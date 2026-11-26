"""Dirichlet distributions sampled in the unconstrained stick-breaking space.

A Dirichlet(alpha) on K states splits into K-1 independent stick fractions
f_i ~ Beta(a_i, b_i), with a_i = alpha_i and b_i = alpha_{i+1} + ... + alpha_K. Writing
f_i = sigmoid(u_i), each u_i has the log-concave density

    log p(u_i) = a_i * log_sigmoid(u_i) + b_i * log_sigmoid(-u_i) - log B(a_i, b_i).

`u` is numpyro's stick-breaking input minus its fixed shift log(K-1-i), so this is the
same density NUTS sees for a plain `dist.Dirichlet`, but it is defined on the whole real
line (no bound needed) and stays finite where the simplex value underflows to 0.

All Dirichlet sites of the model are sampled this way: in float32, a plain
`dist.Dirichlet` with concentration below 1 gives a non-finite density (and divergent
NUTS transitions) once a simplex entry underflows to 0, which happens far in the tails.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.nn import log_sigmoid
from jax.scipy.special import betaln, digamma
from numpyro.distributions import constraints
from numpyro.distributions.transforms import StickBreakingTransform
from numpyro.distributions.util import validate_sample


def _stick_breaking_shift(n_states: int) -> jnp.ndarray:
    """The shift log(K-1-i) that numpyro's stick-breaking transform subtracts."""
    return jnp.log(jnp.arange(n_states - 1, 0, -1))


class LatentDirichlet(dist.Distribution):
    """A Dirichlet distribution expressed in the stick-breaking latent space.

    Samples are unconstrained vectors of length K-1. They map to the simplex by
    `to_simplex`, i.e. by numpyro's `StickBreakingTransform`. The density is that of the
    latent value, so the transform's Jacobian is already included.

    Args:
        concentration: the Dirichlet concentration, shape (..., K)
    """

    arg_constraints = {"concentration": constraints.independent(constraints.positive, 1)}
    support = constraints.real_vector

    def __init__(
        self,
        concentration: jnp.ndarray,
        *,
        validate_args: bool | None = None,
    ):
        self.concentration = jnp.asarray(concentration)
        n_states = self.concentration.shape[-1]
        batch_shape = self.concentration.shape[:-1]
        self._shift = _stick_breaking_shift(n_states)

        # Beta parameters of the stick fractions: a_i = alpha_i, b_i = sum_{j>i} alpha_j
        self._a = self.concentration[..., :-1]
        self._b = jnp.cumsum(self.concentration[..., ::-1], axis=-1)[..., ::-1][..., 1:]

        super().__init__(
            batch_shape=batch_shape,
            event_shape=(n_states - 1,),
            validate_args=validate_args,
        )

    def _to_logits(self, value: jnp.ndarray) -> jnp.ndarray:
        """Logits u of the stick fractions for a latent value."""
        return value - self._shift

    def _from_logits(self, u: jnp.ndarray) -> jnp.ndarray:
        """Latent value for the logits u of the stick fractions."""
        return u + self._shift

    def to_simplex(self, value: jnp.ndarray) -> jnp.ndarray:
        """Map a latent value to the probability simplex."""
        return StickBreakingTransform()(value)

    def sample(self, key: jax.Array, sample_shape: tuple = ()) -> jnp.ndarray:
        # Gamma construction in log space: f_i = G_i / (G_i + sum_{j>i} G_j), so
        # logit(f_i) = log G_i - log sum_{j>i} G_j. Stable for small concentrations.
        shape = sample_shape + self.batch_shape + self.concentration.shape[-1:]
        log_g = jax.random.loggamma(key, self.concentration, shape=shape)
        log_rest = jax.lax.cumlogsumexp(log_g, axis=log_g.ndim - 1, reverse=True)[..., 1:]
        return self._from_logits(log_g[..., :-1] - log_rest)

    @validate_sample
    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        u = self._to_logits(value)
        a, b = self._a, self._b
        return jnp.sum(a * log_sigmoid(u) + b * log_sigmoid(-u) - betaln(a, b), axis=-1)

    @property
    def mean(self) -> jnp.ndarray:
        # E[logit f] = digamma(a) - digamma(b) for f ~ Beta(a, b)
        return self._from_logits(digamma(self._a) - digamma(self._b))


def latent_from_simplex(x: jnp.ndarray) -> jnp.ndarray:
    """The latent value of `dirichlet_from_latent` that maps to the simplex value `x`.

    Args:
        x: a point on the simplex, shape (..., K)

    Returns:
        The latent value, shape (..., K-1)
    """
    return StickBreakingTransform().inv(x)


def dirichlet_from_latent(
    name: str,
    concentration: jnp.ndarray,
) -> jnp.ndarray:
    """Sample from a Dirichlet distribution through the stick-breaking latent space.

    The latent value is sampled at site `{name}_raw` and its value on the simplex is
    recorded as the deterministic site `{name}`.

    Args:
        name: the name of the numpyro sample site
        concentration: the concentration parameter of the Dirichlet distribution

    Returns:
        The sampled value on the probability simplex.
    """
    latent_distr = LatentDirichlet(concentration)
    z = numpyro.sample(f"{name}_raw", latent_distr)
    x = latent_distr.to_simplex(z)
    numpyro.deterministic(name, x)
    return x
