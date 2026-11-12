"""Dirichlet distributions sampled in the unconstrained stick-breaking space.

A Dirichlet(alpha) on K states splits into K-1 independent stick fractions
f_i ~ Beta(a_i, b_i), with a_i = alpha_i and b_i = alpha_{i+1} + ... + alpha_K. Writing
f_i = sigmoid(u_i), each u_i has the log-concave density

    log p(u_i) = a_i * log_sigmoid(u_i) + b_i * log_sigmoid(-u_i) - log B(a_i, b_i).

`u` is numpyro's stick-breaking input minus its fixed shift log(K-1-i), so this is the
same density NUTS sees for a plain `dist.Dirichlet`, but it is defined on the whole real
line (no bound needed) and stays finite where the simplex value underflows to 0.

The reparameterization only changes the sampling geometry if the latent space is
centered on an `offset` that depends on other sampled parameters. For a constant (or
no) offset it is equivalent to a plain `dist.Dirichlet`, which should be used instead.
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
    `to_simplex`, i.e. by adding the latent representation of `offset` and applying
    numpyro's `StickBreakingTransform`. The density is that of the latent value, so
    the transform's Jacobian is already included.

    Args:
        concentration: the Dirichlet concentration, shape (..., K)
        offset: optional point on the simplex to center the latent space on, shape
            (..., K). Without it the latent space is centered on the uniform simplex.
    """

    arg_constraints = {"concentration": constraints.independent(constraints.positive, 1)}
    support = constraints.real_vector

    def __init__(
        self,
        concentration: jnp.ndarray,
        offset: jnp.ndarray | None = None,
        *,
        validate_args: bool | None = None,
    ):
        concentration = jnp.asarray(concentration)
        n_states = concentration.shape[-1]
        if offset is None:
            latent_offset = jnp.zeros(n_states - 1, dtype=concentration.dtype)
        else:
            latent_offset = StickBreakingTransform().inv(jnp.asarray(offset))

        batch_shape = jax.lax.broadcast_shapes(
            concentration.shape[:-1], latent_offset.shape[:-1]
        )
        self.concentration = jnp.broadcast_to(concentration, batch_shape + (n_states,))
        self.latent_offset = jnp.broadcast_to(latent_offset, batch_shape + (n_states - 1,))
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
        return value + self.latent_offset - self._shift

    def _from_logits(self, u: jnp.ndarray) -> jnp.ndarray:
        """Latent value for the logits u of the stick fractions."""
        return u + self._shift - self.latent_offset

    def to_simplex(self, value: jnp.ndarray) -> jnp.ndarray:
        """Map a latent value to the probability simplex."""
        return StickBreakingTransform()(value + self.latent_offset)

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


def dirichlet_from_latent(
    name: str,
    concentration: jnp.ndarray,
    offset: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Sample from a Dirichlet distribution through the stick-breaking latent space.

    The latent value is sampled at site `{name}_raw` and its value on the simplex is
    recorded as the deterministic site `{name}`. Only worth using if `offset` depends on
    other sampled parameters; otherwise a plain `dist.Dirichlet` is equivalent.

    Args:
        name: the name of the numpyro sample site
        concentration: the concentration parameter of the Dirichlet distribution
        offset: optional point on the simplex to center the latent space on

    Returns:
        The sampled value on the probability simplex.
    """
    latent_distr = LatentDirichlet(concentration, offset)
    z = numpyro.sample(f"{name}_raw", latent_distr)
    x = latent_distr.to_simplex(z)
    numpyro.deterministic(name, x)
    return x
