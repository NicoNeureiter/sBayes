"""Tests for the Laplace mass matrix and local evidence in sbayes/sampling/initialization.py."""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from jax import random
from jax.flatten_util import ravel_pytree
from numpyro.infer import MCMC, NUTS, init_to_value
from sbayes.sampling.initialization import laplace_inverse_mass_matrix, log_local_evidence

SD_A = jnp.array([0.1, 1.0, 10.0])
SD_B = 2.0


class _ToyModel:
    """Minimal stand-in for `Model`: only `get_model` is needed."""

    def __init__(self, fn):
        self.get_model = fn


def _gaussian_model():
    numpyro.sample("b", dist.Normal(3.0, SD_B).expand((2, 2)).to_event(2))
    numpyro.sample("a", dist.Normal(0.0, SD_A).to_event(1))


def _mixed_model():
    numpyro.sample("a", dist.Normal(0.0, SD_A).to_event(1))
    numpyro.sample("p", dist.Dirichlet(jnp.array([2.0, 3.0, 4.0])))


def test_gaussian_curvature_gives_the_variances():
    """For a Gaussian the inverse curvature is exactly the variance, in NUTS's order."""
    init = {"a": jnp.array([0.3, -1.0, 5.0]), "b": jnp.full((2, 2), 3.5)}
    imm = laplace_inverse_mass_matrix(_ToyModel(_gaussian_model), init, random.PRNGKey(0))
    expected, _ = ravel_pytree({"a": SD_A ** 2, "b": jnp.full((2, 2), SD_B ** 2)})
    np.testing.assert_allclose(imm, expected, rtol=1e-4)


def test_chunking_does_not_change_the_result():
    init = {"a": jnp.zeros(3), "b": jnp.zeros((2, 2))}
    model = _ToyModel(_gaussian_model)
    full = laplace_inverse_mass_matrix(model, init, random.PRNGKey(0), chunk_size=256)
    chunked = laplace_inverse_mass_matrix(model, init, random.PRNGKey(0), chunk_size=3)
    np.testing.assert_allclose(chunked, full, rtol=1e-6)


def test_non_positive_curvature_keeps_the_default():
    def model():
        x = numpyro.sample("x", dist.Normal(0.0, 1.0).expand((2,)).to_event(1))
        # Curvature 1 - 1.5 < 0 in the first element only
        numpyro.factor("bump", 0.75 * x[0] ** 2)

    imm = laplace_inverse_mass_matrix(_ToyModel(model), {"x": jnp.zeros(2)}, random.PRNGKey(0))
    np.testing.assert_allclose(imm, [1.0, 1.0], rtol=1e-5)


def test_nuts_accepts_the_matrix_for_constrained_sites():
    """The matrix has the size and order of NUTS's unconstrained parameters."""
    init = {"a": jnp.zeros(3), "p": jnp.array([0.2, 0.3, 0.5])}
    model = _ToyModel(_mixed_model)
    imm = laplace_inverse_mass_matrix(model, init, random.PRNGKey(0))
    assert imm.shape == (3 + 2,)   # the 3-simplex has 2 unconstrained parameters
    np.testing.assert_allclose(imm[:3], SD_A ** 2, rtol=1e-4)
    assert jnp.all(imm > 0)

    kernel = NUTS(
        model.get_model,
        init_strategy=init_to_value(values=init),
        inverse_mass_matrix=imm,
        adapt_mass_matrix=False,
    )
    mcmc = MCMC(kernel, num_warmup=5, num_samples=5, progress_bar=False)
    mcmc.run(random.PRNGKey(1))
    # numpyro stores the diagonal as one block for all sites, in sorted site order
    used = mcmc.last_state.adapt_state.inverse_mass_matrix
    assert list(used) == [("a", "p")]
    np.testing.assert_allclose(used[("a", "p")], imm)


def test_local_evidence_is_exact_for_a_gaussian():
    """For a conjugate Gaussian the Laplace proposal is the posterior, so IS is exact."""
    y = 1.5

    def model():
        x = numpyro.sample("x", dist.Normal(0.0, 1.0))
        numpyro.sample("y", dist.Normal(x, 1.0), obs=y)

    # Posterior mode y / 2, evidence N(y; 0, sqrt(2))
    evidence = log_local_evidence(model, {"x": jnp.array(y / 2)}, random.PRNGKey(0))
    np.testing.assert_allclose(evidence, dist.Normal(0.0, jnp.sqrt(2.0)).log_prob(y), rtol=1e-4)


def test_local_evidence_accounts_for_the_transform():
    """A normalized prior alone has evidence 1, also on a constrained site."""

    def model():
        numpyro.sample("p", dist.Dirichlet(jnp.array([2.0, 3.0, 4.0])))

    init = {"p": jnp.array([1 / 6, 2 / 6, 3 / 6])}   # the mode
    evidence = log_local_evidence(model, init, random.PRNGKey(0), num_samples=4000)
    assert abs(evidence) < 0.05


def test_local_evidence_prefers_the_mode_with_more_mass():
    """The narrow mode is higher, but the wide one holds 70% of the mass."""
    narrow = dist.Normal(-3.0, 0.2)
    wide = dist.Normal(3.0, 1.0)

    def model():
        x = numpyro.sample("x", dist.ImproperUniform(dist.constraints.real, (), ()))
        numpyro.factor("mix", jnp.logaddexp(
            jnp.log(0.3) + narrow.log_prob(x), jnp.log(0.7) + wide.log_prob(x)
        ))

    ev_narrow = log_local_evidence(model, {"x": jnp.array(-3.0)}, random.PRNGKey(0))
    ev_wide = log_local_evidence(model, {"x": jnp.array(3.0)}, random.PRNGKey(0))
    np.testing.assert_allclose(ev_narrow, np.log(0.3), atol=0.02)
    np.testing.assert_allclose(ev_wide, np.log(0.7), atol=0.02)
