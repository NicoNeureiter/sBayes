"""Tests for the Laplace-based initial mass matrix in sbayes/sampling/initialization.py."""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from jax import random
from jax.flatten_util import ravel_pytree
from numpyro.infer import MCMC, NUTS, init_to_value
from sbayes.sampling.initialization import laplace_inverse_mass_matrix

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
