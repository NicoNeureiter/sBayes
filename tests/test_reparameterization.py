"""Tests for sbayes/model/reparameterization.py."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from jax import random
from numpyro import handlers
from numpyro.distributions.transforms import StickBreakingTransform
from numpyro.infer import MCMC, NUTS, Predictive
from sbayes.model.reparameterization import LatentDirichlet, dirichlet_from_latent

CONCENTRATIONS = [
    jnp.array([1.0, 1.0]),
    jnp.array([0.1, 0.1, 0.1, 0.1]),
    jnp.array([0.1, 0.5, 2.0, 5.0]),
    jnp.full(9, 0.1),
]
OFFSETS = [None, "uniform", "random"]


def _offset(kind, concentration, key=random.PRNGKey(7)):
    if kind is None:
        return None
    if kind == "uniform":
        return jnp.full(concentration.shape, 1 / concentration.shape[-1])
    return dist.Dirichlet(jnp.ones(concentration.shape)).sample(key)


@pytest.fixture(autouse=True)
def float64():
    """Compare densities in double precision, where the reference is exact."""
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("offset_kind", OFFSETS)
@pytest.mark.parametrize("concentration", CONCENTRATIONS)
def test_density_matches_dirichlet_with_jacobian(concentration, offset_kind):
    """The latent density equals the Dirichlet density times the transform's Jacobian."""
    concentration = concentration.astype(jnp.float64)
    offset = _offset(offset_kind, concentration)
    latent = LatentDirichlet(concentration, offset)
    sb = StickBreakingTransform()

    z = 3.0 * random.normal(random.PRNGKey(0), (500, concentration.shape[-1] - 1))
    y = z + latent.latent_offset
    x = sb(y)
    expected = dist.Dirichlet(concentration).log_prob(x) + sb.log_abs_det_jacobian(y, x)

    np.testing.assert_allclose(latent.log_prob(z), expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(latent.to_simplex(z), x)


@pytest.mark.parametrize("concentration", CONCENTRATIONS)
def test_density_integrates_to_one(concentration):
    """Monte Carlo check that the latent density is normalized (importance sampling)."""
    concentration = concentration.astype(jnp.float64)
    latent = LatentDirichlet(concentration)
    proposal = dist.StudentT(1.0, latent.mean, 10.0).to_event(1)
    z = proposal.sample(random.PRNGKey(1), (400_000,))
    log_w = latent.log_prob(z) - proposal.log_prob(z)
    integral = jnp.exp(jax.scipy.special.logsumexp(log_w) - jnp.log(z.shape[0]))
    assert integral == pytest.approx(1.0, abs=0.05)


def test_density_is_finite_far_in_the_tails():
    """No bound and no underflow: the density stays finite where the simplex is 0."""
    jax.config.update("jax_enable_x64", False)
    latent = LatentDirichlet(jnp.full(4, 0.1, dtype=jnp.float32))
    z = jnp.array([[-1e4, 0.0, 0.0], [0.0, 1e4, -1e4], [500.0, 500.0, 500.0]], dtype=jnp.float32)
    assert jnp.any(latent.to_simplex(z) == 0.0)
    assert jnp.all(jnp.isfinite(latent.log_prob(z)))
    assert jnp.all(jnp.isfinite(jax.grad(lambda v: latent.log_prob(v).sum())(z)))


@pytest.mark.parametrize("offset_kind", OFFSETS)
@pytest.mark.parametrize("concentration", CONCENTRATIONS)
def test_samples_match_dirichlet_moments(concentration, offset_kind):
    """Samples mapped to the simplex have the Dirichlet's mean and variance."""
    concentration = concentration.astype(jnp.float64)
    latent = LatentDirichlet(concentration, _offset(offset_kind, concentration))
    x = latent.to_simplex(latent.sample(random.PRNGKey(2), (200_000,)))

    alpha0 = concentration.sum()
    mean = concentration / alpha0
    var = mean * (1 - mean) / (alpha0 + 1)
    assert jnp.all(jnp.isfinite(x))
    np.testing.assert_allclose(x.mean(0), mean, atol=5 * jnp.sqrt(var.max() / x.shape[0]) + 1e-3)
    np.testing.assert_allclose(x.var(0), var, rtol=0.05, atol=1e-4)


def test_samples_match_dirichlet_marginal_distribution():
    """The first component of the samples follows its Beta marginal (KS test)."""
    from scipy.stats import beta, kstest

    concentration = jnp.array([0.1, 0.5, 2.0], dtype=jnp.float64)
    latent = LatentDirichlet(concentration, jnp.array([0.6, 0.3, 0.1]))
    x = latent.to_simplex(latent.sample(random.PRNGKey(3), (20_000,)))
    a, b = float(concentration[0]), float(concentration[1:].sum())
    assert kstest(np.asarray(x[:, 0]), beta(a, b).cdf).pvalue > 1e-3


def test_mean_is_mean_of_samples():
    latent = LatentDirichlet(jnp.array([0.3, 1.0, 4.0]), jnp.array([0.5, 0.25, 0.25]))
    z = latent.sample(random.PRNGKey(4), (200_000,))
    np.testing.assert_allclose(z.mean(0), latent.mean, atol=0.05)


def test_batch_shapes_broadcast_concentration_and_offset():
    concentration = jnp.ones((5, 3))
    offset = jnp.full((2, 1, 3), 1 / 3)
    latent = LatentDirichlet(concentration, offset)
    assert latent.batch_shape == (2, 5)
    assert latent.event_shape == (2,)
    assert latent.sample(random.PRNGKey(0), (4,)).shape == (4, 2, 5, 2)


def _plate_model(concentration, offset):
    with numpyro.plate("groups", 3):
        dirichlet_from_latent("x", concentration, offset=offset)


def test_sites_of_dirichlet_from_latent():
    concentration = jnp.array([0.5, 1.0, 2.0])
    trace = handlers.trace(handlers.seed(_plate_model, 0)).get_trace(concentration, None)
    assert trace["x_raw"]["type"] == "sample"
    assert trace["x_raw"]["value"].shape == (3, 2)
    assert trace["x"]["type"] == "deterministic"
    np.testing.assert_allclose(trace["x"]["value"].sum(-1), 1.0, rtol=1e-6)
    assert not any(name.endswith("_log_prob") for name in trace)


def test_prior_predictive_matches_dirichlet():
    """`Predictive` draws from the latent site are exact Dirichlet draws."""
    concentration = jnp.array([0.1, 0.5, 2.0])
    offset = jnp.array([0.7, 0.2, 0.1])
    x = Predictive(_plate_model, num_samples=20_000)(random.PRNGKey(5), concentration, offset)["x"]
    np.testing.assert_allclose(x.mean((0, 1)), concentration / concentration.sum(), atol=0.01)


def test_nuts_samples_match_plain_dirichlet():
    """NUTS on the latent site recovers the Dirichlet posterior under a likelihood."""
    concentration = jnp.array([0.3, 0.3, 0.3])
    counts = jnp.array([8.0, 1.0, 0.0])

    def latent_model():
        x = dirichlet_from_latent("x", concentration, offset=jnp.array([0.5, 0.3, 0.2]))
        numpyro.factor("lh", jnp.sum(counts * jnp.log(x + 1e-30)))

    mcmc = MCMC(NUTS(latent_model), num_warmup=500, num_samples=4000, progress_bar=False)
    mcmc.run(random.PRNGKey(6))
    x = mcmc.get_samples()["x"]
    expected = (concentration + counts) / (concentration + counts).sum()   # conjugate update
    np.testing.assert_allclose(x.mean(0), expected, atol=0.03)
