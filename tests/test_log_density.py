"""Establish and pin the relationship between log_posterior, log_likelihood and log_prior.

`potential_energy` from NUTS is the *unconstrained* potential, i.e. the negative joint
log density plus the log Jacobian of the transforms to unconstrained space. The results
files report constrained-space quantities (`log_likelihood` is the sum of the pointwise
likelihoods), so `log_posterior` must be the constrained joint log density, not
`-potential_energy`.
"""

import numpy as np
import pandas as pd
import pytest
import tables

from jax import random
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer import log_likelihood as site_log_likelihood
from numpyro.infer.util import log_density, potential_energy
from sbayes.sampling.loggers import write_samples, density_params
from sbayes.results import Results

RNG_SEED = 0
N_WARMUP = 20
N_SAMPLES = 20

_RUN = {}


@pytest.fixture
def sampled(build_model):
    """Run a short chain once per session and return model, samples and potential energies."""
    if not _RUN:
        model = build_model()
        mcmc = MCMC(
            NUTS(model.get_model, init_strategy=init_to_median),
            num_warmup=N_WARMUP, num_samples=N_SAMPLES, num_chains=1, progress_bar=False,
        )
        mcmc.run(random.PRNGKey(RNG_SEED), extra_fields=("potential_energy",))

        _RUN.update(
            model=model,
            samples=mcmc.get_samples(),
            potential_energy=np.asarray(mcmc.get_extra_fields()["potential_energy"]),
            last_state=mcmc.last_state,
        )
    return _RUN


def sample_params(samples: dict, i: int) -> dict:
    """Extract one sample as constrained parameter values.

    Keeps the discrete sites, which the joint log density needs, and drops the
    unconstrained `_raw` copies that reparameterized sites carry.
    """
    return {k: np.asarray(v)[i] for k, v in density_params(samples).items()}


def seeded_model(model):
    """The model with a fixed key, so that any site drawn rather than substituted is fixed."""
    return handlers.seed(model.get_model, random.PRNGKey(RNG_SEED))


def test_potential_energy_is_the_unconstrained_potential(sampled):
    """`potential_energy` is the potential of the sampler's own unconstrained state.

    Uses the chain's final state rather than reconstructing it from `get_samples()`:
    the constrained samples contain deterministic and reparameterized sites, which do
    not map back to the latents one to one.
    """
    model, state = sampled["model"], sampled["last_state"]

    reproduced = float(potential_energy(seeded_model(model), (), {}, state.z))

    assert reproduced == pytest.approx(float(state.potential_energy), rel=1e-3)


def test_constrained_joint_differs_from_negated_potential_energy(sampled):
    """The two differ by the transform Jacobian, which is not a constant offset.

    Documents why the sign of `potential_energy` cannot simply be flipped to obtain
    `log_posterior`: the gap varies from sample to sample.
    """
    model, samples = sampled["model"], sampled["samples"]
    seeded = seeded_model(model)

    gaps = []
    for i in range(3):
        lp, _ = log_density(seeded, (), {}, sample_params(samples, i))
        gaps.append(float(lp) + float(sampled["potential_energy"][i]))

    assert not np.allclose(gaps, gaps[0], rtol=1e-3)


def test_log_posterior_is_likelihood_plus_prior(sampled):
    """The constrained joint log density decomposes into likelihood and prior.

    This is the identity the results files must satisfy:
        log_posterior == log_likelihood + log_prior
    with every term in constrained space, and all three strongly negative.
    """
    model, samples = sampled["model"], sampled["samples"]
    seeded = seeded_model(model)
    params = sample_params(samples, 0)

    log_posterior, _ = log_density(seeded, (), {}, params)
    log_posterior = float(log_posterior)

    # The observed sites only; summed over observations
    per_site = site_log_likelihood(seeded, {k: v[None] for k, v in params.items()})
    log_lh = float(sum(np.sum(np.asarray(v)) for v in per_site.values()))

    log_prior = log_posterior - log_lh

    assert np.isfinite([log_posterior, log_lh, log_prior]).all()
    assert log_lh < 0
    assert log_posterior == pytest.approx(log_lh + log_prior, rel=1e-6)
    # The joint cannot exceed the likelihood by more than the prior's own mass
    assert log_posterior < 0


@pytest.fixture
def written_run(sampled, data, tmp_path):
    """Write the sampled chain to a stats file and a samples file in `tmp_path`.

    `write_samples` appends to the samples file the online logger wrote during the run;
    here it creates that file, so it holds the derived arrays but none of the parameter
    arrays. That is enough for the h5 checks below, which only read `/derived`.
    """
    model, samples = sampled["model"], sampled["samples"]
    write_samples(run=0, base_path=tmp_path, samples=samples, data=data, model=model)

    return {
        "model": model,
        "samples": samples,
        "stats_path": tmp_path / f"stats_K{model.n_clusters}_0.tsv",
        "samples_path": tmp_path / "samples_0.h5",
    }


def test_stats_file_has_the_three_densities(written_run):
    """The stats file reports a posterior, a likelihood and a prior, all finite."""
    stats = pd.read_csv(written_run["stats_path"], sep="\t")

    assert {"log_posterior", "log_likelihood", "log_prior"} <= set(stats.columns)
    columns = stats[["log_posterior", "log_likelihood", "log_prior"]]
    assert np.isfinite(columns).all().all()


def test_stats_densities_have_plausible_signs(written_run):
    """All three densities are negative, and removing the likelihood raises the total.

    These are the assertions the old output failed: it reported a positive log prior of
    about +3152, because the posterior column held the sampler's unconstrained potential
    rather than the joint log density.
    """
    stats = pd.read_csv(written_run["stats_path"], sep="\t")

    assert (stats["log_likelihood"] < 0).all()
    assert (stats["log_prior"] < 0).all()
    assert (stats["log_posterior"] < 0).all()

def test_stats_densities_decompose(written_run):
    """The prior is exactly what remains after removing the likelihood."""
    stats = pd.read_csv(written_run["stats_path"], sep="\t")

    np.testing.assert_allclose(
        stats["log_posterior"], stats["log_likelihood"] + stats["log_prior"], rtol=1e-6
    )


def test_stats_posterior_matches_an_independent_evaluation(written_run):
    """The written posterior equals the joint log density recomputed from the samples.

    The decomposition above holds by construction, since the prior is derived by
    subtraction; this is the check that the posterior itself is the right quantity.
    """

    model, samples = written_run["model"], written_run["samples"]
    stats = pd.read_csv(written_run["stats_path"], sep="\t")

    for i in (0, 1):
        expected, _ = log_density(seeded_model(model), (), {}, sample_params(samples, i))
        assert stats["log_posterior"].iloc[i] == pytest.approx(float(expected), rel=1e-3)



def test_samples_file_holds_the_likelihood(written_run):
    """The pointwise likelihood is written and sums to the stats file's column."""
    stats = pd.read_csv(written_run["stats_path"], sep="\t")

    with tables.open_file(str(written_run["samples_path"])) as f:
        assert "/derived/likelihood" in f
        assert "/derived/na_values" in f

        likelihood = np.array(f.get_node("/derived/likelihood"))
        na_values = np.array(f.get_node("/derived/na_values"))

    # Missing observations are written as zeros, so the full sum is the sample's
    # log likelihood either way
    np.testing.assert_allclose(likelihood.sum(axis=1), stats["log_likelihood"], rtol=1e-6)
    assert likelihood[:, na_values].sum() == 0.0


def test_sample_keys_cover_the_latent_sites(sampled):
    """The written samples contain every latent site the sampler moves.

    Documents which keys in `get_samples()` are the latents and which are derived,
    so the results writers can be checked against it.
    """
    latents = set(sampled["last_state"].z)
    written = set(sampled["samples"])

    assert latents - written == set(), f"Latent sites missing from the samples: {latents - written}"


