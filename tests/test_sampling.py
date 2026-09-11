import jax.numpy as jnp

from jax import random
from numpyro.infer import init_to_median, MCMC, NUTS
from pathlib import Path
from sbayes.config.config import SBayesConfig
from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMCSetup

TEST_DATA = Path(__file__).parent / "data"

def test_short_mcmc_run(build_model, tmp_path):
    """A short NUTS run produces finite samples for every parameter."""
    model = build_model()

    kernel = NUTS(model.get_model, init_strategy=init_to_median)
    mcmc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1, progress_bar=False)
    mcmc.run(random.PRNGKey(0))

    samples = mcmc.get_samples()
    assert samples, "the sampler produced no samples"
    for name, value in samples.items():
        assert jnp.isfinite(jnp.asarray(value)).all(), f"non-finite samples for {name}"

    # The chain moved
    z = jnp.asarray(samples["z"])
    assert not jnp.allclose(z[0], z[-1])


def test_mcmc_setup_run(tmp_path):
    """A short run through MCMCSetup samples and writes its results."""
    with Experiment(
        config_file=TEST_DATA / "config.yaml",
        experiment_name="test",
        custom_settings={
            "mcmc": {"steps": 20, "samples": 10, "warmup": {"warmup_steps": 10}},
            "results": {"path": str(tmp_path)},
        },
        log=False,
    ) as experiment:
        data = Data.from_config(experiment.config)
        setup = MCMCSetup(data=data, experiment=experiment)
        setup.sample(run=0)

    results_dir = setup.path_results
    assert results_dir.is_dir()

    written = list(results_dir.iterdir())
    assert written, f"no results were written to {results_dir}"
    assert any(f.suffix == ".h5" for f in written), [f.name for f in written]