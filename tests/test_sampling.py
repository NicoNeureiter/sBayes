"""Tests for sbayes/sampling/.

The end-to-end test is the acceptance criterion for this layer: it runs the whole
pipeline from the config file to the results files.
"""
import numpy as np
import numpyro
import pandas as pd
import pytest
import tables
import jax.numpy as jnp

from jax import random
from numpyro.infer import MCMC, NUTS, init_to_median
from pathlib import Path
from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMCSetup
from sbayes.model.model import Model
from sbayes.sampling.loggers import (
    OnlineSampleLogger, samples_array_to_df, write_samples,
)

TEST_DATA = Path(__file__).parent / "data"

def make_chunk(n_samples: int, **param_dims: tuple[int, ...]) -> dict:
    """Build a chunk of samples as the sampler produces it, grouped by chain.

    The chain dimension is always 1 and is squeezed away by the logger, so a parameter
    of shape `dims` arrives as (1, n_samples, *dims).
    """
    return {
        name: np.arange(np.prod((n_samples, *dims)), dtype=float).reshape(
            1, n_samples, *dims
        )
        for name, dims in param_dims.items()
    }


class TestOnlineSampleLogger:

    def test_two_chunks_are_appended(self, tmp_path):
        """Chunks written in sequence end up in one array per parameter."""
        with OnlineSampleLogger(tmp_path, run=0, resume=False) as logger:
            logger.write_sample(make_chunk(4, z=(3, 2), w=(2,)))
            logger.write_sample(make_chunk(4, z=(3, 2), w=(2,)))
            samples = logger.read_samples()

        assert set(samples) == {"z", "w"}
        assert samples["z"].shape == (8, 3, 2)
        assert samples["w"].shape == (8, 2)

    def test_samples_file_is_written(self, tmp_path):
        with OnlineSampleLogger(tmp_path, run=3, resume=False) as logger:
            logger.write_sample(make_chunk(2, z=(3, 2)))

        samples_path = tmp_path / "samples_3.h5"
        assert samples_path.is_file()

        with tables.open_file(str(samples_path)) as f:
            assert f.root.z.shape == (2, 3, 2)

    def test_resume_appends_to_an_existing_file(self, tmp_path):
        """A logger opened with resume=True adds to the samples of a previous run."""
        with OnlineSampleLogger(tmp_path, run=0, resume=False) as logger:
            logger.write_sample(make_chunk(4, z=(3, 2)))

        with OnlineSampleLogger(tmp_path, run=0, resume=True) as logger:
            logger.write_sample(make_chunk(4, z=(3, 2)))
            samples = logger.read_samples()

        assert samples["z"].shape == (8, 3, 2)

    def test_without_resume_the_previous_samples_are_overwritten(self, tmp_path):
        with OnlineSampleLogger(tmp_path, run=0, resume=False) as logger:
            logger.write_sample(make_chunk(4, z=(3, 2)))

        with OnlineSampleLogger(tmp_path, run=0, resume=False) as logger:
            logger.write_sample(make_chunk(4, z=(3, 2)))
            samples = logger.read_samples()

        assert samples["z"].shape == (4, 3, 2)

    def test_state_round_trip(self, tmp_path):
        """The sampler state is written and read back unchanged, for resuming a run."""
        state = {"step_size": 0.25, "position": np.arange(6).reshape(2, 3)}

        with OnlineSampleLogger(tmp_path, run=1, resume=False) as logger:
            logger.dump_state(state)

        assert (tmp_path / "state_1.pkl").is_file()

        with OnlineSampleLogger(tmp_path, run=1, resume=True) as logger:
            loaded = logger.load_state()

        assert loaded["step_size"] == state["step_size"]
        assert np.array_equal(loaded["position"], state["position"])

    def test_reading_before_writing_raises(self, tmp_path):
        with OnlineSampleLogger(tmp_path, run=0, resume=False) as logger:
            with pytest.raises(ValueError, match="not open"):
                logger.read_samples()


class TestSamplesArrayToDf:

    def test_column_names_combine_the_name_groups(self):
        samples = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
        df = samples_array_to_df(
            samples, names=[["a0", "a1", "a2"], ["s0", "s1"]], prefix="areal"
        )

        assert df.shape == (2, 6)
        assert df.columns.tolist() == [
            "areal_a0_s0", "areal_a0_s1",
            "areal_a1_s0", "areal_a1_s1",
            "areal_a2_s0", "areal_a2_s1",
        ]
        # The columns are the flattened parameter, in row-major order
        assert df.iloc[0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_prefix_and_suffix(self):
        samples = np.zeros((2, 2))
        df = samples_array_to_df(samples, names=[["f0", "f1"]], suffix="mean")
        assert df.columns.tolist() == ["f0_mean", "f1_mean"]

    def test_numpy_name_arrays_are_accepted(self):
        samples = np.zeros((2, 2))
        df = samples_array_to_df(samples, names=[np.array(["f0", "f1"])], prefix="w")
        assert df.columns.tolist() == ["w_f0", "w_f1"]

    def test_wrong_name_group_length_raises(self):
        samples = np.zeros((2, 3, 2))
        with pytest.raises(ValueError, match="names imply"):
            samples_array_to_df(samples, names=[["a0", "a1"], ["s0", "s1"]])

    def test_wrong_number_of_name_groups_raises(self):
        samples = np.zeros((2, 3, 2))
        with pytest.raises(ValueError, match="names imply"):
            samples_array_to_df(samples, names=[["a0", "a1", "a2"]])


def test_write_samples(tmp_path, config):
    """The stats file and the samples metadata are written for a completed run."""
    data = Data.from_config(config)
    model = Model(data, config.model)

    n_samples = 5
    with OnlineSampleLogger(tmp_path, run=0, resume=False) as logger:
        with numpyro.handlers.seed(rng_seed=0):
            trace = numpyro.handlers.trace(model.get_model).get_trace()

        # Repeat one prior sample `n_samples` times, as a stand-in for a real chain
        chunk = {
            name: np.broadcast_to(
                np.asarray(site["value"]), (1, n_samples, *np.shape(site["value"]))
            )
            for name, site in trace.items()
            if site["type"] in ("sample", "deterministic")
            and not site.get("is_observed")
        }
        logger.write_sample(chunk)
        samples = logger.read_samples()

    write_samples(
        run=0, base_path=tmp_path, samples=samples, data=data, model=model
    )

    stats_path = tmp_path / f"stats_K{model.n_clusters}_0.tsv"
    assert stats_path.is_file()

    stats = pd.read_csv(stats_path, sep="\t")
    assert len(stats) == n_samples
    assert "log_posterior" in stats.columns
    assert stats["Sample"].tolist() == list(range(n_samples))

    with tables.open_file(str(tmp_path / "samples_0.h5")) as f:
        assert "metadata" in f.root._v_attrs


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