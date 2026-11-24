"""Tests for sbayes/sampling/.

The end-to-end test is the acceptance criterion for this layer: it runs the whole
pipeline from the config file to the results files.
"""
import json
import numpy as np
import numpyro
import pandas as pd
import pytest
import subprocess
import sys
import tables
import jax.numpy as jnp

from jax import random
from numpyro.infer import MCMC, NUTS, init_to_median
from functools import partial
from pathlib import Path
from unittest.mock import patch
from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMCSetup
from sbayes.model.model import Model
from sbayes.results import Results
import sbayes.sampling.loggers as loggers_module
from sbayes.sampling.loggers import (
    OnlineSampleLogger, samples_array_to_df, write_samples
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


class SumOfZ:
    """Stands in for `DerivedValues`: the `log_posterior` of a sample is the sum of `z`."""

    with_likelihood = False

    def __init__(self):
        self.n_calls = 0

    def constants(self) -> dict:
        return {}

    def __call__(self, samples: dict) -> dict:
        self.n_calls += 1
        z = np.asarray(samples["z"])
        return {"log_posterior": z.reshape(len(z), -1).sum(axis=1)}


def read_derived(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The `z` samples and the derived log posterior in a samples file."""
    with tables.open_file(str(path)) as f:
        return f.root.z.read(), f.root.derived.log_posterior.read()


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

    def test_resume_discards_a_corrupt_tail(self, tmp_path):
        """Unreadable chunks at the end of a killed run's file are cut off on resume."""
        rng = np.random.default_rng(0)
        with OnlineSampleLogger(tmp_path, run=0, resume=False) as logger:
            for _ in range(5):
                # Random values, so that the chunks are not trivially compressible
                logger.write_sample({"w": rng.random((1, 400, 50))})

        path = tmp_path / "samples_0.h5"
        with tables.open_file(str(path)) as f:
            chunk_rows = f.root.w.chunkshape[0]
            # The first row of the last chunk
            last_start = (f.root.w.nrows - 1) // chunk_rows * chunk_rows
            info = f.root.w.chunk_info((last_start, 0))

        assert 0 < last_start
        with open(path, "r+b") as f:
            f.seek(info.offset)
            f.write(b"\xff" * info.size)

        with OnlineSampleLogger(tmp_path, run=0, resume=True) as logger:
            assert logger.read_samples()["w"].shape[0] == last_start
            logger.write_sample({"w": rng.random((1, 10, 50))})
            samples = logger.read_samples()

        assert samples["w"].shape == (last_start + 10, 50)

    def test_resume_aligns_arrays_of_different_length(self, tmp_path):
        """A run killed between the appends of two parameters is cut to the shorter one."""
        with OnlineSampleLogger(tmp_path, run=0, resume=False) as logger:
            logger.write_sample(make_chunk(4, z=(3, 2), w=(2,)))
        with tables.open_file(str(tmp_path / "samples_0.h5"), mode="a") as f:
            f.root.w.append(np.zeros((2, 2)))

        with OnlineSampleLogger(tmp_path, run=0, resume=True) as logger:
            samples = logger.read_samples()

        assert samples["z"].shape == (4, 3, 2)
        assert samples["w"].shape == (4, 2)

    def test_resume_removes_stale_derived_values(self, tmp_path):
        """Derived values of an earlier write do not survive a resume."""
        with OnlineSampleLogger(tmp_path, run=0, resume=False) as logger:
            logger.write_sample(make_chunk(4, z=(3, 2)))
        with tables.open_file(str(tmp_path / "samples_0.h5"), mode="a") as f:
            f.create_group(f.root, "derived")
            f.create_array(f.root.derived, "log_posterior", np.zeros(4))

        with OnlineSampleLogger(tmp_path, run=0, resume=True) as logger:
            logger.write_sample(make_chunk(4, z=(3, 2)))

        with tables.open_file(str(tmp_path / "samples_0.h5")) as f:
            assert "/derived" not in f
            assert f.root.z.nrows == 8

    def test_derived_values_are_written_with_each_chunk(self, tmp_path):
        logger = OnlineSampleLogger(tmp_path, run=0, resume=False, derive=SumOfZ())
        logger.write_sample(make_chunk(4, z=(3, 2)))
        logger.write_sample(make_chunk(3, z=(3, 2)))

        z, log_posterior = read_derived(tmp_path / "samples_0.h5")
        assert len(log_posterior) == len(z) == 7
        np.testing.assert_array_equal(log_posterior, z.sum(axis=(1, 2)))

    def test_resume_drops_samples_without_derived_values(self, tmp_path):
        """A run killed between writing a chunk and its derived values loses the chunk."""
        logger = OnlineSampleLogger(tmp_path, run=0, resume=False, derive=SumOfZ())
        logger.write_sample(make_chunk(4, z=(3, 2)))
        with tables.open_file(str(tmp_path / "samples_0.h5"), mode="a") as f:
            f.root.z.append(np.ones((2, 3, 2)))

        logger = OnlineSampleLogger(tmp_path, run=0, resume=True, derive=SumOfZ())
        assert logger.read_samples()["z"].shape == (4, 3, 2)
        z, log_posterior = read_derived(tmp_path / "samples_0.h5")
        np.testing.assert_array_equal(log_posterior, z.sum(axis=(1, 2)))

    def test_resume_recomputes_derived_values_written_at_the_end(self, tmp_path):
        """Files of earlier versions hold `/derived` as fixed arrays, written once."""
        with OnlineSampleLogger(tmp_path, run=0, resume=False) as logger:
            logger.write_sample(make_chunk(4, z=(3, 2)))
        with tables.open_file(str(tmp_path / "samples_0.h5"), mode="a") as f:
            f.create_group(f.root, "derived")
            f.create_carray(f.root.derived, "log_posterior", obj=np.zeros(4))

        derive = SumOfZ()
        logger = OnlineSampleLogger(tmp_path, run=0, resume=True, derive=derive)
        logger.write_sample(make_chunk(2, z=(3, 2)))

        z, log_posterior = read_derived(tmp_path / "samples_0.h5")
        assert len(log_posterior) == len(z) == 6
        np.testing.assert_array_equal(log_posterior, z.sum(axis=(1, 2)))
        # Once for the existing samples, once for the new chunk
        assert derive.n_calls == 2

    def test_resume_keeps_matching_derived_values(self, tmp_path):
        with OnlineSampleLogger(tmp_path, run=0, resume=False, derive=SumOfZ()) as logger:
            logger.write_sample(make_chunk(4, z=(3, 2)))

        derive = SumOfZ()
        logger = OnlineSampleLogger(tmp_path, run=0, resume=True, derive=derive)
        logger.read_samples()
        assert derive.n_calls == 0

    def test_file_can_be_read_between_writes(self, tmp_path):
        """The file is closed after each write, so other readers can open it."""
        metadata = {"cluster_names": ["a0"]}
        logger = OnlineSampleLogger(tmp_path, run=0, resume=False, metadata=metadata)
        logger.write_sample(make_chunk(4, z=(3, 2)))

        with tables.open_file(str(tmp_path / "samples_0.h5"), mode="r") as f:
            assert f.root.z.nrows == 4
            assert json.loads(f.get_node_attr(f.root, "metadata")) == metadata

        logger.write_sample(make_chunk(4, z=(3, 2)))
        assert logger.read_samples()["z"].shape == (8, 3, 2)

    def test_write_waits_for_a_reader_to_release_the_file(self, tmp_path):
        """A write while another process reads the file waits instead of failing."""
        logger = OnlineSampleLogger(tmp_path, run=0, resume=False)
        logger.write_sample(make_chunk(4, z=(3, 2)))

        path = tmp_path / "samples_0.h5"
        reader = subprocess.Popen(
            [sys.executable, "-c",
             f"import tables, time; f = tables.open_file({str(path)!r}, 'r'); "
             f"print('open', flush=True); time.sleep(2); f.close()"],
            stdout=subprocess.PIPE, text=True,
        )
        try:
            assert reader.stdout.readline().strip() == "open"
            with patch.object(loggers_module, "_open_h5",
                              partial(loggers_module._open_h5, interval=0.2)):
                logger.write_sample(make_chunk(4, z=(3, 2)))
        finally:
            reader.wait()

        assert logger.read_samples()["z"].shape == (8, 3, 2)

    def test_resume_does_not_overwrite_a_locked_file(self, tmp_path):
        """A resume that cannot get the file from another process fails instead."""
        with OnlineSampleLogger(tmp_path, run=0, resume=False) as logger:
            logger.write_sample(make_chunk(4, z=(3, 2)))

        path = tmp_path / "samples_0.h5"
        reader = subprocess.Popen(
            [sys.executable, "-c",
             f"import tables, time; f = tables.open_file({str(path)!r}, 'r'); "
             f"print('open', flush=True); time.sleep(3); f.close()"],
            stdout=subprocess.PIPE, text=True,
        )
        try:
            assert reader.stdout.readline().strip() == "open"
            logger = OnlineSampleLogger(tmp_path, run=0, resume=True)
            with patch.object(loggers_module, "_open_h5",
                              partial(loggers_module._open_h5, timeout=0.5, interval=0.1)):
                with pytest.raises(tables.HDF5ExtError):
                    logger.read_samples()
        finally:
            reader.wait()

        with tables.open_file(str(path)) as f:
            assert f.root.z.nrows == 4

    def test_reading_before_writing_raises(self, tmp_path):
        logger = OnlineSampleLogger(tmp_path, run=0, resume=False)
        with pytest.raises(ValueError, match="No samples"):
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


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    """Run a short analysis through MCMCSetup and return its results directory."""
    results_path = tmp_path_factory.mktemp("mcmc_run")

    with Experiment(
        config_file=TEST_DATA / "config.yaml",
        experiment_name="test",
        custom_settings={
            "mcmc": {"steps": 20, "samples": 10, "warmup": {"warmup_steps": 10}},
            "results": {"path": str(results_path)},
        },
        log=False,
    ) as experiment:
        data = Data.from_config(experiment.config)
        setup = MCMCSetup(data=data, experiment=experiment)
        setup.sample(run=0)

    return setup.path_results


def test_mcmc_setup_run(completed_run):
    """A short run through MCMCSetup samples and writes its results."""
    assert completed_run.is_dir()

    written = list(completed_run.iterdir())
    assert written, f"no results were written to {completed_run}"
    assert any(f.suffix == ".h5" for f in written), [f.name for f in written]


def test_samples_file_holds_the_joint_log_density(completed_run):
    """The samples file carries the log posterior, so `from_h5` need not fall back.

    Without `/derived/log_posterior`, `Results.from_h5` reverts to `potential_energy`,
    which is the unconstrained potential and not comparable to the likelihood.
    """
    with tables.open_file(str(completed_run / "samples_0.h5")) as f:
        assert "/derived/log_posterior" in f
        assert "/derived/likelihood" in f


def test_results_densities_are_consistent(completed_run):
    """`Results.from_h5` returns a posterior, a likelihood and a prior.

    All three are in constrained space, so the prior is what remains after removing
    the likelihood from the posterior — and all three are negative for this model.
    """
    results = Results.from_h5(completed_run / "samples_0.h5", burn_in=0)

    assert results.posterior is not None
    assert results.likelihood is not None

    assert np.isfinite(results.posterior).all()
    assert (results.likelihood < 0).all()
    assert (results.prior < 0).all()
    np.testing.assert_allclose(
        results.posterior, results.likelihood + results.prior, rtol=1e-5
    )


