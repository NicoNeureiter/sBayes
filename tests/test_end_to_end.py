#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end runs of the sBayes pipeline on the example case studies.

These are acceptance tests: they exercise config loading, data loading, MCMC setup and
the full inference on real example data, and check that the results files are complete
and internally consistent. They do not check the quality of the inference itself - the
simulation test is the only one with known ground truth, and the calibration study
covers accuracy properly.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import jax.random as random
import numpy as np
import pandas as pd
import tables

from sbayes.cli import main as sbayes_main
from sbayes.simulate.config import load_config
from sbayes.simulate.simulator import Simulator
from sbayes.simulate.validate import compare_parameters, read_simulations

SIMULATE_CONFIG_PATH = Path("experiments/simulation/config_simulate.yaml")
SOUTH_AMERICA_CONFIG = Path("experiments/south_america/config.yaml")
RNG_SEED = 0

CUSTOM_SETTINGS = {
    "mcmc": {
        "steps": 40,
        "samples": 20,
        "runs": 2,
        "warmup": {"warmup_steps": 5, "warmup_chains": 2},
    },
}


def settings(results_path: Path, **overrides) -> dict:
    """Build custom settings that write into `results_path`.

    Args:
        results_path: directory for this run's results, normally a `tmp_path`
        overrides: further config sections, merged into the defaults

    Returns:
        The custom settings to pass to `sbayes_main`.
    """
    custom_settings = deepcopy(CUSTOM_SETTINGS)
    custom_settings["results"] = {"path": str(results_path)}
    for section, values in overrides.items():
        custom_settings.setdefault(section, {}).update(values)
    return custom_settings


def assert_run_is_complete(run_dir: Path, n_samples: int, n_runs: int = 1) -> None:
    """Check that a finished run wrote complete, self-consistent results.

    Args:
        run_dir: the `K{k}` directory of one cluster count
        n_samples: the number of samples the config asked for
        n_runs: the number of runs that should have been written
    """
    samples_files = sorted(run_dir.glob("samples_*.h5"))
    stats_files = sorted(run_dir.glob("stats_*.tsv"))
    assert len(samples_files) == n_runs, [p.name for p in run_dir.iterdir()]
    assert len(stats_files) == n_runs, [p.name for p in run_dir.iterdir()]

    for samples_path in samples_files:
        with tables.open_file(str(samples_path)) as f:
            assert "/derived/log_posterior" in f
            assert "/derived/likelihood" in f

    for stats_path in stats_files:
        stats = pd.read_csv(stats_path, sep="\t")
        assert len(stats) == n_samples

        densities = stats[["log_posterior", "log_likelihood", "log_prior"]]
        assert np.isfinite(densities).all().all()
        np.testing.assert_allclose(
            stats["log_posterior"], stats["log_likelihood"] + stats["log_prior"],
            rtol=1e-5,
        )


# --- The example case studies --------------------------------------------------------

def test_south_america_run(tmp_path):
    """The South America case study runs and writes complete results."""
    sbayes_main(
        config=SOUTH_AMERICA_CONFIG,
        custom_settings=settings(tmp_path),
        experiment_name="test_south_america_run",
    )

    run_dirs = sorted((tmp_path / "test_south_america_run").glob("K*"))
    assert run_dirs, list(tmp_path.iterdir())
    for run_dir in run_dirs:
        assert_run_is_complete(run_dir, n_samples=20, n_runs=2)


def test_custom_settings_as_args(tmp_path):
    """Cluster counts and the run index given as arguments reach the output files."""
    sbayes_main(
        config=SOUTH_AMERICA_CONFIG,
        custom_settings=settings(tmp_path, mcmc={"runs": 1}),
        experiment_name="test_custom_settings",
        i_run=7,
        n_clusters=[2, 3],
    )

    experiment_dir = tmp_path / "test_custom_settings"
    assert {p.name for p in experiment_dir.glob("K*")} == {"K2", "K3"}

    for k in (2, 3):
        run_dir = experiment_dir / f"K{k}"
        assert (run_dir / "samples_7.h5").is_file(), [p.name for p in run_dir.iterdir()]
        assert_run_is_complete(run_dir, n_samples=20, n_runs=1)


def test_sample_prior(tmp_path):
    """Sampling from the prior runs end to end and ignores the observed features."""
    sbayes_main(
        config=SOUTH_AMERICA_CONFIG,
        custom_settings=settings(tmp_path, model={"sample_from_prior": True}),
        experiment_name="test_sample_prior",
    )

    run_dirs = sorted((tmp_path / "test_sample_prior").glob("K*"))
    assert run_dirs, list(tmp_path.iterdir())

    for run_dir in run_dirs:
        for stats_path in run_dir.glob("stats_*.tsv"):
            stats = pd.read_csv(stats_path, sep="\t")
            assert len(stats) == 20
            # No likelihood is evaluated when sampling from the prior
            assert "log_likelihood" not in stats.columns
            assert np.isfinite(stats["log_posterior"]).all()


# --- Simulation, inference and validation --------------------------------------------

def test_simulation_and_run(tmp_path):
    """Simulated data can be inferred from, and the two sides can be compared.

    This is the only test with known ground truth. It checks that the chain
    simulate -> write -> infer -> validate holds together and that every simulated
    parameter finds its inferred counterpart; the accuracy of the inference is a
    question for the calibration study, not for a 20-sample run.
    """
    sim_config = load_config(SIMULATE_CONFIG_PATH)
    sim_config.simulation.output.path = tmp_path

    simulator = Simulator(sim_config)
    simulator.prepare_simulation()
    simulator.simulate(random.PRNGKey(RNG_SEED))
    simulator.write_simulation()
    simulator.infer()

    simulated_k = sim_config.simulation.clusters
    clusters = sim_config.inf_config.model.clusters
    k = clusters if isinstance(clusters, int) else clusters[0]

    comparisons = read_simulations(tmp_path, simulated_k=simulated_k, k=k, burn_in=0)
    assert len(comparisons) == sim_config.simulation.datasets

    for comparison in comparisons.values():
        parameters = list(compare_parameters(comparison))
        assert parameters
        assert all(np.isfinite(p.simulated) for p in parameters)
        assert all(np.isfinite(p.inferred).all() for p in parameters)


def test_simulated_parameters_are_reproducible(tmp_path):
    """Writing the same simulation twice gives the same ground truth.

    The hard cluster assignments in the parameter files are drawn from NumPy's global
    generator, which `write_simulation` seeds from `simulation.seed`.
    """
    stats = []
    for i in (0, 1):
        sim_config = load_config(SIMULATE_CONFIG_PATH)
        sim_config.simulation.output.path = tmp_path / f"attempt_{i}"

        simulator = Simulator(sim_config)
        simulator.prepare_simulation()
        simulator.simulate(random.PRNGKey(RNG_SEED))
        simulator.write_simulation(write_data=False)

        stats_path = next((tmp_path / f"attempt_{i}").rglob("stats_*.tsv"))
        stats.append(pd.read_csv(stats_path, sep="\t"))

    pd.testing.assert_frame_equal(stats[0], stats[1])


def test_experiment_name_is_used_for_the_results_directory(tmp_path):
    """A given `experiment_name` names the results directory, not a timestamp."""
    sbayes_main(
        config=SOUTH_AMERICA_CONFIG,
        custom_settings=settings(tmp_path, mcmc={"runs": 1}),
        experiment_name="my_experiment",
    )
    assert (tmp_path / "my_experiment").is_dir(), [p.name for p in tmp_path.iterdir()]