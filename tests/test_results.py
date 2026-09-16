"""Tests for sbayes/results.py."""

import numpy as np
import pytest

from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMCSetup
from sbayes.results import Results, concat_dicts_recursive, match_clusters
from pathlib import Path


TEST_DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def h5_path(tmp_path_factory) -> Path:
    """Run a short analysis once and return the path to its samples file."""
    results_root = tmp_path_factory.mktemp("results")

    with Experiment(
        config_file=TEST_DATA / "config.yaml",
        experiment_name="test",
        custom_settings={
            "mcmc": {"steps": 20, "samples": 10, "warmup": {"warmup_steps": 10}},
            "results": {"path": str(results_root)},
        },
        log=False,
    ) as experiment:
        data = Data.from_config(experiment.config)
        setup = MCMCSetup(data=data, experiment=experiment)
        setup.sample(run=0)

    return setup.path_results / "samples_0.h5"


def test_h5_round_trip(h5_path, config):
    """Results read back from an h5 file describe the run that wrote it.

    This is the contract between `_build_h5_metadata` and `Results.from_h5`: the effect
    parameters are found through the keys in the metadata, so a mismatch between the two
    would silently produce empty or wrong effects.
    """

    data = Data.from_config(config)
    results = Results.from_h5(h5_path, burn_in=0.0)

    assert results.n_samples == 10
    assert results.n_clusters == config.model.clusters
    assert results.n_objects == len(data.objects.id)
    assert results.feature_names == list(data.features.names)
    assert set(results.groups_by_confounders) == set(data.confounders)

    # Every cluster has effects for every feature, and every feature has its states
    assert set(results.areal_effect) == set(results.cluster_names)
    for cluster_name in results.cluster_names:
        effects = results.areal_effect[cluster_name]
        assert set(effects) == set(results.feature_names)
        for feature_name, samples in effects.items():
            n_states = len(results.get_states_for_feature_name(feature_name))
            assert samples.shape == (results.n_samples, n_states)

    # Every confounder group has effects too
    for conf_name, group_names in results.groups_by_confounders.items():
        assert set(results.confounding_effects[conf_name]) == set(group_names)

    assert results.weights[results.feature_names[0]].shape == (
        results.n_samples, len(results.groups_by_confounders) + 1
    )


def test_h5_burn_in_and_subsampling(tmp_path, h5_path):
    """Burn-in is applied before subsampling, on the full chain."""
    full = Results.from_h5(h5_path, burn_in=0.0)
    assert full.n_samples == 10

    burned = Results.from_h5(h5_path, burn_in=0.2)
    assert burned.n_samples == 8

    subsampled = Results.from_h5(h5_path, burn_in=0.2, subsample_interval=2)
    assert subsampled.n_samples == 4

    with pytest.raises(ValueError, match="burn_in"):
        Results.from_h5(h5_path, burn_in=1.0)
    with pytest.raises(ValueError, match="subsample_interval"):
        Results.from_h5(h5_path, subsample_interval=0)


def test_read_stats_subsampling(tmp_path):
    """Subsampling keeps every n-th sample, starting from the first."""
    stats_path = tmp_path / "stats.tsv"
    rows = "\n".join(f"{i}\t{i * 10}" for i in range(10))
    stats_path.write_text(f"Sample\tx\n{rows}\n")

    assert Results.read_stats(stats_path)["Sample"].tolist() == list(range(10))
    assert Results.read_stats(stats_path, subsample_interval=2)["Sample"].tolist() == [
        0, 2, 4, 6, 8
    ]
    assert Results.read_stats(stats_path, subsample_interval=3)["Sample"].tolist() == [
        0, 3, 6, 9
    ]


def test_match_clusters_aligns_swapped_labels():
    """Two samples describing the same clusters under swapped labels are aligned."""
    # Sample 0 assigns objects 0-1 to cluster 0; sample 1 assigns them to cluster 1
    z = np.array([
        [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.05, 0.9, 0.05]],
        [[0.05, 0.9, 0.05], [0.05, 0.9, 0.05], [0.9, 0.05, 0.05]],
    ])
    # One effect array, distinguishable per cluster
    effects = {"cluster_effect": np.array([[[1.0], [2.0]], [[2.0], [1.0]]])}

    match_clusters(z, effects)

    # After matching, both samples assign objects 0-1 to the same cluster
    assert np.argmax(z[0, 0, :-1]) == np.argmax(z[1, 0, :-1])
    # The effects were permuted along with the clusters
    assert np.allclose(effects["cluster_effect"][0], effects["cluster_effect"][1])


def test_concat_dicts_recursive():
    """Nested dictionaries are combined by concatenating their leaves."""
    a = {"w": np.zeros(2), "eff": {"a0": {"f1": np.zeros((2, 3))}}}
    b = {"w": np.ones(2), "eff": {"a0": {"f1": np.ones((2, 3))}}}

    combined = concat_dicts_recursive([a, b], np.concatenate)

    assert combined["w"].shape == (4,)
    assert combined["eff"]["a0"]["f1"].shape == (4, 3)
    assert combined["w"].tolist() == [0, 0, 1, 1]

    with pytest.raises(ValueError, match="empty"):
        concat_dicts_recursive([], np.concatenate)