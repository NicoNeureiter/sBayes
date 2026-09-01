"""Tests for sbayes/mcmc_setup.py."""

import logging
import pytest
from sbayes import mcmc_setup
from sbayes.mcmc_setup import MCMCSetup
from types import SimpleNamespace

@pytest.fixture
def experiment(tmp_path, monkeypatch, minimal_config):
    monkeypatch.setattr(
        mcmc_setup, "Model", lambda data, config: SimpleNamespace(n_clusters=3)
    )
    return SimpleNamespace(
        config=minimal_config,
        logger=logging.getLogger("test"),
        path_results=tmp_path,
    )


def test_results_dir_is_created_per_n_clusters(experiment):
    setup = MCMCSetup(data=None, experiment=experiment)
    assert setup.path_results == experiment.path_results / "K3"
    assert setup.path_results.is_dir()


def test_missing_results_dir_raises(experiment):
    experiment.path_results = None
    with pytest.raises(ValueError, match="results directory"):
        MCMCSetup(data=None, experiment=experiment)


def test_nested_results_dir_is_created(experiment):
    # parents=True: the parent may not exist yet
    experiment.path_results = experiment.path_results / "a" / "b"
    setup = MCMCSetup(data=None, experiment=experiment)
    assert setup.path_results.is_dir()