import logging
from types import SimpleNamespace

import pytest

from sbayes.config.config import SBayesConfig


@pytest.fixture
def prior_dict() -> dict:
    """A minimal valid prior config section."""
    return {
        "confounding_effects": {},
        "cluster_effect": {},
        "geo": {},
        "cluster_assignment": {"type": "dirichlet"},
        "weights": {},
    }


@pytest.fixture
def minimal_config(tmp_path, prior_dict) -> SBayesConfig:
    """A minimal valid config, sufficient to construct Experiment/MCMCSetup."""
    return SBayesConfig(
        model={"prior": prior_dict},
        mcmc={"steps": 1000, "samples": 100},
        results={"path": str(tmp_path / "results")},
        simulation=True,
    )