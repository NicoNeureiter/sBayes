import pytest

from pathlib import Path
from sbayes.load_data import Data
from sbayes.config.config import SBayesConfig, ModelConfig
from sbayes.model.model import Model

TEST_DATA = Path(__file__).parent / "data"

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

@pytest.fixture
def build_model():
    """Factory: load the baseline config with `overrides` applied and build the model.

    The overrides are deep-merged onto the config dict before validation, so the result
    goes through the same path as a config file.
    """
    def _build(overrides: dict | None = None) -> Model:
        config = SBayesConfig.from_config_file(
            TEST_DATA / "config.yaml", custom_settings=overrides
        )
        return Data.from_config(config), config

    def _build_model(overrides: dict | None = None) -> Model:
        data, config = _build(overrides)
        return Model(data, config.model)

    return _build_model

@pytest.fixture
def config() -> SBayesConfig:
    """The small hand-written test config."""
    return SBayesConfig.from_config_file(TEST_DATA / "config.yaml")


@pytest.fixture
def data(config) -> Data:
    """Data loaded from the small hand-written test dataset."""
    return Data.from_config(config)

@pytest.fixture
def model_config(prior_dict) -> ModelConfig:
    """A minimal ModelConfig with a single cluster count."""
    return ModelConfig.model_validate({"clusters": 2, "prior": prior_dict})

@pytest.fixture
def model_config_sample_from_prior(prior_dict) -> ModelConfig:
    """The same config, but sampling from the prior."""
    return ModelConfig.model_validate({
        "clusters": 2,
        "prior": prior_dict,
        "sample_from_prior": True,
    })