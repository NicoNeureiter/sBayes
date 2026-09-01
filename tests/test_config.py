"""Tests for sbayes/config/config.py.
"""
import json
import warnings
import pytest

from pydantic import ValidationError


from sbayes.config.config import (
    TypedPriorConfig, GaussianMeanPriorConfig, GaussianVariancePriorConfig,
    PoissonPriorConfig, CategoricalPriorConfig, GeoPriorConfig, WeightsPriorConfig,
    ClusterPriorConfig, PriorConfig, ModelConfig, MC3Config, MCMCConfig, SBayesConfig,
)


def test_default_type_warnings():
    for cls, name in [(GaussianMeanPriorConfig, "improper_uniform"),
                      (PoissonPriorConfig, "jeffreys"),
                      (CategoricalPriorConfig, "uniform"),
                      (GeoPriorConfig, "uniform"),
                      (WeightsPriorConfig, "uniform")]:
        with pytest.warns(UserWarning, match=f"Using `{name}` as a default"):
            cfg = cls()
        assert cfg.type.value == name
    # No warning when the type is given explicitly.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PoissonPriorConfig(type="gamma", parameters={"a": 1.0})


def test_required_type_has_no_warning_class():
    assert not issubclass(GaussianVariancePriorConfig, TypedPriorConfig)
    assert not issubclass(ClusterPriorConfig, TypedPriorConfig)
    with pytest.raises(ValidationError, match="Field required"):
        GaussianVariancePriorConfig()


def test_parameter_requirements():
    with pytest.raises(ValidationError, match="Provide `file` or `parameters`"):
        GaussianMeanPriorConfig(type="gaussian")
    with pytest.raises(ValidationError, match="Provide `file` or `parameters`"):
        GaussianVariancePriorConfig(type="inv-gamma")
    with pytest.raises(ValidationError, match="Provide `file` or `parameters`"):
        PoissonPriorConfig(type="gamma")
    with pytest.raises(ValidationError, match="Provide `prior_concentration`"):
        CategoricalPriorConfig(type="symmetric_dirichlet")
    with pytest.raises(ValidationError, match="Provide `logistic_normal_scale`"):
        CategoricalPriorConfig(type="logistic_normal")
    CategoricalPriorConfig(type="logistic_normal", logistic_normal_scale=2.0)


def test_geo_prior_requirements():
    with pytest.raises(ValidationError, match="`rate` is required"):
        GeoPriorConfig(type="cost_based")
    with pytest.raises(ValidationError, match="`inflection_point` is required"):
        GeoPriorConfig(type="cost_based", rate=1.0, probability_function="sigmoid")
    GeoPriorConfig(type="cost_based", rate=1.0, probability_function="sigmoid",
                   inflection_point=0.5)
    assert GeoPriorConfig(type="cost_based", rate=1.0).approx_norm_const["grid_size"] == 40


def test_gamma_tuple_conversion():
    cfg = WeightsPriorConfig(type="uniform", concentration_prior=(2.0, 4.0))
    assert (cfg.concentration_prior.shape, cfg.concentration_prior.rate,
            cfg.concentration_prior.offset) == (2.0, 4.0, 0.0)
    cfg = WeightsPriorConfig(type="uniform", concentration_prior=(2.0, 4.0, 1.0))
    assert cfg.concentration_prior.offset == 1.0
    with pytest.raises(ValidationError, match="must be given as a"):
        WeightsPriorConfig(type="uniform", concentration_prior=(1.0, 2.0, 3.0, 4.0))
    # The caller's dict is not mutated.
    values = {"type": "uniform", "concentration_prior": (2.0, 4.0)}
    WeightsPriorConfig(**values)
    assert values["concentration_prior"] == (2.0, 4.0)


def test_deprecated_attributes_warn_and_drop_without_mutating():
    values = {"chains": 2, "swap_attempts": 5}
    with pytest.warns(UserWarning, match="swap_attempts"):
        cfg = MC3Config(**values)
    assert values == {"chains": 2, "swap_attempts": 5}
    assert cfg.chains == 2


def test_non_dict_input_does_not_crash_validators():
    with pytest.raises(ValidationError, match="valid dict"):
        MC3Config.model_validate(["not", "a", "dict"])
    with pytest.raises(ValidationError, match="valid dict"):
        CategoricalPriorConfig.model_validate(42)


def test_renamed_objects_per_cluster_raises_value_error(prior_dict):
    prior = prior_dict | {"objects_per_cluster": {"type": "dirichlet"}}
    with pytest.raises(ValidationError, match="renamed to `cluster_assignment`"):
        PriorConfig(**prior)


def test_model_config_confounders(prior_dict):
    # Missing `confounders` key must not raise KeyError.
    ModelConfig(prior=prior_dict)
    # Missing `prior` key must give a clean pydantic error, not a KeyError.
    with pytest.raises(ValidationError, match="Field required"):
        ModelConfig(confounders=["family"])
    # An undeclared confounder prior is a ValueError, not a NameError.
    with pytest.raises(ValidationError, match="not defined in the config file"):
        ModelConfig(confounders=["family"], prior=prior_dict)


def test_sample_from_prior_without_categorical_prior(prior_dict):
    # Previously an AttributeError: cluster_effect.categorical is None.
    cfg = ModelConfig(prior=prior_dict, sample_from_prior=True)
    assert cfg.prior.cluster_effect.categorical is None
    # With a categorical prior present the transformation is switched off.
    prior = prior_dict
    prior["cluster_effect"] = {"categorical": {"type": "uniform"}}
    prior["confounding_effects"] = {"family": {"a": {"categorical": {"type": "uniform"}}}}
    cfg = ModelConfig(prior=prior, confounders=["family"], sample_from_prior=True)
    assert cfg.prior.cluster_effect.categorical.use_parameter_transformation is False
    conf = cfg.prior.confounding_effects["family"]["a"]
    assert conf.categorical.use_parameter_transformation is False


def test_mc3_prior_temperature_default():
    cfg = MC3Config(temperature_diff=0.2)
    assert cfg.prior_temperature_diff == 0.2
    cfg = MC3Config(temperature_diff=0.2, prior_temperature_diff=0.5)
    assert cfg.prior_temperature_diff == 0.5
    with pytest.raises(ValidationError):
        MC3Config(prior_temperature_diff=-1.0)


def test_mcmc_sample_spacing():
    MCMCConfig(steps=1000, samples=100)
    with pytest.raises(ValidationError, match="Inconsistent spacing"):
        MCMCConfig(steps=1000, samples=333)


def test_sbayes_config_data_optional(prior_dict):
    model = {"prior": prior_dict}
    mcmc = {"steps": 1000, "samples": 100}
    # Simulation configs may omit the whole data block (previously "Field required").
    cfg = SBayesConfig(model=model, mcmc=mcmc, simulation=True)
    assert cfg.data is None
    with pytest.raises(ValidationError, match="`data` block is required"):
        SBayesConfig(model=model, mcmc=mcmc)


def test_from_config_file_uses_cls_and_suffix(tmp_path,
                                              prior_dict):
    features = tmp_path / "features.csv"
    features.write_text("id,x,y\n")
    types = tmp_path / "feature_types.yaml"
    types.write_text("{}\n")
    config = {
        "data": {"features": "features.csv", "feature_types": "feature_types.yaml"},
        "model": {"prior": prior_dict},
        "mcmc": {"steps": 1000, "samples": 100},
        "results": {"path": str(tmp_path / "results")},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    class SubConfig(SBayesConfig):
        pass

    cfg = SubConfig.from_config_file(path)
    assert type(cfg) is SubConfig            # cls(), not SBayesConfig()
    assert cfg.data.features == features     # relative path resolved against config dir
    assert (tmp_path / "results").is_dir()   # results dir created by the validator

    # .yml is routed to the YAML loader (the old check also matched e.g. "foo.myyml").
    yml_path = tmp_path / "config.yml"
    yml_path.write_text(json.dumps(config))  # JSON is valid YAML
    assert SBayesConfig.from_config_file(yml_path).mcmc.steps == 1000


def test_base_config_helpers():
    cfg = MCMCConfig(steps=1000, samples=100)
    assert cfg["steps"] == 1000
    assert MCMCConfig.get_attr_doc("steps") is None  # __attrdocs__ not harvested
    assert MCMCConfig.annotations("steps") == "PositiveInt"