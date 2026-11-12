"""Tests for sbayes/config/config.py.
"""
import json
import pytest

from pydantic import ValidationError
from sbayes.config.config import (
    GaussianMeanPriorConfig, GaussianVariancePriorConfig,
    PoissonPriorConfig, CategoricalPriorConfig, GeoPriorConfig, WeightsPriorConfig,
    ClusterPriorConfig, PriorConfig, ModelConfig, MC3Config, MCMCConfig, SBayesConfig,
)

def test_type_defaults_to_uniform():
    """Priors with a neutral default need no explicit `type`.

    A uniform prior requires no parameters, so it can be assumed silently.
    """
    for cls in (GeoPriorConfig, CategoricalPriorConfig, WeightsPriorConfig):
        assert cls().type.value == "uniform"


@pytest.mark.parametrize("config_cls", [
    GaussianMeanPriorConfig,
    GaussianVariancePriorConfig,
    ClusterPriorConfig,
    PoissonPriorConfig,
])
def test_type_is_required_without_a_neutral_default(config_cls):
    """Priors whose every type needs parameters must name the type explicitly.

    There is nothing safe to assume: `gaussian` needs `mu_0` and `sigma_0`, `gamma`
    needs `shape` and `rate`, and the parameterless alternatives (`jeffreys`,
    `inv-gamma`) are not implemented.
    """
    with pytest.raises(ValidationError, match="Field required"):
        config_cls()


def test_an_explicit_gaussian_mean_prior_is_accepted():
    """The positive case for the prior whose `type` is now required."""
    cfg = GaussianMeanPriorConfig(type="gaussian", parameters={"mu_0": 0.0, "sigma_0": 1.0})
    assert cfg.type.value == "gaussian"


def test_missing_parameters_are_rejected():
    """A prior type that needs parameters cannot be configured without them."""
    with pytest.raises(ValidationError, match="expects the parameters"):
        GaussianMeanPriorConfig(type="gaussian")
    with pytest.raises(ValidationError, match="expects the parameters"):
        GaussianVariancePriorConfig(type="gamma")
    with pytest.raises(ValidationError, match="expects the parameters"):
        PoissonPriorConfig(type="gamma")
    with pytest.raises(ValidationError, match="Provide `prior_concentration`"):
        CategoricalPriorConfig(type="symmetric_dirichlet")


def test_misspelled_parameters_are_rejected():
    """A typo is reported as both a missing and an unexpected parameter."""
    with pytest.raises(ValidationError, match="unexpected"):
        GaussianVariancePriorConfig(type="gamma", parameters={"shape": 2.0, "scale": 1.0})


def test_unimplemented_types_are_rejected():
    """Prior types the model cannot build are rejected when the config is loaded."""
    cases = [
        (GaussianVariancePriorConfig, "inv-gamma", "not implemented"),
        (GaussianVariancePriorConfig, "jeffreys", "not implemented"),
        (PoissonPriorConfig, "jeffreys", "not implemented"),
        (CategoricalPriorConfig, "logistic_normal", "not implemented"),
        (ClusterPriorConfig, "categorical", "no longer supported"),
        (WeightsPriorConfig, "dirichlet", "not supported for the weights"),
    ]
    for cls, prior_type, message in cases:
        with pytest.raises(ValidationError, match=message):
            cls(type=prior_type)


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
    with pytest.raises(ValidationError, match="No prior is defined for the confounder"):
        ModelConfig(confounders=["family"], prior=prior_dict)


def test_sample_from_prior_without_categorical_prior(prior_dict):
    # Previously an AttributeError: cluster_effect.categorical is None.
    cfg = ModelConfig(prior=prior_dict, sample_from_prior=True)
    assert cfg.prior.cluster_effect.categorical is None


def test_use_parameter_transformation_is_deprecated(prior_dict):
    prior_dict["cluster_effect"] = {
        "categorical": {"type": "uniform", "use_parameter_transformation": False}
    }
    with pytest.warns(UserWarning, match="use_parameter_transformation"):
        cfg = ModelConfig(prior=prior_dict)
    assert not hasattr(cfg.prior.cluster_effect.categorical, "use_parameter_transformation")


def test_offset_reparameterization_defaults_to_true():
    assert MCMCConfig().offset_reparameterization is True


def test_mass_matrix_settings_default_to_true():
    assert MCMCConfig().laplace_mass_matrix is True
    assert MCMCConfig().adapt_mass_matrix is True


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

    # .yml is routed to the YAML loader (the old check also matched e.g. "foo.myyml").
    yml_path = tmp_path / "config.yml"
    yml_path.write_text(json.dumps(config))  # JSON is valid YAML
    assert SBayesConfig.from_config_file(yml_path).mcmc.steps == 1000


def test_base_config_helpers():
    cfg = MCMCConfig(steps=1000, samples=100)
    assert cfg["steps"] == 1000
    assert MCMCConfig.get_attr_doc("steps") is None  # __attrdocs__ not harvested
    assert MCMCConfig.annotations("steps") == "PositiveInt"


def test_model_config_rejects_a_confounder_without_a_prior(prior_dict):
    """Every confounder needs a matching `prior.confounding_effects` section."""
    with pytest.raises(ValidationError, match=r"No prior is defined.*'family'"):
        ModelConfig(confounders=["family"], prior=prior_dict)

