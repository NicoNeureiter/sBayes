"""Numerical soundness of the model across the settings that change its structure.

Each case is the baseline config with one setting overridden. All five bugs found in
the audit were specific to one such setting, so the coverage is in the matrix rather
than in the assertions.
"""
import jax
import jax.numpy as jnp
import numpyro
import pytest

from numpyro.infer.util import log_density
from pathlib import Path
from sbayes.model.model import Model

TEST_DATA = Path(__file__).parent / "data"

# One override per case, applied on top of the baseline config
MODEL_SETTINGS = {
    "baseline": {},
    "no_reparameterization": {
        "prior": {
            "cluster_assignment": {"dirichlet_config": {"use_parameter_transformation": False}},
            "cluster_effect": {"categorical": {"use_parameter_transformation": False}},
        }
    },
    "sample_from_prior": {"sample_from_prior": True},
    "stretch_and_clip": {"prior": {"cluster_assignment": {"stretch_and_clip": True}}},
    "varying_cluster_weights": {"prior": {"weights": {"varying_cluster_weights": True}}},
    "hierarchical_weights": {"prior": {"weights": {"hierarchical": True}}},
    "estimate_no_cluster_concentration": {
        "prior": {"cluster_assignment": {"estimate_no_cluster_concentration": True}}
    },
    "three_clusters": {"clusters": 3},
    "one_cluster": {"clusters": 1},
}

GEO_SETTINGS = {
    "uniform": {},
    "cost_based_exponential": {
        "type": "cost_based", "rate": 100_000.0,
        "probability_function": "exponential", "skeleton": "complete_graph",
    },
    "cost_based_sigmoid": {
        "type": "cost_based", "rate": 100_000.0, "inflection_point": 200_000.0,
        "probability_function": "sigmoid", "skeleton": "complete_graph",
    },
    "cost_based_spectral": {
        "type": "cost_based", "rate": 100_000.0,
        "probability_function": "exponential", "skeleton": "spectral",
    },
    "cost_based_diameter": {
        "type": "cost_based", "rate": 100_000.0,
        "probability_function": "exponential", "skeleton": "diameter",
    },
    "aggregation_sum": {
        "type": "cost_based", "rate": 100_000.0, "aggregation": "sum",
        "probability_function": "exponential", "skeleton": "complete_graph",
    },
}

def assert_model_is_samplable(model: Model) -> None:
    """Every site, the joint log-density and its gradient must be finite."""
    with numpyro.handlers.seed(rng_seed=0):
        trace = numpyro.handlers.trace(model.get_model).get_trace()

    params = {
        name: site["value"]
        for name, site in trace.items()
        if site["type"] == "sample" and not site.get("is_observed")
    }

    joint, _ = log_density(model.get_model, (), {}, params)
    assert jnp.isfinite(joint), "The joint log-density is not finite."

    # Discrete sites are not differentiable, so the gradient is taken with respect to
    # the continuous parameters only.
    continuous = {
        name: value for name, value in params.items()
        if jnp.issubdtype(jnp.asarray(value).dtype, jnp.floating)
    }
    discrete = set(params) - set(continuous)

    def joint_log_density(p: dict) -> jnp.ndarray:
        return log_density(model.get_model, (), {}, {**p, **{k: params[k] for k in discrete}})[0]

    gradient = jax.grad(joint_log_density)(continuous)
    for name, value in gradient.items():
        assert jnp.isfinite(jnp.asarray(value)).all(), (
            f"Site {name} has a non-finite gradient."
        )


@pytest.mark.parametrize("setting", MODEL_SETTINGS, ids=list(MODEL_SETTINGS))
def test_model_is_samplable(build_model, setting):
    """The model is numerically sound for each model setting."""
    assert_model_is_samplable(build_model({"model": MODEL_SETTINGS[setting]}))


@pytest.mark.parametrize("setting", GEO_SETTINGS, ids=list(GEO_SETTINGS))
def test_model_is_samplable_with_geo_prior(build_model, setting):
    """The model is numerically sound for each geo-prior setting."""
    assert_model_is_samplable(
        build_model({"model": {"prior": {"geo": GEO_SETTINGS[setting]}}})
    )