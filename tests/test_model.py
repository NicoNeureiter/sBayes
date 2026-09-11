"""Tests for sbayes/model/model.py."""

import jax.numpy as jnp
import numpy as np
import numpyro
import pytest

from jax import random
from pydantic import ValidationError

from sbayes.config.config import SBayesConfig
from sbayes.load_data import CategoricalFeatures, Data
from sbayes.model.model import Model, normalize, NO_GROUP

COST_BASED_GEO = {
    "type": "cost_based",
    "rate": 100_000.0,
    "probability_function": "exponential",
}

def test_clusters_must_be_a_single_int(data, config):
    config.model.clusters = [1, 2, 3]
    with pytest.raises(ValueError, match="single number of clusters"):
        Model(data, config.model)

def test_group_assignments_mark_objects_without_a_group(data, config):
    model = Model(data, config.model)
    assert model.group_assignments[0, 6] == NO_GROUP
    assert model._has_confounder_component[0, 6] == 0.0
    assert model._has_confounder_component[0, 0] == 1.0


def test_shapes_are_consistent(data, config):
    model = Model(data, config.model)
    assert model.shapes.n_objects == 7
    assert model.shapes.n_confounders == 2
    assert model.shapes.n_components == 3
    assert model.shapes.n_components_expanded == 4

def test_allow_parameterization_follows_sample_from_prior(data, config):
    assert Model(data, config.model).allow_reparameterization is True
    config.model.sample_from_prior = True
    assert Model(data, config.model).allow_reparameterization is False


def test_normalize():
    x = np.ones((2, 4))
    assert np.allclose(normalize(x), 0.25)
    assert np.allclose(normalize(x, axis=0), 0.5)

    # Rows are normalized independently
    x = np.array([[1.0, 3.0], [2.0, 2.0]])
    assert np.allclose(normalize(x), [[0.25, 0.75], [0.5, 0.5]])
    assert np.allclose(normalize(x).sum(axis=-1), 1.0)


def test_get_model_traces(data, config):
    """Every feature type produces a likelihood site, with the expected shapes."""

    model = Model(data, config.model)
    with numpyro.handlers.seed(rng_seed=0):
        trace = numpyro.handlers.trace(model.get_model).get_trace()

    assert trace["w"]["value"].shape == (
        model.shapes.n_features, model.shapes.n_components
    )
    for partition in model.partitions:
        assert f"x_{partition.name}" in trace


@pytest.mark.parametrize("skeleton", ["complete_graph", "spectral", "diameter"])
def test_aggregated_distance_is_finite(build_model, skeleton):
    """Every implemented skeleton aggregates the cluster distances to a finite value."""
    model = build_model({
        "model": {"prior": {"geo": COST_BASED_GEO | {"skeleton": skeleton}}}
    })

    # A plausible fuzzy assignment of 7 objects to 2 clusters
    clusters = jnp.array([
        [0.9, 0.1], [0.8, 0.2], [0.7, 0.3],
        [0.2, 0.8], [0.1, 0.9], [0.3, 0.7], [0.5, 0.5],
    ])

    distance = model.prior.geo_prior.aggregated_distance(clusters)

    assert jnp.isfinite(distance).all()
    assert (distance >= 0).all()


def test_mst_skeleton_is_rejected(build_model):
    """The `mst` skeleton is not supported for fuzzy cluster assignments."""
    with pytest.raises(ValidationError, match="mst"):
        build_model({"model": {"prior": {"geo": COST_BASED_GEO | {"skeleton": "mst"}}}})

def test_generate_initial_params_writes_existing_sites(data, config):
    """The empirical initialization must write to sites that exist in the model.

    The categorical confounding effects are named after the confounder, so a mismatch
    here would silently add unused keys instead of initializing anything.
    """
    model = Model(data, config.model)
    init_params = model.generate_initial_params(random.PRNGKey(0))

    with numpyro.handlers.seed(rng_seed=0):
        trace = numpyro.handlers.trace(model.get_model).get_trace()

    for name in init_params:
        assert name in trace, f"{name} is not a site of the model"


def test_generate_initial_params_initializes_confounding_effects(data, config):
    """Categorical confounding effects are initialized from the observed state counts."""
    model = Model(data, config.model)
    init_params = model.generate_initial_params(random.PRNGKey(0))

    categorical = [p for p in model.partitions if isinstance(p, CategoricalFeatures)]
    assert categorical, "the test data must contain categorical features"

    for conf in model.confounders.values():
        for partition in categorical:
            name = f"conf_effect_{conf.name}_{partition.name}"
            assert name in init_params

            value = jnp.asarray(init_params[name])
            assert value.shape == (
                conf.n_groups, partition.n_features, partition.n_states
            )
            assert jnp.isfinite(value).all()
            # Each row is a probability vector over the states
            assert np.allclose(value.sum(axis=-1), 1.0)