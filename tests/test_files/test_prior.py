"""Tests for sbayes/model/geo_prior.py and the pure helpers in prior.py."""
import jax.numpy as jnp
import numpy as np
import pytest

from sbayes.config.config import CategoricalPriorConfig, GaussianPriorConfig
from sbayes.load_data import FeatureType
from sbayes.model.geo_prior import average_max_distance, first_k_continuous
from sbayes.model.prior import (
    DEFAULT_GROUP, parse_dirichlet_concentration, require_prior_config,
    resolve_group_configs,
)


class TestParseDirichletConcentration:

    def test_uniform(self):
        config = CategoricalPriorConfig.model_validate({"type": "uniform"})
        concentration = parse_dirichlet_concentration(config, shape=(2, 3))
        assert concentration.shape == (2, 3)
        assert jnp.all(concentration == 1.0)

    def test_symmetric_dirichlet(self):
        config = CategoricalPriorConfig.model_validate(
            {"type": "symmetric_dirichlet", "prior_concentration": 2.5}
        )
        concentration = parse_dirichlet_concentration(config, shape=(2, 3))
        assert jnp.all(concentration == 2.5)

    def test_custom_dirichlet(self):
        config = CategoricalPriorConfig.model_validate({
            "type": "dirichlet",
            "parameters": {
                "f1": {"A": 1.0, "B": 2.0},
                "f2": {"A": 3.0, "B": 4.0},
            },
        })
        feature_names = {"f1": ["A", "B"], "f2": ["A", "B"]}
        concentration = parse_dirichlet_concentration(
            config, shape=(2, 2), feature_names=feature_names
        )
        assert np.allclose(concentration, [[1.0, 2.0], [3.0, 4.0]])

    def test_custom_dirichlet_requires_feature_names(self):
        config = CategoricalPriorConfig.model_validate({
            "type": "dirichlet", "parameters": {"f1": {"A": 1.0}},
        })
        with pytest.raises(ValueError, match="requires `feature_names`"):
            parse_dirichlet_concentration(config, shape=(1, 1))

    def test_custom_dirichlet_shape_mismatch(self):
        """The parameters must cover exactly the features and states of the partition."""
        config = CategoricalPriorConfig.model_validate({
            "type": "dirichlet", "parameters": {"f1": {"A": 1.0, "B": 2.0}},
        })
        with pytest.raises(ValueError, match="shape"):
            parse_dirichlet_concentration(
                config, shape=(2, 2), feature_names={"f1": ["A", "B"]}
            )

    def test_missing_state_is_reported(self):
        config = CategoricalPriorConfig.model_validate({
            "type": "dirichlet", "parameters": {"f1": {"A": 1.0}},
        })
        with pytest.raises(ValueError, match="state"):
            parse_dirichlet_concentration(
                config, shape=(1, 2), feature_names={"f1": ["A", "B"]}
            )


class TestResolveGroupConfigs:

    @staticmethod
    def uniform_config() -> CategoricalPriorConfig:
        return CategoricalPriorConfig.model_validate({"type": "uniform"})

    def test_each_group_keeps_its_own_config(self):
        configs = {"a": self.uniform_config(), "b": self.uniform_config()}
        resolved = resolve_group_configs(configs, ["a", "b"], FeatureType.categorical)
        assert list(resolved) == ["a", "b"]
        assert resolved["a"] is configs["a"]

    def test_missing_group_falls_back_to_default(self):
        default = self.uniform_config()
        configs = {"a": self.uniform_config(), DEFAULT_GROUP: default}
        resolved = resolve_group_configs(configs, ["a", "b"], FeatureType.categorical)
        assert resolved["b"] is default

    def test_groups_are_returned_in_the_given_order(self):
        configs = {DEFAULT_GROUP: self.uniform_config()}
        resolved = resolve_group_configs(
            configs, ["c", "a", "b"], FeatureType.categorical
        )
        assert list(resolved) == ["c", "a", "b"]

    def test_missing_group_without_default_raises(self):
        configs = {"a": self.uniform_config()}
        with pytest.raises(ValueError, match="group 'b'"):
            resolve_group_configs(configs, ["a", "b"], FeatureType.categorical)

    def test_the_default_is_not_a_group(self):
        """The `<DEFAULT>` entry itself is never returned as a group."""
        configs = {"a": self.uniform_config(), DEFAULT_GROUP: self.uniform_config()}
        resolved = resolve_group_configs(configs, ["a"], FeatureType.categorical)
        assert DEFAULT_GROUP not in resolved


class TestRequirePriorConfig:

    def test_returns_the_config_unchanged(self, data):
        partition = data.features.partitions[0]
        config = CategoricalPriorConfig.model_validate({"type": "uniform"})
        assert require_prior_config(config, partition, "cluster_effect") is config

    def test_missing_config_names_the_feature_type(self, data):
        partition = data.features.partitions[0]
        with pytest.raises(ValueError) as exc_info:
            require_prior_config(None, partition, "cluster_effect")

        message = str(exc_info.value)
        assert partition.name in message
        assert str(partition.FEATURE_TYPE) in message
        assert "prior.cluster_effect" in message

    def test_missing_group_config_names_the_group(self, data):
        partition = data.features.partitions[0]
        with pytest.raises(ValueError, match="group 'family'"):
            require_prior_config(
                None, partition, "confounding_effects", group="family"
            )


class TestFirstKContinuous:

    def test_keeps_the_first_k_probability_mass(self):
        # 0.8 fits, then 0.2 of the next 0.3, and nothing after that
        result = first_k_continuous(jnp.array([0.8, 0.3, 0.6]), k=1.0)
        assert np.allclose(result, [0.8, 0.2, 0.0])
        assert np.isclose(result.sum(), 1.0)

    def test_less_mass_available_than_requested(self):
        result = first_k_continuous(jnp.array([0.3, 0.2]), k=1.0)
        assert np.allclose(result, [0.3, 0.2])

    def test_along_an_axis(self):
        probs = jnp.array([[0.8, 0.3], [0.1, 0.4]])
        result = first_k_continuous(probs, k=0.5, axis=-1)
        assert np.allclose(result, [[0.5, 0.0], [0.1, 0.4]])


class TestAverageMaxDistance:

    COST = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])

    def test_fuzzy_clusters(self):
        clusters = jnp.array([[0.3, 0.7], [0.4, 0.6], [0.8, 0.2]])
        result = average_max_distance(clusters, self.COST)
        assert np.allclose(result, [3.02, 2.0], atol=1e-4)

    def test_one_value_per_cluster(self):
        clusters = jnp.full((3, 4), 0.5)
        assert average_max_distance(clusters, self.COST).shape == (4,)

    def test_a_single_object_has_no_distance(self):
        clusters = jnp.array([[1.0], [0.0], [0.0]])
        assert np.allclose(average_max_distance(clusters, self.COST), [0.0])