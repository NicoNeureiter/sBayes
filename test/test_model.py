#!/usr/bin/env python3
import unittest

import numpy as np
import numpy.testing
from numpy.typing import NDArray

from sbayes.model import Likelihood, ModelShapes, SourcePrior
from sbayes.sampling.state import Sample
from sbayes.util import log_multinom
from sbayes.load_data import Data, Objects, Features, Confounder


def binary_encoding(data, n_categories=None) -> np.array:
    if n_categories is None:
        n_categories = np.max(data) + 1
    else:
        assert np.max(data) < n_categories

    onehot_vectors = np.eye(n_categories, dtype=bool)
    return onehot_vectors[data]


def generate_features(shape, n_categories, p=None) -> np.array:
    if p is None:
        alpha = np.ones(n_categories)
        p = np.random.dirichlet(alpha)

    if len(p.shape) == 1:
        features_int = np.random.choice(a=n_categories, size=shape, p=p)

    elif len(p.shape) == 2:
        n_features, _ = p.shape
        assert n_features == shape[-1]
        assert _ == n_categories

        features_int = np.zeros(shape, dtype=int)
        for i in range(n_features):
            features_int[..., i] = np.random.choice(
                a=n_categories, size=shape[:-1], p=p[i, :]
            )

    else:
        raise ValueError

    return binary_encoding(features_int, n_categories=n_categories)


def broadcast_weights(w, n_features):
    return np.repeat([w], n_features, axis=0)


def dummy_features_from_values(values: NDArray[bool]) -> Features:
    n_objects, n_features, n_states = values.shape
    feature_names = np.array([f"f{i}" for i in range(n_features)])
    applicable_states = np.ones((n_features, n_states), dtype=bool)
    feature_states = [[f"s{i}" for i in range(n_states)]] * n_features
    return Features(
        values=values,
        names=feature_names,
        states=applicable_states,
        state_names=feature_states,
        na_number=0,
    )


def dummy_applicable_states(n_features: int, n_states: int) -> NDArray[bool]:
    return np.ones((n_features, n_states), dtype=bool)


def dummy_objects(n_objects: int) -> Objects:
    return Objects(
        id=np.arange(n_objects),
        locations=np.random.random((n_objects, 2)),
        names=np.arange(n_objects).astype(str),
    )


def dummy_universal_confounder(n_objects: int) -> Confounder:
    return Confounder(
        name="universal",
        group_assignment=np.ones((1, n_objects), dtype=bool),
        group_names=["<ALL>"],
    )


def dummy_family_confounder(families: NDArray[bool]) -> Confounder:
    n_families = families.shape[0]
    return Confounder(
        name="family",
        group_assignment=families,
        group_names=[f"fam_{i}" for i in range(n_families)],
    )


class TestLikelihood(unittest.TestCase):

    """Test correctness of the model likelihood."""

    def test_minimal_example(self):
        """Test the likelihood of a minimal example. The minimal example only contains
        three objects, and a single fixed binary feature.
        """
        # Define the data shapes
        n_objects = 3
        n_features = 1
        n_states = 2
        n_clusters = 1

        # Define a single Confounder (universal)
        universal = dummy_universal_confounder(n_objects)
        confounders = {universal.name: universal}

        shapes = ModelShapes(
            n_clusters=n_clusters,
            n_sites=n_objects,
            n_features=n_features,
            n_states=n_states,
            n_confounders=1,
            n_groups={universal.name: universal.n_groups},
            states_per_feature=dummy_applicable_states(n_features, n_states),
        )

        # Simple checks on the shapes object
        assert shapes.n_clusters == 1
        assert shapes.n_sites == n_objects
        assert np.all(np.array(shapes.n_states_per_feature) == n_states)

        # Create the data
        feature_values = np.array([[[1, 0]], [[0, 1]], [[0, 1]]])
        features = dummy_features_from_values(feature_values)
        objects = dummy_objects(n_objects)
        data = Data(objects=objects, features=features, confounders=confounders)

        # Make sure dimensions match
        assert data.features.values.shape == (n_objects, n_features, n_states)
        assert data.features.names.shape == (n_features,)
        assert data.features.states.shape == (n_features, n_states)
        assert len(data.objects.id) == n_objects

        source_prior = SourcePrior(na_features=data.features.na_values)

        # Create a simple sample
        p_cluster = broadcast_weights([0.0, 1.0], n_features)[np.newaxis,...]
        p_global = np.full(shape=(1, n_features, n_states), fill_value=0.5)
        source = np.zeros((n_objects, n_features, 2), dtype=bool)
        sample = Sample.from_numpy_arrays(
            clusters=np.ones((1, n_objects),  dtype=np.bool),
            weights=broadcast_weights([0.5, 0.5], n_features),
            confounding_effects={"universal": p_global},
            confounders=confounders,
            source=source,
            feature_counts={'clusters': np.zeros((n_clusters, n_features, n_states)),
                            universal.name: np.zeros((universal.n_groups, n_features, n_states))},
        )

        """Comment on analytical solutions:
        
        Since weights are fixed at 0.5, the source probability is fixed at :
            P( source | weights ) = 2^(- n_objects * n_features) = 0.125 
        
        The cluster is fixed to include all languages, but we vary the `source` array. We 
        will go through the different cases: 
        """
        p_source = np.log(0.125)

        """1. no areal effect means that the likelihood is simply 50/50 for each feature."""
        likelihood_exact = 0.125
        with sample.source.edit() as s:
            s[..., 0] = 0  # index 0 is the area component
            s[..., 1] = 1  # index 1 is the universal component (first confounder)
        likelihood_sbayes = Likelihood(data=data, shapes=shapes)(sample, caching=False)
        np.testing.assert_almost_equal(likelihood_sbayes, np.log(likelihood_exact))
        assert source_prior(sample) == p_source, source_prior(sample)

        """2. assigning the observation of the second object to the cluster effect means 
        that this observation is perfectly explained, increasing the likelihood by a
        factor of 2."""
        likelihood_exact = np.log(0.25)
        with sample.source.edit() as s:
            s[1, :, :] = [[1, 0]]  # switch object 1 to the cluster effect
        likelihood_sbayes = Likelihood(data=data, shapes=shapes)(sample, caching=False)
        np.testing.assert_almost_equal(likelihood_sbayes, likelihood_exact)

        """3. assigning the second object to the cluster effect increases the likelihood
        by another factor of 2."""
        likelihood_exact = np.log(0.5)
        with sample.source.edit() as s:
            s[2, :, :] = [[1, 0]]  # switch object 2 to the cluster effect
        likelihood_sbayes = Likelihood(data=data, shapes=shapes)(sample, caching=False)
        np.testing.assert_almost_equal(likelihood_sbayes, likelihood_exact)

        """4. assigning the first object to the cluster effect results in a likelihood of 
        zero, i.e. a log-likelihood of -inf."""
        likelihood_exact = -np.inf  # == np.log(0.0)
        with sample.source.edit() as s:
            s[0, :, :] = [[1, 0]]  # switch object 1 to the cluster effect
        sample.everything_changed()
        likelihood_sbayes = Likelihood(data=data, shapes=shapes)(sample, caching=False)
        np.testing.assert_almost_equal(likelihood_sbayes, likelihood_exact)

        """5. Integrating over source (setting it to None) averages the component 
        likelihoods for each observation."""
        lh = 0.25 * 0.75 * 0.75
        sample._source = None
        likelihood_sbayes = Likelihood(data=data, shapes=shapes)(sample, caching=False)
        np.testing.assert_almost_equal(likelihood_sbayes, np.log(lh))


# def test_family_cluster_overlap(self):
    #     n_objects = 10
    #     n_features = 5
    #     n_states = 3
    #
    #     shapes = ModelShapes(
    #         n_clusters=1,
    #         n_sites=n_objects,
    #         n_features=n_features,
    #         n_states=n_states,
    #         states_per_feature=dummy_applicable_states(n_features, n_states),
    #     )
    #     objects = dummy_objects(n_objects)
    #
    #     # Generate features from one shared distribution
    #     feature_values = generate_features((n_objects, n_features), n_states)
    #     features = dummy_features_from_values(feature_values)
    #
    #     # Define area and family
    #     areas = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=bool)
    #     families = areas.copy()
    #
    #     confounder_universal = dummy_universal_confounder(n_objects)
    #
    #     confounder_families = dummy_family_confounder(families)
    #
    #     # Dummy ´Data´ class to pass features and families to the likelihood
    #     data_with_family = Data(
    #         objects=objects,
    #         features=features,
    #         confounders={
    #             "universal": confounder_universal,
    #             "family": confounder_families,
    #         },
    #     )
    #
    #     data_without_family = Data(
    #         objects=objects,
    #         features=features,
    #         confounders={"universal": confounder_universal},
    #     )
    #
    #     p_global = np.random.dirichlet(np.ones(n_states), size=(1, n_features))
    #     p_areas = np.random.dirichlet(np.ones(n_states), size=(1, n_features))
    #     p_families = p_areas.copy()
    #
    #     weights_with_family = broadcast_weights([0.4, 0.3, 0.3], n_features)
    #     weights_without_family = broadcast_weights([0.4, 0.6], n_features)
    #
    #     sample_with_family = Sample(
    #         clusters=areas,
    #         weights=weights_with_family,
    #         confounding_effects={"universal": p_global, "family": p_families},
    #         cluster_effect=p_areas,
    #     )
    #     sample_without_family = Sample(
    #         clusters=areas,
    #         weights=weights_without_family,
    #         confounding_effects={"universal": p_global},
    #         cluster_effect=p_areas,
    #     )
    #
    #     # Compute likelihood with and without family
    #     likelihood = Likelihood(data=data_with_family, shapes=shapes)
    #     lh_with_family = likelihood(sample_with_family, caching=False)
    #     likelihood = Likelihood(data=data_without_family, shapes=shapes)
    #     lh_without_family = likelihood(sample_without_family, caching=False)
    #     # self.assertAlmostEqual(lh_with_family, lh_without_family)
    #
    #     # Direct LH computation
    #     p_mixed = 0.4 * p_global + 0.6 * p_areas
    #     p_per_feature = np.repeat(p_global, n_objects, axis=0)
    #     p_per_feature[areas[0]] = np.repeat(p_mixed, np.count_nonzero(areas), axis=0)
    #     lh_direct = np.sum(np.log(p_per_feature[features.values]))
    #
    #     self.assertAlmostEqual(lh_without_family, lh_direct)


class TestSizePrior(unittest.TestCase):

    """Test correctness of the geo prior."""

    def test_symmetry(self):
        clusters = np.array(
            [[1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0]],
            dtype=bool,
        )

        n_clusters, n_sites = clusters.shape
        print(clusters.shape)
        sizes = np.sum(clusters, axis=-1)
        logp = -log_multinom(n_sites, sizes)
        for order in ([0, 2, 1], [1, 0, 2], [2, 1, 0]):
            print(clusters[order])
            sizes = np.sum(clusters[order], axis=-1)
            print(sizes)
            assert logp == -log_multinom(n_sites, sizes)


if __name__ == "__main__":
    unittest.main()
