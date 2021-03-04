#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from collections import namedtuple
import unittest

from sbayes.model import GenerativeLikelihood
from sbayes.sampling.zone_sampling import Sample


def binary_encoding(data, n_categories=None):
    if n_categories is None:
        n_categories = np.max(data) + 1
    else:
        assert np.max(data) < n_categories

    onehot_vectors = np.eye(n_categories, dtype=bool)
    return onehot_vectors[data]


def generate_features(shape, n_categories, p=None):
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
            features_int[..., i] = np.random.choice(a=n_categories, size=shape[:-1], p=p[i, :])

    else:
        raise ValueError

    return binary_encoding(features_int, n_categories=n_categories)


def broadcast_weights(w, n_features):
    return np.repeat([w], n_features, axis=0)


class TestLikelihood(unittest.TestCase):

    """Test correctness of the model likelihood."""

    def test_family_area_overlap(self):
        N_SITES = 10
        N_FEATURES = 5
        N_CATEGORIES = 3

        # Generate features from one shared distribution
        features = generate_features((N_SITES, N_FEATURES), N_CATEGORIES)

        # Define area and family
        areas = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=bool)
        families = areas.copy()

        p_global = np.random.dirichlet(np.ones(N_CATEGORIES), size=(1, N_FEATURES))
        p_areas = np.random.dirichlet(np.ones(N_CATEGORIES), size=(1, N_FEATURES))
        p_families = p_areas.copy()

        weights_with_family = broadcast_weights([0.4, 0.3, 0.3], N_FEATURES)
        weights_without_family = broadcast_weights([0.4, 0.6], N_FEATURES)

        sample_with_family = Sample(areas, weights_with_family,
                                    p_global=p_global, p_zones=p_areas, p_families=p_families)
        sample_without_family = Sample(areas, weights_without_family,
                                       p_global=p_global, p_zones=p_areas, p_families=None)

        # Dummy ´Data´ class to pass features and families to the likelihood
        Data = namedtuple('Data', ['features', 'families'])
        data = Data(features=features, families=families)

        likelihood = GenerativeLikelihood(data=data, inheritance=True)
        lh_with_family = likelihood(sample_with_family, caching=False)
        likelihood = GenerativeLikelihood(data=data, inheritance=False)
        lh_without_family = likelihood(sample_without_family, caching=False)
        self.assertAlmostEqual(lh_with_family, lh_without_family)

        # Direct LH computation
        p_mixed = 0.4*p_global + 0.6*p_areas
        p_per_feature = np.repeat(p_global, N_SITES, axis=0)
        p_per_feature[areas[0]] = np.repeat(p_mixed, np.count_nonzero(areas), axis=0)
        lh_direct = np.sum(np.log(p_per_feature[features]))

        self.assertAlmostEqual(lh_with_family, lh_direct)


if __name__ == '__main__':
    unittest.main()
