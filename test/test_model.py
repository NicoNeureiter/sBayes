#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import unittest
import matplotlib.pyplot as plt

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

        likelihood = GenerativeLikelihood(features)
        lh_with_family = likelihood(sample_with_family, features, inheritance=True, families=families, caching=False)
        lh_without_family = likelihood(sample_without_family, features, inheritance=False, caching=False)
        self.assertAlmostEqual(lh_with_family, lh_without_family)

        # Direct LH computation
        p_mixed = 0.4*p_global + 0.6*p_areas
        p_per_feature = np.repeat(p_global, N_SITES, axis=0)
        p_per_feature[areas[0]] = np.repeat(p_mixed, np.count_nonzero(areas), axis=0)
        lh_direct = np.sum(np.log(p_per_feature[features]))
        self.assertAlmostEqual(lh_with_family, lh_direct)

    def test_family_area_overlap_2(self):
        N_SITES = 3
        N_FEATURES = 100
        N_CATEGORIES = 10

        # Define area and family
        areas = np.zeros((1, N_SITES), dtype=bool)
        areas_noverlap = np.zeros((1, N_SITES), dtype=bool)
        families = np.zeros((1, N_SITES), dtype=bool)

        areas[0, :(N_SITES//2 + 1)] = True
        areas_noverlap[0, :(N_SITES//2)] = True
        families[0, (N_SITES//2):] = True

        print()
        print(areas)
        print(areas_noverlap)
        print(families)

        p_global = np.random.dirichlet(np.ones(N_CATEGORIES), size=(1, N_FEATURES))
        p_areas = np.random.dirichlet(np.ones(N_CATEGORIES), size=(1, N_FEATURES))
        p_families = np.random.dirichlet(np.ones(N_CATEGORIES), size=(1, N_FEATURES))
        # q = np.linspace(1, 0, N_CATEGORIES, endpoint=False)**10.
        # q = q / np.sum(q, keepdims=True)
        # q *= 1
        # p_areas = np.random.dirichlet(q, size=(1, N_FEATURES))
        # p_families = np.random.dirichlet(q[::-1], size=(1, N_FEATURES))

        w_g, w_a, w_f = 0., 0.5, 0.5
        weights = broadcast_weights([w_g, w_a, w_f], N_FEATURES)

        x1 = []
        x2 = []
        for _ in range(100):
            # Generate features from one shared distribution
            features = np.zeros((N_SITES, N_FEATURES, N_CATEGORIES))
            for i in range(N_SITES):
                p_mixture = w_g * p_global[0]
                w_sum = w_g

                if areas[0, i]:
                    p_mixture += w_a * p_areas[0]
                    w_sum += w_a

                if families[0, i]:
                    p_mixture += w_f * p_families[0]
                    w_sum += w_f

                p_mixture /= w_sum

                features[i] = generate_features((N_FEATURES, ), N_CATEGORIES, p=p_mixture)

            sample_overlap = Sample(areas, weights, p_global, p_areas, p_families)
            sample_noverlap = Sample(areas_noverlap, weights, p_global, p_areas, p_families)

            likelihood = GenerativeLikelihood(features)
            lh_overlap = likelihood(sample_overlap, inheritance=True, families=families, caching=False)
            lh_noverlap = likelihood(sample_noverlap, inheritance=True, families=families, caching=False)
            x1.append(lh_overlap)
            x2.append(lh_noverlap)

            print()
            print(lh_overlap)
            print(lh_noverlap)

        plt.hist([x1, x2], label=['Overlap', 'No overlap'])
        plt.legend()
        plt.show()

        # self.assertTrue(lh_noverlap < lh_overlap)


if __name__ == '__main__':
    unittest.main()
