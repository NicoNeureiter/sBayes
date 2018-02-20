#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import math

import numpy as np
from scipy.stats import binom_test, binom, norm as gaussian

EPS = np.finfo(float).eps


def binom_ll(features):
    """Compute the maximum log-likelihood of a binomial distribution."""
    k = features.sum(axis=0)
    n = features.shape[0]
    p = k / n

    ll = binom.logpmf(k, n, p)

    return np.mean(ll)


def compute_likelihood_generative(zone, features, *args):
    """Log-likelihood of a generative model, where the zone and the rest of the
    languages are modeled as independent binomial distributions.

    Args:
        zone (np.ndarray): boolean array representing the zone
        features: feature matrix (boolean)

    Returns:
        float: log-likelihood of the zone
    """

    features_zone = features[zone, :]
    features_rest = features[~zone, :]

    ll_zone = binom_ll(features_zone)
    ll_rest = binom_ll(features_rest)

    return ll_zone + ll_rest


def compute_likelihood(zone, features, ll_lookup):
    """This function computes the likelihood of a zone.
    The function performs a two-sided binomial test. The test computes the probability of
    the observed presence/absence in the zone given the presence/absence in the data.
    Then the function takes the logarithm of the binomial test and multiplies it by -1.
    If the data in the zone follow the general trend, the zone is not indicative of
    language contact. When the binomial test yields a high value, the negative logarithm
    is small."""

    # Retrieve all languages in the zone
    idx = zone.nonzero()[0]
    zone_size = len(idx)

    # Count the presence and absence
    present = features[idx].sum(axis=0)
    log_lh = 0

    for f_idx, f_freq in enumerate(present):
        log_lh += ll_lookup[f_idx][zone_size][f_freq]

    return log_lh


def lookup_log_likelihood(min_size, max_size, feat_prob):
    """This function generates a lookup table of likelihoods
    :In
    - min_size: the minimum number of languages in a zone
    - max_size: the maximum number of languages in a zone
    - feat_prob: the probability of a feature to be present
    :Out
    - lookup_dict: the lookup table of likelihoods for a specific feature, sample size and presence
    """

    lookup_dict = {}
    for i_feat, p_global in enumerate(feat_prob):
        lookup_dict[i_feat] = {}
        for s in range(min_size, max_size + 1):
            lookup_dict[i_feat][s] = {}
            for p_zone in range(s + 1):
                lookup_dict[i_feat][s][p_zone] = \
                    - math.log(binom_test(p_zone, s, p_global))
                    # math.log(1 - binom_test(p_zone, s, p_global) + EPS)

    return lookup_dict


def compute_geo_likelihood(zone: np.ndarray, network: dict):
    """

    Args:
        zone (np.ndarray): boolean array representing the current zone
        network (dict): network containing the graph, location,...

    Returns:
        float: geographical likelihood
    """
    locations = network['locations']
    locations_zone = locations[zone]

    n = locations.shape[0]
    k = locations_zone.shape[0]

    mu = np.mean(locations_zone, axis=0)
    std = np.std(locations, axis=0) * 0.1  # * (k/n + EPS)**0.5

    ll = np.mean(gaussian.logpdf(locations_zone, loc=mu, scale=std))

    return ll


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    for n in [10, 50, 100, 500]:
        k = np.arange(n+1)
        p = k / n
        plt.plot(p, np.log2(binom.pmf(k, n, p)))

    # n = 200
    # for l in [20, 60]:
    #     k = np.arange(n + 1)
    #     plt.plot(k, poisson.pmf(k, l))
    #     plt.plot(k, binom.pmf(k, n, l/n), '--')

    plt.tight_layout(True)
    plt.show()