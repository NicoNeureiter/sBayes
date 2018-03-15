#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import math
import numpy as np

from scipy.stats import binom_test, binom, norm as gaussian, gamma
from src.util import triangulation
from src.config import FEATURE_LL_MODE


EPS = np.finfo(float).eps


# Nico
# -> change to Google stylk, change settings docstring style to google
def binom_ll(features):
    """Compute the maximum log-likelihood of a binomial distribution.

    Args:
        features (np.ndarray): n*m array containing m features of n samples.

    Returns:

    """
    k = features.sum(axis=0)
    n = features.shape[0]
    p = k / n

    ll = binom.logpmf(k, n, p)

    return np.sum(ll)


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
    # features_rest = features[~zone, :]

    ll_zone = binom_ll(features_zone)
    # ll_rest = binom_ll(features_rest)

    return ll_zone


def compute_likelihood(zone, features, ll_lookup):

    """ This function computes the feature likelihood of a zone. The function performs a one-sided binomial test yielding
    a p-value for the observed presence of a feature in the zone given the presence of the feature in the data.
    The likelihood of a zone is the negative logarithm of this p-value. Zones with more present features than expected
    by the presence of a feature in the data have a small p-value and are thus indicative of language contact.

    Args:
        zone(np.ndarray): The current zone (boolean array)
        features(np.ndarray): The feature matrix
        ll_lookup(dict): A lookup table of likelihoods for all features, given a specific zone size

    Returns:
        log_lh(float): The feature-likelihood of the zone
    """

    idx = zone.nonzero()[0]
    zone_size = len(idx)

    # Count the presence and absence
    present = features[idx].sum(axis=0)
    log_lh = 0

    for f_idx, f_freq in enumerate(present):
        log_lh += ll_lookup[f_idx][zone_size][f_freq]

    return log_lh

# Nico
# -> change to Google style, change settings docstring style to google
# flags einbauen
def lookup_log_likelihood(min_size, max_size, feat_prob):
    """This function generates a lookup table of likelihoods
    :In
    - min_size: the minimum number of languages in a zone
    - max_size: the maximum number of languages in a zone
    - feat_prob: the probability of a feature to be present
    :Out
    - lookup_dict: the lookup table of likelihoods for a specific feature, sample size and observed presence
    """
    if FEATURE_LL_MODE == 'binom_test_2':
        # The binomial test computes the p-value of having k or more (!) successes out of n trials,
        # given a specific probability of success
        # For a two-sided binomial test, simply remove "greater"
        def ll(p_zone, s, p_global):
            return math.log(1 - binom_test(p_zone, s, p_global, 'greater') + EPS)

    else:
        # This version of the ll is more sensitive to exceptional observations.
        def ll(p_zone, s, p_global):
            return - math.log(binom_test(p_zone, s, p_global, 'greater'))

    lookup_dict = {}
    for i_feat, p_global in enumerate(feat_prob):
        lookup_dict[i_feat] = {}
        for s in range(min_size, max_size + 1):
            lookup_dict[i_feat][s] = {}
            for p_zone in range(s + 1):
                lookup_dict[i_feat][s][p_zone] = ll(p_zone, s, p_global)

    return lookup_dict

# Nico
# Empirische durschnitteliche Varianz um Gausssche Verteilung zu definieren
def compute_geo_likelihood(zone: np.ndarray, network: dict, std: np.ndarray):
    """

    Args:
        zone (np.ndarray): boolean array representing the current zone
        network (dict): network containing the graph, location,...
        std (np.ndarray): Standard deviation of the gaussian distribution

    Returns:
        float: geographical likelihood
    """
    locations = network['locations']
    locations_zone = locations[zone]

    mu = np.mean(locations_zone, axis=0)

    ll = np.mean(gaussian.logpdf(locations_zone, loc=mu, scale=std))

    return ll


def compute_empirical_geo_likelihood(zone: np.ndarray, net: dict, ecdf_geo: dict, type: str):

    """ This function computes the empirical geo-likelihood of a zone.
    Args:
        zone (np.ndarray): The current zone (boolean array)
        net (dict): A network containing the graph, location,...
        ecdf_geo (dict): The empirically generated geo-likelihood functions
        type (str): Defines which subgraph is used to compute the geo-likelihood, either
                    "minimum spanning tree", "delaunay" or "complete"
    Returns:
        geo_lh(float): The geo-likelihood of the zone
    """
    v = zone.nonzero()[0]

    if type == "complete":

        dist_mat = net['dist_mat']
        dist_mat_zone = dist_mat[zone][:, zone]
        d = dist_mat_zone.sum()

    else:
        triang = triangulation(net, zone)

        if type == "delaunay":
            d = sum(triang.es['weight'])

        elif type == "minimum spanning tree":
            mst = triang.spanning_tree(weights=triang.es["weight"])
            d = sum(mst.es['weight'])

    # Compute likelihood
    x = ecdf_geo[len(v)][type]['fitted_gamma']
    geo_lh = -np.log(gamma.cdf(d, *x))

    return geo_lh

