#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np

import scipy.stats as spstats
from scipy.sparse.csgraph import minimum_spanning_tree

from src.util import triangulation, cache_arg, hash_bool_array, compute_delaunay

EPS = np.finfo(float).eps


def binom_ll(features):
    """Compute the maximum log-likelihood of a binomial distribution.

    Args:
        features (np.array): n*m array containing m features of n samples.

    Returns:
        float: Log-likelihood of MLE binomial distribution.
    """
    k = features.sum(axis=0)
    n = features.shape[0]
    p = (k / n).clip(EPS, 1-EPS)

    # ll = binom.logpmf(k, n, p)
    ll = k * np.log(p) + (n - k) * np.log(1 - p)

    return np.sum(ll)


def compute_likelihood_generative(zone, features, *args):
    """Log-likelihood of a generative model, where the zone is modeled as a
    binomial distributions.

    Args:
        zone (np.array): boolean array representing the zone
        features: feature matrix (boolean)

    Returns:
        float: log-likelihood of the zone
    """

    features_zone = features[zone, :]
    ll_zone = binom_ll(features_zone)

    return ll_zone


@cache_arg(0, hash_bool_array)
def compute_feature_likelihood(zone, features, ll_lookup):

    """ This function computes the feature likelihood of a zone. The function performs a one-sided binomial test yielding
    a p-value for the observed presence of a feature in the zone given the presence of the feature in the data.
    The likelihood of a zone is the negative logarithm of this p-value. Zones with more present features than expected
    by the presence of a feature in the data have a small p-value and are thus indicative of language contact.

    Args:
        zone(np.array): The current zone (boolean array).
        features(np.array): The feature matrix.
        ll_lookup(dict): A lookup table of likelihoods for all features for a specific zone size.

    Returns:
        float: The feature-likelihood of the zone.
    """

    idx = zone.nonzero()[0]
    zone_size = len(idx)

    # Count the presence and absence
    present = features[idx].sum(axis=0)
    log_lh = 0

    for f_idx, f_freq in enumerate(present):
        log_lh += ll_lookup[f_idx][zone_size][f_freq]

    return log_lh


@cache_arg(0, hash_bool_array)
def compute_geo_likelihood_generative(zone: np.array, network: dict, cov: np.array):
    """

    Args:
        zone (np.array): boolean array representing the current zone
        network (dict): network containing the graph, location,...
        cov (np.array): Covariance matrix of the multivariate gaussian.

    Returns:
        float: geographical likelihood
    """
    dist_mat = network['dist_mat'][zone][:, zone]
    locations = network['locations'][zone]

    delaunay = compute_delaunay(locations)
    mst = delaunay
    # mst = minimum_spanning_tree(delaunay.multiply(dist_mat))

    i1, i2 = mst.nonzero()
    diffs = locations[i1] - locations[i2]

    ll = spstats.multivariate_normal.logpdf(diffs, mean=[0, 0], cov=cov)

    return np.mean(ll)


@cache_arg(0, hash_bool_array)
def compute_geo_likelihood_particularity(zone: np.array, net: dict, ecdf_geo: dict, subgraph_type: str):

    """ This function computes the empirical geo-likelihood of a zone.
    Args:
        zone (np.array): The current zone (boolean array)
        net (dict):  The full network containing all sites.
        ecdf_geo (dict): The empirically generated geo-likelihood functions
        subgraph_type (str): Defines which subgraph is used to compute the geo-likelihood, either
                    "complete", "delaunay" or "mst" (short for "minimum spanning tree")
    Returns:
        geo_lh(float): The geo-likelihood of the zone
    """

    v = zone.nonzero()[0]

    if subgraph_type == "complete":

        dist_mat = net['dist_mat']
        dist_mat_zone = dist_mat[zone][:, zone]
        d = dist_mat_zone.sum()

    else:
        triang = triangulation(net, zone)

        if subgraph_type == "delaunay":
            d = triang.sum()

        elif subgraph_type == "mst":
            mst = minimum_spanning_tree(triang)
            d = mst.sum()

        else:
            raise ValueError('Unknown lh_type: %s' % subgraph_type)

    # Compute likelihood
    x = ecdf_geo[len(v)][subgraph_type]['fitted_gamma']

    geo_lh = -spstats.gamma.logcdf(d, *x)

    return geo_lh

