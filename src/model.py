#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
from scipy import sparse
import scipy.stats as spstats
from scipy.sparse.csgraph import minimum_spanning_tree

from src.util import triangulation, cache_arg, hash_bool_array, compute_delaunay

EPS = np.finfo(float).eps


def generalized_bernoulli_ll(features, features_cat):
    """Compute the maximum log-likelihood of a generalized Bernoulli distribution (categorical distribution)

    Args:
        features (np.array): n*m array containing m features of n samples.
        features_cat (list): all categories that the feature f can belong to
    Returns:
        float: Log-likelihood of MLE Bernoulli distribution.
    """

    n = features.shape[0]

    ll = []
    # NAs are assigned to all available categories in equal shares
    # If there are only NAs in the zone, there is no evidence for shared evolution
    # the same is true if there is an equal share of features per category

    # Count all NAs
    k_na = (features == -1).sum(axis=0)
    na_per_cat = k_na/features_cat

    # Assign in equal shares to all categories
    for cat in np.unique(features):
        # - 1 denotes missing data
        if cat != -1:
            k = (features == cat).sum(axis=0) + na_per_cat

            p = (k / n).clip(EPS, 1 - EPS)

            ll.append(k * np.log(p))

    # Old
    # k = features.sum(axis=0)
    # ll = k * np.log(p) + (n - k) * np.log(1 - p)
    return np.sum(ll)


# todo: Remove features_old once everything works, change p_global, alpha
def compute_likelihood_generative(zone, features, features_old, features_cat_old, weight, p_global=1,  *args):
        """Compute the likelihood of all sites in `zone`. The likelihood is
        defined as a mixture of the global distribution `p_global` and the maximum
        likelihood distribution of the zone itself.

        Args:
            features (np.array or 'SparseMatrix'): The feature values for all sites and features.
                shape: (n_sites, n_features, n_categories)
            zone (np.array): Binary array indicating the assignment of a site to the
                current zone.
                shape: (n_sites, )
            p_global (np.array): The global frequencies of every feature and every
                value.
                shape: (n_features, n_categories)
            weight (np.array): The mixture coefficient controlling how much each
                feature is explained by the zone, global distribution and
                language families distribution.
                shape: (n_features, 3)

        TODO:
            p_family (np.array): The probabilities of every category in every
                language family.
                shape: (n_sites, n_features, n_categories)
            families (np.array): Binary array indicating the assignment of a site to
                a language family.
                shape: (n_sites, n_families)
    
        Returns:
            float: The joint likelihood of all sites in the zone.
        """

        # Old likelihood
        features_old_zone = features_old[zone, :]
        ll_zone = generalized_bernoulli_ll(features_old_zone, features_cat_old)

        print('Old likelihood:', ll_zone)

        # New likelihood
        n_sites, n_features, n_categories = features.shape

        assert np.all(np.sum(weight, axis=-1) == 1.)

        idx = zone.nonzero()[0]
        zone_size = len(idx)

        features_zone = features[zone, :, :]
        p_zone = np.sum(features_zone, axis=0) / zone_size
        # p_zone.shape = (n_features, n_categories)

        # Division by zero could cause troubles
        p_zone = p_zone.clip(EPS, 1 - EPS)
        ll = 0.
        for i_f in range(n_features):

            # Get current features vector
            f = features_zone[:, i_f, :]
            # f.shape = (zone_size, n_categories)

            # Compute the feature likelihood vector (for all sites in zone)
            lh_zone = f.dot(p_zone[i_f, :])
            lh_global = f.dot(p_global[i_f, :])
            # lh_family = np.sum(f * p_family([:, i_f, :], axis=-1))
            # lh_family = lh_global            # TODO use real p_family

            # Todo add family
            lh_feature = weight[i_f, 0] * lh_zone\
                         + weight[i_f, 1] * lh_global \
                         # + weight[i_f, 2] * lh_family

            # take log, sum over sites and add to log-likelihood
            ll += np.sum(np.log(lh_feature))
        print('New likelihood:', ll)
        print(llk)

        return ll


@cache_arg(0, hash_bool_array)
def compute_likelihood_particularity(zone, features, ll_lookup):

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

    # Count the number of languages per category in the zone
    log_lh = []
    for f_idx, f in enumerate(np.transpose(features[idx])):

        bin_test_per_cat = []
        for cat in ll_lookup[f_idx].keys():

            # -1 denotes missing values
            #Todo Remove 0, only for testing
            if cat != -1 and cat != 0:
                cat_count = (f == cat).sum(0)
                # Perform Binomial test
                bin_test_per_cat.append(ll_lookup[f_idx][cat][zone_size][cat_count])

        # Keep the most surprising category per feature
        log_lh.append(max(bin_test_per_cat))

    return sum(log_lh)


@cache_arg(0, hash_bool_array)
def compute_feature_likelihood_old(zone, features, ll_lookup):

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
def compute_geo_prior_gaussian(zone: np.array, network: dict, cov: np.array):
    """
    This function computes the two-dimensional Gaussian geo-prior for all edges in the zone
    Args:
        zone (np.array): boolean array representing the current zone
        network (dict): network containing the graph, location,...
        cov (np.array): Covariance matrix of the multivariate gaussian.

    Returns:
        float: geographical prior
    """
    dist_mat = network['dist_mat'][zone][:, zone]
    locations = network['locations'][zone]

    delaunay = compute_delaunay(locations)
    mst = delaunay.multiply(dist_mat)
    # mst = minimum_spanning_tree(delaunay.multiply(dist_mat))

    i1, i2 = mst.nonzero()
    diffs = locations[i1] - locations[i2]

    ll = spstats.multivariate_normal.logpdf(diffs, mean=[0, 0], cov=cov)

    return np.mean(ll)


@cache_arg(0, hash_bool_array)
def compute_geo_prior_distance(zone: np.array, net: dict, ecdf_geo: dict, subgraph_type: str):

    """ This function computes the geo prior for the sum of all distances of the subgraph of a zone
    Args:
        zone (np.array): The current zone (boolean array)
        net (dict):  The full network containing all sites.
        ecdf_geo (dict): The empirically generated geo-prior functions
        subgraph_type (str): Defines which subgraph is used to compute the geo-prior, either
                    "complete", "delaunay" or "mst" (short for "minimum spanning tree")
    Returns:
        geo_prior(float): The geo-prior of the zone
    """

    if subgraph_type == "complete":

        dist_mat = net['dist_mat']

        dist_mat_zone = sparse.triu(dist_mat[zone][:, zone])
        n_dist = dist_mat_zone.nnz

        # Mean
        d = dist_mat_zone.sum()/n_dist
        # Mean squared
        # d = np.sum(dist_mat_zone.toarray() **2.) / n_dist
        # Max
        # d = dist_mat[zone][:, zone].max(axis=None)

    else:
        triang = triangulation(net, zone)

        if subgraph_type == "delaunay":
            # Mean
            dist_mat_zone = sparse.triu(triang)
            n_dist = dist_mat_zone.nnz

            # Mean
            d = dist_mat_zone.sum()/n_dist
            # Mean squared
            # d = np.sum(dist_mat_zone.toarray() ** 2.) / n_dist
            # Max
            # d = triang.max(axis=None)

        elif subgraph_type == "mst":

            dist_mat_zone = minimum_spanning_tree(triang)
            n_dist = dist_mat_zone.nnz

            # Mean
            d = dist_mat_zone.sum()/n_dist
            # Mean squared
            # d = np.sum(dist_mat_zone.toarray() ** 2.) / n_dist
            # Max
            # d = dist_mat_zone.max(axis=None)

        else:
            raise ValueError('Unknown lh_type: %s' % subgraph_type)

    # Compute likelihood
    x = ecdf_geo[subgraph_type]['fitted_gamma']
    geo_prior = np.log(1-spstats.gamma.cdf(d, *x))
    return geo_prior

