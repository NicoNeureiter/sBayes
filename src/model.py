#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
from scipy import sparse
import scipy.stats as spstats
from scipy.sparse.csgraph import minimum_spanning_tree

from src.util import triangulation, cache_arg, hash_bool_array, compute_delaunay, cache_decorator

EPS = np.finfo(float).eps


@cache_decorator
def compute_global_likelihood(features, p_global):
    """Computes the global likelihood, that is the likelihood per site and features
    without knowledge about family or zones.

    Args:
        features(np.array or 'SparseMatrix'): The feature values for all sites and features.
                shape: (n_sites, n_features, n_categories)
        p_global (np.array): The global frequencies of every feature and every value.
                shape: (n_features, n_categories)

    Returns:
        (ndarray): the global probabilities per site and feature
        shape: (n_features, n_sites)
    """
    # Todo: NAs
    n_sites, n_features, n_categories = features.shape
    lh_global = np.full((n_features, n_sites), np.nan)

    for i_f in range(n_features):

        f = features[:, i_f, :]
        # f.shape = (zone_size, n_categories)

        # Compute global likelihood per site and feature
        lh_global[i_f] = f.dot(p_global[i_f, :])

    return lh_global


def compute_zone_likelihood(features, zones):
    """Computes the zone likelihood that is the likelihood per site and feature given zones z1, ... zn
    Args:
        features(np.array or 'SparseMatrix'): The feature values for all sites and features.
            shape: (n_sites, n_features, n_categories)
        zones(np.array): Binary arrays indicating the assignment of a site to the current zones.
                shape: (n_zones, n_sites)

    Returns:
        (np.array): the zone likelihood per site and feature
            shape: (n_features, n_sites)
    """
    n_sites, n_features, n_categories = features.shape
    lh_zone = np.full((n_features, n_sites), np.nan)

    for z in zones:

        idx = z.nonzero()[0]
        zone_size = len(idx)

        # Compute the probability to find a feature in the zone
        features_zone = features[z, :, :]
        p_zone = np.sum(features_zone, axis=0) / zone_size
        # p_zone.shape = (n_features, n_categories)

        # Division by zero could cause troubles
        p_zone = p_zone.clip(EPS, 1 - EPS)

        for i_f in range(n_features):

            f = features_zone[:, i_f, :]
            # f.shape = (zone_size, n_categories)

            # Compute the feature likelihood vector (for all sites in zone)
            lh_zone[i_f, z] = f.dot(p_zone[i_f, :])

    return lh_zone


@cache_decorator
def compute_family_likelihood(features, families=1):
    """Computes the family likelihood, that is the likelihood per site and feature given family f1, ... fn

    Args:
        features(np.array or 'SparseMatrix'): The feature values for all sites and features.
            shape: (n_sites, n_features, n_categories)
        families(np.array): Binary arrays indicating the assignment of a site to a family.
                shape: (n_families, n_sites)

    Returns:
        (np.array): the zone likelihood per site and feature
            shape: (n_features, n_sites)
    """
    n_sites, n_features, n_categories = features.shape
    lh_families = np.full((n_features, n_sites), np.nan)

    return lh_families


def assign_weights_per_site(lh, weight):
    """This function assigns each site a weight if it has a likelihood and zero otherwise

        Args:
            lh(np.array): likelihood value per site
                shape(n_sites)
            weight (double): the weight to normalize

        Return:
            np.array: the weight_per site
            shape(n_sites)
    """

    lh = -np.isnan(lh)
    weights_per_site = weight * lh
    return weights_per_site


# todo: change p_global
def compute_likelihood_generative(zones, features,  weights, p_global=1,  *args):
        """Compute the likelihood of all sites. The likelihood is defined as a mixture of the global distribution and
        the likelihood distribution of the family and the zone.

        Args:
            zones(np.array): Binary arrays indicating the assignment of a site to the current zones.
                shape: (n_zones, n_sites)
            features (np.array or 'SparseMatrix'): The feature values for all sites and features.
                shape: (n_sites, n_features, n_categories)
            weights (np.array): The mixture coefficient controlling how much each feature is explained by each lh
                shape: (n_features, 3)
            p_global (np.array): The global frequencies of every feature and every value.
                shape: (n_features, n_categories)

        TODO:
            p_family (np.array): The probabilities of every feature in every language family.
                shape: (n_sites, n_features, n_categories)
            families (np.array): Binary array indicating the assignment of a site to a language family.
                shape: (n_sites, n_families)
    
        Returns:
            float: The joint likelihood of all sites in the zone.
        """

        # New likelihood
        n_sites, n_features, n_categories = features.shape

        # Weights must sum to one
        assert np.allclose(a=np.sum(weights, axis=-1), b=1., rtol=EPS)

        # Compute likelihood per site # Todo finish family lh
        global_lh = compute_global_likelihood(features, p_global)
        zone_lh = compute_zone_likelihood(features, zones)
        family_lh = compute_family_likelihood(features)

        # Define weights for each site depending on whether the likelihood is available
        # (none, global, global and family, global and zone, global, family and zone)
        weights = np.repeat(weights, n_sites, axis=0)
        ll = 0.

        # Todo: Make this part faster, much faster
        for f in range(n_features):

            weights_per_site = np.array((assign_weights_per_site(global_lh[f], weights[f, 0]),\
                                         assign_weights_per_site(zone_lh[f], weights[f, 1]),\
                                         assign_weights_per_site(family_lh[f], weights[f, 2])))

            norm_weights = weights_per_site/weights_per_site.sum(axis=0, keepdims=True)

            lh = np.nanprod(np.array([norm_weights[0], global_lh[f]]), axis=0) \
                + np.nanprod(np.array([norm_weights[1], zone_lh[f]]), axis=0) \
                + np.nanprod(np.array([norm_weights[2], family_lh[f]]), axis=0)

            ll += np.sum(np.log(lh))

        return ll

def compute_prior_zones(zones, zone_prior_type):
    """Evaluates the prior of a given set of zones.

    Args:
        zones(np.array): Boolean arrays of the current zones.
            shape(n_zones, n_sites)
        zone_prior_type(string): The type of prior (either uniform, geo_empirical, geo_gaussian)

    Returns:
        float: The prior probability of the zone
    """
    if zone_prior_type == 'uniform':
        return 0.


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

