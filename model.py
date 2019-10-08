#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
from scipy import sparse
import scipy.stats as spstats
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import poisson

from src.util import compute_delaunay, transform_weights_from_log, \
    transform_p_from_log, n_smallest_distances

EPS = np.finfo(float).eps


# @cache_global_lh
def compute_global_likelihood(features, global_freq):
    """Computes the global likelihood, that is the likelihood per site and features
    without knowledge about family or zones.

    Args:
        features(np.array or 'SparseMatrix'): The feature values for all sites and features.
                shape: (n_sites, n_features, n_categories)
        global_freq (np.array): The global frequencies of every feature and every category.
                shape: (n_features, n_categories)

    Returns:
        (ndarray): the global probabilities per site and feature
            shape: (n_sites, n_features)
    """

    n_sites, n_features, n_categories = features.shape
    lh_global = np.ones((n_sites, n_features))

    for i_f in range(n_features):

        f = features[:, i_f, :]
        # f.shape = (zone_size, n_categories)

        # Compute global likelihood per site and feature
        lh_global[:, i_f] = f.dot(global_freq[i_f, :])

    return lh_global


# @cache_decorator
def compute_zone_likelihood(features, zones, sample_p_zones, p_zones=None):
    """Computes the zone likelihood that is the likelihood per site and feature given zones z1, ... zn
    Args:
        features(np.array or 'SparseMatrix'): The feature values for all sites and features.
            shape: (n_sites, n_features, n_categories)
        zones(np.array): Binary arrays indicating the assignment of a site to the current zones.
            shape: (n_zones, n_sites)
        sample_p_zones (bool): Sample p_zones (True) or use maximum likelihood estimate (False)
        p_zones(np.array): The estimated probabilities of a category in a zone
            shape: (n_zones, n_features, n_sites)


    Returns:
        (np.array): the zone likelihood per site and feature
            shape: (n_sites, n_features)
    """
    n_sites, n_features, n_categories = features.shape
    lh_zone = np.ones((n_sites, n_features))

    for z in range(len(zones)):

        # Estimate the probability to find a feature/category in the zone, given the counts per category
        features_zone = features[zones[z], :, :]

        if sample_p_zones:
            p_zone = p_zones[z]
        else:
            # Maximum likelihood estimate
            idx = zones[z].nonzero()[0]
            zone_size = len(idx)
            p_zone = np.sum(features_zone, axis=0) / zone_size
        # p_zone.shape = (n_features, n_categories)

        # Division by zero could cause troubles
        p_zone = p_zone.clip(EPS, 1 - EPS)

        for i_f in range(n_features):

            f = features_zone[:, i_f, :]
            # f.shape = (zone_size, n_categories)

            # Compute the feature likelihood vector (for all sites in zone)
            lh_zone[zones[z], i_f] = f.dot(p_zone[i_f, :])

    return lh_zone


# @cache_decorator
def compute_family_likelihood(features, families, sample_p_families, p_families=None):
    """Computes the family likelihood, that is the likelihood per site and feature given family f1, ... fn

    Args:
        features(np.array or 'SparseMatrix'): The feature values for all sites and features.
            shape: (n_sites, n_features, n_categories)
        families(np.array): Binary arrays indicating the assignment of a site to a family.
                shape: (n_families, n_sites)
        sample_p_families (bool): Sample p_families (True) or use maximum likelihood estimate (False)
        p_families(np.array): The estimated probabilities of a category in a family
            shape: (n_zones, n_features, n_sites)

    Returns:
        (np.array): the zone likelihood per site and feature
            shape: (n_sites, n_features)
    """

    n_sites, n_features, n_categories = features.shape
    lh_families = np.ones((n_sites, n_features))

    for fam in range(len(families)):

        # Compute the probability to find a feature in the zone
        features_family = features[families[fam], :, :]

        if sample_p_families:
            p_family = p_families[fam]

        else:
            # maximum likelihood estimate is used
            idx = families[fam].nonzero()[0]
            family_size = len(idx)
            p_family = np.sum(features_family, axis=0) / family_size
        # p_family.shape = (n_features, n_categories)

        # Division by zero could cause troubles
        p_family = p_family.clip(EPS, 1 - EPS)

        for i_f in range(n_features):

            f = features_family[:, i_f, :]

            # Compute the feature likelihood vector (for all sites in family)
            lh_families[families[fam], i_f] = f.dot(p_family[i_f, :])

    return lh_families


def normalize_weights(weights, assignment):
    """This function assigns each site a weight if it has a likelihood and zero otherwise

        Args:
            weights (np.array): the weights to normalize
                shape: (n_sites, n_features, 3)
            assignment (np.array): assignment of sites to global, zone and family.
                shape(n_sites, 3)

        Return:
            np.array: the weight_per site
                shape(n_sites, n_features, 3)
    """

    weights_per_site = weights * assignment[:, np.newaxis, :]
    return weights_per_site / weights_per_site.sum(axis=2, keepdims=True)


class GenerativeLikelihood(object):

    def __init__(self):

        # The assignment and non-normalized likelihood
        self.assignment = None
        self.all_lh = None

        # Assignment and lh per zone, family and global
        self.global_assignment = None
        self.family_assignment = None
        self.zone_assignment = None

        self.global_lh = None
        self.family_lh = None
        self.zone_lh = None

        # Weights
        self.weights = None

    def __call__(self, sample, features,  global_freq, inheritance, families=None, sample_p_zones=True,
                 sample_p_families=True):
            """Compute the likelihood of all sites. The likelihood is defined as a mixture of the global distribution and
            the likelihood distribution of the family and the zone.

            Args:
                sample(Sample): A Sample object consisting of zones and weights
                features (np.array or 'SparseMatrix'): The feature values for all sites and features.
                    shape: (n_sites, n_features, n_categories)
                global_freq (np.array): The observed global frequencies of every feature and every category
                    shape: (n_features, n_categories)
                inheritance (bool): Does the likelihood consider inheritance?

            Kwargs:
                families (np.array): Binary array indicating the assignment of a site to a language family.
                    shape: (n_families, n_sites)
                sample_p_zones (bool): Sample p_zones (True) or use maximum likelihood estimate (False)
                sample_p_families (bool): Sample p_families (True) or use maximum likelihood estimate (False)

            Returns:
                float: The joint likelihood of the current sample.
            """
            n_sites, n_features, n_categories = features.shape
            zones = sample.zones
            #print(np.count_nonzero(sample.zones), "zone size")

            # Find NA features in the data
            na_features = (np.sum(features, axis=-1) == 0)

            # Weights and p_zones must sum to one
            # assert np.allclose(a=np.sum(weights, axis=-1), b=1., rtol=EPS)

            ##################
            # Global
            ##################
            # Global assignment and lh are constants and only evaluated when initialized
            if self.global_assignment is None and self.global_lh is None:
                global_assignment = np.ones(n_sites)
                global_lh = compute_global_likelihood(features=features, global_freq=global_freq)
                self.global_assignment = global_assignment
                self.global_lh = global_lh
            else:
                global_assignment = self.global_assignment
                global_lh = self.global_lh

            ##################
            # Family
            ##################
            # Families are only evaluated if the model considers inheritance
            if inheritance:

                # Family assignment is a constant, and only evaluated when initialized
                if self.family_assignment is None:
                    family_assignment = np.any(families, axis=0)
                    self.family_assignment = family_assignment
                else:
                    family_assignment = self.family_assignment

                # Family lh is evaluated when initialized and when p_families is changed
                if self.family_lh is None or sample.what_changed['lh']['p_families']:
                    # p_families can be sampled or estimated
                    if sample_p_families:
                        p_families = transform_p_from_log(sample.p_families)

                        # assert np.allclose(a=np.sum(p_families, axis=-1), b=1., rtol=EPS)

                        family_lh = compute_family_likelihood(features=features, families=families,
                                                              sample_p_families=sample_p_families,
                                                              p_families=p_families)

                    else:
                        family_lh = compute_family_likelihood(features=features, families=families,
                                                              sample_p_families=sample_p_families)
                    self.family_lh = family_lh

                else:
                    family_lh = self.family_lh

            ##################
            # Zone
            ##################
            # Zone assignment is evaluated when initialized or when zones change
            if self.zone_assignment is None or sample.what_changed['lh']['zones']:

                # Compute the assignment of sites to zones
                zone_assignment = np.any(zones, axis=0)
                self.zone_assignment = zone_assignment

            else:
                zone_assignment = self.zone_assignment

            # Zone lh is evaluated when initialized, or when zones or p_zones change
            if self.zone_lh is None or sample.what_changed['lh']['zones'] or sample.what_changed['lh']['p_zones']:

                # p_zones can be sampled or estimated
                if sample_p_zones:
                    p_zones = transform_p_from_log(sample.p_zones)
                    # assert np.allclose(a=np.sum(p_zones, axis=-1), b=1., rtol=EPS)
                    zone_lh = compute_zone_likelihood(features=features, zones=zones, sample_p_zones=sample_p_zones,
                                                      p_zones=p_zones)
                else:
                    zone_lh = compute_zone_likelihood(features=features, zones=zones, sample_p_zones=sample_p_zones)

                self.zone_lh = zone_lh
            else:
                zone_lh = self.zone_lh

            ##################
            # Combination
            ##################
            # Assignments are recombined when initialized or when zones change
            if self.assignment is None or sample.what_changed['lh']['zones']:

                # Structure of assignment depends on whether inheritance is considered or not
                if inheritance:
                    assignment = np.array([global_assignment, zone_assignment, family_assignment]).T
                else:
                    assignment = np.array([global_assignment, zone_assignment]).T

                self.assignment = assignment
            else:
                assignment = self.assignment

            # Lh is recombined when initialized, when zones change or when p_zones or p_families change
            if self.all_lh is None or sample.what_changed['lh']['zones'] or \
                    sample.what_changed['lh']['p_zones'] or sample.what_changed['lh']['p_families']:

                # Structure of assignment depends on whether inheritance is considered or not
                if inheritance:
                    all_lh = np.array([global_lh, zone_lh, family_lh]).transpose((1, 2, 0))
                else:
                    all_lh = np.array([global_lh, zone_lh]).transpose((1, 2, 0))

                self.all_lh = all_lh
            else:
                all_lh = self.all_lh

            ##################
            # Weights
            ##################
            # weights are evaluated when initialized, when weights change or when assignment changes
            if self.weights is None or sample.what_changed['lh']['weights'] or sample.what_changed['lh']['zones']:

                abnormal_weights = transform_weights_from_log(sample.weights)

                # Extract weights for each site depending on whether the likelihood is available
                # Order of columns in weights: global, contact, inheritance (if available)
                abnormal_weights_per_site = np.repeat(abnormal_weights[np.newaxis, :, :], n_sites, axis=0)
                weights = normalize_weights(abnormal_weights_per_site, assignment)
                self.weights = weights

            else:
                weights = self.weights

            # This is always evaluated

            weighted_lh = np.sum(weights * all_lh, axis=2)
            # Replace na values by 1
            weighted_lh[na_features] = 1.
            log_lh = np.sum(np.log(weighted_lh))

            # todo: try
            # weighted_lh = all_lh.dot(weights)

            # The step is completed. Everything is up-to-date.

            sample.what_changed['lh']['zones'] = False
            sample.what_changed['lh']['p_zones'] = False
            sample.what_changed['lh']['weights'] = False

            if inheritance:
                sample.what_changed['lh']['p_families'] = False

            return log_lh

PRIOR_SIZE_LOGPMF = poisson(15).logpmf

class GenerativePrior(object):

    def __init__(self):
        self.prior_zones = None
        self.prior_weights = None
        self.prior_p_zones = None
        self.prior_p_families = None

    #todo: implement set_prior_size!
    def __call__(self, sample, geo_prior, geo_prior_parameters, prior_weights,
                 prior_p_zones, prior_p_families, prior_p_families_parameters, network,
                 set_prior_size=False):
            """Compute the prior of the current sample.
            Args:
                sample(Sample): A Sample object consisting of zones and weights
                geo_prior(str): The geo-prior used in the analysis ("uniform", "gaussian" or "distance")
                geo_prior_parameters(dict): Additional parameters needed to evaluate the geo-prior
                prior_weights(str): The prior for weights used in the analysis ("uniform")
                prior_p_zones(str): The prior for p_zones ("uniform")
                prior_p_families(str): The prior for p_families ("uniform", or "dirichlet")
                prior_p_families_parameters: (dict): Additional parameters needed to evaluate prior_p_families
                network (dict): network containing the graph, locations,...

            Returns:
                float: The (log)prior of the current sample
            """

            # zones = sample.zones
            # weights = transform_weights_from_log(sample.weights)

            # Weights must sum to one
            # assert np.allclose(a=np.sum(weights, axis=-1), b=1., rtol=EPS)

            # Geo Prior
            if self.prior_zones is None or sample.what_changed['prior']['zones']:

                if geo_prior == "uniform":
                    prior_zones = 0.

                elif geo_prior == "gaussian":
                    prior_zones = geo_prior_gaussian(sample.zones, network, geo_prior_parameters['gaussian'])

                elif geo_prior == "distance":
                    prior_zones = geo_prior_distance(sample.zones, network, geo_prior_parameters['distance'])

                else:
                    raise ValueError('geo_prior must be either "uniform", "gaussian" or "distance"')

                self.prior_zones = prior_zones
            else:
                prior_zones = self.prior_zones

            # Weights
            if self.prior_weights is None or sample.what_changed['prior']['weights']:
                if prior_weights == "uniform":
                    prior_weights = 0.

                else:
                    raise ValueError('prior_weights must be "uniform"')

                self.prior_weights = prior_weights

            else:
                prior_weights = self.prior_weights

            if self.prior_p_zones is None or sample.what_changed['prior']['p_zones']:

                if prior_p_zones == "uniform":
                    prior_p_zones = 0.
                else:
                    raise ValueError('prior_p_zones must be "uniform"')

                self.prior_p_zones = prior_p_zones
            else:
                prior_p_zones = self.prior_p_zones

            if self.prior_p_families is None or sample.what_changed['prior']['p_families']:
                if prior_p_families == "uniform":
                    prior_p_families = 0.

                elif prior_p_families == "dirichlet":

                    prior_p_families = prior_p_families_dirichlet(
                        p_families=transform_p_from_log(sample.p_families),
                        dirichlet=prior_p_families_parameters['dirichlet'],
                        categories=prior_p_families_parameters['categories'])

                else:
                    raise ValueError('prior_p_zones must be "uniform" or "dirichlet')

                self.prior_p_families = prior_p_families

            else:
                prior_p_families = self.prior_p_families

            log_prior = prior_zones + prior_weights + prior_p_zones + prior_p_families

            # The step is completed. Everything is up-to-date.
            sample.what_changed['prior']['zones'] = False
            sample.what_changed['prior']['p_zones'] = False
            sample.what_changed['prior']['weights'] = False
            sample.what_changed['prior']['p_families'] = False

            # TODO implement properly
            if set_prior_size:
                zone_sizes = np.count_nonzero(sample.zones, axis=1)
                log_prior += np.sum(PRIOR_SIZE_LOGPMF(zone_sizes))

            return log_prior

# deprecated
def compute_prior_zones(zones, geo_prior):
    """Evaluates the prior of a given array of zones.
    Args:
        zones(np.array): Boolean arrays of the current zones.
            shape(n_zones, n_sites)
        prior_type(string): The type of prior (either uniform, geo_empirical or geo_gaussian)

    Returns:
        float: The prior probability of the zones
    """
    if geo_prior == "uniform":
        return 0.

    elif geo_prior == "gaussian":
        raise ValueError("not supported yet")

    elif geo_prior == "distance":
        raise ValueError("not supported yet")

    else:
        raise ValueError('geo_prior must be either "uniform", "gaussian" or "distance"')


# deprecated
def prior_weights(weights, prior_weights):
    """Evaluates the prior of a given array of weights

        Args:
            weights(np.array): Boolean arrays of the current weights
                shape(n_features, 3)
            prior_weights(string): The type of prior (currently only 'uniform' is supported)

        Returns:
            float: The prior probability of the weights
    """
    if prior_weights == "uniform":
        return 0.
    else:
        raise ValueError('Only prior_type "uniform" is currently supported')


def geo_prior_gaussian(zones: np.array, network: dict, cov: np.array):
    """
    This function computes the two-dimensional Gaussian geo-prior for all edges in the zone
    Args:
        zones (np.array): boolean array representing the current zone
        network (dict): network containing the graph, location,...
        cov (np.array): Covariance matrix of the multivariate gaussian (estimated from the data)

    Returns:
        float: the geo-prior of the zones
    """
    log_prior = np.ndarray([])
    for z in zones:
        dist_mat = network['dist_mat'][z][:, z]
        locations = network['locations'][z]

        if len(locations) > 3:

            delaunay = compute_delaunay(locations)
            mst = minimum_spanning_tree(delaunay.multiply(dist_mat))
            i1, i2 = mst.nonzero()

        elif len(locations) == 3:
            i1, i2 = n_smallest_distances(dist_mat, n=2, return_idx=True)

        elif len(locations) == 2:
            i1, i2 = n_smallest_distances(dist_mat, n=1, return_idx=True)

        diffs = locations[i1] - locations[i2]
        prior_z = spstats.multivariate_normal.logpdf(diffs, mean=[0, 0], cov=cov)
        log_prior = np.append(log_prior, prior_z)

    return np.mean(log_prior)


def geo_prior_distance(zones: np.array, network: dict, scale: float):

    """ This function computes the geo prior for the sum of all distances of the subgraph of a zone
    Args:
        zones (np.array): The current zones (boolean array)
        network (dict):  The full network containing all sites.
        scale (float): The scale (estimated from the data)

    Returns:
        float: the geo-prior of the zones
    """
    log_prior = np.ndarray([])
    for z in zones:

        dist_mat = network['dist_mat'][z][:, z]
        locations = network['locations'][z]

        if len(locations) > 3:

            delaunay = compute_delaunay(locations)
            mst = minimum_spanning_tree(delaunay.multiply(dist_mat))
            distances = mst.tocsr()[mst.nonzero()]

        elif len(locations) == 3:
            distances = n_smallest_distances(dist_mat, n=2, return_idx=False)

        elif len(locations) == 2:
            distances = n_smallest_distances(dist_mat, n=1, return_idx=False)

        log_prior = spstats.expon.logpdf(distances, loc=0, scale=scale)

    return np.mean(log_prior)


def prior_p_families_dirichlet(p_families, dirichlet, categories):
    """" This function evaluates the prior for p_families
    Args:
        p_families(np.array): p_families from the sample
        dirichlet(list): list of dirichlet distributions
        categories(list): list of available categories per feature

    Returns:
        float: the prior for p_families
    """
    n_fam, n_feat, n_cat = p_families.shape
    log_prior = []

    for fam in range(n_fam):
        for f in range(n_feat):
            idx = categories[f]
            diri = dirichlet[fam][f]
            p_fam = p_families[fam, f, idx]
            log_prior.append(diri.logpdf(p_fam))

    return sum(log_prior)

