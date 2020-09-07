#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import numpy as np
import scipy.stats as stats
from scipy.sparse.csgraph import minimum_spanning_tree

from sbayes.util import (compute_delaunay, n_smallest_distances, log_binom,
                         counts_to_dirichlet, inheritance_counts_to_dirichlet,
                         dirichlet_logpdf)
EPS = np.finfo(float).eps


def compute_global_likelihood(features, sample_p_global, p_global=None,
                              outdated_features=None, cached_lh=None):
    """Computes the global likelihood, that is the likelihood per site and features
    without knowledge about family or zones.

    Args:
        features (np.array or 'SparseMatrix'): The feature values for all sites and features.
                shape: (n_sites, n_features, n_categories)
        sample_p_global(bool): Sample p_global (True) or use maximum likelihood estimate (False)?
    Kwargs:
        p_global (np.array): The estimated global probabilities of all features in all site
            shape: (1, n_features, n_sites)
        outdated_features (IndexSet): Features which changed, i.e. where lh needs to be recomputed.

    Returns:
        (np.array): the global likelihood per site and feature
            shape: (n_sites, n_features)
    """
    n_sites, n_features, n_categories = features.shape

    # Estimate the global probability to find a feature/category
    if sample_p_global:
        p_glob = p_global[0]
    else:
        # Maximum likelihood estimate
        p_glob = np.sum(features, axis=0) / n_sites

    # p_glob.shape = (n_features, n_categories)

    # Division by zero could cause troubles
    p_glob = p_glob.clip(EPS, 1 - EPS)

    if cached_lh is None:
        lh_global = np.ones((n_sites, n_features))
        assert outdated_features.all
    else:
        lh_global = cached_lh

    if outdated_features.all:
        outdated_features = range(n_features)

    for i_f in outdated_features:
        f = features[:, i_f, :]
        # f.shape = (n_sites, n_categories)

        # Compute the feature likelihood vector (for all sites in zone)
        lh_global[:, i_f] = f.dot(p_glob[i_f, :])

    return lh_global


def compute_zone_likelihood(features, zones, sample_p_zones, p_zones=None,
                            outdated_indices=None, outdated_zones=None, cached_lh=None):
    """Computes the zone likelihood that is the likelihood per site and feature given zones z1, ... zn
    Args:
        features(np.array or 'SparseMatrix'): The feature values for all sites and features.
            shape: (n_sites, n_features, n_categories)
        zones(np.array): Binary arrays indicating the assignment of a site to the current zones.
            shape: (n_zones, n_sites)
        sample_p_zones (bool): Sample p_zones (True) or use maximum likelihood estimate (False)
    Kwargs:
        p_zones(np.array): The estimated probabilities of features in all sites according to the zone
            shape: (n_zones, n_features, n_sites)
        outdated_indices (IndexSet): Set of outdated (zone, feature) index-pairs.
        outdated_zones (IndexSet): Set of indices, where the zone changed (=> update across features).
        cached_lh (np.array): The cached set of likelihood values (to be updated, where outdated).


    Returns:
        (np.array): the zone likelihood per site and feature
            shape: (n_sites, n_features)
    """

    n_sites, n_features, n_categories = features.shape
    n_zones = len(zones)

    if cached_lh is None:
        lh_zone = np.zeros((n_sites, n_features))
        assert outdated_indices.all
    else:
        lh_zone = cached_lh

    if outdated_indices.all:
        outdated_indices = itertools.product(range(n_zones), range(n_features))
    else:
        if outdated_zones:
            outdated_zones_expanded = {(zone, feat) for zone in outdated_zones for feat in range(n_features)}
            outdated_indices = set.union(outdated_indices, outdated_zones_expanded)

    # features_by_zone = {features[zones[z], :, :] for z in range(n_zones)}
    features_by_zone = {}

    for z, i_f in outdated_indices:
        # # Estimate the probability to find a feature/category in the zone, given the counts per category
        # features_zone = features[zones[z], :, :]
        #
        # if sample_p_zones:
        #     p_zone = p_zones[z]
        # else:
        #     # Maximum likelihood estimate
        #     idx = zones[z].nonzero()[0]
        #     zone_size = len(idx)
        #     p_zone = np.sum(features_zone, axis=0) / zone_size
        # # p_zone.shape = (n_features, n_categories)
        #
        # # Division by zero could cause troubles
        # p_zone = p_zone.clip(EPS, 1 - EPS)
        #
        # f = features_zone[:, i_f, :]
        # # f.shape = (zone_size, n_categories)

        # Compute the feature likelihood vector (for all sites in zone)
        # f = features[zones[z], i_f, :]
        # f = features_by_zone[z][:, i_f, :]
        if z not in features_by_zone:
            features_by_zone[z] = features[zones[z], :, :]
        f = features_by_zone[z][:, i_f, :]
        p = p_zones[z, i_f, :]
        lh_zone[zones[z], i_f] = f.dot(p)

    return lh_zone


def compute_family_likelihood(features, families, p_families=None,
                              outdated_indices=None, cached_lh=None):
    """Computes the family likelihood, that is the likelihood per site and feature given family f1, ... fn

    Args:
        features(np.array or 'SparseMatrix'): The feature values for all sites and features.
            shape: (n_sites, n_features, n_categories)
        families(np.array): Binary arrays indicating the assignment of a site to a family.
                shape: (n_families, n_sites)
        sample_p_families (bool): Sample p_families (True) or use maximum likelihood estimate (False)?
    Kwargs:
        p_families(np.array): The estimated probabilities of features in all sites according to the family
            shape: (n_families, n_features, n_sites)
        outdated_indices (IndexSet): Set of outdated (family, feature) index-pairs.
        cached_lh (np.array): The cached set of likelihood values (to be updated, where outdated).

    Returns:
        (np.array): the family likelihood per site and feature
            shape: (n_sites, n_features)
    """

    n_sites, n_features, n_categories = features.shape
    n_families = len(families)

    if cached_lh is None:
        lh_families = np.zeros((n_sites, n_features))
        assert outdated_indices.all
    else:
        lh_families = cached_lh

    if outdated_indices.all:
        outdated_indices = itertools.product(range(n_families), range(n_features))

    for fam, i_f in outdated_indices:
        # Compute the feature likelihood vector (for all sites in family)
        f = features[families[fam], i_f, :]
        p = p_families[fam, i_f, :]
        lh_families[families[fam], i_f] = f.dot(p)

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

    def __init__(self, data, inheritance, families=None,
                 sample_p_global=True, sample_p_zones=True,  sample_p_families=True):
        self.data = data
        self.families = np.asarray(families, dtype=bool)
        self.n_sites, self.n_features, self.n_categories = data.shape

        # The assignment (global, zone, family) combined and weighted and the non-normalized likelihood
        self.assignment = None
        self.all_lh = None

        # Assignment and lh (global, per zone and family)
        self.global_assignment = np.ones(self.n_sites)
        self.family_assignment = None
        self.zone_assignment = None

        self.global_lh = None
        self.family_lh = None
        self.zone_lh = None

        # Weights
        self.weights = None

        # Set config flags
        self.inheritance = inheritance
        self.sample_p_global = sample_p_global
        self.sample_p_zones = sample_p_zones
        self.sample_p_families = sample_p_families

    def reset_cache(self):
        # The assignment (global, zone, family) combined and weighted and the non-normalized likelihood
        self.assignment = None
        self.all_lh = None
        # Assignment and lh (global, per zone and family)
        self.family_assignment = None
        self.zone_assignment = None
        self.global_lh = None
        self.family_lh = None
        self.zone_lh = None
        # Weights
        self.weights = None

    def __call__(self, sample, caching=True):
            """Compute the likelihood of all sites. The likelihood is defined as a mixture of the global distribution
            and the likelihood distribution of the family and the zone.

            Args:
                sample(Sample): A Sample object consisting of zones and weights
                features (np.array or 'SparseMatrix'): The feature values for all sites and features.
                    shape: (n_sites, n_features, n_categories)
                inheritance (bool): Does the model consider inheritance?

            Kwargs:
                families (np.array): Binary array indicating the assignment of sites to language families.
                    shape: (n_families, n_sites)
                sample_p_global (bool): Sample p_global (True) or use maximum likelihood estimate (False)
                sample_p_zones (bool): Sample p_zones (True) or use maximum likelihood estimate (False)
                sample_p_families (bool): Sample p_families (True) or use maximum likelihood estimate (False)

            Returns:
                float: The joint likelihood of the current sample.
            """
            if not caching:
                self.reset_cache()

            features = self.data
            n_sites, n_features, n_categories = features.shape

            # Find NA features in the data
            na_features = (np.sum(features, axis=-1) == 0)

            ##############################
            # Component distributions
            ##############################
            global_assignment, global_lh = self.get_global_lh(sample)
            family_assignment, family_lh = self.get_family_lh(sample)
            zone_assignment, zone_lh = self.get_zone_lh(sample)

            ##############################
            # Combination
            ##############################
            # Assignments are recombined when initialized or when zones change
            if self.assignment is None or sample.what_changed['lh']['zones']:

                # Structure of assignment depends on whether inheritance is considered or not
                if self.inheritance:
                    assignment = np.array([global_assignment, zone_assignment, family_assignment]).T
                else:
                    assignment = np.array([global_assignment, zone_assignment]).T

                self.assignment = assignment
            else:
                assignment = self.assignment

            # Lh is recombined when initialized, when zones change or when p_global, p_zones or p_families change
            if self.all_lh is None or sample.what_changed['lh']['zones'] or sample.what_changed['lh']['p_global'] or \
                    sample.what_changed['lh']['p_zones'] or sample.what_changed['lh']['p_families']:

                # Structure of assignment depends on whether inheritance is considered or not
                if self.inheritance:
                    all_lh = np.array([global_lh, zone_lh, family_lh]).transpose((1, 2, 0))
                else:
                    all_lh = np.array([global_lh, zone_lh]).transpose((1, 2, 0))

                self.all_lh = all_lh
            else:
                all_lh = self.all_lh

            ##############################
            # Weights
            ##############################
            # weights are evaluated when initialized, when weights change or when assignment to zones changes
            if self.weights is None or sample.what_changed['lh']['weights'] or sample.what_changed['lh']['zones']:

                abnormal_weights = sample.weights

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

            # The step is completed. Everything is up-to-date.
            sample.what_changed['lh']['zones'].clear()
            sample.what_changed['lh']['p_global'].clear()
            sample.what_changed['lh']['p_zones'].clear()
            sample.what_changed['lh']['weights'] = False
            if self.inheritance:
                sample.what_changed['lh']['p_families'].clear()

            # Find all languages that are in a family, but not in an area. For each add penalty p to the log_lh
            # todo remove after testing
            in_zone = np.any(sample.zones, axis=0)
            in_family = np.any(self.families, axis=0)
            p = -1.6
            total_penalty = p * np.sum(in_family & ~in_zone)
            log_lh += total_penalty

            return log_lh

    def global_lh_outdated(self, sample):
        return (self.global_lh is None) or (sample.what_changed['lh']['p_global'])

    def get_global_lh(self, sample):
        if self.global_lh_outdated(sample):
            # p_global can be sampled or estimated
            if self.sample_p_global:
                self.global_lh = compute_global_likelihood(features=self.data, sample_p_global=True,
                                                           p_global=sample.p_global,
                                                           outdated_features=sample.what_changed['lh']['p_global'],
                                                           cached_lh=self.global_lh)

            else:
                self.global_lh = compute_global_likelihood(features=self.data, sample_p_global=False)
        return self.global_assignment, self.global_lh

    def get_family_lh(self, sample):

        # Families are only evaluated if the model considers inheritance
        if not self.inheritance:
            return None, None

        # Family assignment is a constant, and only evaluated when initialized
        if self.family_assignment is None:
            family_assignment = np.any(self.families, axis=0)
            self.family_assignment = family_assignment

        # Family lh is evaluated when initialized and when p_families is changed
        if self.family_lh is None or sample.what_changed['lh']['p_families']:
            # assert np.allclose(a=np.sum(sample.p_families, axis=-1), b=1., rtol=EPS)
            self.family_lh = compute_family_likelihood(features=self.data, families=self.families,
                                                       p_families=sample.p_families,
                                                       outdated_indices=sample.what_changed['lh']['p_families'],
                                                       cached_lh=self.family_lh)

        return self.family_assignment, self.family_lh

    def get_zone_lh(self, sample):
        if self.zone_assignment is None or sample.what_changed['lh']['zones']:
            # Compute the assignment of sites to zones
            zone_assignment = np.any(sample.zones, axis=0)
            self.zone_assignment = zone_assignment

        # Zone lh is evaluated when initialized, or when zones or p_zones change
        if self.zone_lh is None or sample.what_changed['lh']['zones'] or sample.what_changed['lh']['p_zones']:

            # p_zones can be sampled or estimated
            if self.sample_p_zones:
                # assert np.allclose(a=np.sum(p_zones, axis=-1), b=1., rtol=EPS)
                zone_lh = compute_zone_likelihood(features=self.data, zones=sample.zones,
                                                  sample_p_zones=True, p_zones=sample.p_zones,
                                                  outdated_indices=sample.what_changed['lh']['p_zones'],
                                                  outdated_zones=sample.what_changed['lh']['zones'],
                                                  cached_lh=self.zone_lh)
            else:
                zone_lh = compute_zone_likelihood(features=self.data, zones=sample.zones,
                                                  sample_p_zones=False,
                                                  outdated_indices=sample.what_changed['lh']['p_zones'],
                                                  outdated_zones=sample.what_changed['lh']['zones'],
                                                  cached_lh=self.zone_lh)

            self.zone_lh = zone_lh

        return self.zone_assignment, self.zone_lh


class GenerativePrior(object):

    def __init__(self):
        self.size_prior = None
        self.geo_prior = None
        self.prior_weights = None
        self.prior_p_global = None
        self.prior_p_zones = None
        self.prior_p_families = None
        self.prior_p_zones_distr = None
        self.prior_p_families_distr = None

    def __call__(self, sample, inheritance, geo_prior_meta, prior_weights_meta, prior_p_global_meta,
                 prior_p_zones_meta, prior_p_families_meta, network):
        """Compute the prior of the current sample.
        Args:
            sample (Sample): A Sample object consisting of zones and weights
            inheritance (bool): Does the model consider inheritance?
            geo_prior_meta (dict): The geo-prior used in the analysis
            prior_weights_meta (dict): The prior for weights used in the analysis
            prior_p_global_meta (dict): The prior for p_global
            prior_p_zones_meta (dict): The prior for p_zones
            prior_p_families_meta (dict): The prior for p_families
            network (dict): network containing the graph, locations,...

        Returns:
            float: The (log)prior of the current sample
        """

        # zone-size prior
        size_prior = self.get_size_prior(sample)
        # geo-prior
        geo_prior = self.get_geo_prior(sample, geo_prior_meta, network)
        # weights
        prior_weights = self.get_prior_weights(sample, prior_weights_meta)
        # p_global
        prior_p_global = self.get_prior_p_global(sample, prior_p_global_meta)
        # p_zones
        prior_p_zones = self.get_prior_p_zones(sample, prior_p_zones_meta)
        # p_families
        if inheritance:
            prior_p_families = self.get_prior_p_families(sample, prior_p_families_meta)
        else:
            prior_p_families = None

        # Add up prior components (in log space)
        log_prior = size_prior + geo_prior + prior_weights + prior_p_global + prior_p_zones
        if inheritance:
            log_prior += prior_p_families

        # The step is completed. Everything is up-to-date.
        sample.what_changed['prior']['zones'].clear()
        sample.what_changed['prior']['weights'] = False
        sample.what_changed['prior']['p_global'].clear()
        sample.what_changed['prior']['p_zones'].clear()

        if inheritance:
            sample.what_changed['prior']['p_families'].clear()

        return log_prior

    def weights_prior_outdated(self, sample):
        """Check whether the cached prior_weights is up-to-date or needs to be recomputed."""
        return self.prior_weights is None or sample.what_changed['prior']['weights']

    def get_prior_weights(self, sample, prior_weights_meta):
        """Compute the prior for weights (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.
            prior_weights_meta (dict): Meta-information about the prior.

        Returns:
            float: Logarithm of the prior probability density.
        """
        if self.weights_prior_outdated(sample):
            if prior_weights_meta['type'] == 'uniform':
                prior_weights = 0.
            else:
                raise ValueError('Currently only uniform prior_weights are supported.')

            self.prior_weights = prior_weights

        return self.prior_weights

    def p_zones_prior_outdated(self, sample, prior_type):
        """Check whether the cached prior_p_zones is up-to-date or needs to be recomputed."""
        if self.prior_p_zones is None:
            return True
        elif sample.what_changed['prior']['p_zones']:
            return True
        elif prior_type == 'universal' and sample.what_changed['prior']['p_global']:
            return True
        else:
            return False

    def get_prior_p_zones(self, sample, prior_p_zones_meta):
        """Compute the prior for p_zones (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.
            prior_p_zones_meta (dict): Meta-information about the prior.

        Returns:
            float: Logarithm of the prior probability density.
        """
        prior_type = prior_p_zones_meta['type']
        what_changed = sample.what_changed['prior']

        if self.p_zones_prior_outdated(sample, prior_type):
            if prior_type == 'uniform':
                prior_p_zones = 0.

            elif prior_type == 'universal':
                s = prior_p_zones_meta['strength']
                c_universal = s * sample.p_global[0]

                self.prior_p_zones_distr = counts_to_dirichlet(counts=c_universal,
                                                               categories=prior_p_zones_meta['states'],
                                                               outdated_features=what_changed['p_global'],
                                                               dirichlet=self.prior_p_zones_distr)
                prior_p_zones = prior_p_families_dirichlet(p_families=sample.p_zones,
                                                                dirichlet=self.prior_p_zones_distr,
                                                                categories=prior_p_zones_meta['states'],
                                                                outdated_indices=what_changed['p_zones'],
                                                                outdated_distributions=what_changed['p_global'],
                                                                cached_prior=self.prior_p_zones,
                                                                broadcast=True)
            else:
                raise ValueError('Currently only uniform p_zones priors are supported.')

            self.prior_p_zones = prior_p_zones

        return np.sum(self.prior_p_zones)

    def p_families_prior_outdated(self, sample, prior_type):
        """Check whether the cached prior_p_families is up-to-date or needs to be recomputed."""
        if self.prior_p_families is None:
            return True
        elif sample.what_changed['prior']['p_families']:
            return True
        elif (prior_type in ['universal', 'counts_and_universal']) and (sample.what_changed['prior']['p_global']):
            return True
        else:
            return False

    def get_prior_p_families(self, sample, prior_p_families_meta):
        """Compute the prior for p_families (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.
            prior_p_families_meta (dict): Meta-information about the prior.

        Returns:
            float: Logarithm of the prior probability density.
        """
        prior_type = prior_p_families_meta['type']
        what_changed = sample.what_changed['prior']

        if self.p_families_prior_outdated(sample, prior_type):

            if prior_type == 'uniform':
                prior_p_families = 0.

            elif prior_type == 'pseudocounts':
                prior_p_families = prior_p_families_dirichlet(p_families=sample.p_families,
                                                              dirichlet=prior_p_families_meta['dirichlet'],
                                                              categories=prior_p_families_meta['states'],
                                                              outdated_indices=what_changed['p_families'],
                                                              outdated_distributions=what_changed['p_global'],
                                                              cached_prior=self.prior_p_families,
                                                              broadcast=False)

            elif prior_type == 'universal':
                s = prior_p_families_meta['strength']
                c_universal = s * sample.p_global[0]
                self.prior_p_families_distr = counts_to_dirichlet(counts=c_universal,
                                                                  categories=prior_p_families_meta['states'],
                                                                  outdated_features=what_changed['p_global'],
                                                                  dirichlet=self.prior_p_families_distr)

                prior_p_families = prior_p_families_dirichlet(p_families=sample.p_families,
                                                              dirichlet=self.prior_p_families_distr,
                                                              categories=prior_p_families_meta['states'],
                                                              outdated_indices=what_changed['p_families'],
                                                              outdated_distributions=what_changed['p_global'],
                                                              cached_prior=self.prior_p_families,
                                                              broadcast=True)

            elif prior_type == 'counts_and_universal':
                s = prior_p_families_meta['strength']
                c_pseudocounts = prior_p_families_meta['counts']
                c_universal = s * sample.p_global
                self.prior_p_families_distr = inheritance_counts_to_dirichlet(counts=c_universal + c_pseudocounts,
                                                                         categories=prior_p_families_meta['states'],
                                                                         outdated_features=what_changed['p_global'],
                                                                         dirichlet=self.prior_p_families_distr)

                prior_p_families = prior_p_families_dirichlet(p_families=sample.p_families,
                                                              dirichlet=self.prior_p_families_distr,
                                                              categories=prior_p_families_meta['states'],
                                                              outdated_indices=what_changed['p_families'],
                                                              outdated_distributions=what_changed['p_global'],
                                                              cached_prior=self.prior_p_families,
                                                              broadcast=False)

            else:
                raise ValueError('prior_p_families must be "uniform" or "pseudocounts"')

            self.prior_p_families = prior_p_families

        return np.sum(self.prior_p_families)

    def prior_p_global_outdated(self, sample):
        """Check whether the cached prior_p_global is up-to-date or needs to be recomputed."""
        return self.prior_p_global is None or sample.what_changed['prior']['p_global']

    def get_prior_p_global(self, sample, prior_p_global_meta):
        """Compute the prior for p_global (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.
            prior_p_global_meta (dict): Meta-information about the prior.

        Returns:
            float: Logarithm of the prior probability density.
        """
        if self.prior_p_global_outdated(sample):

            if prior_p_global_meta['type'] == 'uniform':
                prior_p_global = 0.

            elif prior_p_global_meta['type'] == 'pseudocounts':
                prior_p_global = prior_p_global_dirichlet(p_global=sample.p_global,
                                                          dirichlet=prior_p_global_meta['dirichlet'],
                                                          categories=prior_p_global_meta['states'],
                                                          outdated_features=sample.what_changed['prior']['p_global'],
                                                          cached_prior=self.prior_p_global)

            else:
                raise ValueError('Prior for universal pressures must be "uniform" or "pseudocounts')

            self.prior_p_global = prior_p_global

        return np.sum(self.prior_p_global)

    def geo_prior_outdated(self, sample):
        """Check whether the cached geo_prior is up-to-date or needs to be recomputed."""
        return self.geo_prior is None or sample.what_changed['prior']['zones']

    def get_geo_prior(self, sample, geo_prior_meta, network):
        """Compute the geo-prior of the current zones (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.
            geo_prior_meta (dict): Meta-information about the prior.
            network (dict): network containing the graph, location,...

        Returns:
            float: Logarithm of the prior probability density.
        """
        if self.geo_prior_outdated(sample):
            if geo_prior_meta['type'] == 'uniform':
                geo_prior = 0.

            elif geo_prior_meta['type'] == 'gaussian':
                geo_prior = geo_prior_gaussian(sample.zones, network, geo_prior_meta['parameters']['gaussian'])

            elif geo_prior_meta['type'] == 'distance':
                geo_prior = geo_prior_distance(sample.zones, network, geo_prior_meta['parameters']['distance'])

            else:
                raise ValueError('geo_prior must be either \"uniform\", \"gaussian\" or \"distance\".')

            # if "magnification_factor" in geo_prior_meta['parameters']:
            #    geo_prior = geo_prior * geo_prior_meta['parameters']['magnification_factor']

            self.geo_prior = geo_prior

        return self.geo_prior

    def size_prior_outdated(self, sample):
        """Check whether the cached size_prior is up-to-date or needs to be recomputed."""
        return self.size_prior is None or sample.what_changed['prior']['zones']

    def get_size_prior(self, sample):
        """Compute the size-prior of the current zone (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.

        Returns:
            float: Logarithm of the prior probability density.
        """
        if self.size_prior_outdated(sample):
            # size_prior = evaluate_size_prior(sample.zones)
            size_prior = 0.
            self.size_prior = size_prior

        return self.size_prior


def evaluate_size_prior(zones):
    """This function computes the prior probability of a set of zones, based on
    the number of languages in each zone.

    Args:
        zones (np.array): boolean array representing the current zone.
            shape: (n_zones, n_sites)
    Returns:
        float: log-probability of the zone sizes.
    """
    n_zones, n_sites = zones.shape
    sizes = np.sum(zones, axis=-1)

    # P(size)   =   uniform
    # TODO It would be quite natural to allow informative priors here.
    logp = 0.

    # P(zone | size)   =   1 / |{zones of size k}|   =   1 / (n choose k)
    logp += -np.sum(log_binom(n_sites, sizes))

    return logp


def geo_prior_gaussian(zones: np.array, network: dict, cov: np.array):
    """
    This function computes the two-dimensional Gaussian geo-prior for all edges in the zone
    Args:
        zones (np.array): boolean array representing the current zone
        network (dict): network containing the graph, location,...
        cov (np.array): Covariance matrix of the multivariate gaussian (estimated from the data)

    Returns:
        float: the log geo-prior of the zones
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

        else:
            raise ValueError("Too few locations to compute distance.")

        diffs = locations[i1] - locations[i2]
        prior_z = stats.multivariate_normal.logpdf(diffs, mean=[0, 0], cov=cov)
        log_prior = np.append(log_prior, prior_z)

    return np.mean(log_prior)


def geo_prior_distance(zones: np.array, network: dict, scale: float):

    """ This function computes the geo prior for the sum of all distances of the mst of a zone
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
        else:
            raise ValueError("Too few locations to compute distance.")

        log_prior = stats.expon.logpdf(distances, loc=0, scale=scale)

    return np.mean(log_prior)


def prior_p_global_dirichlet(p_global, dirichlet, categories, outdated_features,
                             cached_prior=None):
    """" This function evaluates the prior for p_families
    Args:
        p_global (np.array): p_global from the sample
        dirichlet (list): list of dirichlet distributions
        categories (list): list of available categories per feature
        outdated_features (IndexSet): The features which changed and need to be updated.
    Kwargs:
        cached_prior (list):

    Returns:
        float: the prior for p_global
    """
    _, n_feat, n_cat = p_global.shape

    if outdated_features.all:
        outdated_features = range(n_feat)
        log_prior = np.zeros(n_feat)
    else:
        log_prior = cached_prior

    for f in outdated_features:
        idx = categories[f]
        diri = dirichlet[f]
        p_glob = p_global[0, f, idx]

        if 0. in p_glob:
            p_glob[np.where(p_glob == 0.)] = EPS

        log_prior[f] = dirichlet_logpdf(x=p_glob, alpha=diri)
        # log_prior[f] = diri.logpdf(p_glob)

    return log_prior


def prior_p_families_dirichlet(p_families, dirichlet, categories, outdated_indices, outdated_distributions,
                               cached_prior=None, broadcast=False):
    """" This function evaluates the prior for p_families
    Args:
        p_families(np.array): p_families from the sample
        dirichlet(list): list of dirichlet distributions
        categories(list): list of available categories per feature
        outdated_indices (IndexSet): The features which need to be updated in each family.
        outdated_distributions (IndexSet): The features where the dirichlet distributions changed.
    Kwargs:
        cached_prior (list):
        broadcast (bool):


    Returns:
        float: the prior for p_families
    """
    n_fam, n_feat, n_cat = p_families.shape

    if cached_prior is None:
        assert outdated_indices.all
        log_prior = np.zeros((n_fam, n_feat))
    else:
        log_prior = cached_prior

    if outdated_indices.all or outdated_distributions.all:
        outdated_indices = itertools.product(range(n_fam), range(n_feat))
    else:
        if outdated_distributions:
            outdated_distributions_expanded = {(fam, feat) for feat in outdated_distributions for fam in range(n_fam)}
            outdated_indices = set.union(outdated_indices, outdated_distributions_expanded)

    for fam, feat in outdated_indices:

        if broadcast:
            # One prior is applied to all families
            diri = dirichlet[feat]

        else:
            # One prior per family
            diri = dirichlet[fam][feat]

        idx = categories[feat]
        p_fam = p_families[fam, feat, idx]

        log_prior[fam, feat] = dirichlet_logpdf(x=p_fam, alpha=diri)
        # log_prior[fam, feat] = diri.logpdf(p_fam)

    return log_prior
