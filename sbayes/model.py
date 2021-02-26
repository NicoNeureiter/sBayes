#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
from enum import Enum

import numpy as np

import scipy.stats as stats
from scipy.sparse.csgraph import minimum_spanning_tree

from sbayes.util import (compute_delaunay, n_smallest_distances, log_binom,
                         counts_to_dirichlet, inheritance_counts_to_dirichlet,
                         dirichlet_logpdf, scale_counts)
EPS = np.finfo(float).eps


class Model(object):

    """The sBayes model defining the posterior distribution of areas and parameters.

    Attributes:
        data (Data): data used in the likelihood and empirical priors.
        config (dict): a dictionary containing configuration parameters of the model.
        inheritance (bool): indicator whether or not inheritance is modeled
        likelihood (GenerativeLikelihood): the likelihood of the model.
        prior (GenerativePrior): the prior of the model.

    """

    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.parse_attributes(config)

        # Create likelihood and prior objects
        self.likelihood = GenerativeLikelihood(data=data, inheritance=self.inheritance)
        self.prior = GenerativePrior(data=data,
                                     inheritance=self.inheritance,
                                     prior_config=config['PRIOR'])

    def parse_attributes(self, config):
        """Read attributes from the config dictionary."""
        self.n_zones = config['N_AREAS']
        self.min_size = config['MIN_M']
        self.max_size = config['MAX_M']
        self.inheritance = config['INHERITANCE']
        self.sample_source = config['SAMPLE_SOURCE']

    def __call__(self, sample, caching=True):
        """Evaluate the (non-normalized) posterior probability of the given sample."""
        log_likelihood = self.likelihood(sample, caching=caching)
        log_prior = self.prior(sample)
        return log_likelihood + log_prior

    def __copy__(self):
        return Model(self.data, self.config)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        setup_msg = '\n'.join([
            f'Model',
            f'##########################################',
            f'Number of inferred areas: {self.n_zones}',
            f'Areas have a minimum size of {self.min_size} and a maximum size of {self.max_size}',
            f'Inheritance is considered for inference: {self.inheritance}',
            f'Geo-prior: {self.prior.config["geo"]["type"]}'
            f'Prior on weights: {self.prior.config["weights"]["type"]}',
            f'Prior on universal pressure (alpha): {self.prior.config["universal"]["type"]}',
            f'Prior on contact (gamma): {self.prior.config["contact"]["type"]}'
        ])

        if self.inheritance:
            setup_msg += f'\nPrior on inheritance(beta): {self.prior.config["inheritance"]["type"]}\n'

        return setup_msg


class GenerativeLikelihood(object):

    """Likelihood of the sBayes model.

    Attributes:
        features (np.array): The feature values for all sites and features.
            shape: (n_sites, n_features, n_categories)
        inharitance (bool): flag indicating whether inheritance (i.e. family distributions) is modelled or not.
        families (np.array): assignment of languages to families.
            shape: (n_families, n_sites)
        n_sites (int): number of sites (languages) in the sample.
        n_features (int): number of features in the data-set.
        n_categories (int): the maximum number of categories per feature.
        has_global (np.array): indicator whether a languages is part of global (always true).
            shape: (n_sites)
        has_family (np.array): indicators whether a language is in any family.
            shape: (n_sites)
        has_zone (np.array): indicators whether a languages is in any zone.
            shape: (n_sites)
        assignment (np.array): indicators whether a languages is affected by each
            mixture components (global, family, zone) in one array.
            shape: (n_sites, 3)
        global_lh (np.array): cached likelihood values for each site and feature according to global component.
            shape: (n_sites, n_features)
        family_lh (np.array): cached likelihood values for each site and feature according to family component.
            shape: (n_sites, n_features)
        zone_lh (np.array): cached likelihood values for each site and feature according to zone component.
            shape: (n_sites, n_features)
        all_lh (np.array): cached likelihood values for each site and feature according to each component.
            shape: (n_sites, n_features, 3)
        weights (np.array): cached normalized weights of each component in the final likelihood for each feature.
            shape: (n_sites, n_features, 3)
        na_features (np.array): bool array indicating missing observations
            shape: (n_sites, n_features)
    """

    def __init__(self, data, inheritance):
        self.features = data.features
        self.families = np.asarray(data.families, dtype=bool)
        self.inheritance = inheritance

        # Store relevant dimensions for convenience
        self.n_sites, self.n_features, self.n_categories = data.features.shape
        self.na_features = (np.sum(self.features, axis=-1) == 0)

        # Initialize attributes for caching

        # Assignment of languages to all component (global, per zone and family)
        self.has_global = np.ones(self.n_sites, dtype=bool)
        self.has_family = np.any(self.families, axis=0)
        self.has_zone = None
        self.has_components = None

        # The component likelihoods
        self.global_lh = None
        self.family_lh = None
        self.zone_lh = None

        # The combined and weighted and the non-normalized likelihood
        self.all_lh = None

        # Weights
        self.weights = None

    def reset_cache(self):
        # The assignment (global, zone, family) combined and weighted and the non-normalized likelihood
        self.has_components = None
        self.all_lh = None
        # Assignment and lh (global, per zone and family)
        self.has_zone = None
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

            Returns:
                float: The joint likelihood of the current sample.
            """

        if not caching:
            self.reset_cache()

        # Compute the likelihood values per mixture component
        all_lh = self.update_component_likelihoods(sample)
        # Compute the weights of the mixture component in each feature and site
        weights = self.update_weights(sample)

        # Compute the likelihood per site and feature
        if sample.source is not None:
            # Component contributions are sampled -> lh is defined by source labels
            feature_lh = np.sum(sample.source * weights * all_lh, axis=2)
        else:
            # Compute the weighted likelihood per feature and language
            feature_lh = np.sum(weights * all_lh, axis=2)

        # Sum up likelihood of all sites and features
        log_lh = np.sum(np.log(feature_lh))

        # The step is completed -> everything is up-to-date.
        self.everything_updated(sample)
        return log_lh

    def everything_updated(self, sample):
        sample.what_changed['lh']['zones'].clear()
        sample.what_changed['lh']['p_global'].clear()
        sample.what_changed['lh']['p_zones'].clear()
        sample.what_changed['lh']['weights'] = False
        if self.inheritance:
            sample.what_changed['lh']['p_families'].clear()

    def get_global_lh(self, sample):
        if (self.global_lh is None) or (sample.what_changed['lh']['p_global']):

            self.global_lh = compute_global_likelihood(features=self.features,
                                                       p_global=sample.p_global,
                                                       outdated_indices=sample.what_changed['lh']['p_global'],
                                                       cached_lh=self.global_lh)

        return self.global_lh

    def get_family_lh(self, sample):
        # Families are only evaluated if the model considers inheritance
        if not self.inheritance:
            return None

        # Family lh is evaluated when initialized and when p_families is changed
        if self.family_lh is None or sample.what_changed['lh']['p_families']:
            # assert np.allclose(a=np.sum(sample.p_families, axis=-1), b=1., rtol=EPS)
            self.family_lh = compute_family_likelihood(features=self.features, families=self.families,
                                                       p_families=sample.p_families,
                                                       outdated_indices=sample.what_changed['lh']['p_families'],
                                                       cached_lh=self.family_lh)

        return self.family_lh

    def get_zone_lh(self, sample):
        # Zone lh is evaluated when initialized, or when zones or p_zones change
        if self.zone_lh is None or sample.what_changed['lh']['zones'] or sample.what_changed['lh']['p_zones']:
            # assert np.allclose(a=np.sum(p_zones, axis=-1), b=1., rtol=EPS)
            self.zone_lh = compute_zone_likelihood(features=self.features, zones=sample.zones,
                                                   p_zones=sample.p_zones,
                                                   outdated_indices=sample.what_changed['lh']['p_zones'],
                                                   outdated_zones=sample.what_changed['lh']['zones'],
                                                   cached_lh=self.zone_lh)
        return self.zone_lh

    def update_component_likelihoods(self, sample, caching=True):
        # Update the likelihood valus for each of the mixture components
        global_lh = self.get_global_lh(sample)
        family_lh = self.get_family_lh(sample)
        zone_lh = self.get_zone_lh(sample)

        # Merge the component likelihoods into one array (if something has changed)
        if ((not caching) or (self.all_lh is None)
                or sample.what_changed['lh']['zones'] or sample.what_changed['lh']['p_global']
                or sample.what_changed['lh']['p_zones'] or sample.what_changed['lh']['p_families']):

            # Structure of likelihood depends on whether inheritance is considered or not
            if self.inheritance:
                self.all_lh = np.array([global_lh, zone_lh, family_lh]).transpose((1, 2, 0))
            else:
                self.all_lh = np.array([global_lh, zone_lh]).transpose((1, 2, 0))

            self.all_lh[self.na_features] = 1.

        return self.all_lh

    def get_zone_assignment(self, sample):
        """Update the zone assignment if necessary and return it."""
        if self.has_zone is None or sample.what_changed['lh']['zones']:
            self.has_zone = np.any(sample.zones, axis=0)
        return self.has_zone

    def update_weights(self, sample):
        """Compute the normalized weights of each component at each site.

        Args:
            sample (Sample): the current MCMC sample.

        Returns:
            np.array: normalized weights of each component at each site.
                shape: (n_sites, n_features, 3)
        """

        # The area assignment needs to be updated when the area changes
        self.has_zone = self.get_zone_assignment(sample)

        # Assignments are recombined when initialized or when zones change
        if self.has_components is None or sample.what_changed['lh']['zones']:
            # Structure of assignment depends on
            # whether inheritance is considered or not
            if self.inheritance:
                self.has_components = np.array([self.has_global,
                                                self.has_zone,
                                                self.has_family]).T
            else:
                self.has_components = np.array([self.has_global,
                                                self.has_zone]).T

        # weights are evaluated when initialized, when weights change or when assignment to zones changes
        if self.weights is None or sample.what_changed['lh']['weights'] or sample.what_changed['lh']['zones']:

            abnormal_weights = sample.weights

            # Extract weights for each site depending on whether the likelihood is available
            # Order of columns in weights: global, contact, inheritance (if available)
            # abnormal_weights_per_site = np.repeat(abnormal_weights[np.newaxis, :, :], self.n_sites, axis=0)
            abnormal_weights_per_site = abnormal_weights[np.newaxis, :, :]
            self.weights = normalize_weights(abnormal_weights_per_site, self.has_components)

        return self.weights


def compute_global_likelihood(features, p_global=None,
                              outdated_indices=None, cached_lh=None):
    """Computes the global likelihood, that is the likelihood per site and features
    without knowledge about family or zones.

    Args:
        features (np.array or 'SparseMatrix'): The feature values for all sites and features.
                shape: (n_sites, n_features, n_categories)
        p_global (np.array): The estimated global probabilities of all features in all site
            shape: (1, n_features, n_sites)
        outdated_indices (IndexSet): Features which changed, i.e. where lh needs to be recomputed.
        cached_lh (np.array): the global likelihood computed previously
    Returns:
        (np.array): the global likelihood per site and feature
            shape: (n_sites, n_features)
    """
    n_sites, n_features, n_categories = features.shape

    if cached_lh is None:
        lh_global = np.ones((n_sites, n_features))
        assert outdated_indices.all
    else:
        lh_global = cached_lh

    if outdated_indices.all:
        outdated_indices = range(n_features)

    for i_f in outdated_indices:
        f = features[:, i_f, :]

        # Compute the feature likelihood vector (for all sites in zone)
        lh_global[:, i_f] = f.dot(p_global[0, i_f, :])

    return lh_global


def compute_zone_likelihood(features, zones, p_zones=None,
                            outdated_indices=None, outdated_zones=None, cached_lh=None):
    """Computes the zone likelihood that is the likelihood per site and feature given zones z1, ... zn
    Args:
        features(np.array or 'SparseMatrix'): The feature values for all sites and features.
            shape: (n_sites, n_features, n_categories)
        zones(np.array): Binary arrays indicating the assignment of a site to the current zones.
            shape: (n_zones, n_sites)
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

    # if outdated_indices.all:
    #     outdated_indices = itertools.product(range(n_zones), range(n_features))
    # else:
    #     if outdated_zones:
    #         outdated_zones_expanded = {(zone, feat) for zone in outdated_zones for feat in range(n_features)}
    #         outdated_indices = set.union(outdated_indices, outdated_zones_expanded)
    #
    # # features_by_zone = {features[zones[z], :, :] for z in range(n_zones)}
    # features_by_zone = {}
    #
    # for z, i_f in outdated_indices:
    #     # Compute the feature likelihood vector (for all sites in zone)
    #     if z not in features_by_zone:
    #         features_by_zone[z] = features[zones[z], :, :]
    #     f = features_by_zone[z][:, i_f, :]
    #     p = p_zones[z, i_f, :]
    #     lh_zone[zones[z], i_f] = f.dot(p)

    if outdated_indices.all:
        outdated_zones = range(n_zones)
    else:
        outdated_zones = set.union(outdated_zones,
                                   {i_zone for (i_zone, i_feat) in outdated_indices})

    for z in outdated_zones:
        f_z = features[zones[z], :, :]
        p_z = p_zones[z, :, :]
        lh_zone[zones[z], :] = np.einsum('ijk,jk->ij', f_z, p_z)

    return lh_zone


def compute_family_likelihood(features, families, p_families=None,
                              outdated_indices=None, cached_lh=None):
    """Computes the family likelihood, that is the likelihood per site and feature given family f1, ... fn

    Args:
        features(np.array or 'SparseMatrix'): The feature values for all sites and features.
            shape: (n_sites, n_features, n_categories)
        families(np.array): Binary arrays indicating the assignment of a site to a family.
                shape: (n_families, n_sites)
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


def normalize_weights(weights, has_components):
    """This function assigns each site a weight if it has a likelihood and zero otherwise

        Args:
            weights (np.array): the weights to normalize
                shape: (n_sites, n_features, 3)
            has_components (np.array): boolean indicators, showing whether a language is
                affected by the universal distribution (always true), an areal distribution
                and a family distribution respectively.
                shape: (n_sites, 3)

        Return:
            np.array: the weight_per site
                shape: (n_sites, n_features, 3)
    """
    weights_per_site = weights * has_components[:, np.newaxis, :]
    return weights_per_site / weights_per_site.sum(axis=2, keepdims=True)


class GenerativePrior(object):

    """The joint prior of all parameters in the sBayes model.

    Attributes:
        inheritance (bool): activate family distribution.
        geo_prior_meta (dict): Meta-information about the prior.
        prior_area_size_meta (dict): Meta-information about the area size prior.
        prior_weights_meta (dict): Meta-information about the weights prior.
        prior_p_global_meta (dict): Meta-information about the p_global prior.
        prior_p_zones_meta (dict): Meta-information about the p_zones prior.
        prior_p_families_meta (dict): Meta-information about the p_family prior.
        network (dict): network containing the graph, location,...
        size_prior (float): cached value of the size_prior
        geo_prior (float): cached value of the geo_prior
        prior_weights (float): cached value of the prior_weights
        prior_p_global (float): cached value of the prior_p_global
        prior_p_zones (float): cached value of the prior_p_zones
        prior_p_families (float): cached value of the prior_p_families
        prior_p_families_distr (list): cached list of dirichlet counts for the family prior

    """

    def __init__(self, data, inheritance, prior_config):
        self.inheritance = inheritance
        self.data = data
        self.network = data.network

        self.size_prior = None
        self.geo_prior = None
        self.prior_weights = None
        self.prior_p_global = None
        self.prior_p_zones = None
        self.prior_p_families = None
        self.prior_p_families_distr = None

        self._prior_p_global = PriorPGlobal(config=prior_config['universal'], data=data)
        self._prior_p_families = PriorPFamilies(config=prior_config['universal'], data=data)

        self.parse_attributes(prior_config)

    def parse_attributes(self, config):
        config_parsed = dict.fromkeys(config)

        # TODO once everything is refactored this shouldn't do much -> use config right away

        # geo prior
        if config['geo']['type'] == 'uniform':
            config_parsed['geo'] = {'type': 'uniform'}
        elif config['geo']['type'] == 'cost_based':
            # todo:  change prior if cost matrix is provided
            # todo: move config['model']['scale_geo_prior'] to config['model']['PRIOR']['geo']['scale']
            config_parsed['geo'] = {'type': 'cost_based',
                                            'scale': config['geo']['scale']}
            raise NotImplementedError
        else:
            raise ValueError('Geo prior not supported')

        # area_size prior
        VALID_SIZE_PRIOR_TYPES = ['none', 'uniform', 'quadratic']
        size_prior_type = config['area_size']['type']
        if size_prior_type in VALID_SIZE_PRIOR_TYPES:
            config_parsed['area_size'] = {'type': size_prior_type}
        else:
            raise ValueError(f'Area-size prior ´{size_prior_type}´ not supported' +
                             f'(valid types: {VALID_SIZE_PRIOR_TYPES}).')

        # weights prior
        if config['weights']['type'] == 'uniform':
            config_parsed['weights'] = {'type': 'uniform'}
        else:
            raise ValueError('Currently only uniform prior_weights are supported.')

        # universal preference
        cfg_universal = config['universal']
        if cfg_universal['type'] == 'uniform':
            config_parsed['universal'] = {'type': 'uniform'}

        elif cfg_universal['type'] == 'counts':
            if cfg_universal['scale_counts'] is not None:
                self.data.prior_universal['counts'] = scale_counts(counts=self.data.prior_universal['counts'],
                                                                   scale_to=cfg_universal['scale_counts'])

            dirichlet = counts_to_dirichlet(self.data.prior_universal['counts'],
                                            self.data.states)
            config_parsed['universal'] = {'type': 'counts',
                                                  'dirichlet': dirichlet,
                                                  'states': self.data.states}

        else:
            raise ValueError('Prior for universal must be uniform or counts.')

        # inheritance
        cfg_inheritance = config['inheritance']
        if self.inheritance:
            if cfg_inheritance['type'] == 'uniform':
                config_parsed['inheritance'] = {'type': 'uniform'}

            elif cfg_inheritance['type'] is None:
                config_parsed['inheritance'] = {'type': None}

            elif cfg_inheritance['type'] == 'universal':
                config_parsed['inheritance'] = {'type': 'universal',
                                                        'strength': cfg_inheritance['scale_counts'],
                                                        'states': self.data.state_names['internal']}

            elif cfg_inheritance['type'] == 'counts':
                if cfg_inheritance['scale_counts'] is not None:
                    self.data.prior_inheritance['counts'] = scale_counts(
                        counts=self.data.prior_inheritance['counts'],
                        scale_to=cfg_inheritance['scale_counts'],
                        prior_inheritance=True
                    )
                dirichlet = inheritance_counts_to_dirichlet(self.data.prior_inheritance['counts'],
                                                            self.data.prior_inheritance['states'])
                config_parsed['inheritance'] = {'type': 'counts',
                                                        'dirichlet': dirichlet,
                                                        'states': self.data.prior_inheritance['states']}

            elif cfg_inheritance['type'] == 'counts_and_universal':
                if cfg_inheritance['scale_counts'] is not None:
                    self.data.prior_inheritance['counts'] = scale_counts(
                        counts=self.data.prior_inheritance['counts'],
                        scale_to=cfg_inheritance['scale_counts'],
                        prior_inheritance=True
                    )
                config_parsed['inheritance'] = {'type': 'counts_and_universal',
                                                        'counts': self.data.prior_inheritance['counts'],
                                                        'strength': cfg_inheritance['scale_counts'],
                                                        'states': self.data.prior_inheritance['states']}
            else:
                raise ValueError('Prior for inheritance must be uniform, counts or  counts_and_universal')
        else:
            config_parsed['inheritance'] = None

        # contact
        cfg_contact = config['contact']
        if cfg_contact['type'] == 'uniform':
            config_parsed['contact'] = {'type': 'uniform'}

        elif cfg_contact['type'] == 'universal':
            config_parsed['contact'] = {'type': 'universal',
                                                'strength': cfg_contact['scale_counts'],
                                                'states': self.data.state_names['internal']}
        else:
            raise ValueError('Prior for contact must be uniform or universal.')

        self.config = config_parsed
        self.geo_prior_meta = config_parsed['geo']
        self.prior_area_size_meta = config_parsed['area_size']
        self.prior_weights_meta = config_parsed['weights']
        self.prior_p_global_meta = config_parsed['universal']
        self.prior_p_zones_meta = config_parsed['contact']

        if self.inheritance:
            self.prior_p_families_meta = config_parsed['inheritance']
        else:
            self.prior_p_families_meta = None

    def __call__(self, sample):
        """Compute the prior of the current sample.
        Args:
            sample (Sample): A Sample object consisting of zones and weights

        Returns:
            float: The (log)prior of the current sample
        """
        # Compute the prior components
        size_prior = self.get_size_prior(sample)
        geo_prior = self.get_geo_prior(sample)
        prior_weights = self.get_prior_weights(sample)
        prior_p_global = self.get_prior_p_global(sample)
        prior_p_zones = self.get_prior_p_zones(sample)
        if self.inheritance:
            prior_p_families = self.get_prior_p_families(sample)
        else:
            prior_p_families = None

        # Add up prior components (in log space)
        log_prior = size_prior + geo_prior + prior_weights + prior_p_global + prior_p_zones
        if self.inheritance:
            log_prior += prior_p_families

        # The step is completed. Everything is up-to-date.
        sample.what_changed['prior']['zones'].clear()
        sample.what_changed['prior']['weights'] = False
        sample.what_changed['prior']['p_global'].clear()
        sample.what_changed['prior']['p_zones'].clear()

        if self.inheritance:
            sample.what_changed['prior']['p_families'].clear()

        return log_prior

    def weights_prior_outdated(self, sample):
        """Check whether the cached prior_weights is up-to-date or needs to be recomputed."""
        return self.prior_weights is None or sample.what_changed['prior']['weights']

    def get_prior_weights(self, sample):
        """Compute the prior for weights (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.

        Returns:
            float: Logarithm of the prior probability density.
        """
        if self.weights_prior_outdated(sample):
            if self.prior_weights_meta['type'] == 'uniform':
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

    def get_prior_p_zones(self, sample):
        """Compute the prior for p_zones (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.

        Returns:
            float: Logarithm of the prior probability density.
        """
        prior_type = self.prior_p_zones_meta['type']
        what_changed = sample.what_changed['prior']

        if self.p_zones_prior_outdated(sample, prior_type):
            if prior_type == 'uniform':
                prior_p_zones = 0.

            elif prior_type == 'universal':
                raise ValueError('Currently only uniform p_zones priors are supported.')
                # todo: check!
                # s = prior_p_zones_meta['strength']
                # c_universal = s * sample.p_global[0]
                #
                # self.prior_p_zones_distr = counts_to_dirichlet(counts=c_universal,
                #                                                categories=prior_p_zones_meta['states'],
                #                                                outdated_features=what_changed['p_global'],
                #                                                dirichlet=self.prior_p_zones_distr)
                # prior_p_zones = prior_p_families_dirichlet(p_families=sample.p_zones,
                #                                                 dirichlet=self.prior_p_zones_distr,
                #                                                 categories=prior_p_zones_meta['states'],
                #                                                 outdated_indices=what_changed['p_zones'],
                #                                                 outdated_distributions=what_changed['p_global'],
                #                                                 cached_prior=self.prior_p_zones,
                #                                                 broadcast=True)
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

    def get_prior_p_families(self, sample):
        """Compute the prior for p_families (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.

        Returns:
            float: Logarithm of the prior probability density.
        """
        prior_type = self.prior_p_families_meta['type']
        what_changed = sample.what_changed['prior']

        if self.p_families_prior_outdated(sample, prior_type):

            if prior_type == 'uniform':
                prior_p_families = 0.

            elif prior_type == 'counts':
                prior_p_families = prior_p_families_dirichlet(p_families=sample.p_families,
                                                              dirichlet=self.prior_p_families_meta['dirichlet'],
                                                              states=self.prior_p_families_meta['states'],
                                                              outdated_indices=what_changed['p_families'],
                                                              outdated_distributions=what_changed['p_global'],
                                                              cached_prior=self.prior_p_families,
                                                              broadcast=False)

            elif prior_type == 'universal':
                s = self.prior_p_families_meta['strength']
                c_universal = s * sample.p_global[0]
                self.prior_p_families_distr = counts_to_dirichlet(counts=c_universal,
                                                                  states=self.prior_p_families_meta['states'],
                                                                  outdated_features=what_changed['p_global'],
                                                                  dirichlet=self.prior_p_families_distr)

                prior_p_families = prior_p_families_dirichlet(p_families=sample.p_families,
                                                              dirichlet=self.prior_p_families_distr,
                                                              states=self.prior_p_families_meta['states'],
                                                              outdated_indices=what_changed['p_families'],
                                                              outdated_distributions=what_changed['p_global'],
                                                              cached_prior=self.prior_p_families,
                                                              broadcast=True)

            elif prior_type == 'counts_and_universal':
                s = self.prior_p_families_meta['strength']
                c_pseudocounts = self.prior_p_families_meta['counts']
                c_universal = s * sample.p_global

                self.prior_p_families_distr = \
                    inheritance_counts_to_dirichlet(counts=c_universal + c_pseudocounts,
                                                    states=self.prior_p_families_meta['states'],
                                                    outdated_features=what_changed['p_global'],
                                                    dirichlet=self.prior_p_families_distr)

                prior_p_families = prior_p_families_dirichlet(p_families=sample.p_families,
                                                              dirichlet=self.prior_p_families_distr,
                                                              states=self.prior_p_families_meta['states'],
                                                              outdated_indices=what_changed['p_families'],
                                                              outdated_distributions=what_changed['p_global'],
                                                              cached_prior=self.prior_p_families,
                                                              broadcast=False)

            else:
                raise ValueError('prior_p_families must be "uniform" or "counts"')

            self.prior_p_families = prior_p_families

        return np.sum(self.prior_p_families)

    def prior_p_global_outdated(self, sample):
        """Check whether the cached prior_p_global is up-to-date or needs to be recomputed."""
        return self.prior_p_global is None or sample.what_changed['prior']['p_global']

    def get_prior_p_global(self, sample):
        """Compute the prior for p_global (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.

        Returns:
            float: Logarithm of the prior probability density.
        """
        if self.prior_p_global_outdated(sample):

            if self.prior_p_global_meta['type'] == 'uniform':
                prior_p_global = 0.

            elif self.prior_p_global_meta['type'] == 'counts':
                prior_p_global = prior_p_global_dirichlet(p_global=sample.p_global,
                                                          dirichlet=self.prior_p_global_meta['dirichlet'],
                                                          states=self.prior_p_global_meta['states'],
                                                          outdated_features=sample.what_changed['prior']['p_global'],
                                                          cached_prior=self.prior_p_global)

            else:
                raise ValueError('Prior for universal pressures must be "uniform" or "counts')

            self.prior_p_global = prior_p_global

        return np.sum(self.prior_p_global)

    def geo_prior_outdated(self, sample):
        """Check whether the cached geo_prior is up-to-date or needs to be recomputed."""
        return self.geo_prior is None or sample.what_changed['prior']['zones']

    def get_geo_prior(self, sample):
        """Compute the geo-prior of the current zones (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.

        Returns:
            float: Logarithm of the prior probability density.
        """
        if self.geo_prior_outdated(sample):
            if self.geo_prior_meta['type'] == 'uniform':
                geo_prior = 0.

            elif self.geo_prior_meta['type'] == 'gaussian':
                geo_prior = geo_prior_gaussian(sample.zones, self.network,
                                               self.geo_prior_meta['gaussian'])

            elif self.geo_prior_meta['type'] == 'cost_based':
                geo_prior = geo_prior_distance(sample.zones, self.network,
                                               self.geo_prior_meta['scale'])

            else:
                raise ValueError('geo_prior must be either \"uniform\", \"gaussian\" or \"cost_based\".')

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
            size_prior = evaluate_size_prior(zones=sample.zones,
                                             size_prior_type=self.prior_area_size_meta['type'])
            self.size_prior = size_prior

        return self.size_prior

    def __copy__(self):
        prior_settings = {
            'geo': self.geo_prior_meta,
            'area_size': self.prior_area_size_meta,
            'weights': self.prior_weights_meta,
            'universal': self.prior_p_global_meta,
            'contact': self.prior_p_zones_meta,
            'inheritance': self.prior_p_families_meta
        }
        return GenerativePrior(data=self.data,
                               inheritance=self.inheritance,
                               prior_config=prior_settings)


class DirichletPrior(object):

    class TYPES(Enum):
        ...

    def __init__(self, config, data, initial_counts=1.):
        self.config = config
        self.data = data
        self.states = self.data.states
        self.initial_counts = initial_counts

        self.prior_type = None
        self.counts = None
        self.dirichlet = None

        self.cached = None

        self.parse_attributes(config)

    def parse_attributes(self, config):
        raise NotImplementedError()

    def is_outdated(self, sample):
        return self.cached is None

    def __call__(self, sample):
        raise NotImplementedError()

    def invalid_prior_message(self, s):
        valid_types = ','.join([str(t.value) for t in self.TYPES])
        return f'Invalid prior type {s} for universal prior (choose from [{valid_types}]).'


class PriorPGlobal(DirichletPrior):

    class TYPES(Enum):
        UNIFORM = 'uniform'
        COUNTS = 'counts'

    def parse_attributes(self, config):
        _, n_features, n_states = self.data.features.shape
        if config['type'] == 'uniform':
            self.prior_type = self.TYPES.UNIFORM
            self.counts = np.full(shape=(n_features, n_states),
                                  fill_value=self.initial_counts)

        elif config['type'] == 'counts':
            self.prior_type = self.TYPES.COUNTS
            if config['scale_counts'] is not None:
                self.data.prior_universal['counts'] = scale_counts(counts=self.data.prior_universal['counts'],
                                                                   scale_to=config['scale_counts'])
            self.counts = self.initial_counts + self.data.prior_universal['counts']
            self.dirichlet = counts_to_dirichlet(self.counts,
                                                 self.data.states)

        else:
            raise ValueError(self.invalid_prior_message(config['type']))

    def is_outdated(self, sample):
        """Check whether the cached prior_p_global is up-to-date or needs to be recomputed."""
        return (self.cached is None) or sample.what_changed['prior']['p_global']

    def __call__(self, sample):
        """Compute the prior for p_global (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.

        Returns:
            float: Logarithm of the prior probability density.
        """
        if not self.is_outdated(sample):
            return np.sum(self.cached)

        if self.prior_type is self.TYPES.UNIFORM:
            prior_p_global = 0

        elif self.prior_type is self.TYPES.COUNTS:
            prior_p_global = prior_p_global_dirichlet(
                p_global=sample.p_global,
                dirichlet=self.dirichlet,
                states=self.states,
                outdated_features=sample.what_changed['prior']['p_global'],
                cached_prior=self.cached
            )
        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

        self.cached = prior_p_global
        return np.sum(self.cached)


class PriorPFamilies(DirichletPrior):

    class TYPES(Enum):
        UNIFORM = 'uniform'
        COUNTS = 'counts'
        UNIVERSAL = 'universal'
        COUNTS_AND_UNIVERSAL = 'counts_and_universal'

    def parse_attributes(self, config):
        if config['type'] == 'uniform':
            n_families, _ = self.data.families.shape
            _, n_features, n_states = self.data.features.shape

            self.prior_type = self.TYPES.UNIFORM
            self.counts = np.full(shape=(n_families, n_features, n_states),
                                  fill_value=self.initial_counts)

        elif config['type'] == 'universal':
            self.prior_type = self.TYPES.UNIVERSAL
            self.strength = config['scale_counts']
            # self.states = self.data.state_names['internal']

        elif config['type'] == 'counts':
            self.prior_type = self.TYPES.COUNTS

            if config['scale_counts'] is not None:
                self.data.prior_inheritance['counts'] = scale_counts(
                    counts=self.data.prior_inheritance['counts'],
                    scale_to=config['scale_counts'],
                    prior_inheritance=True
                )
            self.counts = self.initial_counts + self.data.prior_inheritance['counts']
            self.dirichlet = inheritance_counts_to_dirichlet(
                counts=self.counts,
                states=self.states
            )
            # self.states = self.data.state_names['internal']

        elif config['type'] == 'counts_and_universal':
            self.prior_type = self.TYPES.COUNTS_AND_UNIVERSAL

            if config['scale_counts'] is not None:
                self.data.prior_inheritance['counts'] = scale_counts(
                    counts=self.data.prior_inheritance['counts'],
                    scale_to=config['scale_counts'],
                    prior_inheritance=True
                )
            # self.counts = self.initial_counts + self.data.prior_inheritance['counts']
            self.strength = config['scale_counts']
            # self.states = self.data.prior_inheritance['states']
        else:
            raise ValueError(self.invalid_prior_message(config['type']))

    def is_outdated(self, sample):
        """Check whether the cached prior_p_families is up-to-date or needs to be recomputed."""
        if self.cached is None:
            return True
        elif sample.what_changed['prior']['p_families']:
            return True
        elif (self.prior_type in [self.TYPES.UNIVERSAL, self.TYPES.COUNTS_AND_UNIVERSAL]) \
                and (sample.what_changed['prior']['p_global']):
            return True
        else:
            return False

    def __call__(self, sample):
        what_changed = sample.what_changed['prior']

        if not self.is_outdated(sample):
            return np.sum(self.cached)


        if self.prior_type == self.TYPES.UNIFORM:
            prior_p_families = 0.

        elif self.prior_type == self.TYPES.COUNTS:
            prior_p_families = prior_p_families_dirichlet(p_families=sample.p_families,
                                                          dirichlet=self.dirichlet,
                                                          states=self.states,
                                                          outdated_indices=what_changed['p_families'],
                                                          outdated_distributions=what_changed['p_global'],
                                                          cached_prior=self.cached,
                                                          broadcast=False)

        elif self.prior_type == self.TYPES.UNIVERSAL:
            c_universal = self.strength * sample.p_global[0]
            self.prior_p_families_distr = counts_to_dirichlet(counts=c_universal,
                                                              states=self.states,
                                                              outdated_features=what_changed['p_global'],
                                                              dirichlet=self.prior_p_families_distr)

            prior_p_families = prior_p_families_dirichlet(p_families=sample.p_families,
                                                          dirichlet=self.prior_p_families_distr,
                                                          states=self.states,
                                                          outdated_indices=what_changed['p_families'],
                                                          outdated_distributions=what_changed['p_global'],
                                                          cached_prior=self.cached,
                                                          broadcast=True)

        elif self.prior_type == self.TYPES.COUNTS_AND_UNIVERSAL:
            c_pseudocounts = self.counts
            c_universal = self.strength * sample.p_global[0]

            self.prior_p_families_distr = \
                inheritance_counts_to_dirichlet(counts=c_universal + c_pseudocounts,
                                                states=self.states,
                                                outdated_features=what_changed['p_global'],
                                                dirichlet=self.prior_p_families_distr)

            prior_p_families = prior_p_families_dirichlet(p_families=sample.p_families,
                                                          dirichlet=self.prior_p_families_distr,
                                                          states=self.states,
                                                          outdated_indices=what_changed['p_families'],
                                                          outdated_distributions=what_changed['p_global'],
                                                          cached_prior=self.cached,
                                                          broadcast=False)

        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

        self.cached = prior_p_families

        return np.sum(self.cached)


def evaluate_size_prior(zones, size_prior_type):
    """This function computes the prior probability of a set of zones, based on
    the number of languages in each zone.

    Args:
        zones (np.array): boolean array representing the current zone.
            shape: (n_zones, n_sites)
        size_prior_type (str): string identifier describing which type of area size prior to use.
    Returns:
        float: log-probability of the zone sizes.
    """
    # TODO It would be quite natural to allow informative priors here.

    n_zones, n_sites = zones.shape
    sizes = np.sum(zones, axis=-1)

    if size_prior_type == 'uniform':
        # P(size)   =   uniform
        # P(zone | size)   =   1 / |{zones of size k}|   =   1 / (n choose k)
        logp = -np.sum(log_binom(n_sites, sizes))
    elif size_prior_type == 'quadratic':
        # Here we assume that only a quadratically growing subset of zones is plausibly
        # permitted the likelihood and/or geo-prior.
        # P(zone | size) = 1 / |{"plausible" zones of size k}| = 1 / k**2
        log_plausible_zones = np.log(sizes**2)

        # We could bound the number of plausible zones by the number of possible zones:
        # log_possible_zones = log_binom(n_sites, sizes)
        # log_plausible_zones = np.minimum(np.log(sizes**2), log_possible_zones)

        logp = -np.sum(log_plausible_zones)
    elif size_prior_type == 'none':
        # No size prior
        # P(zone | size) = P(zone) = const.
        logp = 0.
    else:
        raise ValueError('Unknown size_prior_type: %s' % size_prior_type)

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


def prior_p_global_dirichlet(p_global, dirichlet, states, outdated_features, cached_prior=None):
    """" This function evaluates the prior for p_families
    Args:
        p_global (np.array): p_global from the sample
        dirichlet (list): list of dirichlet distributions
        states (list): list of available categories per feature
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
        idx = states[f]
        diri = dirichlet[f]
        p_glob = p_global[0, f, idx]

        log_prior[f] = dirichlet_logpdf(x=p_glob, alpha=diri)

    return log_prior


def prior_p_families_dirichlet(p_families, dirichlet, states, outdated_indices, outdated_distributions,
                               cached_prior=None, broadcast=False):
    """" This function evaluates the prior for p_families
    Args:
        p_families(np.array): p_families from the sample
        dirichlet(list): list of dirichlet distributions
        states(list): list of available categories per feature
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

        idx = states[feat]
        p_fam = p_families[fam, feat, idx]
        log_prior[fam, feat] = dirichlet_logpdf(x=p_fam, alpha=diri)
        # log_prior[fam, feat] = diri.logpdf(p_fam)

    return log_prior
