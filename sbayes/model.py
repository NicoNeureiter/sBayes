#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
from enum import Enum
import json
from typing import List

import numpy as np

import scipy.stats as stats
from scipy.sparse.csgraph import minimum_spanning_tree, csgraph_from_dense

from sbayes.util import (compute_delaunay, n_smallest_distances, log_binom, log_multinom,
                         counts_to_dirichlet, inheritance_counts_to_dirichlet,
                         dirichlet_logpdf, scale_counts)
EPS = np.finfo(float).eps


class Model(object):

    """The sBayes model defining the posterior distribution of areas and parameters.

    Attributes:
        data (Data): data used in the likelihood and empirical priors.
        config (dict): a dictionary containing configuration parameters of the model.
        inheritance (bool): indicator whether or not inheritance is modeled
        likelihood (Likelihood): the likelihood of the model.
        prior (Prior): the prior of the model.

    """

    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.parse_attributes(config)

        # Create likelihood and prior objects
        self.likelihood = Likelihood(
            data=data,
            inheritance=self.inheritance
            # missing_family_as_universal=self.config['missing_family_as_universal']
        )
        self.prior = Prior(
            data=data,
            inheritance=self.inheritance,
            prior_config=config['prior']
        )

    def parse_attributes(self, config):
        """Read attributes from the config dictionary."""
        self.n_zones = config['areas']
        self.min_size = config['prior']['languages_per_area']['min']
        self.max_size = config['prior']['languages_per_area']['max']
        self.inheritance = config['inheritance']
        self.sample_source = config['sample_source']

    def __call__(self, sample, caching=True):
        """Evaluate the (non-normalized) posterior probability of the given sample."""
        log_likelihood = self.likelihood(sample, caching=caching)
        log_prior = self.prior(sample)
        return log_likelihood + log_prior

    def __copy__(self):
        return Model(self.data, self.config)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        setup_msg = '\n'
        setup_msg += 'Model\n'
        setup_msg += '##########################################\n'
        setup_msg += f'Number of inferred areas: {self.n_zones}\n'
        setup_msg += f'Areas have a minimum size of {self.min_size} and a maximum ' \
                     f'size of {self.max_size}\n'
        setup_msg += f'Inheritance is considered for inference: {self.inheritance}\n'
        setup_msg += f'Family weights are added to universal weights for languages ' \
                     f'without family : {self.likelihood.missing_family_as_universal}\n'
        setup_msg += self.prior.get_setup_message()
        return setup_msg


class Likelihood(object):

    """Likelihood of the sBayes model.

    Attributes:
        features (np.array): The feature values for all sites and features.
            shape: (n_sites, n_features, n_categories)
        inheritance (bool): flag indicating whether inheritance (i.e. family distributions) is modelled or not.
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
        has_components (np.array): indicators whether a languages is affected by each
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

    def __init__(self, data, inheritance, missing_family_as_universal=False):
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

        self.missing_family_as_universal = missing_family_as_universal

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

        # Compute the total log-likelihood
        observation_lhs = self.get_observation_lhs(all_lh, weights, sample.source)
        sample.observation_lhs = observation_lhs
        log_lh = np.sum(np.log(observation_lhs))

        # Add the probability of observing the sampled (if sampled)
        if sample.source is not None:
            is_source = np.where(sample.source.ravel())
            p_source = weights.ravel()[is_source]
            log_lh += np.sum(np.log(p_source))

        # The step is completed -> everything is up-to-date.
        self.everything_updated(sample)

        return log_lh

    def get_observation_lhs(self, all_lh, weights, source):
        if source is None:
            return np.sum(weights * all_lh, axis=2).ravel()
        else:
            is_source = np.where(source.ravel())
            return all_lh.ravel()[is_source]

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
            # Extract weights for each site depending on whether the likelihood is available
            # Order of columns in weights: global, contact, inheritance (if available)
            self.weights = normalize_weights(
                weights=sample.weights,
                has_components=self.has_components,
                missing_family_as_universal=self.missing_family_as_universal
            )

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


def normalize_weights(weights, has_components, missing_family_as_universal=False):
    """This function assigns each site a weight if it has a likelihood and zero otherwise

    Args:
        weights (np.array): the weights to normalize
            shape: (n_features, 3)
        has_components (np.array): boolean indicators, showing whether a language is
            affected by the universal distribution (always true), an areal distribution
            and a family distribution respectively.
            shape: (n_sites, 3)
        missing_family_as_universal (bool): Add family weights to the universal distribution instead
            of re-normalizing when family is absent.

    Return:
        np.array: the weight_per site
            shape: (n_sites, n_features, 3)

    == Usage ===
    >>> normalize_weights(weights=np.array([[0.2, 0.2, 0.6],
    ...                                     [0.2, 0.6, 0.2]]),
    ...                   has_components=np.array([[True, False, True],
    ...                                            [True, True, False]]),
    ...                   missing_family_as_universal=False)
    array([[[0.25, 0.  , 0.75],
            [0.5 , 0.  , 0.5 ]],
    <BLANKLINE>
           [[0.5 , 0.5 , 0.  ],
            [0.25, 0.75, 0.  ]]])
    >>> normalize_weights(weights=np.array([[0.2, 0.2, 0.6],
    ...                                     [0.2, 0.6, 0.2]]),
    ...                   has_components=np.array([[True, False, True],
    ...                                            [True, True, False]]),
    ...                   missing_family_as_universal=True)
    array([[[0.25, 0.  , 0.75],
            [0.5 , 0.  , 0.5 ]],
    <BLANKLINE>
           [[0.8 , 0.2 , 0.  ],
            [0.4 , 0.6 , 0.  ]]])

    """
    inheritance = ((weights.shape[-1]) == 3)

    # Broadcast weights to each site and mask with the has_components arrays (so that
    # area-/family-weights in languages without area/family are set to 0.
    # Broadcasting:
    #   `weights` doesnt know about sites -> add axis to broadcast to the sites-dimension of `has_component`
    #   `has_components` doesnt know about features -> add axis to broadcast to the features-dimension of `weights`
    weights_per_site = weights[np.newaxis, :, :] * has_components[:, np.newaxis, :]

    if inheritance and missing_family_as_universal:
        # If `missing_family_as_universal` is set, we assume that the missing family
        # distribution for isolates (or languages who are the only sample from their
        # family) is best replaced by the universal distribution -> shift the weight
        # accordingly from w[l, :, 2] (family weight) to w[l, :, 0] (universal weight).
        without_family = ~has_components[:, 2]
        weights_per_site[without_family, :, 0] += weights[:, 2]
        assert np.all(weights_per_site[without_family, :, 2] == 0.)

    # Re-normalize the weights, where weights were masked (and not added to universal)
    return weights_per_site / weights_per_site.sum(axis=2, keepdims=True)


class Prior(object):

    """The joint prior of all parameters in the sBayes model.

    Attributes:
        inheritance (bool): activate family distribution.
        network (dict): network containing the graph, location,...
        size_prior (ZoneSizePrior): prior on the area size
        geo_prior (GeoPrior): prior on the geographic spread of an area
        prior_weights (WeightsPrior): prior on the mixture weights
        prior_p_global (PGlobalPrior): prior on the p_global parameters
        prior_p_zones (PZonesPrior): prior on the p_zones parameters
        prior_p_families (PFamiliesPrior): prior on the p_families parameters
    """

    def __init__(self, data, inheritance, prior_config):
        self.inheritance = inheritance
        self.data = data
        self.network = data.network
        self.config = prior_config

        self.size_prior = ZoneSizePrior(config=prior_config['languages_per_area'], data=data)
        self.geo_prior = GeoPrior(config=prior_config['geo'], data=data)
        self.prior_weights = WeightsPrior(config=prior_config['weights'], data=data)
        self.prior_p_global = PGlobalPrior(config=prior_config['universal'], data=data)
        self.prior_p_zones = PZonesPrior(config=prior_config['contact'], data=data)
        if self.inheritance:
            self.prior_p_families = PFamiliesPrior(config=prior_config['inheritance'], data=data)

    def __call__(self, sample):
        """Compute the prior of the current sample.
        Args:
            sample (Sample): A Sample object consisting of zones and weights

        Returns:
            float: The (log)prior of the current sample
        """
        log_prior = 0

        # Sum all prior components (in log-space)
        log_prior += self.size_prior(sample)
        log_prior += self.geo_prior(sample)
        log_prior += self.prior_weights(sample)
        log_prior += self.prior_p_global(sample)
        log_prior += self.prior_p_zones(sample)
        if self.inheritance:
            log_prior += self.prior_p_families(sample)

        self.everything_updated(sample)

        return log_prior

    def everything_updated(self, sample):
        """Mark all parameters in ´sample´ as updated."""
        sample.what_changed['prior']['zones'].clear()
        sample.what_changed['prior']['weights'] = False
        sample.what_changed['prior']['p_global'].clear()
        sample.what_changed['prior']['p_zones'].clear()
        if self.inheritance:
            sample.what_changed['prior']['p_families'].clear()

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        setup_msg = self.geo_prior.get_setup_message()
        setup_msg += self.size_prior.get_setup_message()
        setup_msg += self.prior_weights.get_setup_message()
        setup_msg += self.prior_p_global.get_setup_message()
        setup_msg += self.prior_p_zones.get_setup_message()
        if self.inheritance:
            setup_msg += self.prior_p_families.get_setup_message()
        return setup_msg

    def __copy__(self):
        return Prior(data=self.data,
                     inheritance=self.inheritance,
                     prior_config=self.config)


class DirichletPrior(object):

    class TYPES(Enum):
        UNIFORM = 'uniform'
        DIRICHLET = 'dirichlet'

    def __init__(self, config, data, initial_counts=1.):

        self.config = config
        self.data = data
        self.states = self.data.states
        self.initial_counts = initial_counts

        self.prior_type = None
        self.counts = None
        self.concentration = None

        self.cached = None

        self.n_sites, self.n_features, self.n_states = data.features.shape

        self.parse_attributes(config)

    def load_concentration(self, config: dict) -> List[np.ndarray]:
        if 'file' in config:
            return self.parse_concentration_json(config['file'])
        elif 'parameters' in config:
            return self.parse_concentration_dict(config['parameters'])

    def parse_concentration_json(self, json_path: str) -> List[np.ndarray]:
        # Read the concentration parameters from the JSON file
        with open(json_path, 'r') as f:
            concentration_dict = json.load(f)

        # Parse the resulting dictionary
        return self.parse_concentration_dict(concentration_dict)

    def parse_concentration_dict(self, concentration_dict: dict) -> List[np.ndarray]:
        # Get feature_names and state_names lists to put parameters in the right order
        feature_names = self.data.feature_names['external']
        state_names = self.data.state_names['external']
        assert len(state_names) == len(feature_names) == self.n_features

        # Compile the array with concentration parameters
        concentration = []
        for f, state_names_f in zip(feature_names, state_names):
            conc_f = [concentration_dict[f][s] for s in state_names_f]
            concentration.append(np.array(conc_f))

        return concentration

    def get_uniform_concentration(self) -> List[np.ndarray]:
        concentration = []
        for state_names_f in self.data.state_names['external']:
            concentration.append(
                np.ones(shape=len(state_names_f))
            )
        return concentration


    def parse_attributes(self, config):
        raise NotImplementedError()

    def is_outdated(self, sample):
        return self.cached is None

    def __call__(self, sample):
        raise NotImplementedError()

    def invalid_prior_message(self, s):
        name = self.__class__.__name__
        valid_types = ','.join([str(t.value) for t in self.TYPES])
        return f'Invalid prior type {s} for {name} (choose from [{valid_types}]).'


class PGlobalPrior(DirichletPrior):

    def parse_attributes(self, config):
        if config['type'] == 'uniform':
            self.prior_type = self.TYPES.UNIFORM
            self.concentration = self.get_uniform_concentration()
        elif config['type'] == 'dirichlet':
            self.prior_type = self.TYPES.DIRICHLET
            self.concentration = self.load_concentration(config)
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

        elif self.prior_type is self.TYPES.DIRICHLET:
            prior_p_global = prior_p_global_dirichlet(
                p_global=sample.p_global,
                concentration=self.concentration,
                applicable_states=self.states,
                outdated_features=sample.what_changed['prior']['p_global'],
                cached_prior=self.cached
            )
        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

        self.cached = prior_p_global
        return np.sum(self.cached)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        msg = f'Universal-prior type: {self.prior_type.value}\n'
        if self.prior_type == self.TYPES.DIRICHLET:
            msg += f'\tCounts file: {self.config["file"]}\n'
        return msg


class PFamiliesPrior(DirichletPrior):

    # TODO (NN): Turn prior_type UNIVERSAL into a separate flag (can be combined with any other fixed prior).

    def __init__(self, config, data, initial_counts=1.):
        self.universal_as_prior = False
        self.universal_concentration = 2.0
        super(PFamiliesPrior, self).__init__(config, data, initial_counts=initial_counts)

    def parse_attributes(self, config):
        n_families, _ = self.data.families.shape
        _, n_features, n_states = self.data.features.shape

        self.concentration = [np.empty(0) for _ in range(n_families)]

        # @Nico: Define uniform prior and adapt per family if necessary
        for i_fam, family in enumerate(self.data.family_names['external']):
            if (family not in config) or (config[family]['type'] == 'uniform'):
                self.concentration[i_fam] = self.get_uniform_concentration()
            elif config[family]['type'] == 'dirichlet':
                self.concentration[i_fam] = self.load_concentration(config[family])
            else:
                raise ValueError(self.invalid_prior_message(config['type']))

        # elif config['type'] == 'counts_and_universal':
        #     self.prior_type = self.TYPES.COUNTS_AND_UNIVERSAL
        #
        #     if config['scale_counts'] is not None:
        #         self.data.prior_inheritance['counts'] = scale_counts(
        #             counts=self.data.prior_inheritance['counts'],
        #             scale_to=config['scale_counts'],
        #             prior_inheritance=True
        #         )
        #     # self.counts = self.initial_counts + self.data.prior_inheritance['counts']
        #     self.strength = config['scale_counts']
        #     # self.states = self.data.prior_inheritance['states']
        # else:
        #     raise ValueError(self.invalid_prior_message(config['type']))

    def is_outdated(self, sample):
        """Check whether the cached prior_p_families is up-to-date or needs to be recomputed."""
        if self.cached is None:
            return True
        elif sample.what_changed['prior']['p_families']:
            return True
        elif self.universal_as_prior and sample.what_changed['prior']['p_global']:
            return True
        else:
            return False

    def __call__(self, sample):
        what_changed = sample.what_changed['prior']

        if not self.is_outdated(sample):
            return np.sum(self.cached)

        current_concentration = self.concentration
        if self.universal_as_prior:
            # TODO Update the concentration parameter with hyperprior_concentration * p_global
            ...

        prior_p_families = prior_p_families_dirichlet(p_families=sample.p_families,
                                                      concentration=current_concentration,
                                                      applicable_states=self.states,
                                                      outdated_indices=what_changed['p_families'],
                                                      outdated_distributions=what_changed['p_global'], # TODO this is only necessary if we us unversal_as_hyperprior
                                                      cached_prior=self.cached,
                                                      broadcast=False)

        # elif self.prior_type == self.TYPES.UNIVERSAL:
        #     c_universal = self.strength * sample.p_global[0]
        #     self.prior_p_families_distr = counts_to_dirichlet(counts=c_universal,
        #                                                       states=self.states,
        #                                                       outdated_features=what_changed['p_global'],
        #                                                       dirichlet=self.prior_p_families_distr)
        #
        #     prior_p_families = prior_p_families_dirichlet(p_families=sample.p_families,
        #                                                   concentration=self.prior_p_families_distr,
        #                                                   applicable_states=self.states,
        #                                                   outdated_indices=what_changed['p_families'],
        #                                                   outdated_distributions=what_changed['p_global'],
        #                                                   cached_prior=self.cached,
        #                                                   broadcast=True)
        #
        # elif self.prior_type == self.TYPES.COUNTS_AND_UNIVERSAL:
        #     c_pseudocounts = self.counts
        #     c_universal = self.strength * sample.p_global[0]
        #
        #     self.prior_p_families_distr = \
        #         inheritance_counts_to_dirichlet(counts=c_universal + c_pseudocounts,
        #                                         states=self.states,
        #                                         outdated_features=what_changed['p_global'],
        #                                         dirichlet=self.prior_p_families_distr)
        #
        #     prior_p_families = prior_p_families_dirichlet(p_families=sample.p_families,
        #                                                   concentration=self.prior_p_families_distr,
        #                                                   applicable_states=self.states,
        #                                                   outdated_indices=what_changed['p_families'],
        #                                                   outdated_distributions=what_changed['p_global'],
        #                                                   cached_prior=self.cached,
        #                                                   broadcast=False)
        #
        # else:
        #     raise ValueError(self.invalid_prior_message(self.prior_type))

        self.cached = prior_p_families

        return np.sum(self.cached)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        msg = f'Prior on inheritance (beta):\n'

        for i_fam, family in enumerate(self.data.family_names['external']):
            if family not in self.config:
                msg += f'\tNo inheritance prior defined for family {family}, using a uniform prior.\n'
            elif self.config[family]['type'] == 'uniform':
                msg += f'\tUniform prior for family {family}.\n'
            elif self.config[family]['type'] == 'dirichlet':
                msg += f'\tDirichlet prior for family {family}.'
                if 'parameters' in self.config[family]:
                    msg += f'Parameters defined in config.\n'
                elif 'file' in self.config[family]:
                    msg += f'Parameters defined in {self.config[family]["file"]}.\n'
            else:
                raise ValueError(self.invalid_prior_message(self.config['type']))

        if self.universal_as_prior:
            msg += f'\tUniversal hyperprior concentration: {self.universal_concentration}\n'

        return msg


class PZonesPrior(DirichletPrior):

    class TYPES(Enum):
        UNIFORM = 'uniform'
        UNIVERSAL = 'universal'

        # TODO like in PFamilyPrior: Turn the type UNIVERSAL into a separate flag.

    def parse_attributes(self, config):
        _, n_features, n_states = self.data.features.shape

        if config['type'] == 'uniform':
            self.prior_type = self.TYPES.UNIFORM
            self.counts = np.full(shape=(n_features, n_states),
                                  fill_value=self.initial_counts)

        elif config['type'] == 'universal':
            self.prior_type = self.TYPES.UNIVERSAL
            self.strength = config['scale_counts']
            self.states = self.data.state_names['internal']

        else:
            raise ValueError(self.invalid_prior_message(config['type']))

    def is_outdated(self, sample):
        """Check whether the cached prior_p_global is up-to-date or needs to be recomputed."""
        if self.cached is None:
            return True
        elif sample.what_changed['prior']['p_zones']:
            return True
        elif self.prior_type == self.TYPES.UNIVERSAL and sample.what_changed['prior']['p_global']:
            return True
        else:
            return False

    def __call__(self, sample):
        """Compute the prior for p_zones (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.

        Returns:
            float: Logarithm of the prior probability density.
        """
        if not self.is_outdated(sample):
            return np.sum(self.cached)

        if self.prior_type == self.TYPES.UNIFORM:
            prior_p_zones = 0.

        elif self.prior_type == self.TYPES.UNIVERSAL:
            raise ValueError('Currently only uniform p_zones priors are supported.')
            # todo: check!
            # s = self.strength
            # c_universal = s * sample.p_global[0]
            #
            # self.prior_p_zones_distr = counts_to_dirichlet(counts=c_universal,
            #                                                categories=self.states,
            #                                                outdated_features=what_changed['p_global'],
            #                                                dirichlet=self.prior_p_zones_distr)
            # prior_p_zones = prior_p_families_dirichlet(p_families=sample.p_zones,
            #                                                 dirichlet=self.prior_p_zones_distr,
            #                                                 categories=self.states,
            #                                                 outdated_indices=what_changed['p_zones'],
            #                                                 outdated_distributions=what_changed['p_global'],
            #                                                 cached_prior=self.prior_p_zones,
            #                                                 broadcast=True)
        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

        self.cached = prior_p_zones
        return np.sum(self.cached)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f'Prior on contact (gamma): {self.prior_type.value}\n'


class WeightsPrior(DirichletPrior):

    class TYPES(Enum):
        UNIFORM = 'uniform'

    def parse_attributes(self, config):
        _, n_features, n_states = self.data.features.shape

        if config['type'] == 'uniform':
            self.prior_type = self.TYPES.UNIFORM
            self.counts = np.full(shape=(n_features, n_states),
                                  fill_value=self.initial_counts)

        else:
            raise ValueError(self.invalid_prior_message(config['type']))

    def is_outdated(self, sample):
        return self.cached is None or sample.what_changed['prior']['weights']

    def __call__(self, sample):
        """Compute the prior for weights (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.

        Returns:
            float: Logarithm of the prior probability density.
        """
        if not self.is_outdated(sample):
            return np.sum(self.cached)

        if self.prior_type == self.TYPES.UNIFORM:
            prior_weights = 0.
        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

        self.cached = prior_weights
        return np.sum(self.cached)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f'Prior on weights: {self.prior_type.value}\n'


class SourcePrior(object):

    def __init__(self, config, likelihood: Likelihood):
        self.config = config
        self.likelihood = likelihood

    def __call__(self, sample):
        """Compute the prior for weights (or load from cache).
        Args:
            sample (Sample): Current MCMC sample.
        Returns:
            float: Logarithm of the prior probability density.
        """
        weights = self.likelihood.update_weights(sample)
        is_source = np.where(sample.source.ravel())
        observation_weights = weights.ravel()[is_source]
        return np.sum(np.log(observation_weights))


class ZoneSizePrior(object):

    class TYPES(Enum):
        UNIFORM_AREA = 'uniform_area'
        UNIFORM_SIZE = 'uniform_size'
        QUADRATIC_SIZE = 'quadratic'

    def __init__(self, config, data, initial_counts=1.):
        self.config = config
        self.data = data
        self.initial_counts = initial_counts

        self.prior_type = None
        self.counts = None
        self.dirichlet = None

        self.cached = None

        self.parse_attributes(config)

    def invalid_prior_message(self, s):
        valid_types = ','.join([str(t.value) for t in self.TYPES])
        return f'Invalid prior type {s} for size prior (choose from [{valid_types}]).'

    def parse_attributes(self, config):
        size_prior_type = config['type']
        if size_prior_type == 'uniform_area':
            self.prior_type = self.TYPES.UNIFORM_AREA
        elif size_prior_type == 'uniform_size':
            self.prior_type = self.TYPES.UNIFORM_SIZE
        elif size_prior_type == 'quadratic':
            self.prior_type = self.TYPES.QUADRATIC_SIZE
        else:
            raise ValueError(self.invalid_prior_message(size_prior_type))

    def is_outdated(self, sample):
        return self.cached is None or sample.what_changed['prior']['zones']

    def __call__(self, sample):
        """Compute the prior probability of a set of zones, based on the number of
        languages in each zone.

        Args:
            sample (Sample): Current MCMC sample.

        Returns:
            float: Log-probability of the zone sizes.
        """
        # TODO It would be quite natural to allow informative priors here.

        if self.is_outdated(sample):
            n_zones, n_sites = sample.zones.shape
            sizes = np.sum(sample.zones, axis=-1)

            if self.prior_type == self.TYPES.UNIFORM_SIZE:
                # P(size)   =   uniform
                # P(zone | size)   =   1 / |{zones of size k}|   =   1 / (n choose k)
                # logp = -np.sum(log_binom(n_sites, sizes))
                logp = -log_multinom(n_sites, sizes)

            elif self.prior_type == self.TYPES.QUADRATIC_SIZE:
                # Here we assume that only a quadratically growing subset of zones is
                # plausibly permitted by the likelihood and/or geo-prior.
                # P(zone | size) = 1 / |{"plausible" zones of size k}| = 1 / k**2
                log_plausible_zones = np.log(sizes ** 2)

                # We could bound the number of plausible zones by the number of possible zones:
                # log_possible_zones = log_binom(n_sites, sizes)
                # log_plausible_zones = np.minimum(np.log(sizes**2), log_possible_zones)

                logp = -np.sum(log_plausible_zones)
            elif self.prior_type == self.TYPES.UNIFORM_AREA:
                # No size prior
                # P(zone | size) = P(zone) = const.
                logp = 0.
            else:
                raise ValueError(self.invalid_prior_message(self.prior_type))

            self.cached = logp

        return self.cached

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f'Prior on area size: {self.prior_type.value}\n'

    @staticmethod
    def sample(prior_type, n_zones, n_sites):
        if prior_type == ZoneSizePrior.TYPES.UNIFORM_AREA:
            onehots = np.eye(n_zones+1, n_zones, dtype=bool)
            return onehots[np.random.randint(0, n_zones+1, size=n_sites)].T
        else:
            raise NotImplementedError()

class GeoPrior(object):

    class TYPES(Enum):
        UNIFORM = 'uniform'
        GAUSSIAN = 'gaussian'
        COST_BASED = 'cost_based'

    def __init__(self, config, data, initial_counts=1.):
        self.config = config
        self.data = data
        self.initial_counts = initial_counts

        self.prior_type = None
        self.cost_matrix = None
        self.scale = None
        self.cached = None

        self.parse_attributes(config)

    def parse_attributes(self, config):
        if config['type'] == 'uniform':
            self.prior_type = self.TYPES.UNIFORM

        elif config['type'] == 'gaussian':
            self.prior_type = self.TYPES.GAUSSIAN
            self.covariance = config['covariance']

        elif config['type'] == 'cost_based':
            self.prior_type = self.TYPES.COST_BASED
            self.cost_matrix = self.data.geo_prior['cost_matrix']
            self.scale = config['rate']
            self.aggregation = config['aggregation']
        else:
            raise ValueError('Geo prior not supported')

    def is_outdated(self, sample):
        return self.cached is None or sample.what_changed['prior']['zones']

    def __call__(self, sample):
        """Compute the size-prior of the current zone (or load from cache).

        Args:
            sample (Sample): Current MCMC sample.

        Returns:
            float: Logarithm of the prior probability density.
        """
        if self.is_outdated(sample):
            if self.prior_type is self.TYPES.UNIFORM:
                geo_prior = 0.

            elif self.prior_type is self.TYPES.GAUSSIAN:
                geo_prior = geo_prior_gaussian(
                    zones=sample.zones,
                    network=self.data.network,
                    cov=self.covariance
                )

            elif self.prior_type is self.TYPES.COST_BASED:
                geo_prior = geo_prior_distance(
                    zones=sample.zones,
                    cost_mat=self.cost_matrix,
                    scale=self.scale,
                    aggregation=self.aggregation
                )

            else:
                raise ValueError('geo_prior must be either \"uniform\", \"gaussian\" or \"cost_based\".')

            self.cached = geo_prior

        return self.cached

    def invalid_prior_message(self, s):
        valid_types = ','.join([str(t.value) for t in self.TYPES])
        return f'Invalid prior type {s} for geo-prior (choose from [{valid_types}]).'

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        msg = f'Geo-prior: {self.prior_type.value}\n'
        if self.prior_type == self.TYPES.COST_BASED:
            msg += f'\tScale: {self.scale}\n'
            if self.config['costs'] == 'from_data':
                msg += '\tCost-matrix inferred from geo-locations.\n'
            else:
                msg += f'\tCost-matrix file: {self.config["costs"]}\n'
        return msg


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


def geo_prior_distance(zones: np.array,
                       cost_mat: np.array,
                       scale: float,
                       aggregation: str):
    """ This function computes the geo prior for the sum of all distances of the mst of a zone
    Args:
        zones (np.array): The current zones (boolean array)
        cost_mat (np.array): The cost matrix between locations
        scale (float): The scale parameter of an exponential distribution
        aggregation (str): The aggregation policy, defining how the single edge
            costs are combined into one joint cost for the area.

    Returns:
        float: the log geo-prior of the zones
    """

    AGGREGATORS = {
        'mean': np.mean,
        'sum': np.sum,
        'max': np.min,  # sic: max distance == min log-likelihood
    }
    if aggregation in AGGREGATORS:
        aggregator = AGGREGATORS[aggregation]
    else:
        raise ValueError(f'Unknown aggregation policy "{aggregation}" in geo prior.')


    log_prior = 0.0
    for z in zones:
        cost_mat_z = cost_mat[z][:, z]

        # if len(locations) > 3:
        #
        #     delaunay = compute_delaunay(locations)
        #     mst = minimum_spanning_tree(delaunay.multiply(dist_mat))
        #     distances = mst.tocsr()[mst.nonzero()]
        #
        # elif len(locations) == 3:
        #     distances = n_smallest_distances(dist_mat, n=2, return_idx=False)
        #
        # elif len(locations) == 2:
        #     distances = n_smallest_distances(dist_mat, n=1, return_idx=False)

        if cost_mat_z.shape[0] > 1:
            graph = csgraph_from_dense(cost_mat_z, null_value=np.inf)
            mst = minimum_spanning_tree(graph)

            # When there are zero costs between languages the MST might be 0
            if mst.nnz > 0:

                distances = mst.tocsr()[mst.nonzero()]
            else:
                distances = 0
        else:
            raise ValueError("Too few locations to compute distance.")

        log_prior_per_edge = stats.expon.logpdf(distances, loc=0, scale=scale)
        log_prior += aggregator(log_prior_per_edge)

    return log_prior


def prior_p_global_dirichlet(p_global, concentration, applicable_states, outdated_features,
                             cached_prior=None):
    """" This function evaluates the prior for p_families
    Args:
        p_global (np.array): p_global from the sample
        concentration (list): list of Dirichlet concentration parameters
        applicable_states (list): list of available categories per feature
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
        states_f = applicable_states[f]
        log_prior[f] = dirichlet_logpdf(
            x=p_global[0, f, states_f],
            alpha=concentration[f]
        )

    return log_prior


def prior_p_families_dirichlet(p_families, concentration, applicable_states,
                               outdated_indices, outdated_distributions,
                               cached_prior=None, broadcast=False):
    """" This function evaluates the prior for p_families
    Args:
        p_families (np.array): p_families from the sample
        concentration (list): List of Dirichlet concentration parameters
        applicable_states (list): List of available categories per feature
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
            outdated_distributions_expanded = {(fam, f) for f in outdated_distributions for fam in range(n_fam)}
            outdated_indices = set.union(outdated_indices, outdated_distributions_expanded)

    for fam, f in outdated_indices:

        if broadcast:
            # One prior is applied to all families
            conc_f = concentration[f]

        else:
            # One prior per family
            conc_f = concentration[fam][f]

        states_f = applicable_states[f]
        p_fam = p_families[fam, f, states_f]
        log_prior[fam, f] = dirichlet_logpdf(x=p_fam, alpha=conc_f)
        # log_prior[fam, f] = diri.logpdf(p_fam)

    return log_prior



if __name__ == '__main__':
    import doctest
    doctest.testmod()
