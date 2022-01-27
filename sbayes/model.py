#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
from enum import Enum
from typing import List

import numpy as np

import scipy.stats as stats
from scipy.sparse.csgraph import minimum_spanning_tree, csgraph_from_dense

from sbayes.util import (compute_delaunay, n_smallest_distances,
                         log_multinom, dirichlet_logpdf)
EPS = np.finfo(float).eps


class Model(object):
    """The sBayes model: posterior distribution of clusters and parameters.

    Attributes:
        data (Data): The data used in the likelihood
        config (dict): A dictionary containing configuration parameters of the model
        confounders (dict): A ict of all confounders and group names
        shapes (dict): A dictionary with shape information for building the Likelihood and Prior objects
        likelihood (Likelihood): The likelihood of the model
        prior (Prior): Rhe prior of the model

    """
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.confounders = config['confounding_effects']
        self.n_clusters = config['clusters']
        self.min_size = config['prior']['objects_per_cluster']['min']
        self.max_size = config['prior']['objects_per_cluster']['max']
        self.sample_source = config['sample_source']
        n_sites, n_features, n_states = self.data.features['values'].shape

        self.shapes = {
            'n_sites': n_sites,
            'n_features': n_features,
            'n_states': n_states,
            'states_per_feature': self.data.features['states']
        }

        # Create likelihood and prior objects
        self.likelihood = Likelihood(data=self.data,
                                     shapes=self.shapes)
        self.prior = Prior(shapes=self.shapes,
                           config=self.config['prior'])

    def __call__(self, sample, caching=True):
        """Evaluate the (non-normalized) posterior probability of the given sample."""
        log_likelihood = self.likelihood(sample, caching=caching)
        log_prior = self.prior(sample)
        return log_likelihood + log_prior

    def __copy__(self):
        return Model(self.data, self.config)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        setup_msg = "\n"
        setup_msg += "Model\n"
        setup_msg += "##########################################\n"
        setup_msg += f"Number of clusters: {self.config['clusters']}\n"
        setup_msg += f"Clusters have a minimum size of {self.config['prior']['objects_per_cluster']['min']} " \
                     f"and a maximum size of {self.config['prior']['objects_per_cluster']['max']}\n"
        setup_msg += self.prior.get_setup_message()
        return setup_msg


class Likelihood(object):

    """Likelihood of the sBayes model.

    Attributes:
        features (np.array): The values for all sites and features.
            shape: (n_sites, n_features, n_categories)
        confounders (dict): Assignment of objects to confounders. For each confounder (c) one np.array
            with shape: (n_groups(c), n_sites)
        shapes (dict): A dictionary with shape information for building the Likelihood and Prior objects
        has (dict): Indicates if objects are in a cluster, have a confounder or are affected by a mixture component
        lh_components (dict): A dictionary of cached likelihood values for each site and feature
        weights (np.array): The cached normalized weights of each component in the final likelihood for each feature
            shape: (n_sites, n_features, 3)
        na_features (np.array): A boolean array indicating missing observations
            shape: (n_sites, n_features)
    """

    def __init__(self, data, shapes):

        self.features = data.features['values']
        self.confounders = data.confounders

        self.shapes = shapes
        self.na_features = (np.sum(self.features, axis=-1) == 0)

        # Initialize attributes for caching: assignment of objects to clusters, confounders and mixture components
        self.has = {
            'clusters': None,
            'confounders': dict(),
            'components': None
        }

        # The likelihood components
        self.lh_components = {
            'areal_effect': None,
            'confounding_effects': dict(),
            'all': None
        }

        for k, v in data.confounders.items():
            self.has['confounders'][k] = np.any(v['values'], axis=0)
            self.lh_components['confounding_effects'][k] = None

        # Weights
        self.weights = None

    def reset_cache(self):
        # Assignments
        for k in self.has:
            # Assignment to confounders is fixed
            if k == "confounders":
                pass
            else:
                self.has[k] = None

        # Lh
        for k in self.lh_components:
            if k == "confounding_effects":
                for c in self.lh_components[k]:
                    self.lh_components[k][c] = None
            else:
                self.lh_components[k] = None

        # Weights
        self.weights = None

    def __call__(self, sample, caching=True):
        """Compute the likelihood of all sites. The likelihood is defined as a mixture of areal and confounding effects.
            Args:
                sample(Sample): A Sample object consisting of clusters and weights
            Returns:
                float: The joint likelihood of the current sample
            """

        if not caching:
            self.reset_cache()

        # Compute the likelihood values per mixture component
        all_lh = self.update_component_likelihoods(sample)

        # Compute the weights of the mixture component in each feature and site
        weights = self.update_weights(sample)

        # Compute the total weighted log-likelihood
        log_lh = self.combine_lh(all_lh, weights, sample.source)

        # The step is completed -> everything is up-to-date.
        self.everything_updated(sample)

        return log_lh

    @staticmethod
    def combine_lh(all_lh, weights, source):
        if source is None:
            feature_lh = np.sum(weights * all_lh, axis=2)
            return np.sum(np.log(feature_lh))
        else:
            is_source = np.where(source.ravel())
            observation_weights = weights.ravel()[is_source]
            observation_lhs = all_lh.ravel()[is_source]
            if np.any(observation_weights == 0):
                return -np.inf

            return np.sum(np.log(observation_weights * observation_lhs))

    def everything_updated(self, sample):
        sample.what_changed['lh']['clusters'].clear()
        sample.what_changed['lh']['weights'] = False
        sample.what_changed['lh']['areal_effect'].clear()
        for k, v in self.confounders.items():
            sample.what_changed['lh']['confounding_effects'][k].clear()

    def get_confounding_effects_lh(self, sample, conf):
        """Get the lh for the confounding effect[i]"""
        # The lh for a confounding effect[i] is evaluated when initialized and when the confounding_effect[i] is changed
        if (self.lh_components['confounding_effects'][conf] is None
                or sample.what_changed['lh']['confounding_effects'][conf]):

            # assert np.allclose(a=np.sum(sample.p_families, axis=-1), b=1., rtol=EPS)
            lh = compute_confounding_effects_lh(features=self.features, groups=self.confounders[conf]['values'],
                                                confounding_effect=sample.confounding_effects[conf],
                                                shapes=self.shapes,
                                                outdated_indices=sample.what_changed['lh']['confounding_effects'][conf],
                                                cached_lh=self.lh_components['confounding_effects'][conf])
            self.lh_components['confounding_effects'][conf] = lh
        return self.lh_components['confounding_effects'][conf]

    def get_areal_effect_lh(self, sample):
        """Get the lh for the areal effect"""

        # The lh for the areal effect is evaluated when initialized or when the clusters or the areal_effect changes
        if (self.lh_components['areal_effect'] is None
                or sample.what_changed['lh']['clusters']
                or sample.what_changed['lh']['areal_effect']):

            # assert np.allclose(a=np.sum(p_zones, axis=-1), b=1., rtol=EPS)
            lh = compute_areal_effect_lh(features=self.features, clusters=sample.clusters,
                                         areal_effect=sample.areal_effect, shapes=self.shapes,
                                         outdated_indices=sample.what_changed['lh']['areal_effect'],
                                         outdated_clusters=sample.what_changed['lh']['clusters'],
                                         cached_lh=self.lh_components['areal_effect'])
            self.lh_components['areal_effect'] = lh
        return self.lh_components['areal_effect']

    def update_component_likelihoods(self, sample, caching=True):
        """Update the likelihood values for each of the mixture components"""
        lh = [self.get_areal_effect_lh(sample=sample)]

        for k in self.confounders:
            lh.append(self.get_confounding_effects_lh(sample=sample, conf=k))

        # Merge the component likelihoods into one array (if something has changed)
        if ((not caching) or (self.lh_components['all'] is None)
                or sample.what_changed['lh']['clusters'] or sample.what_changed['lh']['areal_effect']
                or any([v for v in sample.what_changed['lh']['confounding_effects'].values()])):

            self.lh_components['all'] = np.array(lh).transpose((1, 2, 0))
            self.lh_components['all'][self.na_features] = 1.

        return self.lh_components['all']

    def get_cluster_assignment(self, sample):
        """Update the cluster assignment if necessary and return it."""

        if self.has['clusters'] is None or sample.what_changed['lh']['clusters']:
            self.has['clusters'] = np.any(sample.clusters, axis=0)
        return self.has['clusters']

    def update_weights(self, sample):
        """Compute the normalized weights of each component at each site.
        Args:
            sample (Sample): the current MCMC sample.
        Returns:
            np.array: normalized weights of each component at each site.
                shape: (n_sites, n_features, 1 + n_confounders)
        """
        # Assignments are recombined when initialized or when clusters change
        if self.has['components'] is None or sample.what_changed['lh']['clusters']:
            components = [self.get_cluster_assignment(sample)]

            for k in self.confounders:
                components.append(self.has['confounders'][k])

            self.has['components'] = np.array(components).T

        # The weights are evaluated when initialized, or when the weights or the assignment to clusters changes
        if (self.weights is None or sample.what_changed['lh']['weights']
                or sample.what_changed['lh']['clusters']):

            # Extract weights for each site depending on whether the likelihood is available
            self.weights = normalize_weights(weights=sample.weights,
                                             has_components=self.has['components'])

        return self.weights


def compute_areal_effect_lh(features, clusters, shapes, areal_effect=None,
                            outdated_indices=None, outdated_clusters=None, cached_lh=None):
    """Computes the areal effect lh, that is the likelihood per site and feature given clusters z1, ... zn
    Args:
        features(np.array or 'SparseMatrix'): The feature values for all sites and features
            shape: (n_sites, n_features, n_categories)
        clusters(np.array): Binary arrays indicating the assignment of a site to the current clusters
            shape: (n_clusters, n_sites)
        areal_effect(np.array): The areal effect in the current sample
            shape: (n_clusters, n_features, n_sites)
        shapes (dict): A dictionary with shape information for building the Likelihood and Prior objects
        outdated_indices (IndexSet): Set of outdated (cluster, feature) index-pairs
        outdated_clusters (IndexSet): Set of indices, where the cluster changed (=> update across features)
        cached_lh (np.array): The cached set of likelihood values (to be updated where outdated).

    Returns:
        (np.array): the areal effect likelihood per site and feature
            shape: (n_sites, n_features)
    """

    n_clusters = len(clusters)

    if cached_lh is None:
        lh_areal_effect = np.zeros((shapes['n_sites'], shapes['n_features']))
        assert outdated_indices.all
    else:
        lh_areal_effect = cached_lh

    if outdated_indices.all:
        outdated_clusters = range(n_clusters)
    else:
        outdated_clusters = set.union(outdated_clusters,
                                      {i_cluster for (i_cluster, i_feat) in outdated_indices})

    for z in outdated_clusters:
        f_z = features[clusters[z], :, :]
        p_z = areal_effect[z, :, :]
        lh_areal_effect[clusters[z], :] = np.einsum('ijk,jk->ij', f_z, p_z)

    return lh_areal_effect


def compute_confounding_effects_lh(features, groups, shapes, confounding_effect,
                                   outdated_indices=None, cached_lh=None):
    """Computes the confounding effects lh that is the likelihood per site and feature given confounder[i]

    Args:
        features(np.array or 'SparseMatrix'): The feature values for all sites and features
            shape: (n_sites, n_features, n_categories)
        groups(np.array): Binary arrays indicating the assignment of a site to each group of the confounder[i]
            shape: (n_groups, n_sites)
        confounding_effect(np.array): The confounding effect of confounder[i] in the current sample
            shape: (n_clusters, n_features, n_sites)
        shapes (dict): A dictionary with shape information for building the Likelihood and Prior objects
        outdated_indices (IndexSet): Set of outdated (group, feature) index-pairs
        cached_lh (np.array): The cached set of likelihood values (to be updated where outdated).

    Returns:
        (np.array): the family likelihood per site and feature
            shape: (n_sites, n_features)
    """

    n_groups = len(groups)
    if cached_lh is None:
        lh_confounding_effect = np.zeros((shapes['n_sites'], shapes['n_features']))
        assert outdated_indices.all
    else:
        lh_confounding_effect = cached_lh

    if outdated_indices.all:
        outdated_indices = itertools.product(range(n_groups), range(shapes['n_features']))

    for g, i_f in outdated_indices:
        # Compute the feature likelihood vector (for all sites in family)
        f = features[groups[g], i_f, :]
        p = confounding_effect[g, i_f, :]
        lh_confounding_effect[groups[g], i_f] = f.dot(p)

    return lh_confounding_effect


def normalize_weights(weights, has_components):
    """This function assigns each site a weight if it has a likelihood and zero otherwise

    Args:
        weights (np.array): the weights to normalize
            shape: (n_features, 1 + n_confounders)
        has_components (np.array): boolean indicators, showing whether an object is
            affected by an areal or confounding effect
            shape: (n_sites, 1 + n_confounders)

    Return:
        np.array: the weight_per site
            shape: (n_sites, n_features, 1 + n_confounders)
    """

    # Broadcast weights to each site and mask with the has_components arrays
    # Broadcasting:
    #   `weights` doesnt know about sites -> add axis to broadcast to the sites-dimension of `has_component`
    #   `has_components` doesnt know about features -> add axis to broadcast to the features-dimension of `weights`
    weights_per_site = weights[np.newaxis, :, :] * has_components[:, np.newaxis, :]

    # Re-normalize the weights, where weights were masked
    return weights_per_site / weights_per_site.sum(axis=2, keepdims=True)


class Prior(object):
    """The joint prior of all parameters in the sBayes model.

    Attributes:
        size_prior (ClusterSizePrior): prior on the cluster size
        geo_prior (GeoPrior): prior on the geographic spread of a cluster
        prior_weights (WeightsPrior): prior on the mixture weights
        prior_areal_effect (ArealEffectPrior): prior on the areal effect
        prior_confounding_effects (ConfoundingEffectsPrior): prior on all confounding effects
    """

    def __init__(self, shapes, config):
        self.shapes = shapes
        self.config = config

        self.size_prior = ClusterSizePrior(config=self.config['objects_per_cluster'],
                                           shapes=self.shapes)
        self.geo_prior = GeoPrior(config=self.config['geo'])
        self.prior_weights = WeightsPrior(config=self.config['weights'],
                                          shapes=self.shapes)
        self.prior_areal_effect = ArealEffectPrior(config=self.config['areal_effect'],
                                                   shapes=self.shapes)
        self.prior_confounding_effects = dict()
        for k, v in self.config['confounding_effects'].items():
            self.prior_confounding_effects[k] = ConfoundingEffectsPrior(config=v,
                                                                        shapes=self.shapes,
                                                                        conf=k)

    def __call__(self, sample):
        """Compute the prior of the current sample.
        Args:
            sample (Sample): A Sample object consisting of clusters, weights, areal and confounding effects
        Returns:
            float: The (log)prior of the current sample
        """
        log_prior = 0

        # Sum all prior components (in log-space)
        log_prior += self.size_prior(sample)
        log_prior += self.geo_prior(sample)
        log_prior += self.prior_weights(sample)
        log_prior += self.prior_areal_effect(sample)
        for k, v in self.prior_confounding_effects.items():
            log_prior += v(sample)

        self.everything_updated(sample)

        return log_prior

    @staticmethod
    def everything_updated(sample):
        """Mark all parameters in ´sample´ as updated."""
        sample.what_changed['prior']['clusters'].clear()
        sample.what_changed['prior']['weights'] = False
        sample.what_changed['prior']['areal_effect'].clear()
        for k, v in sample.what_changed['prior']['confounding_effects'].items():
            v.clear()

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        setup_msg = self.geo_prior.get_setup_message()
        setup_msg += self.size_prior.get_setup_message()
        setup_msg += self.prior_weights.get_setup_message()
        setup_msg += self.prior_areal_effect.get_setup_message()
        for k, v in self.prior_confounding_effects.items():
            setup_msg += v.get_setup_message()

        return setup_msg

    def __copy__(self):
        return Prior(shapes=self.shapes,
                     config=self.config['prior'])


class DirichletPrior(object):

    class TYPES(Enum):
        UNIFORM = 'uniform'
        DIRICHLET = 'dirichlet'

    def __init__(self, config, shapes, conf=None, initial_counts=1.):

        self.config = config
        self.shapes = shapes
        self.conf = conf

        self.initial_counts = initial_counts
        self.prior_type = None
        self.counts = None
        self.concentration = None
        self.cached = None

        self.parse_attributes()

    # todo: reactivate
    # def load_concentration(self, config: dict) -> List[np.ndarray]:
    #     if 'file' in config:
    #         return self.parse_concentration_json(config['file'])
    #     elif 'parameters' in config:
    #         return self.parse_concentration_dict(config['parameters'])

    # todo: reactivate
    # def parse_concentration_json(self, json_path: str) -> List[np.ndarray]:
    #     # Read the concentration parameters from the JSON file
    #     with open(json_path, 'r') as f:
    #         concentration_dict = json.load(f)
    #
    #     # Parse the resulting dictionary
    #     return self.parse_concentration_dict(concentration_dict)

    # todo: reactivate
    # def parse_concentration_dict(self, concentration_dict: dict) -> List[np.ndarray]:
    #     # Get feature_names and state_names lists to put parameters in the right order
    #     feature_names = self.data.feature_names['external']
    #     state_names = self.data.state_names['external']
    #     assert len(state_names) == len(feature_names) == self.n_features
    #
    #     # Compile the array with concentration parameters
    #     concentration = []
    #     for f, state_names_f in zip(feature_names, state_names):
    #         conc_f = [concentration_dict[f][s] for s in state_names_f]
    #         concentration.append(np.array(conc_f))
    #
    #     return concentration

    def get_uniform_concentration(self) -> List[np.ndarray]:
        concentration = []
        for state_names_f in self.shapes['states_per_feature']:
            concentration.append(
                np.full(shape=len(state_names_f), fill_value=self.initial_counts))
        return concentration

    def parse_attributes(self):
        raise NotImplementedError()

    def is_outdated(self, sample):
        return self.cached is None

    def __call__(self, sample):
        raise NotImplementedError()

    def invalid_prior_message(self, s):
        name = self.__class__.__name__
        valid_types = ','.join([str(t.value) for t in self.TYPES])
        return f'Invalid prior type {s} for {name} (choose from [{valid_types}]).'


class ConfoundingEffectsPrior(DirichletPrior):

    def __init__(self, config, shapes, conf, initial_counts=1.):
        super(ConfoundingEffectsPrior, self).__init__(config, shapes, conf=conf,
                                                      initial_counts=initial_counts)

    def parse_attributes(self):
        n_groups = len(self.config)

        self.concentration = [np.empty(0) for _ in range(n_groups)]
        for i_g, group in enumerate(self.config):
            if self.config[group]['type'] == 'uniform':
                self.concentration[i_g] = self.get_uniform_concentration()
            # todo: reactivate
            # elif config[family]['type'] == 'dirichlet':
            #     self.concentration[i_fam] = self.load_concentration(config[family])
            else:
                raise ValueError(self.invalid_prior_message(self.config['type']))

    def is_outdated(self, sample):
        """Check whether the cached prior for the confounding effect [i] is up-to-date or needs to be recomputed."""
        if self.cached is None:
            return True
        elif sample.what_changed['prior']['confounding_effects'][self.conf]:
            return True
        else:
            return False

    def __call__(self, sample):

        what_changed = sample.what_changed['prior']['confounding_effects'][self.conf]
        if not self.is_outdated(sample):
            return np.sum(self.cached)

        current_concentration = self.concentration

        prior = compute_confounding_effects_prior(confounding_effect=sample.confounding_effects[self.conf],
                                                  concentration=current_concentration,
                                                  applicable_states=self.shapes['states_per_feature'],
                                                  outdated_indices=what_changed,
                                                  cached_prior=self.cached,
                                                  broadcast=False)
        self.cached = prior

        return np.sum(self.cached)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        msg = f"Prior on confounding effect {self.conf}:\n"

        for i_g, group in enumerate(self.config):
            if self.config[group]['type'] == 'uniform':
                msg += f'\tUniform prior for confounder {self.conf} = {group}.\n'
            elif self.config[group]['type'] == 'dirichlet':
                msg += f'\tDirichlet prior for confounder {self.conf} = {group}.\n'
            else:
                raise ValueError(self.invalid_prior_message(self.config['type']))
        return msg


class ArealEffectPrior(DirichletPrior):

    class TYPES(Enum):
        UNIFORM = 'uniform'

    def parse_attributes(self):

        if self.config['type'] == 'uniform':
            self.prior_type = self.TYPES.UNIFORM
            self.counts = np.full(shape=(self.shapes['n_features'], self.shapes['n_states']),
                                  fill_value=self.initial_counts)

        else:
            raise ValueError(self.invalid_prior_message(self.config['type']))

    def is_outdated(self, sample):
        """Check whether the cached prior for the areal effect is up-to-date or needs to be recomputed."""
        if self.cached is None:
            return True
        elif sample.what_changed['prior']['areal_effect']:
            return True
        else:
            return False

    def __call__(self, sample):
        """Compute the prior for the areal effect (or load from cache).
        Args:
            sample (Sample): Current MCMC sample.
        Returns:
            float: Logarithm of the prior probability density.
        """
        if not self.is_outdated(sample):
            return np.sum(self.cached)

        if self.prior_type == self.TYPES.UNIFORM:
            prior_areal_effect = 0.
        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

        self.cached = prior_areal_effect
        return np.sum(self.cached)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f'Prior on areal effect: {self.prior_type.value}\n'


class WeightsPrior(DirichletPrior):

    class TYPES(Enum):
        UNIFORM = 'uniform'

    def parse_attributes(self):
        if self.config['type'] == 'uniform':
            self.prior_type = self.TYPES.UNIFORM
            self.counts = np.full(shape=(self.shapes['n_features'], self.shapes['n_states']),
                                  fill_value=self.initial_counts)

        else:
            raise ValueError(self.invalid_prior_message(self.config['type']))

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


class ClusterSizePrior(object):

    class TYPES(Enum):
        UNIFORM_AREA = 'uniform_area'
        UNIFORM_SIZE = 'uniform_size'
        QUADRATIC_SIZE = 'quadratic'

    def __init__(self, config, shapes, initial_counts=1.):
        self.config = config
        self.shapes = shapes
        self.initial_counts = initial_counts
        self.prior_type = None

        self.cached = None
        self.parse_attributes()

    def invalid_prior_message(self, s):
        valid_types = ','.join([str(t.value) for t in self.TYPES])
        return f'Invalid prior type {s} for size prior (choose from [{valid_types}]).'

    def parse_attributes(self):
        size_prior_type = self.config['type']
        if size_prior_type == 'uniform_area':
            self.prior_type = self.TYPES.UNIFORM_AREA
        elif size_prior_type == 'uniform_size':
            self.prior_type = self.TYPES.UNIFORM_SIZE
        elif size_prior_type == 'quadratic':
            self.prior_type = self.TYPES.QUADRATIC_SIZE
        else:
            raise ValueError(self.invalid_prior_message(size_prior_type))

    def is_outdated(self, sample):
        return self.cached is None or sample.what_changed['prior']['clusters']

    def __call__(self, sample):
        """Compute the prior probability of a set of clusters, based on its number of objects.
        Args:
            sample (Sample): Current MCMC sample.
        Returns:
            float: log-probability of the cluster size.
        """

        if self.is_outdated(sample):
            sizes = np.sum(sample.clusters, axis=-1)

            if self.prior_type == self.TYPES.UNIFORM_SIZE:
                # P(size)   =   uniform
                # P(zone | size)   =   1 / |{zones of size k}|   =   1 / (n choose k)
                logp = -log_multinom(self.shapes['n_sites'], sizes)

            elif self.prior_type == self.TYPES.QUADRATIC_SIZE:
                # Here we assume that only a quadratically growing subset of zones is
                # plausibly permitted by the likelihood and/or geo-prior.
                # P(zone | size) = 1 / |{"plausible" zones of size k}| = 1 / k**2
                log_plausible_clusters = np.log(sizes ** 2)
                logp = -np.sum(log_plausible_clusters)

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

    # @staticmethod
    # def sample(prior_type, n_clusters, n_sites):
    #     if prior_type == ClusterSizePrior.TYPES.UNIFORM_AREA:
    #         onehots = np.eye(n_clusters+1, n_clusters, dtype=bool)
    #         return onehots[np.random.randint(0, n_clusters+1, size=n_sites)].T
    #     else:
    #         raise NotImplementedError()


class GeoPrior(object):

    class TYPES(Enum):
        UNIFORM = 'uniform'
        COST_BASED = 'cost_based'

    def __init__(self, config, initial_counts=1.):
        self.config = config
        self.initial_counts = initial_counts

        self.prior_type = None
        self.cost_matrix = None
        self.scale = None
        self.cached = None

        self.parse_attributes(config)

    def parse_attributes(self, config):
        if config['type'] == 'uniform':
            self.prior_type = self.TYPES.UNIFORM
        # todo: read cost matrix from cost-matrix file
        # elif config['type'] == 'cost_based':
        #     self.prior_type = self.TYPES.COST_BASED
        #
        #     self.cost_matrix = self.data.geo_prior['cost_matrix']
        #     self.scale = config['rate']
        #     self.aggregation = config['aggregation']
        #     self.linkage = config['linkage']
        else:
            raise ValueError('Geo prior not supported')

    def is_outdated(self, sample):
        return self.cached is None or sample.what_changed['prior']['clusters']

    def __call__(self, sample):
        """Compute the geo-prior of the current cluster (or load from cache).
        Args:
            sample (Sample): Current MCMC sample
        Returns:
            float: Logarithm of the prior probability density
        """
        if self.is_outdated(sample):

            if self.prior_type == self.TYPES.UNIFORM:
                geo_prior = 0.

            # elif self.prior_type == self.TYPES.GAUSSIAN:
            #     geo_prior = compute_gaussian_geo_prior(sample.zones, self.data.network,
            #     self.config['gaussian'])

            # todo: implement geo-prior
            # elif self.prior_type == self.TYPES.COST_BASED:
            #     geo_prior = compute_cost_based_geo_prior(
            #         zones=sample.zones,
            #         cost_mat=self.cost_matrix,
            #         scale=self.scale,
            #         aggregation=self.aggregation
            #     )

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
            if 'file' in self.config:
                msg += f'\tCost-matrix file: {self.config["file"]}\n'
        return msg


def compute_gaussian_geo_prior(zones: np.array, network: dict, cov: np.array):
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


def compute_cosed_based_geo_prior(zones: np.array,
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
        float: the geo-prior of the zones
    """

    log_prior = np.ndarray([])
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

        log_prior = stats.expon.logpdf(distances, loc=0, scale=scale)

    if aggregation == 'mean':
        return np.mean(log_prior)
    elif aggregation == 'sum':
        return np.sum(log_prior)
    elif aggregation == 'max':
        return np.min(log_prior)
    else:
        raise ValueError(f'Unknown aggregation policy "{aggregation}" in geo prior.')


# def prior_p_global_dirichlet(p_global, concentration, applicable_states, outdated_features,
#                              cached_prior=None):
#     """" This function evaluates the prior for p_families
#     Args:
#         p_global (np.array): p_global from the sample
#         concentration (list): list of Dirichlet concentration parameters
#         applicable_states (list): list of available categories per feature
#         outdated_features (IndexSet): The features which changed and need to be updated.
#     Kwargs:
#         cached_prior (list):
#
#     Returns:
#         float: the prior for p_global
#     """
#     _, n_feat, n_cat = p_global.shape
#
#     if outdated_features.all:
#         outdated_features = range(n_feat)
#         log_prior = np.zeros(n_feat)
#     else:
#         log_prior = cached_prior
#
#     for f in outdated_features:
#         states_f = applicable_states[f]
#         log_prior[f] = dirichlet_logpdf(
#             x=p_global[0, f, states_f],
#             alpha=concentration[f]
#         )
#
#     return log_prior

def compute_confounding_effects_prior(confounding_effect, concentration, applicable_states,
                                      outdated_indices, cached_prior=None, broadcast=False):
    """" This function evaluates the prior for p_families
    Args:
        confounding_effect (np.array): The confounding effect [i] from the sample
        concentration (list): List of Dirichlet concentration parameters
        applicable_states (list): List of available states per feature
        outdated_indices (IndexSet): The features which need to be updated in each group
        cached_prior (np.array): The cached prior for confound effect [i]
        broadcast (bool): Apply the same prior for all groups (TRUE)?

    Returns:
        float: the prior for confounding effect [i]
    """
    n_groups, n_features, n_states = confounding_effect.shape

    if cached_prior is None:
        assert outdated_indices.all
        log_prior = np.zeros((n_groups, n_features))
    else:
        log_prior = cached_prior

    for group, f in outdated_indices:

        if broadcast:
            # One prior is applied to all groups
            concentration_group = concentration[f]

        else:
            # One prior per group
            concentration_group = concentration[group][f]

        states_f = applicable_states[f]
        conf_group = confounding_effect[group, f, states_f]

        log_prior[group, f] = dirichlet_logpdf(x=conf_group, alpha=conf_group)

    return log_prior


if __name__ == '__main__':
    import doctest
    doctest.testmod()
