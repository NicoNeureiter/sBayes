#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Dict, Sequence, Callable, Optional, Any
from enum import Enum
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import scipy.stats as stats
from scipy.sparse.csgraph import minimum_spanning_tree, csgraph_from_dense

from sbayes.sampling.state import Sample
from sbayes.util import (compute_delaunay, n_smallest_distances, log_multinom,
                         dirichlet_logpdf, log_expit)
from sbayes.config.config import ModelConfig, PriorConfig, DirichletPriorConfig

EPS = np.finfo(float).eps


@dataclass
class ModelShapes:
    n_clusters: int
    n_sites: int
    n_features: int
    n_states: int
    states_per_feature: NDArray[bool]

    @property
    def n_states_per_feature(self):
        return [sum(applicable) for applicable in self.states_per_feature]

    def __getitem__(self, key):
        """Getter for backwards compatibility with dict-notation."""
        return getattr(self, key)


class Model:
    """The sBayes model: posterior distribution of clusters and parameters.

    Attributes:
        data (Data): The data used in the likelihood
        config (ModelConfig): A dictionary containing configuration parameters of the model
        confounders (dict): A ict of all confounders and group names
        shapes (ModelShapes): A dictionary with shape information for building the Likelihood and Prior objects
        likelihood (Likelihood): The likelihood of the model
        prior (Prior): Rhe prior of the model

    """
    def __init__(self, data: 'Data', config: ModelConfig):
        self.data = data
        self.config = config
        self.confounders = config.confounders
        self.n_clusters = config.clusters
        self.min_size = config.prior.objects_per_cluster.min
        self.max_size = config.prior.objects_per_cluster.max
        self.sample_source = config.sample_source
        n_sites, n_features, n_states = self.data.features.values.shape

        self.shapes = ModelShapes(
            n_clusters=self.n_clusters,
            n_sites=n_sites,
            n_features=n_features,
            n_states=n_states,
            states_per_feature=self.data.features.states
        )

        # Create likelihood and prior objects
        self.likelihood = Likelihood(data=self.data, shapes=self.shapes)
        self.prior = Prior(shapes=self.shapes, config=self.config.prior, data=data)

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
        setup_msg += f"Number of clusters: {self.config.clusters}\n"
        setup_msg += f"Clusters have a minimum size of {self.config.prior.objects_per_cluster.min} " \
                     f"and a maximum size of {self.config.prior.objects_per_cluster.max}\n"
        setup_msg += self.prior.get_setup_message()
        return setup_msg


class Likelihood(object):

    """Likelihood of the sBayes model.

    Attributes:
        features (np.array): The values for all sites and features.
            shape: (n_objects, n_features, n_categories)
        confounders (dict): Assignment of objects to confounders. For each confounder (c) one np.array
            with shape: (n_groups(c), n_objects)
        shapes (ModelShapes): A dataclass with shape information for building the Likelihood and Prior objects
        na_features (np.array): A boolean array indicating missing observations
            shape: (n_objects, n_features)
    """

    def __init__(self, data: 'Data', shapes: ModelShapes):
        self.features = data.features.values
        self.confounders = data.confounders
        self.shapes = shapes
        self.na_features = (np.sum(self.features, axis=-1) == 0)

    def __call__(self, sample, caching=True):
        """Compute the likelihood of all sites. The likelihood is defined as a mixture of areal and confounding effects.
            Args:
                sample(Sample): A Sample object consisting of clusters and weights
            Returns:
                float: The joint likelihood of the current sample
            """
        if not caching:
            sample.cache.clear()

        # Compute the likelihood values per mixture component
        component_lhs = self.update_component_likelihoods(sample, caching=caching)

        # Compute the weights of the mixture component in each feature and site
        weights = update_weights(sample, caching=caching)

        # Compute the total log-likelihood
        observation_lhs = self.get_observation_lhs(component_lhs, weights, sample.source.value)
        sample.observation_lhs = observation_lhs
        log_lh = np.sum(np.log(observation_lhs))

        # Add the probability of observing the sources (if sampled)
        if sample.source.value is not None:
            is_source = np.where(sample.source.value.ravel())
            p_source = weights.ravel()[is_source]
            log_lh += np.sum(np.log(p_source))

        assert not np.isnan(log_lh)

        return log_lh

    @staticmethod
    def get_observation_lhs(
        all_lh: NDArray,                # shape: (n_objects, n_features, n_components)
        weights: NDArray[float],        # shape: (n_objects, n_features, n_components)
        source: NDArray[bool] | None,   # shape: (n_objects, n_features, n_components)
    ) -> NDArray[float]:                # shape: (n_objects, n_features)
        """Combine likelihood from the selected source distributions."""
        if source is None:
            return np.sum(weights * all_lh, axis=2).ravel()
        else:
            is_source = np.where(source.ravel())
            return all_lh.ravel()[is_source]

    def update_component_likelihoods(self, sample: Sample, caching=True) -> NDArray[float]:
        """Update the likelihood values for each of the mixture components"""
        CHECK_CACHING = False

        cache = sample.cache.component_likelihoods
        if caching and not cache.is_outdated():
            x = sample.cache.component_likelihoods.value
            if CHECK_CACHING:
                assert np.all(x == self.update_component_likelihoods(sample, caching=False))
            return x

        with cache.edit() as component_likelihood:
            # TODO: Not sure whether a context manager is the best way to do this. Discuss!
            # Update component likelihood for cluster effects:
            compute_component_likelihood(
                features=self.features,
                probs=sample.cluster_effect.value,
                groups=sample.clusters.value,
                changed_groups=cache.what_changed(['cluster_effect', 'clusters'], caching),
                out=component_likelihood[..., 0],
            )
            if caching and CHECK_CACHING:
                x = component_likelihood[..., 0].copy()
                y = compute_component_likelihood(
                    features=self.features,
                    probs=sample.cluster_effect.value,
                    groups=sample.clusters.value,
                    changed_groups=cache.what_changed(['cluster_effect', 'clusters'], caching=False),
                    out=component_likelihood[..., 0],
                )
                assert np.all(x[~self.na_features] == y[~self.na_features])

            # Update component likelihood for confounding effects:
            for i, conf in enumerate(self.confounders, start=1):
                compute_component_likelihood(
                    features=self.features,
                    probs=sample.confounding_effects[conf].value,
                    groups=sample.confounders[conf].group_assignment,
                    changed_groups=cache.what_changed(f'c_{conf}', caching),
                    out=component_likelihood[..., i],
                )
                if caching and CHECK_CACHING:
                    x = component_likelihood[..., i].copy()
                    y: object = compute_component_likelihood(
                        features=self.features,
                        probs=sample.confounding_effects[conf].value,
                        groups=sample.confounders[conf].group_assignment,
                        changed_groups=cache.what_changed(f'c_{conf}', caching=False),
                        out=component_likelihood[..., i],
                    )
                    assert np.all(x[~self.na_features] == y[~self.na_features])

            component_likelihood[self.na_features] = 1.

        # cache.set_up_to_date()

        return cache.value


def compute_component_likelihood(
        features: NDArray[bool],
        probs: NDArray[float],
        groups: NDArray[bool],  # (n_groups, n_sites)
        changed_groups: set[int],
        out: NDArray[float]
) -> NDArray[float]:  # shape: (n_sites, n_features)
    out[~groups.any(axis=0), :] = 0.
    for i in changed_groups:
        g = groups[i]
        f_g = features[g, :, :]
        p_g = probs[i, :, :]
        out[g, :] = np.einsum('ijk,jk->ij', f_g, p_g)
    return out


# def compute_cluster_effect_lh(features, clusters, shapes, cluster_effect,
#                               outdated_indices=None, outdated_clusters=None, cached_lh=None):
#     """Computes the areal effect lh, that is the likelihood per site and feature given clusters z1, ... zn
#     Args:
#         features(np.array or 'SparseMatrix'): The feature values for all sites and features
#             shape: (n_sites, n_features, n_categories)
#         clusters(np.array): Binary arrays indicating the assignment of a site to the current clusters
#             shape: (n_clusters, n_sites)
#         cluster_effect(np.array): The areal effect in the current sample
#             shape: (n_clusters, n_features, n_sites)
#         shapes (ModelShapes): A dictionary with shape information for building the Likelihood and Prior objects
#         outdated_indices (IndexSet): Set of outdated (cluster, feature) index-pairs
#         outdated_clusters (IndexSet): Set of indices, where the cluster changed (=> update across features)
#         cached_lh (np.array): The cached set of likelihood values (to be updated where outdated).
#
#     Returns:
#         (np.array): the areal effect likelihood per site and feature
#             shape: (n_sites, n_features)
#     """
#
#     n_clusters = len(clusters)
#
#     if cached_lh is None:
#         lh_cluster_effect = np.zeros((shapes.n_sites, shapes.n_features))
#         assert outdated_indices.all
#     else:
#         lh_cluster_effect = cached_lh
#
#     if (outdated_indices is None) or outdated_indices.all:
#         outdated_clusters = range(n_clusters)
#     else:
#         outdated_clusters = set.union(outdated_clusters,
#                                       {i_cluster for (i_cluster, i_feat) in outdated_indices})
#
#     for z in outdated_clusters:
#         f_z = features[clusters[z], :, :]
#         p_z = cluster_effect[z, :, :]
#         lh_cluster_effect[clusters[z], :] = np.einsum('ijk,jk->ij', f_z, p_z)
#
#     return lh_cluster_effect
#
#
# def compute_confounding_effects_lh(
#         features: NDArray[bool],
#         groups: NDArray[bool],
#         shapes: ModelShapes,
#         confounding_effect: NDArray[float],
#         outdated_indices: IndexSet = None,
#         cached_lh: NDArray[float] = None,
# ) -> NDArray[float]:
#     """Computes the confounding effects lh that is the likelihood per site and feature given confounder[i]
#
#     Args:
#         features: The feature values for all sites and features
#             shape: (n_sites, n_features, n_categories)
#         groups: Binary arrays indicating the assignment of a site to each group of the confounder[i]
#             shape: (n_groups, n_sites)
#         confounding_effect: The confounding effect of confounder[i] in the current sample
#             shape: (n_clusters, n_features, n_sites)
#         shapes: A dictionary with shape information for building the Likelihood and Prior objects
#         outdated_indices: Set of outdated (group, feature) index-pairs
#         cached_lh: The cached set of likelihood values (to be updated where outdated).
#
#     Returns:
#         The family likelihood per site and feature
#             shape: (n_sites, n_features)
#     """
#
#     n_groups = len(groups)
#     if cached_lh is None:
#         lh_confounding_effect = np.zeros((shapes.n_sites, shapes.n_features))
#         assert outdated_indices.all
#     else:
#         lh_confounding_effect = cached_lh
#
#     # print(outdated_indices.all, outdated_indices)
#
#     if outdated_indices.all:
#         outdated_indices = itertools.product(range(n_groups), range(shapes.n_features))
#
#     for g, i_f in outdated_indices:
#         # Compute the feature likelihood vector (for all sites in family)
#         f = features[groups[g], i_f, :]
#         p = confounding_effect[g, i_f, :]
#         lh_confounding_effect[groups[g], i_f] = f.dot(p)
#
#     return lh_confounding_effect


def update_weights(sample: Sample, caching=True) -> NDArray[float]:
    """Compute the normalized weights of each component at each site.
    Args:
        sample: the current MCMC sample.
        caching: ignore cache if set to false.
    Returns:
        np.array: normalized weights of each component at each site.
            shape: (n_objects, n_features, 1 + n_confounders)
    """
    cache = sample.cache.weights_normalized

    if (not caching) or cache.is_outdated():
        w_normed = normalize_weights(sample.weights.value, sample.cache.has_components.value)
        cache.update_value(w_normed)

    return cache.value


def normalize_weights(
    weights: NDArray[float],  # shape: (n_features, 1 + n_confounders)
    has_components: NDArray[bool]  # shape: (n_objects, 1 + n_confounders)
) -> NDArray[float]:  # shape: (n_objects, n_features, 1 + n_confounders)
    """This function assigns each site a weight if it has a likelihood and zero otherwise
    Args:
        weights: the weights to normalize
        has_components: indicators for which objects are affected by cluster and confounding effects
    Return:
        the weight_per site
    """
    # Broadcast weights to each site and mask with the has_components arrays
    # Broadcasting:
    #   `weights` doesnt know about sites -> add axis to broadcast to the sites-dimension of `has_component`
    #   `has_components` doesnt know about features -> add axis to broadcast to the features-dimension of `weights`
    weights_per_site = weights[np.newaxis, :, :] * has_components[:, np.newaxis, :]

    # Re-normalize the weights, where weights were masked
    return weights_per_site / weights_per_site.sum(axis=2, keepdims=True)


class Prior:
    """The joint prior of all parameters in the sBayes model.

    Attributes:
        size_prior (ClusterSizePrior): prior on the cluster size
        geo_prior (GeoPrior): prior on the geographic spread of a cluster
        prior_weights (WeightsPrior): prior on the mixture weights
        prior_cluster_effect (ClusterEffectPrior): prior on the areal effect
        prior_confounding_effects (ConfoundingEffectsPrior): prior on all confounding effects
    """

    def __init__(self, shapes: ModelShapes, config: PriorConfig, data: 'Data'):
        self.shapes = shapes
        self.config = config
        self.data = data

        self.size_prior = ClusterSizePrior(config=self.config.objects_per_cluster,
                                           shapes=self.shapes)
        self.geo_prior = GeoPrior(config=self.config.geo,
                                  cost_matrix=data.geo_cost_matrix)
        self.prior_weights = WeightsPrior(config=self.config.weights,
                                          shapes=self.shapes)
        self.prior_cluster_effect = ClusterEffectPrior(config=self.config.cluster_effect,
                                                       shapes=self.shapes)
        self.prior_confounding_effects = {
            k: ConfoundingEffectsPrior(config=v, shapes=self.shapes, conf=k)
            for k, v in self.config.confounding_effects.items()
        }

    def __call__(self, sample: Sample, caching=True) -> float:
        """Compute the prior of the current sample.
        Args:
            sample: A Sample object consisting of clusters, weights, areal and confounding effects
        Returns:
            The (log)prior of the current sample
        """
        log_prior = 0

        # Sum all prior components (in log-space)
        log_prior += self.size_prior(sample, caching=caching)
        log_prior += self.geo_prior(sample, caching=caching)
        log_prior += self.prior_weights(sample, caching=caching)
        log_prior += self.prior_cluster_effect(sample, caching=caching)
        for k, v in self.prior_confounding_effects.items():
            log_prior += v(sample, caching=caching)

        # CHECK_CACHING = True
        # if CHECK_CACHING:
        #     assert self.size_prior(sample, caching=True) == self.size_prior(sample, caching=False)
        #     assert self.geo_prior(sample, caching=True) == self.geo_prior(sample, caching=False)
        #     assert self.prior_weights(sample, caching=True) == self.prior_weights(sample, caching=False)
        #     assert self.prior_cluster_effect(sample, caching=True) == self.prior_cluster_effect(sample, caching=False)
        #     for k, v in self.prior_confounding_effects.items():
        #         assert v(sample, caching=True) == v(sample, caching=False)

        return log_prior

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        setup_msg = self.geo_prior.get_setup_message()
        setup_msg += self.size_prior.get_setup_message()
        setup_msg += self.prior_weights.get_setup_message()
        setup_msg += self.prior_cluster_effect.get_setup_message()
        for k, v in self.prior_confounding_effects.items():
            setup_msg += v.get_setup_message()

        return setup_msg

    def __copy__(self):
        return Prior(shapes=self.shapes, config=self.config, data=self.data)


class DirichletPrior:

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
        self.concentration: list[NDArray] = None

        self.parse_attributes()

    # todo: reactivate
    # def load_concentration(self, config: DirichletPriorConfig) -> List[np.ndarray]:
    #     if config.file:
    #         return self.parse_concentration_json(config['file'])
    #     elif 'parameters' in config:
    #         return self.parse_concentration_dict(config['parameters'])
    #
    # def parse_concentration_json(self, json_path: str) -> List[np.ndarray]:
    #     # Read the concentration parameters from the JSON file
    #     with open(json_path, 'r') as f:
    #         concentration_dict = json.load(f)
    #
    #     # Parse the resulting dictionary
    #     return self.parse_concentration_dict(concentration_dict)
    #
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
        for state_names_f in self.shapes.n_states_per_feature:
            concentration.append(
                np.ones(shape=np.sum(state_names_f))
            )
        return concentration

    def parse_attributes(self):
        raise NotImplementedError()

    def __call__(self, sample: Sample):
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

            # todo reactivate
            # elif self.config[group]['type'] == 'dirichlet':
            #     self.concentration[i_g] = self.load_concentration(self.config[group])

            else:
                raise ValueError(self.invalid_prior_message(self.config['type']))

    def __call__(self, sample: Sample, caching=True) -> float:
        """"Calculate the log PDF of the confounding effects prior.

        Args:
            sample: Current MCMC sample.

        Returns:
            float: Logarithm of the prior probability density.
        """
        parameter = sample.confounding_effects[self.conf]
        cache = sample.cache.confounding_effects_prior[self.conf]
        if caching and not cache.is_outdated():
            return np.sum(cache.value)

        prior = compute_confounding_effects_prior(
            confounding_effect=parameter.value,
            concentration=self.concentration,
            applicable_states=self.shapes.states_per_feature,
            outdated_groups=cache.what_changed(f'c_{self.conf}', caching=caching),
            cached_prior=cache.value,
            broadcast=False
        )

        # assert np.all(
        #     prior == compute_confounding_effects_prior(
        #         confounding_effect=parameter.value,
        #         concentration=self.concentration,
        #         applicable_states=self.shapes.states_per_feature,
        #         outdated_groups=cache.what_changed(f'c_{self.conf}', caching=False),
        #         cached_prior=cache.value,
        #         broadcast=False
        #     )
        # )

        cache.update_value(prior)
        return np.sum(cache.value)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        msg = f"Prior on confounding effect {self.conf}:\n"

        for i_g, group in enumerate(self.config):
            if self.config[group]['type'] == 'uniform':
                msg += f'\tUniform prior for confounder {self.conf} = {group}.\n'
            elif self.config[group]['type'] == 'dirichlet':
                msg += f'\tDirichlet prior for confounder {self.conf} = {group}.\n'
            else:
                raise ValueError(self.invalid_prior_message(self.config.type))
        return msg


class ClusterEffectPrior(DirichletPrior):

    class TYPES(Enum):
        UNIFORM = 'uniform'

    def parse_attributes(self):

        if self.config.type == 'uniform':
            self.prior_type = self.TYPES.UNIFORM
            self.counts = np.full(shape=(self.shapes['n_features'], self.shapes['n_states']),
                                  fill_value=self.initial_counts)

        else:
            raise ValueError(self.invalid_prior_message(self.config.type))

    def __call__(self, sample: Sample, caching=True) -> float:
        """Compute the prior for the areal effect (or load from cache).
        Args:
            sample: Current MCMC sample.
        Returns:
            Logarithm of the prior probability density.
        """
        # TODO: reactivate caching once we implement more complex priors
        # parameter = sample.clusters
        # cache = sample.cache.cluster_effect_prior
        # if not cache.is_outdated():
        #     return np.sum(cache.value)

        if self.prior_type == self.TYPES.UNIFORM:
            prior_cluster_effect = 0.
        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

        # cache.update_value(prior_cluster_effect)
        # return np.sum(cache.value)

        return prior_cluster_effect

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f'Prior on cluster effect: {self.prior_type.value}\n'


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

    def __call__(self, sample: Sample, caching=True) -> float:
        """Compute the prior for weights (or load from cache).
        Args:
            sample: Current MCMC sample.
        Returns:
            Logarithm of the prior probability density.
        """
        # TODO: reactivate caching once we implement more complex priors
        # parameter = sample.weights
        # cache = sample.cache.weights_prior
        # if not cache.is_outdated():
        #     return np.sum(cache.value)

        if self.prior_type == self.TYPES.UNIFORM:
            prior_weights = 0.
        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

        # cache.update_value(prior_weights)
        # return np.sum(cache.value)
        return prior_weights

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f'Prior on weights: {self.prior_type.value}\n'


class SourcePrior(object):

    def __init__(self, config):
        self.config = config

    def __call__(self, sample: Sample, caching=True) -> float:
        """Compute the prior for weights (or load from cache).
        Args:
            sample: Current MCMC sample.
        Returns:
            Logarithm of the prior probability density.
        """
        cache = sample.cache.source_prior
        if caching and not cache.is_outdated():
            return cache.value

        weights = update_weights(sample, caching=caching)
        is_source = np.where(sample.source.value.ravel())
        observation_weights = weights.ravel()[is_source]
        source_prior = np.sum(np.log(observation_weights))

        cache.update_value(source_prior)
        return source_prior


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

    def __call__(self, sample: Sample, caching=True) -> float:
        """Compute the prior probability of a set of clusters, based on its number of objects.
        Args:
            sample: Current MCMC sample.
        Returns:
            Log-probability of the cluster size.
        """
        cache = sample.cache.cluster_size_prior
        if caching and not cache.is_outdated():
            return cache.value

        sizes = np.sum(sample.clusters.value, axis=-1)

        if self.prior_type == self.TYPES.UNIFORM_SIZE:
            # P(size)   =   uniform
            # P(zone | size)   =   1 / |{clusters of size k}|   =   1 / (n choose k)
            logp = -log_multinom(self.shapes['n_sites'], sizes)

        elif self.prior_type == self.TYPES.QUADRATIC_SIZE:
            # Here we assume that only a quadratically growing subset of clusters is
            # plausibly permitted by the likelihood and/or geo-prior.
            # P(zone | size) = 1 / |{"plausible" clusters of size k}| = 1 / k**2
            log_plausible_clusters = np.log(sizes ** 2)
            logp = -np.sum(log_plausible_clusters)

        elif self.prior_type == self.TYPES.UNIFORM_AREA:
            # No size prior
            # P(zone | size) = P(zone) = const.
            logp = 0.
        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

        cache.update_value(logp)
        return logp

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f'Prior on cluster size: {self.prior_type.value}\n'

    # @staticmethod
    # def sample(prior_type, n_clusters, n_sites):
    #     if prior_type == ClusterSizePrior.TYPES.UNIFORM_AREA:
    #         onehots = np.eye(n_clusters+1, n_clusters, dtype=bool)
    #         return onehots[np.random.randint(0, n_clusters+1, size=n_sites)].T
    #     else:
    #         raise NotImplementedError()


Aggregator = Callable[[Sequence[float]], float]
"""A type describing functions that aggregate costs in the geo-prior."""


class GeoPrior(object):

    class TYPES(Enum):
        UNIFORM = 'uniform'
        COST_BASED = 'cost_based'

    AGGREGATORS: Dict[str, Aggregator] = {
        'mean': np.mean,
        'sum': np.sum,
        'max': np.max,
    }

    def __init__(self, config, cost_matrix=None, initial_counts=1.):
        self.config = config
        self.cost_matrix = cost_matrix
        self.initial_counts = initial_counts

        self.prior_type = None
        self.covariance = None
        self.cost_matrix = None
        self.aggregator = None
        self.aggregation_policy = None
        self.probability_function = None
        self.scale = None
        self.cached = None
        self.aggregation = None
        self.linkage = None

        self.parse_attributes(config)

    def parse_attributes(self, config):
        if config['type'] == 'uniform':
            self.prior_type = self.TYPES.UNIFORM

        elif config['type'] == 'cost_based':
            if self.cost_matrix is None:
                ValueError('`cost_based` geo-prior requires a cost_matrix.')

            self.prior_type = self.TYPES.COST_BASED
            self.cost_matrix = self.cost_matrix
            self.scale = config['rate']
            self.aggregation_policy = config['aggregation']
            assert self.aggregation_policy in ['mean', 'sum', 'max']
            self.aggregator = self.AGGREGATORS.get(self.aggregation_policy)
            if self.aggregator is None:
                raise ValueError(f'Unknown aggregation policy "{self.aggregation_policy}" in geo prior.')

            assert config['probability_function'] in ['exponential', 'sigmoid']
            self.probability_function = self.parse_prob_function(
                prob_function_type=config['probability_function'],
                scale=self.scale,
                inflection_point=config.get('inflection_point', None)
            )
        else:
            raise ValueError('Geo prior not supported')

    @staticmethod
    def parse_prob_function(
            prob_function_type: str,
            scale: float,
            inflection_point: Optional[float] = None
    ) -> callable:

        if prob_function_type == 'exponential':
            return lambda x: -x / scale
            # == log(e**(-x/scale))

        elif prob_function_type == 'sigmoid':
            assert inflection_point is not None

            x0 = inflection_point
            return lambda x: log_expit(-(x - x0) / scale) - log_expit(x0 / scale)
            # The last term `- log_expit(x0/scale)` scales the sigmoid to be 1 at distance 0

        else:
            raise ValueError(f'Unknown probability_function `{prob_function_type}`')

    def __call__(self, sample: Sample, caching=True):
        """Compute the geo-prior of the current cluster (or load from cache).
        Args:
            sample (Sample): Current MCMC sample
        Returns:
            float: Logarithm of the prior probability density
        """
        cache = sample.cache.geo_prior
        if caching and not cache.is_outdated():
            return cache.value

        if self.prior_type is self.TYPES.UNIFORM:
            geo_prior = 0.
        elif self.prior_type is self.TYPES.COST_BASED:
            geo_prior = compute_cost_based_geo_prior(
                clusters=sample.clusters.value,
                cost_mat=self.cost_matrix,
                aggregator=self.aggregator,
                probability_function=self.probability_function,
            )
        else:
            raise ValueError('geo_prior must be either \"uniform\" or \"cost_based\".')

        cache.update_value(geo_prior)
        return geo_prior

    def invalid_prior_message(self, s):
        valid_types = ','.join([str(t.value) for t in self.TYPES])
        return f'Invalid prior type {s} for geo-prior (choose from [{valid_types}]).'

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        msg = f'Geo-prior: {self.prior_type.value}\n'
        if self.prior_type == self.TYPES.COST_BASED:
            prob_fun = self.config["probability_function"]
            msg += f'\tProbability function: {prob_fun}\n'
            msg += f'\tAggregation policy: {self.aggregation_policy}\n'
            msg += f'\tScale: {self.scale}\n'
            if self.config['probability_function'] == 'sigmoid':
                msg += f'\tInflection point: {self.config["inflection_point"]}\n'
            if self.config['costs'] == 'from_data':
                msg += '\tCost-matrix inferred from geo-locations.\n'
            else:
                msg += f'\tCost-matrix file: {self.config["costs"]}\n'

        return msg


def compute_gaussian_geo_prior(
        cluster: np.array,
        network: 'ComputeNetwork',
        cov: np.array,
) -> float:
    """This function computes the 2D Gaussian geo-prior for all edges in the cluster.

    Args:
        cluster: boolean array representing the current zone
        network: network containing the graph, location,...
        cov: Covariance matrix of the multivariate gaussian (estimated from the data)

    Returns:
        float: the log geo-prior of the clusters
    """
    log_prior = np.ndarray([])
    for z in cluster:
        dist_mat = network.dist_mat[z][:, z]
        locations = network.locations[z]

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


def compute_cost_based_geo_prior(
        clusters: np.array,
        cost_mat: np.array,
        aggregator: Aggregator,
        probability_function: Callable[[float], float],
) -> float:
    """ This function computes the geo prior for the sum of all distances of the mst of a zone
    Args:
        clusters: The current cluster (boolean array)
        cost_mat: The cost matrix between locations
        aggregator: The aggregation policy, defining how the single edge
            costs are combined into one joint cost for the area.
        probability_function: Function mapping aggregate distances to log-probabilities

    Returns:
        float: the log geo-prior of the cluster
    """
    log_prior = 0.0
    for z in clusters:
        cost_mat_z = cost_mat[z][:, z]

        if cost_mat_z.shape[0] > 1:
            graph = csgraph_from_dense(cost_mat_z, null_value=np.inf)
            mst = minimum_spanning_tree(graph)

            # When there are zero costs between languages the MST might be 0
            if mst.nnz > 0:
                distances = mst.tocsr()[mst.nonzero()]
            else:
                distances = [0.0]
        else:
            raise ValueError("Too few locations to compute distance.")

        agg_distance = aggregator(distances)
        log_prior += probability_function(agg_distance)

    return log_prior


def compute_confounding_effects_prior(
        confounding_effect: NDArray[float],  # shape: (n_groups, n_features, n_states)
        concentration: List[NDArray],  # shape: (n_applicable_states[f],) for f in features
        applicable_states: List[NDArray],  # shape: (n_applicable_states[f],) for f in features
        outdated_groups: set[int],
        cached_prior: NDArray[float] = None,  # shape: (n_groups)
        broadcast: bool = False
) -> NDArray[float]:  # shape: (n_groups, n_features)
    """" This function evaluates the prior for p_families
    Args:
        confounding_effect: The confounding effect [i] from the sample
        concentration: List of Dirichlet concentration parameters.
        applicable_states: List of available states per feature
        outdated_groups: The features which need to be updated in each group
        cached_prior: The cached prior for confound effect [i]
        broadcast: Apply the same prior for all groups (TRUE)?
    Returns:
        The prior log-pdf of the confounding effect for each group and feature
    """
    n_groups, n_features, n_states = confounding_effect.shape

    for group in outdated_groups:
        cached_prior[group] = 0.0
        for f in range(n_features):
            if broadcast:
                # One prior is applied to all groups
                concentration_group = concentration[f]

            else:
                # One prior per group
                concentration_group = concentration[group][f]

            states_f = applicable_states[f]
            conf_group = confounding_effect[group, f, states_f]

            cached_prior[group] += dirichlet_logpdf(x=conf_group, alpha=concentration_group)

    # for group, f in outdated_indices:
    #     if broadcast:
    #         # One prior is applied to all groups
    #         concentration_group = concentration[f]
    #
    #     else:
    #         # One prior per group
    #         concentration_group = concentration[group][f]
    #
    #     states_f = applicable_states[f]
    #     conf_group = confounding_effect[group, f, states_f]
    #
    #     log_prior[group, f] = dirichlet_logpdf(x=conf_group, alpha=concentration_group)
    return cached_prior


if __name__ == '__main__':
    import doctest
    doctest.testmod()
