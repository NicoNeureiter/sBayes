#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Dict, Sequence, Callable, Optional, OrderedDict
from dataclasses import dataclass
import json

import numpy as np
from numpy.typing import NDArray

import scipy.stats as stats
from scipy.sparse.csgraph import minimum_spanning_tree, csgraph_from_dense

from sbayes.sampling.state import Sample, CalculationNode, ArrayParameter
from sbayes.util import (compute_delaunay, n_smallest_distances, log_multinom,
                         dirichlet_logpdf, log_expit, PathLike)
from sbayes.config.config import ModelConfig, PriorConfig, DirichletPriorConfig, GeoPriorConfig, ClusterSizePriorConfig
from sbayes.load_data import Data, ComputeNetwork, GroupName, ConfounderName, StateName, FeatureName


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
    def __init__(self, data: Data, config: ModelConfig):
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
        self.prior = Prior(shapes=self.shapes, config=self.config.prior, data=data, sample_source=self.sample_source)

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

    def __init__(self, data: Data, shapes: ModelShapes):
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
        observation_lhs = self.get_observation_lhs(component_lhs, weights, sample.source)
        sample.observation_lhs = observation_lhs
        log_lh = np.sum(np.log(observation_lhs))

        return log_lh

    @staticmethod
    def get_observation_lhs(
        all_lh: NDArray,                # shape: (n_objects, n_features, n_components)
        weights: NDArray[float],        # shape: (n_objects, n_features, n_components)
        source: ArrayParameter | None,   # shape: (n_objects, n_features, n_components)
    ) -> NDArray[float]:                # shape: (n_objects, n_features)
        """Combine likelihood from the selected source distributions."""
        if source is None:
            return np.sum(weights * all_lh, axis=2).ravel()
        else:
            is_source = np.where(source.value.ravel())
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

    def __init__(self, shapes: ModelShapes, config: PriorConfig, data: Data, sample_source: bool):
        self.shapes = shapes
        self.config = config
        self.data = data
        self.sample_source = sample_source

        self.size_prior = ClusterSizePrior(config=self.config.objects_per_cluster,
                                           shapes=self.shapes)
        self.geo_prior = GeoPrior(config=self.config.geo,
                                  cost_matrix=data.geo_cost_matrix)
        self.prior_weights = WeightsPrior(config=self.config.weights,
                                          shapes=self.shapes,
                                          feature_names=data.features.feature_and_state_names)
        self.prior_cluster_effect = ClusterEffectPrior(config=self.config.cluster_effect,
                                                       shapes=self.shapes,
                                                       feature_names=data.features.feature_and_state_names)
        self.prior_confounding_effects = {}
        for k, v in self.config.confounding_effects.items():
            self.prior_confounding_effects[k] = ConfoundingEffectsPrior(
                config=v,
                shapes=self.shapes,
                conf=k,
                feature_names=data.features.feature_and_state_names
            )

        if self.sample_source:
            self.source_prior = SourcePrior()
        else:
            self.source_prior = lambda *args, **kwargs: 0.0

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

        if self.sample_source:
            assert sample.source is not None
            log_prior += self.source_prior(sample, caching=caching)
        else:
            assert sample.source is None
            pass

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
        return Prior(
            shapes=self.shapes,
            config=self.config,
            data=self.data,
            sample_source=self.sample_source,
        )


Concentration = List[NDArray[float]]


class DirichletPrior:

    PriorType = DirichletPriorConfig.Types

    config: DirichletPriorConfig | dict[GroupName, DirichletPriorConfig]
    shapes: ModelShapes
    initial_counts: float
    prior_type: Optional[PriorType]
    concentration: Concentration | dict[GroupName, Concentration]

    def __init__(
        self,
        config: DirichletPriorConfig | dict[GroupName, DirichletPriorConfig],
        shapes: ModelShapes,
        feature_names: OrderedDict[FeatureName, list[StateName]],
        initial_counts=1.
    ):
        self.config = config
        self.shapes = shapes
        self.initial_counts = initial_counts
        self.feature_names = feature_names
        self.prior_type = None
        self.concentration = None

        self.parse_attributes()

    @property
    def n_features(self):
        return len(self.feature_names)

    def load_concentration(self, config: DirichletPriorConfig) -> List[np.ndarray]:
        if config.file:
            return self.parse_concentration_json(config.file)
        elif config.parameters:
            return self.parse_concentration_dict(config.parameters)
        else:
            raise ValueError('DirichletPrior requires a file or parameters.')

    def parse_concentration_json(self, json_path: PathLike) -> List[np.ndarray]:
        # Read the concentration parameters from the JSON file
        with open(json_path, 'r') as f:
            concentration_dict = json.load(f)

        # Parse the resulting dictionary
        return self.parse_concentration_dict(concentration_dict)

    def parse_concentration_dict(
        self,
        concentration_dict: dict[FeatureName, dict[StateName, float]]
    ) -> List[np.ndarray]:
        """Compile the array with concentration parameters"""
        concentration = []
        for f, state_names_f in self.feature_names.items():
            conc_f = [
                self.initial_counts + concentration_dict[f][s]
                for s in state_names_f
            ]
            concentration.append(np.array(conc_f))
        return concentration

    def get_uniform_concentration(self) -> List[np.ndarray]:
        concentration = []
        for n_states_f in self.shapes.n_states_per_feature:
            concentration.append(
                np.ones(shape=n_states_f)
            )
        return concentration

    def parse_attributes(self):
        raise NotImplementedError()

    def __call__(self, sample: Sample):
        raise NotImplementedError()

    def invalid_prior_message(self, s):
        name = self.__class__.__name__
        valid_types = ', '.join(self.PriorType)
        return f'Invalid prior type {s} for {name} (choose from [{valid_types}]).'


class ConfoundingEffectsPrior(DirichletPrior):

    conf: ConfounderName

    def __init__(self, *args, conf: ConfounderName, **kwargs):
        super(ConfoundingEffectsPrior, self).__init__(*args, **kwargs)
        self.conf = conf

    def parse_attributes(self):
        self.concentration = {}
        for group in self.config.keys():
            if self.config[group].type is self.PriorType.UNIFORM:
                self.concentration[group] = self.get_uniform_concentration()

            elif self.config[group].type is self.PriorType.DIRICHLET:
                self.concentration[group] = self.load_concentration(self.config[group])

            else:
                raise ValueError(self.invalid_prior_message(self.config[group].type))

    def __call__(self, sample: Sample, caching=True) -> float:
        """"Calculate the log PDF of the confounding effects prior.

        Args:
            sample: Current MCMC sample.

        Returns:
            Logarithm of the prior probability density.
        """

        parameter = sample.confounding_effects[self.conf]
        cache = sample.cache.confounding_effects_prior[self.conf]
        if caching and not cache.is_outdated():
            return cache.value.sum()

        group_names = sample.confounders[self.conf].group_names
        with cache.edit() as cached_priors:
            for i_group in cache.what_changed(f'c_{self.conf}', caching=caching):
                group = group_names[i_group]

                if self.config[group].type is self.PriorType.UNIFORM:
                    cached_priors[i_group] = 0.0
                    continue

                cached_priors[i_group] = compute_group_effect_prior(
                    group_effect=parameter.value[i_group],
                    concentration=self.concentration[group],
                    applicable_states=self.shapes.states_per_feature,
                )

        return cache.value.sum()

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        msg = f"Prior on confounding effect {self.conf}:\n"

        for i_g, group in enumerate(self.config):
            if self.config[group].type is self.PriorType.UNIFORM:
                msg += f'\tUniform prior for confounder {self.conf} = {group}.\n'
            elif self.config[group].type is self.PriorType.DIRICHLET:
                msg += f'\tDirichlet prior for confounder {self.conf} = {group}.\n'
            else:
                raise ValueError(self.invalid_prior_message(self.config.type))
        return msg


class ClusterEffectPrior(DirichletPrior):

    def parse_attributes(self):
        self.prior_type = self.config.type
        if self.prior_type is self.PriorType.UNIFORM:
            self.concentration = self.get_uniform_concentration()
        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

    def __call__(self, sample: Sample, caching=True) -> float:
        """Compute the prior for the areal effect (or load from cache).
        Args:
            sample: Current MCMC sample.
        Returns:
            Logarithm of the prior probability density.
        """
        parameter = sample.cluster_effect
        cache = sample.cache.cluster_effect_prior
        if not cache.is_outdated():
            # return np.sum(cache.value)
            return cache.value

        log_p = 0.0
        if self.prior_type is self.PriorType.UNIFORM:
            pass
        else:
            for i_cluster in range(sample.n_clusters):
                log_p += compute_group_effect_prior(
                    group_effect=parameter.value[i_cluster],
                    concentration=self.concentration,
                    applicable_states=self.shapes.states_per_feature,
                )

        cache.update_value(log_p)
        # return np.sum(cache.value)
        return log_p

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f'Prior on cluster effect: {self.prior_type.value}\n'


class WeightsPrior(DirichletPrior):

    def parse_attributes(self):
        self.prior_type = self.config.type
        if self.prior_type is self.PriorType.UNIFORM:
            self.counts = np.full(shape=(self.shapes['n_features'], self.shapes['n_states']),
                                  fill_value=self.initial_counts)
        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

    def __call__(self, sample: Sample, caching=True) -> float:
        """Compute the prior for weights (or load from cache).
        Args:
            sample: Current MCMC sample.
        Returns:
            Logarithm of the prior probability density.
        """
        # TODO: reactivate caching once we implement more complex priors
        parameter = sample.weights
        cache = sample.cache.weights_prior
        if not cache.is_outdated():
            # return np.sum(cache.value)
            return cache.value

        log_p = 0.0
        if self.prior_type is self.PriorType.UNIFORM:
            pass
        elif self.prior_type is self.PriorType.DIRICHLET:
            log_p = compute_group_effect_prior(
                group_effect=parameter.value,
                concentration=self.concentration,
                applicable_states=np.ones_like(parameter.value),
            )
        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

        cache.update_value(log_p)
        # return np.sum(cache.value)

        return log_p

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f'Prior on weights: {self.prior_type.value}\n'


class SourcePrior(object):

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

        weights = update_weights(sample)
        is_source = np.where(sample.source.value.ravel())
        observation_weights = weights.ravel()[is_source]
        source_prior = np.log(observation_weights).sum()

        cache.update_value(source_prior)
        return source_prior


class ClusterSizePrior:

    PriorType = ClusterSizePriorConfig.Types

    def __init__(self, config: ClusterSizePriorConfig, shapes: ModelShapes, initial_counts=1.):
        self.config = config
        self.shapes = shapes
        self.initial_counts = initial_counts
        self.prior_type = config.type

        self.cached = None

    def invalid_prior_message(self, s):
        valid_types = ','.join(self.PriorType)
        return f'Invalid prior type {s} for size prior (choose from [{valid_types}]).'

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

        if self.prior_type is self.PriorType.UNIFORM_SIZE:
            # P(size)   =   uniform
            # P(zone | size)   =   1 / |{clusters of size k}|   =   1 / (n choose k)
            logp = -log_multinom(self.shapes['n_sites'], sizes)

        elif self.prior_type is self.PriorType.QUADRATIC_SIZE:
            # Here we assume that only a quadratically growing subset of clusters is
            # plausibly permitted by the likelihood and/or geo-prior.
            # P(zone | size) = 1 / |{"plausible" clusters of size k}| = 1 / k**2
            log_plausible_clusters = np.log(sizes ** 2)
            logp = -np.sum(log_plausible_clusters)

        elif self.prior_type is self.PriorType.UNIFORM_AREA:
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
    #     if prior_type is ClusterSizePrior.PriorTypes.UNIFORM_AREA:
    #         onehots = np.eye(n_clusters+1, n_clusters, dtype=bool)
    #         return onehots[np.random.randint(0, n_clusters+1, size=n_sites)].T
    #     else:
    #         raise NotImplementedError()


Aggregator = Callable[[Sequence[float]], float]
"""A type describing functions that aggregate costs in the geo-prior."""


class GeoPrior(object):

    PriorTypes = GeoPriorConfig.Types
    AggrStrats = GeoPriorConfig.AggregationStrategies

    AGGREGATORS: Dict[str, Aggregator] = {
        AggrStrats.MEAN: np.mean,
        AggrStrats.SUM: np.sum,
        AggrStrats.MAX: np.max,
    }

    def __init__(self, config: GeoPriorConfig, cost_matrix=None, initial_counts=1.):
        self.config = config
        self.cost_matrix = cost_matrix
        self.initial_counts = initial_counts
        self.prior_type = config.type

        self.covariance = None
        self.aggregator = None
        self.aggregation_policy = None
        self.probability_function = None
        self.scale = None
        self.cached = None
        self.aggregation = None
        self.linkage = None

        self.parse_attributes(config)

    def parse_attributes(self, config: GeoPriorConfig):
        if self.prior_type is config.Types.COST_BASED:
            if self.cost_matrix is None:
                ValueError('`cost_based` geo-prior requires a cost_matrix.')

            self.prior_type = self.PriorTypes.COST_BASED
            self.scale = config.rate
            self.aggregation_policy = config.aggregation
            self.aggregator = self.AGGREGATORS[self.aggregation_policy]

            self.probability_function = self.parse_prob_function(
                prob_function_type=config.probability_function,
                scale=self.scale,
                inflection_point=config.inflection_point
            )

    @staticmethod
    def parse_prob_function(
            prob_function_type: str,
            scale: float,
            inflection_point: Optional[float] = None
    ) -> Callable[[float], float]:

        if prob_function_type is GeoPriorConfig.probability_function.EXPONENTIAL:
            return lambda x: -x / scale
            # == log(e**(-x/scale))

        elif prob_function_type is GeoPriorConfig.probability_function.SIGMOID:
            assert inflection_point is not None
            x0 = inflection_point
            return lambda x: log_expit(-(x - x0) / scale) - log_expit(x0 / scale)
            # The last term `- log_expit(x0/scale)` scales the sigmoid to be 1 at distance 0

        else:
            raise ValueError(f'Unknown probability_function `{prob_function_type}`')

    def __call__(self, sample: Sample, caching=True) -> float:
        """Compute the geo-prior of the current cluster (or load from cache).
        Args:
            sample: Current MCMC sample
        Returns:
            Logarithm of the prior probability density
        """
        cache = sample.cache.geo_prior
        if caching and not cache.is_outdated():
            return cache.value

        if self.prior_type is self.PriorTypes.UNIFORM:
            geo_prior = 0.
        elif self.prior_type is self.PriorTypes.COST_BASED:
            geo_prior = compute_cost_based_geo_prior(
                clusters=sample.clusters.value,
                cost_mat=self.cost_matrix,
                aggregator=self.aggregator,
                probability_function=self.probability_function,
            )
        elif self.prior_type is self.TYPES.DIAMETER_BASED:
            geo_prior = compute_diameter_based_geo_prior(
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
        valid_types = ','.join(self.PriorTypes)
        return f'Invalid prior type {s} for geo-prior (choose from [{valid_types}]).'

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        msg = f'Geo-prior: {self.prior_type.value}\n'
        if self.prior_type is self.PriorTypes.COST_BASED:
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
        network: ComputeNetwork,
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

    return log_prior.mean()


def compute_diameter_based_geo_prior(
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
        log_prior += probability_function(cost_mat_z.max())

    return log_prior


def compute_cost_based_geo_prior(
    clusters: NDArray[bool],  # shape: (n_clusters, n_objects)
    cost_mat: NDArray,        # shape: (n_objects, n_objects)
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


def compute_group_effect_prior(
        group_effect: NDArray[float],  # shape: (n_features, n_states)
        concentration: List[NDArray],  # shape: (n_applicable_states[f],) for f in features
        applicable_states: List[NDArray],  # shape: (n_applicable_states[f],) for f in features
) -> float:
    """" This function evaluates the prior on probability vectors in a cluster or confounder group.
    Args:
        group_effect: The group effect for a confounder
        concentration: List of Dirichlet concentration parameters.
        applicable_states: List of available states per feature
    Returns:
        The prior log-pdf of the confounding effect for each feature
    """
    n_features, n_states = group_effect.shape

    log_p = 0.0
    for f in range(n_features):
        states_f = applicable_states[f]
        conf_group = group_effect[f, states_f]
        log_p += dirichlet_logpdf(x=conf_group, alpha=concentration[f])

    return log_p


if __name__ == '__main__':
    import doctest
    doctest.testmod()
