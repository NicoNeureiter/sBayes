#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Sequence, Callable, Optional, OrderedDict
import json

import numpy as np
from numpy.typing import NDArray

import scipy.stats as stats
from scipy.sparse.csgraph import minimum_spanning_tree, csgraph_from_dense

from sbayes.model.likelihood import update_weights, ModelShapes, normalize_weights
from sbayes.preprocessing import sample_categorical
from sbayes.sampling.state import Sample
from sbayes.util import (compute_delaunay, n_smallest_distances, log_multinom,
                         dirichlet_logpdf, log_expit, PathLike, log_binom, normalize)
from sbayes.config.config import PriorConfig, DirichletPriorConfig, GeoPriorConfig, ClusterSizePriorConfig, \
    ConfoundingEffectPriorConfig
from sbayes.load_data import Data, ComputeNetwork, GroupName, ConfounderName, StateName, FeatureName, Confounder


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
                                  cost_matrix=data.geo_cost_matrix,
                                  network=data.network)
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
                feature_names=data.features.feature_and_state_names,
                group_names=data.confounders[k].group_names,
                conf_effect_priors = self.prior_confounding_effects,
            )

        self.source_prior = SourcePrior(na_features=self.data.features.na_values)

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

        # AFTER THE DIRI-MULT MODEL CHANGES, THE FOLLOWING PRIORS ARE NOT NEEDED:
        # log_prior += self.prior_cluster_effect(sample, caching=caching)
        # for k, v in self.prior_confounding_effects.items():
        #     log_prior += v(sample, caching=caching)

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

    def generate_sample(self) -> Sample:
        """Generate a sample from the prior distribution."""
        confounders = self.data.confounders

        # Generate samples from the independent prior distributions
        weights = self.prior_weights.generate_sample()
        clusters = self.size_prior.generate_sample()

        # Sample the source array, conditioned on the weights and clusters
        has_components = compute_has_components(clusters, confounders)
        source = self.source_prior.generate_sample(weights, has_components)

        feature_counts_cluster = np.zeros((self.shapes.n_clusters, self.shapes.n_features, self.shapes.n_states))
        feature_counts_by_confounder = {}
        for conf, n_groups in self.shapes.n_groups.items():
            feature_counts_by_confounder[conf] = np.zeros((n_groups, self.shapes.n_features, self.shapes.n_states))

        # Create a sample object from the generated numpy arrays
        return Sample.from_numpy_arrays(
            clusters=clusters,
            weights=weights,
            confounding_effects={conf: np.empty(n_groups) for conf, n_groups in self.shapes.n_groups.items()},
            confounders=confounders,
            feature_counts={'clusters': feature_counts_cluster, **feature_counts_by_confounder},
            source=source
        )

    def generate_samples(self, n_samples: int) -> list[Sample]:
        """Generate `n_samples` samples from the prior distribution"""
        return [self.generate_sample() for _ in range(n_samples)]


def compute_has_components(clusters: NDArray[bool], confounders: dict[str, Confounder]):
    n_components = len(confounders) + 1
    n_objects = clusters.shape[1]

    has_components = np.empty((n_objects, n_components))
    has_components[:, 0] = np.any(clusters, axis=0)
    for i, conf in enumerate(confounders.values(), start=1):
        has_components[:, i] = conf.any_group()

    return np.array(has_components)


Concentration = List[NDArray[float]]


class DirichletPrior:

    PriorType = DirichletPriorConfig.Types

    config: DirichletPriorConfig | dict[GroupName, DirichletPriorConfig]
    shapes: ModelShapes
    initial_counts: float
    prior_type: Optional[PriorType]
    concentration: Concentration | dict[GroupName, Concentration]
    concentration_array: NDArray[float]

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

    def load_concentration(self, config: DirichletPriorConfig) -> list[np.ndarray]:
        if config.file:
            return self.parse_concentration_json(config.file)
        elif config.parameters:
            return self.parse_concentration_dict(config.parameters)
        else:
            raise ValueError('DirichletPrior requires a file or parameters.')

    def parse_concentration_json(self, json_path: PathLike) -> list[np.ndarray]:
        # Read the concentration parameters from the JSON file
        with open(json_path, 'r') as f:
            concentration_dict = json.load(f)
        # Parse the resulting dictionary
        return self.parse_concentration_dict(concentration_dict)

    def parse_concentration_dict(
        self,
        concentration_dict: dict[FeatureName, dict[StateName, float]]
    ) -> list[np.ndarray]:
        """Compile the array with concentration parameters"""
        concentration = []
        for f, state_names_f in self.feature_names.items():
            conc_f = [
                self.initial_counts + concentration_dict[f][s]
                for s in state_names_f
            ]
            concentration.append(np.array(conc_f))
        return concentration

    def get_symmetric_concentration(self, c: float) -> list[np.ndarray]:
        concentration = []
        for n_states_f in self.shapes.n_states_per_feature:
            concentration.append(
                np.full(shape=n_states_f, fill_value=c)
            )
        return concentration

    def get_oneovern_concentration(self) -> list[np.ndarray]:
        concentration = []
        for n_states_f in self.shapes.n_states_per_feature:
            concentration.append(
                np.ones(shape=n_states_f) / n_states_f
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

    def __init__(
        self,
        *args,
        conf: ConfounderName,
        group_names: list[str],
        conf_effect_priors: dict[str, ConfoundingEffectsPrior],
        **kwargs
    ):
        self.conf = conf
        self.group_names = group_names
        self.any_dynamic_priors = False
        self.conf_effect_priors = conf_effect_priors
        self.precision = 0.0
        super(ConfoundingEffectsPrior, self).__init__(*args, **kwargs)

    def get_default_prior_config(self) -> ConfoundingEffectPriorConfig:
        return ConfoundingEffectPriorConfig()

    def parse_attributes(self):
        n_groups = len(self.group_names)
        self.concentration = {}
        self._concentration_array = np.zeros((n_groups, self.shapes.n_features, self.shapes.n_states), dtype=float)
        self.any_dynamic_priors = False

        default_config = self.config.get("<DEFAULT>", None)

        for i_g, group in enumerate(self.group_names):
            if group not in self.config:
                if default_config is None:
                    self.config[group] = self.get_default_prior_config()
                else:
                    self.config[group] = default_config

            config_g = self.config[group]
            group_prior_type = config_g.type
            if group_prior_type is self.PriorType.UNIFORM:
                self.concentration[group] = self.get_symmetric_concentration(1.0)
            elif group_prior_type is self.PriorType.JEFFREYS:
                self.concentration[group] = self.get_symmetric_concentration(0.5)
            elif group_prior_type is self.PriorType.BBS:
                self.concentration = self.get_oneovern_concentration()
            elif group_prior_type is self.PriorType.DIRICHLET:
                self.concentration[group] = self.load_concentration(config_g)
            elif self.prior_type is self.PriorType.SYMMETRIC_DIRICHLET:
                self.concentration[group] = self.get_symmetric_concentration(config_g.prior_concentration)
            elif group_prior_type is self.PriorType.UNIVERSAL:
                # Concentration will change based on sample of universal distribution...
                # For initialization: use uniform as dummy concentration to avoid invalid states
                self.concentration[group] = self.get_symmetric_concentration(config_g.prior_concentration)
                self.precision = config_g.prior_concentration
                self.any_dynamic_priors = True

            else:
                raise ValueError(self.invalid_prior_message(config_g.type))

            for i_f, conc_f in enumerate(self.concentration[group]):
                self._concentration_array[i_g, i_f, :len(conc_f)] = conc_f

        if not self.any_dynamic_priors:
            self._concentration_array.setflags(write=False)

    def concentration_array(self, sample: Sample):
        if self.any_dynamic_priors:
            universal_prior_counts = self.conf_effect_priors['universal'].concentration_array(sample)[0]
            universal_feature_counts = sample.feature_counts['universal'].value[0]
            univ_counts = universal_prior_counts + universal_feature_counts
            mean = normalize(univ_counts, axis=-1)
            precision = np.minimum(self.precision, univ_counts.sum(axis=-1, keepdims=True))

            # We add 1 to the precision for each possible feature state
            # TODO: discuss whether this makes sense
            precision += np.asarray(self.shapes.n_states_per_feature)[:, np.newaxis]

            concentration = mean * precision

            for i_g, group in enumerate(self.group_names):
                if self.config[group].type is self.PriorType.UNIVERSAL:
                    self._concentration_array[i_g] = concentration

        return self._concentration_array

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

    def generate_sample(self) -> dict[GroupName, NDArray[float]]:  # shape: (n_samples, n_features, n_states)
        group_effects = {}
        for i_g, conc_g in enumerate(self.concentration):
            group_effects[...] = np.array([

            ])
        return group_effects

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        msg = f"Prior on confounding effect {self.conf}:\n"

        for i_g, group in enumerate(self.group_names):
        # for i_g, group in enumerate(self.config):
            if self.config[group].type is self.PriorType.UNIFORM:
                msg += f'\tUniform prior for confounder {self.conf} = {group}.\n'
            elif self.config[group].type is self.PriorType.DIRICHLET:
                msg += f'\tDirichlet prior for confounder {self.conf} = {group}.\n'
            elif self.config[group].type is self.PriorType.UNIVERSAL:
                msg += f'\tDirichlet prior with universal distribution as mean for confounder {self.conf} = {group}.\n'
            else:
                raise ValueError(self.invalid_prior_message(self.config.type))
        return msg


class ClusterEffectPrior(DirichletPrior):

    def parse_attributes(self):
        self.prior_type = self.config.type
        if self.prior_type is self.PriorType.UNIFORM:
            self.concentration = self.get_symmetric_concentration(1.0)
        elif self.prior_type is self.PriorType.JEFFREYS:
            self.concentration = self.get_symmetric_concentration(0.5)
        elif self.prior_type is self.PriorType.BBS:
            self.concentration = self.get_oneovern_concentration()
        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

        self.concentration_array = np.zeros((self.shapes.n_features, self.shapes.n_states))
        for i_f, conc_f in enumerate(self.concentration):
            self.concentration_array[i_f, :len(conc_f)] = conc_f

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

    def get_symmetric_concentration(self, c: float) -> list[np.ndarray]:
        return [np.full(self.shapes.n_components, c) for _ in range(self.shapes.n_features)]

    def get_oneovern_concentration(self) -> list[np.ndarray]:
        return self.get_symmetric_concentration(1 / self.shapes.n_components)

    def parse_attributes(self):
        self.prior_type = self.config.type
        if self.prior_type is self.PriorType.UNIFORM:
            self.concentration = list(np.full(
                shape=(self.shapes.n_features, self.shapes.n_components),
                fill_value=self.initial_counts
            ))
        elif self.prior_type is self.PriorType.JEFFREYS:
            self.concentration = self.get_symmetric_concentration(0.5)
        elif self.prior_type is self.PriorType.BBS:
            self.concentration = self.get_oneovern_concentration()
        elif self.prior_type is self.PriorType.SYMMETRIC_DIRICHLET:
            self.concentration = self.get_symmetric_concentration(self.config.prior_concentration)
        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

        self.concentration_array = np.array(self.concentration)

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
            return cache.value

        log_p = 0.0
        if self.prior_type is self.PriorType.UNIFORM:
            pass
        elif self.prior_type in [self.PriorType.DIRICHLET,
                                 self.PriorType.JEFFREYS,
                                 self.PriorType.BBS,
                                 self.PriorType.SYMMETRIC_DIRICHLET]:
            log_p = compute_group_effect_prior(
                group_effect=parameter.value,
                concentration=self.concentration,
                applicable_states=np.ones_like(parameter.value, dtype=bool),
            )
        else:
            raise ValueError(self.invalid_prior_message(self.prior_type))

        cache.update_value(log_p)
        return log_p

    def pointwise_prior(self, sample: Sample) -> NDArray[float]:
        return compute_group_effect_prior_pointwise(
            group_effect=sample.weights.value,
            concentration=self.concentration,
            applicable_states=np.ones_like(sample.weights.value, dtype=bool),
        )

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f'Prior on weights: {self.prior_type.value}\n'

    def generate_sample(self) -> NDArray[float]:  # shape: (n_features, n_states)
        return np.array([np.random.dirichlet(c) for c in self.concentration])


class SourcePrior:

    def __init__(
        self,
        na_features: NDArray[bool]
    ):
        self.valid_observations = ~na_features

    def __call__(self, sample: Sample, caching=True) -> float:
        """Compute the prior for weights (or load from cache).
        Args:
            sample: Current MCMC sample.
        Returns:
            Logarithm of the prior probability density.
        """
        cache = sample.cache.source_prior
        # if caching and not cache.is_outdated():
        #     return cache.value

        weights = update_weights(sample)
        is_source = np.where(sample.source.value[self.valid_observations].ravel())
        observation_weights = weights[self.valid_observations].ravel()[is_source]
        source_prior = np.log(observation_weights).sum()

        cache.update_value(source_prior)
        return source_prior

    def old_source_prior(self, sample):
        weights = update_weights(sample)
        is_source = np.where(sample.source.value.ravel())
        observation_weights = weights.ravel()[is_source]
        return np.log(observation_weights).sum()

    def generate_sample(
            self,
            weights: NDArray[float],
            has_components: NDArray[bool],
    ) -> NDArray[float]:  # shape: (n_objects, n_features)
        p = normalize_weights(weights, has_components)
        return sample_categorical(p, binary_encoding=True)


class ClusterSizePrior:

    PriorType = ClusterSizePriorConfig.Types

    def __init__(self, config: ClusterSizePriorConfig, shapes: ModelShapes, initial_counts=1.):
        self.config = config
        self.shapes = shapes
        self.initial_counts = initial_counts
        self.prior_type = config.type
        self.min = self.config.min
        self.max = self.config.max

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
            logp = -log_multinom(self.shapes.n_sites, sizes)
            # logp = -log_binom(self.shapes.n_sites, sizes).sum()

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

    def generate_sample(self) -> NDArray[bool]:  # shape: (n_clusters, n_objects)
        n_clusters = self.shapes.n_clusters
        n_objects = self.shapes.n_sites
        if self.prior_type is self.PriorType.UNIFORM_AREA:
            onehots = np.eye(n_clusters + 1, n_clusters, dtype=bool)

            clusters = np.zeros((n_clusters, n_objects))
            while not np.all(self.min <= np.sum(clusters, axis=-1) <= self.max):
                clusters = onehots[np.random.randint(0, n_clusters+1, size=n_objects)].T
            return clusters
        else:
            raise NotImplementedError()


Aggregator = Callable[[Sequence[float]], float]
"""A type describing functions that aggregate costs in the geo-prior."""


class GeoPrior(object):

    PriorTypes = GeoPriorConfig.Types
    AggrStrats = GeoPriorConfig.AggregationStrategies

    AGGREGATORS: dict[str, Aggregator] = {
        AggrStrats.MEAN: np.mean,
        AggrStrats.SUM: np.sum,
        AggrStrats.MAX: np.max,
    }

    def __init__(
        self,
        config: GeoPriorConfig,
        cost_matrix: NDArray[float] = None,
        network: ComputeNetwork = None,
    ):
        self.config = config
        self.cost_matrix = cost_matrix
        self.network = network
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

        self.mean_edge_length = float(np.mean(
            network.dist_mat,
            where=network.adj_mat.toarray().astype(bool)
        ))

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

        if prob_function_type is GeoPriorConfig.ProbabilityFunction.EXPONENTIAL:
            return lambda x: -x / scale
            # == log(e**(-x/scale))

        elif prob_function_type is GeoPriorConfig.ProbabilityFunction.SIGMOID:
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
        elif self.prior_type is self.PriorTypes.DIAMETER_BASED:
            geo_prior = compute_diameter_based_geo_prior(
                clusters=sample.clusters.value,
                cost_mat=self.cost_matrix,
                aggregator=self.aggregator,
                probability_function=self.probability_function,
            )
        elif self.prior_type is self.PriorTypes.SIMULATED:
            geo_prior = compute_simulation_based_geo_prior(
                clusters=sample.clusters.value,
                cost_mat=self.cost_matrix,
                mean_edge_length=self.mean_edge_length,
            )
        else:
            raise ValueError('geo_prior must be either \"uniform\" or \"cost_based\".')

        cache.update_value(geo_prior)
        return geo_prior

    def get_costs_per_object(
        self,
        sample: Sample,
        i_cluster: int,
    ) -> NDArray[float]:  # shape: (n_objects)
        """Compute the change in the geo-prior (difference in log-probability) when adding each possible object."""

        if self.prior_type is self.PriorTypes.UNIFORM:
            # Uniform prior -> no changes in cost possible
            return np.zeros(sample.n_objects)

        cluster = sample.clusters.value[i_cluster]
        m = np.count_nonzero(cluster)
        cost_to_cluster = np.min(self.cost_matrix[cluster], axis=0)

        distances_before = compute_mst_distances(self.cost_matrix[cluster][:, cluster])
        aggr_cost_before = self.aggregator(distances_before)

        if self.aggregation_policy == self.AggrStrats.MEAN:
            aggr_cost_after = (cost_to_cluster + m*aggr_cost_before) / (1 + m)
        elif self.aggregation_policy == self.AggrStrats.SUM:
            aggr_cost_after = cost_to_cluster + aggr_cost_before
        elif self.aggregation_policy == self.AggrStrats.MAX:
            aggr_cost_after = np.maximum(cost_to_cluster, aggr_cost_before)
        else:
            raise ValueError(f"Aggregation strategy {self.aggregation_policy} not fully implemented yet.")

        return self.probability_function(aggr_cost_after) - self.probability_function(aggr_cost_before)

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

def compute_simulation_based_geo_prior(
    clusters: NDArray[bool],    # (n_clusters, n_objects)
    cost_mat: NDArray[float],   # (n_objects, n_objects)
    mean_edge_length: float,
) -> float:
    """Compute the geo-prior based on characteristics of areas and non-areas in the
    simulation in [https://github.com/Anaphory/area-priors]. The prior probability is
    conditioned on the area size and given by as sigmoid curve that is fitted using
    logistic regression to predict areality/non-areality of a group of languages based on
    their MST."""

    def sigmoid_intercept(n: int) -> float:
        a = -1.62973132061948
        b = 12.7679075267602
        c = -25.4137798184766
        d = 17.237407405487
        logn = np.log(n)
        return a * logn**3 + b * logn**2 + c * logn + d

    def sigmoid_coeff(n: int) -> float:
        a = -31.397363895626
        b = 1.02000702311327
        c = -94.0788824218419
        d = 0.93626444975598
        return a*b**(-n) + c/n + d

    log_prior = 0.0
    for z in clusters:
        n = np.count_nonzero(z)
        y0 = sigmoid_intercept(n)
        k = sigmoid_coeff(n)


        cost_mat_z = cost_mat[z][:, z]

        cost_mat_z = cost_mat_z * 0.039 / mean_edge_length

        if cost_mat_z.shape[0] > 1:
            graph = csgraph_from_dense(cost_mat_z, null_value=np.inf)
            mst = minimum_spanning_tree(graph)

            # When there are zero costs between languages the MST might be 0
            if mst.nnz > 0:
                distances = mst.tocsr()[mst.nonzero()]
            else:
                distances = [0.0]
        else:
            print(cost_mat_z)
            raise ValueError("Too few locations to compute distance.")

        x = np.sum(distances)
        log_prior += log_expit(k*x + y0)

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
    for c in clusters:
        distances = compute_mst_distances(cost_mat[c][:, c])
        agg_distance = aggregator(distances)
        log_prior += probability_function(agg_distance)

    return log_prior


def compute_mst_distances(cost_mat):
    if cost_mat.shape[0] <= 1:
        raise ValueError("Too few locations to compute distance.")

    graph = csgraph_from_dense(cost_mat, null_value=np.inf)
    mst = minimum_spanning_tree(graph)

    # When there are zero costs between languages the MST might be 0
    if mst.nnz == 0:
        return np.zeros(1)
    else:
        return mst.tocsr()[mst.nonzero()]


def compute_group_effect_prior(
        group_effect: NDArray[float],  # shape: (n_features, n_states)
        concentration: list[NDArray],  # shape: (n_applicable_states[f],) for f in features
        applicable_states: list[NDArray],  # shape: (n_applicable_states[f],) for f in features
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


def compute_group_effect_prior_pointwise(
        group_effect: NDArray[float],  # shape: (n_features, n_states)
        concentration: list[NDArray],  # shape: (n_applicable_states[f],) for f in features
        applicable_states: list[NDArray],  # shape: (n_applicable_states[f],) for f in features
) -> NDArray[float]:
    """" This function evaluates the prior on probability vectors in a cluster or confounder group.
    Args:
        group_effect: The group effect for a confounder
        concentration: List of Dirichlet concentration parameters.
        applicable_states: List of available states per feature
    Returns:
        The prior log-pdf of the confounding effect for each feature
    """
    n_features, n_states = group_effect.shape

    p = np.zeros(n_features)
    for f in range(n_features):
        states_f = applicable_states[f]
        conf_group = group_effect[f, states_f]
        p[f] = dirichlet_logpdf(x=conf_group, alpha=concentration[f])

    return p


if __name__ == '__main__':
    import doctest
    doctest.testmod()
