from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Callable
import json

import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from scipy.sparse.csgraph import minimum_spanning_tree, csgraph_from_dense
from scipy.sparse import csr_matrix
import libpysal as pysal

from sbayes.model.model_shapes import ModelShapes
from sbayes.util import dirichlet_logpdf, log_expit, FLOAT_TYPE, normalize_weights
from sbayes.config.config import PriorConfig, DirichletPriorConfig, GeoPriorConfig, ClusterPriorConfig, \
    ConfoundingEffectConfig, GaussianVariancePriorConfig, GaussianMeanPriorConfig, GaussianPriorConfig, \
    ClusterEffectConfig
from sbayes.load_data import Data, ComputeNetwork, GroupName, StateName, FeatureName, Confounder, \
    CategoricalFeatures, GaussianFeatures, GenericTypeFeatures


class Prior:
    """The joint prior of all parameters in the sBayes model.

    Attributes:
        cluster_prior (ClusterPrior): prior on the cluster size
        geo_prior (GeoPrior): prior on the geographic spread of a cluster
        weights_prior (WeightsPrior): prior on the mixture weights
        cluster_effect_prior (ClusterEffectPrior): prior on the areal effect
        confounding_effects_prior (CategoricalConfoundingEffectsPrior): prior on all confounding effects
    """

    def __init__(self, shapes: ModelShapes, config: PriorConfig, data: Data):
        self.shapes = shapes
        self.config = config
        self.data = data

        self.cluster_prior = ClusterPrior(config=config.cluster_assignment,
                                          shapes=self.shapes)
        self.geo_prior = GeoPrior(config=config.geo,
                                  cost_matrix=data.geo_cost_matrix,
                                  network=data.network)
        self.weights_prior = WeightsPrior(config=config.weights, shapes=self.shapes)

        self.cluster_effect_prior = ClusterEffectPrior(config=config.cluster_effect, partitions=data.features.partitions)
        self.confounding_effects_prior = {
            c: ConfoundingEffectsPrior(config=config.confounding_effects[c], conf=conf, partitions=data.features.partitions)
            for c, conf in data.confounders.items()
        }

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        setup_msg = self.geo_prior.get_setup_message()
        setup_msg += self.cluster_prior.get_setup_message()
        setup_msg += self.weights_prior.get_setup_message()
        for partition in self.data.features.partitions:
            setup_msg += f"Priors for partition {partition.name}:\n"
            setup_msg += self.cluster_effect_prior[partition.name].get_setup_message()
            for k, v in self.confounding_effects_prior.items():
                setup_msg += v[partition.name].get_setup_message()

        return setup_msg

    def __copy__(self):
        return Prior(
            shapes=self.shapes,
            config=self.config,
            data=self.data,
        )


def compute_has_components(clusters: NDArray[bool], confounders: dict[str, Confounder]):
    n_components = len(confounders) + 1
    n_objects = clusters.shape[1]

    has_components = np.empty((n_objects, n_components))
    has_components[:, 0] = np.any(clusters, axis=0)
    for i, conf in enumerate(confounders.values(), start=1):
        has_components[:, i] = conf.any_group()

    return np.array(has_components)


def parse_concentration_dict(
    concentration_dict: dict[FeatureName, dict[StateName, float]],
    feature_names: dict[FeatureName, Sequence[StateName]],
) -> list[np.ndarray]:
    """Compile the array with concentration parameters"""
    concentration = []
    for f, state_names_f in feature_names.items():
        conc_f = [concentration_dict[f][s] for s in state_names_f]
        concentration.append(np.array(conc_f, dtype=FLOAT_TYPE))
    return concentration


def parse_custom_concentration(config: DirichletPriorConfig, feature_names: dict[FeatureName, Sequence[StateName]]) -> list[np.ndarray]:
    # Get the concentration parameters from config or JSON file
    if config.file:
        with open(config.file, 'r') as f:
            concentration_dict = json.load(f)
    elif config.parameters:
        concentration_dict = config.parameters
    else:
        raise ValueError('DirichletPrior requires a file or parameters.')

    # Parse the config parameters (requires feature_names)
    return parse_concentration_dict(concentration_dict, feature_names)


def parse_dirichlet_concentration(
    config: DirichletPriorConfig,
    shape: tuple[int,...],
    feature_names: dict[FeatureName, Sequence[StateName]] | None = None,
) -> jnp.array:
    """Parse the concentration parameter of a Dirichlet prior."""
    if config.type == DirichletPriorConfig.Types.UNIFORM:
        return jnp.full(shape, 1.0)
    elif config.type == DirichletPriorConfig.Types.SYMMETRIC_DIRICHLET:
        assert config.prior_concentration is not None
        return jnp.full(shape, config.prior_concentration)
    elif config.type == DirichletPriorConfig.Types.DIRICHLET:
        assert feature_names is not None
        return parse_custom_concentration(config, feature_names)
    else:
        raise ValueError(f"Invalid Dirichlet prior type: {config.type}")


class CategoricalConfoundingEffectsPrior:

    def __init__(
        self,
        config: dict[GroupName, DirichletPriorConfig],
        conf: Confounder,
        partition: CategoricalFeatures,
    ):
        self.config = config
        self.conf = conf
        self.partition = partition
        self.group_names = conf.group_names

        n_groups = len(self.group_names)
        self.concentration = {}
        self.concentration_array = np.zeros((n_groups, partition.n_features, partition.n_states), dtype=float)
        default_config = config.get("<DEFAULT>", DirichletPriorConfig())
        for i_g, group in enumerate(self.group_names):
            # If config is not provided for this group, use the default config
            if group not in config:
                config[group] = default_config

            # Parse the concentration parameters into the concentration dictionary
            self.concentration[group] = parse_dirichlet_concentration(
                config=config[group],
                shape=(partition.n_features, partition.n_states),
                feature_names=partition.state_names_dict,
            )

            # Compile the concentration array
            self.concentration_array[i_g, ...] = np.array(self.concentration[group], dtype=FLOAT_TYPE)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        msg = f"Prior on confounding effect {self.conf.name}:\n"
        for i_g, group in enumerate(self.group_names):
            msg += f"\tPrior {self.config[group].type.value} for confounder {self.conf.name} = {group}.\n"
        return msg


class CategoricalClusterEffectPrior:

    def __init__(
        self,
        config: DirichletPriorConfig,
        partition: CategoricalFeatures,
    ):
        self.config = config
        self.partition = partition
        self.concentration = parse_dirichlet_concentration(
            config=self.config,
            shape=(partition.n_features, partition.n_states),
            feature_names=partition.state_names_dict,
        )
        self.concentration_array = np.array(self.concentration, dtype=FLOAT_TYPE)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f'Prior on cluster effect for {self.partition.name} features: {self.config.type.value}\n'


class GaussianMeanPrior:

    def __init__(
        self,
        config: GaussianPriorConfig  | dict[GroupName, GaussianPriorConfig],
        partition: GaussianFeatures,
        group_names: Sequence[GroupName] | None = None,
    ):
        self.config = config
        self.partition = partition
        self.group_names = group_names

        if isinstance(config, GaussianPriorConfig):
            # Parse the prior mean and variance from config
            mu_0_array, sigma_0_array = self.parse_group_prior(config.mean)

        elif isinstance(config, dict):
            if group_names is None:
                raise ValueError('Group names are required for multiple Gaussian priors.')

            default_config = config.get("<DEFAULT>", None)

            # Parse the prior mean and variance for each group from respective config
            n_groups = len(group_names)
            mu_0_array = np.zeros((n_groups, partition.n_features))
            sigma_0_array = np.zeros((n_groups, partition.n_features))
            for i_g, group in enumerate(group_names):
                group_config = config.get(group, default_config)
                if group_config is None:
                    raise ValueError(f'No gaussian prior config for group `{group}`.')
                mu_0_array[i_g, :], sigma_0_array[i_g, :] = self.parse_group_prior(group_config.mean)

        else:
            raise ValueError(f'Invalid Gaussian prior config: {config}')

        # Convert to jax arrays
        self.mu_0_array = jnp.array(mu_0_array)
        self.sigma_0_array = jnp.array(sigma_0_array)

    def parse_group_prior(self, config: GaussianMeanPriorConfig):
        n_features = self.partition.n_features
        if config.type is config.Types.GAUSSIAN:
            mu_0_array = jnp.full(n_features, config.parameters['mu_0'])
            sigma_0_array = jnp.full(n_features, config.parameters['sigma_0'])
        else:
            raise ValueError(self.invalid_prior_message(config.type))
        return mu_0_array, sigma_0_array

    def invalid_prior_message(self, s):
        name = self.__class__.__name__
        valid_types = ', '.join(self.config.Types)
        return f'Invalid prior type {s} for {name} (choose from [{valid_types}]).'


class GaussianVariancePrior:

    def __init__(
        self,
        config: GaussianPriorConfig | dict[GroupName, GaussianPriorConfig],
        partition: GaussianFeatures,
        group_names: Sequence[GroupName] | None = None,
    ):
        self.config = config
        self.partition = partition
        self.group_names = group_names

        if isinstance(config, GaussianPriorConfig):
            self.parameters = self.parse_group_prior(config.variance)
        elif isinstance(config, dict):
            if group_names is None:
                raise ValueError('Group names are required for multiple Gaussian priors.')
            default_config = config.get("<DEFAULT>", None)
            group_parameters = []
            for group in group_names:
                group_config = config.get(group, default_config)
                if group_config is None:
                    raise ValueError(f'No gaussian prior config for group `{group}`.')
                group_parameters.append(
                    self.parse_group_prior(group_config.variance)
                )
            self.parameters = jnp.array(group_parameters).transpose((1, 0, 2))
        else:
            raise ValueError(f'Invalid Gaussian prior config: {config}')

    def parse_group_prior(self, config: GaussianVariancePriorConfig):
        n_features = self.partition.n_features
        if config.type is config.Types.EXPONENTIAL:
            return jnp.full(n_features, config.parameters['rate'])
        if config.type is config.Types.GAMMA:
            return jnp.array([
                jnp.full(n_features, config.parameters['shape']),
                jnp.full(n_features, config.parameters['rate']),
            ])

        else:
            raise ValueError(self.invalid_prior_message(config.type))

    def invalid_prior_message(self, s):
        name = self.__class__.__name__
        valid_types = ', '.join(GaussianVariancePriorConfig.Types)
        return f'Invalid prior type {s} for {name} (choose from [{valid_types}]).'

    def get_numpyro_distr(self):
        if isinstance(self.config, GaussianPriorConfig):
            typ = self.config.variance.type
        else:
            assert isinstance(self.config, dict), self.config
            typ = next(iter(self.config.values())).variance.type

        if typ is GaussianVariancePriorConfig.Types.EXPONENTIAL:
            return dist.Exponential(rate=self.parameters)
        elif typ is GaussianVariancePriorConfig.Types.INV_GAMMA:
            raise NotImplementedError('InverseGamma prior not implemented.')
        elif typ is GaussianVariancePriorConfig.Types.GAMMA:
            return dist.Gamma(concentration=self.parameters[0], rate=self.parameters[1])


class GaussianConfoundingEffectsPrior:

    def __init__(
        self,
        config: dict[GroupName, GaussianPriorConfig],
        conf: Confounder,
        partition: GaussianFeatures,
    ):
        self.config = config
        self.conf = conf
        self.partition = partition
        self.mean = GaussianMeanPrior(config=config, partition=partition, group_names=conf.group_names)
        self.variance = GaussianVariancePrior(config=config, partition=partition, group_names=conf.group_names)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        msg = f"Prior on confounding effect {self.conf.name} for {self.partition.name} features:\n"
        for group in self.config.keys():
            msg += f"\tPrior for group {group}: (mean={self.config[group].mean.type.value}, variance={self.config[group].variance.type.value}).\n"
        return msg

class GaussianClusterEffectPrior:

    def __init__(
        self,
        config: GaussianPriorConfig,
        partition: GaussianFeatures,
    ):
        self.config = config
        self.partition = partition
        self.mean = GaussianMeanPrior(config=self.config, partition=partition)
        self.variance = GaussianVariancePrior(config=self.config, partition=partition)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f"Prior on cluster effect for {self.partition.name} features:  (mean={self.config.mean.type.value}, variance={self.config.variance.type.value})\n"


class ClusterEffectPrior:

    def __init__(
        self,
        config: ClusterEffectConfig,
        partitions: list[GenericTypeFeatures],
    ):
        self.config = config
        self.partition_priors = {}

        # Create prior for  each partition
        for p in partitions:
            if isinstance(p, CategoricalFeatures):
                self.partition_priors[p.name] = CategoricalClusterEffectPrior(config.categorical, p)
            elif isinstance(p, GaussianFeatures):
                self.partition_priors[p.name] = GaussianClusterEffectPrior(config.gaussian, p)
            else:
                raise NotImplementedError(f'Partition type {type(p)} is not supported.')

    def __getitem__(self, partition_name):
        return self.partition_priors[partition_name]

    def get_setup_message(self):
        return "".join(prior.get_setup_message() for prior in self.partition_priors.values())

class ConfoundingEffectsPrior:

    def __init__(
        self,
        config: dict[GroupName, ConfoundingEffectConfig],
        conf: Confounder,
        partitions: list[GenericTypeFeatures],
    ):
        self.config = config
        self.partition_priors = {}

        # Create prior for  each partition
        for p in partitions:
            if isinstance(p, CategoricalFeatures):
                categorical_configs = {g: c.categorical for g, c in config.items()}
                self.partition_priors[p.name] = CategoricalConfoundingEffectsPrior(categorical_configs, conf, p)
            elif isinstance(p, GaussianFeatures):
                gaussian_configs = {g: c.gaussian for g, c in config.items()}
                self.partition_priors[p.name] = GaussianConfoundingEffectsPrior(gaussian_configs, conf, p)
            else:
                raise NotImplementedError(f'Partition type {type(p)} is not supported.')

    def __getitem__(self, partition_name):
        return self.partition_priors[partition_name]

    def get_setup_message(self):
        return "".join(prior.get_setup_message() for prior in self.partition_priors.values())

class WeightsPrior:

    def __init__(
        self,
        config: DirichletPriorConfig | dict[GroupName, DirichletPriorConfig],
        shapes: ModelShapes,
    ):
        self.config = config
        self.shapes = shapes
        self.concentration = parse_dirichlet_concentration(
            config=self.config,
            shape=(shapes.n_features, self.shapes.n_components),
        )
        self.concentration_array = np.array(self.concentration, dtype=FLOAT_TYPE)

    def get_numpyro_distr(self) -> float:
        ...  # TODO: implement

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f'Prior on weights: {self.config.type.value}\n'


class ClusterPrior:

    PriorType = ClusterPriorConfig.Types

    def __init__(self, config: ClusterPriorConfig, shapes: ModelShapes):
        self.config = config
        self.shapes = shapes
        self.prior_type = config.type
        self.min = self.config.min
        self.max = self.config.max

        self.concentration = None
        self.logi_norm_loc = None
        self.logi_norm_scale = None

        self.parse_attributes()

    def parse_attributes(self):
        """Parse the attributes of the cluster assignment prior."""
        if self.prior_type is self.PriorType.CATEGORICAL:
            pass
        elif self.prior_type is self.PriorType.DIRICHLET:
            self.concentration = parse_dirichlet_concentration(
                config=self.config.dirichlet_config,
                shape=(self.shapes.n_objects, self.shapes.n_clusters + 1),
            )
        elif self.prior_type is self.PriorType.LOGISTIC_NORMAL:
            self.logi_norm_loc = self.config.logistic_normal_config.loc
            self.logi_norm_scale = self.config.logistic_normal_config.scale
        else:
            raise ValueError(f'Invalid prior type {self.prior_type} for cluster assignment.')

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f'Prior on cluster assignment: {self.prior_type.value}\n'

    def get_numpyro_distr(self):
        K = self.shapes.n_clusters
        if self.prior_type is self.PriorType.CATEGORICAL:
            with numpyro.plate("plate_objects_z", self.shapes.n_objects, dim=-1):
                z_int = numpyro.sample("z_int", dist.Categorical(jnp.ones(K + 1) / (K + 1)))
                z = jax.nn.one_hot(z_int, K+1)
                numpyro.deterministic("z", z)

        elif self.prior_type is self.PriorType.DIRICHLET:
            if self.config.hierarchical:
                c = numpyro.sample("z_concentration", dist.Uniform(0, 1))
                concentration = jnp.full((self.shapes.n_clusters + 1, ), c)
            else:
                concentration = self.concentration
            with numpyro.plate("plate_objects_z", self.shapes.n_objects, dim=-1):
                z = numpyro.sample("z", dist.Dirichlet(concentration))

        elif self.prior_type is self.PriorType.LOGISTIC_NORMAL:
            if self.config.hierarchical:
                scale = numpyro.sample("z_concentration", dist.LogNormal(0.0, 1.0))
                # scale = jnp.full((self.shapes.n_clusters + 1, ), s)
            else:
                scale = self.logi_norm_scale

            with numpyro.plate("plate_objects_z", self.shapes.n_objects, dim=-2):
                with numpyro.plate("plate_clusters_z", self.shapes.n_clusters + 1, dim=-1):
                    z_logit_unscaled = numpyro.sample("z_logit", dist.Normal(self.logi_norm_loc, 1.0))
                    z_logit = z_logit_unscaled * scale
                    z = jax.nn.softmax(z_logit, axis=-1)
                    numpyro.deterministic("z", z)
        else:
            raise ValueError(f'Invalid prior type {self.prior_type} for cluster assignment.')

        return z

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
        self.prob_func_type = None
        self.scale = None
        self.inflection_point = None
        self.cached = None
        self.aggregation = None
        self.linkage = None

        self.parse_attributes(config)

        # x = network.dist_mat[network.adj_mat.nonzero()].mean()
        self.mean_edge_length = compute_mst_distances(network.dist_mat).mean()

    def parse_attributes(self, config: GeoPriorConfig):
        if self.prior_type is config.Types.COST_BASED:
            if self.cost_matrix is None:
                ValueError('`cost_based` geo-prior requires a cost_matrix.')

            self.prior_type = self.PriorTypes.COST_BASED
            self.scale = config.rate
            self.aggregation_policy = config.aggregation
            self.aggregator = self.AGGREGATORS[self.aggregation_policy]

            self.prob_func_type = config.probability_function
            self.inflection_point = config.inflection_point

    def get_probability_function(self) -> Callable[[float], float]:
        if self.prob_func_type is GeoPriorConfig.ProbabilityFunction.EXPONENTIAL:
            return lambda x: -x / self.scale  # == log(e**(-x/scale))

        if self.prob_func_type is GeoPriorConfig.ProbabilityFunction.SQUARED_EXPONENTIAL:
            return lambda x: -(x / self.scale)**2  # == log(e**(-(x/scale)**2))

        elif self.prob_func_type is GeoPriorConfig.ProbabilityFunction.SIGMOID:
            # assert self.inflection_point is not None
            x0 = self.inflection_point
            s = self.scale
            return lambda x: log_expit(-(x - x0) / s) - log_expit(x0 / s)
            # return lambda x: jnp.log(1E-100 + jax.scipy.special.expit(-(x - x0) / s))
            # The last term `- log_expit(x0/s)` scales the sigmoid to be 1 at distance 0

        else:
            raise ValueError(f'Unknown probability_function `{self.prob_func_type}`')

    def get_numpyro_distr(self, clusters) -> float:
        """Compute the geo-prior of a fuzzy cluster.
        Args:
            clusters: Current sample of the fuzzy cluster assignments.
        Returns:
            Logarithm of the prior probability density
        """
        if self.prior_type is self.PriorTypes.UNIFORM:
            return 0.0

        n_objects, n_clusters = clusters.shape
        cluster_size = jnp.sum(clusters, axis=-2)
        prob_func = self.get_probability_function()

        dist_mat = self.cost_matrix

        # distances, weights = self.compute_distances_along_skeleton(clusters)
        if self.config.skeleton is GeoPriorConfig.Skeleton.MST:
            aggregated_distance = 0.0
            for cluster in clusters.T:
                aggregated_distance += self.compute_fuzzy_mst_distance(cluster)
        elif self.config.skeleton is GeoPriorConfig.Skeleton.COMPLETE:
            clusters_normed = clusters / cluster_size[None, :]
            same_cluster_prob = clusters_normed @ clusters.T  # Expected distance to a random language
            aggregated_distance = jnp.sum(same_cluster_prob * dist_mat)

        elif self.config.skeleton == GeoPriorConfig.Skeleton.SPECTRAL:
            def get_spectrum(C):
                    L = jnp.fill_diagonal(C, jnp.sum(C, axis=-1), inplace=False)
                    eigvals = jnp.linalg.eigvals(L)
                    return jnp.sum(jnp.real(eigvals))

            # Compute spectral geo-prior
            connectivities = clusters.T[:, :, None] * clusters.T[:, None, :]  # shape (n_clusters, n_objects, n_objects)
            mats = connectivities * dist_mat  # shape (n_clusters, n_objects, n_objects)
            eigvals_batched = jax.vmap(get_spectrum, in_axes=0, out_axes=0)(mats)
            # eigvals_batched = jax.vmap(jnp.linalg.eigvals, in_axes=0, out_axes=0)(mats)
            aggregated_distance = jnp.sum(jnp.real(eigvals_batched))
        elif self.config.skeleton is GeoPriorConfig.Skeleton.DIAMETER:
            aggregated_distance = jnp.sum(average_max_distance(clusters, self.cost_matrix))
        else:
            raise ValueError(f'Unknown skeleton type `{self.config.skeleton}`')

        # for i_c in range(n_clusters):
        #     c = clusters[i_c]
        #     if self.prior_type is self.PriorTypes.COST_BASED:
        #         distances = self.compute_distances_along_skeleton(c)
        #         agg_distance = self.aggregator(distances)
        #         geo_prior += prob_func(agg_distance)
        #     else:
        #         raise ValueError('geo_prior must be either \"uniform\" or \"cost_based\".')

        log_geo_priors = prob_func(aggregated_distance)
        numpyro.factor('geo_prior', log_geo_priors)

        return log_geo_priors

    def compute_fuzzy_mst_distance(self, cluster):
        C = jnp.array(self.cost_matrix)

        # Objects are considered the `core` if they are more likely in the cluster than not
        core = cluster > 0.5

        # # Compute the distance of each object to the core
        dist_to_core = jnp.min(C, axis=1, where=core, initial=jnp.inf)

        # Compute the minimum spanning tree within the core objects
        C_core = C[core][:, core]
        core_mst = minimum_spanning_tree(C_core)

        return core_mst.sum()



    def compute_distances_along_skeleton(self, cluster):
        skeleton = self.config.skeleton
        skeleton_types = GeoPriorConfig.Skeleton

        cost_mat = self.cost_matrix[cluster][:, cluster]
        locations = self.network.lat_lon[cluster]

        if skeleton is skeleton_types.MST:
            return compute_mst_distances(cost_mat)
        elif skeleton is skeleton_types.DELAUNAY:
            return compute_delaunay_distances(locations, cost_mat)
        elif skeleton is skeleton_types.DIAMETER:
            raise NotImplementedError
        elif skeleton is skeleton_types.COMPLETE:
            return cost_mat

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

    def local_costs(self, clusters, k=3) -> NDArray[float]:
        """Compute the local costs for all clusters."""
        n_objects, n_clusters = clusters.shape
        neighbour_sort_idxs = jnp.argsort(self.cost_matrix, axis=-1)
        row_idxs = jnp.arange(n_objects)[:, None]
        cost_sorted = self.cost_matrix[row_idxs, neighbour_sort_idxs]
        probs_sorted = clusters[row_idxs, neighbour_sort_idxs]
        first_k = first_k_continuous(probs_sorted, k, axis=-1)
        return cost_sorted.dot(first_k)

    def continuous_diameters(self, clusters) -> jnp.array:
        """Compute the continuous diameter of all clusters."""
        cost_mat = self.cost_matrix
        diameters = []
        for c in clusters.T:
            edge_probs = jnp.ravel(c[:, None] * c[None, :])
            edge_costs = jnp.ravel(cost_mat)
            max_cost_order = jnp.argsort(edge_costs, descending=True)
            max_cost_distr = first_k_continuous(edge_probs[max_cost_order], k=1)
            d = edge_costs[max_cost_order].dot(max_cost_distr)
            diameters.append(d)

        return jnp.array(diameters)

def average_max_distance(clusters: jnp.array, cost_matrix: jnp.array) -> jnp.array:
    """Compute the average distance from each node to the farthest node in the cluster.

    Args:
        clusters: The fuzzy cluster assignments.
        cost_matrix: The cost matrix between locations

    Usage:
    >>> clusters = np.array([[0.3, 0.7], [0.4, 0.6], [0.8, 0.2]])
    >>> cost_matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    >>> jnp.round(average_max_distance(clusters, cost_matrix), 4)  # round for numerically stable comparison
    Array([2.0133, 1.3333], dtype=float32)
    """
    n_objects, n_clusters = clusters.shape

    max_cost_order = jnp.argsort(cost_matrix, axis=-1, descending=True)
    sorted_cost = jnp.take_along_axis(cost_matrix, max_cost_order, axis=-1)
    # clusters_normed = clusters / jnp.sum(clusters, axis=0, keepdims=True)

    max_distances = []
    for i in range(n_clusters):
        c = clusters[:, i]
        other_probs = jnp.repeat(c[None, :], n_objects, axis=0)
        sorted_probs = jnp.take_along_axis(other_probs, max_cost_order, axis=-1)
        max_cost_distr = first_k_continuous(sorted_probs, k=1, axis=-1)
        d = jnp.sum(max_cost_distr * sorted_cost, axis=-1)
        # d_expected = jnp.dot(clusters_normed[:, i], d)
        d_expected = jnp.dot(c, d)
        max_distances.append(d_expected)
    return jnp.array(max_distances)


def first_k_continuous(probs: jnp.array, k: float, axis: int = -1) -> jnp.array:
    """Clip the probabilities in `probs` to only contain the first `k` probability mass.

    Args:
        probs: The probabilities to clip.
        k: The number of probabilities to keep.
    Returns:
        The clipped probabilities `probs_to_k` with `sum(probs_to_k) == k`.

    == Usage ===
    >>> np.round(first_k_continuous(jnp.array([0.8, 0.3, 0.6]), 1.0), 1)
    Array([0.8, 0.2, 0. ], dtype=float32)
    """
    cum_probs = jnp.cumsum(probs, axis=axis)
    cum_probs_to_k = cum_probs.clip(0, k)
    probs_to_k = jnp.diff(cum_probs_to_k, prepend=0, axis=axis)
    return probs_to_k


def compute_diameter_based_geo_prior(
        clusters: NDArray[bool],
        cost_mat: NDArray[float],
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

class SimulatedSigmoid:

    @staticmethod
    @lru_cache(maxsize=128)
    def intercept(n: int) -> float:
        a = -1.62973132061948
        b = 12.7679075267602
        c = -25.4137798184766
        d = 17.237407405487
        logn = np.log(n)
        return a * logn**3 + b * logn**2 + c * logn + d

    @staticmethod
    @lru_cache(maxsize=128)
    def coeff(n: int) -> float:
        a = -31.397363895626
        b = 1.02000702311327
        c = -94.0788824218419
        d = 0.93626444975598
        return a*b**(-n) + c/n + d

    @staticmethod
    def sigmoid(total_distance: float, n: int) -> float:
        y0 = SimulatedSigmoid.intercept(n)
        k = SimulatedSigmoid.coeff(n)
        return log_expit(k * total_distance + y0)


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

    log_prior = 0.0
    for z in clusters:
        n = np.count_nonzero(z)
        # cost_mat_z = cost_mat[z][:, z] * 0.039 / mean_edge_length
        cost_mat_z = cost_mat[z][:, z] * 0.020838 / mean_edge_length
        distances = compute_mst_distances(cost_mat_z)
        log_prior += SimulatedSigmoid.sigmoid(distances.sum(), n)

    return log_prior


def compute_mst_distances(cost_mat: NDArray[float]) -> csr_matrix:
    if cost_mat.shape[0] <= 1:
        return np.zeros_like(cost_mat)
        # raise ValueError("Too few locations to compute distance.")

    graph = csgraph_from_dense(cost_mat, null_value=np.inf)
    mst = minimum_spanning_tree(graph)

    # When there are zero costs between languages the MST might be 0
    if mst.nnz == 0:
        return np.zeros(1)
    else:
        return mst.tocsr()[mst.nonzero()]


def compute_delaunay_distances(
    locations: NDArray[float],
    cost_mat: NDArray[float],
) -> csr_matrix:
    if cost_mat.shape[0] <= 1:
        raise ValueError("Too few locations to compute distance.")


    # graph = csgraph_from_dense(cost_mat, null_value=np.inf)
    cells = pysal.cg.voronoi_frames(locations, return_input=False, as_gdf=True)
    delaunay = pysal.weights.Rook.from_dataframe(cells, use_index=False).to_sparse()
    dists = delaunay.multiply(cost_mat)

    # When there are zero costs between languages the MST might be 0
    if dists.nnz == 0:
        return np.zeros(1)
    else:
        return dists.tocsr()[dists.nonzero()]



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


def update_weights(sample, caching: bool = True) -> NDArray[float]:
    """Compute the normalized mixture weights of each component at each object.
    Args:
        sample: the current MCMC sample.
        caching: ignore cache if set to false.
    Returns:
        np.array: normalized weights of each component at each object.
            shape: (n_objects, n_features, 1 + n_confounders)
    """
    cache = sample.cache.weights_normalized

    if (not caching) or cache.is_outdated():
        w_normed = normalize_weights(sample.weights.value, sample.cache.has_components.value)
        cache.update_value(w_normed)

    return cache.value


if __name__ == '__main__':
    import doctest
    doctest.testmod()
