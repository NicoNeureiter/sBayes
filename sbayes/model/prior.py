from __future__ import annotations

from functools import lru_cache
from matplotlib import pyplot as plt
from typing import Sequence, Callable
import json

import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import StickBreakingTransform, AffineTransform, Transform
from scipy.sparse.csgraph import minimum_spanning_tree, csgraph_from_dense
from scipy.sparse import csr_matrix
import libpysal as pysal

from sbayes.model.geoprior import estimate_marginal_log_likelihood_curve
from sbayes.model.model_shapes import ModelShapes
from sbayes.util import log_expit, FLOAT_TYPE, normalize_weights, EPS, normalize
from sbayes.config.config import PriorConfig, CategoricalPriorConfig, GeoPriorConfig, ClusterPriorConfig, \
    ConfoundingEffectConfig, GaussianVariancePriorConfig, GaussianMeanPriorConfig, GaussianPriorConfig, \
    ClusterEffectConfig, PoissonPriorConfig
from sbayes.load_data import Data, ComputeNetwork, GroupName, StateName, FeatureName, Confounder, \
    CategoricalFeatures, GaussianFeatures, GenericTypeFeatures, PoissonFeatures


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
        concentration.append(jnp.array(conc_f, dtype=FLOAT_TYPE))
    return concentration


def parse_custom_concentration(config: CategoricalPriorConfig, feature_names: dict[FeatureName, Sequence[StateName]]) -> list[np.ndarray]:
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
    config: CategoricalPriorConfig,
    shape: tuple[int,...],
    feature_names: dict[FeatureName, Sequence[StateName]] | None = None,
) -> jnp.array:
    """Parse the concentration parameter of a Dirichlet prior."""
    if config.type == CategoricalPriorConfig.Types.UNIFORM:
        return jnp.full(shape, 1.0)
    elif config.type == CategoricalPriorConfig.Types.SYMMETRIC_DIRICHLET:
        assert config.prior_concentration is not None
        return jnp.full(shape, config.prior_concentration)
    elif config.type == CategoricalPriorConfig.Types.DIRICHLET:
        assert feature_names is not None
        return parse_custom_concentration(config, feature_names)
    else:
        raise ValueError(f"Invalid Dirichlet prior type: {config.type}")


class CategoricalConfoundingEffectsPrior:

    def __init__(
        self,
        config: dict[GroupName, CategoricalPriorConfig],
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
        default_config = config.get("<DEFAULT>", None)
        for i_g, group in enumerate(self.group_names):
            # If config is not provided for this group, use the default config
            if group not in config:
                config[group] = default_config
                if default_config is None:
                    raise ValueError("Provide a confounding effects prior for every group or specify a default prior")

            # Parse the concentration parameters into the concentration dictionary
            self.concentration[group] = parse_dirichlet_concentration(
                config=config[group],
                shape=(partition.n_features, partition.n_states),
                feature_names=partition.state_names_dict,
            )

            # Compile the concentration array
            self.concentration_array[i_g, ...] = np.array(self.concentration[group], dtype=FLOAT_TYPE)

        self.use_parameter_transformation = self.config[self.group_names[0]].use_parameter_transformation

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        msg = f"Prior on confounding effect {self.conf.name}:\n"
        for i_g, group in enumerate(self.group_names):
            msg += f"\tPrior {self.config[group].type.value} for confounder {self.conf.name} in partition {self.partition.name} = {group}.\n"
        return msg

    def get_numpyro_distr(self, allow_reparameterization: bool = True):
        p_name = self.partition.name
        c_name = self.conf.name
        with numpyro.plate(f"plate_groups_{c_name}_{p_name}", self.conf.n_groups, dim=-2):
            with numpyro.plate(f"plate_features_{c_name}_{p_name}", self.partition.n_features, dim=-1):
                if allow_reparameterization and self.use_parameter_transformation:
                    conf_eff = dirichlet_from_latent(f"conf_effect_{c_name}_{p_name}", self.concentration_array)
                else:
                    conf_eff_distr = dist.Dirichlet(self.concentration_array)
                    conf_eff = numpyro.sample(f"conf_effect_{c_name}_{p_name}", conf_eff_distr)

        return conf_eff

class CategoricalClusterEffectPrior:

    PriorType = CategoricalPriorConfig.Types

    def __init__(
        self,
        config: CategoricalPriorConfig,
        partition: CategoricalFeatures,
    ):
        self.config = config
        self.partition = partition

        self.prior_type = self.config.type
        if self.prior_type in [self.PriorType.UNIFORM,
                               self.PriorType.DIRICHLET,
                               self.PriorType.SYMMETRIC_DIRICHLET]:
            self.concentration = parse_dirichlet_concentration(
                config=self.config,
                shape=(partition.n_features, partition.n_states),
                feature_names=partition.state_names_dict,
            )
            self.concentration_array = jnp.array(self.concentration, dtype=FLOAT_TYPE)
        elif self.prior_type is self.PriorType.LOGISTIC_NORMAL:
            self.logi_norm_loc = 0.0
            self.logi_norm_scale = self.config.logistic_normal_scale
        else:
            raise ValueError(f'Invalid prior type {self.prior_type} for cluster assignment.')

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        return f'Prior on cluster effect for {self.partition.name} features: {self.config.type.value}\n'

    def get_data_dependent_contributions(
        self,
        clusters_weights: jnp.array,  # shape: (n_clusters, n_objects, n_features)
        additive_smoothing: float = 0.5,
    ):
        x = self.partition.to_binary().astype(jnp.float32)
        # shape: (n_objects, n_features, n_states)

        # prior_counts = self.concentration_array
        feature_counts = jnp.einsum("ijk,jkl->ikl", clusters_weights, x)
        return normalize(feature_counts + additive_smoothing, axis=-1)

    def get_numpyro_distr(
        self,
        n_clusters: int,
        clust_eff_pred: jnp.array,  # (n_clusters, n_features, n_states)
        allow_reparameterization: bool = False,
    ):
        n_features = self.partition.n_features
        p_name = self.partition.name

        # clust_eff_offset = numpyro.param(f"clust_eff_offset_{p_name}", jnp.zeros_like(clust_eff_pred, dtype=jnp.float32))
        with numpyro.plate(f"plate_clusters_{p_name}_offset", n_clusters, dim=-2):
            with numpyro.plate(f"plate_features_{p_name}_offset", n_features, dim=-1):
                if allow_reparameterization and self.config.use_parameter_transformation:
                    clust_eff = dirichlet_from_latent(
                        name=f"cluster_effect_{p_name}",
                        concentration=self.concentration,
                        offset=clust_eff_pred,
                    )
                else:
                    clust_eff_distr = dist.Dirichlet(self.concentration)
                    clust_eff = numpyro.sample(f"cluster_effect_{p_name}", clust_eff_distr)
        #
        # if self.prior_type == self.PriorType.LOGISTIC_NORMAL:
        #     clust_eff_pred_latent = jnp.log(clust_eff_pred * n_states)
        #     cluster_eff_raw = clust_eff_pred_latent + clust_eff_offset
        # else:
        #     transform = StickBreakingTransform()
        #     clust_eff_pred_latent = transform.inv(clust_eff_pred)
        #     cluster_eff_raw = transform(clust_eff_pred_latent + clust_eff_offset)
        #
        # numpyro.deterministic(f"clust_eff_pred_{p_name}", clust_eff_pred_latent)
        # numpyro.deterministic(f"clust_eff_raw_{p_name}", cluster_eff_raw)
        #
        # with numpyro.plate(f"plate_clusters_{p_name}", n_clusters, dim=-2):
        #     with numpyro.plate(f"plate_features_{p_name}", n_features, dim=-1):
        #         if self.prior_type == self.PriorType.LOGISTIC_NORMAL:
        #             with numpyro.plate(f"plate_states_{p_name}", n_states, dim=-1):
        #             # if self.prior_type is self.PriorType.LOGISTIC_NORMAL:
        #                 clust_eff_distr = dist.Normal(self.logi_norm_loc, 1.0)
        #         else:
        #             clust_eff_distr = dist.Dirichlet(self.concentration)
        #
        #         prior_log_prob = clust_eff_distr.log_prob(cluster_eff_raw) + transform.log_abs_det_jacobian(clust_eff_pred_latent + clust_eff_offset, cluster_eff_raw)
        #         numpyro.factor(f"cluster_effect_factor_{p_name}", prior_log_prob)
        #         # cluster_eff_raw = numpyro.sample(f"cluster_effect_raw_{p_name}", clust_eff_distr)
        #
        #     # clust_eff_per_clust.append(clust_eff_i)
        #
        #
        # numpyro.deterministic(f"cluster_effect_{p_name}", clust_eff)

        return clust_eff


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

    def get_data_dependent_contributions(
        self,
        clusters_weights: jnp.array,  # shape: (n_clusters, n_objects, n_features)
    ):
        x = self.partition.values
        # shape: (n_objects, n_features)

        observation_counts = jnp.sum(clusters_weights, axis=1)
        prec_0 = 1.0 / self.sigma_0_array
        return (self.mu_0_array / prec_0 + jnp.sum(clusters_weights * x[None, :, :], axis=1)) / (prec_0 + observation_counts)

    def get_numpyro_distr(
        self,
        n_clusters: int,
        clust_eff_mean_pred: jnp.array,  # (n_clusters, n_features,)
    ):
        n_features = self.partition.n_features
        p_name = self.partition.name

        # The prior distribution is Normal with mean mu_0 and standard deviation sigma_0
        mean_dist = dist.Normal(self.mu_0_array, self.sigma_0_array)

        # Define the samples from this prior. Either transformed or directly...
        if not self.config.use_parameter_transformation:
            with numpyro.plate(f"plate_clusters_{p_name}", n_clusters, dim=-2):
                with numpyro.plate(f"plate_features_{p_name}", n_features, dim=-1):
                    mean = numpyro.sample(f"cluster_effect_{p_name}_mean", mean_dist)
            return mean
        else:
            offset_dist = dist.Uniform(-1, 1)
            with numpyro.plate(f"plate_clusters_{p_name}", n_clusters, dim=-2):
                with numpyro.plate(f"plate_features_{p_name}", n_features, dim=-1):
                    offset_latent = self.sigma_0_array * numpyro.sample(f"cluster_effect_{p_name}_mean_offset", offset_dist)

            trans = AffineTransform(loc=0.0, scale=10.0)
            offset = trans(offset_latent)

            # Calculate the mean as sum of off
            effect_mean = clust_eff_mean_pred + offset
            numpyro.deterministic(f"cluster_effect_{p_name}_mean", effect_mean)

            # Add actual prior probability as factor
            prior_log_prob = mean_dist.log_prob(effect_mean)
            prior_correction_factor = trans.log_abs_det_jacobian(offset_latent, offset) - offset_dist.log_prob(offset_latent)
            corrected_log_prob = prior_log_prob + prior_correction_factor
            numpyro.factor(f"cluster_effect_{p_name}_mean_log_prob", corrected_log_prob)

            return effect_mean


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
        elif config.type is config.Types.GAMMA:
            return jnp.array([
                jnp.full(n_features, config.parameters['shape']),
                jnp.full(n_features, config.parameters['rate']),
            ])
        elif config.types is config.Types.FIXED:
            return jnp.full(n_features, config.parameters['value'])
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
        elif typ is GaussianVariancePriorConfig.Types.FIXED:
            return dist.Delta(v=self.parameters)


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


class PoissonRatePrior:

    def __init__(
        self,
        config: PoissonPriorConfig | dict[GroupName, PoissonPriorConfig],
        partition: PoissonFeatures,
        group_names: Sequence[GroupName] | None = None,
    ):
        self.config = config
        self.partition = partition
        self.group_names = group_names

        if isinstance(config, PoissonPriorConfig):
            self.parameters = self.parse_group_prior(config)
        elif isinstance(config, dict):
            if group_names is None:
                raise ValueError('Group names are required for poisson priors.')
            default_config = config.get("<DEFAULT>", None)
            group_parameters = []
            for group in group_names:
                group_config = config.get(group, default_config)
                if group_config is None:
                    raise ValueError(f'No poisson prior config for group `{group}`.')
                group_parameters.append(
                    self.parse_group_prior(group_config)
                )
            self.parameters = jnp.array(group_parameters).transpose((1, 0, 2))
        else:
            raise ValueError(f'Invalid Gaussian prior config: {config}')

    def parse_group_prior(self, config: PoissonPriorConfig):
        n_features = self.partition.n_features
        if config.type is config.Types.JEFFREYS:
            return None
        if config.type is config.Types.GAMMA:
            return jnp.array([
                jnp.full(n_features, config.parameters['shape']),
                jnp.full(n_features, config.parameters['rate']),
            ])
        else:
            raise ValueError(self.invalid_prior_message(config.type))

    def invalid_prior_message(self, s):
        name = self.__class__.__name__
        valid_types = ', '.join(PoissonPriorConfig.Types)
        return f'Invalid prior type {s} for {name} (choose from [{valid_types}]).'

    def get_numpyro_distr(self):
        if isinstance(self.config, PoissonPriorConfig):
            typ = self.config.type
        else:
            assert isinstance(self.config, dict), self.config
            typ = next(iter(self.config.values())).type
            #assert (all(v.variance.type == typ for v in self.config.values()))

        if typ is PoissonPriorConfig.Types.JEFFREYS:
            return dist.Exponential(rate=self.parameters)
        elif typ is PoissonPriorConfig.Types.GAMMA:
            return dist.Gamma(concentration=self.parameters[0], rate=self.parameters[1])
        else:
            raise ValueError(f'Invalid prior type {typ} for Poisson rate prior.')


class PoissonConfoundingEffectsPrior:

    def __init__(
        self,
        config: dict[GroupName, PoissonPriorConfig],
        conf: Confounder,
        partition: PoissonFeatures,
    ):
        self.config = config
        self.conf = conf
        self.partition = partition
        self.rate = PoissonRatePrior(config=config, partition=partition, group_names=conf.group_names)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        msg = f"Prior on confounding effect {self.conf.name} for {self.partition.name} features:\n"
        for group in self.config.keys():
            msg += f"\tPrior for group {group}: (mean={self.config[group].type.value}).\n"
        return msg


class PoissonClusterEffectPrior:

    def __init__(
        self,
        config: PoissonPriorConfig,
        partition: PoissonFeatures,
    ):
        self.config = config
        self.partition = partition
        self.mean = PoissonRatePrior(config=self.config, partition=partition)

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
            elif isinstance(p, PoissonFeatures):
                self.partition_priors[p.name] = PoissonClusterEffectPrior(config.poisson, p)
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
            elif isinstance(p, PoissonFeatures):
                poisson_configs = {g: c.poisson for g, c in config.items()}
                self.partition_priors[p.name] = PoissonConfoundingEffectsPrior(poisson_configs, conf, p)
            else:
                raise NotImplementedError(f'Partition type {type(p)} is not supported.')

    def __getitem__(self, partition_name):
        return self.partition_priors[partition_name]

    def get_setup_message(self):
        return "".join(prior.get_setup_message() for prior in self.partition_priors.values())

class WeightsPrior:

    def __init__(
        self,
        config: CategoricalPriorConfig | dict[GroupName, CategoricalPriorConfig],
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
                # shape=(self.shapes.n_objects, self.shapes.n_clusters + 1),
                shape=(self.shapes.n_clusters + 1,),
            )
        elif self.prior_type is self.PriorType.LOGISTIC_NORMAL:
            self.logi_norm_loc = self.config.logistic_normal_config.loc
            self.logi_norm_scale = self.config.logistic_normal_config.scale
        else:
            raise ValueError(f'Invalid prior type {self.prior_type} for cluster assignment.')

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        msg = f'Prior on cluster assignment: {self.prior_type.value}\n'
        if self.config.hierarchical:
            msg += f'\tEstimate cluster prior concentration\n'
        else:
            msg += f'\tFixed cluster prior concentration at c={self.concentration[0]}\n'
        if self.config.estimate_no_cluster_concentration:
            msg += f'\tEstimate non-cluster concentration.\n'

        return msg

    def get_numpyro_distr(self, allow_reparameterization: bool = True):
        K = self.shapes.n_clusters
        if self.prior_type is self.PriorType.CATEGORICAL:
            with numpyro.plate("plate_objects_z", self.shapes.n_objects, dim=-1):
                z_int = numpyro.sample("z_int", dist.Categorical(jnp.ones(K + 1) / (K + 1)))
                z = jax.nn.one_hot(z_int, K+1)
            numpyro.deterministic("z_raw", z)
            numpyro.deterministic("z", z)

        elif self.prior_type is self.PriorType.DIRICHLET:
            if self.config.hierarchical:
                c = numpyro.sample("z_concentration", dist.Uniform(0, 1))
                # c = numpyro.sample("z_concentration", dist.Beta(4., 4.))
                concentration = jnp.full((self.shapes.n_clusters + 1, ), c)
            else:
                # concentration = np.full((self.shapes.n_clusters + 1,), self.concentration)
                concentration = self.concentration

            if self.config.estimate_no_cluster_concentration:
                # c_nocluster = numpyro.sample("z_concentration_nocluster", dist.Uniform(0, 1))
                # c_nocluster = numpyro.sample("z_concentration_nocluster", dist.Exponential(1.0))
                c_nocluster = numpyro.sample("z_concentration_nocluster", dist.LogNormal(0.0, 1.0))
            else:
                c_nocluster = self.config.no_cluster_concentration

            if c_nocluster is not None:
                concentration = concentration.at[-1].set(c_nocluster)

            with numpyro.plate("plate_objects_z", self.shapes.n_objects, dim=-1):
                if allow_reparameterization and self.config.dirichlet_config.use_parameter_transformation:
                    z = dirichlet_from_latent("z_unstretched", concentration, offset=normalize(concentration))
                else:
                    z = numpyro.sample("z_unstretched", dist.Dirichlet(concentration))

        elif self.prior_type is self.PriorType.LOGISTIC_NORMAL:
            if self.config.hierarchical:
                scale = numpyro.sample("z_concentration", dist.LogNormal(0.0, 1.0))
            else:
                scale = self.logi_norm_scale

            with numpyro.plate("plate_objects_z", self.shapes.n_objects, dim=-2):
                with numpyro.plate("plate_clusters_z", self.shapes.n_clusters + 1, dim=-1):
                    z_raw = numpyro.sample("z_raw", dist.Normal(self.logi_norm_loc, 1.0))

            z_logit = z_raw * scale
            z = jax.nn.softmax(z_logit, axis=-1)

        else:
            raise ValueError(f'Invalid prior type {self.prior_type} for cluster assignment.')

        if self.config.stretch_and_clip:
            # s = self.config.stretch_factor
            s = 1 + numpyro.sample("z0_stretch_factor", dist.Exponential(1.))
            z_stretched = z.at[:, -1].multiply(s)
            d = z_stretched[:, -1:] - z[:, -1:]
            z_stretched = z_stretched.at[:, :-1] \
                .subtract((z[:, :-1] / (jnp.sum(z[:, :-1], axis=-1, keepdims=True) + 1E6)) * d)

            z_stretched = jnp.clip(z_stretched, 1E-9, 1 - 1E-9)

            # Shouldn't be necessary, but renormalize for numerical stability
            z_stretched = normalize(z_stretched, axis=-1)

            z = z_stretched

        numpyro.deterministic("z", z)

        # # Penalty on clusters without a core
        # highest_z_per_cluster = jnp.max(z, axis=-2)[:-1]
        # # highest_z_distr = dist.Beta(1.0, 0.2)
        # # numpyro.factor("highest_z_penalty", highest_z_distr.log_prob(highest_z_per_cluster))
        # highest_z_penalty = numpyro.sample("highest_z_penalty", dist.LogNormal(np.log(1.0), 1.))
        # numpyro.factor("", -highest_z_penalty * jnp.abs(1 - highest_z_per_cluster))

        return z


Aggregator = Callable[[Sequence[float]], float]
"""A type describing functions that aggregate costs in the geo-prior."""


class GeoPrior(object):

    PriorTypes = GeoPriorConfig.Types
    AggrStrats = GeoPriorConfig.AggregationStrategies

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
        self.aggregation_policy = None
        self.prob_func_type = None
        self.scale = None
        self.inflection_point = None
        self.cached = None
        self.linkage = None

        self.parse_attributes(config)

        self._norm_const_interpolator: callable | None = None

    def parse_attributes(self, config: GeoPriorConfig):
        if self.prior_type is config.Types.COST_BASED:
            if self.cost_matrix is None:
                ValueError('`cost_based` geo-prior requires a cost_matrix.')

            self.prior_type = self.PriorTypes.COST_BASED
            self.scale = config.rate
            self.aggregation_policy = config.aggregation

            self.prob_func_type = config.probability_function
            self.inflection_point = config.inflection_point

    def calibrate(self, cluster_prior: ClusterPrior):
        self.cluster_prior = cluster_prior

        def cost(z: jax.Array, scale: float):
            clusters = z[..., :-1]                                          # (n_obj, n_clust)
            cluster_size = jnp.sum(clusters, axis=-2)                       # (n_clust,)
            clusters_normed = clusters / cluster_size[None, :]              # (n_obj, n_clust)

            # Compute expected distance to a random language
            # same_cluster_prob = clusters_normed @ clusters.T                # ()
            # total_dist = jnp.sum(same_cluster_prob * self.cost_matrix)
            if self.aggregation_policy is self.AggrStrats.MEAN:
                same_cluster_prob = jnp.einsum("ik,jk->ijk", clusters_normed, clusters_normed)
            elif self.aggregation_policy is self.AggrStrats.SUM_OF_MEAN:
                same_cluster_prob = jnp.einsum("ik,jk->ijk", clusters_normed, clusters)
            else:
                raise ValueError(f'Invalid aggregation policy {self.aggregation_policy}.')

            total_dist_per_cluster = jnp.sum(same_cluster_prob * self.cost_matrix[..., None], axis=(0,1))
            c = self.probability_function(total_dist_per_cluster, scale)
            return c

        grid_size = self.config.approx_norm_const["grid_size"]
        grid_min = self.config.rate / 8.0
        grid_max = self.config.rate * 4.0
        r_grid = jnp.linspace(grid_min**0.5, grid_max**0.5, grid_size) ** 2

        self._norm_const_interpolator, _, _ = estimate_marginal_log_likelihood_curve(
            base_prior=lambda : self.cluster_prior.get_numpyro_distr(allow_reparameterization=False),
            log_g_fn=lambda z, s: cost(z, s),
            # r_grid=r_grid[::-1],
            r_grid=r_grid,
            num_samples=self.config.approx_norm_const["steps_per_setting"],
        )

        # Store the grid bounds for clamping in norm_const_function
        self._norm_const_grid_min = float(jnp.min(r_grid))
        self._norm_const_grid_max = float(jnp.max(r_grid))

    def norm_const_function(self, scale):
        # Clamp scale to interpolation grid bounds to avoid NaN from out-of-bounds extrapolation
        scale_clamped = jnp.clip(jnp.atleast_1d(scale), self._norm_const_grid_min, self._norm_const_grid_max)
        return self._norm_const_interpolator(scale_clamped)

    def probability_function(self, x: float, scale: float) -> float:
        x_agg = jnp.sum(x)
        if self.prob_func_type is GeoPriorConfig.ProbabilityFunction.EXPONENTIAL:
            return -x_agg / scale              # == log(e**(-x/scale))
        elif self.prob_func_type is GeoPriorConfig.ProbabilityFunction.GAMMA_EXPONENTIAL:
            # Hierarchical model:
            #   x_i | lambda_i ~ Exponential(rate=lambda_i)
            #   lambda_i ~ Gamma(shape=alpha, rate=beta)
            # Marginal for each x_i: p(x_i) = alpha * beta**alpha / (x_i + beta)**(alpha+1)
            # Log-pdf (summed over elements of x):
            alpha = 5.0
            beta = scale * (alpha - 1.0)
            # add small EPS for numerical stability
            return jnp.sum(jnp.log(alpha) + alpha * jnp.log(beta) - (alpha + 1.0) * jnp.log(x + beta))

        elif self.prob_func_type is GeoPriorConfig.ProbabilityFunction.SQUARED_EXPONENTIAL:
            return -(x_agg / scale)**2         # == log(e**(-(x/scale)**2))
        elif self.prob_func_type is GeoPriorConfig.ProbabilityFunction.SIGMOID:
            x0 = self.inflection_point
            return log_expit(-(x_agg - x0) / scale) - log_expit(x0 / scale)
            # return jnp.log(1E-100 + jax.scipy.special.expit(-(x - x0) / s))
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

        dist_mat = self.cost_matrix

        # distances, weights = self.compute_distances_along_skeleton(clusters)
        if self.config.skeleton is GeoPriorConfig.Skeleton.MST:
            aggregated_distance = 0.0
            for cluster in clusters.T:
                aggregated_distance += self.compute_fuzzy_mst_distance(cluster)
        elif self.config.skeleton is GeoPriorConfig.Skeleton.COMPLETE:
            clusters_normed = clusters / cluster_size[None, :]

            if self.aggregation_policy is GeoPrior.AggrStrats.MEAN:
                same_cluster_prob = jnp.einsum("ik,jk->ijk", clusters_normed, clusters_normed)
            elif self.aggregation_policy is GeoPrior.AggrStrats.SUM:
                same_cluster_prob = jnp.einsum("ik,jk->ijk", clusters, clusters)
            elif self.aggregation_policy is GeoPrior.AggrStrats.SUM_OF_MEAN:
                same_cluster_prob = jnp.einsum("ik,jk->ijk", clusters_normed, clusters)
            else:
                raise ValueError(f'Unknown aggregation policy `{self.aggregation_policy}`')
            aggregated_distance = jnp.sum(same_cluster_prob * self.cost_matrix[..., None], axis=(0,1))

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

        if self.config.estimate_rate:
            sigma = 1.0
            # mu = jnp.log(self.config.rate) - sigma * sigma / 2
            mu = jnp.log(self.config.rate)
            log_scale = numpyro.sample("geoprior_log_scale", dist.Normal(mu, sigma))
            # scale = numpyro.sample("geoprior_scale", dist.LogNormal(mu, sigma))
            scale = jnp.exp(log_scale)
            numpyro.deterministic("geoprior_scale", scale)
            norm_const = self.norm_const_function(scale)
        else:
            scale = self.config.rate
            norm_const = 1.0

        log_geo_priors = self.probability_function(aggregated_distance, scale)
        numpyro.factor("geoprior", log_geo_priors - norm_const)

        numpyro.deterministic("geoprior_total_dist", aggregated_distance)
        # for i_c in range(n_clusters):
        #     numpyro.deterministic(f"geo_dist_cluster_{i_c}", aggregated_distance[i_c])

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
            msg += f'\tProbability function: {prob_fun.value}\n'
            msg += f'\tAggregation policy: {self.aggregation_policy.value}\n'
            msg += f'\tScale: {self.scale}\n'
            if self.config['probability_function'] == 'sigmoid':
                msg += f'\tInflection point: {self.config["inflection_point"]}\n'
            if self.config['costs'] == 'from_data':
                msg += '\tCost-matrix inferred from geo-locations.\n'
            else:
                msg += f'\tCost-matrix file: {self.config["costs"]}\n'
        if self.config.estimate_rate:
            msg += f'\tEstimating geo-prior rate: {self.config.approx_norm_const}\n'

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


class PowerTransform(Transform):
    """A signed power transform: y = sign(x) * |x|^beta.

    This transform expands or compresses regions around zero depending on beta:
    - beta > 1: compresses values near zero, expands tails
    - beta < 1: expands values near zero, compresses tails

    For use with low-concentration Dirichlet priors, we want beta > 1 in the
    forward direction (from raw latent to stick-breaking latent), which means
    the inverse (beta < 1) expands the near-zero region in the raw space.
    """

    def __init__(self, beta: float = 2.0):
        """
        Args:
            beta: The power exponent. Values > 1 compress near-zero regions
                  in the forward direction.
        """
        self.beta = beta

    def __call__(self, x):
        """Forward transform: y = sign(x) * |x|^beta"""
        return jnp.sign(x) * jnp.abs(x) ** self.beta

    def _inverse(self, y):
        """Inverse transform: x = sign(y) * |y|^(1/beta)"""
        return jnp.sign(y) * jnp.abs(y) ** (1.0 / self.beta)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        """Log absolute determinant of the Jacobian.

        For y = sign(x) * |x|^beta, we have dy/dx = beta * |x|^(beta-1)
        3*x^2
        The log determinant is: sum(log(beta) + (beta-1) * log(|x|))
        """
        # Add small epsilon to avoid log(0) at x=0
        log_abs_x = jnp.log(jnp.abs(x) + 1e-10)
        log_det_per_dim = jnp.log(self.beta) + (self.beta - 1) * log_abs_x
        return jnp.sum(log_det_per_dim, axis=-1)

    @property
    def domain(self):
        return dist.constraints.real_vector

    @property
    def codomain(self):
        return dist.constraints.real_vector

    def tree_flatten(self):
        """Flatten the transform for JAX pytree compatibility."""
        return (self.beta,), (("beta",),)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        """Unflatten the transform from JAX pytree format."""
        (beta,) = params
        return cls(beta=beta)


class RadialPowerTransform(Transform):
    """A radial power transform: y = z * ||z||^(beta-1).

    This transform applies a power scaling to the radius (length) of the vector
    while preserving its direction. This avoids axis-aligned artifacts that occur
    with element-wise power transforms.

    - beta > 1: compresses vectors near the origin, expands those far from origin
    - beta < 1: expands vectors near the origin, compresses those far from origin

    The transform maps: z -> z * r^(beta-1) where r = ||z||
    Equivalently: the radius transforms as r -> r^beta while direction is preserved.
    """

    def __init__(self, beta: float = 2.0, eps: float = 1e-10):
        """
        Args:
            beta: The power exponent for the radius.
            eps: Small constant for numerical stability near origin.
        """
        self.beta = beta
        self.eps = eps

    def __call__(self, z):
        """Forward transform: y = z * ||z||^(beta-1)"""
        r = jnp.linalg.norm(z, axis=-1, keepdims=True)
        # For numerical stability, use r + eps in the power
        scale = (r + self.eps) ** (self.beta - 1)
        return z * scale

    def _inverse(self, y):
        """Inverse transform: z = y * ||y||^(1/beta - 1)"""
        r_y = jnp.linalg.norm(y, axis=-1, keepdims=True)
        # r_y = r_z^beta, so r_z = r_y^(1/beta)
        # scale factor: r_z / r_y = r_y^(1/beta) / r_y = r_y^(1/beta - 1)
        scale = (r_y + self.eps) ** (1.0 / self.beta - 1)
        return y * scale

    def log_abs_det_jacobian(self, z, y, intermediates=None):
        """Log absolute determinant of the Jacobian.

        For the transform y = z * r^(beta-1) where r = ||z||:

        The Jacobian matrix J has the form:
            J_ij = d(y_i)/d(z_j) = r^(beta-1) * delta_ij + (beta-1) * r^(beta-3) * z_i * z_j

        Using the matrix determinant lemma for (aI + b*uv^T):
            det(aI + b*uv^T) = a^(n-1) * (a + b*||u||^2)  when u=v

        Here a = r^(beta-1), b = (beta-1)*r^(beta-3), ||z||^2 = r^2
            det(J) = [r^(beta-1)]^(n-1) * [r^(beta-1) + (beta-1)*r^(beta-3)*r^2]
                   = r^((beta-1)*(n-1)) * [r^(beta-1) + (beta-1)*r^(beta-1)]
                   = r^((beta-1)*n) * beta

        So: log|det(J)| = n*(beta-1)*log(r) + log(beta)
        """
        n = z.shape[-1]
        r = jnp.linalg.norm(z, axis=-1)
        log_r = jnp.log(r + self.eps)
        return n * (self.beta - 1) * log_r + jnp.log(self.beta)

    @property
    def domain(self):
        return dist.constraints.real_vector

    @property
    def codomain(self):
        return dist.constraints.real_vector

    def tree_flatten(self):
        """Flatten the transform for JAX pytree compatibility."""
        return (self.beta, self.eps), (("beta", "eps"),)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        """Unflatten the transform from JAX pytree format."""
        beta, eps = params
        return cls(beta=beta, eps=eps)


class ArcsinhTransform(Transform):
    """Element-wise arcsinh transform: y = scale * arcsinh(x / scale).

    This transform compresses the tails while remaining approximately linear near zero.
    For |x| << scale: y ≈ x (identity)
    For |x| >> scale: y ≈ scale * sign(x) * log(2|x|/scale) (logarithmic compression)

    The scale parameter controls where the transition from linear to logarithmic occurs.
    Smaller scale = more compression of tails.
    """

    def __init__(self, scale: float = 1.0):
        """
        Args:
            scale: Controls the transition point between linear and log behavior.
                   Smaller values compress more aggressively.
        """
        self.scale = scale

    def __call__(self, x):
        """Forward transform: y = scale * arcsinh(x / scale)"""
        return self.scale * jnp.arcsinh(x / self.scale)

    def _inverse(self, y):
        """Inverse transform: x = scale * sinh(y / scale)"""
        return self.scale * jnp.sinh(y / self.scale)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        """Log absolute determinant of the Jacobian.

        For y_i = scale * arcsinh(x_i / scale):
            dy_i/dx_i = 1 / sqrt(1 + (x_i/scale)^2)

        Log det J = sum_i log(1 / sqrt(1 + (x_i/scale)^2))
                  = -0.5 * sum_i log(1 + (x_i/scale)^2)
        """
        return -0.5 * jnp.sum(jnp.log(1 + (x / self.scale) ** 2), axis=-1)

    @property
    def domain(self):
        return dist.constraints.real_vector

    @property
    def codomain(self):
        return dist.constraints.real_vector

    def tree_flatten(self):
        """Flatten the transform for JAX pytree compatibility."""
        return (self.scale,), (("scale",),)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        """Unflatten the transform from JAX pytree format."""
        (scale,) = params
        return cls(scale=scale)


class RadialArcsinhTransform(Transform):
    """Radial arcsinh transform: y = z * arcsinh(||z|| / scale) / (||z|| / scale).

    This transform applies arcsinh compression to the radius while preserving direction.
    It compresses the tails (large ||z||) while remaining approximately identity near origin.

    For ||z|| << scale: y ≈ z (identity)
    For ||z|| >> scale: y ≈ z * scale * log(2||z||/scale) / ||z|| (logarithmic compression)

    The scale parameter controls where the transition from linear to logarithmic occurs.
    """

    def __init__(self, scale: float = 1.0, eps: float = 1e-10):
        """
        Args:
            scale: Controls the transition point between linear and log behavior.
            eps: Small constant for numerical stability near origin.
        """
        self.scale = scale
        self.eps = eps

    def __call__(self, z):
        """Forward transform: y = z * arcsinh(r/scale) / (r/scale) where r = ||z||"""
        r = jnp.linalg.norm(z, axis=-1, keepdims=True)
        # For numerical stability near origin, use series expansion
        # arcsinh(u)/u ≈ 1 - u^2/6 + ... for small u
        u = r / self.scale
        # Use the stable form: arcsinh(u)/u, handling u->0
        ratio = jnp.where(
            u > self.eps,
            jnp.arcsinh(u) / (u + self.eps),
            1.0 - u ** 2 / 6.0  # Taylor expansion for small u
        )
        return z * ratio

    def _inverse(self, y):
        """Inverse transform: find z such that y = z * arcsinh(||z||/scale) / (||z||/scale)

        Since direction is preserved: z = y * ||z|| / ||y||
        And ||y|| = ||z|| * arcsinh(||z||/scale) / (||z||/scale)
        So we need to solve: ||y|| = arcsinh(r/scale) * scale, i.e., r = scale * sinh(||y||/scale)
        """
        r_y = jnp.linalg.norm(y, axis=-1, keepdims=True)
        # r_z = scale * sinh(r_y / scale)
        r_z = self.scale * jnp.sinh(r_y / self.scale)
        # z = y * (r_z / r_y)
        ratio = jnp.where(
            r_y > self.eps,
            r_z / (r_y + self.eps),
            1.0  # Near origin, transform is identity
        )
        return y * ratio

    def log_abs_det_jacobian(self, z, y, intermediates=None):
        """Log absolute determinant of the Jacobian.

        For y = z * f(r) where f(r) = arcsinh(r/s) / (r/s) and r = ||z||:

        The Jacobian has the form:
            J_ij = f(r) * delta_ij + f'(r) * z_i * z_j / r

        Using the matrix determinant lemma:
            det(J) = f(r)^(n-1) * (f(r) + f'(r) * r)

        For f(r) = arcsinh(r/s) / (r/s) = s * arcsinh(r/s) / r:
            f(r) = s * arcsinh(u) / r  where u = r/s
            f'(r) = d/dr [s * arcsinh(r/s) / r]
                  = s * [1/(s*sqrt(1+u^2)) * 1/r - arcsinh(u)/r^2]
                  = 1/(r*sqrt(1+u^2)) - s*arcsinh(u)/r^2

            f(r) + r*f'(r) = s*arcsinh(u)/r + 1/sqrt(1+u^2) - s*arcsinh(u)/r
                           = 1/sqrt(1+u^2)

        So: det(J) = f(r)^(n-1) * 1/sqrt(1+u^2)
                   = [s*arcsinh(u)/r]^(n-1) / sqrt(1+u^2)

        log|det(J)| = (n-1)*log(s*arcsinh(u)/r) - 0.5*log(1+u^2)
                    = (n-1)*[log(s) + log(arcsinh(u)) - log(r)] - 0.5*log(1+u^2)
        """
        n = z.shape[-1]
        r = jnp.linalg.norm(z, axis=-1)
        u = r / self.scale

        # Handle small r case where arcsinh(u)/u -> 1
        log_arcsinh_u = jnp.where(
            u > self.eps,
            jnp.log(jnp.arcsinh(u) + self.eps),
            jnp.log(u + self.eps) - u ** 2 / 6.0  # log(arcsinh(u)) ≈ log(u) for small u
        )
        log_r = jnp.log(r + self.eps)

        log_det = (n - 1) * (jnp.log(self.scale) + log_arcsinh_u - log_r) - 0.5 * jnp.log(1 + u ** 2)

        # For very small r, the transform is identity, so log_det -> 0
        log_det = jnp.where(r > self.eps, log_det, 0.0)

        return log_det

    @property
    def domain(self):
        return dist.constraints.real_vector

    @property
    def codomain(self):
        return dist.constraints.real_vector

    def tree_flatten(self):
        """Flatten the transform for JAX pytree compatibility."""
        return (self.scale, self.eps), (("scale", "eps"),)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        """Unflatten the transform from JAX pytree format."""
        scale, eps = params
        return cls(scale=scale, eps=eps)

def compute_adaptive_beta(concentration: jnp.array, min_beta: float = 1.0, max_beta: float = 4.0) -> float:
    """Compute an adaptive beta value based on the Dirichlet concentration parameter.

    For very low concentration (sparse Dirichlet), we want higher beta to compress
    the low-density interior region more aggressively. For moderate concentration,
    we need less compression.

    The scaling is based on the minimum concentration value:
    - alpha_min >= 1.0: beta = min_beta (no extra compression needed)
    - alpha_min -> 0: beta approaches max_beta

    Args:
        concentration: The Dirichlet concentration parameter array.
        min_beta: Minimum beta value (for alpha >= 1).
        max_beta: Maximum beta value (for very small alpha).

    Returns:
        Adaptive beta value.
    """
    alpha_min = jnp.min(concentration)

    # Use a smooth transition: beta = min_beta + (max_beta - min_beta) * exp(-k * alpha_min)
    # where k controls how quickly beta decreases as alpha increases
    k = 3.0  # Decay rate
    beta = min_beta + (max_beta - min_beta) * jnp.exp(-k * alpha_min)

    return float(beta)


def dirichlet_from_latent(
    name: str,
    concentration: jnp.array,
    offset=None,
    use_double_transform: bool = False,
) -> NDArray[float]:
    """Sample from a Dirichlet distribution using a latent space representation.

    This function samples from a uniform distribution in a latent space and transforms
    it to the probability simplex using a stick-breaking transform. Optionally, a
    power transform can be applied before the stick-breaking transform to improve
    sampling efficiency for low-concentration Dirichlet distributions.

    Args:
        name: The name for the numpyro sample site.
        concentration: The Dirichlet concentration parameter array.
        offset: Optional offset to add in latent space (on the simplex).
        use_double_transform: If True, apply a radial arcsinh transform before stick-breaking to
                improve sampling for low-concentration Dirichlet distributions.
    Returns:
        Sampled value on the probability simplex.
    """
    n_states = concentration.shape[-1]
    n_states_latent = n_states - 1

    # Sample from uniform distribution in latent space that spans wide enough to cover the tails
    x_latent_distr = dist.Uniform(-200, 200).expand((n_states_latent,)).to_event()
    z = numpyro.sample(f"{name}_raw", x_latent_distr)
    prior_correction_factor = -x_latent_distr.log_prob(z)

    # Define transforms
    stick_breaking = StickBreakingTransform()

    if use_double_transform:
        # Define the pre-transform with a fixed scale
        pretransform = RadialArcsinhTransform(scale=5.)

        # Apply radial arcsinh transform
        x_latent = pretransform(z)

        # Update the prior_concentration_factor to reflect 'squashing' by the transformation
        jacobian_power = pretransform.log_abs_det_jacobian(z, x_latent)
        prior_correction_factor += jacobian_power

    else:
        x_latent = z

    # If offset is provided, add it in latent space
    if offset is not None:
        offset_latent = stick_breaking.inv(offset)
        x_latent = x_latent + offset_latent

    # Transform to probability simplex
    x = stick_breaking(x_latent)

    # Define the dirichlet distribution on probability simplex
    x_distr = dist.Dirichlet(concentration)
    prior_log_prob = x_distr.log_prob(x)

    # Compute the correction factor to get the correct probability density on the simplex
    jacobian_stick = stick_breaking.log_abs_det_jacobian(x_latent, x)
    prior_correction_factor += jacobian_stick

    # Add the corrected log probability as a factor
    corrected_log_prob = prior_log_prob + prior_correction_factor
    numpyro.factor(f"{name}_log_prob", corrected_log_prob)

    # Add the sampled transformed value to the state
    numpyro.deterministic(name, x)

    return x


if __name__ == '__main__':
    import doctest
    doctest.testmod()
