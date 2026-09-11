from __future__ import annotations

import jax
import jax.numpy as jnp
import json
import numpy as np
import numpyro
import numpyro.distributions as dist

from numpyro.distributions.transforms import AffineTransform
from sbayes.config.config import PriorConfig, CategoricalPriorConfig, \
    ConfoundingEffectConfig, GaussianVariancePriorConfig, GaussianMeanPriorConfig, GaussianPriorConfig, \
    ClusterEffectConfig, PoissonPriorConfig, WeightsPriorConfig, ClusterPriorConfig
from sbayes.load_data import Data, GroupName, StateName, FeatureName, Confounder, \
    CategoricalFeatures, GaussianFeatures, GenericTypeFeatures, PoissonFeatures, FeatureType
from sbayes.model.reparameterization import dirichlet_from_latent
from sbayes.model.geo_prior import GeoPrior
from sbayes.model.model_shapes import ModelShapes
from sbayes.util import  FLOAT_TYPE, normalize

from typing import Sequence, TypeVar

T = TypeVar("T")

DEFAULT_GROUP = "<DEFAULT>"
"""Key of the fallback prior config, used for groups without their own config."""

def resolve_group_configs(
    config: dict[GroupName, T],
    group_names: Sequence[GroupName],
    feature_type: FeatureType,
) -> dict[GroupName, T]:
    """Resolve the prior config of each group, falling back to the `<DEFAULT>` entry.

    Args:
        config: the prior config per group, optionally with a `<DEFAULT>` entry
        group_names: the groups of the confounder
        feature_type: the feature type this prior applies to, used in the error message

    Returns:
        The config of every group in `group_names`, in that order.

    Raises:
        ValueError: if a group has no config of its own and no `<DEFAULT>` is defined.
    """
    default_config = config.get(DEFAULT_GROUP)
    resolved = {}
    for group in group_names:
        group_config = config.get(group, default_config)
        if group_config is None:
            raise ValueError(
                f"No {feature_type} prior defined for group '{group}'. Provide a prior for "
                f"every group or specify a `{DEFAULT_GROUP}` prior."
            )
        resolved[group] = group_config
    return resolved

def require_prior_config(
    config: T | None,
    partition: GenericTypeFeatures,
    section: str,
    group: GroupName | None = None,
) -> T:
    """Return the prior config of a feature type, or raise if none is defined.

    Args:
        config: the prior config for this feature type, or None if it is not defined
        partition: the features the prior applies to, used in the error message
        section: the config section the prior belongs to, e.g. "cluster_effect"
        group: the group the prior applies to, for confounding effects

    Returns:
        The prior config, guaranteed not to be None.

    Raises:
        ValueError: if no prior is defined for this feature type.
    """
    if config is None:
        where = f" in group '{group}'" if group else ""
        raise ValueError(
            f"No prior defined for {partition.name} features{where}. Add a "
            f"`{partition.FEATURE_TYPE}` section to `prior.{section}`."
        )
    return config

def invalid_prior_message(
    config: GaussianMeanPriorConfig | GaussianVariancePriorConfig
            | PoissonPriorConfig | CategoricalPriorConfig | ClusterPriorConfig,
    class_name: str,
) -> str:
    """Compile an error message for a prior type that is not implemented.

    Args:
        config: the prior config whose `type` is not supported
        class_name: the prior class that received it

    Returns:
        The error message, listing the types the class does support.
    """
    valid_types = ', '.join(t.value for t in config.Types)
    return (
        f"Prior type `{config.type}` is not implemented for {class_name} "
        f"(choose from: {valid_types})."
    )

def parse_concentration_dict(
    concentration_dict: dict[FeatureName, dict[StateName, float]],
    feature_names: dict[FeatureName, Sequence[StateName]],
) -> list[jnp.ndarray]:
    """Compile the concentration parameters of a custom Dirichlet prior.

    Args:
        concentration_dict: concentration parameter per feature and state
        feature_names: the states of each feature, in the order used by the model

    Returns:
        One array of concentration parameters per feature, ordered as in
        `feature_names`. States not listed in `feature_names` are ignored.

    Raises:
        ValueError: if a feature or one of its states is missing from
            `concentration_dict`.
    """
    concentration = []
    for f, state_names_f in feature_names.items():
        if f not in concentration_dict:
            raise ValueError(
                f"No concentration parameters defined for feature '{f}'."
            )
        missing = [s for s in state_names_f if s not in concentration_dict[f]]
        if missing:
            raise ValueError(
                f"No concentration parameters defined for feature '{f}', "
                f"state(s) {missing}."
            )
        conc_f = [concentration_dict[f][s] for s in state_names_f]
        concentration.append(jnp.array(conc_f, dtype=FLOAT_TYPE))

    return concentration

def parse_custom_concentration(
    config: CategoricalPriorConfig,
    feature_names: dict[FeatureName, Sequence[StateName]],
) -> list[jnp.ndarray]:
    """Load the concentration parameters of a custom Dirichlet prior.

    The parameters are read from a JSON file if `config.file` is set, otherwise
    directly from `config.parameters`.

    Args:
        config: the categorical prior configuration
        feature_names: the states of each feature, in the order used by the model

    Returns:
        One array of concentration parameters per feature.
    """
    if config.file:
        with open(config.file, 'r') as f:
            try:
                concentration_dict = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Could not parse the prior parameters in {config.file}: {e}"
                ) from e
    elif config.parameters:
        concentration_dict = config.parameters
    else:
        raise ValueError(
            f"A `{config.type.value}` prior requires either `file` or `parameters`."
        )

    return parse_concentration_dict(concentration_dict, feature_names)

def parse_dirichlet_concentration(
    config: CategoricalPriorConfig,
    shape: tuple[int, ...],
    feature_names: dict[FeatureName, Sequence[StateName]] | None = None,
) -> jnp.ndarray:
    """Parse the concentration parameter of a Dirichlet prior.

    Args:
        config: the categorical prior configuration
        shape: the shape of the concentration array for uniform and symmetric priors
        feature_names: the states of each feature; required for a custom `dirichlet`
            prior, ignored otherwise

    Returns:
        One concentration array per feature.
    """
    if config.type == CategoricalPriorConfig.Types.UNIFORM:
        return jnp.full(shape, 1.0)

    elif config.type == CategoricalPriorConfig.Types.SYMMETRIC_DIRICHLET:
        if config.prior_concentration is None:
            raise ValueError(
                f"A `{config.type.value}` prior requires `prior_concentration`."
            )
        return jnp.full(shape, config.prior_concentration)

    elif config.type == CategoricalPriorConfig.Types.DIRICHLET:
        if feature_names is None:
            raise ValueError(
                f"A `{config.type.value}` prior requires `feature_names` to map the "
                f"concentration parameters onto features and states."
            )
        concentration = jnp.stack(parse_custom_concentration(config, feature_names))
        if concentration.shape != shape:
            raise ValueError(
                f"The concentration parameters in the `{config.type.value}` prior have "
                f"shape {concentration.shape}, but the partition requires {shape}."
            )
        return concentration

    else:
        raise ValueError(f"Unsupported categorical prior type: {config.type}")


class Prior:
    """The joint prior of all parameters in the sBayes model.

    Attributes:
        cluster_prior (ClusterPrior): prior on the cluster assignment
        geo_prior (GeoPrior): prior on the geographic spread of a cluster
        weights_prior (WeightsPrior): prior on the mixture weights
        cluster_effect_prior (ClusterEffectPrior): prior on the areal effect
        confounding_effects_prior (CategoricalConfoundingEffectsPrior): prior on all confounding effects
    """

    def __init__(self, shapes: ModelShapes, config: PriorConfig, data: Data) -> None:
        """
        Args:
            shapes: shape information for building the prior components
            config: the prior configuration
            data: the data the model is fitted to
        """
        self.shapes = shapes
        self.config = config
        self.partitions = data.features.partitions

        self.cluster_prior = ClusterPrior(
            config=config.cluster_assignment,
            shapes=self.shapes
        )
        self.geo_prior = GeoPrior(
            config=config.geo,
            cost_matrix=data.geo_cost_matrix,
            network=data.network,
        )
        self.weights_prior = WeightsPrior(
            config=config.weights,
            shapes=self.shapes
        )
        self.cluster_effect_prior = ClusterEffectPrior(
            config=config.cluster_effect,
            partitions=self.partitions
        )
        self.confounding_effects_prior = {
            c: ConfoundingEffectsPrior(
                config=config.confounding_effects[c], conf=conf,
                partitions=self.partitions
            )
            for c, conf in data.confounders.items()
        }

    def get_setup_message(self) -> str:
        """Compile a set-up message for logging."""
        setup_msg = self.geo_prior.get_setup_message()
        setup_msg += self.cluster_prior.get_setup_message()
        setup_msg += self.weights_prior.get_setup_message()
        for partition in self.partitions:
            setup_msg += f"Priors for partition {partition.name}:\n"
            setup_msg += self.cluster_effect_prior[partition.name].get_setup_message()
            for conf_name, conf_prior in self.confounding_effects_prior.items():
                setup_msg += f"Confounder {conf_name}:\n"
                setup_msg += conf_prior[partition.name].get_setup_message()
        return setup_msg


class CategoricalConfoundingEffectsPrior:
    """The prior on confounding effects for categorical feature."""

    def __init__(
            self,
            config: dict[GroupName, CategoricalPriorConfig],
            conf: Confounder,
            partition: CategoricalFeatures,
    ) -> None:
        """
        Args:
            config: the prior configuration per group, optionally with a `<DEFAULT>`
                entry used for groups that have no configuration of their own
            conf: the confounder this prior applies to
            partition: the categorical features this prior applies to
        """
        self.config = config
        self.conf = conf
        self.partition = partition
        self.group_names = conf.group_names

        n_groups = len(self.group_names)
        concentration_array = np.zeros(
            (n_groups, partition.n_features, partition.n_states), dtype=FLOAT_TYPE
        )

        self.group_configs = resolve_group_configs(config, conf.group_names, partition.FEATURE_TYPE)
        for i_g, group_config in enumerate(self.group_configs.values()):
            concentration_array[i_g, ...] = parse_dirichlet_concentration(
                config=group_config,
                shape=(partition.n_features, partition.n_states),
                feature_names=partition.state_names_dict,
            )

        self.concentration_array = jnp.asarray(concentration_array)

        # The Dirichlet is sampled for all groups at once, so the transformation flag
        # has to be the same for every group of this confounder.
        flags = {g: cfg.use_parameter_transformation
                 for g, cfg in self.group_configs.items()}
        if len(set(flags.values())) > 1:
            raise ValueError(
                f"`use_parameter_transformation` must be the same for all groups of "
                f"confounder '{conf.name}', but got {flags}."
            )
        self.use_parameter_transformation = next(iter(flags.values()))

    def get_setup_message(self) -> str:
        """Compile a set-up message for logging."""
        msg = f"Prior on confounding effect {self.conf.name}:\n"
        for group, group_config in self.group_configs.items():
            msg += f"\tPrior for group {group}: {group_config.type.value}\n"
        return msg

    def get_numpyro_distr(self, allow_reparameterization: bool = False) -> jnp.ndarray:
        """Sample the state probabilities of each group of this confounder.

        Args:
            allow_reparameterization: if False, the Dirichlet is sampled directly
                instead of being constructed from latent parameters

        Returns:
            The state probabilities per group and feature.
            shape: (n_groups, n_features, n_states)
        """
        p_name = self.partition.name
        c_name = self.conf.name

        with numpyro.plate(f"plate_groups_{c_name}_{p_name}", self.conf.n_groups, dim=-2):
            with numpyro.plate(f"plate_features_{c_name}_{p_name}", self.partition.n_features, dim=-1):
                site_name = f"conf_effect_{c_name}_{p_name}"
                if allow_reparameterization and self.use_parameter_transformation:
                    conf_eff = dirichlet_from_latent(site_name, self.concentration_array)
                else:
                    conf_eff = numpyro.sample(
                        site_name, dist.Dirichlet(self.concentration_array)
                    )
        return jnp.asarray(conf_eff)


class CategoricalClusterEffectPrior:
    """The prior on cluster effects for categorical feature."""

    def __init__(
        self,
        config: CategoricalPriorConfig,
        partition: CategoricalFeatures,
    ) -> None:
        """
        Args:
            config: the prior configuration for this partition's cluster effect
            partition: the categorical features this prior applies to
        """
        self.config = config
        self.partition = partition
        self.prior_type = config.type

        if self.prior_type in (
            CategoricalPriorConfig.Types.UNIFORM,
            CategoricalPriorConfig.Types.DIRICHLET,
            CategoricalPriorConfig.Types.SYMMETRIC_DIRICHLET,
        ):
            self.concentration_array = parse_dirichlet_concentration(
                config=config,
                shape=(partition.n_features, partition.n_states),
                feature_names=partition.state_names_dict,
            )
        else:
            raise ValueError(
                f"Prior type `{self.prior_type.value}` is not implemented for "
                f"categorical cluster effects."
            )

        # One-hot encoded features as floats
        self._x_float = partition.to_binary().astype(FLOAT_TYPE)

    def get_setup_message(self) -> str:
        """Compile a set-up message for logging."""
        return (
            f"Prior on cluster effect for {self.partition.name} features: "
            f"{self.config.type.value}\n"
        )

    def get_data_dependent_contributions(
        self,
        clusters_weights: jnp.ndarray,   # shape: (n_clusters, n_objects, n_features)
        additive_smoothing: float = 0.5,
    ) -> jnp.ndarray:
        """Estimate the state probabilities of each cluster from the observed data.

        Used as the offset of the reparameterized cluster effect, so that sampling
        starts from a data-informed point rather than from the prior.

        Args:
            clusters_weights: how much each object contributes to each cluster
            additive_smoothing: pseudo-count added to every state, so that states not
                observed in a cluster keep a non-zero probability

        Returns:
            The estimated state probabilities per cluster and feature.
            shape: (n_clusters, n_features, n_states)
        """

        # Weighted count of each state in each cluster
        # (i: cluster, j: object, k: feature, l: state)
        feature_counts = jnp.einsum("ijk,jkl->ikl", clusters_weights, self._x_float)
        return normalize(feature_counts + additive_smoothing, axis=-1)

    def get_numpyro_distr(
        self,
        n_clusters: int,
        clust_eff_pred: jnp.ndarray,   # shape: (n_clusters, n_features, n_states)
        allow_reparameterization: bool = False,
    ) -> jnp.ndarray:
        """Sample the state probabilities of each cluster.

        Args:
            n_clusters: the number of clusters
            clust_eff_pred: data-dependent estimate of the state probabilities, used as
                the offset when the effect is reparameterized
            allow_reparameterization: if False, the Dirichlet is sampled directly
                instead of being constructed from latent parameters

        Returns:
            The state probabilities per cluster and feature.
            shape: (n_clusters, n_features, n_states)
        """
        p_name = self.partition.name
        site_name = f"cluster_effect_{p_name}"

        with numpyro.plate(f"plate_clusters_{p_name}", n_clusters, dim=-2):
            with numpyro.plate(f"plate_features_{p_name}", self.partition.n_features, dim=-1):
                if allow_reparameterization and self.config.use_parameter_transformation:
                    clust_eff = dirichlet_from_latent(
                        name=site_name,
                        concentration=self.concentration_array,
                        offset=clust_eff_pred,
                    )
                else:
                    clust_eff = numpyro.sample(
                        site_name, dist.Dirichlet(self.concentration_array)
                    )

        return jnp.asarray(clust_eff)


class GaussianMeanPrior:
    """The prior on the mean for Gaussian features."""

    def __init__(
        self,
        config: GaussianPriorConfig | dict[GroupName, GaussianPriorConfig],
        partition: GaussianFeatures,
        group_names: Sequence[GroupName] | None = None,
    ) -> None:
        """
        Args:
            config: the prior configuration, either a single config for a cluster effect
                or one config per group for a confounding effect
            partition: the Gaussian features this prior applies to
            group_names: the groups of the confounder; required if `config` is a dict
        """
        self.config = config
        self.partition = partition
        self.group_names = group_names

        if isinstance(config, GaussianPriorConfig):
            # Parse the prior mean and variance from config
            mu_0_array, sigma_0_array = self.parse_group_prior(config.mean)

        elif isinstance(config, dict):
            if group_names is None:
                raise ValueError(
                    "Group names are required for a Gaussian prior with one config "
                    "per group."
                )

            n_groups = len(group_names)
            mu_0_array = np.zeros((n_groups, partition.n_features), dtype=FLOAT_TYPE)
            sigma_0_array = np.zeros((n_groups, partition.n_features), dtype=FLOAT_TYPE)

            self.group_configs = resolve_group_configs(config, group_names, partition.FEATURE_TYPE)

            # Parse the prior mean and variance for each group from its own config
            for i_g, group_config in enumerate(self.group_configs.values()):
                mu_0_array[i_g, :], sigma_0_array[i_g, :] = self.parse_group_prior(
                    group_config.mean
                )

        else:
            raise ValueError(f"Invalid Gaussian prior config: {config}")

        self.mu_0_array = jnp.asarray(mu_0_array, dtype=FLOAT_TYPE)
        self.sigma_0_array = jnp.asarray(sigma_0_array, dtype=FLOAT_TYPE)


    def parse_group_prior(
        self, config: GaussianMeanPriorConfig
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Parse the mean and standard deviation of the prior on the Gaussian mean.

        Args:
            config: the prior configuration of one group or of the cluster effect

        Returns:
            The prior mean and standard deviation, one value per feature.
            shape of each: (n_features)
        """
        if config.type is not config.Types.GAUSSIAN:
            raise ValueError(invalid_prior_message(config, type(self).__name__))

        n_features = self.partition.n_features
        try:
            mu_0 = config.parameters['mu_0']
            sigma_0 = config.parameters['sigma_0']
        except KeyError as e:
            raise ValueError(
                f"A `gaussian` prior on the mean requires the parameters `mu_0` and "
                f"`sigma_0`, but {e} is missing."
            ) from e

        mu_0_array = jnp.full(n_features, mu_0)
        sigma_0_array = jnp.full(n_features, sigma_0)
        return mu_0_array, sigma_0_array

    def get_data_dependent_contributions(
        self,
        clusters_weights: jnp.ndarray,   # shape: (n_clusters, n_objects, n_features)
    ) -> jnp.ndarray:
        """Estimate the mean of each cluster from the observed data.

        Computes the posterior mean under the conjugate normal-normal update, using the
        cluster weights as observation weights. Used as the offset of the
        reparameterized cluster effect.

        Args:
            clusters_weights: how much each object contributes to each cluster

        Returns:
            The estimated mean per cluster and feature.
            shape: (n_clusters, n_features)
        """
        # Missing values contribute neither to the sum nor to the counts
        x = jnp.where(self.partition.na_values, 0.0, self.partition.values)
        observed = ~self.partition.na_values
        # shape of both: (n_objects, n_features)

        weighted_sum = jnp.sum(clusters_weights * x[None, :, :], axis=1)
        observation_counts = jnp.sum(clusters_weights * observed[None, :, :], axis=1)

        prec_0 = 1.0 / self.sigma_0_array ** 2
        return (self.mu_0_array * prec_0 + weighted_sum) / (prec_0 + observation_counts)

    def get_numpyro_distr(
        self,
        n_clusters: int,
        clust_eff_mean_pred: jnp.ndarray,   # shape: (n_clusters, n_features)
        allow_reparameterization: bool = False
    ) -> jnp.ndarray:
        """Sample the mean of each cluster.

        Args:
            n_clusters: the number of clusters
            clust_eff_mean_pred: data-dependent estimate of the cluster means, used as
                the offset when the effect is reparameterized
            allow_reparameterization: if False, the Gaussian is sampled directly rather
                than through an affine reparameterization
        Returns:
            The mean per cluster and feature. shape: (n_clusters, n_features)
        """
        p_name = self.partition.name
        mean_dist = dist.Normal(self.mu_0_array, self.sigma_0_array)

        if not allow_reparameterization:
            with numpyro.plate(f"plate_clusters_{p_name}", n_clusters, dim=-2):
                with numpyro.plate(f"plate_features_{p_name}", self.partition.n_features, dim=-1):
                    return jnp.asarray(numpyro.sample(f"cluster_effect_{p_name}_mean", mean_dist))

        # Reparameterized: sample a bounded offset around the data-dependent estimate
        # instead of the mean itself, which keeps the sampler in a well-scaled space.
        offset_dist = dist.Uniform(-1, 1)
        with numpyro.plate(f"plate_clusters_{p_name}", n_clusters, dim=-2):
            with numpyro.plate(f"plate_features_{p_name}", self.partition.n_features, dim=-1):
                offset_latent = self.sigma_0_array * numpyro.sample(
                    f"cluster_effect_{p_name}_mean_offset", offset_dist
                )

        # Widen the offset to +-10 prior standard deviations
        trans = AffineTransform(loc=0.0, scale=10.0)
        offset = trans(offset_latent)
        effect_mean = clust_eff_mean_pred + offset
        numpyro.deterministic(f"cluster_effect_{p_name}_mean", effect_mean)

        # The latent parameterization is uniform, so the actual prior density has to be
        # added explicitly, corrected for the transform and the latent's own density.
        prior_log_prob = mean_dist.log_prob(effect_mean)
        prior_correction_factor = (
            trans.log_abs_det_jacobian(offset_latent, offset)
            - offset_dist.log_prob(offset_latent)
        )
        numpyro.factor(
            f"cluster_effect_{p_name}_mean_log_prob",
            prior_log_prob + prior_correction_factor,
        )

        return jnp.asarray(effect_mean)


class GaussianVariancePrior:
    """The prior on the variance for Gaussian features."""

    def __init__(
        self,
        config: GaussianPriorConfig | dict[GroupName, GaussianPriorConfig],
        partition: GaussianFeatures,
        group_names: Sequence[GroupName] | None = None,
    ) -> None:
        """
        Args:
            config: the prior configuration, either a single config for a cluster effect
                or one config per group for a confounding effect
            partition: the Gaussian features this prior applies to
            group_names: the groups of the confounder; required if `config` is a dict

        The parsed parameters are stored in `self.parameters`, whose shape depends on
        `self.prior_type`: (n_features,) for `exponential` and `fixed`, and
        (2, n_features) for `gamma`, with one leading group axis in the dict case.
        """
        self.config = config
        self.partition = partition
        self.group_names = group_names

        if isinstance(config, GaussianPriorConfig):
            self.prior_type = config.variance.type
            self.parameters = self.parse_group_prior(config.variance)

        elif isinstance(config, dict):
            if group_names is None:
                raise ValueError(
                    "Group names are required for a Gaussian prior with one config "
                    "per group."
                )
            self.group_configs = resolve_group_configs(
                config, group_names, partition.FEATURE_TYPE
            )
            variance_configs = [c.variance for c in self.group_configs.values()]

            # All groups are sampled from a single distribution, so they have to share
            # the same prior type.
            types = {cfg.type for cfg in variance_configs}
            if len(types) > 1:
                raise ValueError(
                    f"All groups of a confounder must use the same variance prior type, "
                    f"but got {sorted(t.value for t in types)}."
                )
            self.prior_type = types.pop()

            group_parameters = jnp.stack(
                [self.parse_group_prior(cfg) for cfg in variance_configs]
            )
            # shape: (n_groups, n_features) or (n_groups, 2, n_features) for gamma
            if self.prior_type is GaussianVariancePriorConfig.Types.GAMMA:
                group_parameters = group_parameters.transpose((1, 0, 2))
            self.parameters = group_parameters

        else:
            raise ValueError(f"Invalid Gaussian prior config: {config}")

    def parse_group_prior(
        self, config: GaussianVariancePriorConfig
    ) -> jnp.ndarray:
        """Parse the parameters of the prior on the Gaussian variance.

        Args:
            config: the variance prior configuration of one group or of the cluster effect

        Returns:
            The prior parameters, one value per feature: shape (n_features,) for
            `exponential` and `fixed`, and (2, n_features) for `gamma`, holding the
            shape and rate.
        """
        n_features = self.partition.n_features
        try:
            if config.type is config.Types.EXPONENTIAL:
                return jnp.full(n_features, config.parameters['rate'])
            elif config.type is config.Types.GAMMA:
                return jnp.stack([
                    jnp.full(n_features, config.parameters['shape']),
                    jnp.full(n_features, config.parameters['rate']),
                ])
            elif config.type is config.Types.FIXED:
                return jnp.full(n_features, config.parameters['value'])
            else:
                raise ValueError(invalid_prior_message(config, type(self).__name__))
        except KeyError as e:
            raise ValueError(
                f"A `{config.type.value}` prior on the variance is missing the "
                f"parameter {e}."
            ) from e

    def get_numpyro_distr(self) -> dist.Distribution:
        """Return the prior distribution on the variance of the Gaussian features.

        Unlike most `get_numpyro_distr` methods in this module, this returns a
        distribution rather than a sample: the caller declares the plates and samples
        from it.

        Returns:
            The prior distribution, batched over features (and groups, for a
            confounding effect).
        """

        if self.prior_type is GaussianVariancePriorConfig.Types.EXPONENTIAL:
            return dist.Exponential(rate=self.parameters)
        elif self.prior_type is GaussianVariancePriorConfig.Types.GAMMA:
            return dist.Gamma(concentration=self.parameters[0], rate=self.parameters[1])
        elif self.prior_type is GaussianVariancePriorConfig.Types.FIXED:
            return dist.Delta(v=self.parameters)
        else:
            valid_types = ', '.join(t.value for t in GaussianVariancePriorConfig.Types)
            raise ValueError(
                "Prior type `%s` is not implemented for %s (choose from: %s)."
                % (self.prior_type.value, type(self).__name__, valid_types)
            )


class GaussianConfoundingEffectsPrior:

    """Prior on the confounding effects of a partition of Gaussian features."""

    def __init__(
        self,
        config: dict[GroupName, GaussianPriorConfig],
        conf: Confounder,
        partition: GaussianFeatures,
    ) -> None:
        """
        Args:
            config: the prior configuration per group, optionally with a `<DEFAULT>` entry
            conf: the confounder this prior applies to
            partition: the Gaussian features this prior applies to
        """
        self.config = config
        self.conf = conf
        self.partition = partition
        self.mean = GaussianMeanPrior(
            config=config, partition=partition, group_names=conf.group_names
        )
        self.variance = GaussianVariancePrior(
            config=config, partition=partition, group_names=conf.group_names
        )

    def get_setup_message(self) -> str:
        """Compile a set-up message for logging."""
        msg = (
            f"Prior on confounding effect {self.conf.name} for "
            f"{self.partition.name} features:\n"
        )
        for group, group_config in self.mean.group_configs.items():
            msg += (
                f"\tPrior for group {group}: "
                f"(mean={group_config.mean.type.value}, "
                f"variance={group_config.variance.type.value}).\n"
            )
        return msg


class GaussianClusterEffectPrior:

    """Prior on the cluster effect of a partition of Gaussian features."""

    def __init__(
        self,
        config: GaussianPriorConfig,
        partition: GaussianFeatures,
    ) -> None:
        """
        Args:
            config: the prior configuration for this partition's cluster effect
            partition: the Gaussian features this prior applies to
        """
        self.config = config
        self.partition = partition
        self.mean = GaussianMeanPrior(config=config, partition=partition)
        self.variance = GaussianVariancePrior(config=config, partition=partition)

    def get_setup_message(self) -> str:
        """Compile a set-up message for logging."""
        return (
            f"Prior on cluster effect for {self.partition.name} features: "
            f"(mean={self.config.mean.type.value}, "
            f"variance={self.config.variance.type.value})\n"
        )


class PoissonRatePrior:

    """Prior on the rate of a partition of Poisson features."""

    def __init__(
        self,
        config: PoissonPriorConfig | dict[GroupName, PoissonPriorConfig],
        partition: PoissonFeatures,
        group_names: Sequence[GroupName] | None = None,
    ) -> None:
        """
        Args:
            config: the prior configuration, either a single config for a cluster effect
                or one config per group for a confounding effect
            partition: the Poisson features this prior applies to
            group_names: the groups of the confounder; required if `config` is a dict
        """
        self.config = config
        self.partition = partition
        self.group_names = group_names

        if isinstance(config, PoissonPriorConfig):
            self.prior_type = config.type
            self.parameters = self.parse_group_prior(config)

        elif isinstance(config, dict):
            if group_names is None:
                raise ValueError(
                    "Group names are required for a Poisson prior with one config "
                    "per group."
                )
            self.group_configs = resolve_group_configs(
                config, group_names, partition.FEATURE_TYPE
            )
            rate_configs = list(self.group_configs.values())

            # All groups are sampled from a single distribution, so they have to share
            # the same prior type.
            types = {cfg.type for cfg in rate_configs}
            if len(types) > 1:
                raise ValueError(
                    f"All groups of a confounder must use the same Poisson prior type, "
                    f"but got {sorted(t.value for t in types)}."
                )
            self.prior_type = types.pop()

            group_parameters = jnp.stack(
                [self.parse_group_prior(cfg) for cfg in rate_configs]
            )
            # shape: (n_groups, 2, n_features)
            self.parameters = group_parameters.transpose((1, 0, 2))

        else:
            raise ValueError(f"Invalid Poisson prior config: {config}")

    def parse_group_prior(self, config: PoissonPriorConfig) -> jnp.ndarray:
        """Parse the parameters of the prior on the Poisson rate.

        Args:
            config: the prior configuration of one group or of the cluster effect

        Returns:
            The shape and rate of the Gamma prior. shape: (2, n_features)
        """
        if config.type is not config.Types.GAMMA:
            raise ValueError(invalid_prior_message(config, type(self).__name__))

        n_features = self.partition.n_features
        try:
            return jnp.stack([
                jnp.full(n_features, config.parameters['shape']),
                jnp.full(n_features, config.parameters['rate']),
            ])
        except KeyError as e:
            raise ValueError(
                f"A `{config.type.value}` prior on the Poisson rate is missing the "
                f"parameter {e}."
            ) from e


    def get_numpyro_distr(self) -> dist.Distribution:
        """Return the prior distribution on the Poisson rate."""
        return dist.Gamma(
            concentration=self.parameters[0], rate=self.parameters[1]
        )


class PoissonConfoundingEffectsPrior:

    """Prior on the confounding effects of a partition of Poisson features."""

    def __init__(
        self,
        config: dict[GroupName, PoissonPriorConfig],
        conf: Confounder,
        partition: PoissonFeatures,
    ) -> None:
        """
        Args:
            config: the prior configuration per group, optionally with a `<DEFAULT>` entry
            conf: the confounder this prior applies to
            partition: the Poisson features this prior applies to
        """
        self.config = config
        self.conf = conf
        self.partition = partition
        self.rate = PoissonRatePrior(
            config=config, partition=partition, group_names=conf.group_names
        )

    def get_setup_message(self) -> str:
        """Compile a set-up message for logging."""
        msg = (
            f"Prior on confounding effect {self.conf.name} for "
            f"{self.partition.name} features:\n"
        )
        for group, group_config in self.rate.group_configs.items():
            msg += f"\tPrior for group {group}: (rate={group_config.type.value}).\n"
        return msg


class PoissonClusterEffectPrior:

    """Prior on the cluster effect of a partition of Poisson features."""

    def __init__(
        self,
        config: PoissonPriorConfig,
        partition: PoissonFeatures,
    ) -> None:
        """
        Args:
            config: the prior configuration for this partition's cluster effect
            partition: the Poisson features this prior applies to
        """
        self.config = config
        self.partition = partition
        self.rate = PoissonRatePrior(config=config, partition=partition)


    def get_setup_message(self) -> str:
        """Compile a set-up message for logging."""
        return (
            f"Prior on cluster effect for {self.partition.name} features: "
            f"(rate={self.config.type.value})\n"
        )


class ClusterEffectPrior:

    """Priors on the cluster effects, one per partition."""

    def __init__(
        self,
        config: ClusterEffectConfig,
        partitions: list[GenericTypeFeatures],
    ) -> None:
        """
        Args:
            config: the cluster effect prior configuration per feature type
            partitions: the feature partitions of the data
        """
        self.config = config
        self.partition_priors = {}

        # Create a prior for each partition. Note: LogitNormalFeatures subclasses
        # GaussianFeatures and is handled by the Gaussian branch.
        for p in partitions:
            if isinstance(p, CategoricalFeatures):
                self.partition_priors[p.name] = CategoricalClusterEffectPrior(
                    require_prior_config(config.categorical, p, "cluster_effect"), p
                )
            elif isinstance(p, GaussianFeatures):
                self.partition_priors[p.name] = GaussianClusterEffectPrior(
                    require_prior_config(config.gaussian, p, "cluster_effect"), p
                )
            elif isinstance(p, PoissonFeatures):
                self.partition_priors[p.name] = PoissonClusterEffectPrior(
                    require_prior_config(config.poisson, p, "cluster_effect"), p
                )
            else:
                raise NotImplementedError(
                    f"Partition type {type(p).__name__} is not supported."
                )

    def __getitem__(self, partition_name: str):
        return self.partition_priors[partition_name]

    def get_setup_message(self) -> str:
        """Compile a set-up message for logging."""
        return "".join(
            prior.get_setup_message() for prior in self.partition_priors.values()
        )

class ConfoundingEffectsPrior:

    """Priors on the confounding effects of one confounder, one per partition."""

    def __init__(
        self,
        config: dict[GroupName, ConfoundingEffectConfig],
        conf: Confounder,
        partitions: list[GenericTypeFeatures],
    ) -> None:
        """
        Args:
            config: the confounding effect prior configuration per group
            conf: the confounder these priors apply to
            partitions: the feature partitions of the data
        """
        self.config = config
        self.partition_priors = {}

        # Create a prior for each partition. Note: LogitNormalFeatures subclasses
        # GaussianFeatures and is handled by the Gaussian branch.
        for p in partitions:
            if isinstance(p, CategoricalFeatures):
                categorical_configs = {
                    g: require_prior_config(c.categorical, p, "confounding_effects", g)
                    for g, c in config.items()
                }
                self.partition_priors[p.name] = CategoricalConfoundingEffectsPrior(
                    categorical_configs, conf, p
                )
            elif isinstance(p, GaussianFeatures):
                gaussian_configs = {
                    g: require_prior_config(c.gaussian, p, "confounding_effects", g)
                    for g, c in config.items()
                }
                self.partition_priors[p.name] = GaussianConfoundingEffectsPrior(
                    gaussian_configs, conf, p
                )
            elif isinstance(p, PoissonFeatures):
                poisson_configs = {
                    g: require_prior_config(c.poisson, p, "confounding_effects", g)
                    for g, c in config.items()
                }
                self.partition_priors[p.name] = PoissonConfoundingEffectsPrior(
                    poisson_configs, conf, p
                )
            else:
                raise NotImplementedError(
                    f"Partition type {type(p).__name__} is not supported."
                )

    def __getitem__(self, partition_name: str):
        return self.partition_priors[partition_name]

    def get_setup_message(self) -> str:
        """Compile a set-up message for logging."""
        return "".join(
            prior.get_setup_message() for prior in self.partition_priors.values()
        )


class WeightsPrior:

    """Prior on the mixture weights of the components."""

    def __init__(self, config: WeightsPriorConfig, shapes: ModelShapes) -> None:
        """
        Args:
            config: the prior configuration for the weights
            shapes: shape information of the model
        """
        self.config = config
        self.shapes = shapes
        self.concentration_array = parse_dirichlet_concentration(
            config=config,
            shape=(shapes.n_features, shapes.n_components),
        )

    def get_setup_message(self) -> str:
        """Compile a set-up message for logging."""
        return f"Prior on weights: {self.config.type.value}\n"


class ClusterPrior:

    def __init__(self, config: ClusterPriorConfig, shapes: ModelShapes) -> None:
        """
        Args:
            config: the prior configuration for the cluster assignments
            shapes: shape information of the model
        """
        self.config = config
        self.shapes = shapes
        self.prior_type = config.type

        self.concentration = None
        self.logi_norm_loc = None
        self.logi_norm_scale = None
        self.use_parameter_transformation = False

        if self.prior_type is ClusterPriorConfig.Types.CATEGORICAL:
            pass
        elif self.prior_type is ClusterPriorConfig.Types.DIRICHLET:
            if config.dirichlet_config is None:
                raise ValueError(
                    "A `dirichlet` cluster prior requires a `dirichlet_config` section "
                    "in `prior.cluster_assignment`."
                )
            self.concentration = parse_dirichlet_concentration(
                config=config.dirichlet_config,
                shape=(shapes.n_clusters + 1,),
            )
            self.use_parameter_transformation = (
                config.dirichlet_config.use_parameter_transformation
            )

        elif self.prior_type is ClusterPriorConfig.Types.LOGISTIC_NORMAL:
            if config.logistic_normal_config is None:
                raise ValueError(
                    "A `logistic_normal` cluster prior requires a "
                    "`logistic_normal_config` section in `prior.cluster_assignment`."
                )
            self.logi_norm_loc = config.logistic_normal_config.loc
            self.logi_norm_scale = config.logistic_normal_config.scale
        else:
            raise ValueError(
                invalid_prior_message(config, type(self).__name__)
            )

    def get_setup_message(self) -> str:
        """Compile a set-up message for logging."""
        msg = f"Prior on cluster assignment: {self.prior_type.value}\n"

        if self.prior_type is ClusterPriorConfig.Types.DIRICHLET:
            if self.config.hierarchical:
                msg += "\tEstimate cluster prior concentration\n"
            else:
                msg += (
                    f"\tFixed cluster prior concentration at "
                    f"c={self.concentration[0]}\n"
                )
            if self.config.estimate_no_cluster_concentration:
                msg += "\tEstimate non-cluster concentration\n"

        elif self.prior_type is ClusterPriorConfig.Types.LOGISTIC_NORMAL:
            if self.config.hierarchical:
                msg += "\tEstimate logistic normal scale\n"
            else:
                msg += f"\tFixed logistic normal scale at {self.logi_norm_scale}\n"

        return msg

    @staticmethod
    def _stretch_and_clip(z: jnp.ndarray) -> jnp.ndarray:
        """Inflate the no-cluster component and rescale the clusters accordingly.

        The no-cluster probability is multiplied by a sampled factor `s >= 1` and the
        added mass is subtracted from the cluster components, proportionally to their
        current size. The result is clipped away from 0 and 1 and renormalized, so that
        it stays a valid simplex.

        This shifts the prior towards smaller clusters without changing the relative
        weights of the clusters themselves.

        Args:
            z: cluster assignments with the no-cluster component last.
                shape: (n_objects, n_clusters + 1)

        Returns:
            The stretched assignments, same shape.
        """
        s = 1 + numpyro.sample("z0_stretch_factor", dist.Exponential(1.0))
        z_stretched = z.at[:, -1].multiply(s)

        # Subtract the added no-cluster mass from the clusters, proportionally
        d = z_stretched[:, -1:] - z[:, -1:]
        cluster_share = z[:, :-1] / (jnp.sum(z[:, :-1], axis=-1, keepdims=True) + 1E-6)
        z_stretched = z_stretched.at[:, :-1].subtract(cluster_share * d)

        z_stretched = jnp.clip(z_stretched, 1E-9, 1 - 1E-9)
        return normalize(z_stretched, axis=-1)

    def get_numpyro_distr(self, allow_reparameterization: bool = False) -> jnp.ndarray:
        """Sample the cluster assignment of each object.

        Args:
            allow_reparameterization: if False, the Dirichlet is sampled directly
                instead of being constructed from latent parameters

        Returns:
            The assignment of each object to each cluster, with a dummy "no-cluster"
            component in the last column. shape: (n_objects, n_clusters + 1)
        """

        n_clusters = self.shapes.n_clusters

        if self.prior_type is ClusterPriorConfig.Types.CATEGORICAL:
            # Note: `z` is a hard one-hot assignment here, which is not differentiable
            # and can therefore not be sampled by NUTS.
            with numpyro.plate("plate_objects_z", self.shapes.n_objects, dim=-1):
                z_int = numpyro.sample(
                    "z_int", dist.Categorical(jnp.ones(n_clusters + 1) / (n_clusters + 1))
                )
                z = jax.nn.one_hot(z_int, n_clusters + 1)

        elif self.prior_type is ClusterPriorConfig.Types.DIRICHLET:
            if self.config.hierarchical:
                c = numpyro.sample("z_concentration", dist.Uniform(0, 1))
                concentration = jnp.full((n_clusters + 1,), c)
            else:
                concentration = self.concentration

            if self.config.estimate_no_cluster_concentration:
                c_nocluster = numpyro.sample(
                    "z_concentration_nocluster", dist.LogNormal(0.0, 1.0)
                )
            else:
                c_nocluster = self.config.no_cluster_concentration

            if c_nocluster is not None:
                concentration = concentration.at[-1].set(c_nocluster)

            with numpyro.plate("plate_objects_z", self.shapes.n_objects, dim=-1):
                if allow_reparameterization and self.use_parameter_transformation:
                    z = dirichlet_from_latent(
                        "z_unstretched", concentration, offset=normalize(concentration)
                    )
                else:
                    z = numpyro.sample("z_unstretched", dist.Dirichlet(concentration))

        elif self.prior_type is ClusterPriorConfig.Types.LOGISTIC_NORMAL:
            if self.config.hierarchical:
                scale = numpyro.sample("z_concentration", dist.LogNormal(0.0, 1.0))
            else:
                scale = self.logi_norm_scale

            with numpyro.plate("plate_objects_z", self.shapes.n_objects, dim=-2):
                with numpyro.plate("plate_clusters_z", n_clusters + 1, dim=-1):
                    z_raw = numpyro.sample("z_raw", dist.Normal(self.logi_norm_loc, 1.0))

            z = jax.nn.softmax(z_raw * scale, axis=-1)

        else:
            raise ValueError(invalid_prior_message(self.config, type(self).__name__))

        if self.config.stretch_and_clip:
            z = self._stretch_and_clip(jnp.asarray(z))

        numpyro.deterministic("z", z)
        return jnp.asarray(z)

