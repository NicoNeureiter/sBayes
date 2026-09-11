#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from jax.typing import ArrayLike
from numpyro.infer import init_to_median
from numpyro.infer.util import initialize_model, constrain_fn
from sbayes.load_data import Data, CategoricalFeatures, GaussianFeatures, PoissonFeatures
from sbayes.config.config import ModelConfig
from sbayes.model.model_shapes import ModelShapes
from sbayes.model.prior import (
    Prior,
    GaussianConfoundingEffectsPrior,
    PoissonConfoundingEffectsPrior,
    PoissonClusterEffectPrior,
    dirichlet_from_latent,
)
from sbayes.util import onehot_to_integer_encoding, FLOAT_TYPE, EPS


NO_GROUP = -1
CLUSTER_COMPONENT = 0

def indent(text, amount, ch=' '):
    padding = amount * ch
    return ''.join(padding + line for line in text.splitlines(True))


def normalize(x: ArrayLike, axis: int = -1) -> jnp.ndarray:
    """Normalize `x` so that it sums up to 1 along the given axis.

    Args:
        x: array to be normalized
        axis: the axis to be normalized (will sum up to 1)

    Returns:
        `x` normalized along `axis`

    Note:
        No check is done for zero sums: an all-zero slice yields NaN.
    """
    x = jnp.asarray(x)
    return x / jnp.sum(x, axis=axis, keepdims=True)


class Model:
    """The sBayes model: posterior distribution of clusters and parameters.

    Attributes:
        data (Data): The data used in the likelihood
        config (ModelConfig): A dictionary containing configuration parameters of the model
        confounders (dict): A dict of all confounders and group names
        shapes (sbayes.model.ModelShapes): A dictionary with shape information for building the Likelihood and Prior objects
        prior (Prior): The prior of the model

    """

    def __init__(self, data: Data, config: ModelConfig) -> None:
        """
        Args:
            data: the data the model is fitted to
            config: the model configuration, for a single number of clusters
        """
        if not isinstance(config.clusters, int):
            raise ValueError(
                f"Model requires a single number of clusters, but the config specifies "
                f"{config.clusters}. cli.py splits runs over multiple cluster counts."
            )

        self.data = data
        self.config = config
        self.confounders = data.confounders

        if not self.confounders:
            raise ValueError("The model requires at least one confounder.")

        n_objects, n_features = self.data.features.all_features.shape

        self.shapes = ModelShapes(
            n_clusters=config.clusters,
            n_objects=n_objects,
            n_features=n_features,
            n_confounders=len(self.confounders),
            n_groups={name: conf.n_groups for name, conf in self.confounders.items()},
        )


        # Initialise the prior
        self.prior = Prior(shapes=self.shapes, config=self.config.prior, data=data)

        # Group names and group assignments of the confounders. The group assignments are
        # translated from binary (one-hot) to integer group indices, with NO_GROUP for
        # objects that belong to no group of a confounder.
        self.group_names = [conf.group_names for conf in self.confounders.values()]
        self.group_assignments = jnp.stack([
            onehot_to_integer_encoding(conf.group_assignment, none_index=NO_GROUP, axis=0)
            for conf in self.confounders.values()
        ])

        # Whether each object belongs to any group of each confounder.
        # shape: (n_confounders, n_objects)
        self._has_confounder_component = (self.group_assignments != NO_GROUP).astype(float)

        self.w_prior_conc = jnp.asarray(
            self.prior.weights_prior.concentration_array, dtype=FLOAT_TYPE
        )

        self.partitions = self.data.features.partitions

    def calibrate(self) -> None:
        """Run any calibration procedures that are required before the MCMC run."""
        self.prior.geo_prior.calibrate(self.prior.cluster_prior)

    @property
    def n_clusters(self) -> int:
        """The number of clusters in this model."""
        return self.shapes.n_clusters

    @property
    def allow_reparameterization(self) -> bool:
        """Whether parameters may be sampled through a latent reparameterization.

        The reparameterization is not supported when sampling from the prior, where the
        parameters are sampled from their prior distributions directly.
        """
        return not self.config.sample_from_prior

    def get_model(self) -> None:
        """Define the sBayes model for NumPyro.

        Samples the cluster assignments, the mixture weights and the effect parameters of
        each partition, and conditions them on the observed features. Nothing is returned:
        the model is defined through the side effects of the `numpyro.sample` calls.
        """

        # Add the cluster prior to the model
        z = self.prior.cluster_prior.get_numpyro_distr(allow_reparameterization=self.allow_reparameterization)

        # `z` has a dummy "no-cluster" at the end, which we want to remove for most purposes
        clusters = z[..., :-1]

        # Add geo-prior as a factor to the model
        self.prior.geo_prior.add_geo_prior(clusters)

        # Participation of each object in each component. Row 0 is the cluster component:
        # `z[..., -1]` is the dummy "no-cluster" probability, so `1 - z[..., -1]` is the
        # probability that an object belongs to some cluster.
        has_component = jnp.concatenate(
            [(1 - z[..., -1])[None, :], self._has_confounder_component], axis=0
        )
        mixture_weights = self.add_weights_prior(clusters, has_component)
        # shape: (n_clusters + n_confounders, n_objects, n_features)

        for partition in self.partitions:
            if isinstance(partition, CategoricalFeatures):
                self.add_partition_categorical(partition, mixture_weights)
            elif isinstance(partition, GaussianFeatures):
                self.add_partition_gaussian(partition, mixture_weights)
            elif isinstance(partition, PoissonFeatures):
                self.add_partition_poisson(partition, mixture_weights)
            else:
                raise ValueError(f"Partition type {partition.__class__.__name__} not supported.")

    def add_partition_categorical(
        self,
        partition: CategoricalFeatures,
        mixture_weights: jnp.ndarray,   # shape: (n_clusters+n_confounders, n_objects, n_features)
    ) -> None:
        """Add the likelihood of a partition of categorical features to the model.

        Samples the state probabilities of each cluster and each confounder group, mixes
        them according to `mixture_weights` and conditions them on the observed features.
        """

        p_name = partition.name
        n_clusters = self.shapes.n_clusters

        p_data_by_comp = jnp.zeros((
            self.shapes.n_components_expanded,
            self.shapes.n_objects,
            partition.n_features,
            partition.n_states,
        ))

        # Weights of this partition's features only
        p_weights = mixture_weights[:, :, partition.feature_indices]
        # shape: (n_clusters+n_confounders, n_objects, partition.n_features)

        # Sample and assign cluster effects
        cluster_effect_prior = self.prior.cluster_effect_prior[p_name]
        cluster_effect_pred = cluster_effect_prior.get_data_dependent_contributions(
            p_weights[:n_clusters]
        )
        cluster_effect = cluster_effect_prior.get_numpyro_distr(
            n_clusters,
            cluster_effect_pred,
            allow_reparameterization=self.allow_reparameterization,
        )
        p_data_by_comp = p_data_by_comp.at[:n_clusters].set(cluster_effect[:, None, :, :])

        # Sample and assign confounding effects
        for i_c, conf in enumerate(self.confounders.values()):
            conf_effect = self.prior.confounding_effects_prior[conf.name][p_name].get_numpyro_distr()
            # Objects in no group have index NO_GROUP (-1) and pick up the last group's
            # effect here. They are masked out by `has_component` in the mixture weights.
            g = self.group_assignments[i_c]
            p_data_by_comp = p_data_by_comp.at[n_clusters + i_c].set(conf_effect[g, :, :])

        # Define mixture likelihood (k: component, i: object, f: feature, s: state)
        # Adding EPS makes log(p) finite everywhere, so the gradient exists.
        p_data_mixed = normalize(
            jnp.einsum('kif,kifs->ifs', p_weights, p_data_by_comp) + EPS, axis=-1
        )
        # shape: (n_objects, n_features, n_states)

        with numpyro.plate(f"plate_objects_lh_{p_name}", self.shapes.n_objects, dim=-2):
            with numpyro.plate(f"plate_features_lh_{p_name}", partition.n_features, dim=-1):
                with numpyro.handlers.mask(mask=~partition.na_values):
                    numpyro.sample(
                        f"x_{p_name}",
                        dist.Categorical(probs=p_data_mixed),
                        obs=None if self.config.sample_from_prior else partition.values,
                    )

    def add_partition_gaussian(
            self,
            partition: GaussianFeatures,
            mixture_weights: jnp.ndarray,  # shape: (n_clusters+n_confounders, n_objects, n_features)
    ) -> None:
        """Add the likelihood of a partition of Gaussian features to the model.

        Samples the mean and variance of each cluster and each confounder group and
        conditions a Gaussian mixture on the observed features. Logit-normal features
        are handled here as well: their values are already logit-transformed, so the
        likelihood applies on the transformed scale.
        """
        p_name = partition.name
        n_clusters = self.shapes.n_clusters

        shape_by_comp = (
            self.shapes.n_components_expanded,
            self.shapes.n_objects,
            partition.n_features,
        )
        mean_by_comp = jnp.zeros(shape_by_comp)
        variance_by_comp = jnp.zeros(shape_by_comp)


        # Weights of this partition's features only
        p_weights = mixture_weights[:, :, partition.feature_indices]
        # shape: (n_clusters+n_confounders, n_objects, partition.n_features)

        # Sample and assign cluster effects
        cluster_eff_prior = self.prior.cluster_effect_prior[p_name]
        cluster_effect_pred = cluster_eff_prior.mean.get_data_dependent_contributions(
            p_weights[:n_clusters]
        )
        cluster_mean = cluster_eff_prior.mean.get_numpyro_distr(
            n_clusters, cluster_effect_pred,
            allow_reparameterization=self.allow_reparameterization,
        )
        # Note: `mean.get_numpyro_distr` samples internally (including its own plates),
        # while `variance.get_numpyro_distr` returns a distribution to sample here.
        # TODO: unify this convention when auditing model/prior.py.

        with numpyro.plate(f"plate_clusters_{p_name}", n_clusters, dim=-2):
            with numpyro.plate(f"plate_features_{p_name}", partition.n_features, dim=-1):
                cluster_variance = numpyro.sample(
                    f"cluster_effect_{p_name}_variance",
                    cluster_eff_prior.variance.get_numpyro_distr(),
                )

        mean_by_comp = mean_by_comp.at[:n_clusters].set(cluster_mean[:, None, :])
        variance_by_comp = variance_by_comp.at[:n_clusters].set(cluster_variance[:, None, :])

        # Sample and assign confounding effects
        for i_c, conf in enumerate(self.confounders.values()):
            conf_eff_prior: GaussianConfoundingEffectsPrior = (
                self.prior.confounding_effects_prior[conf.name][p_name]
            )
            with numpyro.plate(f"plate_groups_{conf.name}_{p_name}", conf.n_groups, dim=-2):
                with numpyro.plate(f"plate_features_{conf.name}_{p_name}", partition.n_features, dim=-1):
                    mean_prior = dist.Normal(
                        conf_eff_prior.mean.mu_0_array, conf_eff_prior.mean.sigma_0_array
                    )
                    conf_eff_mean = numpyro.sample(
                        f"conf_effect_{conf.name}_{p_name}_mean", mean_prior
                    )
                    conf_eff_variance = numpyro.sample(
                        f"conf_effect_{conf.name}_{p_name}_variance",
                        conf_eff_prior.variance.get_numpyro_distr(),
                    )
                    # shape of both: (n_groups, partition.n_features)

            # Objects in no group have index NO_GROUP (-1) and pick up the last group's
            # effect here. They are masked out by `has_component` in the mixture weights.
            g = self.group_assignments[i_c]
            mean_by_comp = mean_by_comp.at[n_clusters + i_c].set(conf_eff_mean[g, :])
            variance_by_comp = variance_by_comp.at[n_clusters + i_c].set(conf_eff_variance[g, :])

        # Move the component axis last, as expected by MixtureSameFamily
        # shape: (n_objects, partition.n_features, n_clusters+n_confounders)
        with numpyro.plate(f"plate_objects_lh_{p_name}", self.shapes.n_objects, dim=-2):
            with numpyro.plate(f"plate_features_lh_{p_name}", partition.n_features, dim=-1):
                with numpyro.handlers.mask(mask=~partition.na_values):
                    numpyro.sample(
                        f"x_{p_name}",
                        dist.MixtureSameFamily(

                            mixing_distribution=dist.Categorical(
                                # Adding EPS makes log(p) finite everywhere, so the gradient exists.
                                probs=normalize(p_weights.transpose((1, 2, 0)) + EPS, axis=-1)
                            ),
                            component_distribution=dist.Normal(
                                loc=mean_by_comp.transpose((1, 2, 0)),
                                scale=variance_by_comp.transpose((1, 2, 0)) ** 0.5,
                            ),
                        ),
                        obs=None if self.config.sample_from_prior else partition.values,
                    ) # Shape: [n_objects, n_features, n_components]

    def add_partition_poisson(
        self,
        partition: PoissonFeatures,
        mixture_weights: jnp.ndarray,   # shape: (n_clusters+n_confounders, n_objects, n_features)
    ) -> None:
        """Add the likelihood of a partition of Poisson features to the model.

        Samples the rate of each cluster and each confounder group and conditions a
        Poisson mixture on the observed features.
        """
        p_name = partition.name
        n_clusters = self.shapes.n_clusters

        rate_by_comp = jnp.zeros((
            self.shapes.n_components_expanded,
            self.shapes.n_objects,
            partition.n_features,
        ))

        # Weights of this partition's features only
        p_weights = mixture_weights[:, :, partition.feature_indices]
        # shape: (n_clusters+n_confounders, n_objects, partition.n_features)

        # Sample and assign cluster effects
        cluster_eff_prior: PoissonClusterEffectPrior = self.prior.cluster_effect_prior[p_name]
        with numpyro.plate(f"plate_clusters_{p_name}", n_clusters, dim=-2):
            with numpyro.plate(f"plate_features_{p_name}", partition.n_features, dim=-1):
                cluster_rate = numpyro.sample(
                    f"cluster_effect_{p_name}_rate",
                    cluster_eff_prior.rate.get_numpyro_distr(),
                )

        rate_by_comp = rate_by_comp.at[:n_clusters].set(cluster_rate[:, None, :])

        # Sample and assign confounding effects
        for i_c, conf in enumerate(self.confounders.values()):
            conf_eff_prior: PoissonConfoundingEffectsPrior = (
                self.prior.confounding_effects_prior[conf.name][p_name]
            )
            with numpyro.plate(f"plate_groups_{conf.name}_{p_name}", conf.n_groups, dim=-2):
                with numpyro.plate(f"plate_features_{conf.name}_{p_name}", partition.n_features, dim=-1):
                    conf_eff_rate = numpyro.sample(
                        f"conf_effect_{conf.name}_{p_name}_rate",
                        conf_eff_prior.rate.get_numpyro_distr(),
                    )
                    # shape: (n_groups, partition.n_features)

            # Objects in no group have index NO_GROUP (-1) and pick up the last group's
            # effect here. They are masked out by `has_component` in the mixture weights.
            g = self.group_assignments[i_c]
            rate_by_comp = rate_by_comp.at[n_clusters + i_c].set(conf_eff_rate[g, :])

        # Move the component axis last, as expected by MixtureSameFamily
        # shape: (n_objects, partition.n_features, n_clusters+n_confounders)
        with numpyro.plate(f"plate_objects_lh_{p_name}", self.shapes.n_objects, dim=-2):
            with numpyro.plate(f"plate_features_lh_{p_name}", partition.n_features, dim=-1):
                with numpyro.handlers.mask(mask=~partition.na_values):
                    numpyro.sample(
                        name=f"x_{p_name}",
                        fn=dist.MixtureSameFamily(
                            mixing_distribution=dist.Categorical(
                                # Adding EPS makes log(p) finite everywhere, so the gradient exists.
                                probs=normalize(p_weights.transpose((1, 2, 0)) + EPS, axis=-1)
                            ),
                            component_distribution=dist.Poisson(
                                rate=rate_by_comp.transpose((1, 2, 0))
                            ),
                        ),
                        obs=None if self.config.sample_from_prior else partition.values,
                    )

    def add_weights_prior(
        self,
        clusters: jnp.ndarray,          # shape: (n_objects, n_clusters)
        has_component: jnp.ndarray,     # shape: (n_components, n_objects)
    ) -> jnp.ndarray:
        """Sample the mixture weights and expand them over objects and clusters.

        The weights define how much each component (the cluster or one of the
        confounders) contributes to each feature of each object. Components an object
        does not belong to are masked out via `has_component` and the remaining weights
        are renormalized. The single cluster component is then expanded into one
        component per cluster, weighted by the object's cluster assignments.

        Args:
            clusters: the cluster assignments of each object
            has_component: how much each object participates in each component

        Returns:
            The mixture weights per component, object and feature.
            shape: (n_clusters + n_confounders, n_objects, n_features)
        """
        weights_config = self.config.prior.weights

        if weights_config.hierarchical:
            conc_cfg = weights_config.concentration_prior
            with numpyro.plate("plate_components_w_prior", self.shapes.n_components, dim=-1):
                # Sample from Gamma distribution: raw_conc ~ Gamma(shape, rate)
                raw_conc = numpyro.sample(
                    "w_concentration_raw", dist.Gamma(conc_cfg.shape, conc_cfg.rate)
                )
                # Apply the lower bound (offset): w_concentration = offset + raw_conc
                w_concentration = conc_cfg.offset + raw_conc
            numpyro.deterministic("w_concentration", w_concentration)
        else:
            w_concentration = self.w_prior_conc


        with numpyro.plate("plate_objects_w", self.shapes.n_features, dim=-1):
            if self.allow_reparameterization:
                w = numpyro.sample("w", dist.Dirichlet(w_concentration))
            else:
                w = dirichlet_from_latent(
                    "w", w_concentration, offset=normalize(w_concentration, axis=-1)
                )
        # w.shape: (n_features, n_components)

        # Mask out components that the object does not belong to
        w_per_object = w.T[:, None, :] * has_component[:, :, None]
        # shape: (n_components, n_objects, n_features)

        if weights_config.varying_cluster_weights:
            # Sample cluster weight factor concentrations from Gamma with configurable
            # offsets. `w_cluster_factor` is a 2-simplex per (cluster, feature):
            # component 1 is the factor applied to the cluster weight, component 0 its
            # complement.
            factor_conc_cfg = weights_config.cluster_weight_factor_concentration
            factor_conc_distr = dist.Gamma(factor_conc_cfg.shape, factor_conc_cfg.rate)
            factor_conc_raw = numpyro.sample(
                "w_cluster_factor_c_raw", factor_conc_distr.expand((2,)).to_event(1)
            )
            factor_conc = factor_conc_raw + factor_conc_cfg.offset
            numpyro.deterministic("w_cluster_factor_c", factor_conc)

            with numpyro.plate("plate_clusters_w", self.shapes.n_clusters, dim=-2):
                with numpyro.plate("plate_features_w", self.shapes.n_features, dim=-1):
                    cluster_factor = dirichlet_from_latent(
                        "w_cluster_factor", factor_conc, offset=normalize(factor_conc)
                    )[..., 1]

            w_cluster = cluster_factor * w[:, CLUSTER_COMPONENT]
            w_cluster_mixed = clusters @ w_cluster
            # shape: (n_objects, n_features)

            w_per_object = w_per_object.at[CLUSTER_COMPONENT].set(w_cluster_mixed)

        # Normalize over the components
        w_per_object = w_per_object / jnp.maximum(w_per_object.sum(axis=0, keepdims=True), EPS)


        # Expand the cluster component into one component per cluster
        clusters_normalized = clusters / jnp.maximum(has_component[CLUSTER_COMPONENT, :, None], EPS)
        # shape: (n_objects, n_clusters)

        # Row 0 is the cluster component, rows 1: are the confounders
        per_cluster_weights = (
            clusters_normalized.T[:, :, None] * w_per_object[:1, :, :]
        )
        # shape: (n_clusters, n_objects, n_features)

        return jnp.concatenate(
            [per_cluster_weights, w_per_object[1:, :, :]], axis=0
        )

    def get_tempered_model(self, temperature: float = 1.0) -> None:
        """Define the model with its log-density scaled by `1 / temperature`.

        Used for annealing during warm-up: higher temperatures flatten the posterior and
        make it easier to traverse.
        """
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be positive, but was {temperature}.")

        with numpyro.handlers.scale(scale=1 / temperature):
            self.get_model()

    def generate_initial_params(self, rng_key) -> dict:
        """Generate initial parameter values for the MCMC run.

        Starts from NumPyro's own initialization and overwrites the categorical
        confounding effects with an empirical estimate: the prior counts plus the
        observed state counts of each group.

        Args:
            rng_key: random key for NumPyro's model initialization

        Returns:
            The initial value of each parameter, in constrained space.
        """

        # `init_to_uniform` (NumPyro's default) draws in unconstrained space, which puts
        # the bounded uniform latents of this model at their boundaries, where the
        # density is zero. `init_to_median` draws from the prior instead.
        init_params = initialize_model(
            rng_key, self.get_model, init_strategy=init_to_median,
        )[0].z

        init_params = constrain_fn(self.get_model, (), {}, init_params)

        # Empirical initialization is currently only implemented for categorical
        # features. Other feature types keep the values from `initialize_model`.
        # TODO: consider empirical initialization for Gaussian and Poisson effects.
        for partition in self.partitions:
            if not isinstance(partition, CategoricalFeatures):
                continue

            p = partition.name
            features_binary = partition.to_binary()

            # Sample and assign confounding effects
            for conf in self.confounders.values():
                prior_counts = self.prior.confounding_effects_prior[conf.name][p].concentration_array
                for i_g, g in enumerate(conf.group_assignment):
                    # Count the observed states of the objects in this group
                    feature_counts = jnp.sum(features_binary, axis=0, where=g[:, None, None])
                    conf_eff_c_g = normalize(prior_counts[i_g] + feature_counts, axis=-1)
                    init_params[f"conf_effect_{conf.name}_{p}"] = (
                        init_params[f"conf_effect_{conf.name}_{p}"].at[i_g].set(conf_eff_c_g)
                    )

        # TODO: initialize the cluster assignments and cluster effects as well - they
        #   currently keep the values from `initialize_model`.

        return init_params


    def __copy__(self) -> Model:
        """Build an independent Model from the same data and config.

        Note: this rebuilds the model rather than copying its state, so the new instance
        shares no JAX arrays with the original.
        """
        return Model(self.data, self.config)

    def get_setup_message(self) -> str:
        """Compile a set-up message for logging."""
        return (
            f"\n"
            f"Model\n"
            f"##########################################\n"
            f"Number of clusters: {self.shapes.n_clusters}\n"
            f"{self.prior.get_setup_message()}"
        )