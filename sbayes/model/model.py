#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from numpyro import handlers
from jax import random
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model, constrain_fn

from sbayes.model.model_shapes import ModelShapes
from sbayes.model.prior import (
    Prior,
    GaussianConfoundingEffectsPrior,
    PoissonConfoundingEffectsPrior,
    PoissonClusterEffectPrior,
    dirichlet_from_latent,
)
from sbayes.config.config import ModelConfig
from sbayes.load_data import Data, CategoricalFeatures, GaussianFeatures, PoissonFeatures
from sbayes.util import onehot_to_integer_encoding


def indent(text, amount, ch=' '):
    padding = amount * ch
    return ''.join(padding + line for line in text.splitlines(True))


def normalize(x, axis=-1):
    """Normalize ´x´ s.t. the last axis sums up to 1.

    Args:
        x (np.array): Array to be normalized.
        axis (int): The axis to be normalized (will sum up to 1).

    Returns:
         jnp.array: x with normalized s.t. the last axis sums to 1.

    == Usage ===
    >>> normalize(np.ones((2, 4))).tolist()
    [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
    >>> normalize(np.ones((2, 4)), axis=0).tolist()
    [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]
    """
    return (x / jnp.sum(x, axis=axis, keepdims=True))


class Model:
    """The sBayes model: posterior distribution of clusters and parameters.

    Attributes:
        data (Data): The data used in the likelihood
        config (ModelConfig): A dictionary containing configuration parameters of the model
        confounders (dict): A dict of all confounders and group names
        shapes (sbayes.model.ModelShapes): A dictionary with shape information for building the Likelihood and Prior objects
        prior (Prior): Rhe prior of the model

    """

    def __init__(self, data: Data, config: ModelConfig):
        self.data = data
        self.config = config
        self.confounders = data.confounders
        self.n_clusters = config.clusters
        n_objects, n_features = self.data.features.all_features.shape

        self.shapes = ModelShapes(
            n_clusters=self.n_clusters,
            n_objects=n_objects,
            n_features=n_features,
            n_confounders=len(self.confounders),
            n_groups={name: conf.n_groups for name, conf in self.confounders.items()}
        )

        # Initialize the prior
        self.prior = Prior(shapes=self.shapes, config=self.config.prior, data=data)

        # Create a list of group names and group assignments for the cluster and confounder effects
        self.group_names = []
        self.group_assignments = -jnp.ones((self.shapes.n_confounders, self.shapes.n_objects), dtype=int)
        for i_c, confounder in enumerate(self.confounders.values()):
            self.group_names.append(confounder.group_names)

            # Translate binary (one-hot) group assignments to integer values
            group_indexes = onehot_to_integer_encoding(confounder.group_assignment, none_index=-1, axis=0)
            self.group_assignments = self.group_assignments.at[i_c].set(group_indexes)

        self.w_prior_conc = jnp.array(self.prior.weights_prior.concentration_array).astype(jnp.float32)

        # self._has_component = jnp.concate(self.group_assignments != -1, jnp.float32)
        self._has_component = jnp.concatenate([-jnp.ones((1, self.shapes.n_objects)), self.group_assignments != -1], axis=0)

        self.partitions = self.data.features.partitions

        # self.get_model = handlers.seed(self._get_model, rng_seed=0)

        self.sample_from_prior = config.sample_from_prior
        self.allow_dirichlet_transform = not self.sample_from_prior

    def calibrate(self):
        """Run any potential calibration procedures that are required before the MCMC run."""

        # Estimate the normalization constant of the geo_prior for different `scale` values.
        if self.prior.geo_prior.config.estimate_rate:
            self.prior.geo_prior.calibrate(self.prior.cluster_prior)

    def get_model(self, no_clusters: bool = False):
        """Return the model function for the sBayes model."""

        # Add the cluster prior to the model
        z = self.prior.cluster_prior.get_numpyro_distr(allow_reparameterization=self.allow_dirichlet_transform)

        # `z` has a dummy "no-cluster" at the end, which we want to remove for most purposes
        clusters = z[..., :-1]

        if not no_clusters:
            # Add geo-prior as a factor to the model
            self.prior.geo_prior.get_numpyro_distr(clusters)

            # Update the `cluster` component of `has_component` to reflect the cluster assignments
            self.has_component = self._has_component.at[0, :].set(1 - z[..., -1])

        mixture_weights = self.add_weights_prior(clusters)
        # shape: (n_components, n_objects, n_features)

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
        mixture_weights: jnp.ndarray,       # shape: (n_clusters+n_confounders, n_objects, n_features)
    ):
        # Short alias for partition_name
        p_name = partition.name

        n_flat_components = self.shapes.n_clusters + self.shapes.n_confounders
        p_data_by_comp = jnp.zeros((n_flat_components, self.shapes.n_objects, partition.n_features, partition.n_states))

        # Sample and assign cluster effects
        cluster_effect_pred = self.prior.cluster_effect_prior[p_name].get_data_dependent_contributions(mixture_weights[:self.n_clusters, :, partition.feature_indices])
        cluster_effect = self.prior.cluster_effect_prior[p_name].get_numpyro_distr(self.n_clusters, cluster_effect_pred, allow_reparameterization=self.allow_dirichlet_transform)

        p_data_by_comp = p_data_by_comp.at[:self.n_clusters].set(cluster_effect[:, None, :, :])

        # Sample and assign confounding effects
        for i_c, conf in enumerate(self.confounders.values()):
            conf_effect = self.prior.confounding_effects_prior[conf.name][p_name].get_numpyro_distr()
            g = self.group_assignments[i_c]
            p_data_by_comp = p_data_by_comp.at[self.n_clusters + i_c].set(conf_effect[g, :, :])

        # Define mixture likelihood
        p_data_mixed = jnp.einsum(
            'kif,kifs->ifs',
            mixture_weights[:, :, partition.feature_indices],
            p_data_by_comp
        )  # shape: (n_objects, n_features, n_states)
        with numpyro.plate(f"plate_objects_lh_{p_name}", self.shapes.n_objects, dim=-2):
            with numpyro.plate(f"plate_features_lh_{p_name}", partition.n_features, dim=-1):
                with numpyro.handlers.mask(mask=~partition.na_values):
                    numpyro.sample(
                        f"x_{p_name}",
                        dist.Categorical(probs=p_data_mixed),
                        obs=None if self.sample_from_prior else partition.values,
                    )

    def add_partition_gaussian(
        self,
        partition: GaussianFeatures,
        mixture_weights: jnp.ndarray,       # shape: (n_clusters+n_confounders, n_objects, n_features)
    ):
        # Short alias for partition_name
        p_name = partition.name

        #
        n_flat_components = self.shapes.n_clusters + self.shapes.n_confounders
        mean_by_comp = jnp.zeros((n_flat_components, self.shapes.n_objects, partition.n_features))
        variance_by_comp = jnp.zeros((n_flat_components, self.shapes.n_objects, partition.n_features))

        # Sample and assign cluster effects
        cluster_effect_pred = self.prior.cluster_effect_prior[p_name].mean.get_data_dependent_contributions(mixture_weights[:self.n_clusters, :, partition.feature_indices])
        cluster_mean = self.prior.cluster_effect_prior[p_name].mean.get_numpyro_distr(self.n_clusters, cluster_effect_pred)

        cluster_eff_prior = self.prior.cluster_effect_prior[partition.name]
        with numpyro.plate(f"plate_clusters_{p_name}", self.n_clusters, dim=-2):
            with numpyro.plate(f"plate_features_{p_name}", partition.n_features, dim=-1):
                # cluster_mean_dist = dist.Normal(cluster_eff_prior.mean.mu_0_array, cluster_eff_prior.mean.sigma_0_array)
                # cluster_mean = numpyro.sample(f"cluster_effect_{p_name}_mean", cluster_mean_dist)

                # cluster_scale_dist = dist.Exponential(rate=cluster_eff_prior.variance.rate)
                cluster_scale_dist = cluster_eff_prior.variance.get_numpyro_distr()
                cluster_scale = numpyro.sample(f"cluster_effect_{p_name}_variance", cluster_scale_dist)

        mean_by_comp = mean_by_comp.at[:self.n_clusters].set(cluster_mean[:, None, :])
        variance_by_comp = variance_by_comp.at[:self.n_clusters].set(cluster_scale[:, None, :])

        # Sample and assign confounding effects
        for i_c, conf in enumerate(self.confounders.values()):
            conf_eff_prior: GaussianConfoundingEffectsPrior = self.prior.confounding_effects_prior[conf.name][p_name]
            with numpyro.plate(f"plate_groups_{i_c}", conf.n_groups, dim=-2):
                with numpyro.plate(f"plate_features_{i_c}_{p_name}", partition.n_features, dim=-1):
                    mean_prior = dist.Normal(conf_eff_prior.mean.mu_0_array, conf_eff_prior.mean.sigma_0_array)
                    conf_eff_mean = numpyro.sample(f"conf_effect_{i_c}_{p_name}_mean", mean_prior)
                    # shape: (n_groups, n_features)

                    # variance_prior = dist.Exponential(conf_eff_prior.variance.rate)
                    variance_prior = conf_eff_prior.variance.get_numpyro_distr()
                    conf_eff_variance = numpyro.sample(f"conf_effect_{i_c}_{p_name}_variance", variance_prior)
                    # shape: (n_groups, n_features)

            g = self.group_assignments[i_c]
            mean_by_comp = mean_by_comp.at[self.n_clusters + i_c].set(conf_eff_mean[g, :])
            variance_by_comp = variance_by_comp.at[self.n_clusters + i_c].set(conf_eff_variance[g, :])

        with numpyro.plate(f"plate_objects_lh_{p_name}", self.shapes.n_objects, dim=-2):
            with numpyro.plate(f"plate_features_lh_{p_name}", partition.n_features, dim=-1):
                with numpyro.handlers.mask(mask=~partition.na_values):
                    numpyro.sample(
                        f"x_{p_name}",
                        dist.MixtureSameFamily(
                            mixing_distribution=dist.Categorical(probs=mixture_weights[:, :, partition.feature_indices].transpose((1, 2, 0))),
                            component_distribution=dist.Normal(loc=mean_by_comp.transpose((1, 2, 0)), scale=variance_by_comp.transpose((1, 2, 0))**0.5)
                        ),
                        obs=None if self.sample_from_prior else partition.values
                    )
                    # Shape: [n_objects, n_features, n_components]

                    # ALTERNATIVE MANUAL IMPLEMENTATION
                    # log_weights = jnp.log(mixture_weights[:, :, partition.feature_indices].transpose((1, 2, 0)))  # (n_obj, n_feat, n_comp)
                    # means = mean_by_comp.transpose((1, 2, 0))  # (n_obj, n_feat, n_comp)
                    # stds = variance_by_comp.transpose((1, 2, 0)) ** 0.5  # (n_obj, n_feat, n_comp)
                    #
                    # # Broadcast obs to match component shape: (n_obj, n_feat, 1)
                    # observations = partition.values[..., None]
                    #
                    # # Compute log prob for each component
                    # log_probs = dist.Normal(loc=means, scale=stds).log_prob(observations)  # (n_obj, n_feat, n_comp)
                    #
                    # # Log-sum-exp over components to marginalize the mixture
                    # total_log_prob = jax.scipy.special.logsumexp(log_weights + log_probs, axis=-1)  # (n_obj, n_feat)
                    #
                    # # Contribute to the joint log prob
                    # numpyro.factor(f"log_prob_{p_name}", total_log_prob.sum())

    def add_partition_poisson(
        self,
        partition: PoissonFeatures,
        mixture_weights: jnp.ndarray,       # shape: (n_clusters+n_confounders, n_objects, n_features)
    ):
        # Short alias for partition_name
        p_name = partition.name

        # Initialize rate_by_comp
        n_flat_components = self.shapes.n_clusters + self.shapes.n_confounders
        rate_by_comp = jnp.zeros((n_flat_components, self.shapes.n_objects, partition.n_features))

        # Sample and assign cluster effects
        cluster_eff_prior: PoissonClusterEffectPrior = self.prior.cluster_effect_prior[partition.name]
        with numpyro.plate(f"plate_clusters_{p_name}", self.n_clusters, dim=-2):
            with numpyro.plate(f"plate_features_{p_name}", partition.n_features, dim=-1):
                cluster_rate_dist = cluster_eff_prior.mean.get_numpyro_distr()
                cluster_rate = numpyro.sample(f"cluster_effect_{p_name}_rate", cluster_rate_dist)

        rate_by_comp = rate_by_comp.at[:self.n_clusters].set(cluster_rate[:, None, :])

        # Sample and assign confounding effects
        for i_c, conf in enumerate(self.confounders.values()):
            conf_eff_prior: PoissonConfoundingEffectsPrior = self.prior.confounding_effects_prior[conf.name][p_name]
            with numpyro.plate(f"plate_groups_{i_c}", conf.n_groups, dim=-2):
                with numpyro.plate(f"plate_features_{i_c}_{p_name}", partition.n_features, dim=-1):
                    rate_prior = conf_eff_prior.rate.get_numpyro_distr()
                    conf_eff_rate = numpyro.sample(f"conf_effect_{i_c}_{p_name}_rate", rate_prior)
                    # shape: (n_groups, n_features)

            g = self.group_assignments[i_c]
            rate_by_comp = rate_by_comp.at[self.n_clusters + i_c].set(conf_eff_rate[g, :])

        with numpyro.plate(f"plate_objects_lh_{p_name}", self.shapes.n_objects, dim=-2):
            with numpyro.plate(f"plate_features_lh_{p_name}", partition.n_features, dim=-1):
                with numpyro.handlers.mask(mask=~partition.na_values):
                    numpyro.sample(
                        name=f"x_{p_name}",
                        fn=dist.MixtureSameFamily(
                            mixing_distribution=dist.Categorical(probs=mixture_weights[:, :, partition.feature_indices].transpose((1, 2, 0))),
                            component_distribution=dist.Poisson(rate=rate_by_comp.transpose((1, 2, 0)))
                        ),
                        obs=None if self.sample_from_prior else partition.values
                    )
                    # Shape: [n_objects, n_features, n_components]

    def add_weights_prior(self, clusters):
        weights_config = self.config.prior.weights

        if weights_config.hierarchical:
            with numpyro.plate("plate_components_w_prior", self.shapes.n_components, dim=-1):
                w_concentration = numpyro.sample("w_concentration", dist.Gamma(*weights_config.concentration_prior))
            with numpyro.plate("plate_objects_w", self.shapes.n_features, dim=-1):
                if self.allow_dirichlet_transform:
                    w = numpyro.sample("w", dist.Dirichlet(w_concentration))
                else:
                    w = dirichlet_from_latent("w", w_concentration, offset=normalize(w_concentration, axis=-1))
        else:
            with numpyro.plate("plate_objects_w", self.shapes.n_features, dim=-1):
                if self.allow_dirichlet_transform:
                    w = numpyro.sample("w", dist.Dirichlet(self.w_prior_conc))
                else:
                    w = dirichlet_from_latent("w", self.w_prior_conc)
            # w = numpyro.sample("w", dist.Dirichlet(self.w_prior_conc))
        # shape: (n_features, n_components)


        # Multiply weights with `has_component` to mask out components that are not present in the group and normalize
        w_per_object = w.T[:, None, :] * self.has_component[:, :, None]
        # shape: (n_components, n_objects, n_features)

        if weights_config.varying_cluster_weights:
            c0 = numpyro.sample("w_cluster_concentration_0", dist.Gamma(*weights_config.mask_prior_concentration_0))
            c1 = numpyro.sample("w_cluster_concentration_1", dist.Gamma(*weights_config.mask_prior_concentration_1))
            cluster_factor_concentration = jnp.stack([c0, c1], axis=-1)
            with numpyro.plate("plate_clusters_w", self.n_clusters, dim=-2):
                with numpyro.plate("plate_features_w", self.shapes.n_features, dim=-1):
                    cluster_factor = dirichlet_from_latent("w_cluster_factor", cluster_factor_concentration, offset=normalize(cluster_factor_concentration))[..., 1]

            w_cluster = cluster_factor * w[:, 0]
            w_cluster_mixed = clusters @ w_cluster
            # shape: (n_objects, n_features)

            # Update the weights
            w_per_object = w_per_object.at[0].set(w_cluster_mixed)

        # Normalize weights_per_object
        w_per_object = w_per_object / w_per_object.sum(axis=-3, keepdims=True)

        # Flatten the weights into one array for all clusters and confounders
        clusters_normalized = clusters / self.has_component[0, :, None]                             # (objects, clusters)
        per_cluster_weights = clusters_normalized.T[:, :, None] * w_per_object[:1, :, :]            # (clusters, objects, features)
        mixture_weights = jnp.concat([per_cluster_weights, w_per_object[1:, :, :]], axis=0)  # (clusters+confounders, objects, features)

        return mixture_weights

    def get_tempered_model(self, temperature=1.0, no_clusters=False):
        """Perform tempered MCMC sampling of the model."""
        with numpyro.handlers.scale(scale=1/temperature):
            return self.get_model(no_clusters=no_clusters)

    def generate_initial_params(self, rng_key, n_chains: int = 1) -> dict:
        """Initialize the sBayes model and return the model function."""

        init_params = initialize_model(rng_key, self.get_model)[0].z
        init_params = constrain_fn(self.get_model, (), {}, init_params)

        for partition in self.partitions:
            if not isinstance(partition, CategoricalFeatures):
                continue

            p = partition.name
            features_binary = partition.to_binary()

            # Sample and assign confounding effects
            for i_c, conf in enumerate(self.confounders.values()):
                prior_counts = self.prior.confounding_effects_prior[conf.name][p].concentration_array
                for i_g, g in enumerate(conf.group_assignment):
                    feature_counts = jnp.sum(features_binary, axis=0, where=g[:, None, None])
                    conf_eff_c_g = normalize(prior_counts[i_g] + feature_counts, axis=-1)
                    # conf_eff_c_g_latent = simplex_transform._inverse(conf_eff_c_g)
                    init_params[f"conf_effect_{i_c}_{p}"] = init_params[f"conf_effect_{i_c}_{p}"].at[i_g].set(conf_eff_c_g)

        # Sample clusters to initialize the model


        # # Broadcast to multiple chains
        # for key, value in init_params.items():
        #     init_params[key] = jnp.broadcast_to(value, (n_chains,) + value.shape)

        return init_params

    # def get_svi_guide(self):
    #     """Return a custom SVI guide for the sBayes model."""
    #
    #     n_clusters = self.n_clusters
    #     n_objects = self.shapes.n_objects
    #     n_features = self.shapes.n_features
    #
    #     features = self.data.features.values
    #     confounders = list(self.data.confounders.values())
    #     counts_by_conf = [
    #         jnp.array([
    #                       jnp.sum(features[grps, :, :], axis=0)
    #                       for grps in conf.group_assignment
    #                   ] + [jnp.zeros((n_features, self.shapes.n_states))])  # Add a dummy group
    #         for conf in confounders
    #     ]
    #
    #     def guide(*args, **kwargs):
    #         # Guide for `z` (cluster assignments)
    #         # # z = add_logistic_normal_distribution(n_clusters + 1, n_objects, "z")
    #         # z = add_logistic_normal_distribution(n_objects, n_clusters + 1, "z")
    #         # # z_posterior_conc = numpyro.param("z_posterior_conc", jnp.ones((n_objects, n_clusters + 1)), constraint=constraints.positive)
    #         # # numpyro.sample("z", dist.Dirichlet(z_posterior_conc))
    #
    #         # Parameters for the logistic normal
    #         # z_mean = numpyro.param("z_mean", jnp.zeros((n_clusters, n_objects)))
    #         z_mean = numpyro.param("z_mean", 0.1 * jax.random.normal(jax.random.PRNGKey(1), (n_clusters, n_objects)))
    #         z_cov = numpyro.param("z_cov", jnp.eye(n_objects), constraints=constraints.positive_semidefinite)
    #         z_cov += 1e-6 * jnp.eye(n_objects)  # Add a small diagonal noise term for stability
    #
    #         # Sample from the normal distribution with object-wise correlation
    #         z_logit = numpyro.sample("z_logit", dist.MultivariateNormal(z_mean, z_cov),
    #                                  infer={'is_auxiliary': True})
    #
    #         # Apply the softmax transform
    #         z_logit_scale = numpyro.param("z_logit_scale", 0.1, constraint=constraints.positive)
    #         z = numpyro.sample("z", dist.TransformedDistribution(
    #             dist.Normal(z_logit.T, z_logit_scale),
    #             transforms=[transforms.StickBreakingTransform()],
    #             # transforms=[SoftmaxTransform()],
    #         ))
    #
    #         # z = numpyro.sample("z", dist.TransformedDistribution(
    #         #     dist.MultivariateNormal(z_mean, z_cov),
    #         #     transforms=[Transpose(), SoftmaxTransform()],
    #         # ))
    #
    #         print(z.shape)
    #
    #         n_states = self.shapes.n_states
    #         # cluster_eff_logit_mean = fnn_two_layers(z.T, D_H = 5 * n_features, D_Y=n_features * n_states)[:-1, :].reshape((n_clusters, n_features, n_states))
    #         # cluster_eff_logit_mean = numpyro.param(f"cluster_eff_mean", jnp.zeros((n_clusters, n_features, n_states-1)))
    #         cluster_eff_logit_mean = numpyro.param(f"cluster_eff_mean", 0.1 * jax.random.normal(jax.random.PRNGKey(1), (
    #         n_clusters, n_features, n_states - 1)))
    #         cluster_eff_logit_scale = numpyro.param(f"cluster_eff_scale",
    #                                                 jnp.ones((n_clusters, n_features, n_states - 1)),
    #                                                 constraint=constraints.positive)
    #         numpyro.sample("cluster_effect", dist.TransformedDistribution(
    #             dist.Normal(cluster_eff_logit_mean, cluster_eff_logit_scale),
    #             transforms=[transforms.StickBreakingTransform()],
    #             # transforms=[SoftmaxTransform()],
    #         ))
    #
    #         # cluster_effect_conc = numpyro.param("cluster_effect_conc", jnp.ones((n_clusters, n_features, n_states)), constraint=constraints.positive)
    #         # print('cluster_effect_conc', cluster_effect_conc.shape)
    #
    #         # cluster_effect_conc = numpyro.param("cluster_effect_conc", jnp.repeat(self.clust_eff_prior_conc[None,...], n_clusters, axis=0), constraint=constraints.positive)
    #         # numpyro.sample("cluster_effect", dist.Dirichlet(cluster_effect_conc))
    #
    #         # Guide for confounding effects
    #         for i_c in range(1, self.shapes.n_confounders + 1):
    #             n_groups = len(self.group_names[i_c])
    #             with numpyro.plate(f"plate_groups_{i_c}", n_groups + 1, dim=-2):
    #                 with numpyro.plate(f"plate_features_{i_c}", self.shapes.n_features, dim=-1):
    #                     conf_effect_conc = numpyro.param(f"conf_effect_conc_{i_c - 1}",
    #                                                      self.conf_eff_prior_params[i_c - 1] + counts_by_conf[
    #                                                          i_c - 1] / self.shapes.n_components,
    #                                                      constraint=constraints.positive)
    #                     numpyro.sample(f"conf_eff_{i_c - 1}", dist.Dirichlet(conf_effect_conc))
    #
    #         # Guide for weights (`w`)
    #         w_posterior_conc = numpyro.param("w_posterior_conc", self.w_prior_conc,
    #                                          constraint=constraints.positive)
    #         numpyro.sample("w", dist.Gamma(w_posterior_conc, 1))
    #
    #     return guide

    def __copy__(self):
        return Model(self.data, self.config)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        setup_msg = "\n"
        setup_msg += "Model\n"
        setup_msg += "##########################################\n"
        setup_msg += f"Number of clusters: {self.config.clusters}\n"
        setup_msg += self.prior.get_setup_message()
        return setup_msg
