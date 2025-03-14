#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import StickBreakingTransform, ExpTransform
from numpyro.infer.util import initialize_model

from sbayes.model.model_shapes import ModelShapes
from sbayes.model.prior import Prior, GeoPrior, GaussianConfoundingEffectsPrior
from sbayes.config.config import ModelConfig
from sbayes.load_data import Data, FeatureType, GenericTypeFeatures, CategoricalFeatures, GaussianFeatures, \
    PoissonFeatures
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

    def get_model(self, no_clusters: bool = False):
        """Return the model function for the sBayes model."""

        # Add the cluster prior to the model
        z = self.prior.cluster_prior.get_numpyro_distr()

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
                # self.add_partition_poisson(partition, clusters, mixture_weights)
                raise NotImplementedError
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
        with numpyro.plate(f"plate_clusters_{p_name}", self.n_clusters, dim=-2):
            with numpyro.plate(f"plate_features_{p_name}", partition.n_features, dim=-1):
                cluster_effect_prior = dist.Dirichlet(self.prior.cluster_effect_prior[p_name].concentration_array)
                cluster_effect = numpyro.sample(f"cluster_effect_{p_name}", cluster_effect_prior)
                # shape: (n_clusters, n_features, n_states)

        p_data_by_comp = p_data_by_comp.at[:self.n_clusters].set(cluster_effect[:, None, :, :])

        # Sample and assign confounding effects
        for i_c, conf in enumerate(self.confounders.values()):
            concentration = self.prior.confounding_effects_prior[conf.name][p_name].concentration_array
            with numpyro.plate(f"plate_groups_{i_c}", conf.n_groups, dim=-2):
                with numpyro.plate(f"plate_features_{i_c}_{p_name}", partition.n_features, dim=-1):
                    conf_effect = numpyro.sample(f"conf_effect_{i_c}_{p_name}", dist.Dirichlet(concentration))
                    # shape: (n_groups, n_features, n_states)

            g = self.group_assignments[i_c]
            p_data_by_comp = p_data_by_comp.at[self.n_clusters + i_c].set(conf_effect[g, :, :])

        if not self.config.sample_from_prior:
            # Define mixture likelihood
            p_data_mixed = jnp.einsum('kif,kifs->ifs', mixture_weights[:, :, partition.feature_indices], p_data_by_comp)  # shape: (n_objects, n_features, n_states)
            with numpyro.plate(f"plate_objects_lh_{p_name}", self.shapes.n_objects, dim=-2):
                with numpyro.plate(f"plate_features_lh_{p_name}", partition.n_features, dim=-1):
                    with numpyro.handlers.mask(mask=~partition.na_values):
                        numpyro.sample(f"x_{p_name}", dist.Categorical(probs=p_data_mixed), obs=partition.values)

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
        cluster_eff_prior = self.prior.cluster_effect_prior[partition.name]
        with numpyro.plate(f"plate_clusters_{p_name}", self.n_clusters, dim=-2):
            with numpyro.plate(f"plate_features_{p_name}", partition.n_features, dim=-1):
                cluster_loc_dist = dist.Normal(cluster_eff_prior.mean.mu_0_array, cluster_eff_prior.mean.sigma_0_array)
                cluster_loc = numpyro.sample(f"cluster_effect_{p_name}_mean", cluster_loc_dist)

                cluster_scale_dist = dist.Exponential(rate=cluster_eff_prior.variance.rate)
                cluster_scale = numpyro.sample(f"cluster_effect_{p_name}_variance", cluster_scale_dist)

        mean_by_comp = mean_by_comp.at[:self.n_clusters].set(cluster_loc[:, None, :])
        variance_by_comp = variance_by_comp.at[:self.n_clusters].set(cluster_scale[:, None, :])

        # Sample and assign confounding effects
        for i_c, conf in enumerate(self.confounders.values()):
            conf_eff_prior: GaussianConfoundingEffectsPrior = self.prior.confounding_effects_prior[conf.name][p_name]
            with numpyro.plate(f"plate_groups_{i_c}", conf.n_groups, dim=-2):
                with numpyro.plate(f"plate_features_{i_c}_{p_name}", partition.n_features, dim=-1):
                    mean_prior = dist.Normal(conf_eff_prior.mean.mu_0_array, conf_eff_prior.mean.sigma_0_array)
                    conf_eff_mean = numpyro.sample(f"conf_effect_{i_c}_{p_name}_mean", mean_prior)
                    # shape: (n_groups, n_features)

                    variance_prior = dist.Exponential(conf_eff_prior.variance.rate)
                    conf_eff_variance = numpyro.sample(f"conf_effect_{i_c}_{p_name}_variance", variance_prior)
                    # shape: (n_groups, n_features)

            g = self.group_assignments[i_c]
            mean_by_comp = mean_by_comp.at[self.n_clusters + i_c].set(conf_eff_mean[g, :])
            variance_by_comp = variance_by_comp.at[self.n_clusters + i_c].set(conf_eff_variance[g, :])

        with numpyro.plate(f"plate_objects_lh_{p_name}", self.shapes.n_objects, dim=-2):
            with numpyro.plate(f"plate_features_lh_{p_name}", partition.n_features, dim=-1):
                with numpyro.handlers.mask(mask=~partition.na_values):
                    numpyro.sample(f"x_{p_name}", dist.MixtureSameFamily(
                        mixing_distribution=dist.Categorical(probs=mixture_weights[:, :, partition.feature_indices].transpose((1, 2, 0))),
                        component_distribution=dist.Normal(loc=mean_by_comp.transpose((1, 2, 0)), scale=variance_by_comp.transpose((1, 2, 0)))
                    ), obs=partition.values)

    def add_weights_prior(self, clusters):
        w = numpyro.sample("w", dist.Dirichlet(self.w_prior_conc))
        # w = numpyro.sample("w", dist.Gamma(self.w_prior_conc, 1))
        # After normalization, this is equivalent to  Dirichlet(w_prior_conc)
        # shape: (n_features, n_components)

        varying_weights_per_cluster = False  # TODO: Move to config
        if varying_weights_per_cluster:
            with numpyro.plate("plate_clusters_w", self.n_clusters, dim=-2):
                with numpyro.plate("plate_features_w", self.shapes.n_features, dim=-1):
                    w_cluster = numpyro.sample("w_cluster", dist.Gamma(w[:, 0], 1))
            w_cluster_mixed = clusters @ w_cluster
            # shape: (n_objects, n_features)

            # TODO: aggregating across cluster here and then splitting to get per_cluster_weights below feels redundant. Try to avoid this.

        # Multiply weights with `has_component` to mask out components that are not present in the group and normalize
        w_per_object = w.T[:, None, :] * self.has_component[:, :, None]
        if varying_weights_per_cluster:
            w_per_object = w_per_object.at[0].set(w_cluster_mixed)
        w_per_object = w_per_object / w_per_object.sum(axis=-3, keepdims=True)
        # shape: (n_components, n_objects, n_features)

        clusters_normalized = clusters / self.has_component[0, :, None]                             # (objects, clusters)
        per_cluster_weights = clusters_normalized.T[:, :, None] * w_per_object[:1, :, :]            # (clusters, objects, features)
        mixture_weights = jnp.concat([per_cluster_weights, w_per_object[1:, :, :]], axis=0)  # (clusters+confounders, objects, features)

        return mixture_weights

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