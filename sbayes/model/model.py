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
from sbayes.model.prior import Prior, GeoPrior
from sbayes.config.config import ModelConfig
from sbayes.load_data import Data, Partition, FeatureType
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
        self.min_size = config.prior.objects_per_cluster.min
        self.max_size = config.prior.objects_per_cluster.max
        n_sites, n_features, n_states_f = self.data.features.values.shape

        self.shapes = ModelShapes(
            n_clusters=self.n_clusters,
            n_sites=n_sites,
            n_features=n_features,
            n_states=n_states_f,
            states_per_feature=self.data.features.states,
            n_confounders=len(self.confounders),
            n_groups={name: conf.n_groups for name, conf in self.confounders.items()}
        )

        # Initialize the prior
        self.prior = Prior(shapes=self.shapes, config=self.config.prior, data=data)

        # Set geo-prior flag (todo: add geo prior types from config)
        self.activate_geo_prior = (self.prior.geo_prior.prior_type is not GeoPrior.PriorTypes.UNIFORM)


        # Extract and clip prior concentrations arrays for cluster effect
        self.effect_priors: dict[str, list] = {}
        self.clust_eff_prior_conc = self.prior.prior_cluster_effect.concentration_array
        self.clust_eff_prior_conc.setflags(write=True)
        self.clust_eff_prior_conc[self.clust_eff_prior_conc == 0.0] = 1e-3

        # Extract and clip prior concentration arrays for confounding effects
        self.conf_eff_prior_conc = []
        for i_c, conf in enumerate(self.confounders):
            conc = self.prior.prior_confounding_effects[conf]._concentration_array

            # Add a dummy dimension for the "non-assigned" groups
            conc = np.concatenate([conc, jnp.ones((1, *conc.shape[1:]))], axis=0)

            # Replace zeros by epsilon
            conc[conc == 0.0] = 1e-3

            # Place back into list of concentrations
            self.conf_eff_prior_conc.append(
                jnp.array(conc, dtype=jnp.float32)
            )

        self.conf_eff_prior_conc_by_partition = {}
        for partition in self.data.partitions:
            self.conf_eff_prior_conc_by_partition[partition.name] = [
                conc[:, partition.feature_indices, :partition.n_states]
                for conc in self.conf_eff_prior_conc
            ]

        self.cluster_prior_probs = jnp.ones(self.n_clusters + 1, dtype=jnp.float32)  # +1 for "non-assigned"
        # Alternative cluster prior with equal weight on "cluster" and "not cluster"
        # self.cluster_prior_probs = self.cluster_prior_probs.at[-1].set(self.n_clusters)
        # self.cluster_prior_probs = self.cluster_prior_probs.at[-1].set(2)

        # Create a list of group names and group assignments for the cluster and confounder effects
        self.group_names = [[f"cluster_{i_c}" for i_c in range(self.n_clusters)]]
        self.group_assignments = -jnp.ones((self.shapes.n_components, self.shapes.n_sites), dtype=int)
        for i_c, confounder in enumerate(self.confounders.values(), start=1):
            self.group_names.append(confounder.group_names)

            # Translate binary (one-hot) group assignments to integer values
            group_indexes = onehot_to_integer_encoding(confounder.group_assignment, none_index=-1, axis=0)
            self.group_assignments = self.group_assignments.at[i_c].set(group_indexes)

        self.w_prior_conc = jnp.array(self.prior.prior_weights.concentration_array).astype(jnp.float32)

        self.features = jnp.array(self.data.features.values, dtype=jnp.bool)
        self.features_int = onehot_to_integer_encoding(self.data.features.values, none_index=-1, axis=-1)
        self.not_missing = (self.features_int != -1)

        self.valid_states = self.data.features.states

        self._has_component = jnp.astype(self.group_assignments != -1, jnp.float32)

        self.partitions = self.data.partitions
        self.partition_inices = jnp.stack([p.feature_indices for p in self.partitions])
        # shape: (n_partitions, n_features)

    def get_model(self, no_clusters: bool = False):
        """Return the model function for the sBayes model."""

        # Add the cluster prior to the model
        z = self.add_cluster_prior()

        # `z` has a dummy "no-cluster" at the end, which we want to remove for most purposes
        clusters = z[..., :-1]

        if not no_clusters:
            # Add geo-prior as a factor to the model
            if self.activate_geo_prior:
                self.add_geo_prior(clusters)

            # Update the `cluster` component of `has_component` to reflect the cluster assignments
            self.has_component = self._has_component.at[0, :].set(1 - z[..., -1])


        weights = self.add_weights_prior(clusters)
        # shape: (n_components, n_objects, n_features)

        for partition in self.partitions:
            if partition.feature_type is FeatureType.categorical:
                self.add_partition_categorical(partition, clusters, weights, no_clusters=no_clusters)
            else:
                raise NotImplementedError

    def add_partition_categorical(
        self,
        partition: Partition,
        clusters: jnp.ndarray,      # shape: (n_objects, n_clusters)
        all_weights: jnp.ndarray,       # shape: (n_components, n_objects, n_features)
        no_clusters: bool = False
    ):
        # Short alias for partition_name
        p_name = partition.name

        # p_data_by_comp = self.p_data_by_comp
        p_data_by_comp = jnp.zeros((self.shapes.n_components, self.shapes.n_sites, partition.n_features, partition.n_states))

        # Sample and assign cluster effects
        with numpyro.plate(f"plate_clusters_{p_name}", self.n_clusters, dim=-2):
            with numpyro.plate(f"plate_features_{p_name}", partition.n_features, dim=-1):
                cluster_effect = numpyro.sample(f"cluster_effect_{p_name}",
                                                dist.Dirichlet(self.clust_eff_prior_conc[partition.feature_indices, :partition.n_states]))
                # shape: (n_clusters, n_features, n_states)

        if not no_clusters:
            clusters_normalized = clusters / self.has_component[0, :, None]
            p_data_by_comp = p_data_by_comp.at[0].set(
                jnp.einsum('nk,kfs->nfs', clusters_normalized, cluster_effect)
            )

        # Sample and assign confounding effects
        for i_c in range(1, self.shapes.n_confounders + 1):
            concentration = self.conf_eff_prior_conc_by_partition[p_name][i_c - 1]
            n_groups = len(self.group_names[i_c])
            with numpyro.plate(f"plate_groups_{i_c}", n_groups + 1, dim=-2):  # +1 for "non-assigned"
                with numpyro.plate(f"plate_features_{i_c}_{p_name}", partition.n_features, dim=-1):
                    conf_effect = numpyro.sample(f"conf_eff_{i_c - 1}_{p_name}", dist.Dirichlet(concentration))
                    # shape: (n_groups + 1, n_features, n_states)

            p_data_by_comp = p_data_by_comp.at[i_c].set(conf_effect[self.group_assignments[i_c], :, :])

        # Define mixture likelihood
        partition_weights = all_weights[:, :, partition.feature_indices]
        p_data_mixed = jnp.einsum('kif,kifs->ifs', partition_weights, p_data_by_comp)  # shape: (n_objects, n_features, n_states)

        with numpyro.plate(f"plate_objects_lh_{p_name}", self.shapes.n_sites, dim=-2):
            with numpyro.plate(f"plate_features_lh_{p_name}", partition.n_features, dim=-1):
                with numpyro.handlers.mask(mask=self.not_missing[:, partition.feature_indices]):
                    numpyro.sample(f"x_{p_name}", dist.Categorical(probs=p_data_mixed), obs=partition.features)
        #             # numpyro.sample('x', dist.MixtureSameFamily(likelihood_by_comp, w_per_object), obs=self.features_int)

    def add_cluster_prior(self):

        # LOGISTIC NORMAL MODEL
        with numpyro.plate("plate_objects_1", self.shapes.n_sites, dim=-2):
            with numpyro.plate("plate_clusters_1", self.shapes.n_clusters + 1, dim=-1):
                z_logit = numpyro.sample("z_logit", dist.Normal(
                    loc=jnp.log(self.cluster_prior_probs),
                    scale=jnp.full_like(self.cluster_prior_probs, 3.0)
                ))
                z = numpyro.deterministic("z", softmax(z_logit, axis=-1))

        # with numpyro.plate("plate_objects_1", self.shapes.n_sites, dim=-1):
        #     z = numpyro.sample("z", dist.Dirichlet(1.0 * self.cluster_prior_probs))

        # with numpyro.plate("plate_objects_1", self.shapes.n_sites, dim=-1):
        #     k1 = self.n_clusters + 1
        #     z_int = numpyro.sample("z_int", dist.Categorical(jnp.ones(k1) / k1))
        #     z = numpyro.deterministic("z", jax.nn.one_hot(z_int, k1))

        return z

    def add_geo_prior(self, clusters, geo_prior_type = 'fully_connected'):
        cluster_size = jnp.sum(clusters, axis=-2)

        dist_mat = self.prior.geo_prior.cost_matrix
        if geo_prior_type == 'fully_connected':
            clusters_normed = clusters / cluster_size[None, :]
            # same_cluster_prob = clusters_normed @ clusters_normed.T   # Random language to random language
            # same_cluster_prob = clusters @ clusters.T                 # Expected total distance
            same_cluster_prob = clusters_normed @ clusters.T            # Expected distance to a random language
            expected_distance_in_clusters = jnp.sum(same_cluster_prob * dist_mat)
            log_geo_priors = -expected_distance_in_clusters / 500_000
            # log_geo_priors = -expected_distance_in_clusters / self.prior.geo_prior.scale
        else:
            raise NotImplementedError

            # # Local connectivity prior
            # k = 3
            # nearest_neighbours = jnp.argsort(dist_mat, axis=-1)
            # knn_dist = ...

            # def get_spectrum(C):
            #     L = jnp.fill_diagonal(C, jnp.sum(C, axis=-1), inplace=False)
            #     eigvals = jnp.linalg.eigvals(L)
            #     return jnp.sum(jnp.real(eigvals))
            #
            # # Compute spectral geo-prior
            # connectivities = clusters.T[:, :, None] * clusters.T[:, None, :]  # shape (n_clusters, n_features, n_features)
            # mats = connectivities * dist_mat  # shape (n_clusters, n_features, n_features)
            # eigvals_batched = jax.vmap(get_spectrum, in_axes=0, out_axes=0)(mats)
            # # eigvals_batched = jax.vmap(jnp.linalg.eigvals, in_axes=0, out_axes=0)(mats)
            # log_geo_priors = -jnp.sum(jnp.real(eigvals_batched))

        # Add geo-prior as a factor to the model
        numpyro.factor('geo_prior', log_geo_priors)

    def add_weights_prior(self, clusters):
        # with numpyro.plate("plate_features_w", self.shapes.n_features, dim=-2):
        #     with numpyro.plate("plate_components_w", self.shapes.n_components, dim=-1):
        w = numpyro.sample("w", dist.Gamma(self.w_prior_conc, 1))
        # w = numpyro.sample("w", dist.Dirichlet(self.w_prior_conc))
        # After normalization, this is equivalent to  Dirichlet(w_prior_conc)
        # shape: (n_features, n_components)

        varying_weights_per_cluster = True  # TODO: Move to config
        if varying_weights_per_cluster:
            with numpyro.plate("plate_clusters_w", self.n_clusters, dim=-2):
                with numpyro.plate("plate_features_w", self.shapes.n_features, dim=-1):
                    w_cluster = numpyro.sample("w_cluster", dist.Gamma(w[:, 0], 1))
            w_cluster_mixed = clusters @ w_cluster
            # shape: (n_objects, n_features)

        # Multiply weights with `has_component` to mask out components that are not present in the group and normalize
        w_per_object = w.T[:, None, :] * self.has_component[:, :, None]
        if varying_weights_per_cluster:
            w_per_object = w_per_object.at[0].set(w_cluster_mixed)
        w_per_object = w_per_object / w_per_object.sum(axis=-3, keepdims=True)
        # shape: (n_components, n_objects, n_features)

        return w_per_object

    def generate_initial_params(self, rng_key) -> dict:
        """Initialize the sBayes model and return the model function."""
        simplex_transform = StickBreakingTransform()
        positive_transform = ExpTransform()
        init_params = initialize_model(rng_key, self.get_model)[0].z
        print(init_params.keys())

        for i_c, conf in enumerate(self.confounders.values(), start=1):
            for i_g, g in enumerate(conf.group_assignment):
                prior_counts = self.conf_eff_prior_conc[i_c - 1][i_g]
                feature_counts = jnp.sum(self.data.features.values, axis=0, where=g[:, None, None])
                conf_eff_c_g = normalize(prior_counts + feature_counts, axis=-1)
                conf_eff_c_g_latent = simplex_transform._inverse(conf_eff_c_g)
                init_params[f"conf_eff_{i_c - 1}"] = init_params[f"conf_eff_{i_c - 1}"].at[i_g].set(conf_eff_c_g_latent)

        return init_params

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