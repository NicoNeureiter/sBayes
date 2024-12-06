#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from tokenize import group

import numpy as np
import jax.numpy as jnp
from jax import random
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist
from sqlalchemy.testing.plugin.plugin_base import config

from sbayes.model.model_shapes import ModelShapes
from sbayes.model.prior import Prior
from sbayes.config.config import ModelConfig
from sbayes.load_data import Data
from sbayes.util import onehot_to_integer_encoding


def indent(text, amount, ch=' '):
    padding = amount * ch
    return ''.join(padding + line for line in text.splitlines(True))

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

        # Set geo-prior flag (todo: get from config)
        self.activate_geo_prior = 0

        # Extract prior concentrations for the cluster and confounder effects
        self.clust_eff_prior_conc = self.prior.prior_cluster_effect.concentration_array
        self.conf_eff_prior_conc = [
            self.prior.prior_confounding_effects[conf]._concentration_array
            for conf in self.confounders
        ]

        # Adapt concentration arrays some practical issues in the numpyro model
        for i_c, conc in enumerate(self.conf_eff_prior_conc):
            # Add a dummy dimension for the "non-assigned" groups
            conc = np.concatenate([conc, jnp.ones((1, *conc.shape[1:]))], axis=0)
            # Replace zeros by epsilon
            conc[conc == 0.0] = 1e-10
            # Place back into list of concentrations
            self.conf_eff_prior_conc[i_c] = conc

            # print(i_c, conc)

        # self.cluster_prior_probs = jnp.ones(self.n_clusters + 1) / (self.n_clusters + 1)  # +1 for "non-assigned"
        self.cluster_prior_probs = jnp.ones(self.n_clusters + 1)  # +1 for "non-assigned"

        self.clust_eff_prior_conc.setflags(write=True)
        self.clust_eff_prior_conc[self.clust_eff_prior_conc == 0.0] = 1e-10

        # Create a list of group names and group assignments for the cluster and confounder effects
        self.group_names = [[f"cluster_{i_c}" for i_c in range(self.n_clusters)]]
        self.group_assignments = -jnp.ones((self.shapes.n_components, self.shapes.n_sites), dtype=int)
        for i_c, confounder in enumerate(self.confounders.values(), start=1):
            self.group_names.append(confounder.group_names)

            # Translate binary (one-hot) group assignments to integer values
            group_indexes = onehot_to_integer_encoding(confounder.group_assignment, none_index=-1, axis=0)
            self.group_assignments = self.group_assignments.at[i_c].set(group_indexes)

        self.p_data_by_comp = jnp.empty((self.shapes.n_components, self.shapes.n_sites, self.shapes.n_features, self.shapes.n_states))

        self.w_prior_conc = jnp.ones(self.shapes.n_components)

        self.features_int = onehot_to_integer_encoding(self.data.features.values, none_index=-1, axis=-1)
        self.not_missing = (self.features_int != -1)

        self.has_component = jnp.astype(self.group_assignments == -1, jnp.float32)

    def get_model(self):

        # Sample continuous area assignment from a Dirichlet distribution
        with numpyro.plate("plate_objects", self.shapes.n_sites, dim=-1):
            # z = numpyro.sample("z", dist.Dirichlet(0.05 * self.cluster_prior_probs.at[-1].set(20)))
            z = numpyro.sample("z", dist.Dirichlet(self.cluster_prior_probs))

        has_component = self.has_component.at[0, :].set(1 - z[..., -1])

        clusters = z[..., :-1]
        cluster_size = jnp.sum(clusters, axis=-2)
        # numpyro.factor('size_prior', -10 * np.sum(cluster_size**2))

        # Add geo-prior as a factor to the model
        if self.activate_geo_prior:
            dist_mat = self.prior.geo_prior.cost_matrix
            # z_peaky = softmax(self.shapes.n_sites * z.T, axis=1)**2
            # z_peaky = z.T**2
            # avg_dist_to_cluster = z_peaky.dot(dist_mat)
            # log_geo_priors = -avg_dist_to_cluster / geo_prior.scale / 2

            clusters_normed = clusters / cluster_size[None, :]
            same_cluster_prob = (clusters_normed) @ clusters_normed.T
            expected_distance_in_clusters = np.sum(same_cluster_prob * dist_mat)
            log_geo_priors = -expected_distance_in_clusters / 100_000

            # Add geo-prior as a factor to the model
            numpyro.factor('geo_prior', log_geo_priors)

        with numpyro.plate("plate_features", self.shapes.n_features, dim=-1):
            w = numpyro.sample("w", dist.Dirichlet(self.w_prior_conc))
            # shape: (n_features, n_components)

        # Multiply weights with `has_component` to mask out components that are not present in the group and normalize
        w_per_object = w.T[:, None, :] * has_component[:, :, None]
        w_per_object = w_per_object / w_per_object.sum(axis=-3, keepdims=True)
        # shape: (n_components, n_objects, n_features)

        # Take the cashed "p_data_by_comp" and update it with the new component effects
        p_data_by_comp = self.p_data_by_comp

        # Sample and assign cluster effects
        with numpyro.plate(f"plate_clusters", self.n_clusters, dim=-2):
            with numpyro.plate(f"plate_features", self.shapes.n_features, dim=-1):
                cluster_effect = numpyro.sample(f"cluster_effect", dist.Dirichlet(self.clust_eff_prior_conc))
                # shape: (n_clusters, n_features, n_states)

        p_data_by_comp = p_data_by_comp.at[0].set(
            jnp.einsum('nk,kfs->nfs', z[..., :-1], cluster_effect)
        )

        # Sample and assign confounding effects
        for i_c in range(1, self.shapes.n_confounders + 1):
            n_groups = len(self.group_names[i_c])
            with numpyro.plate(f"plate_groups_{i_c}", n_groups + 1, dim=-2):  # +1 for "non-assigned"
                with numpyro.plate(f"plate_features_{i_c}", self.shapes.n_features, dim=-1):
                    conf_effect = numpyro.sample(f"conf_eff_{i_c}", dist.Dirichlet(self.conf_eff_prior_conc[i_c - 1]))
                    # shape: (n_groups + 1, n_features, n_states)
            p_data_by_comp = p_data_by_comp.at[i_c].set(
                conf_effect[self.group_assignments[i_c], :, :]
            )
            # shape: (n_objects, n_features, n_states) for each component

        # Define mixture likelihood
        p_data_mixed = jnp.einsum('kif,kifs->ifs', w_per_object, p_data_by_comp)
        # shape: (n_objects, n_features, n_states)

        with numpyro.plate("plate_objects", self.shapes.n_sites, dim=-2):
            with numpyro.plate("plate_features", self.shapes.n_features, dim=-1):
                with numpyro.handlers.mask(mask=self.not_missing):
                    numpyro.sample("x", dist.Categorical(probs=p_data_mixed), obs=self.features_int)

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
