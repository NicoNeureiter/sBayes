#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import random as _random

import numpy as np

from sbayes.sampling.mcmc import MCMC
from sbayes.sampling.state import Sample
from sbayes.sampling.operators import (
    Operator,
    AlterWeights,
    AlterClusterEffect,
    GibbsSampleWeights,
    AlterClusterGibbsish,
    AlterClusterGibbsish2,
    AlterCluster,
    GibbsSampleSource,
    AlterConfoundingEffects,
    GibbsSampleClusterEffect,
    GibbsSampleConfoundingEffects,
)
from sbayes.util import get_neighbours, normalize, get_max_size_list
from sbayes.config.config import OperatorsConfig


class ClusterMCMC(MCMC):

    def __init__(self, model, data, p_grow_connected, initial_sample, initial_size,
                 **kwargs):
        """
        Args:
            p_grow_connected (float): Probability at which grow operator only considers neighbours to add to the cluster
            initial_sample (Sample): The starting sample
            initial_size (int): The initial size of a cluster
            **kwargs: Other arguments that are passed on to MCMC
        """

        # Data
        self.features = data.features.values
        self.applicable_states = data.features.states
        self.n_states_by_feature = np.sum(data.features.states, axis=-1)
        self.n_features = self.features.shape[1]
        self.n_sites = self.features.shape[0]

        # Locations and network
        self.locations = data.network.locations
        self.adj_mat = data.network.adj_mat

        # Sampling
        self.p_grow_connected = p_grow_connected

        # Clustering
        self.n_clusters = model.n_clusters
        self.min_size = model.min_size
        self.max_size = model.max_size
        self.initial_size = initial_size

        # Confounders and sources
        self.confounders = model.confounders
        self.n_sources = 1 + len(self.confounders)
        self.source_index = self.get_source_index()
        self.n_groups = self.get_groups_per_confounder()

        super().__init__(model=model, data=data, **kwargs)

        # Initial Sample
        if initial_sample is None:
            # self.initial_sample = Sample.empty_sample(self.confounders)
            self.initial_sample = None
        else:
            self.initial_sample = initial_sample

        # Variance of the proposal distribution
        self.var_proposal_weight = 10
        self.var_proposal_cluster_effect = 20
        self.var_proposal_confounding_effects = 10

    def get_groups_per_confounder(self):
        n_groups = dict()
        for k, v in self.confounders.items():
            n_groups[k] = len(v)
        return n_groups

    def get_source_index(self):
        source_index = {'cluster_effect': 0, 'confounding_effects': dict()}

        for i, k, in enumerate(self.confounders):
            source_index['confounding_effects'][k] = i+1

        return source_index

    def generate_initial_clusters(self):
        """For each chain (c) generate initial clusters by
        A) growing through random grow-steps up to self.min_size,
        B) using the last sample of a previous run of the MCMC

        Returns:
            np.array: The generated initial clusters.
                shape(n_clusters, n_sites)
        """

        # If there are no clusters in the model, return empty matrix
        if self.n_clusters == 0:
            return np.zeros((self.n_clusters, self.n_sites), bool)

        occupied = np.zeros(self.n_sites, bool)
        initial_clusters = np.zeros((self.n_clusters, self.n_sites), bool)
        n_generated = 0

        # B: When clusters from a previous run exist use them as the initial sample
        if self.initial_sample is not None:
            for i in range(self.initial_sample.clusters.n_clusters):
                initial_clusters[i, :] = self.initial_sample.clusters.value[i]
                occupied += self.initial_sample.clusters.value[i]
                n_generated += 1

        not_initialized = range(n_generated, self.n_clusters)

        # A: Grow the remaining clusters
        # With many clusters new ones can get stuck due to unfavourable seeds.
        # We perform several attempts to initialize the clusters.
        attempts = 0
        max_attempts = 1000

        while True:
            for i in not_initialized:
                try:
                    initial_size = self.initial_size
                    cl, in_cl = self.grow_cluster_of_size_k(k=initial_size, already_in_cluster=occupied)

                except self.ClusterError:
                    # Rerun: Error might be due to an unfavourable seed
                    if attempts < max_attempts:
                        attempts += 1
                        not_initialized = range(n_generated, self.n_clusters)
                        break
                    # Seems there is not enough sites to grow n_clusters of size k
                    else:
                        raise ValueError(f"Failed to add additional cluster. Try fewer clusters "
                                         f"or set initial_sample to None.")

                n_generated += 1
                initial_clusters[i, :] = cl
                occupied = in_cl

            if n_generated == self.n_clusters:
                return initial_clusters

    def grow_cluster_of_size_k(self, k, already_in_cluster=None):
        """ This function grows a cluster of size k excluding any of the sites in <already_in_cluster>.
        Args:
            k (int): The size of the cluster, i.e. the number of sites in the cluster
            already_in_cluster (np.array): All sites already assigned to a cluster (boolean)

        Returns:
            np.array: The newly grown cluster (boolean).
            np.array: all sites already assigned to a cluster (boolean).

        """
        if already_in_cluster is None:
            already_in_cluster = np.zeros(self.n_sites, bool)

        # Initialize the cluster
        cluster = np.zeros(self.n_sites, bool)

        # Find all sites that are occupied by a cluster and those that are still free
        sites_occupied = np.nonzero(already_in_cluster)[0]
        sites_free = list(set(range(self.n_sites)) - set(sites_occupied))

        # Take a random free site and use it as seed for the new cluster
        try:
            i = _random.sample(sites_free, 1)[0]
            cluster[i] = already_in_cluster[i] = 1
        except ValueError:
            raise self.ClusterError

        # Grow the cluster if possible
        for _ in range(k - 1):
            neighbours = get_neighbours(cluster, already_in_cluster, self.adj_mat)
            if not np.any(neighbours):
                raise self.ClusterError

            # Add a neighbour to the cluster
            site_new = _random.choice(neighbours.nonzero()[0])
            cluster[site_new] = already_in_cluster[site_new] = 1

        return cluster, already_in_cluster

    def generate_initial_weights(self):
        """This function generates initial weights for the Bayesian additive mixture model, either by
        A) proposing them (randomly) from scratch
        B) using the last sample of a previous run of the MCMC
        Weights are in log-space and not normalized.

        Returns:
            np.array: weights for cluster_effect and each of the i confounding_effects
            """

        # B: Use weights from a previous run
        if self.initial_sample is not None:
            initial_weights = self.initial_sample.weights.value

        # A: Initialize new weights
        else:
            initial_weights = np.full((self.n_features, self.n_sources), 1.)

        return normalize(initial_weights)

    def generate_initial_cluster_effect(self, initial_clusters):
        """This function generates initial state probabilities for each of the clusters, either by
        A) proposing them (randomly) from scratch
        B) using the last sample of a previous run of the MCMC
        Probabilities are in log-space and not normalized.
        Args:
            initial_clusters: The assignment of sites to clusters
            (n_clusters, n_sites)
        Returns:
            np.array: probabilities for categories in each cluster
                shape (n_clusters, n_features, max(n_states))
        """
        # We place the cluster_effect of all features in one array, even though not all have the same number of states
        initial_cluster_effect = np.zeros((self.n_clusters, self.n_features, self.features.shape[2]))
        n_generated = 0

        # B: Use cluster_effect from a previous run
        if self.initial_sample is not None:

            for i in range(self.initial_sample.cluster_effect.n_groups):
                initial_cluster_effect[i, :] = self.initial_sample.cluster_effect.value[i]
                n_generated += 1

        not_initialized = range(n_generated, self.n_clusters)

        # A: Initialize a new cluster_effect using a value close to the MLE of the current cluster
        for i in not_initialized:
            idx = initial_clusters[i].nonzero()[0]
            features_cluster = self.features[idx, :, :]

            sites_per_state = np.nansum(features_cluster, axis=0)

            # Some clusters have nan for all states, resulting in a non-defined MLE
            # other clusters have only a single state, resulting in an MLE including 1.
            # to avoid both, we add 1 to all applicable states of each feature,
            # which gives a well-defined initial cluster_effect slightly nudged away from the MLE

            sites_per_state[np.isnan(sites_per_state)] = 0
            sites_per_state[self.applicable_states] += 1

            site_sums = np.sum(sites_per_state, axis=1)
            cluster_effect = sites_per_state / site_sums[:, np.newaxis]

            initial_cluster_effect[i, :, :] = cluster_effect

        return initial_cluster_effect

    def generate_initial_confounding_effect(self, conf: str):
        """This function generates initial state probabilities for each group in confounding effect [i], either by
        A) using the MLE
        B) using the last sample of a previous run of the MCMC
        Probabilities are in log-space and not normalized.
        Args:
            conf: The confounding effect [i]
        Returns:
            np.array: probabilities for states in each group of confounding effect [i]
                shape (n_groups, n_features, max(n_states))
        """

        n_groups = self.n_groups[conf]
        groups = self.data.confounders[conf].group_assignment

        initial_confounding_effect = np.zeros((n_groups, self.n_features, self.features.shape[2]))

        # B: Use confounding_effect from a previous run
        if self.initial_sample is not None:
            for i in range(self.initial_sample.confounding_effects[conf].n_groups):
                initial_confounding_effect[i, :] = self.initial_sample.confounding_effects[conf].value[i]

        # A: Initialize new confounding_effect using the MLE
        else:
            for g in range(n_groups):

                idx = groups[g].nonzero()[0]
                features_group = self.features[idx, :, :]

                sites_per_state = np.nansum(features_group, axis=0)

                # Compute the MLE for each state and each group in the confounding effect
                # Some groups have only NAs for some features, resulting in a non-defined MLE
                # other groups have only a single state, resulting in an MLE including 1.
                # To avoid both, we add 1 to all applicable states of each feature,
                # which gives a well-defined initial confounding effect, slightly nudged away from the MLE

                sites_per_state[np.isnan(sites_per_state)] = 0
                sites_per_state[self.applicable_states] += 1

                state_sums = np.sum(sites_per_state, axis=1)
                p_group = sites_per_state / state_sums[:, np.newaxis]
                initial_confounding_effect[g, :, :] = p_group

        return initial_confounding_effect

    def generate_initial_sample(self, c=0):
        """Generate initial Sample object (clusters, weights, cluster_effect, confounding_effects)
        Kwargs:
            c (int): index of the MCMC chain
        Returns:
            Sample: The generated initial Sample
        """
        # Clusters
        initial_clusters = self.generate_initial_clusters()

        # Weights
        initial_weights = self.generate_initial_weights()

        # Areal effect
        initial_cluster_effect = self.generate_initial_cluster_effect(initial_clusters)

        # Confounding effects
        initial_confounding_effects = dict()
        for k, v in self.confounders.items():
            initial_confounding_effects[k] = self.generate_initial_confounding_effect(k)

        if self.model.sample_source:
            initial_source = np.empty((self.n_sites, self.n_features, self.n_sources), dtype=bool)
        else:
            initial_source = None

        sample = Sample.from_numpy_arrays(
            clusters=initial_clusters,
            weights=initial_weights,
            cluster_effect=initial_cluster_effect,
            confounding_effects=initial_confounding_effects,
            confounders=self.data.confounders,
            source=initial_source,
            chain=c,
        )

        assert ~np.any(np.isnan(initial_weights)), initial_weights

        # Generate the initial source using a Gibbs sampling step
        if self.model.sample_source:
            sample.everything_changed()
            sample.source.set_value(
                self.callable_operators['gibbs_sample_sources'].function(sample)[0].source.value
            )

        sample.everything_changed()
        return sample

    # @staticmethod
    # def get_removal_candidates(cluster):
    #     """Finds sites which can be removed from the given zone.
    #     Args:
    #         cluster (np.array): The zone for which removal candidates are found.
    #             shape(n_sites)
    #     Returns:
    #         (list): Index-list of removal candidates.
    #     """
    #     return cluster.nonzero()[0]

    class ClusterError(Exception):
        pass

    def get_operators(self, operators_config: OperatorsConfig) -> dict[str, Operator]:
        """Get all relevant operator functions for proposing MCMC update steps and their probabilities
        Args:
            operators_config: dictionary with names of all operators (keys) and their weights (values)
        Returns:
            Dictionary mapping operator names to operator objects
        """

        if self.model.sample_source:

            operators = {
                # 'sample_cluster': AlterCluster(
                #     weight=operators_config.clusters,
                #     adjacency_matrix=self.data.network.adj_mat,
                #     p_grow_connected=0.5,
                #     model_by_chain=self.posterior_per_chain,
                #     resample_source=True,
                #     sample_from_prior=True,
                # ),
                'gibbsish_sample_cluster': AlterClusterGibbsish(
                    weight=0.5 * operators_config.clusters,
                    adjacency_matrix=self.data.network.adj_mat,
                    p_grow_connected=self.p_grow_connected,
                    model_by_chain=self.posterior_per_chain,
                    features=self.features,
                    resample_source=self.model.sample_source,
                    sample_from_prior=self.sample_from_prior,
                ),
                'gibbsish_sample_cluster_2"': AlterClusterGibbsish2(
                    weight=0.5 * operators_config.clusters,
                    adjacency_matrix=self.data.network.adj_mat,
                    p_grow_connected=self.p_grow_connected,
                    model_by_chain=self.posterior_per_chain,
                    features=self.features,
                    resample_source=self.model.sample_source,
                    sample_from_prior=self.sample_from_prior,
                ),
                'gibbs_sample_sources': GibbsSampleSource(
                    weight=operators_config.source,
                    model_by_chain=self.posterior_per_chain,
                    sample_from_prior=self.sample_from_prior
                ),
                'gibbs_sample_weights': GibbsSampleWeights(
                    weight=operators_config.weights,
                    model_by_chain=self.posterior_per_chain,
                    sample_from_prior=self.sample_from_prior,
                ),
                'gibbs_sample_cluster_effect': GibbsSampleClusterEffect(
                    weight=operators_config.cluster_effect,
                    model_by_chain=self.posterior_per_chain,
                    applicable_states=self.applicable_states,
                    sample_from_prior=self.sample_from_prior,
                ),
            }

            r = float(1 / len(self.model.confounders))
            for k in self.model.confounders:
                op_name = f"gibbs_sample_confounding_effects_{k}"
                operators[op_name] = GibbsSampleConfoundingEffects(
                    weight=r * operators_config.confounding_effects,
                    confounder=k,
                    source_index=self.source_index['confounding_effects'][k],
                    model_by_chain=self.posterior_per_chain,
                    applicable_states=self.applicable_states,
                    sample_from_prior=self.sample_from_prior,
                )

        else:
            operators = {
                'sample_cluster': AlterCluster(
                    weight=operators_config.clusters,
                    adjacency_matrix=self.data.network.adj_mat,
                    p_grow_connected=self.p_grow_connected,
                    model_by_chain=self.posterior_per_chain,
                    resample_source=self.model.sample_source and not self.sample_from_prior,
                    sample_from_prior=self.sample_from_prior,
                ),
                'alter_weights': AlterWeights(operators_config.weights),
                'alter_cluster_effect': AlterClusterEffect(
                    weight=operators_config.cluster_effect,
                    applicable_states=self.data.features.states,
                )
            }

            r = float(1 / len(self.model.confounders))
            for k in self.model.confounders:
                op_name = "alter_confounding_effects_" + str(k)
                operators.update({
                    op_name: AlterConfoundingEffects(
                        weight=operators_config.confounding_effects * r,
                        applicable_states=self.data.features.states,
                        confounder=k,
                    )
                })

        normalize_operator_weights(operators)
        return operators


def normalize_operator_weights(operators: dict):
    total = sum(op['weight'] for op in operators.values())
    for op in operators.values():
        op['weight'] /= total


class ClusterMCMCWarmup(ClusterMCMC):

    IS_WARMUP = True

    def __init__(self, **kwargs):
        super(ClusterMCMCWarmup, self).__init__(**kwargs)

        # In warmup chains can have a different max_size for clusters
        self.max_size = get_max_size_list(
            start=int((self.initial_size + self.max_size) / 2),
            end=int(1.5*self.max_size),
            n_total=self.n_chains,
            k_groups=4
        )

        # Some chains only have connected steps, whereas others also have random steps
        self.p_grow_connected = _random.choices(
            population=[0.95, self.p_grow_connected],
            k=self.n_chains
        )
