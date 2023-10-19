#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import random as _random

import numpy as np

from sbayes.load_data import Data
from sbayes.model import update_categorical_weights, update_gaussian_weights, \
                         update_poisson_weights, update_logitnormal_weights
from sbayes.sampling.conditionals import sample_source_from_prior
from sbayes.sampling.counts import recalculate_feature_counts
from sbayes.sampling.mcmc import MCMC
from sbayes.sampling.state import Sample, Source
from sbayes.sampling.operators import (
    Operator,
    AlterWeights,
    GibbsSampleWeights,
    AlterClusterGibbsish,
    AlterClusterGibbsishWide,
    AlterCluster,
    GibbsSampleSource,
    ObjectSelector, ResampleSourceMode, ClusterJump, ClusterEffectProposals
)
from sbayes.util import get_neighbours, normalize
from sbayes.config.config import OperatorsConfig
import sbayes.model


class ClusterMCMC(MCMC):

    """sBayes specific subclass of MCMC for sampling clusters (and other parameters)."""

    def __init__(
        self,
        model: Model,
        data: Data,
        p_grow_connected: float,
        initial_size: int,
        **kwargs
    ):
        """
        Args:
            p_grow_connected: Probability at which grow operator only considers neighbours to add to the cluster

            initial_size: The initial size of a cluster
            **kwargs: Other arguments that are passed on to MCMC
        """

        # Data
        self.shapes = model.shapes
        self.features = data.features

        # Sampling
        self.p_grow_connected = p_grow_connected

        # Clustering
        self.initial_size = initial_size

        # Confounders and sources
        self.confounders = model.confounders
        self.n_sources = 1 + len(self.confounders)
        self.source_index = self.get_source_index()
        self.n_groups = self.get_groups_per_confounder()

        super().__init__(model=model, data=data, **kwargs)

    def get_groups_per_confounder(self):
        n_groups = dict()
        for k, v in self.confounders.items():
            n_groups[k] = v.n_groups
        return n_groups

    def get_source_index(self):
        source_index = {'cluster_effect': 0, 'confounding_effects': dict()}

        for i, k, in enumerate(self.confounders):
            source_index['confounding_effects'][k] = i+1

        return source_index

    def generate_initial_clusters(self):
        """For each chain generate initial clusters by growing through random grow-steps up to self.min_size,

        Returns:
            np.array: The generated initial clusters.
                shape(n_clusters, n_sites)
        """

        # No clusters in the model --> return empty matrix
        if self.shapes.n_clusters == 0:
            return np.zeros((self.shapes.n_clusters, self.shapes.n_objects), bool)

        initial_clusters = np.zeros((self.shapes.n_clusters, self.shapes.n_objects), bool)
        occupied = np.zeros(self.shapes.n_objects, bool)

        # With many clusters new ones can get stuck due to unfavourable seeds.
        # We perform several attempts to initialize the clusters.
        attempts = 0
        max_attempts = 1000

        while True:
            for i in range(self.shapes.n_clusters):
                try:
                    initial_size = self.initial_size
                    cl, in_cl = self.grow_cluster_of_size_k(k=initial_size, already_in_cluster=occupied)

                except self.ClusterError:
                    # Rerun: Error might be due to an unfavourable seed
                    if attempts < max_attempts:
                        attempts += 1
                        if attempts % 10 == 0 and self.initial_size > 3:
                            self.initial_size -= 1
                            self.logger.warning(f"Reduced 'initial_size' to {self.initial_size} after "
                                                f"{attempts} unsuccessful initialization attempts.")
                        break
                    # Seems there is not enough sites to grow n_clusters of size k
                    else:
                        raise ValueError(f"Failed to add additional cluster. Try fewer clusters "
                                         f"or set initial_sample to None.")

                initial_clusters[i, :] = cl
            else:  # No break -> no cluster exception
                return initial_clusters

    def grow_cluster_of_size_k(self, k, already_in_cluster=None):
        """ This function grows a cluster of size k, excluding any of the objects in <already_in_cluster>.
        Args:
            k (int): The size of the cluster, i.e. the number of sites in the cluster
            already_in_cluster (np.array): All sites already assigned to a cluster (boolean)

        Returns:
            np.array: The newly grown cluster (boolean).
            np.array: all sites already assigned to a cluster (boolean).

        """
        if already_in_cluster is None:
            already_in_cluster = np.zeros(self.shapes.n_objects, bool)

        # Initialize the cluster
        cluster = np.zeros(self.shapes.n_objects, bool)

        # Find all sites that are occupied by a cluster and those that are still free
        sites_occupied = np.nonzero(already_in_cluster)[0]
        sites_free = list(set(range(self.shapes.n_objects)) - set(sites_occupied))

        # Take a random free site and use it as seed for the new cluster
        try:
            i = _random.sample(sites_free, 1)[0]
            cluster[i] = already_in_cluster[i] = 1
        except ValueError:
            raise self.ClusterError

        # Grow the cluster if possible
        for _ in range(k - 1):
            neighbours = get_neighbours(cluster, already_in_cluster, self.data.network.adj_mat)
            if not np.any(neighbours):
                raise self.ClusterError

            # Add a neighbour to the cluster
            site_new = _random.choice(list(neighbours.nonzero()[0]))
            cluster[site_new] = already_in_cluster[site_new] = 1

        return cluster, already_in_cluster

    def generate_initial_weights(self):
        """This function generates initial weights for each feature in the Bayesian additive mixture model

        Returns:
            np.array: weights for cluster_effect and each of the i confounding_effects
        """
        initial_weights = {}
        if self.features.categorical is not None:
            initial_weights['categorical'] = normalize(np.ones((self.shapes.n_features_categorical, self.n_sources)))
        else:
            initial_weights['categorical'] = None

        if self.features.gaussian is not None:
            initial_weights['gaussian'] = normalize(np.ones((self.shapes.n_features_gaussian, self.n_sources)))
        else:
            initial_weights['gaussian'] = None

        if self.features.poisson is not None:
            initial_weights['poisson'] = normalize(np.ones((self.shapes.n_features_poisson, self.n_sources)))
        else:
            initial_weights['poisson'] = None

        if self.features.logitnormal is not None:
            initial_weights['logitnormal'] = normalize(np.ones((self.shapes.n_features_logitnormal, self.n_sources)))
        else:
            initial_weights['logitnormal'] = None

        return initial_weights

    def generate_initial_source(self) -> Source:
        """Generate initial source arrays
            Returns:
                probabilities for states in each group of confounding effect [i]
        """
        initial_source = dict()

        if self.features.categorical is not None:
            initial_source['categorical'] = np.empty((self.shapes.n_objects,
                                                      self.shapes.n_features_categorical,
                                                      self.n_sources), dtype=bool)
        else:
            initial_source['categorical'] = None

        if self.features.gaussian is not None:
            initial_source['gaussian'] = np.empty((self.shapes.n_objects,
                                                   self.shapes.n_features_gaussian,
                                                   self.n_sources), dtype=bool)
        else:
            initial_source['gaussian'] = None

        if self.features.poisson is not None:
            initial_source['poisson'] = np.empty((self.shapes.n_objects,
                                                  self.shapes.n_features_poisson,
                                                  self.n_sources), dtype=bool)
        else:
            initial_source['gaussian'] = None

        if self.features.logitnormal is not None:
            initial_source['logitnormal'] = np.empty((self.shapes.n_objects,
                                                      self.shapes.n_features_logitnormal,
                                                      self.n_sources), dtype=bool)
        else:
            initial_source['logitnormal'] = None

        return Source(**initial_source)

    def generate_initial_confounding_effect(self, conf: str) -> dict[str, np.ndarray | None]:
        """Generates initial state probabilities for each group in confounding effect conf
        Args:
            conf: The confounding effect
        Returns:
            probabilities for states in each group of confounding effect [i]
        """

        n_groups = self.n_groups[conf]
        groups = self.confounders[conf].group_assignment
        initial_confounding_effect = dict()

        if self.features.categorical is not None:
            initial_confounding_effect['categorical'] = np.zeros((n_groups,
                                                                  self.shapes.n_features_categorical,
                                                                  self.shapes.n_states_categorical))

            for g in range(n_groups):
                idx = groups[g].nonzero()[0]
                features_group = self.features.categorical.values[idx, :, :]
                objects_per_state = np.nansum(features_group, axis=0)

                # Compute the MLE for each state and each group in the confounding effect
                # Some groups have only NAs for some features, resulting in a non-defined MLE
                # other groups have only a single state, resulting in an MLE of 1.
                # To avoid both, we add 1 to all applicable states in this case
                # which gives a well-defined initial confounding effect, slightly nudged away from the MLE

                objects_per_state[np.count_nonzero(objects_per_state, axis=1) < 2] += 1
                p = objects_per_state / np.sum(objects_per_state, axis=1, keepdims=True)

                initial_confounding_effect['categorical'][g, :, :] = p

        else:
            initial_confounding_effect['categorical'] = None

        if self.features.gaussian is not None:
            initial_confounding_effect['gaussian'] = np.zeros((n_groups, self.shapes.n_features_gaussian, 2))
            for g in range(n_groups):
                idx = groups[g].nonzero()[0]
                features_group = self.features.gaussian.values[idx, :]

                m = np.nanmean(features_group, axis=0)
                std = np.nanstd(features_group, axis=0)

                initial_confounding_effect['gaussian'][g, :, 0] = m
                initial_confounding_effect['gaussian'][g, :, 1] = std

        else:
            initial_confounding_effect['gaussian'] = None

        if self.features.poisson is not None:
            initial_confounding_effect['poisson'] = np.zeros((n_groups, self.shapes.n_features_poisson))
            for g in range(n_groups):
                idx = groups[g].nonzero()[0]
                features_group = self.features.poisson.values[idx, :]

                r = np.nanmean(features_group, axis=0)
                initial_confounding_effect['poisson'][g, :] = r
        else:
            initial_confounding_effect['poisson'] = None

        if self.features.logitnormal is not None:
            initial_confounding_effect['logitnormal'] = np.zeros((n_groups, self.shapes.n_features_logitnormal, 2))
            for g in range(n_groups):
                idx = groups[g].nonzero()[0]
                features_group = self.features.logitnormal.values[idx, :]

                m = np.nanmean(features_group, axis=0)
                std = np.nanstd(features_group, axis=0)

                initial_confounding_effect['logitnormal'][g, :, 0] = m
                initial_confounding_effect['logitnormal'][g, :, 1] = std

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

        # Confounding effects
        initial_confounding_effects = dict()
        for conf_name in self.confounders:
            initial_confounding_effects[conf_name] = self.generate_initial_confounding_effect(conf_name)

        # Source arrays
        if self.model.sample_source:
            initial_source = self.generate_initial_source()
        else:
            initial_source = None

        feature_counts = {'clusters': np.zeros((self.shapes.n_clusters,
                                                self.shapes.n_features_categorical,
                                                self.shapes.n_states_categorical)),
                          **{conf: np.zeros((n_groups, self.shapes.n_features_categorical,
                                             self.shapes.n_states_categorical))
                             for conf, n_groups in self.n_groups.items()}}

        sample = Sample.from_numpy_arrays(
            clusters=initial_clusters,
            weights=initial_weights,
            # confounding_effects=initial_confounding_effects,
            confounders=self.data.confounders,
            source=initial_source,
            feature_counts=feature_counts,
            chain=c,
            model_shapes=self.model.shapes,
        )

        for t in ['categorical', 'gaussian', 'poisson', 'logitnormal']:
            assert ~np.any(np.isnan(initial_weights[t])), initial_weights[t]

        self.generate_inital_source_with_gibbs(sample)

        sample.everything_changed()

        return sample

    class ClusterError(Exception):
        pass

    def generate_inital_source_with_gibbs(self, sample):

        # Generate the initial source using a Gibbs sampling step
        sample.everything_changed()
        source = sample_source_from_prior(sample)

        feature_types = ["categorical", "gaussian", "poisson", "logitnormal"]

        full_source_operator = GibbsSampleSource(
            weight=1,
            model_by_chain=self.posterior_per_chain,
            sample_from_prior=False,
            object_selector=ObjectSelector.ALL
        )

        for ft in feature_types:
            if getattr(sample, ft) is not None:
                source[ft][getattr(self.data.features, ft).na_values] = 0
                getattr(sample, ft).source.set_value(source[ft])

                if ft == "categorical":
                    # Recalculate the counts for categorical features
                    recalculate_feature_counts(self.features.categorical.values, sample)

                update_weights = getattr(sbayes.model, "update_" + ft + "_weights")
                w = update_weights(sample, caching=False)
                s = getattr(sample, ft).source.value
                assert np.all(s <= (w > 0)), np.max(w)

        # Propose a new sample
        new_sample = full_source_operator.function(sample)[0]

        for ft in feature_types:
            if getattr(sample, ft) is not None:
                source[ft] = getattr(new_sample, ft).source.value
                getattr(sample, ft).source.set_value(source[ft])
                if ft == "categorical":
                    recalculate_feature_counts(self.features.categorical.values, sample)

        return source

    def get_operators(self, operators_config: OperatorsConfig) -> dict[str, Operator]:
        """Get all relevant operator functions for proposing MCMC update steps and their probabilities
        Args:
            operators_config: dictionary with names of all operators (keys) and their weights (values)
        Returns:
            Dictionary mapping operator names to operator objects
        """

        operators = {
            # 'sample_cluster': AlterCluster(
            #     weight=operators_config.clusters,
            #     adjacency_matrix=self.data.network.adj_mat,
            #     p_grow_connected=self.p_grow_connected,
            #     model_by_chain=self.posterior_per_chain,
            #     resample_source=self.model.sample_source,
            #     resample_source_mode=ResampleSourceMode.PRIOR,
            #     sample_from_prior=self.sample_from_prior,
            # ),
            'gibbsish_sample_cluster': AlterClusterGibbsish(
                # weight=0.2 * operators_config.clusters,
                weight=1 * operators_config.clusters,
                adjacency_matrix=self.data.network.adj_mat,
                model_by_chain=self.posterior_per_chain,
                features=self.features,
                resample_source=self.model.sample_source,
                resample_source_mode=ResampleSourceMode.GIBBS,
                sample_from_prior=self.sample_from_prior,
                n_changes=1,
            ),
            # 'gibbsish_sample_cluster_geo': AlterClusterGibbsish(
            #     weight=0.2 * operators_config.clusters,
            #     adjacency_matrix=self.data.network.adj_mat,
            #     model_by_chain=self.posterior_per_chain,
            #     features=self.features,
            #     resample_source=self.model.sample_source,
            #     resample_source_mode=ResampleSourceMode.GIBBS,
            #     sample_from_prior=self.sample_from_prior,
            #     n_changes=1,
            #     consider_geo_prior=self.model.prior.geo_prior.prior_type == self.model.prior.geo_prior.prior_type.COST_BASED,
            # ),
            # 'gibbsish_sample_cluster_2_geo': AlterClusterGibbsish(
            #     weight=0.2 * operators_config.clusters,
            #     adjacency_matrix=self.data.network.adj_mat,
            #     model_by_chain=self.posterior_per_chain,
            #     features=self.features,
            #     resample_source=self.model.sample_source,
            #     resample_source_mode=ResampleSourceMode.GIBBS,
            #     sample_from_prior=self.sample_from_prior,
            #     consider_geo_prior=self.model.prior.geo_prior.prior_type == self.model.prior.geo_prior.prior_type.COST_BASED,
            #     n_changes=2,
            # ),
            # 'gibbsish_sample_cluster_wide': AlterClusterGibbsishWide(
            #     weight=0.15 * operators_config.clusters,
            #     adjacency_matrix=self.data.network.adj_mat,
            #     model_by_chain=self.posterior_per_chain,
            #     features=self.features,
            #     resample_source=self.model.sample_source,
            #     resample_source_mode=ResampleSourceMode.GIBBS,
            #     sample_from_prior=self.sample_from_prior,
            #     w_stay=0.6,
            #     consider_geo_prior=self.model.prior.geo_prior.prior_type == self.model.prior.geo_prior.prior_type.COST_BASED,
            # ),
            # 'gibbsish_sample_cluster_wide_residual': AlterClusterGibbsishWide(
            #     weight=0.05 * operators_config.clusters,
            #     adjacency_matrix=self.data.network.adj_mat,
            #     model_by_chain=self.posterior_per_chain,
            #     features=self.features,
            #     resample_source=self.model.sample_source,
            #     resample_source_mode=ResampleSourceMode.GIBBS,
            #     sample_from_prior=self.sample_from_prior,
            #     w_stay=0.0,
            #     cluster_effect_proposal=ClusterEffectProposals.residual_counts,
            # ),
            # 'cluster_jump': ClusterJump(
            #     weight=0.1 * operators_config.clusters,
            #     model_by_chain=self.posterior_per_chain,
            #     features=self.features,
            #     resample_source=True,
            #     sample_from_prior=self.sample_from_prior,
            #     gibbsish=False
            # ),

            # 'cluster_jump_gibbsish': ClusterJump(
            #     weight=0.2 * operators_config.clusters if self.shapes.n_clusters > 1 else 0.0,
            #     model_by_chain=self.posterior_per_chain,
            #     features=self.features,
            #     resample_source=True,
            #     sample_from_prior=self.sample_from_prior,
            #     gibbsish=True
            # ),

            'gibbs_sample_sources': GibbsSampleSource(
                weight=0.5*operators_config.source,
                model_by_chain=self.posterior_per_chain,
                sample_from_prior=self.sample_from_prior,
                object_selector=ObjectSelector.RANDOM_SUBSET,
                max_size=4,
            ),
            # 'gibbs_sample_sources_groups': GibbsSampleSource(
            #     weight=0.5*operators_config.source,
            #     model_by_chain=self.posterior_per_chain,
            #     sample_from_prior=self.sample_from_prior,
            #     object_selector=ObjectSelector.GROUPS,
            # ),

            'gibbs_sample_weights': GibbsSampleWeights(
                weight=operators_config.weights,
                model_by_chain=self.posterior_per_chain,
                sample_from_prior=self.sample_from_prior,
            ),
        }

        normalize_operator_weights(operators)

        return operators


def normalize_operator_weights(operators: dict[str, Operator]):
    total = sum(op.weight for op in operators.values())
    for op in operators.values():
        op.weight /= total
