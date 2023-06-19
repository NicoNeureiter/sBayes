#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from sbayes.load_data import Data
from sbayes.model import Model
from sbayes.sampling.mcmc import MCMC
from sbayes.config.config import OperatorsConfig
from sbayes.sampling.operators import (
    Operator,
    GibbsSampleWeights,
    AlterClusterGibbsish,
    AlterClusterGibbsishWide,
    GibbsSampleSource,
    ObjectSelector,
    ResampleSourceMode,
    ClusterJump,
    ClusterEffectProposals,
)


class ClusterMCMC(MCMC):

    """sBayes specific subclass of MCMC for sampling clusters (and other parameters)."""

    def __init__(
        self,
        model: Model,
        data: Data,
        p_grow_connected: float,
        **kwargs
    ):
        """
        Args:
            p_grow_connected: Probability at which grow operator only considers neighbours to add to the cluster

            initial_size: The initial size of a cluster
            **kwargs: Other arguments that are passed on to MCMC
        """
        super().__init__(model=model, data=data, **kwargs)

        # Parameters for operators
        self.p_grow_connected = p_grow_connected
        self.var_proposal_weight = 10
        self.var_proposal_cluster_effect = 20
        self.var_proposal_confounding_effects = 10

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
                weight=0.2 * operators_config.clusters,
                adjacency_matrix=self.data.network.adj_mat,
                model_by_chain=self.posterior_per_chain,
                features=self.data.features.values,
                resample_source=self.model.sample_source,
                resample_source_mode=ResampleSourceMode.GIBBS,
                sample_from_prior=self.sample_from_prior,
                n_changes=1,
            ),
            'gibbsish_sample_cluster_geo': AlterClusterGibbsish(
                weight=0.2 * operators_config.clusters,
                adjacency_matrix=self.data.network.adj_mat,
                model_by_chain=self.posterior_per_chain,
                features=self.data.features.values,
                resample_source=self.model.sample_source,
                resample_source_mode=ResampleSourceMode.GIBBS,
                sample_from_prior=self.sample_from_prior,
                n_changes=1,
                consider_geo_prior=self.model.prior.geo_prior.prior_type == self.model.prior.geo_prior.prior_type.COST_BASED,
            ),
            'gibbsish_sample_cluster_2_geo': AlterClusterGibbsish(
                weight=0.2 * operators_config.clusters,
                adjacency_matrix=self.data.network.adj_mat,
                model_by_chain=self.posterior_per_chain,
                features=self.data.features.values,
                resample_source=self.model.sample_source,
                resample_source_mode=ResampleSourceMode.GIBBS,
                sample_from_prior=self.sample_from_prior,
                consider_geo_prior=self.model.prior.geo_prior.prior_type == self.model.prior.geo_prior.prior_type.COST_BASED,
                n_changes=2,
            ),
            'gibbsish_sample_cluster_wide': AlterClusterGibbsishWide(
                weight=0.15 * operators_config.clusters,
                adjacency_matrix=self.data.network.adj_mat,
                model_by_chain=self.posterior_per_chain,
                features=self.data.features.values,
                resample_source=self.model.sample_source,
                resample_source_mode=ResampleSourceMode.GIBBS,
                sample_from_prior=self.sample_from_prior,
                w_stay=0.6,
                consider_geo_prior=self.model.prior.geo_prior.prior_type == self.model.prior.geo_prior.prior_type.COST_BASED,
            ),
            'gibbsish_sample_cluster_wide_residual': AlterClusterGibbsishWide(
                weight=0.05 * operators_config.clusters,
                adjacency_matrix=self.data.network.adj_mat,
                model_by_chain=self.posterior_per_chain,
                features=self.data.features.values,
                resample_source=self.model.sample_source,
                resample_source_mode=ResampleSourceMode.GIBBS,
                sample_from_prior=self.sample_from_prior,
                w_stay=0.0,
                cluster_effect_proposal=ClusterEffectProposals.residual_counts,
            ),
            # 'cluster_jump': ClusterJump(
            #     weight=0.1 * operators_config.clusters,
            #     model_by_chain=self.posterior_per_chain,
            #     resample_source=True,
            #     sample_from_prior=self.sample_from_prior,
            #     gibbsish=False
            # ),
            'cluster_jump_gibbsish': ClusterJump(
                weight=0.2 * operators_config.clusters,
                model_by_chain=self.posterior_per_chain,
                resample_source=True,
                sample_from_prior=self.sample_from_prior,
                gibbsish=True
            ),

            'gibbs_sample_sources': GibbsSampleSource(
                weight=0.5*operators_config.source,
                model_by_chain=self.posterior_per_chain,
                sample_from_prior=self.sample_from_prior,
                object_selector=ObjectSelector.RANDOM_SUBSET,
                max_size=40,
            ),
            'gibbs_sample_sources_groups': GibbsSampleSource(
                weight=0.5*operators_config.source,
                model_by_chain=self.posterior_per_chain,
                sample_from_prior=self.sample_from_prior,
                object_selector=ObjectSelector.GROUPS,
            ),

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
