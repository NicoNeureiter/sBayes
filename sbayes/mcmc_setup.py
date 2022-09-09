#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the MCMC process """
from __future__ import annotations

from sbayes.sampling.sbayes_sampling import ClusterMCMC, ClusterMCMCWarmup
from sbayes.sampling.state import Sample
from sbayes.model import Model
from sbayes.sampling.loggers import ResultsLogger, ParametersCSVLogger, ClustersLogger, LikelihoodLogger, \
    OperatorStatsLogger
from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data


class MCMCSetup:
    def __init__(self, data: Data, experiment: Experiment):

        # Retrieve the data
        self.data = data

        # Retrieve the configurations
        self.config = experiment.config

        # Paths
        self.path_log = experiment.path_results / 'experiment.log'
        self.path_results = experiment.path_results

        # Create the model to sample from
        self.model = Model(data=self.data, config=self.config.model)

        # Samples
        self.sampler = None
        self.samples = None
        self.sample_from_warm_up = None

        self.logger = experiment.logger

    def log_setup(self):
        mcmc_cfg = self.config.mcmc
        wu_cfg = mcmc_cfg.warmup
        op_cfg = mcmc_cfg.operators
        self.logger.info(self.model.get_setup_message())
        self.logger.info(f'''
MCMC SETUP
##########################################
MCMC with {mcmc_cfg.steps} steps and {mcmc_cfg.samples} samples
Warm-up: {wu_cfg.warmup_chains} chains exploring the parameter space in {wu_cfg.warmup_steps} steps
Ratio of cluster steps (growing, shrinking, swapping clusters): {op_cfg.clusters}
Ratio of weight steps (changing weights): {op_cfg.weights}
Ratio of cluster_effect steps (changing probabilities in clusters): {op_cfg.cluster_effect}
Ratio of confounding_effects steps (changing probabilities in confounders): {op_cfg.confounding_effects}''')
        if self.model.sample_source:
            self.logger.info(f'Ratio of source steps (changing source component assignment): {op_cfg.source}')
        self.logger.info('\n')

    def sample(self, initial_sample: Sample | None = None, run: int = 1):
        mcmc_config = self.config.mcmc

        if initial_sample is None:
            initial_sample = self.sample_from_warm_up

        sample_loggers = self.get_sample_loggers(run=run)

        self.sampler = ClusterMCMC(
            data=self.data,
            model=self.model,
            sample_loggers=sample_loggers,
            initial_sample=initial_sample,
            operators=mcmc_config.operators,
            p_grow_connected=mcmc_config.grow_to_adjacent,
            initial_size=mcmc_config.init_objects_per_cluster,
            sample_from_prior=mcmc_config.sample_from_prior,
            logger=self.logger,
        )

        self.sampler.generate_samples(mcmc_config.steps, mcmc_config.samples)

        self.samples = self.sampler.statistics  # TODO do we still need this?
        self.sampler.print_statistics()

    def warm_up(self):
        mcmc_config = self.config.mcmc
        warmup = ClusterMCMCWarmup(
            data=self.data,
            model=self.model,
            sample_loggers=[],
            n_chains=mcmc_config.warmup.warmup_chains,
            operators=mcmc_config.operators,
            p_grow_connected=mcmc_config.grow_to_adjacent,
            initial_sample=None,
            initial_size=mcmc_config.init_objects_per_cluster,
            sample_from_prior=mcmc_config.sample_from_prior,
            logger=self.logger,
        )

        self.sample_from_warm_up = warmup.generate_samples(n_steps=0,
                                                           n_samples=0,
                                                           warm_up=True,
                                                           warm_up_steps=mcmc_config.warmup.warmup_steps)

    def get_sample_loggers(self, run=1) -> list[ResultsLogger]:
        k = self.model.n_clusters
        base_dir = self.path_results / f'K{k}'
        base_dir.mkdir(exist_ok=True)
        params_path = base_dir / f'stats_K{k}_{run}.txt'
        clusters_path = base_dir / f'clusters_K{k}_{run}.txt'
        likelihood_path = base_dir / f'likelihood_K{k}_{run}.h5'
        op_stats_path = base_dir / f'operator_stats_K{k}_{run}.txt'

        sample_loggers = [
            ParametersCSVLogger(params_path, self.data, self.model),
            ClustersLogger(clusters_path, self.data, self.model),
            OperatorStatsLogger(op_stats_path, self.data, self.model, operators=[])
        ]

        if not self.config.mcmc.sample_from_prior:
            sample_loggers.append(LikelihoodLogger(likelihood_path, self.data, self.model))

        return sample_loggers
