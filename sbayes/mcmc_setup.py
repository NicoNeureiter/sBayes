#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the MCMC process """
from __future__ import annotations

import time
from datetime import timedelta
from multiprocessing import Process, Pipe
from pathlib import Path
from time import sleep

from jax import random
import numpy as np
from numpy.typing import NDArray
import numpyro
from numpyro.infer import MCMC, NUTS

from sbayes.results import Results
from sbayes.model import Model
from sbayes.sampling.loggers import ResultsLogger, ParametersCSVLogger, ClustersLogger, LikelihoodLogger, \
    OperatorStatsLogger, StateDumper
from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.util import RNG, process_memory, format_cluster_columns


numpyro.set_host_device_count(4)


class MCMCSetup:

    swap_matrix: NDArray[int] = None
    last_swap_matrix_save: int = 0

    def __init__(self, data: Data, experiment: Experiment):
        self.data = data
        self.config = experiment.config

        # Create the model to sample from
        self.model = Model(data=self.data, config=self.config.model)

        # Set the results directory based on the number of clusters
        self.path_results = experiment.path_results / f'K{self.model.n_clusters}'
        self.path_results.mkdir(exist_ok=True)

        # Samples
        self.sampler = None
        self.samples = None

        self.logger = experiment.logger

        self.t_start = None

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
Ratio of source steps (changing source component assignment): {op_cfg.source}''')
        self.logger.info('\n')

    def sample(
        self,
        initial_sample = None,
        resume: bool = True,
        run: int = 1
    ):
        mcmc_config = self.config.mcmc
        # Sample the model parameters and latent variables using MCMC
        mcmc = MCMC(
            sampler=NUTS(self.model.get_model),
            num_warmup=200,
            num_samples=1000,
            num_chains=1,
            thinning=10,
        )
        mcmc.run(random.PRNGKey(3))
        samples = mcmc.get_samples()

        print(samples)
        print(samples['z'].shape)

        clusters_path = self.path_results / f'clusters_K{self.model.n_clusters}_{run}.txt'
        with open(clusters_path, "w", buffering=1) as clusters_file:
            for clusters in np.array(samples['z']).transpose((0, 2, 1)):
                clusters_binary = np.random.random(clusters.shape) < (clusters ** 2)
                clusters_string = format_cluster_columns(clusters_binary[:-1, :])
                clusters_file.write(clusters_string + '\n')



    def get_sample_loggers(self, run: int, resume: bool, chain: int = 0) -> list[ResultsLogger]:
        k = self.model.n_clusters
        base_dir = self.path_results
        if chain == 0:
            chain_str = ''
        else:
            chain_str = f'.chain{chain}'
            base_dir = base_dir / 'hot_chains'
            base_dir.mkdir(exist_ok=True)

        state_path = base_dir / f'state_K{k}_{run}{chain_str}.pickle'
        params_path = base_dir / f'stats_K{k}_{run}{chain_str}.txt'
        clusters_path = base_dir / f'clusters_K{k}_{run}{chain_str}.txt'
        likelihood_path = base_dir / f'likelihood_K{k}_{run}{chain_str}.h5'
        op_stats_path = base_dir / f'operator_stats_K{k}_{run}{chain_str}.txt'

        # Always include the StateDumper to make sure we can resume runs later on
        sample_loggers = [StateDumper(state_path, self.data, self.model, resume=resume)]
        if not self.config.results.log_hot_chains and chain > 0:
            return sample_loggers

        sample_loggers += [
            ParametersCSVLogger(params_path, self.data, self.model,
                                log_source=self.config.results.log_source,
                                float_format=f"%.{self.config.results.float_precision}g",
                                resume=resume),
            ClustersLogger(clusters_path, self.data, self.model, resume=resume),
            OperatorStatsLogger(op_stats_path, self.data, self.model, operators=[], resume=resume),
        ]

        if (not self.config.mcmc.sample_from_prior      # When sampling from prior, the likelihood is not interesting
                and self.config.results.log_likelihood  # Likelihood logger can be manually deactivated
                and chain == 0):                        # No likelihood logger for hot chains
            sample_loggers.append(LikelihoodLogger(likelihood_path, self.data, self.model, resume=resume))

        return sample_loggers

    def get_results_file_path(
        self,
        prefix: str,
        run: int,
        chain: int = 0,
        suffix: str = 'txt'
    ) -> Path:
        if chain == 0:
            base_dir = self.path_results
            chain_str = ""
        else:
            base_dir = self.path_results / "hot_chains"
            chain_str = f".chain{chain}"

        k = self.model.n_clusters
        return base_dir / f'{prefix}_K{k}_{run}{chain_str}.{suffix}'

    def print_screen_log(self, i_step: int, likelihood: float):
        i_step_str = f"{i_step:<12}"
        likelihood_str = f'log-likelihood of the cold chain:  {likelihood:<19.2f}'
        time_per_million = (time.time() - self.t_start) / (i_step + 1) * 1000000
        time_str = f'{timedelta(seconds=int(time_per_million))} / million steps'
        self.logger.info(i_step_str + likelihood_str + time_str)
