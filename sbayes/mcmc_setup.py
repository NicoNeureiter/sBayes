#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the MCMC process """
from __future__ import annotations

from jax import random
import numpy as np
from numpy.typing import NDArray
import numpyro

from sbayes.preprocessing import sample_categorical

from sbayes.model import Model
from sbayes.sampling.loggers import write_samples
from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.sampling.numpyro_sampling import sample_nuts, sample_svi, get_manual_guide
from sbayes.util import format_cluster_columns, get_best_permutation

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
        self.logger.info(self.model.get_setup_message())
        self.logger.info(f'''
MCMC SETUP
##########################################
MCMC with {mcmc_cfg.steps} steps and {mcmc_cfg.samples} samples
Warm-up: {mcmc_cfg.warmup.warmup_steps} steps''')
        self.logger.info('\n')

    def sample(
        self,
        initial_sample = None,
        resume: bool = True,
        run: int = 1
    ):
        mcmc_config = self.config.mcmc

        inference_mode = "MCMC"
        # inference_mode = "SVI"

        if inference_mode == "MCMC":
            rng_key = random.PRNGKey(seed=124 * run)
            sampler, samples = sample_nuts(
                model=self.model.get_model,
                num_warmup=mcmc_config.warmup.warmup_steps,
                num_samples=mcmc_config.steps,
                num_chains=1,  # NN: Could be configurable, but I don't see a clear advantage over parallel runs
                rng_key=rng_key,
                thinning=mcmc_config.steps // mcmc_config.samples,
                # show_inference_summary=show_inference_summary,
                # init_params=self.model.generate_initial_params(rng_key),
            )

        elif inference_mode == "SVI":
            sampler, samples = sample_svi(
                model=self.model.get_model,
                num_warmup=mcmc_config.warmup.warmup_steps,
                num_samples=mcmc_config.samples,
                num_chains=1,  # NN: Could be configurable, but I don't see a clear advantage over parallel runs
                rng_key=random.PRNGKey(seed=123 * run),
                thinning=mcmc_config.steps // mcmc_config.samples,
                # show_inference_summary=show_inference_summary,
                # guide=get_manual_guide(self.model),
            )
        else:
            raise ValueError(f"Unknown inference mode: {inference_mode}")

        K = self.model.n_clusters
        write_samples(
            samples=samples,
            clusters_path=self.path_results / f'clusters_K{K}_{run}.txt',
            params_path=self.path_results / f'stats_K{K}_{run}.txt',
            data=self.data,
            model=self.model,
        )
