#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the MCMC process """
from __future__ import annotations

import json
import pickle

from jax import random
import numpy as np
from numpy.typing import NDArray
import numpyro
from numpyro.diagnostics import summary

from sbayes.preprocessing import sample_categorical

from sbayes.model import Model
from sbayes.sampling.loggers import write_samples, OnlineSampleLogger
from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.sampling.numpyro_sampling import sample_nuts, sample_svi, get_manual_guide, sample_nuts_with_annealing
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
        resume: bool = True,
        # run: int = 1,
    ):
        mcmc_config = self.config.mcmc
        results_config = self.config.results

        inference_mode = "MCMC"
        # inference_mode = "SVI"

        # rng_key = random.PRNGKey(seed=124 * run)
        rng_key = random.key(0)

        sample_logger = OnlineSampleLogger(self.path_results / 'samples.h5', self.data, self.model, resume)

        if inference_mode == "MCMC":
            # If resuming, read the initial sample from the samples.h5 file
            if resume:
                initial_sample = sample_logger.read_samples()
            else:
                initial_sample = None

            # sampler, samples = sample_nuts_with_annealing(
            #     model=self.model,
            sampler, samples = sample_nuts(
                model=self.model,
                num_warmup=mcmc_config.warmup.warmup_steps,
                num_samples=mcmc_config.steps,
                # num_chains=1,  # NN: Could be configurable, but I don't see a clear advantage over parallel runs
                num_chains=mcmc_config.runs,
                rng_key=rng_key,
                write_interval=results_config.write_interval,
                thinning=mcmc_config.steps // mcmc_config.samples,
                init_sample=initial_sample,
                sample_logger=sample_logger,
            )

        elif inference_mode == "SVI":
            sampler, samples = sample_svi(
                model=self.model,
                num_warmup=mcmc_config.warmup.warmup_steps,
                num_samples=mcmc_config.samples,
                # num_chains=1,  # NN: Could be configurable, but I don't see a clear advantage over parallel runs
                num_chains=mcmc_config.runs,
                rng_key=rng_key,
                thinning=mcmc_config.steps // mcmc_config.samples,
                # show_inference_summary=show_inference_summary,
                # guide=get_manual_guide(self.model),
            )
        else:
            raise ValueError(f"Unknown inference mode: {inference_mode}")

        self.logger.info("Writing samples to disk")

        if not results_config.samples_file_only:
            # Write the raw numpyro samples and the mcmc summary to separate files
            if isinstance(sampler, numpyro.infer.mcmc.MCMC):
                with open(self.path_results / f'samples.pkl', 'wb') as f:
                    pickle.dump(samples, f)

                with open(self.path_results / f'mcmc_summary.pkl', 'wb') as f:
                    pickle.dump(summary(samples, group_by_chain=True), f)

                # Write the numpyro mcmc checkpoint to a file
                with open(self.path_results / f'state.pkl', 'wb') as f:
                    pickle.dump(sampler.last_state, f)

                # Write results to sBayes results files (separate files for clusters and other parameters)
                for i in range(mcmc_config.runs):
                    samples_i = {k: v[i] for k, v in samples.items()}
                    write_samples(
                        run=i,
                        base_path=self.path_results,
                        samples=samples_i,
                        data=self.data,
                        model=self.model,
                    )
            else:
                with open(self.path_results / f'samples.pkl', 'wb') as f:
                    pickle.dump(samples, f)

                # Write results to sBayes results files (separate files for clusters and other parameters)
                    write_samples(
                        run=0,
                        base_path=self.path_results,
                        samples=samples,
                        data=self.data,
                        model=self.model,
                    )

        sample_logger.close()

