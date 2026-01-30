""" Setup of the MCMC process """
from __future__ import annotations

import pickle
import time

from jax import random
import numpyro
from numpyro.diagnostics import summary

from sbayes.model import Model
from sbayes.sampling.loggers import write_samples, OnlineSampleLogger
from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.sampling.numpyro_sampling import sample_nuts, sample_svi
from sbayes.tools.realign_clusters_within_run import align_clusters


class MCMCSetup:

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
        resume: bool = False,
        run: int = 1,
    ):
        mcmc_config = self.config.mcmc
        results_config = self.config.results

        self.t_start = time.time()

        inference_mode = "MCMC"
        # inference_mode = "SVI"

        rng_key = random.PRNGKey(seed=124 * run)
        # rng_key = random.key(0)

        sample_logger = OnlineSampleLogger(self.path_results, self.data, self.model, run, resume)

        self.model.calibrate()

        if inference_mode == "MCMC":
            # If resuming, read the initial sample from the samples.h5 file
            if resume:
                sample_logger.open()
                initial_sample = sample_logger.load_state()
            else:
                initial_sample = None

            # sampler, samples = sample_nuts_with_annealing(
            sampler, samples = sample_nuts(
                model=self.model,
                num_warmup=mcmc_config.warmup.warmup_steps,
                num_samples=mcmc_config.steps,
                num_chains=mcmc_config.runs,
                rng_key=rng_key,
                write_interval=results_config.write_interval,
                thinning=mcmc_config.steps // mcmc_config.samples,
                init_sample=initial_sample,
                init_strategy=mcmc_config.initialization_strategy,
                sample_logger=sample_logger,
            )

        elif inference_mode == "SVI":
            sampler, samples = sample_svi(
                model=self.model,
                num_warmup=mcmc_config.warmup.warmup_steps,
                num_samples=mcmc_config.samples,
                num_chains=mcmc_config.runs,
                rng_key=rng_key,
                thinning=mcmc_config.steps // mcmc_config.samples,
                # guide=get_manual_guide(self.model),
            )
        else:
            raise ValueError(f"Unknown inference mode: {inference_mode}")

        self.logger.info("Writing samples to disk")

        if not results_config.samples_file_only:
            # align_clusters()
            # Write the raw numpyro samples and the mcmc summary to separate files
            if isinstance(sampler, numpyro.infer.mcmc.MCMC):
                with open(self.path_results / f'samples_{run}.pkl', 'wb') as f:
                    pickle.dump(samples, f)

                with open(self.path_results / f'mcmc_summary_{run}.pkl', 'wb') as f:
                    pickle.dump(summary(samples, group_by_chain=True), f)

                # Write results to sBayes results files (separate files for clusters and other parameters)
                assert mcmc_config.runs == 1
                # for i in range(mcmc_config.runs):
                # samples_i = {k: v[run] for k, v in samples.items()}
                samples_i = {k: v[0] for k, v in samples.items()}
                write_samples(
                    run=run,
                    base_path=self.path_results,
                    samples=samples_i,
                    data=self.data,
                    model=self.model,
                )
            else:
                with open(self.path_results / f'samples_{run}.pkl', 'wb') as f:
                    pickle.dump(samples, f)

                # Write results to sBayes results files (separate files for clusters and other parameters)
                    write_samples(
                        run=run,
                        base_path=self.path_results,
                        samples=samples,
                        data=self.data,
                        model=self.model,
                    )

        sample_logger.close()

        runtime = time.time() - self.t_start
        self.logger.info(f"Runtime: {runtime:.2f} seconds")