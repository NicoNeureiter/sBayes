""" Setup of the MCMC process """
from __future__ import annotations

import numpy as np
import time

from jax import random
from sbayes.config.config import MCMCConfig
from sbayes.model import Model
from sbayes.sampling.loggers import write_samples, OnlineSampleLogger
from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.sampling.numpyro_sampling import sample_nuts


class MCMCSetup:

    """Sets up and runs the MCMC sampling for one run of an experiment."""

    def __init__(self, data: Data, experiment: Experiment):
        """
        Args:
            data: the data to run the analysis on
            experiment: the experiment providing the config, logger and results path
        """
        if experiment.path_results is None:
            raise ValueError(
                "MCMCSetup requires a results directory, but the experiment was "
                "created without one (create_experiment_folder=False)."
            )

        self.data = data
        self.config = experiment.config
        self.logger = experiment.logger


        # Create the model to sample from
        self.model = Model(data=self.data, config=self.config.model)

        # Set the results directory based on the number of clusters
        self.path_results = experiment.path_results / f"K{self.model.n_clusters}"
        self.path_results.mkdir(parents=True, exist_ok=True)


    def log_setup(self) -> None:
        """Log the model and MCMC settings of this run."""
        mcmc_cfg = self.config.mcmc
        self.logger.info(self.model.get_setup_message())
        self.logger.info(f'''
    MCMC SETUP
    ##########################################
    MCMC with {mcmc_cfg.steps} steps and {mcmc_cfg.samples} samples
    Warm-up: {mcmc_cfg.warmup.warmup_steps} steps
    ##########################################
    ''')

    def sample(self, run: int, resume: bool = False) -> None:
        """Run the inference and write the resulting samples to disk.

        Args:
            run: index of this run, used to offset the random seed and to label the
                output files
            resume: if True, continue a previous run from the last sample in its
                samples.h5 file
        """
        mcmc_config = self.config.mcmc
        results_config = self.config.results
        rng_key = random.PRNGKey(mcmc_config.seed + run)
        t_start = time.time()

        if mcmc_config.inference_mode is MCMCConfig.InferenceMode.SVI:
            raise NotImplementedError(
                "SVI inference is not implemented: `sample_svi` has not been updated "
                "for the current model. Use `inference_mode: mcmc`."
            )
        if mcmc_config.inference_mode is not MCMCConfig.InferenceMode.MCMC:
            raise ValueError(f"Unknown inference mode: {mcmc_config.inference_mode}")

        self.model.calibrate()

        with OnlineSampleLogger(self.path_results, run, resume) as sample_logger:
            if resume:
                # The file has to be open for the samples to be read back at the end
                sample_logger.open()
                initial_state = sample_logger.load_state()
            else:
                initial_state = None

            samples = sample_nuts(
                model=self.model,
                num_warmup=mcmc_config.warmup.warmup_steps,
                num_samples=mcmc_config.steps,
                # One chain per run; cli.py loops over runs
                num_chains=1,
                rng_key=rng_key,
                write_interval=results_config.write_interval,
                thinning=mcmc_config.steps // mcmc_config.samples,
                init_state=initial_state,
                init_strategy=mcmc_config.initialization_strategy,
                sample_logger=sample_logger,
                svi_guide=mcmc_config.svi_guide,
                svi_steps=mcmc_config.svi_steps,
            )

        if not results_config.samples_file_only:
            self.logger.info("Writing samples to disk")
            # Write results to sBayes results files (stats TSV + likelihood into samples h5)
            np.random.seed(mcmc_config.seed + run)
            write_samples(
                run=run,
                base_path=self.path_results,
                samples=samples,
                data=self.data,
                model=self.model,
            )

        runtime = time.time() - t_start
        self.logger.info(f"Runtime: {runtime:.2f} seconds")