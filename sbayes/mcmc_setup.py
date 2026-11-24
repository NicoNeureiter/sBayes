""" Setup of the MCMC process """
from __future__ import annotations

import shutil
import time

from jax import random
from sbayes.config.config import MCMCConfig
from sbayes.model import Model
from sbayes.sampling.loggers import (
    DerivedValues, OnlineSampleLogger, h5_metadata, write_samples
)
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
        self.model = Model(
            data=self.data,
            config=self.config.model,
            offset_reparameterization=self.config.mcmc.offset_reparameterization,
        )

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

    def calibrate(self) -> None:
        """Run the model calibration, reusing a cached result if one exists.

        The calibration file that was used is copied into the results directory, so
        that each result records the normalization constant it was sampled with.
        """
        cache_dir = None
        if self.config.results.cache_calibration:
            cache_dir = self.config.results.path / ".cache"

        t_start = time.time()
        self.model.calibrate(cache_dir=cache_dir)

        geo_prior = self.model.prior.geo_prior
        if geo_prior.config.estimate_rate:
            source = "Loaded cached" if geo_prior.calibration_from_cache else "Estimated"
            self.logger.info(
                f"{source} geo-prior normalization constant "
                f"({time.time() - t_start:.1f} seconds)"
            )
        if geo_prior.calibration_file is not None:
            shutil.copy(geo_prior.calibration_file, self.path_results / "geo_norm_const.npz")

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

        self.calibrate()

        thinning = mcmc_config.steps // mcmc_config.samples
        num_steps = mcmc_config.steps

        sample_logger = OnlineSampleLogger(
            self.path_results, run, resume,
            metadata=h5_metadata(self.data, self.model.n_clusters,
                                 self.model.fixed_parameter_names()),
            derive=DerivedValues(self.model, self.data),
        )
        with sample_logger:
            initial_state = None
            if resume:
                initial_state = sample_logger.load_state()

                # Only sample the steps that the interrupted run did not get to
                num_done = sample_logger.num_written_samples() * thinning
                num_steps = max(mcmc_config.steps - num_done, 0)
                self.logger.info(
                    f"Resuming run {run}: {num_done} of {mcmc_config.steps} steps "
                    f"already sampled, {num_steps} remaining."
                )
                # Avoid repeating the random numbers from the start of the run
                rng_key = random.fold_in(rng_key, num_done)

            if num_steps == 0:
                samples = sample_logger.read_samples()
            else:
                samples = sample_nuts(
                    model=self.model,
                    num_warmup=mcmc_config.warmup.warmup_steps,
                    num_samples=num_steps,
                    # One chain per run; cli.py loops over runs
                    num_chains=1,
                    rng_key=rng_key,
                    write_interval=results_config.write_interval,
                    thinning=thinning,
                    init_state=initial_state,
                    init_strategy=mcmc_config.initialization_strategy,
                    sample_logger=sample_logger,
                    svi_guide=mcmc_config.svi_guide,
                    svi_steps=mcmc_config.svi_steps,
                    laplace_mass_matrix=mcmc_config.laplace_mass_matrix,
                    adapt_mass_matrix=mcmc_config.adapt_mass_matrix,
                    logger=self.logger,
                )

        if not results_config.samples_file_only:
            self.logger.info("Writing samples to disk")
            # Write the metadata to the samples file and the stats TSV from it
            write_samples(
                run=run,
                base_path=self.path_results,
                samples=samples,
                data=self.data,
                model=self.model,
            )

        runtime = time.time() - t_start
        self.logger.info(f"Runtime: {runtime:.2f} seconds")