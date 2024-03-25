#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the MCMC process """
from __future__ import annotations

import logging
import math
import os
import pickle
import random
import sys
import time
from datetime import timedelta
from multiprocessing import Process, Pipe
from time import sleep

import numpy as np
from numpy.typing import NDArray

from sbayes.config.config import MCMCConfig, WarmupConfig
from sbayes.results import Results
from sbayes.sampling.conditionals import impute_source
from sbayes.sampling.counts import recalculate_feature_counts
from sbayes.sampling.initializers import SbayesInitializer
from sbayes.sampling.mcmc import MCMC
from sbayes.sampling.mcmc_chain import MCMCChain
from sbayes.sampling.state import Sample
from sbayes.model import Model
from sbayes.sampling.loggers import ResultsLogger, ParametersCSVLogger, ClustersLogger, LikelihoodLogger, \
    OperatorStatsLogger, StateDumper
from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.util import RNG, process_memory


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
        self.sample_from_warm_up = None
        self.swap_attempts = 0
        self.swap_accepts = 0

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
        initial_sample: Sample | None = None,
        resume: bool = True,
        run: int = 1
    ):
        mcmc_config = self.config.mcmc

        # Initialize loggers
        sample_loggers = self.get_sample_loggers(run, resume)

        if initial_sample is not None:
            pass
        elif resume:
            # Load results
            results = self.read_previous_results(run)
            initial_sample = self.last_sample(results)

        else:
            warmup = MCMC(
                data=self.data,
                model=self.model,
                sample_loggers=[],
                n_chains=mcmc_config.warmup.warmup_chains,
                operators=mcmc_config.operators,
                p_grow_connected=mcmc_config.grow_to_adjacent,
                sample_from_prior=mcmc_config.sample_from_prior,
                logger=self.logger,
            )
            initializer = SbayesInitializer(
                model=self.model,
                data=self.data,
                initial_size=mcmc_config.initialization.objects_per_cluster,
                attempts=mcmc_config.initialization.attempts,
                initial_cluster_steps=mcmc_config.initialization._initial_cluster_steps,
            )
            initial_sample = warmup.generate_samples(
                n_steps=0, n_samples=0, warm_up=True,
                warm_up_steps=mcmc_config.warmup.warmup_steps,
                initializer=initializer,
            )
            initial_sample.i_step = 0

        self.sampler = MCMC(
            data=self.data,
            model=self.model,
            sample_loggers=sample_loggers,
            operators=mcmc_config.operators,
            p_grow_connected=mcmc_config.grow_to_adjacent,
            sample_from_prior=mcmc_config.sample_from_prior,
            logger=self.logger,
            screen_log_interval=mcmc_config.screen_log_interval,
        )
        self.sampler.generate_samples(
            mcmc_config.steps, mcmc_config.samples,
            initial_sample=initial_sample
        )

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

    def read_previous_results(self, run, chain=0) -> Results:
        if chain == 0:
            base_dir = self.path_results
            chain_str = ""
        else:
            base_dir = self.path_results / "hot_chains"
            chain_str = f".chain{chain}"

        k = self.model.n_clusters
        state_path = base_dir / f'state_K{k}_{run}{chain_str}.pickle'
        if state_path.exists():
            # New resume, based on pickled Sample objects (preferred if pickle file exists)
            with open(state_path, 'rb') as pickle_file:
                return pickle.load(state_path)
        else:
            # Old resume, based on results files
            params_path = base_dir / f'stats_K{k}_{run}{chain_str}.txt'
            clusters_path = base_dir / f'clusters_K{k}_{run}{chain_str}.txt'
            return Results.from_csv_files(clusters_path, params_path)

    def last_sample(self, results: Results) -> Sample:
        shapes = self.model.shapes
        clusters = results.clusters[:, -1, :]
        weights = np.array([results.weights[f][-1] for f in self.data.features.names])

        # Confounding effects are not used in `Sample` anymore.
        # TODO: Maybe use them to get a better initial state for `source`
        # conf_effects = {}
        # for conf, conf_eff in results.confounding_effects.items():
        #     conf_effects[conf] = np.zeros((shapes.n_groups[conf],
        #                                    shapes.n_features,
        #                                    shapes.n_states))
        #
        #     for g in self.model.confounders[conf].group_names:
        #         for i_f, f in enumerate(self.data.features.names):
        #             n_states_f = shapes.n_states_per_feature[i_f]
        #             conf_effects[conf][:, i_f, :n_states_f] = conf_eff[g][f][-1]

        source_shapes = (shapes.n_sites, shapes.n_features, shapes.n_components)
        dummy_source = np.empty(source_shapes, dtype=bool)

        dummy_feature_counts = {
            'clusters': np.zeros((shapes.n_clusters, shapes.n_features, shapes.n_states))
        } | {
            conf: np.zeros((n_groups, shapes.n_features, shapes.n_states))
            for conf, n_groups in shapes.n_groups.items()
        }

        sample = Sample.from_numpy_arrays(
            clusters=clusters,
            weights=weights,
            confounders=self.data.confounders,
            source=dummy_source,
            feature_counts=dummy_feature_counts,
            model_shapes=self.model.shapes,
        )
        sample.i_step = results.sample_id[-1] + 1

        # Next iteration: sample source from prior (allows calculating feature counts)
        impute_source(sample, self.model)
        recalculate_feature_counts(self.data.features.values, sample)

        return sample

    def sample_mc3(self, resume: bool = False, run: int = 1):
        """Generate samples using MC3.
        Args:
            resume: whether to resume a previous run.
            run: ID of this sampling run (when multiple runs of the same configuration are started)
        """
        mcmc_config = self.config.mcmc
        mc3_config = mcmc_config.mc3
        n_chains = mc3_config.chains
        logging_interval = int(np.ceil(mcmc_config.steps / mcmc_config.samples))
        n_swaps = int(mcmc_config.steps / mc3_config.swap_interval)

        # Remember time before initializing MC3 chains
        t_pre_init = time.time()

        # Calculate temperatures for each chain
        chain_idxs = np.arange(n_chains)
        if mc3_config.exponential_temperatures:
            temperatures = (1 + mc3_config.temperature_diff) ** chain_idxs
            prior_temperatures = (1 + mc3_config.prior_temperature_diff) ** chain_idxs
        else:
            temperatures = 1 + mc3_config.temperature_diff * chain_idxs
            prior_temperatures = 1 + mc3_config.prior_temperature_diff * chain_idxs

        processes = []
        for c in chain_idxs:
            # Create and start MCMCChainProcess for chain c
            proc = MCMCChainProcess(
                subchain_length=mc3_config.swap_interval,
                logging_interval=logging_interval,
                i_chain=c,
                temperature=temperatures[c],
                prior_temperature=prior_temperatures[c],
            )
            proc.start()
            processes.append(proc)

            if resume:
                # Load results
                results = self.read_previous_results(run, chain=c)
                initial_sample = self.last_sample(results)
            else:
                initial_sample = None

            # Start the initializer for chain c
            logger = self.get_sample_loggers(run, resume, chain=c)
            proc.send_initialize_chain(mcmc_config, self.model, self.data, logger, initial_sample=initial_sample)

        # Wait for each chain to finish initialization and warm-up and store the initial samples
        samples: list[Sample] = [None] * n_chains
        self.receive_samples(processes, samples)

        # Sanity checks
        assert len(processes) == n_chains
        assert len(samples) == n_chains

        # Initialize counters and timer
        self.swap_attempts = 0
        self.swap_accepts = 0
        self.swap_matrix = np.zeros((n_chains, n_chains), dtype=int)
        self.t_start = time.time()

        self.logger.info(f"Initialization and warm-up time: {timedelta(seconds=int(self.t_start - t_pre_init))}")
        self.logger.info("Sampling from posterior...")

        for i_swap in range(n_swaps):
            # Send the current sample to each process to start the next MCMC sub-chain
            for c in chain_idxs:
                processes[c].send_run_chain(samples[c])

            # Wait for the next final sample of each chain
            self.receive_samples(processes, samples)

            # Swap the chains of the current samples
            self.swap_chains(samples, temperatures, prior_temperatures,
                             attempts=mc3_config._swap_attempts,
                             only_swap_neighbours=mc3_config._only_swap_adjacent_chains)

            new_swap = self.last_swap_matrix_save < self.swap_attempts
            if mc3_config.log_swap_matrix and new_swap:
                self.swap_matrix_path = self.path_results / f"mc3_swaps_K{self.model.n_clusters}_{run}.txt"
                np.savetxt(self.swap_matrix_path, self.swap_matrix, fmt="%i")
                self.last_swap_matrix_save = self.swap_accepts

            if i_swap % 10 == 0:
                memory_parent = process_memory(unit="MB")
                memory_by_chain = [process_memory(proc.pid, unit="MB") for proc in processes]
                self.logger.info(f"Memory usage of parent process: {memory_parent} MB")
                self.logger.info(f"Memory usage of MC3 chain processes: {memory_by_chain} MB")

        self.logger.info(f"MCMC run finished after {timedelta(seconds=int(time.time() - self.t_start))}")
        self.terminate_chains(processes)

    def receive_samples(self, processes: list[MCMCChainProcess], samples: list[Sample]):
        """Receive the current samples from all MC3 chain processes, handling forwarded exceptions."""
        exceptions = []
        for c, proc in enumerate(processes):
            try:
                samples[c] = processes[c].receive()
            except SubprocessException as e:
                exceptions.append(e)

        if exceptions:
            self.terminate_chains(processes)
            raise exceptions[0]

    @staticmethod
    def terminate_chains(processes: list[MCMCChainProcess]):
        for proc in processes:
            proc.send_terminate()
            proc.join()

    def swap_chains(
        self,
        samples: list[Sample],
        temperatures: list[float],
        prior_temperatures: list[float],
        attempts: int,
        only_swap_neighbours: bool = False,
    ):
        """Chose random chains and try to swap with chain."""
        n_chains = len(samples)

        if only_swap_neighbours:
            # Only consecutive indices are possible swaps
            possible_swaps = [(i, i + 1) for i in range(n_chains - 1)]
        else:
            # All pairs of indices are possible swaps
            possible_swaps = [(i, j) for i in range(n_chains - 1) for j in range(i + 1, n_chains)]

        accepted_swaps = []

        # Choose `attempts` index pairs to propose swaps
        for swap_from, swap_to in RNG.choice(possible_swaps, size=attempts, replace=False):

            # Compute prior ratio for both chains
            log_prior_ratio = self.model.prior(samples[swap_from]) - self.model.prior(samples[swap_to])
            log_lh_ratio = self.model.likelihood(samples[swap_from]) - self.model.likelihood(samples[swap_to])

            prior_exp_diff = (1 / prior_temperatures[swap_from]) - (1 / prior_temperatures[swap_to])
            lh_exp_diff = (1 / temperatures[swap_from]) - (1 / temperatures[swap_to])

            mh_ratio = -(log_prior_ratio * prior_exp_diff + log_lh_ratio * lh_exp_diff)

            # mh_ratio = (posterior_from - posterior_to) * (pow_to - pow_from)
            # Equivalent to:
            #   joint_posterior_noswap = posterior_from * temp_from + posterior_to * temp_to
            #   joint_posterior_swap = posterior_from * temp_to + posterior_to * temp_from
            #   mh_ratio2 = joint_posterior_swap - joint_posterior_noswap

            accept = math.log(random.random()) < mh_ratio
            # Swap chains according to MH-ratio and update
            if accept:
                samples[swap_from], samples[swap_to] = samples[swap_to], samples[swap_from]
                self.swap_accepts += 1
                self.swap_matrix[swap_from, swap_to] += 1
                accepted_swaps.append((swap_from, swap_to))

            self.swap_attempts += 1

        self.logger.info(
            f"Accepted swaps ({attempts} attempts): {accepted_swaps}".ljust(60) +
            f"accept-rate={self.swap_accepts/self.swap_attempts}"
        )

        self.print_screen_log(samples[0].i_step, self.model.likelihood(samples[0]))

    def print_screen_log(self, i_step: int, likelihood: float):
        i_step_str = f"{i_step:<12}"
        likelihood_str = f'log-likelihood of the cold chain:  {likelihood:<19.2f}'
        time_per_million = (time.time() - self.t_start) / (i_step + 1) * 1000000
        time_str = f'{timedelta(seconds=int(time_per_million))} / million steps'
        self.logger.info(i_step_str + likelihood_str + time_str)


class MCMCChainProcess(Process):

    TERMINATE = 'terminate'

    def __init__(
        self,
        subchain_length: int,
        logging_interval: int,
        i_chain: int,
        temperature: float,
        prior_temperature: float,
    ):
        super().__init__()
        self.parent_connection, self.connection = Pipe()
        self.subchain_length = subchain_length
        self.logging_interval = logging_interval
        self.i_chain = i_chain
        self.temperature = temperature
        self.prior_temperature = prior_temperature

        # Will be initialized for each process in MCMCChainProcess.initialize
        self.mcmc_chain = None

    def run(self):
        while True:
            # Get method name and args from the parent process
            method_name, *args = self.connection.recv()

            if method_name == self.TERMINATE:
                self.shut_down()
                break

            # Get the method from the current object and call it
            method = getattr(self, method_name)
            try:
                result, send_back = method(*args)
            except Exception as exception:
                self.connection.send(exception)
                raise exception

            # Send the result back to the parent process
            if send_back:
                self.connection.send(result)

    def initialize_chain(
        self,
        mcmc_config: MCMCConfig,
        model: Model,
        data: Data,
        sample_loggers: list[ResultsLogger],
        initial_sample: Sample = None,
    ) -> (None, bool):
        """Initialize the MCMC chain in this process"""
        logging.info(f"Initializing MCMCChain {self.i_chain} in worker process {os.getpid()}")

        self.mcmc_chain = MCMCChain(
            model=model,
            data=data,
            operators=mcmc_config.operators,
            sample_from_prior=mcmc_config.sample_from_prior,
            screen_log_interval=mcmc_config.screen_log_interval,
            sample_loggers=sample_loggers,
            temperature=self.temperature,
            prior_temperature=self.prior_temperature,
        )

        if initial_sample is None:
            initializer = SbayesInitializer(
                model=model,
                data=data,
                initial_size=mcmc_config.initialization.objects_per_cluster,
                attempts=mcmc_config.initialization.attempts,
                initial_cluster_steps=mcmc_config.initialization._initial_cluster_steps,
                n_em_steps=mcmc_config.initialization.em_steps,
            )
            initial_sample = self.initialize_sample(initializer, self.mcmc_chain, mcmc_config.warmup)

        return initial_sample, True

    def initialize_sample(self, initializer: SbayesInitializer, mcmc_chain: MCMCChain, cfg: WarmupConfig) -> Sample:
        best_sample = None
        best_ll = -np.inf
        for i_attempt in range(cfg.warmup_chains):
            sample = initializer.generate_sample(c=self.i_chain)
            sample = mcmc_chain.run(
                n_steps=cfg.warmup_steps,
                logging_interval=sys.maxsize,
                initial_sample=sample
            )

            ll = mcmc_chain._ll
            if ll > best_ll:
                best_sample = sample
                best_ll = ll

            mcmc_chain.reset_posterior_cache()

        best_sample.i_step = 0
        return best_sample

    def run_chain(self, sample: Sample) -> (Sample, bool):
        sample = self.mcmc_chain.run(
            initial_sample=sample,
            n_steps=self.subchain_length,
            logging_interval=self.logging_interval,
        )
        return sample, True

    def shut_down(self):
        self.mcmc_chain.close_loggers()

    @staticmethod
    def print(s: str):
        """Print in the child process with process ID prefix. """
        print(f"{os.getpid()} >> {s}", flush=True)

    """ Interface for the parent process """

    def send_initialize_chain(
        self,
        mcmc_config: MCMCConfig,
        model: Model,
        data: Data,
        sample_loggers: list[ResultsLogger],
        initial_sample: Sample = None,
    ):
        self.parent_connection.send(("initialize_chain", mcmc_config, model, data, sample_loggers, initial_sample))

    def send_run_chain(self, sample: Sample):
        self.parent_connection.send(("run_chain", sample))

    def send_terminate(self):
        self.parent_connection.send((self.TERMINATE,))

    def receive(self):
        result = self.parent_connection.recv()

        if isinstance(result, BaseException):
            # Wait so that exception can be printed cleanly in the MCMC chain process
            sleep(0.5)
            raise SubprocessException(self.i_chain)

        return result


class SubprocessException(Exception):

    """Exception in a subprocess that is re-raised in the parent process"""

    def __init__(self, i_chain: int):
        super().__init__(f"An error occurred in the MC3ChainProcess for chain {i_chain}")
