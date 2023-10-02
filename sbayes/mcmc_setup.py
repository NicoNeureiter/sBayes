#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the MCMC process """
from __future__ import annotations
import math
import random
import time
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

import numpy as np

from sbayes.results import Results
from sbayes.sampling.conditionals import impute_source
from sbayes.sampling.counts import recalculate_feature_counts
from sbayes.sampling.initializers import SbayesInitializer
from sbayes.sampling.mcmc import MCMC
from sbayes.sampling.mcmc_chain import MCMCChainProcess
from sbayes.sampling.state import Sample
from sbayes.model import Model
from sbayes.sampling.loggers import ResultsLogger, ParametersCSVLogger, ClustersLogger, LikelihoodLogger, \
    OperatorStatsLogger
from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data


class MCMCSetup:
    def __init__(self, data: Data, experiment: Experiment):
        self.data = data
        self.config = experiment.config
        self.path_results = experiment.path_results

        # Create the model to sample from
        self.model = Model(data=self.data, config=self.config.model)

        # Samples
        self.sampler = None
        self.samples = None
        self.sample_from_warm_up = None
        self.swap_attempts = 0
        self.swap_accepts = 0

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
Ratio of confounding_effects steps (changing probabilities in confounders): {op_cfg.confounding_effects}''')
        if self.model.sample_source:
            self.logger.info(f'Ratio of source steps (changing source component assignment): {op_cfg.source}')
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
                initial_size=mcmc_config.init_objects_per_cluster,
                attempts=mcmc_config.initialization.attempts,
                initial_cluster_steps=mcmc_config.initialization.initial_cluster_steps,
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
        base_dir = self.path_results / f'K{k}'
        base_dir.mkdir(exist_ok=True)
        if chain == 0:
            params_path = base_dir / f'stats_K{k}_{run}.txt'
            clusters_path = base_dir / f'clusters_K{k}_{run}.txt'
            likelihood_path = base_dir / f'likelihood_K{k}_{run}.h5'
            op_stats_path = base_dir / f'operator_stats_K{k}_{run}.txt'
        else:
            params_path = base_dir / f'stats_K{k}_{run}.chain{chain}.txt'
            clusters_path = base_dir / f'clusters_K{k}_{run}.chain{chain}.txt'
            likelihood_path = base_dir / f'likelihood_K{k}_{run}.chain{chain}.h5'
            op_stats_path = base_dir / f'operator_stats_K{k}_{run}.chain{chain}.txt'

        sample_loggers = [
            ParametersCSVLogger(params_path, self.data, self.model,
                                log_source=self.config.results.log_source,
                                float_format=f"%.{self.config.results.float_precision}g",
                                resume=resume),
            ClustersLogger(clusters_path, self.data, self.model, resume=resume),
            OperatorStatsLogger(op_stats_path, self.data, self.model, operators=[], resume=resume)
        ]

        if not self.config.mcmc.sample_from_prior and self.config.results.log_likelihood:
            sample_loggers.append(LikelihoodLogger(likelihood_path, self.data, self.model, resume=resume))

        return sample_loggers

    def read_previous_results(self, run=1) -> Results:
        k = self.model.n_clusters
        params_path = self.path_results / f'K{k}' / f'stats_K{k}_{run}.txt'
        clusters_path = self.path_results / f'K{k}' / f'clusters_K{k}_{run}.txt'
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
        mcmc_config = self.config.mcmc
        n_chains = mcmc_config.mc3_chains
        logging_interval = int(np.ceil(mcmc_config.steps / mcmc_config.samples))
        n_swaps = int(mcmc_config.steps / mcmc_config.mc3_swap_interval)

        initializer = SbayesInitializer(
            model=self.model,
            data=self.data,
            initial_size=mcmc_config.init_objects_per_cluster,
            attempts=mcmc_config.initialization.attempts,
            initial_cluster_steps=mcmc_config.initialization.initial_cluster_steps,
        )
        samples = [initializer.generate_sample(c) for c in range(n_chains)]

        mcmc_chain_args = dict(
            model=self.model,
            data=self.data,
            operators=mcmc_config.operators,
            sample_from_prior=mcmc_config.sample_from_prior,
            screen_log_interval=mcmc_config.screen_log_interval,
        )
        temperatures = [1 + (c * mcmc_config.mc3_temperature_diff) for c in range(n_chains)]
        loggers = [self.get_sample_loggers(run, resume, chain=c) for c in range(n_chains)]

        connections: list[Connection] = []
        processes: list[MCMCChainProcess] = []
        for c in range(n_chains):
            parent_conn, child_conn = Pipe()
            proc = MCMCChainProcess(
                conn=child_conn,
                subchain_length=mcmc_config.mc3_swap_interval,
                logging_interval=logging_interval
            )
            proc.start()
            parent_conn.send(('initialize', mcmc_chain_args, loggers[c], temperatures[c]))
            processes.append(proc)
            connections.append(parent_conn)

        # Remember starting time for runtime estimates
        self.swap_attempts = 0
        self.swap_accepts = 0
        self.t_start = time.time()
        self.logger.info("Sampling from posterior...")

        for i_swap in range(n_swaps):

            # Send the current sample to each process to start the next MCMC sub-chain
            for c in range(n_chains):
                connections[c].send(('run_chain', samples[c]))

            # Wait for the next final sample of each chain
            for c in range(n_chains):
                samples[c] = connections[c].recv()

            # Swap the chains of the current samples
            self.swap_chains(samples, temperatures)

        self.logger.info(f"MCMC run finished after {(time.time() - self.t_start):.1f} seconds")

        for conn in connections:
            conn.send(MCMCChainProcess.TERMINATE)
        for proc in processes:
            proc.join()

    def swap_chains(self, samples: list[Sample], temperatures: list[float]):
        """Chose random chains and try to swap with first chain"""
        n_chains = len(samples)

        swap_from = np.random.randint(n_chains - 1)
        swap_to = swap_from + 1

        # Compute posterior ratio for both chains
        posterior_from = self.model(samples[swap_from])
        pow_from = 1 / temperatures[swap_from]

        posterior_to = self.model(samples[swap_to])
        pow_to = 1 / temperatures[swap_to]

        mh_ratio = (posterior_from - posterior_to) * (pow_to - pow_from)
        # Equivalent to:
        #   joint_posterior_noswap = posterior_from * temp_from + posterior_to * temp_to
        #   joint_posterior_swap = posterior_from * temp_to + posterior_to * temp_from
        #   mh_ratio2 = joint_posterior_swap - joint_posterior_noswap

        accept = math.log(random.random()) < mh_ratio
        # Swap chains according to MH-ratio and update
        if accept:
            samples[swap_from], samples[swap_to] = samples[swap_to], samples[swap_from]
            self.swap_accepts += 1
        self.swap_attempts += 1

        accept_str = 'ACCEPT' if accept else 'REJECT'
        self.logger(f"swap chains {(swap_from, swap_to)}?   " +
                    f"{accept_str}  (p_accept={np.exp(mh_ratio):.2f})    " +
                    f"accept-rate={self.swap_accepts/self.swap_attempts}")
        self.print_screen_log(samples[0].i_step, self.model.likelihood(samples[0]))

    def print_screen_log(self, i_step: int, likelihood: float):
        i_step_str = f"{i_step:<12}"
        likelihood_str = f'log-likelihood of the cold chain:  {likelihood:<19.2f}'
        time_per_million = (time.time() - self.t_start) / (i_step + 1) * 1000000
        time_str = f'{time_per_million:.0f} seconds / million steps'
        self.logger.info(i_step_str + likelihood_str + time_str)
