#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import logging
import math
from abc import ABC, abstractmethod
import random as _random
import time as _time

import numpy as _np
from copy import copy
import typing as typ

import numpy as np

from sbayes.model import Model
from sbayes.load_data import Data
from sbayes.sampling.loggers import ResultsLogger, OperatorStatsLogger
from sbayes.sampling.operators import Operator
from sbayes.config.config import OperatorsConfig

from sbayes.sampling.state import Sample


class MCMC(ABC):

    """Base-class for MCMC samplers for generative model. Instantiable sub-classes have to implement
    some methods, like propose_step() and likelihood().

    Attributes:
        statistics (dict): Container for a set of statistics about the sampling run.
    """

    CHECK_CACHING = __debug__

    def __init__(
            self,
            model: Model,
            data: Data,
            operators: OperatorsConfig,
            sample_loggers: typ.List[ResultsLogger],
            n_chains: int = 1,
            mc3: bool = False,
            swap_period: int = None,
            chain_swaps: int = None,
            sample_from_prior: bool = False,
            show_screen_log: bool = False,
            logger: logging.Logger = None,
            **kwargs
    ):
        # The model and data defining the posterior distribution
        self.model = model
        self.data = data

        # Sampling attributes
        self.n_chains = n_chains
        self.chain_idx = list(range(self.n_chains))
        self.sample_from_prior = sample_from_prior

        # Copy posterior instance for each chain
        self.posterior_per_chain: typ.List[Model] = [copy(model) for _ in range(self.n_chains)]

        # Operators
        self.callable_operators: dict[str, Operator] = self.get_operators(operators)

        # Loggers to write results to files
        self.sample_loggers = sample_loggers

        # State attributes
        self._ll = _np.full(self.n_chains, -_np.inf)
        self._prior = _np.full(self.n_chains, -_np.inf)

        self.show_screen_log = show_screen_log
        self.t_start = None

        if logger is None:
            import logging
            self.logger = logging.getLogger()
        else:
            self.logger = logger

        for logger in self.sample_loggers:
            if isinstance(logger, OperatorStatsLogger):
                logger.operators = list(self.callable_operators.values())

    def prior(self, sample, chain):
        """Compute the (log) prior of a sample.
        Args:
            sample (Sample): The current sample.
            chain (int): The current chain.
        Returns:
            float: The (log) prior of the sample
        """
        log_prior = self.posterior_per_chain[chain].prior(sample=sample)

        if self.CHECK_CACHING and (sample.i_step < 1000) and (sample.i_step % 10 == 0):
            log_prior_stable = self.posterior_per_chain[chain].prior(sample=sample, caching=False)
            assert log_prior == log_prior_stable, f'{log_prior} != {log_prior_stable}'

        sample.last_prior = log_prior
        return log_prior

    def likelihood(self, sample, chain):
        """Compute the (log) likelihood of the given sample.

        Args:
            sample (Sample): The current sample.
            chain (int): The current chain.
        Returns:
            float: (log) likelihood of x
        """
        if self.sample_from_prior:
            sample.last_lh = 0.
            return 0.

        # Compute the likelihood
        log_lh = self.posterior_per_chain[chain].likelihood(sample=sample, caching=True)

        if self.CHECK_CACHING and (sample.i_step < 1000) and (sample.i_step % 10 == 0):
            log_lh_stable = self.posterior_per_chain[chain].likelihood(sample=sample, caching=False)
            assert np.allclose(log_lh, log_lh_stable), f'{log_lh} != {log_lh_stable}'
            # assert log_lh == log_lh_stable, f'{log_lh} != {log_lh_stable}'

        sample.last_lh = log_lh
        return log_lh

    @abstractmethod
    def generate_initial_sample(self, c=0):
        """Generate an initial sample from which the run should be started.
        Preferably in high density areas.
        Returns:
            SampleType: Initial sample.
        """
        pass

    @abstractmethod
    def get_operators(self, operators: OperatorsConfig) -> dict[str, Operator]:
        """Get relevant operators and weights for proposing MCMC update steps

        Args:
            operators: dictionary mapping operator names to operator objects
        Returns:
            The operator objects with a proposal function and weights
        """

    def generate_samples(self, n_steps, n_samples, warm_up=False, warm_up_steps=None):
        """Run the MCMC sampling procedure with Metropolis Hastings rejection step and options for multiple chains. \
        Samples are returned, statistics saved in self.statistics.
        Args:
            n_steps (int): The number of steps taken by the sampler (without burn-in steps)
            n_samples (int): The number of samples
            warm_up (bool): Warm-up run or real sampling?
            warm_up_steps (int): Number of warm-up steps
        Returns:
            list: The generated samples
        """

        # Generate initial samples
        sample = []
        for c in self.chain_idx:
            sample.append(
                self.generate_initial_sample(c)
            )

            # Compute the (log)-likelihood and the prior for each sample
            self._ll[c] = self.likelihood(sample[c], c)
            self._prior[c] = self.prior(sample[c], c)

        # # Probability of operators is different if there are zero clusters
        # if self.n_clusters == 0:
        #
        #     operator_names = [f.__name__ for f in self.fn_operators]
        #     cluster_op = [operator_names.index('alter_cluster_effect'), operator_names.index('shrink_cluster'),
        #                   operator_names.index('grow_cluster'), operator_names.index('swap_cluster')]
        #
        #     for op in cluster_op:
        #         self.p_operators[op] = 0.
        #     self.p_operators = [p / sum(self.p_operators) for p in self.p_operators]

        self.t_start = _time.time()

        # Function is called in warmup-mode
        if warm_up:
            print("Tuning parameters in warm-up...")
            for i_warmup in range(warm_up_steps):
                warmup_progress = (i_warmup / warm_up_steps) * 100
                if warmup_progress % 10 == 0:
                    print("warm-up", int(warmup_progress), "%")
                for c in self.chain_idx:
                    sample[c] = self.step(sample[c], c)
                    sample[c].i_step = i_warmup

            self.logger.info(f"Warm-up finished after {(_time.time() - self.t_start):.1f} seconds")

            # For the last sample find the best chain (highest posterior)
            posterior_samples = [self._ll[c] + self._prior[c] for c in self.chain_idx]
            best_chain = posterior_samples.index(max(posterior_samples))

            self.logger.info(f"Starting state taken from warmup chain {best_chain} with "
                             f"posterior density {posterior_samples[best_chain]} (posterior "
                             f"densities of all chains: {posterior_samples}).")

            # Return the best sample
            return sample[best_chain]

        # Function is called to sample from posterior
        else:
            print("Sampling from posterior...")
            steps_per_sample = int(_np.ceil(n_steps / n_samples))

            for i_step in range(n_steps):
                # Generate samples for each chain
                for c in self.chain_idx:
                    sample[c] = self.step(sample[c], c)
                    sample[c].i_step = i_step

                # Log samples at fixed intervals
                if i_step % steps_per_sample == 0:
                    # Log sample from the first chain
                    for logger in self.sample_loggers:
                        logger.write_sample(sample[self.chain_idx[0]])

                # Print work status and likelihood at fixed intervals
                if (i_step+1) % 1000 == 0:
                    self.print_screen_log(i_step+1, sample)

        # Close files of all sample_loggers
        for logger in self.sample_loggers:
            logger.close()

        self.logger.info(f"MCMC run finished after {(_time.time() - self.t_start):.1f} seconds")

    def choose_operator(self) -> Operator:
        # Randomly choose one operator to propose a new sample
        step_weights = [w.weight for w in self.callable_operators.values()]
        possible_steps = list(self.callable_operators.keys())
        operator_name = _np.random.choice(possible_steps, 1, p=step_weights)[0]

        operator = self.callable_operators[operator_name]
        operator['name'] = operator_name
        return operator

    def step(self, sample, c):
        """This function performs a full MH step: first, a new candidate sample is proposed
        for either the clusters or the other parameters. Then the candidate is evaluated against the current sample
        and accepted with Metropolis-Hastings acceptance probability
        Args:
            sample(Sample): A Sample object consisting of clusters, weights and source array
            c(int): the current chain
        Returns:
            Sample: A Sample object consisting of clusters, weights and source array"""
        operator = self.choose_operator()
        step_function = operator['function']

        step_time_start = _time.time()

        candidate, log_q, log_q_back = step_function(sample, c=c)

        # Compute the log-likelihood of the candidate
        ll_candidate = self.likelihood(candidate, c)

        # Compute the prior of the candidate
        prior_candidate = self.prior(candidate, c)

        # Evaluate the metropolis-hastings ratio
        if log_q_back == -_np.inf:
            accept = False
        elif log_q == -_np.inf:
            accept = True
        else:
            mh_ratio = self.metropolis_hastings_ratio(
                ll_new=ll_candidate, ll_prev=self._ll[c],
                prior_new=prior_candidate, prior_prev=self._prior[c],
                log_q=log_q, log_q_back=log_q_back
            )

            # Accept/reject according to MH-ratio and update
            accept = math.log(_random.random()) < mh_ratio

        step_time = _time.time() - step_time_start

        if accept:
            operator.register_accept(step_time=step_time, sample_old=sample, sample_new=candidate)
            sample = candidate
            self._ll[c] = ll_candidate
            self._prior[c] = prior_candidate
        else:
            operator.register_reject(step_time=step_time)

        return sample

    @staticmethod
    def metropolis_hastings_ratio(ll_new, ll_prev, prior_new, prior_prev, log_q, log_q_back, temperature=1.):
        """ Computes the metropolis-hastings ratio.
        Args:
            ll_new(float): the likelihood of the candidate
            ll_prev(float): the likelihood of the current sample
            prior_new(float): the prior of the candidate
            prior_prev(float): the prior of the current sample
            log_q (float): the transition probability
            log_q_back (float): the back-probability
            temperature(float): the temperature of the MCMC
        Returns:
            (float): the metropolis-hastings ratio
        """
        ll_ratio = ll_new - ll_prev
        log_q_ratio = log_q - log_q_back

        prior_ratio = prior_new - prior_prev
        mh_ratio = (ll_ratio * temperature) - log_q_ratio + prior_ratio

        return mh_ratio

    def print_screen_log(self, i_step, sample):
        i_step_str = f"{i_step:<12}"

        likelihood = self.likelihood(sample[self.chain_idx[0]], self.chain_idx[0])
        likelihood_str = f'log-likelihood:  {likelihood:<19.2f}'

        time_per_million = (_time.time() - self.t_start) / (i_step + 1) * 1000000
        time_str = f'{time_per_million:.0f} seconds / million steps'

        print(i_step_str + likelihood_str + time_str)







