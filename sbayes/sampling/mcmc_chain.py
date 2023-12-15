#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import psutil
import logging
import math
import random as _random
import time as _time
import typing as typ

import numpy as np

from sbayes.model import Model
from sbayes.load_data import Data
from sbayes.sampling.loggers import ResultsLogger, OperatorStatsLogger
from sbayes.sampling.operators import Operator, get_operator_schedule
from sbayes.config.config import OperatorsConfig

from sbayes.sampling.state import Sample


class MCMCChain:

    """Base-class for MCMC samplers for generative model. Instantiable subclasses
     have to implement some methods, like propose_step() and likelihood()."""

    CHECK_CACHING = __debug__

    def __init__(
        self,
        model: Model,
        data: Data,
        operators: OperatorsConfig,
        sample_loggers: typ.List[ResultsLogger],
        sample_from_prior: bool = False,
        show_screen_log: bool = False,
        logger: logging.Logger = None,
        screen_log_interval: int = 1000,
        temperature: float = 1.0,
        prior_temperature: float = 1.0
    ):
        # The model and data defining the posterior distribution
        self.model = model
        self.data = data
        self.temperature = temperature
        self.prior_temperature = prior_temperature

        # Sampling attributes
        self.sample_from_prior = sample_from_prior

        # Operators
        self.callable_operators: dict[str, Operator] = self.get_operators(operators)

        # Loggers to write results to files
        self.sample_loggers = sample_loggers

        # State attributes
        self._ll = -np.inf
        self._prior = -np.inf

        self.show_screen_log = show_screen_log
        self.screen_log_interval = screen_log_interval
        self.t_start = None

        self.screen_logger = logger
        for logger in self.sample_loggers:
            if isinstance(logger, OperatorStatsLogger):
                logger.operators = list(self.callable_operators.values())

        self.i_step_start = 0
        self.previous_operator = None

    def prior(self, sample: Sample):
        """Compute the log-prior of a sample.
        Args:
            sample (Sample): The current sample.
        Returns:
            float: The (log) prior of the sample
        """
        log_prior = self.model.prior(sample=sample)

        if self.CHECK_CACHING and (sample.i_step < 1000) and (sample.i_step % 10 == 0):
            log_prior_stable = self.model.prior(sample=sample, caching=False)
            assert log_prior == log_prior_stable, f'{log_prior} != {log_prior_stable}'

        sample.last_prior = log_prior
        return log_prior / self.prior_temperature

    def likelihood(self, sample: Sample):
        """Compute the (log) likelihood of the given sample.
        Args:
            sample (Sample): The current sample.
        Returns:
            float: (log) likelihood of the sample.
        """
        if self.sample_from_prior:
            sample.last_lh = 0.
            return 0.

        # Compute the likelihood
        log_lh = self.model.likelihood(sample=sample, caching=True)

        if self.CHECK_CACHING and (sample.i_step < 1000) and (sample.i_step % 10 == 0):
            log_lh_stable = self.model.likelihood(sample=sample, caching=False)
            assert np.allclose(log_lh, log_lh_stable), f'{log_lh} != {log_lh_stable}'

        sample.last_lh = log_lh
        return log_lh / self.temperature

    def get_operators(self, operators: OperatorsConfig) -> dict[str, Operator]:
        """Get operators and weights for proposing MCMC update steps

        Args:
            operators: dictionary mapping operator names to operator objects
        Returns:
            The operator objects with a proposal function and weights
        """
        return get_operator_schedule(
            operators_config=operators,
            model=self.model,
            data=self.data,
            temperature=self.temperature,
            prior_temperature=self.prior_temperature,
            sample_from_prior=self.sample_from_prior
        )

    def run(
        self,
        n_steps: int,
        logging_interval: int,
        initial_sample: Sample,
        log_memory_usage: bool = True,
    ) -> Sample:
        """Run the MCMC sampling procedure with Metropolis Hastings rejection step.
        Samples are returned, statistics are written to loggers.

        Args:
            n_steps: The number of steps taken by the sampler (without burn-in steps)
            logging_interval: The interval
            initial_sample: The starting sample
            log_memory_usage: print memory usage in the logger?
        Returns:
            Sample: The final state in the MCMC chain
        """

        # Generate initial samples
        sample = initial_sample

        # Compute the (log)-likelihood and the prior for each sample
        self._ll = self.likelihood(sample)
        self._prior = self.prior(sample)

        # Remember starting time for runtime estimates
        self.t_start = _time.time()

        # Function is called to sample from posterior
        self.i_step_start = sample.i_step + 1
        for i_step in range(self.i_step_start, self.i_step_start + n_steps):
            sample = self.step(sample)
            sample.i_step = i_step

            # Log samples at fixed intervals
            if i_step % logging_interval == 0:
                for logger in self.sample_loggers:
                    logger.write_sample(sample)

            # Print work status and likelihood at fixed intervals
            if (i_step+1) % self.screen_log_interval == 0:
                self.print_screen_log(i_step + 1, sample)

        return sample

    def close_loggers(self):
        """Close files of all sample_loggers"""
        for logger in self.sample_loggers:
            logger.close()

    def choose_operator(self) -> Operator:
        # Randomly choose one operator to propose a new sample
        step_weights = [w.weight for w in self.callable_operators.values()]
        possible_steps = list(self.callable_operators.keys())
        operator_name = np.random.choice(possible_steps, 1, p=step_weights)[0]
        return self.callable_operators[operator_name]

    def step(self, sample: Sample) -> Sample:
        """This function performs a full MH step: first, a new candidate sample is proposed
        for either the clusters or the other parameters. Then the candidate is evaluated against the current sample
        and accepted with Metropolis-Hastings acceptance probability
        Args:
            sample(Sample): A Sample object consisting of clusters, weights and source array
        Returns:
            Sample: A Sample object consisting of clusters, weights and source array"""
        step_time_start = _time.time()

        # Chose and apply an MCMC operator
        operator = self.choose_operator()
        candidate, log_q, log_q_back = operator.function(sample)

        # Compute the log-likelihood and log-prior of the candidate
        ll_candidate = self.likelihood(candidate)
        prior_candidate = self.prior(candidate)

        # Evaluate the metropolis-hastings ratio
        if log_q_back == -np.inf:
            accept = False
        elif log_q == -np.inf:
            accept = True
        else:
            mh_ratio = self.metropolis_hastings_ratio(
                ll_new=ll_candidate, ll_prev=self._ll,
                prior_new=prior_candidate, prior_prev=self._prior,
                log_q=log_q, log_q_back=log_q_back
            )

            # Accept/reject according to MH-ratio and update
            accept = math.log(_random.random()) < mh_ratio

        step_time = _time.time() - step_time_start

        if accept:
            operator.register_accept(step_time=step_time, sample_old=sample, sample_new=candidate, prev_operator=self.previous_operator)
            sample = candidate
            self._ll = ll_candidate
            self._prior = prior_candidate
        else:
            operator.register_reject(step_time=step_time, prev_operator=self.previous_operator)

        self.previous_operator = operator

        if not np.isfinite(self._ll):
            raise ValueError(f'Non finite log-likelihood ({self._ll}) was accepted '
                             f'after MCMC operator {operator.operator_name}.')
        if not np.isfinite(self._prior):
            raise ValueError(f'Non finite log-prior ({self._prior}) was accepted '
                             f'after MCMC operator {operator.operator_name}.')

        return sample

    @staticmethod
    def metropolis_hastings_ratio(ll_new, ll_prev, prior_new, prior_prev, log_q, log_q_back):
        """ Computes the metropolis-hastings ratio.
        Args:
            ll_new(float): the likelihood of the candidate
            ll_prev(float): the likelihood of the current sample
            prior_new(float): the prior of the candidate
            prior_prev(float): the prior of the current sample
            log_q (float): the transition probability
            log_q_back (float): the back-probability
        Returns:
            (float): the metropolis-hastings ratio
        """
        log_posterior_ratio = (ll_new + prior_new) - (ll_prev + prior_prev)
        log_q_ratio = log_q - log_q_back
        return log_posterior_ratio - log_q_ratio

    def print_screen_log(self, i_step, sample):
        if self.screen_logger is None:
            return

        i_step_str = f"{i_step:<12}"

        likelihood = self.likelihood(sample)
        likelihood_str = f'log-likelihood:  {likelihood:<19.2f}'

        time_per_million = (_time.time() - self.t_start) / (i_step + 1 - self.i_step_start) * 1000000
        time_str = f'{time_per_million:.0f} seconds / million steps'

        self.screen_logger.info(i_step_str + likelihood_str + time_str)

    def reset_posterior_cache(self):
        """Reset the cached likelihood and prior for when the chain is repurposed to use a different sample"""
        self._ll = -np.inf
        self._prior = -np.inf
