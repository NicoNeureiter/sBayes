#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import math as _math
import abc as _abc
import random as _random
import time as _time
import numpy as _np


class MCMC_generative(metaclass=_abc.ABCMeta):

    """Base-class for MCMC samplers for generative model. Instantiable sub-classes have to implement
    some methods, like propose_step() and likelihood().
    The base-class provides options for Markov coupled MCMC (MC3)[2].

    Attributes:
        mc3_chains (int): Number of coupled chains for MC3 (ignored when MC 3 is False)
        mc3_delta_t (float):
        mc3_swap_period (int):

        temperature (float): Simulated annealing temperature. Changes according to
            schedule when simulated_annealing is True, is constant (1) otherwise.
        statistics (dict): Container for a set of statistics about the sampling run.

    [1]  Altekar, Gautam, et al. "Parallel metropolis coupled Markov chain Monte Carlo for Bayesian phylogenetic inference." Bioinformatics 20.3 (2004): 407-415.
    """

    def __init__(self, operators, n_chains=4, swap_period=1000, chain_swaps=1, temperature=1.):

        # Sampling attributes
        self.n_chains = n_chains
        self.swap_period = swap_period
        self.chain_swaps = chain_swaps
        self.chain_idx = list(range(self.n_chains))
        self.temperature = temperature

        # Operators
        self.fn_operators, self.p_operators = self.get_operators(operators)

        # Initialize statistics
        self.statistics = {'sample_likelihood': [],
                           'sample_prior': [],
                           'sample_zones': [],
                           'sample_weights': [],
                           'last_sample': [],
                           'acceptance_ratio': _math.nan,
                           'accepted_steps': 0,
                           'n_swaps': 0,
                           'accepted_swaps': 0,
                           'swap_ratio': []}

        # State attributes
        self._ll = _np.full(self.n_chains, -_np.inf)
        self._prior = _np.full(self.n_chains, -_np.inf)

    @_abc.abstractmethod
    def prior(self, x):
        """Compute the prior of the sample
        Args:
            x (Sample): Sample object
        Returns:
            float: the prior of x
        """
        pass

    @_abc.abstractmethod
    def likelihood(self, x):
        """Compute the (log) likelihood of the given sample.

        Args:
            x (Sample): The current sample.

        Returns:
            float: (log)likelihood of x
        """
        pass

    @_abc.abstractmethod
    def generate_initial_sample(self, c):
        """Generate an initial sample from which the run should be started.
        Preferably in high density areas.
        Args:
            c(int): number of chain
        Returns:
            SampleType: Initial zone.
        """
        pass

    @_abc.abstractmethod
    def get_operators(self, operators):
        """Get relevant operators and weights for proposing MCMC update steps

        Args:
            operators (dict): dictionary with names of all operators (keys) and their weights (values)
        Returns:
            list, list: the operators (callable), their weights (float)
        """

    def generate_samples(self, n_steps, n_samples, burn_in_steps):
        """Run the MCMC sampling procedure for the Generative model with Metropolis Hastings rejection
        step and options for multiple chains. Samples are returned, statistics saved in self.statistics.

        Args:
            n_steps (int): The number of steps the sampler takes (without burn-in steps)
            n_samples (int): The number of samples
            burn_in_steps (int): The number of burn in steps before the first sample is taken
        Returns:
            list: The generated samples.
        """

        steps_per_sample = int(_np.ceil(n_steps / n_samples))
        t_start = _time.time()

        # Generate samples using MCMC with several chains
        sample = [None] * self.n_chains

        # Generate initial samples
        for c in self.chain_idx:

            sample[c] = self.generate_initial_sample(c)

            # Compute the (log)-likelihood for the sample and the prior
            self._ll[c] = self.likelihood(sample[c])
            self._prior[c] = self.prior(sample[c])

        # Generate burn-in samples for each chain
        for c in self.chain_idx:
            for i_step in range(burn_in_steps):
                sample[c] = self.step(sample[c], c)

        # Generate post burn-in samples
        for i_step in range(n_steps):

                # Generate samples for each chain
                for c in self.chain_idx:
                    sample[c] = self.step(sample[c], c)

                    # Log samples, but only from the first chain
                    if self.chain_idx[c] == 0:

                        # Log samples at fixed intervals
                        if i_step % steps_per_sample == 0:
                            self.log_sample_statistics(sample[c], c)

                # Exchange chains at fixed intervals
                if i_step % self.swap_period == 0:
                    self.swap_chains(sample, self.temperature)

                # Print work status at fixed intervals
                if i_step % 1000 == 0:
                    print(i_step, ' steps taken')

                # Log the complete last sample
                if i_step % (n_steps-1) == 0 and i_step != 0:
                    self.log_last_sample(sample)

        t_end = _time.time()
        self.statistics['sampling_time'] = t_end - t_start
        self.statistics['time_per_sample'] = (t_end - t_start) / n_samples
        self.statistics['acceptance_ratio'] = (self.statistics['accepted_steps'] / n_steps)
        self.statistics['swap_ratio'] = (self.statistics['accepted_swaps'] / self.statistics['n_swaps'])

        return

    def swap_chains(self, sample, temperature=1.):

        for _ in range(self.chain_swaps):

            self.statistics['n_swaps'] += 1

            # Chose random chains and try to swap with first chain
            swap_to, = _np.random.choice(range(1, self.n_chains), 1)
            swap_from = 0

            # Compute lh and prior ratio for both chains
            ll_from = self.likelihood(sample[swap_from])
            prior_from = self.prior(sample[swap_from])

            ll_to = self.likelihood(sample[swap_to])
            prior_to = self.prior(sample[swap_to])
            q_to = q_from = 1.

            # Evaluate the metropolis-hastings ratio
            mh_ratio = self.metropolis_hastings_ratio(ll_new=ll_to, ll_prev=ll_from,
                                                      prior_new=prior_to, prior_prev=prior_from,
                                                      q=q_to, q_back=q_from, temperature=self.temperature)

            # Swap chains according to MH-ratio and update
            if _math.log(_random.random()) < mh_ratio:
                self.chain_idx[swap_from] = swap_to
                self.chain_idx[swap_to] = swap_from
                self.statistics['accepted_swaps'] += 1

    def step(self, sample, c):
        """This function performs a full MH step: first, a new candidate sample is proposed
        for either the zones or the weights, then the candidate is evaluated against the current sample
        and accepted with metropolis hastings acceptance probability

        Args:
            sample(Sample): A Sample object consisting of zones and weights
            c(int): the current chain of the MC3
        Returns:
            Sample: A Sample object consisting of zones and weights"""

        # Randomly choose one of the operators to propose a new sample (grow/shrink/swap zones, alter weights)
        propose_step = _np.random.choice(self.fn_operators, 1, p=self.p_operators)[0]
        candidate, q, q_back = propose_step(sample)

        # Compute the log-likelihood of the candidate
        ll_candidate = self.likelihood(candidate, step_type=propose_step.__name__)

        # Compute the prior of the candidate
        prior_candidate = self.prior(candidate)

        # Evaluate the metropolis-hastings ratio
        mh_ratio = self.metropolis_hastings_ratio(ll_new=ll_candidate, ll_prev=self._ll[c],
                                                  prior_new=prior_candidate, prior_prev=self._prior[c],
                                                  q=q, q_back=q_back, temperature=self.temperature)

        # Accept/reject according to MH-ratio and update
        if _math.log(_random.random()) < mh_ratio:
            sample = candidate
            self._ll[c] = ll_candidate
            self._prior[c] = prior_candidate
            self.statistics['accepted_steps'] += 1

        return sample

    def metropolis_hastings_ratio(self, ll_new, ll_prev, prior_new, prior_prev, q, q_back, temperature=1.):
        """ Computes the metropolis-hastings ratio.
        Args:
            ll_new(float): the likelihood of the candidate
            ll_prev(float): the likelihood of the current sample
            prior_new(float): the prior of the candidate
            prior_prev(float): tself.fn_operatorshe prior of the current sample
            q (float): the transition probability
            q (float): the back-probability
            temperature(float): the temperature of the MCMC
        Returns:
            (float): the metropolis-hastings ratio
        """

        ll_ratio = ll_new - ll_prev
        try:
            log_q_ratio = _math.log(q / q_back)
        except ZeroDivisionError:
            log_q_ratio = _math.inf

        prior_ratio = prior_new - prior_prev
        mh_ratio = (ll_ratio * temperature) - log_q_ratio + prior_ratio
        return mh_ratio

    def log_sample_statistics(self, sample, c):
        """ This function logs the statistics of an MCMC sample.
        Args:
            sample (Sample): A Sample object consisting of zones and weights
            c (int): The current chain
        """
        self.statistics['sample_zones'].append(sample.zones)
        self.statistics['sample_weights'].append(sample.weights)
        self.statistics['sample_likelihood'].append(self._ll[c])
        self.statistics['sample_prior'].append(self._prior[c])

    def log_last_sample(self, last_sample):
        """ This function logs the complete last sample of an MCMC run.
        Args:
            last_sample (Sample): A Sample object consisting of zones and weights
        """
        self.statistics['last_sample'].append(last_sample)

