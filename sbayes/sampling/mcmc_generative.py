#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import math as _math
import abc as _abc
import random as _random
import time as _time
import numpy as _np
from copy import copy

from collections import defaultdict


class MCMCGenerative(metaclass=_abc.ABCMeta):

    """Base-class for MCMC samplers for generative model. Instantiable sub-classes have to implement
    some methods, like propose_step() and likelihood().
    The base-class provides options for Markov coupled MCMC (MC3)[2].

    Attributes:
        statistics (dict): Container for a set of statistics about the sampling run.
    """

    IS_WARMUP = False

    def __init__(self, model, data, operators, n_chains,
                 mc3=False, swap_period=None, chain_swaps=None,
                 sample_from_prior=False, show_screen_log=False, **kwargs):

        # The model and data defining the posterior distribution
        self.model = model
        self.data = data

        # Sampling attributes
        self.n_chains = n_chains
        self.chain_idx = list(range(self.n_chains))
        self.sample_from_prior = sample_from_prior

        # Operators
        self.fn_operators, self.p_operators = self.get_operators(operators)

        # MC3
        self.mc3 = mc3
        self.swap_period = swap_period
        self.chain_swaps = chain_swaps

        # Initialize statistics
        self.statistics = {'sample_id': [],
                           'sample_likelihood': [],
                           'sample_prior': [],
                           'sample_zones': [],
                           'sample_weights': [],
                           'sample_p_global': [],
                           'sample_p_zones': [],
                           'sample_p_families': [],
                           'last_sample': [],
                           'acceptance_ratio': _math.nan,
                           'accepted_steps': 0,
                           'n_swaps': 0,
                           'accepted_swaps': 0,
                           'swap_ratio': [],
                           'accept_operator': defaultdict(int),
                           'reject_operator': defaultdict(int)}

        # State attributes
        self._ll = _np.full(self.n_chains, -_np.inf)
        self._prior = _np.full(self.n_chains, -_np.inf)

        self.show_screen_log = show_screen_log
        self.t_start = _time.time()

        self.posterior_per_chain = [copy(model) for _ in range(self.n_chains)]

    def prior(self, sample, chain):
        """Compute the (log) prior of a sample.
        Args:
            sample (Sample): The current sample.
            chain (int): The current chain.
        Returns:
            float: The (log) prior of the sample"""
        # Compute the prior
        log_prior = self.posterior_per_chain[chain].prior(sample=sample)

        check_caching = False
        if check_caching:
            sample.everything_changed()
            log_prior_stable = self.posterior_per_chain[chain].prior(sample=sample)
            assert log_prior == log_prior_stable, f'{log_prior} != {log_prior_stable}'

        return log_prior

    def likelihood(self, sample, chain):
        """Compute the (log) likelihood of the given sample.

        Args:
            sample (Sample): The current sample.
            chain (int): The current chain.
        Returns:
            float: (log)likelihood of x
        """
        if self.sample_from_prior:
            return 0.

        # Compute the likelihood
        log_lh = self.posterior_per_chain[chain].likelihood(sample=sample)

        check_caching = False
        if check_caching:
            sample.everything_changed()
            log_lh_stable = self.posterior_per_chain[chain].likelihood(sample=sample, caching=False)
            assert log_lh == log_lh_stable, f'{log_lh} != {log_lh_stable}'

        return log_lh


    @_abc.abstractmethod
    def generate_initial_sample(self, c=0):
        """Generate an initial sample from which the run should be started.
        Preferably in high density areas.
        Returns:
            SampleType: Initial sample.
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

    def generate_samples(self, n_steps, n_samples, warm_up=False, warm_up_steps=None):
        """Run the MCMC sampling procedure for the Generative model with Metropolis Hastings rejection
        step and options for multiple chains. Samples are returned, statistics saved in self.statistics.

        Args:
            n_steps (int): The number of steps the sampler takes (without burn-in steps)
            n_samples (int): The number of samples
            warm_up (bool): Warm-up run or real sampling?
            warm_up_steps (int): Number of warm-up steps
        Returns:
            list: The generated samples.
        """

        # Generate samples using MCMC with several chains
        sample = [None] * self.n_chains

        # Generate initial samples
        for c in self.chain_idx:

            sample[c] = self.generate_initial_sample(c)
            # Compute the (log)-likelihood and the prior for each sample
            self._ll[c] = self.likelihood(sample[c], c)
            self._prior[c] = self.prior(sample[c], c)

        # # Probability of operators is different if there are zero zones
        # if self.n_zones == 0:
        #
        #     operator_names = [f.__name__ for f in self.fn_operators]
        #     zone_op = [operator_names.index('alter_p_zones'), operator_names.index('shrink_zone'),
        #                operator_names.index('grow_zone'), operator_names.index('swap_zone')]
        #
        #     for op in zone_op:
        #         self.p_operators[op] = 0.
        #     self.p_operators = [p / sum(self.p_operators) for p in self.p_operators]

        # Function is called in warmup-mode
        if warm_up:
            print("Tuning parameters in warm-up...")
            for i_warmup in range(warm_up_steps):
                warmup_progress = (i_warmup / warm_up_steps) * 100
                if warmup_progress % 10 == 0:
                    print("warm-up", int(warmup_progress), "%")
                for c in self.chain_idx:
                    sample[c] = self.step(sample[c], c)

            # For the last sample find the best chain (highest posterior)
            posterior_samples = [self._ll[c] + self._prior[c] for c in self.chain_idx]

            best_chain = posterior_samples.index(max(posterior_samples))

            # Return the best sample
            return sample[best_chain]

        # Function is called to sample from posterior
        else:
            print("Sampling from posterior...")
            steps_per_sample = int(_np.ceil(n_steps / n_samples))
            t_start = _time.time()

            for i_step in range(n_steps):
                # Generate samples for each chain
                for c in self.chain_idx:
                    sample[c] = self.step(sample[c], c)

                # Log samples at fixed intervals
                if i_step % steps_per_sample == 0:

                    # Log samples, but only from the first chain
                    self.log_sample_statistics(sample[self.chain_idx[0]], c=self.chain_idx[0],
                                               sample_id=int(i_step/steps_per_sample))

                # For mc3: Exchange chains at fixed intervals
                if self.mc3:
                    if (i_step+1) % self.swap_period == 0:
                        self.swap_chains(sample)

                # Print work status and likelihood at fixed intervals
                if (i_step+1) % 1000 == 0:
                    self.print_screen_log(i_step+1, sample)

                # Log the last sample of the first chain
                if i_step % (n_steps-1) == 0 and i_step != 0:
                    self.log_last_sample(sample[self.chain_idx[0]])

            t_end = _time.time()
            self.statistics['sampling_time'] = t_end - t_start
            self.statistics['time_per_sample'] = (t_end - t_start) / n_samples
            self.statistics['acceptance_ratio'] = (self.statistics['accepted_steps'] / n_steps)
            if self.statistics['n_swaps'] > 0:
                self.statistics['swap_ratio'] = (self.statistics['accepted_swaps'] / self.statistics['n_swaps'])
            else:
                self.statistics['swap_ratio'] = 0

    def swap_chains(self, sample):

        for _ in range(self.chain_swaps):

            self.statistics['n_swaps'] += 1

            # Chose random chains and try to swap with first chain
            swap_from_idx = 0
            swap_from = self.chain_idx[swap_from_idx]

            swap_to_idx, = _np.random.choice(range(1, self.n_chains), 1)
            swap_to = self.chain_idx[swap_to_idx]

            # Compute lh and prior ratio for both chains
            ll_from = self.likelihood(sample[swap_from], swap_from)
            prior_from = self.prior(sample[swap_from], swap_from)

            ll_to = self.likelihood(sample[swap_to], swap_to)
            prior_to = self.prior(sample[swap_to], swap_to)
            q_to = q_from = 1.

            # Evaluate the metropolis-hastings ratio
            mh_ratio = self.metropolis_hastings_ratio(ll_new=ll_to, ll_prev=ll_from,
                                                      prior_new=prior_to, prior_prev=prior_from,
                                                      q=q_to, q_back=q_from)

            # Swap chains according to MH-ratio and update
            if _math.log(_random.random()) < mh_ratio:
                self.chain_idx[swap_from_idx] = swap_to
                self.chain_idx[swap_to_idx] = swap_from
                self.statistics['accepted_swaps'] += 1

            #     print("swapped")
            # else:
            #     print("not swapped")

        # Set all 'what_changed' flags to true (to avoid caching errors)
        for s in sample:
            s.everything_changed()

    def step(self, sample, c):
        """This function performs a full MH step: first, a new candidate sample is proposed
        for either the zones or the weights, then the candidate is evaluated against the current sample
        and accepted with metropolis hastings acceptance probability

        Args:
            sample(Sample): A Sample object consisting of zones and weights
            c(int): the current chain
        Returns:
            Sample: A Sample object consisting of zones and weights"""

        # Randomly choose one operator to propose new sample (grow/shrink/swap zones, alter weights/p_zones/p_families)
        propose_step = _np.random.choice(self.fn_operators, 1, p=self.p_operators)[0]
        if self.IS_WARMUP:
            candidate, q, q_back = propose_step(sample, c)
        else:
            candidate, q, q_back = propose_step(sample)

        # Compute the log-likelihood of the candidate
        ll_candidate = self.likelihood(candidate, c)

        # Compute the prior of the candidate
        prior_candidate = self.prior(candidate, c)

        # Evaluate the metropolis-hastings ratio
        if q_back == 0:
            accept = False
        elif q == 0:
            accept = True
        else:
            mh_ratio = self.metropolis_hastings_ratio(ll_new=ll_candidate, ll_prev=self._ll[c],
                                                      prior_new=prior_candidate, prior_prev=self._prior[c],
                                                      q=q, q_back=q_back)

            # Accept/reject according to MH-ratio and update
            accept = _math.log(_random.random()) < mh_ratio

        if accept:
            sample = candidate
            self._ll[c] = ll_candidate
            self._prior[c] = prior_candidate
            self.statistics['accepted_steps'] += 1
            self.statistics['accept_operator'][propose_step.__name__] += 1
        else:
            self.statistics['reject_operator'][propose_step.__name__] += 1

        return sample

    @staticmethod
    def metropolis_hastings_ratio(ll_new, ll_prev, prior_new, prior_prev, q, q_back, temperature=1.):
        """ Computes the metropolis-hastings ratio.
        Args:
            ll_new(float): the likelihood of the candidate
            ll_prev(float): the likelihood of the current sample
            prior_new(float): the prior of the candidate
            prior_prev(float): the prior of the current sample
            q (float): the transition probability
            q_back (float): the back-probability
            temperature(float): the temperature of the MCMC
        Returns:
            (float): the metropolis-hastings ratio
        """
        ll_ratio = ll_new - ll_prev
        try:
            with _np.errstate(divide='ignore'):
                log_q_ratio = _math.log(q / q_back)

        except ZeroDivisionError:
            log_q_ratio = _math.inf

        prior_ratio = prior_new - prior_prev
        mh_ratio = (ll_ratio * temperature) - log_q_ratio + prior_ratio

        return mh_ratio

    def log_sample_statistics(self, sample, c, sample_id):
        """ This function logs the statistics of an MCMC sample.
        Args:
            sample (Sample): A Sample object consisting of zones and weights
            c (int): The current chain
            sample_id (int): Index of the logged sample.
        """
        self.statistics['sample_id'].append(sample_id)
        self.statistics['sample_zones'].append(sample.zones)
        self.statistics['sample_weights'].append(sample.weights)
        self.statistics['sample_p_global'].append(sample.p_global)
        self.statistics['sample_p_zones'].append(sample.p_zones)
        self.statistics['sample_p_families'].append(sample.p_families)
        self.statistics['sample_likelihood'].append(self._ll[c])
        self.statistics['sample_prior'].append(self._prior[c])

        if self.show_screen_log:
            print('Log-likelihood: %.2f' % self._ll[c])
            print('Accepted steps: %i' % self.statistics['accepted_steps'])

    def log_last_sample(self, last_sample):
        """ This function logs the last sample of the first chain of an MCMC run.
        Args:
            last_sample (Sample): A Sample object consisting of zones and weights
        """
        self.statistics['last_sample'] = last_sample

    def print_screen_log(self, i_step, sample):
        i_step_str = str.ljust(str(i_step), 12)

        likelihood = self.likelihood(sample[self.chain_idx[0]], self.chain_idx[0])
        likelihood_str = str.ljust('log-likelihood:  %.2f' % likelihood, 36)

        time_per_million = (_time.time() - self.t_start) / (i_step + 1) * 1000000
        time_str = '%i seconds / million steps' % time_per_million

        print(i_step_str + likelihood_str + time_str)
        # print('size0 =', 'sum(sample[self.chain_idx[0]].zones[0]))
