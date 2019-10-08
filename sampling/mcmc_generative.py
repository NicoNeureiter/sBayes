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


        statistics (dict): Container for a set of statistics about the sampling run.

    [1]  Altekar, Gautam, et al. "Parallel metropolis coupled Markov chain Monte Carlo for Bayesian phylogenetic inference." Bioinformatics 20.3 (2004): 407-415.
    """

    def __init__(self, operators, inheritance, families, prior, sample_p,  global_freq,
                 known_initial_weights=None, known_initial_zones=None, n_chains=4, swap_period=1000, chain_swaps=1,
                 show_screen_log=False):

        # Sampling attributes
        self.n_chains = n_chains
        self.swap_period = swap_period
        self.chain_swaps = chain_swaps
        self.chain_idx = list(range(self.n_chains))

        # Todo: Remove after testing
        self.known_initial_weights = known_initial_weights
        self.known_initial_zones = known_initial_zones

        # Operators
        self.fn_operators, self.p_operators = self.get_operators(operators)

        # Is inheritance (information on language families) available?
        self.inheritance = inheritance
        self.families = families

        # Prior
        self.geo_prior = prior['geo_prior']
        self.geo_prior_parameters = prior['geo_prior_parameters']
        self.prior_weights = prior['weights']
        self.prior_p_zones = prior['p_zones']
        self.prior_p_families = prior['p_families']
        self.prior_p_families_parameters = prior['p_families_parameters']

        # Global frequencies
        self.global_freq = global_freq

        # Sample the probabilities of categories (global, in zones and in families)?
        self.sample_p_global = sample_p['global']
        self.sample_p_zones = sample_p['zones']
        self.sample_p_families = sample_p['families']

        # Initialize statistics
        self.statistics = {'sample_likelihood': [],
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
                           'swap_ratio': []}

        # State attributes
        self._ll = _np.full(self.n_chains, -_np.inf)
        self._prior = _np.full(self.n_chains, -_np.inf)

        self.show_screen_log = show_screen_log

    @_abc.abstractmethod
    def prior(self, x, c):
        """Compute the prior of the sample
        Args:
            x (Sample): Sample object
            c (int): Current chain
        Returns:
            float: the prior of x
        """
        pass

    @_abc.abstractmethod
    def likelihood(self, x, c):
        """Compute the (log) likelihood of the given sample.

        Args:
            x (Sample): The current sample.
            c (int): The current chain.
        Returns:
            float: (log)likelihood of x
        """
        pass

    @_abc.abstractmethod
    def generate_initial_sample(self):
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

            sample[c] = self.generate_initial_sample()
            # Compute the (log)-likelihood and the prior for each sample
            self._ll[c] = self.likelihood(sample[c], c)
            self._prior[c] = self.prior(sample[c], c)

        # Probability of operators in burn-in is different to post-burn-in
        p_operators_post_burn_in = list(self.p_operators)

        # For burn-in we set the weights operator to 0. to make it easier to find zones
        operator_names = [f.__name__ for f in self.fn_operators]
        i_weights = operator_names.index('alter_weights')
        self.p_operators[i_weights] = 0.

        # Re-normalize
        self.p_operators = [p/sum(self.p_operators) for p in self.p_operators]

        # Generate burn-in samples for each chain
        for c in self.chain_idx:
            for i_step in range(burn_in_steps):
                sample[c] = self.step(sample[c], c)

        # Generate post burn-in samples
        self.p_operators = p_operators_post_burn_in

        for i_step in range(n_steps):

                # Generate samples for each chain
                for c in self.chain_idx:
                    sample[c] = self.step(sample[c], c)

                # Log samples at fixed intervals
                if i_step % steps_per_sample == 0:

                    # Log samples, but only from the first chain
                    self.log_sample_statistics(sample[self.chain_idx[0]], c=self.chain_idx[0])

                # Exchange chains at fixed intervals
                if i_step % self.swap_period == 0:
                    self.swap_chains(sample)

                # Print work status at fixed intervals
                if i_step % 1000 == 0:
                    print(i_step, ' steps taken')

                # Log the last sample of the first chain
                if i_step % (n_steps-1) == 0 and i_step != 0:
                    self.log_last_sample(sample[self.chain_idx[0]])

        t_end = _time.time()
        self.statistics['sampling_time'] = t_end - t_start
        self.statistics['time_per_sample'] = (t_end - t_start) / n_samples
        self.statistics['acceptance_ratio'] = (self.statistics['accepted_steps'] / n_steps)
        self.statistics['swap_ratio'] = (self.statistics['accepted_swaps'] / self.statistics['n_swaps'])

        return

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

        # Set all 'what_changed' flags to true (to avoid caching errors)
        for s in sample:
            for d in s.what_changed.values():
                for k in d.keys():
                    d[k] = True

    def step(self, sample, c):
        """This function performs a full MH step: first, a new candidate sample is proposed
        for either the zones or the weights, then the candidate is evaluated against the current sample
        and accepted with metropolis hastings acceptance probability

        Args:
            sample(Sample): A Sample object consisting of zones and weights
            c(int): the current chain of the MC3
        Returns:
            Sample: A Sample object consisting of zones and weights"""

        # Randomly choose one operator to propose new sample (grow/shrink/swap zones, alter weights/p_zones/p_families)

        propose_step = _np.random.choice(self.fn_operators, 1, p=self.p_operators)[0]
        candidate, q, q_back = propose_step(sample)

        # Compute the log-likelihood of the candidate
        ll_candidate = self.likelihood(candidate, c)

        # Compute the prior of the candidate
        prior_candidate = self.prior(candidate, c)

        # Evaluate the metropolis-hastings ratio
        mh_ratio = self.metropolis_hastings_ratio(ll_new=ll_candidate, ll_prev=self._ll[c],
                                                  prior_new=prior_candidate, prior_prev=self._prior[c],
                                                  q=q, q_back=q_back)

        # Accept/reject according to MH-ratio and update
        if _math.log(_random.random()) < mh_ratio:
            sample = candidate
            self._ll[c] = ll_candidate
            self._prior[c] = prior_candidate
            self.statistics['accepted_steps'] += 1

            #print(propose_step.__name__, " accepted", _np.count_nonzero(sample.zones), "zone size")
            # if (propose_step.__name__ == 'grow_zone'):
            #     print('grow step accepted')
            #     pass
            # elif(propose_step.__name__ == 'shrink_zone'):
            #     pass
            #     print('shrink step accepted')

        else:
            if not (propose_step.__name__ == 'alter_weights'):
                pass
                #print('zone step rejected')
            else:
                pass
                #print('weight step rejected')

        return sample

    def metropolis_hastings_ratio(self, ll_new, ll_prev, prior_new, prior_prev, q, q_back, temperature=1.):
        """ Computes the metropolis-hastings ratio.
        Args:
            ll_new(float): the likelihood of the candidate
            ll_prev(float): the likelihood of the current sample
            prior_new(float): the prior of the candidate
            prior_prev(float): tself.fn_operatorshe prior of the current sample
            q (float): the transition probability
            q_back (float): the back-probability
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

