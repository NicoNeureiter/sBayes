#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import math as _math
import abc as _abc
import random as _random
import time as _time
import numpy as _np


class MCMC_particularity(metaclass=_abc.ABCMeta):

    """Base-class for MCMC samplers inpaticularity mode. Instantiable sub-classes have to implement
    some methods, like propose_step() and log_likelihood().
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

    def __init__(self, n_chains=4, swap_period=1000, chain_swaps=1, temperature=1.):

        # Sampling attributes
        self.n_chains = n_chains
        self.swap_period = swap_period
        self.chain_swaps = chain_swaps
        self.chain_idx = list(range(self.n_chains))
        self.temperature = temperature

        # Initialize statistics
        self.statistics = self.init_statistics()

        # State attributes
        self._ll = _np.full(self.n_chains, -_np.inf)
        self._prior = _np.full(self.n_chains, -_np.inf)
        self.zone = []

    def init_statistics(self):
        return {'sample_likelihoods': [],
                'sample_priors': [],
                'sample_zones': [],
                'step_likelihoods': [],
                'step_priors': [],
                'step_zones': [],
                'last_sample': [],
                'acceptance_ratio': _math.nan,
                'accepted': 0}

    @_abc.abstractmethod
    def prior_zone(self, x):
        """Compute the geo prior of the given sample.
        Args:
            x (SampleType): Sample.
        Returns:
            float: Geo prior of x
        """
        pass

    @_abc.abstractmethod
    def log_likelihood(self, x):
        """Compute the log-likelihood of the given sample.

        Args:
            x (SampleType): Sample.

        Returns:
            float: Log-likelihood of x
        """
        pass

    @_abc.abstractmethod
    def generate_initial_sample(self, c):
        """Generate an initial sample from which the run should be started.
        Preferably in high density areas.
        Args:
            c(int): number of chains
        Returns:
            SampleType: Initial zone.
        """
        pass

    @_abc.abstractmethod
    def propose_step(self, x_prev):
        """Propose a new candidate sample. Might be rejected later, due to the
        Metropolis-Hastings rejection step.

        Args:
            x_prev (SampleType): The previous sample.

        Returns:
            SampleType: The proposed new sample.
            float: The probability of the proposed step.
            float: The probability of a transition back (x_new -> x_prev).
        """
        pass

    def metropolis_hastings_ratio(self, ll_new, ll_prev, prior_new, prior_prev,
                                  q, q_back, temperature=1.):
        ll_ratio = ll_new - ll_prev
        try:
            log_q_ratio = _math.log(q / q_back)
        except ZeroDivisionError:
            log_q_ratio = _math.inf

        prior_ratio = prior_new - prior_prev
        return (ll_ratio * temperature) - log_q_ratio + prior_ratio

    def generate_samples(self, n_steps, n_samples, burn_in_steps, return_steps=False):
        """Run the MCMC sampling procedure for the particularity model with Metropolis Hastings rejection
        step and options for multiple chains. Samples are
        returned, statistics saved in self.statistics.

        Args:
            n_steps (int): The number of steps the sampler should make in total.
            n_samples (int): The number of samples the sampler should take.
            burn_in_steps (int): The number of burn in steps performed before the first sample.
            return_steps (boolean): Return only samples or each step?
        Returns:
            list: The generated samples.
        """

        steps_per_sample = int(_np.ceil(n_steps / n_samples))
        t_start = _time.time()

        # Initialize stats
        self.statistics['accepted_exchanges'] = 0
        self.statistics['attempted_exchanges'] = 0
        self.statistics['exchange_ratio'] = []

        # Generate samples using MCMC with several chains
        sample = [None] * self.n_chains

        # Generate initial samples for each chain
        for c in self.chain_idx:
            sample[c] = self.generate_initial_sample(c)

            # Compute the (log)-likelihood for the sample
            self._ll[c] = self.log_likelihood(sample[c])
            self._prior[c] = self.prior_zone(sample[c])

        # Generate burn-in samples for each chain
        for c in self.chain_idx:
            for i_step in range(burn_in_steps):
                sample[c] = self.step(sample[c], c)

        # Update samples
        for i_step in range(n_steps):

                # Generate samples for each chain
                for c in self.chain_idx:

                    sample[c] = self.step(sample[c], c)

                    # Only take samples from the first chain
                    if self.chain_idx[c] == 0:
                        self.zone = sample[c]

                        if return_steps:
                            self.log_step_statistics(c)

                        if i_step % steps_per_sample == 0:
                            self.log_sample_statistics(c)

                if i_step % self.swap_period == 0:

                    self.swap_chains(sample, self.temperature)

                if i_step % 1000 == 0:
                    print(i_step, ' steps taken')

                # Save the complete last sample
                if i_step % (n_steps-1) == 0 and i_step != 0:
                    for c in self.chain_idx:
                        self.log_last_sample(sample[c])

        t_end = _time.time()
        self.statistics['sampling_time'] = t_end - t_start
        self.statistics['time_per_sample'] = (t_end - t_start) / n_samples
        self.statistics['acceptance_ratio'] = (self.statistics['accepted'] / n_steps)

        self.statistics['exchange_ratio'] = (self.statistics['accepted_exchanges'] /
                                             self.statistics['attempted_exchanges'])

        return

    def swap_chains(self, sample, temperature=1.):

        for _ in range(self.chain_swaps):

            self.statistics['attempted_exchanges'] += 1

            # Chose random chains and try to swap them
            swap_from = 0
            swap_to, = _np.random.choice(range(1, self.n_chains), 1)

            # Compute lh and prior ratio for both chains
            ll_from = self.log_likelihood(sample[swap_from])
            prior_from = self.prior_zone(sample[swap_from])

            ll_to = self.log_likelihood(sample[swap_to])
            prior_to = self.prior_zone(sample[swap_to])

            ll_ratio = (ll_to - ll_from) * temperature
            prior_ratio = prior_to - prior_from

            ex_ratio = ll_ratio + prior_ratio

            # Swap in Metropolis Hastings step
            if _math.log(_random.random()) < ex_ratio:
                self.chain_idx[swap_from] = swap_to
                self.chain_idx[swap_to] = swap_from
                self.statistics['accepted_exchanges'] += 1

    def step(self, sample, c=None):

        # Get a candidate
        candidate, q, q_back = self.propose_step(sample)

        # Compute the log-likelihood of the candidate
        ll_candidate = self.log_likelihood(candidate)

        # Compute the prior for the candidate
        prior_candidate = self.prior_zone(candidate)

        if c is None:
            # Single chain sampling
            mh_ratio = self.metropolis_hastings_ratio(ll_new=ll_candidate,
                                                      ll_prev=self._ll,
                                                      prior_new=prior_candidate, prior_prev=self._prior,
                                                      q=q, q_back=q_back,
                                                      temperature=self.temperature)

            # Accept/reject according to MH-ratio and update
            if _math.log(_random.random()) < mh_ratio:
                sample = candidate
                self._ll = ll_candidate
                self._prior = prior_candidate
                self.statistics['accepted'] += 1

        else:
            # Multi chain sampling
            mh_ratio = self.metropolis_hastings_ratio(ll_new=ll_candidate,
                                                      ll_prev=self._ll[c],
                                                      prior_new=prior_candidate, prior_prev=self._prior[c],
                                                      q=q, q_back=q_back,
                                                      temperature=self.temperature)

            # Accept/reject according to MH-ratio
            if _math.log(_random.random()) < mh_ratio:
                sample = candidate
                self._ll[c] = ll_candidate
                self._prior[c] = prior_candidate
                self.statistics['accepted'] += 1

        return sample

    def log_sample_statistics(self, c=None):

        self.statistics['sample_zones'].append(self.zone)
        self.statistics['sample_likelihoods'].append(self._ll[c])
        self.statistics['sample_priors'].append(self._prior[c])

    def log_step_statistics(self, c=None):

        self.statistics['step_zones'].append(self.zone)
        self.statistics['step_likelihoods'].append(self._ll[c])
        self.statistics['step_priors'].append(self._prior[c])

    def log_last_sample(self, sample):
        self.statistics['last_sample'].append(sample)


class ComponentMCMC_particularity(MCMC_particularity, metaclass=_abc.ABCMeta):

    """Extends the default MCMC sampler to problems with multiple components.
    Every component is updated separately conditioned on the others (like Gibbs
    sampling). The update schedule is simple round-robin.

    Attributes:
        n_components (int): The number of components.

        _lls (_np.array): The log-likelihood for every component. Total
            log-likelihood is given by sum(_lls).
    """

    def __init__(self, n_components, **kwargs):
        self.n_components = n_components
        super(ComponentMCMC_particularity, self).__init__(**kwargs)

        self._lls = _np.full((self.n_chains, self.n_components), -_np.inf)
        self._priors = _np.full((self.n_chains, self.n_components), -_np.inf)
        self._current_zone = 0

    def step(self, sample, c=None):
        """Perform one MCMC step. Here, in one step every component is updated.

        Returns:
            SampleType: the updated sample.
        """

        sample = sample.copy()

        for i_component in range(self.n_components):

            self._current_zone = i_component

            # Get a candidate
            candidate_i, q, q_back = self.propose_step(sample)
            candidate = sample.copy()
            candidate[i_component] = candidate_i

            # Compute the log-likelihood
            ll_candidate = self.log_likelihood(candidate_i)

            # Evaluate the prior probability
            prior_candidate = self.prior_zone(candidate_i)

            # One chain
            if c is None:
                mh_ratio = self.metropolis_hastings_ratio(ll_new=ll_candidate,
                                                          ll_prev=self._lls[i_component],  #
                                                          prior_new=prior_candidate,
                                                          prior_prev=self._priors[i_component],
                                                          q=q, q_back=q_back,
                                                          temperature=self.temperature)
                if _math.log(_random.random()) < mh_ratio:
                    sample[i_component] = candidate_i
                    self._lls[i_component] = ll_candidate
                    self._priors[i_component] = prior_candidate

                    self._ll = _np.sum(self._lls)
                    self._prior = _np.sum(self._priors)
                    self.statistics['accepted'] += 1

            # Multiple chains
            else:
                mh_ratio = self.metropolis_hastings_ratio(ll_new=ll_candidate,
                                                          ll_prev=self._lls[c][i_component],
                                                          prior_new=prior_candidate,
                                                          prior_prev=self._priors[c][i_component],
                                                          q=q, q_back=q_back,
                                                          temperature=self.temperature)

                # Accept/reject according to MH-ratio
                if _math.log(_random.random()) < mh_ratio:
                    sample[i_component] = candidate_i
                    self._lls[c][i_component] = ll_candidate
                    self._priors[c][i_component] = prior_candidate

                    self._ll[c] = _np.sum(self._lls[c])
                    self._prior[c] = _np.sum(self._priors[c])
                    self.statistics['accepted'] += 1

        return sample

    def init_statistics(self):
        return {'sample_likelihoods': [[] for _ in range(self.n_components)],
                'sample_priors': [[] for _ in range(self.n_components)],
                'sample_zones': [[] for _ in range(self.n_components)],
                'step_likelihoods': [[] for _ in range(self.n_components)],
                'step_priors': [[] for _ in range(self.n_components)],
                'step_zones': [[] for _ in range(self.n_components)],
                'last_sample': [[] for _ in range(self.n_components)],
                'acceptance_ratio': _math.nan,
                'accepted': 0}

    def swap_chains(self, sample, temperature=1.):
        for _ in range(self.chain_swaps):

            self.statistics['attempted_exchanges'] += 1

            # Chose random chains and try to swap them
            swap_from = [i for i in range(len(self.chain_idx)) if self.chain_idx[i] == 0][0]
            rest = [i for i in range(len(self.chain_idx)) if self.chain_idx[i] != 0]
            swap_to, = _np.random.choice(rest, 1)

            # Compute lh and prior ratio for both chains
            ll_from = self.log_likelihood(sample[swap_from])
            prior_from = self.prior_zone(sample[swap_from])

            ll_to = self.log_likelihood(sample[swap_to])
            prior_to = self.prior_zone(sample[swap_to])

            ll_ratio = (ll_to - ll_from) * temperature
            prior_ratio = prior_to - prior_from

            ex_ratio = ll_ratio + prior_ratio

            # Swap in Metropolis Hastings step
            if _math.log(_random.random()) < ex_ratio:
                self.chain_idx[swap_from] = self.chain_idx[swap_to]
                self.chain_idx[swap_to] = 0
                self.statistics['accepted_exchanges'] += 1

            else:
                pass

    def log_sample_statistics(self, c=None):

        for i in range(self.n_components):

            self.statistics['sample_zones'][i].append(self.zone[i])
            self.statistics['sample_likelihoods'][i].append(self._lls[c][i])
            self.statistics['sample_priors'][i].append(self._priors[c][i])

    def log_step_statistics(self, c=None):

        for i in range(self.n_components):
            self.statistics['step_zones'].append(self.zone[i])
            self.statistics['step_likelihoods'].append(self._lls[c][i])
            self.statistics['step_priors'].append(self._priors[c][i])

    def log_last_sample(self, sample):
        for i in range(self.n_components):
            self.statistics['last_sample'][i].append(sample[i])
