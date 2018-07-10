#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import math as _math
import abc as _abc
import random as _random
import time as _time

import numpy as _np


class MCMC(metaclass=_abc.ABCMeta):

    """Base-class for MCMC samplers. Instantiable sub-classes have to implement
    some methods, like propose_step() and log_likelihood().
    The base-class provides options for simulated annealing [1], for Markov coupled MCMC (MC3)[2].

    Attributes:
        simulated_annealing (bool): Use simulated annealing to avoid getting
            stuck in local optima [1].
        alpha_annealing (float): The alpha parameter, controlling the cooling schedule
            for simulated annealing (ignored when simulated_annealing is False).
        mc3 (bool): Use MC3 to explore the features space with several coupled chains
        mc3_chains (int): Number of coupled chains for MC3 (ignored when MC 3 is False)
        mc3_delta_t (float):
        mc3_swap_period (int):
        plot_samples (bool): Plot the generated sample after every run.

        temperature (float): Simulated annealing temperature. Changes according to
            schedule when simulated_annealing is True, is constant (1) otherwise.
        statistics (dict): Container for a set of statistics about the sampling run.

    [1] https://en.wikipedia.org/wiki/Simulated_annealing
    [2] Altekar, Gautam, et al. "Parallel metropolis coupled Markov chain Monte Carlo for Bayesian phylogenetic inference." Bioinformatics 20.3 (2004): 407-415.
    """

    def __init__(self, simulated_annealing=False, alpha_annealing=1.,
                 mc3=False, n_mc3_chains=4, mc3_exchange_period=1000, mc3_delta_t=0.5, plot_samples=True):

        self.simulated_annealing = simulated_annealing
        self.alpha_annealing = alpha_annealing
        self.mc3 = mc3
        self.n_mc3_chains = n_mc3_chains
        self.mc3_exchange_period = mc3_exchange_period
        self.mc3_delta_t = mc3_delta_t
        self.plot_samples = plot_samples

        self.statistics = {
            'sample_likelihoods': [],
            'step_likelihoods': [],
            'accepted': 0,
            'acceptance_ratio': _math.nan
        }

        # State attributes
        if self.mc3:
            self._ll = _np.full(self.n_mc3_chains, -_np.inf)
            self._prior = _np.full(self.n_mc3_chains, -_np.inf)
            self.temperature = [1 / (1 + self.mc3_delta_t * i) for i in range(self.n_mc3_chains)]
            print(self.temperature)
        else:
            self._ll = -float('inf')
            self._prior = -float('inf')
            self.temperature = 1.

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
    def generate_initial_sample(self):
        """Generate an initial sample from which the run should be started.
        Preferably in high density areas.

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
        log_q_ratio = _math.log(q / q_back)
        prior_ratio = prior_new - prior_prev
        return (ll_ratio * temperature) - log_q_ratio + prior_ratio

    def generate_samples(self, n_steps, n_samples, burn_in_steps, return_steps=False):
        """Run the MCMC sampling procedure with Metropolis Hastings rejection
        step and options for simulated annealing and mc3. Samples are
        returned, statistics saved in self.statistics.

        Args:
            n_steps (int): The number of steps the sampler should make in total.
            n_samples (int): The number of samples the sampler should take.
            burn_in_steps (int): The number of burn in steps performed before the first sample.

        Returns:
            list: The generated samples.
        """

        # Initialize statistics
        self.statistics = {
            'sample_likelihoods': [],
            'sample_priors': [],
            'step_likelihoods': [],
            'step_priors': [],
            'acceptance_ratio': _math.nan,
            'accepted': 0}

        steps_per_sample = int(_np.ceil(n_steps / n_samples))
        samples = []
        t_start = _time.time()

        # Generate samples using Metropolis coupled MCMC
        if self.mc3:

            self.statistics['accepted_exchanges'] = 0
            self.statistics['exchange_ratio'] = []

            # We only store statistics for the chain with temperature = 1
            sample = [None] * self.n_mc3_chains

            # Generate initial samples for each chain
            for c in range(self.n_mc3_chains):

                sample[c] = self.generate_initial_sample()
                self._ll[c] = self.log_likelihood(sample[c])
                self._prior[c] = self.prior_zone(sample[c])

            # Generate burn-in samples for each chain
            for c in range(self.n_mc3_chains):
                for i_step in range(burn_in_steps):
                    sample[c] = self.step(sample[c], c)

            # Update samples for each chain
            for i_step in range(n_steps):

                    # Generate samples for each chain
                    for c in range(self.n_mc3_chains):

                            sample[c] = self.step(sample[c], c)

                            if self.temperature[c] == 1.:
                                samples.append(sample[c])

                                if return_steps:
                                    self.log_step_statistics(c)

                                if i_step % steps_per_sample == 0:
                                    self.log_sample_statistics(c)

                    # print('likelihood before:', self._ll)
                    # print('likelihood per zone before:', self._lls)
                    # print('prior before:', self._prior)
                    # print('prior per zone before:', self._priors)
                    # print('temperature before:', self.temperature)

                    if i_step % self.mc3_exchange_period == 0:
                        # Exchange the temperature
                        self.exchange_temperature(sample)

                        # print('likelihood after', self._ll)
                        # print('likelihood per zone after:', self._lls)
                        # print('prior after:', self._prior)
                        # print('prior per zone after:', self._priors)
                        # print('temperature after:', self.temperature)

                    if i_step % 1000 == 0:
                        print(i_step, ' steps taken')

        else:

            sample = self.generate_initial_sample()
            self._ll = self.log_likelihood(sample)
            self._prior = self.prior_zone(sample)

            for i_step in range(burn_in_steps):
                sample = self.step(sample)

            for i_step in range(n_steps):

                # Set SA temperature
                if self.simulated_annealing:

                    self.temperature = (i_step / n_steps) ** self.alpha_annealing

                sample = self.step(sample)
                samples.append(sample)

                if return_steps:
                    self.log_step_statistics()

                if i_step % steps_per_sample:
                    self.log_sample_statistics()

                if i_step % 1000 == 0:
                    print(i_step, ' steps taken')

            if self.plot_samples:
                self.plot_sample(sample)

        t_end = _time.time()
        self.statistics['sampling_time'] = t_end - t_start
        self.statistics['time_per_sample'] = (t_end - t_start) / n_samples
        self.statistics['acceptance_ratio'] = (self.statistics['accepted'] / n_steps)

        if self.mc3:
            self.statistics['exchange_ratio'] = (self.statistics['accepted_exchanges'] /
                                                 _np.floor(n_steps/self.mc3_exchange_period))

        return samples

    def exchange_temperature(self, sample):

        ex_from, ex_to = _np.random.choice(range(len(self.temperature)), 2, replace=False)

        ll_from = self.log_likelihood(sample[ex_from])
        temperature_from = self.temperature[ex_from]

        ll_to = self.log_likelihood(sample[ex_to])
        temperature_to = self.temperature[ex_to]

        ex_ratio = ll_from**temperature_to * ll_to**temperature_from /\
                   (ll_to**temperature_to * ll_from**temperature_from)

        if _random.random() < ex_ratio:
            self.temperature[ex_from] = temperature_to
            self.temperature[ex_to] = temperature_from
            self.statistics['accepted_exchanges'] =+ 1

    def step(self, sample, c=None):
        # Get a candidate
        candidate, q, q_back = self.propose_step(sample)

        # Compute the log-likelihood and Metropolis-Hastings ratio
        ll_candidate = self.log_likelihood(candidate)
        prior_candidate = self.prior_zone(candidate)

        if c is None:
            # Regular sampling
            mh_ratio = self.metropolis_hastings_ratio(ll_new=ll_candidate, ll_prev=self._ll,
                                                      prior_new=prior_candidate, prior_prev=self._prior,
                                                      q=q, q_back=q_back,
                                                      temperature=self.temperature)

            # Accept/reject according to MH-ratio
            if _math.log(_random.random()) < mh_ratio:
                sample = candidate
                self._ll = ll_candidate
                self._prior = prior_candidate
                self.statistics['accepted'] += 1

        else:
            # MC3 sampling
            mh_ratio = self.metropolis_hastings_ratio(ll_new=ll_candidate, ll_prev=self._ll[c],
                                                      prior_new=prior_candidate, prior_prev=self._prior[c],
                                                      q=q, q_back=q_back,
                                                      temperature=self.temperature[c])

            # Accept/reject according to MH-ratio
            if _math.log(_random.random()) < mh_ratio:
                sample = candidate
                self._ll[c] = ll_candidate
                self._prior[c] = prior_candidate
                self.statistics['accepted'] += 1

        return sample

    def log_sample_statistics(self, c=None):

        if c is None:
            self.statistics['sample_likelihoods'].append(self._ll)
            self.statistics['sample_priors'].append(self._prior)
        else:
            self.statistics['sample_likelihoods'].append(self._ll[c])
            self.statistics['sample_priors'].append(self._prior[c])

    def log_step_statistics(self, c=None):

        if c is None:
            self.statistics['step_likelihoods'].append(self._ll)
            self.statistics['step_priors'].append(self._prior)
        else:
            self.statistics['step_likelihoods'].append(self._ll[c])
            self.statistics['step_priors'].append(self._prior[c])

    @_abc.abstractmethod
    def plot_sample(self, x):
        """A single sample plotting method. This is called after the generation of
        each sample in generate_samples() if self.plot_samples is set.

        Args:
            x (SampleType): The sample to be plotted
        """
        pass


class ComponentMCMC(MCMC, metaclass=_abc.ABCMeta):

    """Extends the default MCMC sampler to problems with multiple components.
    Every component is updated separately conditioned on the others (like Gibbs
    sampling). The update schedule is simple round-robin.

    Attributes:
        n_components (int): The number of components.

        _lls (_np.array): The log-likelihood for every component. Total
            log-likelihood is given by sum(_lls).

    TODO Resolve the need to reset _lls after resetting the chain
         (now in generate_initial_sample)
    """

    def __init__(self, n_components, **kwargs):
        super(ComponentMCMC, self).__init__(**kwargs)

        self.n_components = n_components

        if self.mc3:

            self._lls = _np.full((self.n_mc3_chains, self.n_components), -_np.inf)
            self._priors = _np.full((self.n_mc3_chains, self.n_components), -_np.inf)

        else:

            self._lls = _np.full(self.n_components, -_np.inf)
            self._priors = _np.full(self.n_components, -_np.inf)

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

            # Compute the log-likelihood and Metropolis-Hastings ratio
            ll_candidate = self.log_likelihood(candidate_i)
            prior_candidate = self.prior_zone(candidate_i)

            if c is None:
                mh_ratio = self.metropolis_hastings_ratio(ll_new=ll_candidate, ll_prev=self._lls[i_component],
                                                          prior_new=prior_candidate, prior_prev=self._priors[i_component],
                                                          q=q, q_back=q_back,
                                                          temperature=self.temperature)
                if _math.log(_random.random()) < mh_ratio:
                    sample[i_component] = candidate_i
                    self._lls[i_component] = ll_candidate
                    self._priors[i_component] = prior_candidate
                    self._ll = _np.sum(self._lls)
                    self._prior = _np.sum(self._priors)
                    self.statistics['accepted'] += 1

            else:
                mh_ratio = self.metropolis_hastings_ratio(ll_new=ll_candidate, ll_prev=self._lls[c][i_component],
                                                          prior_new=prior_candidate, prior_prev=self._priors[c][i_component],
                                                          q=q, q_back=q_back,
                                                          temperature=self.temperature[c])

                # Accept/reject according to MH-ratio
                if _math.log(_random.random()) < mh_ratio:
                    sample[i_component] = candidate_i
                    self._lls[c][i_component] = ll_candidate
                    self._priors[c][i_component] = prior_candidate
                    self._ll[c] = _np.sum(self._lls[c])
                    self._prior[c] = _np.sum(self._priors[c])
                    self.statistics['accepted'] += 1

        return sample

    def exchange_temperature(self, sample):

        ex_from, ex_to = _np.random.choice(range(len(self.temperature)), 2, replace=False)

        ll_from = self.log_likelihood(sample[ex_from])
        temperature_from = self.temperature[ex_from]

        ll_to = self.log_likelihood(sample[ex_to])
        temperature_to = self.temperature[ex_to]

        ex_ratio = ll_from ** temperature_to * ll_to ** temperature_from / \
                   (ll_to ** temperature_to * ll_from ** temperature_from)

        print(ex_ratio)
        if _random.random() < ex_ratio:

            print('Exchange accepted')
            self.temperature[ex_from] = temperature_to
            self.temperature[ex_to] = temperature_from

            self.statistics['accepted_exchanges'] = + 1

        else:
            print('Exchange rejected')

    @_abc.abstractmethod
    def log_likelihood_rest(self, x):
        """

        Args:
            x (SampleType): The current posterior sample.

        Returns:
            float: Likelihood of the not-assigned samples
        """
        pass