#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import math as _math
import abc as _abc
import random as _random

import numpy as _np


class MCMC(metaclass=_abc.ABCMeta):

    """Base-class for MCMC samplers. Instantiable sub-classes have to implement
    some methods, like propose_step() and log_likelihood().
    The base-class provides options for simulated annealing [1] and for
    restarting of the markov chain after every sample.

    Attributes:
        restart_interval (int): Number of samples after which the markov chain is re-initialized.
        simulated_annealing (bool): Use simulated annealing to avoid getting
            stuck in local optima [1].
        alpha_annealing (float): The alpha parameter, controlling the cooling schedule
            for simulated annealing (ignored when simulated_annealing is False).
        plot_samples (bool): Plot the generated sample after every run.

        temperature (float): Simulated annealing temperature. Changes according to
            schedule when simulated_annealing is True, is constanty 1 otherwise.
        statistics (dict): Container for a set of statistics about the sampling run.

    [1] https://en.wikipedia.org/wiki/Simulated_annealing
    """

    def __init__(self, restart_interval=_np.inf, simulated_annealing=False,
                 alpha_annealing=1., plot_samples=True):

        self.restart_interval = restart_interval
        self.simulated_annealing = simulated_annealing
        self.alpha_annealing = alpha_annealing
        self.temperature = 1.

        self.plot_samples = plot_samples

        self.statistics = {
            'sample_likelihoods': [],
            'step_likelihoods': [],
            'accepted': 0,
            'acceptance_ratio': 0.
        }

        # State attributes
        self._ll = -float('inf')

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
        Preferably in high density areas. If the chain is restarted between samples,
        this should cover wide parts of the parameter-space (cover multiple modi).

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

    def metropolis_hastings_ratio(self, ll_new, ll_prev, q, q_back, temperature=1.):
        ll_ratio = ll_new - ll_prev
        log_q_ratio = _math.log(q / q_back)

        return (ll_ratio * temperature) - log_q_ratio

    def generate_samples(self, n_samples, n_steps, burn_in_steps):
        """Run the MCMC sampling procedure with Metropolis Hastings rejection
        step and options for simulated annealing, restarting. Samples are
        returned, statistics saved in self.statistics.

        Args:
            n_samples (int):The number of samples to generate.
            n_steps (int): The number of steps, the sampler should make before every sample.
            burn_in_steps (int): The number of burn in steps performed before the first sample.
                NB: If self.restart is active, a burn-in is needed after every sample

        Returns:
            list: The generated samples.
        """
        # Initialize statistics
        self.statistics = {
            'sample_likelihoods': [],
            'step_likelihoods': [],
            'acceptance_ratio': _math.nan,
            'accepted': 0}
        samples = []

        # Generate samples...

        for i_sample in range(n_samples):

            # Mode for restarting from an initial position for every sample
            if i_sample % self.restart_interval == 0:
                sample = self.generate_initial_sample()
                self._ll = self.log_likelihood(sample)

                for i_step in range(burn_in_steps):
                    sample = self.step(sample)

            # Run the chain to generate a sample
            for i_step in range(n_steps):
                # Set SA temperature
                if self.simulated_annealing:
                    self.temperature = ((i_step + 1) / n_steps) ** self.alpha_annealing

                sample = self.step(sample)

            samples.append(sample)
            self.log_sample_statistics(sample)

            if self.plot_samples:
                self.plot_sample(sample)

        self.statistics['acceptance_ratio'] = (self.statistics['accepted'] / (n_steps * n_samples))

        return samples

    def step(self, sample):
        # Get a candidate
        candidate, q, q_back = self.propose_step(sample)

        # Compute the log-likelihood and Metropolis-Hastings ratio
        ll_candidate = self.log_likelihood(candidate)
        mh_ratio = self.metropolis_hastings_ratio(ll_candidate, self._ll,q, q_back,
                                                  temperature=self.temperature)

        # Accept/reject according to MH-ratio
        if _math.log(_random.random()) < mh_ratio:
            sample = candidate
            self._ll = ll_candidate

            self.statistics['accepted'] += 1

        self.log_step_statistics(sample)

        return sample

    def log_sample_statistics(self, sample):
        self.statistics['sample_likelihoods'].append(self._ll)

    def log_step_statistics(self, sample):
        self.statistics['step_likelihoods'].append(self._ll)

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

        self._lls = _np.full(n_components, -_np.inf)
        self._ll_rest = - _np.inf

        self._current_zone = 0

    def step(self, sample):
        """Perform one MCMC step. Here, in one step every component is updated.

        Returns:
            SampleType: the updated sample.
        """
        for i_component in range(self.n_components):
            self._current_zone = i_component

            # Get a candidate
            candidate_i, q, q_back = self.propose_step(sample)
            candidate = sample.copy()
            candidate[i_component] = candidate_i

            # Compute the log-likelihood and Metropolis-Hastings ratio
            ll_candidate = self.log_likelihood(candidate_i)
            ll_rest = self.log_likelihood_rest(candidate)
            mh_ratio = self.metropolis_hastings_ratio(ll_candidate + ll_rest,
                                                      self._lls[i_component] + self._ll_rest,
                                                      q, q_back, temperature=self.temperature)

            # Accept/reject according to MH-ratio
            if _math.log(_random.random()) < mh_ratio:
                sample[i_component] = candidate_i
                self._lls[i_component] = ll_candidate
                self._ll_rest = ll_rest
                self._ll = _np.sum(self._lls) + ll_rest

                self.statistics['accepted'] += 1

            self.log_step_statistics(sample)

        return sample

    @_abc.abstractmethod
    def log_likelihood_rest(self, x):
        """

        Args:
            x (SampleType): The current posterior sample.

        Returns:
            float: Likelihood of the not-assigned samples
        """
        pass