#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import math as _math
import abc as _abc
import random as _random


class MCMC(metaclass=_abc.ABCMeta):

    def __init__(self, n_steps, restart_chain=False,
                 simulated_annealing=False, alpha_annealing=1.):
        self.n_steps = n_steps

        self.restart_chain = restart_chain
        self.simulated_annealing = simulated_annealing
        self.alpha_annealing = alpha_annealing
        self.temperature = 1.

        self.statistics = {
            'sample_likelihoods': [],
            'step_likelihoods': [],
            'acceptance_ratio': 0.
        }


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
        """

        Args:
            x_prev: The previous sample.

        Returns:
            SampleType: The proposed new sample.
            float: The probability of the proposed step.
            float: The probability of a transition back (x_new -> x_prev).
        """
        pass

    def metropolis_hastings_ratio(self, ll_new, ll_prev, q, q_back, temperature=1.):
        ll_ratio = ll_new - ll_prev
        log_q_ratio = _math.log(q / q_back)

        return (ll_ratio ** temperature) - log_q_ratio

    def generate_samples(self, n_samples):
        # Initialize statistics
        self.statistics = {
            'sample_likelihoods': [],
            'step_likelihoods': [],
            'acceptance_ratio': _math.nan}
        accepted = 0
        samples = []

        x = self.generate_initial_sample()
        ll = self.log_likelihood(x)

        for i_sample in range(n_samples):
            if self.restart_chain:
                x = self.generate_initial_sample()

            for i_step in range(self.n_steps):
                # Set SA temperatire
                if self.simulated_annealing:
                    temperature = ((i_step + 1) / self.n_steps) ** self.alpha_annealing
                else:
                    temperature = 1.

                # Get a candidate
                x_candidate, q, q_back = self.propose_step(x)

                # Compute the log-likelihood and Metropolis-Hastings ratio
                ll_candidate = self.log_likelihood(x_candidate)
                mh_ratio = self.metropolis_hastings_ratio(ll_candidate, ll, q, q_back,
                                                          temperature=temperature)

                # Accept/reject according to MH-ratio
                if _math.log(_random.random()) < mh_ratio:
                    x = x_candidate
                    ll = ll_candidate
                    accepted += 1

                self.statistics['step_likelihoods'].append(ll)

            samples.append(x)
            self.statistics['sample_likelihoods'].append(ll)

        self.statistics['acceptance_ratio'] = accepted / (self.n_steps * n_samples)

        return samples
