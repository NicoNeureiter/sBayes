#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import random as _random
import numpy as np

from src.sampling.mcmc import MCMC
from src.util import grow_zone, get_neighbours


class ContactZoneMCMC(MCMC):

    def __init__(self, n_steps, net, min_size, max_size, p_transition_mode,
                 fixed_size=True, **kwargs):
        super(ContactZoneMCMC, self).__init__(n_steps, **kwargs)

        self.network = net
        self.adj_mat = net['adj_mat']

        self.min_size = min_size
        self.max_size = max_size

        self.p_transition_mode = p_transition_mode

        self.fixed_size = fixed_size

    def log_likelihood(self, x):
        return x.sum()

    def swap_step(self, x_prev):
        pass

    def grow_step(self, x_prev):
        pass

    def shrink_step(self, x_prev):
        pass

    def propose_step(self, x_prev):
        """This function proposes a new candidate zone in the network. The new zone differs
        from the previous one by exactly one vertex. An exception are global update steps, which
        are performed with probability p_global and should avoid getting stuck in local
        modes.

        Args:
            x_prev (np.array): the previous zone, which will be modified to generate
                the new one.
        Returns
            new_zone (np.array): The proposed new contact zone.
            q (float): The proposal probability.
            q_back (float): The probability to transition back (new_zone -> prev_zone)
        """
        modeselektor = _random.random()

        # Swap step
        if modeselektor < self.p_transition_mode['swap']:
            return self.swap_step(x_prev)

        # Grow or shrink?
        grow = (modeselektor < self.p_transition_mode['grow'])

        # Ensure we don't exceed size limits
        prev_size = np.count_nonzero(x_prev)
        if prev_size <= self.min_size:
            grow = True
        if prev_size >= self.max_size:
            grow = False

        if grow:
            return self.grow_step(x_prev, self.max_size)
        else:
            result = self.shrink_step(x_prev, self.max_size)

            if result:
                return result
            else:
                # No way to shrink while keeping the zone connected
                return x_prev

    def generate_initial_sample(self):
        if self.fixed_size:
            return grow_zone(self.max_size, self.adj_mat)
        else:
            return grow_zone(self.min_size, self.adj_mat)
