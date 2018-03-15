#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import random as _random
import numpy as np

from src.sampling.mcmc import MCMC
from src.model import compute_likelihood, compute_geo_likelihood,\
    compute_empirical_geo_likelihood
from src.util import grow_zone, get_neighbours


class ZoneMCMC(MCMC):

    def __init__(self, network, features, n_steps, min_size, max_size, p_transition_mode,
                 geo_weight, lh_lookup, ecdf_geo,
                 feature_ll_mode='generative', geo_ll_mode='gaussian',
                 ecdf_type='complete', geo_std=None, fixed_size=True,
                 connected_only=False, **kwargs):

        super(ZoneMCMC, self).__init__(n_steps, **kwargs)

        # Data
        self.network = network
        self.features = features
        self.adj_mat = network['adj_mat']
        self.locations = network['locations']
        self.graph = network['graph']

        # Prior assumptions about zone size / connectedness
        self.min_size = min_size
        self.max_size = max_size
        self.fixed_size = fixed_size
        self.connected_only = connected_only

        # Mode frequency / weight,
        self.p_transition_mode = p_transition_mode
        self.geo_weight = geo_weight

        # Likelihood modes
        self.feature_ll_mode = feature_ll_mode
        self.geo_ll_mode = geo_ll_mode
        self.ecdf_type = ecdf_type
        if geo_std:
            self.geo_std = geo_std
        else:
            self.geo_std = 0.1 * np.std(self.locations, axis=0)

        # Look-up tables
        self.lh_lookup = lh_lookup
        self.ecdf_geo = ecdf_geo


    def log_likelihood(self, x):
        ll_feature = compute_likelihood(x, self.features, self.lh_lookup)

        if self.geo_ll_mode == "Gaussian":
            # Compute the geo-likelihood based on a gaussian distribution.
            ll_geo = compute_geo_likelihood(x, self.network, self.geo_std)
        elif self.geo_ll_mode == "Empirical":
            # Compute the empirical geo-likelihood of a zone
            ll_geo = compute_empirical_geo_likelihood(x, self.network, self.ecdf_geo,
                                                      self.ecdf_type)
        else:
            raise ValueError('Unknown geo_ll_mode: %s' % self.geo_ll_mode)

        return ll_feature + self.geo_weight * ll_geo

    def swap_step(self, x_prev, n_swaps=1):
        """Propose a transition by removing a vertex and adding another one from the
        neighbourhood.

        Args:
            x_prev (np.ndarray): Current zone given as a boolean array.
            n_swaps (int): The maximum number of swaps to perform in this step.

        Returns:
            np.array: The proposed new contact zone.
            float: The proposal probability.
            float: The probability to transition back (new_zone -> prev_zone)
        """
        x_new = x_prev.copy()

        size = np.count_nonzero(x_prev)

        # Compute the neighbourhood
        neighbours = get_neighbours(x_prev, self.adj_mat)
        neighbours_idx = neighbours.nonzero()[0]

        # Add a neighbour to the zone
        i_new = _random.choice(neighbours_idx)
        x_new[i_new] = 1

        zone_idx = x_prev.nonzero()[0]

        if self.connected_only:
            # # Compute cut_vertices (can be removed while keeping zone connected)
            G_zone = self.graph.induced_subgraph(zone_idx)
            assert G_zone.is_connected()

            cut_vs_idx = G_zone.cut_vertices()
            if len(cut_vs_idx) == size + 1:
                return None

            cut_vertices = G_zone.vs[cut_vs_idx]['name']
            removal_candidates = [v for v in zone_idx if v not in cut_vertices]
        else:
            removal_candidates = zone_idx

        # Remove a vertex from the zone.
        i_out = _random.choice(removal_candidates)
        x_new[i_out] = 0

        back_neighbours = get_neighbours(x_new, self.adj_mat)
        q = 1. / np.count_nonzero(neighbours)
        q_back = 1. / np.count_nonzero(back_neighbours)

        return x_new, q, q_back

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
            return self.grow_step(x_prev)
        else:
            result = self.shrink_step(x_prev)

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
