#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import random as _random
import logging

import numpy as np

from src.sampling.mcmc_particularity import ComponentMCMC_particularity
from src.model import compute_likelihood_particularity, \
    compute_geo_prior_distance, compute_likelihood_generative
from src.util import grow_zone, get_neighbours, ZoneError, categories_from_features
from src.plotting import plot_zones
from src.config import *


class ZoneMCMC_particularity(ComponentMCMC_particularity):

    def __init__(self, network, features, min_size, max_size, start_size,
                 p_transition_mode, lh_lookup, ecdf_type,
                 geo_prior_mode, initial_zones, lh_weight=1, n_zones=1,
                 connected_only=False,
                 print_logs=True, ecdf_geo=None, **kwargs):

        super(ZoneMCMC_particularity, self).__init__(n_components=n_zones, **kwargs)

        # Data
        self.features = features
        self.features_cat = categories_from_features(features)

        # Network
        self.network = network
        self.adj_mat = network['adj_mat']
        self.locations = network['locations']
        self.graph = network['graph']

        # Sampling
        self.n = self.adj_mat.shape[0]

        # Prior assumptions about zone size / connectedness /initial sample
        self.min_size = min_size
        self.max_size = max_size
        self.start_size = start_size
        self.connected_only = connected_only
        self.initial_zones = initial_zones

        # Mode frequency / weight
        self.p_transition_mode = p_transition_mode
        self.nr_features = features.shape[1]
        self.lh_weight = lh_weight

        # Likelihood modes
        self.geo_prior_mode = geo_prior_mode
        self.ecdf_type = ecdf_type

        # Look-up tables
        self.lh_lookup = lh_lookup
        self.ecdf_geo = ecdf_geo

        self.print_logs = print_logs

    def prior_zone(self, zone):

        if np.ndim(zone) == 2:
            return sum([self.prior_zone(z) for z in zone])

        # Compute the geographic prior

        elif self.geo_prior_mode == 'distance':
            # Compute the empirical geo-prior of a zone
            prior_geo = compute_geo_prior_distance(zone, self.network, self.ecdf_geo,
                                                   subgraph_type=self.ecdf_type)

        elif self.geo_prior_mode == 'none':
            prior_geo = 0
        else:
            raise ValueError('Unknown geo_prior_mode: %s' % self.geo_prior_mode)

        # Normalize geo-prior
        prior_geo *= self.nr_features

        return prior_geo

    def log_likelihood(self, zone):

        if np.ndim(zone) == 2:
            return sum([self.log_likelihood(z) for z in zone])

        ll_feature = compute_likelihood_particularity(zone, self.features, self.lh_lookup)

        return self.lh_weight * ll_feature

    def swap_step(self, x_prev):

        """Propose a transition by removing a vertex and adding another one from the
        neighbourhood.

        Args:
            x_prev (np.ndarray): Current zone given as a boolean array.

        Returns:
            np.array: The proposed new contact zone.
            float: The proposal probability.
            float: The probability to transition back (new_zone -> prev_zone)
        """

        # Select current zone
        i_zone = self._current_zone
        zone_prev = x_prev[i_zone, :]
        zone_new = zone_prev.copy()
        occupied = np.any(x_prev, axis=0)

        # Compute the neighbourhood
        neighbours = get_neighbours(zone_prev, occupied, self.adj_mat)

        if not np.any(neighbours):
            # When stuck, stay in zone and do not accept
            return zone_prev, 1, 0

        neighbours_idx = neighbours.nonzero()[0]
        i_new = _random.choice(neighbours_idx)
        zone_new[i_new] = occupied[i_new] = 1

        # Remove a vertex from the zone.
        removal_candidates = self.get_removal_candidates(zone_new)
        i_out = _random.choice(removal_candidates)
        zone_new[i_out] = occupied[i_out] = 0

        # Compute transition probabilities
        # back_neighbours = get_neighbours(zone_new, occupied, self.adj_mat)
        # q = 1. / np.count_nonzero(neighbours)
        # q_back = 1. / np.count_nonzero(back_neighbours)

        return zone_new, 1., 1. #q, q_back

    def grow_step(self, x_prev):

        # Select current zone
        i_zone = self._current_zone
        zone_prev = x_prev[i_zone]
        zone_new = zone_prev.copy()
        occupied = np.any(x_prev, axis=0)

        # Add a neighbour to the zone
        neighbours = get_neighbours(zone_prev, occupied, self.adj_mat)
        if not np.any(neighbours):
            # When stuck, stay in zone and do not accept
            return zone_prev, 1, 0

        i_new = _random.choice(neighbours.nonzero()[0])
        zone_new[i_new] = occupied[i_new] = 1

        # size_prev = np.count_nonzero(zone_prev)
        # size_new = size_prev + 1

        # # Transition probability when growing
        # q = 1 / np.count_nonzero(neighbours)
        # if size_prev > self.min_size:
        #     q /= 2
        #
        # # Back-probability (-> shrinking)
        # q_back = 1 / size_new
        # if size_new < self.max_size:
        #     q_back /= 2

        return zone_new, 1., 1.  # , q, q_back

    def shrink_step(self, x_prev):

        # Select current zone
        i_zone = self._current_zone
        zone_prev = x_prev[i_zone]
        zone_new = zone_prev.copy()
        occupied = np.any(x_prev, axis=0)

        # Remove a vertex from the zone.
        removal_candidates = self.get_removal_candidates(zone_prev)
        i_out = _random.choice(removal_candidates)
        zone_new[i_out] = occupied[i_out] = 0

        # size_prev = np.count_nonzero(zone_prev)
        # size_new = size_prev - 1

        # # Transition probability when shrinking.
        # q = 1 / len(removal_candidates)
        # if size_prev < self.max_size:
        #     q /= 2
        #
        # # Back-probability (-> growing)
        # back_neighbours = get_neighbours(zone_new, occupied, self.adj_mat)
        # q_back = 1 / np.count_nonzero(back_neighbours)
        # if size_new > self.min_size:
        #     q_back /= 2

        return zone_new, 1., 1.  # q, q_back

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
        i_zone = self._current_zone
        zone_prev = x_prev[i_zone]

        modeselektor = _random.random()

        if modeselektor < self.p_transition_mode['swap']:
            # Swap step
            return self.swap_step(x_prev)

        else:
            # Grow or shrink?
            grow = (modeselektor < self.p_transition_mode['grow'])

            # Ensure we don't exceed size limits
            prev_size = np.count_nonzero(zone_prev)
            assert self.min_size <= prev_size <= self.max_size
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
                    return x_prev[self._current_zone], 1., 1.

    def generate_initial_sample(self, c):
        """Generate initial sample zones by A) growing a zone through random
        grow-steps up to self.min_size, B) using the last sample of a previous run of the MCMC

        Returns:
            np.array: The generated initial zones.
        """

        occupied = np.zeros(self.n, bool)
        init_zones = np.zeros((self.n_components, self.n), bool)
        zones_generated = 0

        for i in range(len(self.initial_zones)):

            # Test if initial samples exist
            if self.initial_zones[i][c] is not None:

                init_zones[i, :] = self.initial_zones[i][c]
                occupied += self.initial_zones[i][c]
                zones_generated += 1

        components_with_iz = zones_generated

        # For the rest: create all zones from scratch
        while True:
            for i in range(components_with_iz, self.n_components):
                try:
                    g = grow_zone(self.start_size, self.network, occupied)
                    zones_generated += 1
                except ZoneError:
                    break

                init_zones[i, :] = g[0]
                occupied = g[1]

            if zones_generated == self.n_components:
                return init_zones

    def get_removal_candidates(self, zone):
        """Nodes which can be removed from the given zone. If connectedness is
        required (connected_only = True), only non-cutvertices are returned.

        Args:
            zone (np.array): The zone from which removal candidate is selected.

        Returns:
            list: Index-list of removal candidates.
        """
        zone_idx = zone.nonzero()[0]
        size = len(zone_idx)

        if not self.connected_only:
            # If connectedness is not required, all nodes are candidates.
            return zone_idx

        else:
            # Compute non cut-vertices (can be removed while keeping zone connected).
            g_zone = self.graph.induced_subgraph(zone_idx)
            assert g_zone.is_connected()

            cut_vs_idx = g_zone.cut_vertices()
            if len(cut_vs_idx) == size:
                return []

            cut_vertices = g_zone.vs[cut_vs_idx]['name']

            return [v for v in zone_idx if v not in cut_vertices]

    def log_sample_statistics(self, c=None):
        super(ZoneMCMC_particularity, self).log_sample_statistics(c)
        if self.print_logs:
            logging.info('Current zones log-likelihood: %s' % self._lls)
            logging.info('Current total log-likelihood: %s' % self._ll)

