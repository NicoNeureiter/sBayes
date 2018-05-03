#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import random as _random
import logging

import numpy as np

from src.sampling.mcmc import ComponentMCMC
from src.model import compute_feature_likelihood, compute_geo_likelihood_generative,\
    compute_geo_likelihood_particularity, compute_likelihood_generative
from src.util import grow_zone, get_neighbours
from src.plotting import plot_zones
from src.config import *


class ZoneMCMC(ComponentMCMC):

    def __init__(self, network, features, min_size, max_size, p_transition_mode,
                 geo_weight, lh_lookup, n_zones=1, ecdf_geo=None,
                 feature_ll_mode=FEATURE_LL_MODE, geo_ll_mode=GEO_LL_MODE,
                 ecdf_type=GEO_ECDF_TYPE, random_walk_cov=None, connected_only=False,
                 print_logs=True, **kwargs):

        super(ZoneMCMC, self).__init__(n_zones, **kwargs)

        # Data
        self.network = network
        self.features = features
        self.adj_mat = network['adj_mat']
        self.locations = network['locations']
        self.graph = network['graph']

        # Sampling
        self.n_zones = n_zones
        self.n = self.adj_mat.shape[0]

        # Prior assumptions about zone size / connectedness
        self.min_size = min_size
        self.max_size = max_size
        self.connected_only = connected_only

        # Mode frequency / weight,
        self.p_transition_mode = p_transition_mode
        self.geo_weight = geo_weight * features.shape[1]

        # Likelihood modes
        self.feature_ll_mode = feature_ll_mode
        self.geo_ll_mode = geo_ll_mode
        self.ecdf_type = ecdf_type
        self.random_walk_cov = random_walk_cov

        # Look-up tables
        self.lh_lookup = lh_lookup
        self.ecdf_geo = ecdf_geo

        self.print_logs = print_logs

    def log_likelihood(self, zone):
        if np.ndim(zone) == 2:
            return sum([self.log_likelihood(z) for z in zone])

        # Compute the feature likelihood

        if self.feature_ll_mode == 'generative':
            # Compute the feature-likelihood based on a binomial distribution.
            ll_feature = compute_likelihood_generative(zone, self.features)
        elif self.feature_ll_mode == 'particularity':
            ll_feature = compute_feature_likelihood(zone, self.features, self.lh_lookup)
        else:
            raise ValueError('Unknown feature_ll_mode: %s' % self.feature_ll_mode)

        # Compute the geographic likelihood

        if self.geo_ll_mode == 'generative':
            # Compute the geo-likelihood based on a gaussian distribution.
            ll_geo = compute_geo_likelihood_generative(zone, self.network, self.random_walk_cov)
        elif self.geo_ll_mode == 'particularity':
            # Compute the empirical geo-likelihood of a zone
            ll_geo = compute_geo_likelihood_particularity(zone, self.network, self.ecdf_geo,
                                                          subgraph_type=self.ecdf_type)
        elif self.geo_ll_mode == 'none':
            ll_geo = 0
        else:
            raise ValueError('Unknown geo_ll_mode: %s' % self.geo_ll_mode)

        # Weight geo-likelihood
        ll_geo *= self.geo_weight

        return ll_feature + ll_geo

    def log_likelihood_rest(self, x):
        if self.feature_ll_mode == 'particularity':
            return 0.

        elif self.feature_ll_mode == 'generative':
            occupied = np.any(x, axis=0)

            features_free = self.features[~occupied, :]
            n, n_features = features_free.shape
            k = features_free.sum(axis=0)

            n_all = self.features.shape[0]
            k_all = self.features.sum(axis=0)
            p_all = (k_all / n_all)  # TODO pre-compute

            ll = np.sum(k * np.log(p_all) + (n - k) * np.log(1 - p_all))

            return ll

        else:
            raise ValueError

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

        # Select current zone
        i_zone = self._current_zone
        zone_prev = x_prev[i_zone, :]
        zone_new = zone_prev.copy()
        occupied = np.any(x_prev, axis=0)

        # Compute the neighbourhood
        neighbours = get_neighbours(zone_prev, occupied, self.adj_mat)
        neighbours_idx = neighbours.nonzero()[0]

        # Add a neighbour to the zone
        try:
            i_new = _random.choice(neighbours_idx)
        except IndexError as e:
            print(zone_new.nonzero(), neighbours_idx)
            raise e

        zone_new[i_new] = occupied[i_new] = 1

        # Remove a vertex from the zone.
        removal_candidates = self.get_removal_candidates(zone_new)
        i_out = _random.choice(removal_candidates)
        zone_new[i_out] = occupied[i_out] = 0

        # Compute transition probabilities
        back_neighbours = get_neighbours(zone_new, occupied, self.adj_mat)
        q = 1. / np.count_nonzero(neighbours)
        q_back = 1. / np.count_nonzero(back_neighbours)

        return zone_new, q, q_back

    def grow_step(self, x_prev):

        # Select current zone
        i_zone = self._current_zone
        zone_prev = x_prev[i_zone]
        zone_new = zone_prev.copy()
        occupied = np.any(x_prev, axis=0)

        size_prev = np.count_nonzero(zone_prev)
        size_new = size_prev + 1

        # Add a neighbour to the zone
        neighbours = get_neighbours(zone_prev, occupied, self.adj_mat)
        i_new = _random.choice(neighbours.nonzero()[0])
        zone_new[i_new] = occupied[i_new] = 1

        # Transition probability when growing
        q = 1 / np.count_nonzero(neighbours)
        if size_prev > self.min_size:
            q /= 2

        # Back-probability (-> shrinking)
        q_back = 1 / size_new
        if size_new < self.max_size:
            q_back /= 2

        return zone_new, q, q_back

    def shrink_step(self, x_prev):

        # Select current zone
        i_zone = self._current_zone
        zone_prev = x_prev[i_zone]
        zone_new = zone_prev.copy()
        occupied = np.any(x_prev, axis=0)

        size_prev = np.count_nonzero(zone_prev)
        size_new = size_prev - 1

        # Remove a vertex from the zone.
        removal_candidates = self.get_removal_candidates(zone_prev)
        i_out = _random.choice(removal_candidates)
        zone_new[i_out] = occupied[i_out] = 0

        # Transition probability when shrinking.
        q = 1 / len(removal_candidates)
        if size_prev < self.max_size:
            q /= 2

        # Back-probability (-> growing)
        back_neighbours = get_neighbours(zone_new, occupied, self.adj_mat)
        q_back = 1 / np.count_nonzero(back_neighbours)
        if size_new > self.min_size:
            q_back /= 2

        return zone_new, q, q_back

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

    def generate_initial_sample(self):
        """Generate initial sample zones by growing a zone through random
        grow-steps up to self.min_size.

        Returns:
            np.array: The generated initial zones.
        """
        self._lls = np.full(self.n_components, -np.inf)

        occupied = np.zeros(self.n, bool)
        init_zones = np.zeros((self.n_zones, self.n), bool)

        for i in range(self.n_zones):
            g = grow_zone(self.min_size, self.network, occupied)
            init_zones[i, :] = g[0]
            occupied = g[1]

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
            G_zone = self.graph.induced_subgraph(zone_idx)
            assert G_zone.is_connected()

            cut_vs_idx = G_zone.cut_vertices()
            if len(cut_vs_idx) == size:
                return []

            cut_vertices = G_zone.vs[cut_vs_idx]['name']

            return [v for v in zone_idx if v not in cut_vertices]

    def plot_sample(self, sample, ax=None):
        return plot_zones(sample, self.network, ax=ax)

    def log_sample_statistics(self, sample):
        super(ZoneMCMC, self).log_sample_statistics(sample)
        if self.print_logs:
            logging.info('Current zones log-likelihood: %s' % self._lls)
            logging.info('Current total log-likelihood: %s' % self._ll)
            logging.info('Current zone sizes:           %s' %
                         np.count_nonzero(sample, axis=-1))
