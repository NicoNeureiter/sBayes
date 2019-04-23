#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import random as _random
import logging

import numpy as np

from src.sampling.mcmc_generative import MCMC_generative
from src.model import compute_geo_prior_distance, compute_likelihood_generative, compute_prior_zones
from src.util import get_neighbours, categories_from_features
from src.preprocessing import compute_p_global
from src.plotting import plot_zones
from src.config import *


class Sample(object):
    """
    Attributes:
        zones (np.array): Assignment of sites to zones.
            shape: (n_sites, )
        weights (np.array): Weights of zone, family and global likelihood for different features.
            shape: (n_features, 3)
    """

    def __init__(self, zones, weights):
        self.zones = zones
        self.weights = weights

    def copy(self):
        return Sample(self.zones.copy(), self.weights.copy())


class ZoneMCMC_generative(MCMC_generative):

    def __init__(self, network, features, min_size, max_size, initial_size, background,
                 zones_previous_run, n_zones=1,
                 connected_only=False, **kwargs):

        super(ZoneMCMC_generative, self).__init__(**kwargs)

        # Data
        self.features = features
        self.n_features = features.shape[1]

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
        self.initial_size = initial_size
        self.connected_only = connected_only
        self.n_zones = n_zones
        self.zones_previous_run = zones_previous_run

        # Global and family probabilities
        self.p_global = compute_p_global(features)

        # Todo compute_p_family
        #self.p.family = compute_p_family()

        # Proposal distribution
        self.var_proposal_weight = 0.1

    def prior(self, sample):
        """Compute the (log) prior of a sample.
        Args:
            sample(Sample): A Sample object consisting of zones and weights

        Returns:
            float: The (log) prior of the sample"""

        #Todo: Define and compute prior_zone and prior_weights
        prior_zones = compute_prior_zones(sample.zones, 'uniform')
        prior_weights = compute_prior_weights(sample.weights, 'uniform)
        prior = prior_zones + prior_weights
        return prior

    def likelihood(self, sample, step_type='nüt'):
        """Compute the (log) likelihood of a sample.
        Args:
            sample(Sample): A Sample object consisting of zones and weights

        Kwargs:
            step_type (str): What operator was used in the current step?

        Returns:
            float: The (log) likelihood of the sample"""
        # Normalize the weight
        weights_norm = self.normalize_weights(sample.weights)

        # Compute the likelihood
        is_zone_update = not (step_type == 'alter_weights')
        ll_feature = compute_likelihood_generative(zones=sample.zones, features=self.features,
                                                   p_global=self.p_global, weights=weights_norm,
                                                   is_zone_update=is_zone_update)

        return ll_feature

    # todo

    def alter_weights(self, sample):
        """This function modifies one weight of a random feature in the current sample

        Args:
            sample(Sample): The current sample with zones and weights.
        Returns:
            Sample: The modified sample
        """
        sample_new = sample.copy()
        weights_current = sample.weights

        # Randomly choose one of the features
        f_id = np.random.choice(range(weights_current.shape[0]))
        # Randomly choose to modify the global or the zone weight (family weight is adjusted during normalization)
        w_id = np.random.choice(range(weights_current.shape[1] - 1))

        weight_current = weights_current[f_id, w_id]
        # Sample new weight from normal distribution centered at the current weight and with fixed variance
        weight_new = np.random.normal(loc=weight_current, scale=self.var_proposal_weight)
        sample_new.weights[f_id, w_id] = weight_new

        # The proposal distribution is normal, so the transition and back probability are equal
        q = q_back = 1.

        return sample_new, q, q_back

    def swap_zone(self, sample):
        """ This functions swaps sites in one of the zones of the current sample
        (i.e. in of the zones a site is removed and another one added)
        Args:
            sample(Sample): The current sample with zones and weights.

        Returns:
            Sample: The modified sample.
         """

        sample_new = sample.copy()
        zones_current = sample.zones
        occupied = np.any(zones_current, axis=0)

        # Randomly choose one of the zones to modify
        z_id = np.random.choice(range(zones_current.shape[0]))
        zone_current = zones_current[z_id, :]

        # Find all neighbors that are not yet occupied by other zones
        neighbours = get_neighbours(zone_current, occupied, self.adj_mat)

        # When stuck (all neighbors occupied) return current sample and reject the step (q_back = 0)
        if not np.any(neighbours):
            q, q_back = 1., 0.
            return sample, q, q_back

        # Add a site to the zone
        site_new = _random.choice(neighbours.nonzero()[0])
        sample_new.zones[z_id, site_new] = 1

        # Remove a site from the zone
        removal_candidates = self.get_removal_candidates(zone_current)
        site_removed = _random.choice(removal_candidates)
        sample_new.zones[z_id, site_removed] = 0
        q = q_back = 1.

        return sample_new, q, q_back

    def grow_zone(self, sample):
        """ This functions grows one of the zones in the current sample (i.e. it adds a new site to one of the zones)
        Args:
            sample(Sample): The current sample with zones and weights.

        Returns:
            (Sample): The modified sample.
        """

        sample_new = sample.copy()
        zones_current = sample.zones
        occupied = np.any(zones_current, axis=0)

        # Randomly choose one of the zones to modify
        z_id = np.random.choice(range(zones_current.shape[0]))
        zone_current = zones_current[z_id, :]

        # Check if zone is small enough to grow
        current_size = np.count_nonzero(zone_current)

        if current_size > self.max_size:
            # Zone too big to grow: don't modify the sample and reject the step (q_back = 0)
            q, q_back = 1., 0.
            return sample, q, q_back

        # Find all neighbors that are not yet occupied by other zones
        neighbours = get_neighbours(zone_current, occupied, self.adj_mat)

        # When stuck (all neighbors occupied) return current sample and reject the step (q_back = 0)
        if not np.any(neighbours):
            q, q_back = 1., 0.
            return sample, q, q_back

        site_new = _random.choice(neighbours.nonzero()[0])
        sample_new.zones[z_id, site_new] = 1
        q = q_back = 1.

        return sample_new, q, q_back

    def shrink_zone(self, sample):
        """ This functions shrinks one of the zones in the current sample (i.e. it removes one site from one zone)

        Args:
            sample(Sample): The current sample with zones and weights.

        Returns:
            (Sample): The modified sample.
        """
        sample_new = sample.copy()
        zones_current = sample.zones
        z_id = np.random.choice(range(zones_current.shape[0]))

        # Randomly choose one of the zones to modify
        zone_current = zones_current[z_id, :]

        # Check if zone is big enough to shrink
        current_size = np.count_nonzero(zone_current)
        if current_size < self.min_size:
            # Zone is too small to shrink: don't modify the sample and reject the step (q_back = 0)
            q, q_back = 1., 0.
            return sample, q, q_back

        # Zone is big enough: shrink
        removal_candidates = self.get_removal_candidates(zone_current)
        site_removed = _random.choice(removal_candidates)
        sample_new.zones[z_id, site_removed] = 0
        q = q_back = 1.

        return sample_new, q, q_back

    def generate_initial_zones(self, c):
        """For each chain (c) generate initial zones by
        A) growing through random grow-steps up to self.min_size,
        B) using the last sample of a previous run of the MCMC
        Args:
            c (int): the current chain
        Returns:
            np.array: The generated initial zones.
        """

        occupied = np.zeros(self.n, bool)
        initial_zones = np.zeros((self.n_zones, self.n), bool)
        n_generated = 0

        # B: For those zones where a sample from a previous run exists we use this as the initial sample
        for i in range(len(self.zones_previous_run)):

            if self.zones_previous_run[i][c] is not None:
                initial_zones[i, :] = self.zones_previous_run[i][c]
                occupied += self.zones_previous_run[i][c]
                n_generated += 1

        not_initialized = range(n_generated, self.n_zones)

        # A: The zones that are not initialized yet are grown
        # When growing many zones some can get stuck due to an unfavourable seed.
        # That's why we perform several (15) attempts to initialize them.
        grow_attempts = 0
        while True:
            for i in not_initialized:
                try:
                    g = self.grow_zone_of_size_k(self.initial_size, occupied)

                except self.ZoneError:
                        # Might be due to an unfavourable seed
                        if grow_attempts < 15:
                            grow_attempts += 1
                            not_initialized = range(n_generated, self.n_zones)
                            break
                        # Seems there is not enough sites to grow n_zones of size k

                        else:
                            raise ValueError("Seems there are not enough sites (%i) to grow %i zones of size %i" %
                                             (self.n, self.n_zones, self.initial_size))
                n_generated += 1
                initial_zones[i, :] = g[0]
                occupied = g[1]

            if n_generated == self.n_zones:
                return initial_zones

    def grow_zone_of_size_k(self, k, already_in_zone=None):
        """ This function grows a zone of size k excluding any of the sites in <already_in_zone>.
        Args:
            k (int): The size of the zone, i.e. the number of sites in the zone.
            already_in_zone (np.array): All sites already assigned to a zone (boolean)

        Returns:
            np.array: The newly grown zone (boolean).
            np.array: all nodes in the network already assigned to a zone (boolean).

        """
        n_sites = self.n

        if already_in_zone is None:
            already_in_zone = np.zeros(n_sites, bool)

        # Initialize the zone
        zone = np.zeros(n_sites, bool)

        # Find all sites that already belong to a zone (sites_occupied) and those that don't (sites_free)
        sites_occupied = np.nonzero(already_in_zone)[0]
        sites_free = set(range(n_sites)) - set(sites_occupied)

        # Take a random free site and use it as seed for the new zone
        try:
            i = _random.sample(sites_free, 1)[0]
            zone[i] = already_in_zone[i] = 1
        except ValueError:
            raise self.ZoneError

        # Grow the zone if possible
        for _ in range(k - 1):

            neighbours = get_neighbours(zone, already_in_zone, self.adj_mat)
            if not np.any(neighbours):
                raise self.ZoneError

            # Add a neighbour to the zone
            site_new = _random.choice(neighbours.nonzero()[0])
            zone[site_new] = already_in_zone[site_new] = 1

        return zone, already_in_zone

    def generate_initial_weights(self):
        """This function generates initial weights for the Bayesian additive mixture model.
        Weights are in log-space and not normalized.

        Returns:
            list: weights for global, zone and family influence
            """
        # todo: smart initial value
        weights = np.full((self.n_features, 3), 1.)
        return weights

    def generate_initial_sample(self, c):
        """Generate initial Sample object (zones and weights)

        Returns:
            Sample: The generated initial Sample
        """
        initial_zones = self.generate_initial_zones(c)
        initial_weights = self.generate_initial_weights()
        sample = Sample(zones=initial_zones, weights=initial_weights)

        return sample

    def get_removal_candidates(self, zone):
        """Finds sites which can be removed from the given zone. If connectedness is
        required (connected_only = True), only non-cut vertices are returned.

        Args:
            zone (np.array): The zone for which removal candidates are found.

        Returns:
            (list): Index-list of removal candidates.
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

    class ZoneError(Exception):
        pass

    def get_operators(self, operators):
        """Get all relevant operator functions for proposing MCMC update steps and their probabilities

        Args:
            operators(dict): dictionary with names of all operators (keys) and their weights (values)

        Returns:
            list, list: the operator functions (callable), their weights (float)
        """
        fn_operators = []
        p_operators = []

        for k, v in operators.items():
            fn_operators.append(getattr(self, k))
            p_operators.append(v)

        return fn_operators, p_operators

    def normalize_weights(self, log_weights):
        """Transform the weights from log space and normalize them such that they sum to 1

        Args:
            log_weights (np.array): The non-normalized weights in log space
                shape(n_features, 3)
        Returns:
            (np.array): transformed and normalized weights
         """
        # Transform to original space
        weights = np.exp(log_weights)
        # Normalize
        weights_norm = weights/weights.sum(axis=1, keepdims=True)

        return weights_norm
