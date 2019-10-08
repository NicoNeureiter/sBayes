#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import random as _random
import logging
from copy import deepcopy

import numpy as np

from src.sampling.mcmc_generative import MCMC_generative
from src.model import GenerativeLikelihood, GenerativePrior
from src.util import get_neighbours, transform_p_to_log, balance_p_array


class Sample(object):
    """
    Attributes:
        zones (np.array): Assignment of sites to zones.
            shape: (n_sites, )
        weights (np.array): Weights of zone, family and global likelihood for different features.
            shape: (n_features, 3)
        p_zones (np.array): Probabilities of categories in zones
            shape: (n_zones, n_features, n_categories)
        p_families (np.array): Probabilities of categories in families
            shape: (n_families, n_features, n_categories)
    """

    def __init__(self, zones, weights, p_zones, p_families):
        self.zones = zones
        self.weights = weights
        self.p_zones = p_zones
        self.p_families = p_families

        # The sample contains information about which of its parameters was changed in the last MCMC step
        self.what_changed = {'lh': {'zones': True, 'weights': True, 'p_zones': True, 'p_families': True},
                             'prior': {'zones': True, 'weights': True, 'p_zones': True, 'p_families': True}}

    def copy(self):
        zone_copied = deepcopy(self.zones)
        weights_copied = deepcopy(self.weights)
        what_changed_copied = deepcopy(self.what_changed)

        if self.p_zones is not None:
            p_zones_copied = self.p_zones.copy()
        else:
            p_zones_copied = None

        if self.p_families is not None:
            p_families_copied = self.p_families.copy()
        else:
            p_families_copied = None

        new_sample = Sample(zones=zone_copied, weights=weights_copied,
                            p_zones=p_zones_copied, p_families=p_families_copied)

        new_sample.what_changed = what_changed_copied

        return new_sample


class ZoneMCMC_generative(MCMC_generative):

    def __init__(self, network, features, min_size, max_size, initial_size,
                 initial_sample, n_zones=1, connected_only=False, **kwargs):

        super(ZoneMCMC_generative, self).__init__(**kwargs)

        # Data
        self.features = features
        self.n_features = features.shape[1]
        self.sites_per_category = np.count_nonzero(features, axis=0)

        # Network
        self.network = network
        self.adj_mat = network['adj_mat']
        self.locations = network['locations']
        self.graph = network['graph']

        # Sampling
        self.n = self.adj_mat.shape[0]

        # Zone size / connectedness /initial sample
        self.min_size = min_size
        self.max_size = max_size
        self.initial_size = initial_size
        self.connected_only = connected_only
        self.n_zones = n_zones
        self.initial_sample = initial_sample

        # Families
        if self.inheritance:
            self.n_families = self.families.shape[0]

        # Proposal distribution
        self.var_proposal_weight = 0.1
        self.var_proposal_p_zones = 0.1
        self.var_proposal_p_families = 0.1

        self.compute_lh_per_chain = [
            GenerativeLikelihood() for _ in range(self.n_chains)
        ]

        self.compute_prior_per_chain = [
            GenerativePrior() for _ in range(self.n_chains)
        ]

    def prior(self, sample, chain):
        """Compute the (log) prior of a sample.
        Args:
            sample(Sample): A Sample object consisting of zones and parameters
            chain(int): The current chain
        Returns:
            float: The (log) prior of the sample"""

        # Compute the prior
        log_prior = self.compute_prior_per_chain[chain](sample=sample, geo_prior=self.geo_prior,
                                                        geo_prior_parameters=self.geo_prior_parameters,
                                                        prior_weights=self.prior_weights,
                                                        prior_p_zones=self.prior_p_zones,
                                                        prior_p_families=self.prior_p_families,
                                                        prior_p_families_parameters=self.prior_p_families_parameters,
                                                        network=self.network)
        return log_prior

    def likelihood(self, sample, chain):
        """Compute the (log) likelihood of a sample.
        Args:
            sample(Sample): A Sample object consisting of zones and parameters.
            chain (int): The current chain
        Returns:
            float: The (log) likelihood of the sample"""

        # Compute the likelihood
        log_lh = self.compute_lh_per_chain[chain](sample=sample, features=self.features, families=self.families,
                                                  global_freq=self.global_freq,
                                                  inheritance=self.inheritance,
                                                  sample_p_zones=self.sample_p_zones,
                                                  sample_p_families=self.sample_p_families)

        return log_lh

    def alter_weights(self, sample):
        """This function modifies one weight of one feature in the current sample

        Args:
            sample(Sample): The current sample with zones and parameters.
        Returns:
            Sample: The modified sample
        """
        sample_new = sample.copy()
        # The step changed the weights (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['weights'] = sample_new.what_changed['prior']['weights'] = True
        sample.what_changed['lh']['weights'] = sample.what_changed['prior']['weights'] = True

        weights_current = sample.weights

        # Randomly choose one of the features
        f_id = np.random.choice(range(weights_current.shape[0]))

        if not self.inheritance:
            # The contact weights (column 1 in weights) is modified,
            # the inheritance weight is not relevant, the global weight is adjusted during normalization
            w_id = 1

        else:
            # The contact or family weights (column 1 or 2 in weights) are modified,
            # the global weight is adjusted during normalization
            w_id = np.random.choice([1, 2])

        weight_current = weights_current[f_id, w_id]
        # Sample new weight from normal distribution centered at the current weight and with fixed variance
        weight_new = np.random.normal(loc=weight_current, scale=self.var_proposal_weight)
        sample_new.weights[f_id, w_id] = weight_new

        # The proposal distribution is normal, so the transition and back probability are equal
        q = q_back = 1.

        return sample_new, q, q_back

    def alter_p_zones(self, sample):
        """This function modifies one p_zones of one category, one feature and in zone in the current sample
            Args:
                sample(Sample): The current sample with zones and parameters.
            Returns:
                Sample: The modified sample
                """
        sample_new = sample.copy()
        # The step changed p_zones (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['p_zones'] = sample_new.what_changed['prior']['p_zones'] = True
        sample.what_changed['lh']['p_zones'] = sample.what_changed['prior']['p_zones'] = True

        p_zones_current = sample.p_zones

        # Randomly choose one of the zones, one of the features and one of the categories
        z_id = np.random.choice(range(self.n_zones))
        f_id = np.random.choice(range(self.n_features))

        # Different features have different numbers of categories
        cat_f = np.where(self.sites_per_category[f_id] != 0)[0]
        cat_id = np.random.choice(cat_f)

        p_current = p_zones_current[z_id, f_id, cat_id]

        # Sample new p from normal distribution centered at the current p and with fixed variance
        p_new = np.random.normal(loc=p_current, scale=self.var_proposal_p_zones)

        sample_new.p_zones[z_id, f_id, cat_id] = p_new

        # The proposal distribution is normal, so the transition and back probability are equal
        q = q_back = 1.
        return sample_new, q, q_back

    def alter_p_families(self, sample):
        """This function modifies one p_families of one category, one feature and one family in the current sample
            Args:
                 sample(Sample): The current sample with zones and parameters.
            Returns:
                 Sample: The modified sample
        """

        sample_new = sample.copy()
        # The step changed p_families (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['p_families'] = sample_new.what_changed['prior']['p_families'] = True
        sample.what_changed['lh']['p_families'] = sample.what_changed['prior']['p_families'] = True

        p_families_current = sample.p_families

        # Randomly choose one of the families, one of the features and one of the categories
        fam_id = np.random.choice(range(self.n_families))
        f_id = np.random.choice(range(self.n_features))

        # Different features have different numbers of categories
        cat_f = np.where(self.sites_per_category[f_id] != 0)[0]
        cat_id = np.random.choice(cat_f)

        p_current = p_families_current[fam_id, f_id, cat_id]

        # Sample new p from normal distribution centered at the current p and with fixed variance
        p_new = np.random.normal(loc=p_current, scale=self.var_proposal_p_families)

        sample_new.p_families[fam_id, f_id, cat_id] = p_new

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
        # The step changed the zone (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['zones'] = sample.what_changed['lh']['zones'] = True
        sample_new.what_changed['prior']['zones'] = sample.what_changed['prior']['zones'] = True

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
        # The step changed the zone (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['zones'] = sample.what_changed['lh']['zones'] = True
        sample_new.what_changed['prior']['zones'] = sample.what_changed['prior']['zones'] = True

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
        # The step changed the zone (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['zones'] = sample.what_changed['lh']['zones'] = True
        sample_new.what_changed['prior']['zones'] = sample.what_changed['prior']['zones'] = True

        zones_current = sample.zones

        # Randomly choose one of the zones to modify
        z_id = np.random.choice(range(zones_current.shape[0]))
        zone_current = zones_current[z_id, :]

        # Check if zone is big enough to shrink
        current_size = np.count_nonzero(zone_current)
        if current_size <= self.min_size:
            # Zone is too small to shrink: don't modify the sample and reject the step (q_back = 0)
            q, q_back = 1., 0.
            return sample, q, q_back

        # Zone is big enough: shrink
        removal_candidates = self.get_removal_candidates(zone_current)
        site_removed = _random.choice(removal_candidates)
        sample_new.zones[z_id, site_removed] = 0
        q = q_back = 1.

        return sample_new, q, q_back

    def generate_initial_zones(self):
        """For each chain (c) generate initial zones by
        A) growing through random grow-steps up to self.min_size,
        B) using the last sample of a previous run of the MCMC

        Returns:
            np.array: The generated initial zones.
        """

        occupied = np.zeros(self.n, bool)
        initial_zones = np.zeros((self.n_zones, self.n), bool)
        n_generated = 0

        # B: For those zones where a sample from a previous run exists we use this as the initial sample
        if self.initial_sample.zones is not None:
            for i in range(len(self.initial_sample.zones)):

                initial_zones[i, :] = self.initial_sample.zones[i]
                occupied += self.initial_sample.zones[i]
                n_generated += 1

        not_initialized = range(n_generated, self.n_zones)

        # A: The zones that are not initialized yet are grown
        # When growing many zones, some can get stuck due to an unfavourable seed.
        # That's why we perform several attempts to initialize them.
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
        """This function generates initial weights for the Bayesian additive mixture model, either by
        A) proposing them (randomly) from scratch
        B) using the last sample of a previous run of the MCMC
        Weights are in log-space and not normalized.

        Returns:
            np.array: weights for global, zone and family influence
            """

        # B: Use weights from a previous run
        if self.initial_sample.weights is not None:
            initial_weights = self.initial_sample.weights

        # A: Initialize new weights
        else:
            # When the algorithm does not include inheritance then there are only 2 weights (global and contact)
            if not self.inheritance:
                initial_weights = np.full((self.n_features, 2), 1.)

            else:
                initial_weights = np.full((self.n_features, 3), 1.)

        return initial_weights

    def generate_initial_p_zones(self):
        """This function generates initial probabilities for categories in each of the zones, either by
        A) proposing them (randomly) from scratch
        B) using the last sample of a previous run of the MCMC
        Probabilities are in log-space and not normalized.

        Returns:
            np.array: probabilities for categories in each zones
                shape (n_zones, n_features, max(n_categories))
        """
        # For convenience all p_zones go in one array, even though not all features have the same number of categories
        initial_p_zones = np.zeros((self.n_zones, self.n_features, self.features.shape[2]))
        n_generated = 0

        # B: Use p_zones from a previous run
        if self.initial_sample.p_zones is not None:

            for i in range(len(self.initial_sample.p_zones)):
                initial_p_zones[i, :] = self.initial_sample.p_zones[i]
                n_generated += 1

        not_initialized = range(n_generated, self.n_zones)

        # A: Initialize new p_zones using the global MLE
        for i in not_initialized:

            l_per_cat = np.sum(self.features, axis=0)
            l_sums = np.sum(l_per_cat, axis=1)
            p_zones = l_per_cat / l_sums[:, np.newaxis]

            initial_p_zones[i, :, :] = transform_p_to_log(p_zones)

            # The probabilities of categories without data are set to 0 (or -inf in log space)
            sites_per_category = np.count_nonzero(self.features, axis=0)
            initial_p_zones[i, sites_per_category == 0] = -np.inf

        return initial_p_zones

    def generate_initial_p_families(self):
        """This function generates initial probabilities for categories in each of the families, either by
        A) using the MLE
        B) using the last sample of a previous run of the MCMC
        Probabilities are in log-space and not normalized.

        Returns:
            np.array: probabilities for categories in each family
                shape (n_families, n_features, max(n_categories))
        """
        initial_p_families = np.zeros((self.n_families, self.n_features, self.features.shape[2]))

        # B: Use p_families from a previous run
        if self.initial_sample.p_families is not None:
            for i in range(len(self.initial_sample.p_families)):
                initial_p_families[i, :] = self.initial_sample.p_families[i]

        # A: Initialize new p_families using the MLE
        for fam in range(len(self.families)):
            idx = self.families[fam].nonzero()[0]
            features_family = self.features[idx, :, :]

            # Compute the MLE for each category and each family
            # Some families have only NAs for some features, of course.
            # FInd these in the data and set equal pseudo counts (1) such that the start value is 0.5

            for feat in range(len(features_family[0])):
                idx = np.where(self.sites_per_category[feat] != 0)[0]
                if sum(features_family[0][feat][idx]) == 0:
                    features_family[0][feat][idx] = 1

            l_per_cat = np.sum(features_family, axis=0)
            l_sums = np.sum(l_per_cat, axis=1)
            p_family = l_per_cat/l_sums[:, np.newaxis]

            # If one of the p_family values is 0 balance the p_array
            for p in range(len(p_family)):

                # Check for nonzero p_idx
                p_idx = np.where(self.sites_per_category[p] != 0)[0]

                # If any of the remaining is zero -> balance
                if 0. in p_family[p, p_idx]:
                    p_family[p, p_idx] = balance_p_array(p_array=p_family[p, p_idx], balance_by=0.2)

            initial_p_families[fam, :, :] = transform_p_to_log(p_family)

            # The probabilities of categories without data are set to 0 (or -inf in log space)
            sites_per_category = np.count_nonzero(self.features, axis=0)
            initial_p_families[fam, sites_per_category == 0] = -np.inf

        return initial_p_families

    def generate_initial_sample(self):
        """Generate initial Sample object (zones, weights)

        Returns:
            Sample: The generated initial Sample
        """
        # Zones
        if self.known_initial_zones is None:
            initial_zones = self.generate_initial_zones()
        # for testing: set initial_zones to known start zones
        else:
            initial_zones = self.known_initial_zones

        # Weights
        if self.known_initial_weights is None:
            initial_weights = self.generate_initial_weights()
        # for testing: set initial_weights to  known start weights
        else:
            initial_weights = self.known_initial_weights

        # p_zones
        # p_zones can be sampled or derived from the maximum likelihood estimate
        if self.sample_p_zones:
            initial_p_zones = self.generate_initial_p_zones()
        else:
            initial_p_zones = None

        # p_families
        # p_families can be sampled or derived from the maximum likelihood estimate
        if self.sample_p_families:
            initial_p_families = self.generate_initial_p_families()
        else:
            initial_p_families = None

        sample = Sample(zones=initial_zones, weights=initial_weights,
                        p_zones=initial_p_zones, p_families=initial_p_families)

        return sample

    def get_removal_candidates(self, zone):
        """Finds sites which can be removed from the given zone. If connectedness is
        required (connected_only = True), only non-cut vertices are returned.

        Args:
            zone (np.array): The zone for which removal candidates are found.
                shape(n_sites)
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

