#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import random as _random
from copy import deepcopy

import numpy as np

from sbayes.sampling.mcmc_generative import MCMCGenerative
from sbayes.model import GenerativeLikelihood, GenerativePrior
from sbayes.util import get_neighbours, normalize, dirichlet_pdf


class IndexSet(set):

    def __init__(self, all_i=True):
        super().__init__()
        self.all = all_i

    def add(self, element):
        super(IndexSet, self).add(element)

    def clear(self):
        super(IndexSet, self).clear()
        self.all = False

    def __bool__(self):
        return self.all or (len(self) > 0)

    def __copy__(self):
        # other = deepcopy(super(IndexSet, self))
        other = IndexSet(all_i=self.all)
        for element in self:
            other.add(element)

        return other

    def __deepcopy__(self, memo):
        # other = deepcopy(super(IndexSet, self))
        other = IndexSet(all_i=self.all)
        for element in self:
            other.add(deepcopy(element))

        return other


class Sample(object):
    """
    Attributes:
        zones (np.array): Assignment of sites to zones.
            shape: (n_zones, n_sites)
        weights (np.array): Weights of zone, family and global likelihood for different features.
            shape: (n_features, 3)
        p_global (np.array): Global probabilities of categories
            shape(n_features, n_categories)
        p_zones (np.array): Probabilities of categories in zones
            shape: (n_zones, n_features, n_categories)
        p_families (np.array): Probabilities of categories in families
            shape: (n_families, n_features, n_categories)
    """

    def __init__(self, zones, weights, p_global, p_zones, p_families):
        self.zones = zones
        self.weights = weights
        self.p_global = p_global
        self.p_zones = p_zones
        self.p_families = p_families

        # The sample contains information about which of its parameters was changed in the last MCMC step
        self.what_changed = {}
        self.everything_changed()

    def everything_changed(self):
        self.what_changed = {'lh': {'zones': IndexSet(), 'weights': True,
                                    'p_global': IndexSet(), 'p_zones': IndexSet(), 'p_families': IndexSet()},
                             'prior': {'zones': IndexSet(), 'weights': True,
                                       'p_global': IndexSet(), 'p_zones': IndexSet(), 'p_families': IndexSet()}}

    def copy(self):
        zone_copied = deepcopy(self.zones)
        weights_copied = deepcopy(self.weights)
        what_changed_copied = deepcopy(self.what_changed)

        if self.p_global is not None:
            p_global_copied = self.p_global.copy()
        else:
            p_global_copied = None

        if self.p_zones is not None:
            p_zones_copied = self.p_zones.copy()
        else:
            p_zones_copied = None

        if self.p_families is not None:
            p_families_copied = self.p_families.copy()
        else:
            p_families_copied = None

        new_sample = Sample(zones=zone_copied, weights=weights_copied,
                            p_global=p_global_copied, p_zones=p_zones_copied, p_families=p_families_copied)

        new_sample.what_changed = what_changed_copied

        return new_sample


class ZoneMCMCGenerative(MCMCGenerative):

    """float: Probability at which grow operator only considers neighbours to add to the zone."""

    def __init__(self, network, features, min_size, max_size, var_proposal,
                 p_grow_connected, initial_sample, initial_size, sample_from_prior=False, **kwargs):

        super(ZoneMCMCGenerative, self).__init__(**kwargs)

        # Data
        self.features = features
        self.n_features = features.shape[1]
        try:
            # If possible, applicable states per feature are deduced from the prior
            self.applicable_states = self.prior_p_global['states']

        except KeyError:
            # Applicable states per feature are deduced from the data
            counts_per_states = np.count_nonzero(features, axis=0)
            self.applicable_states = counts_per_states > 0

        # Network
        self.network = network
        self.adj_mat = network['adj_mat']
        self.locations = network['locations']

        # Sampling
        self.n = self.adj_mat.shape[0]
        self.p_grow_connected = p_grow_connected

        # Zone size /initial sample
        self.min_size = min_size
        self.max_size = max_size
        self.initial_sample = initial_sample
        self.initial_size = initial_size

        # Families
        if self.inheritance:
            self.n_families = self.families.shape[0]

        # Variance of the proposal distribution
        self.var_proposal_weight = var_proposal['weights']
        self.var_proposal_p_global = var_proposal['universal']
        self.var_proposal_p_zones = var_proposal['contact']
        try:
            self.var_proposal_p_families = var_proposal['inheritance']
        except KeyError:
            pass

        self.sample_from_prior = sample_from_prior

        self.compute_lh_per_chain = [
            # GenerativeLikelihood(features, self.inheritance, self.families) for _ in range(self.n_chains)
            GenerativeLikelihood(data=features, families=self.families, inheritance=self.inheritance)
            for _ in range(self.n_chains)
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
        log_prior = self.compute_prior_per_chain[chain](sample=sample, inheritance=self.inheritance,
                                                        geo_prior_meta=self.geo_prior,
                                                        prior_weights_meta=self.prior_weights,
                                                        prior_p_global_meta=self.prior_p_global,
                                                        prior_p_zones_meta=self.prior_p_zones,
                                                        prior_p_families_meta=self.prior_p_families,
                                                        network=self.network)

        check_caching = False
        if check_caching:
            sample.everything_changed()
            log_prior_stable = self.compute_prior_per_chain[chain](sample=sample, inheritance=self.inheritance,
                                                                   geo_prior_meta=self.geo_prior,
                                                                   prior_weights_meta=self.prior_weights,
                                                                   prior_p_global_meta=self.prior_p_global,
                                                                   prior_p_zones_meta=self.prior_p_zones,
                                                                   prior_p_families_meta=self.prior_p_families,
                                                                   network=self.network)
            assert log_prior == log_prior_stable, f'{log_prior} != {log_prior_stable}'

        return log_prior

    def likelihood(self, sample, chain):
        """Compute the (log) likelihood of a sample.
        Args:
            sample(Sample): A Sample object consisting of zones and parameters.
            chain (int): The current chain
        Returns:
            float: The (log) likelihood of the sample"""
        if self.sample_from_prior:
            return 0.

        # Compute the likelihood
        log_lh = self.compute_lh_per_chain[chain](sample=sample)

        check_caching = False
        if check_caching:
            sample.everything_changed()
            log_lh_stable = self.compute_lh_per_chain[chain](sample=sample, caching=False)
            assert log_lh == log_lh_stable, f'{log_lh} != {log_lh_stable}'

        return log_lh

    def alter_weights(self, sample):
        """This function modifies one weight of one feature in the current sample

        Args:
            sample(Sample): The current sample with zones and parameters.
        Returns:
            Sample: The modified sample
        """
        sample_new = sample.copy()
        weights_current = sample.weights

        # Randomly choose one of the features
        f_id = np.random.choice(range(weights_current.shape[0]))

        # if not self.inheritance:
        #     # The contact weights (column 1 in weights) is modified,
        #     # the inheritance weight is not relevant, the global weight is adjusted during normalization
        #     w_id = 1
        #
        # else:
        #     # The contact or family weights (column 1 or 2 in weights) are modified,
        #     # the global weight is adjusted during normalization
        #     w_id = np.random.choice([1, 2])
        #
        # weight_current = weights_current[f_id, w_id]

        # Sample new weight from dirichlet distribution with given precision
        weights_new, q, q_back = self.dirichlet_proposal(weights_current[f_id, :], self.var_proposal_weight)
        sample_new.weights[f_id, :] = weights_new

        # The step changed the weights (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['weights'] = True
        sample_new.what_changed['prior']['weights'] = True
        sample.what_changed['lh']['weights'] = True
        sample.what_changed['prior']['weights'] = True

        return sample_new, q, q_back

    def alter_p_global(self, sample):
        """This function modifies one p_global of one category and one feature in the current sample
            Args:
                 sample(Sample): The current sample with zones and parameters.
            Returns:
                 Sample: The modified sample
        """
        sample_new = sample.copy()
        p_global_current = sample.p_global

        # Randomly choose one of the features and one of the categories
        f_id = np.random.choice(range(self.n_features))

        # Different features have different numbers of categories
        f_cats = self.applicable_states[f_id]
        p_current = p_global_current[0, f_id, f_cats]

        # Sample new p from dirichlet distribution with given precision
        p_new, q, q_back = self.dirichlet_proposal(p_current, step_precision=self.var_proposal_p_global)
        sample_new.p_global[0, f_id, f_cats] = p_new

        # The step changed p_global (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['p_global'].add(f_id)
        sample_new.what_changed['prior']['p_global'].add(f_id)
        sample.what_changed['lh']['p_global'].add(f_id)
        sample.what_changed['prior']['p_global'].add(f_id)

        return sample_new, q, q_back

    def alter_p_zones(self, sample):
        """This function modifies one p_zones of one category, one feature and in zone in the current sample
            Args:
                sample(Sample): The current sample with zones and parameters.
            Returns:
                Sample: The modified sample
                """
        sample_new = sample.copy()
        p_zones_current = sample.p_zones

        # Randomly choose one of the zones, one of the features and one of the categories
        z_id = np.random.choice(range(self.n_zones))
        f_id = np.random.choice(range(self.n_features))

        # Different features have different numbers of categories
        f_cats = self.applicable_states[f_id]
        p_current = p_zones_current[z_id, f_id, f_cats]

        # Sample new p from dirichlet distribution with given precision
        p_new, q, q_back = self.dirichlet_proposal(p_current, step_precision=self.var_proposal_p_zones)
        sample_new.p_zones[z_id, f_id, f_cats] = p_new

        # The step changed p_zones (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['p_zones'].add((z_id, f_id))
        sample_new.what_changed['prior']['p_zones'].add((z_id, f_id))
        sample.what_changed['lh']['p_zones'].add((z_id, f_id))
        sample.what_changed['prior']['p_zones'].add((z_id, f_id))

        return sample_new, q, q_back

    @staticmethod
    def dirichlet_proposal(w, step_precision):
        """ A proposal distribution for normalized weight and probability vectors (summing to 1).

        Args:
            w (np.array): The weight vector, which is being resampled.
                Shape: (n_categories, )
            step_precision (float): The precision parameter controlling how narrow/wide the proposal
                distribution is. Low precision -> wide, high precision -> narrow.

        Returns:
            np.array: The newly proposed weights w_new (same shape as w).
            float: The transition probability q.
            float: The back probability q_back
        """
        # assert np.allclose(np.sum(w, axis=-1), 1.), w

        alpha = 1 + step_precision * w
        w_new = np.random.dirichlet(alpha)
        q = dirichlet_pdf(w_new, alpha)

        alpha_back = 1 + step_precision * w_new
        q_back = dirichlet_pdf(w, alpha_back)

        if not np.all(np.isfinite(w_new)):
            logging.warning(f'Dirichlet step resulted in NaN or Inf:')
            logging.warning(f'\tOld sample: {w}')
            logging.warning(f'\tstep_precision: {step_precision}')
            logging.warning(f'\tNew sample: {w_new}')
            # return w, 1., 0.

        return w_new, q, q_back

    def alter_p_families(self, sample):
        """This function modifies one p_families of one category, one feature and one family in the current sample
            Args:
                 sample(Sample): The current sample with zones and parameters.
            Returns:
                 Sample: The modified sample
        """

        sample_new = sample.copy()
        p_families_current = sample.p_families

        # Randomly choose one of the families, one of the features and one of the categories
        fam_id = np.random.choice(range(self.n_families))
        f_id = np.random.choice(range(self.n_features))

        # Different features have different numbers of categories
        f_cats = self.applicable_states[f_id]
        p_current = p_families_current[fam_id, f_id, f_cats]

        # Sample new p from dirichlet distribution with given precision
        p_new, q, q_back = self.dirichlet_proposal(p_current, step_precision=self.var_proposal_p_families)

        sample_new.p_families[fam_id, f_id, f_cats] = p_new

        # The step changed p_families (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['p_families'].add((fam_id, f_id))
        sample_new.what_changed['prior']['p_families'].add((fam_id, f_id))
        sample.what_changed['lh']['p_families'].add((fam_id, f_id))
        sample.what_changed['prior']['p_families'].add((fam_id, f_id))

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

        neighbours = get_neighbours(zone_current, occupied, self.adj_mat)
        connected_step = (_random.random() < self.p_grow_connected)
        if connected_step:
            # All neighbors that are not yet occupied by other zones are candidates
            candidates = neighbours
        else:
            # All free sites are candidates
            candidates = ~occupied

        # When stuck (all neighbors occupied) return current sample and reject the step (q_back = 0)
        if not np.any(candidates):
            q, q_back = 1., 0.
            return sample, q, q_back

        # Add a site to the zone
        site_new = _random.choice(candidates.nonzero()[0])
        sample_new.zones[z_id, site_new] = 1

        # Remove a site from the zone
        removal_candidates = self.get_removal_candidates(zone_current)
        site_removed = _random.choice(removal_candidates)
        sample_new.zones[z_id, site_removed] = 0

        # # Compute transition probabilities
        back_neighbours = get_neighbours(zone_current, occupied, self.adj_mat)
        # q = 1. / np.count_nonzero(candidates)
        # q_back = 1. / np.count_nonzero(back_neighbours)

        # Transition probability growing to the new zone
        q_non_connected = 1 / np.count_nonzero(~occupied)
        q = (1 - self.p_grow_connected) * q_non_connected
        if neighbours[site_new]:
            q_connected = 1 / np.count_nonzero(neighbours)
            q += self.p_grow_connected * q_connected

        # Transition probability of growing back to the original zone
        q_back_non_connected = 1 / np.count_nonzero(~occupied)
        q_back = (1 - self.p_grow_connected) * q_back_non_connected
        # If z is a neighbour of the new zone, the back step could also be a connected grow step
        if back_neighbours[site_removed]:
            q_back_connected = 1 / np.count_nonzero(back_neighbours)
            q_back += self.p_grow_connected * q_back_connected

        # The step changed the zone (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['zones'].add(z_id)
        sample.what_changed['lh']['zones'].add(z_id)
        sample_new.what_changed['prior']['zones'].add(z_id)
        sample.what_changed['prior']['zones'].add(z_id)

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

        if current_size >= self.max_size:
            # Zone too big to grow: don't modify the sample and reject the step (q_back = 0)
            q, q_back = 1., 0.
            return sample, q, q_back

        neighbours = get_neighbours(zone_current, occupied, self.adj_mat)
        connected_step = (_random.random() < self.p_grow_connected)
        if connected_step:
            # All neighbors that are not yet occupied by other zones are candidates
            candidates = neighbours
        else:
            # All free sites are candidates
            candidates = ~occupied

        # When stuck (no candidates) return current sample and reject the step (q_back = 0)
        if not np.any(candidates):
            q, q_back = 1., 0.
            return sample, q, q_back

        # Choose a random candidate and add it to the zone
        site_new = _random.choice(candidates.nonzero()[0])
        sample_new.zones[z_id, site_new] = 1

        # Transition probability when growing
        q_non_connected = 1 / np.count_nonzero(~occupied)
        q = (1 - self.p_grow_connected) * q_non_connected
        if neighbours[site_new]:
            q_connected = 1 / np.count_nonzero(neighbours)
            q += self.p_grow_connected * q_connected

        # Back-probability (shrinking)
        q_back = 1 / (current_size + 1)

        # q = q_back = 1.

        # The step changed the zone (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['zones'].add(z_id)
        sample.what_changed['lh']['zones'].add(z_id)
        sample_new.what_changed['prior']['zones'].add(z_id)
        sample.what_changed['prior']['zones'].add(z_id)

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

        # Transition probability when shrinking.
        q = 1 / len(removal_candidates)
        # Back-probability (growing)
        zone_new = sample_new.zones[z_id]
        occupied_new = np.any(sample_new.zones, axis=0)
        back_neighbours = get_neighbours(zone_new, occupied_new, self.adj_mat)

        # The back step could always be a non-connected grow step
        q_back_non_connected = 1 / np.count_nonzero(~occupied_new)
        q_back = (1 - self.p_grow_connected) * q_back_non_connected

        # If z is a neighbour of the new zone, the back step could also be a connected grow step
        if back_neighbours[site_removed]:
            q_back_connected = 1 / np.count_nonzero(back_neighbours)
            q_back += self.p_grow_connected * q_back_connected

        # q = q_back = 1.

        # The step changed the zone (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['zones'].add(z_id)
        sample.what_changed['lh']['zones'].add(z_id)
        sample_new.what_changed['prior']['zones'].add(z_id)
        sample.what_changed['prior']['zones'].add(z_id)

        return sample_new, q, q_back

    def generate_initial_zones(self):
        """For each chain (c) generate initial zones by
        A) growing through random grow-steps up to self.min_size,
        B) using the last sample of a previous run of the MCMC

        Returns:
            np.array: The generated initial zones.
                shape(n_zones, n_sites)
        """

        # If there are no zones, return empty matrix
        if self.n_zones == 0:
            return np.zeros((self.n_zones, self.n), bool)

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

        # A: The areas that are not initialized yet are grown
        # When there are already many areas, new ones can get stuck due to an unfavourable seed.
        # That's why we perform several attempts to initialize areas
        attempts = 0
        max_attempts = 1000

        while True:
            for i in not_initialized:
                try:
                    initial_size = self.initial_size
                    g = self.grow_zone_of_size_k(initial_size, occupied)

                except self.ZoneError:
                    # Might be due to an unfavourable seed

                    if attempts < max_attempts:
                        attempts += 1
                        not_initialized = range(n_generated, self.n_zones)
                        break
                    # Seems there is not enough sites to grow n_zones of size k
                    else:
                        raise ValueError("Failed to add additional area. Try fewer areas"
                                         "or set initial_sample to None")
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

        return normalize(initial_weights)

    def generate_initial_p_global(self):
        """This function generates initial global probabilities for each category either by
        A) using the MLE
        B) using the last sample of a previous run of the MCMC
        Probabilities are in log-space and not normalized.

        Returns:
            np.array: probabilities for categories in each family
                shape (1, n_features, max(n_categories))
        """
        initial_p_global = np.zeros((1, self.n_features, self.features.shape[2]))

        # B: Use p_global from a previous run
        if self.initial_sample.p_global is not None:
            initial_p_global = self.initial_sample.p_global

        # A: Initialize new p_global using the MLE
        else:

            sites_per_state = np.count_nonzero(self.features, axis=0)
            # Some areas have nan for all states, resulting in a non-defined MLE
            # other areas have only a single state, resulting in an MLE including 1.
            # to avoid both, we add 1 to all applicable states of each feature,
            # which gives a well-defined initial p_zone without 1., slightly nudged away from the MLE

            sites_per_state[np.isnan(sites_per_state)] = 0
            sites_per_state[self.applicable_states] += 1
            site_sums = np.sum(sites_per_state, axis=1)
            p_global = sites_per_state / site_sums[:, np.newaxis]

            initial_p_global[0, :, :] = p_global
        return initial_p_global

    def set_p_zones_to_mle(self, updated_zone):
        """This function sets the p_zones to the MLE of the current zone
        Probabilities are in log-space and not normalized.
        Args:
            updated_zone (np.array): The currently updated zone
            (n_sites)
        Returns:
            np.array: probabilities for categories in each zones
                shape (n_zones, n_features, max(n_categories))
        """

        idx = updated_zone.nonzero()[0]
        features_zone = self.features[idx, :, :]
        l_per_cat = np.sum(features_zone, axis=0)
        p_zones = normalize(l_per_cat)

        return p_zones

    def generate_initial_p_zones(self, initial_zones):
        """This function generates initial probabilities for categories in each of the zones, either by
        A) proposing them (randomly) from scratch
        B) using the last sample of a previous run of the MCMC
        Probabilities are in log-space and not normalized.
        Args:
            initial_zones: The assignment of sites to zones
            (n_zones, n_sites)
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

        # A: Initialize new p_zones using a value close to the MLE of the current zone
        for i in not_initialized:
            idx = initial_zones[i].nonzero()[0]
            features_zone = self.features[idx, :, :]

            sites_per_state = np.nansum(features_zone, axis=0)

            # Some areas have nan for all states, resulting in a non-defined MLE
            # other areas have only a single state, resulting in an MLE including 1.
            # to avoid both, we add 1 to all applicable states of each feature,
            # which gives a well-defined initial p_zone without 1., slightly nudged away from the MLE

            sites_per_state[np.isnan(sites_per_state)] = 0
            sites_per_state[self.applicable_states] += 1

            site_sums = np.sum(sites_per_state, axis=1)
            p_zones = sites_per_state / site_sums[:, np.newaxis]

            initial_p_zones[i, :, :] = p_zones

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
        else:

            for fam in range(len(self.families)):
                idx = self.families[fam].nonzero()[0]
                features_family = self.features[idx, :, :]

                sites_per_state = np.nansum(features_family, axis=0)

                # Compute the MLE for each category and each family
                # Some families have only NAs for some features, resulting in a non-defined MLE
                # other families have only a single state, resulting in an MLE including 1.
                # to avoid both, we add 1 to all applicable states of each feature,
                # which gives a well-defined initial p_family without 1., slightly nudged away from the MLE

                sites_per_state[np.isnan(sites_per_state)] = 0
                sites_per_state[self.applicable_states] += 1

                state_sums = np.sum(sites_per_state, axis=1)
                p_family = sites_per_state / state_sums[:, np.newaxis]
                initial_p_families[fam, :, :] = p_family

        return initial_p_families

    def generate_initial_sample(self):
        """Generate initial Sample object (zones, weights)

        Returns:
            Sample: The generated initial Sample
        """
        # Zones
        initial_zones = self.generate_initial_zones()

        # Weights
        initial_weights = self.generate_initial_weights()

        # p_global (alpha)

        initial_p_global = self.generate_initial_p_global()

        # p_zones (gamma)
        initial_p_zones = self.generate_initial_p_zones(initial_zones)

        # p_families (beta)
        if self.inheritance:
            initial_p_families = self.generate_initial_p_families()
        else:
            initial_p_families = None

        sample = Sample(zones=initial_zones, weights=initial_weights,
                        p_global=initial_p_global, p_zones=initial_p_zones, p_families=initial_p_families)

        return sample

    @staticmethod
    def get_removal_candidates(zone):
        """Finds sites which can be removed from the given zone.

        Args:
            zone (np.array): The zone for which removal candidates are found.
                shape(n_sites)
        Returns:
            (list): Index-list of removal candidates.
        """
        return zone.nonzero()[0]

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

    def log_sample_statistics(self, sample, c, sample_id):
        super(ZoneMCMCGenerative, self).log_sample_statistics(sample, c, sample_id)
