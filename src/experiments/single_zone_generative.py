#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import logging
import itertools
import datetime
import multiprocessing.pool

import numpy as np

from src.util import dump, load_from
from src.preprocessing import (get_network,
                               generate_ecdf_geo_prior,
                               get_contact_zones,
                               sample_from_likelihood,
                               simulate_contact, assign_na)
from src.sampling.zone_sampling_generative import ZoneMCMC_generative


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

now = datetime.datetime.now().__str__().rsplit('.')[0]
now = now.replace(':', '-')
now = now.replace(' ', '_')


TEST_SAMPLING_DIRECTORY = 'data/results/test/zones/{experiment}/'.format(experiment=now)
TEST_SAMPLING_RESULTS_PATH = TEST_SAMPLING_DIRECTORY + 'zone_z{z}_e{e}_{run}.pkl'
TEST_SAMPLING_LOG_PATH = TEST_SAMPLING_DIRECTORY + 'zone.log'


# Make directory if it doesn't exist yet
if not os.path.exists(TEST_SAMPLING_DIRECTORY):
    os.mkdir(TEST_SAMPLING_DIRECTORY)


logging.basicConfig(filename=TEST_SAMPLING_LOG_PATH, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())


################################
# Simulation
################################

# Zones
test_zone = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Intensity of contact, passed as alpha when drawing samples from dirichlet distribution
# lower i's result in zones with more features belonging to the same category
i = [1.5, 1.25, 1, 0.75, 0.5, 0.25]

# Number of contact features, passed as alpha when drawing samples from dirichlet distribution
# higher f's correspond to more features for which the influence of contact is strong
f = [1.25, 1.5, 1.75, 2, 2.25, 2.5]

test_ease = range(0, len(f))

# [0] Hard: Unfavourable zones with low intensity and few features affected by contact
# [5] Easy: Favourable zones  with high intensity and many features affected by contact

# Feature probabilities
TOTAL_N_FEATURES = 30
P_SUCCESS_MIN = 0.2
P_SUCCESS_MAX = 0.8


#################################
# MCMC
#################################

# General
BURN_IN = 0
N_STEPS = 50000
N_SAMPLES = 1000
N_RUNS = 1


# Zone sampling
MIN_SIZE = 5
MAX_SIZE = 200
INITIAL_SIZE = 25
CONNECTED_ONLY = False
OPERATORS = {'shrink_zone': 0.05,
             'grow_zone': 0.05,
             'swap_zone': 0.05,
             'alter_weights': 0.85}

# Markov chain coupled MC (mc3)
N_CHAINS = 5     # Number of independent chains
SWAP_PERIOD = 200
N_SWAPS = 1      # Attempted inter-chain swaps after each SWAP_PERIOD

# Number of zones
N_ZONES = 1
# Todo background in LH?
BACKGROUND = True

# Geo-prior (none, distance or gaussian)
GEO_PRIOR = 'none'


sampling_param_grid = list(itertools.product(test_ease, test_zone))
print(sampling_param_grid)


def evaluate_sampling_parameters(params):
    e, z = params

    # Retrieve the network from the DB
    network = get_network(reevaluate=True)

    stats = []
    samples = []

    contact_zones_idxs = get_contact_zones(z)
    n_zones = len(contact_zones_idxs)
    contact_zones = np.zeros((n_zones, network['n']), bool)

    for k, cz_idxs in enumerate(contact_zones_idxs.values()):
        contact_zones[k, cz_idxs] = True

    for run in range(N_RUNS):
        # todo alpha and np.random.dirichlet([1.0] control
        # Single zones, hence there are no initial zones
        initial_zones = [[None] * N_CHAINS] * N_ZONES

        # Simulate features
        n_features = 30
        max_categories = 4
        n_families = 2

        p_global = np.zeros((n_features, max_categories), dtype=float)
        p_zones = np.zeros((n_zones, n_features, max_categories), dtype=float)
        p_families = np.zeros((n_families, n_features, max_categories), dtype=float)

        # Most features are binary (70%), some have three and four categories (20 and 10%)
        p_2, p_3, p_4 = 0.7, 0.2, 0.1

        for feat in range(n_features):
            n_categories = np.random.choice(a=[2, 3, 4], size=1, p=[p_2, p_3, p_4])[0]

            # Todo Change to intensity value for family
            # Sample zone probabilities
            alpha_p_zones = [i[e]] * n_categories + [0] * (max_categories - n_categories)
            for z in range(n_zones):
                p_zones[z, feat, :] = np.random.dirichlet(alpha_p_zones, size=1)

            # Sample global probabilities
            alpha_p_global = [1.5] * n_categories + [0] * (max_categories - n_categories)
            p_global[feat, :] = np.random.dirichlet(alpha_p_global, size=1)

            # Sample family probabilities
            alpha_p_families = [1] * n_categories + [0] * (max_categories - n_categories)
            for fam in range(n_families):
                p_families[fam, feat, :] = np.random.dirichlet(alpha_p_families, size=1)

        # Random family sites
        n_sites = network['n']
        q_families = np.random.dirichlet([1.0] * n_families, size=n_sites)
        families = (q_families == np.max(q_families, axis=-1)[:, None])

        # We sample weights that control the influence of (zone, global and family) probabilities when sampling features
        alpha = [f[e], 1.0, 0.0001]
        weights = np.random.dirichlet(alpha, n_features)

        # Sample features
        # Todo: There is a bias for category 1! Remove!
        features, features_old = sample_from_likelihood(zones=np.transpose(contact_zones), families=families,
                                                        p_zones=p_zones, p_global=p_global, p_families=p_families,
                                                        weights=weights, cat_axis=True)

        # Todo: Fix!
        # Assign some features to NA (makes testing more realistic)
        #features = assign_na(features=features, n_na=20)

        zone_sampler = ZoneMCMC_generative(network=network, features=features,
                                           min_size=MIN_SIZE, max_size=MAX_SIZE, initial_size=INITIAL_SIZE,
                                           n_zones=N_ZONES, connected_only=CONNECTED_ONLY, background=BACKGROUND,
                                           n_chains=N_CHAINS, zones_previous_run=initial_zones,
                                           swap_period=SWAP_PERIOD, operators=OPERATORS,
                                           chain_swaps=N_SWAPS)

        zone_sampler.generate_samples(N_STEPS, N_SAMPLES, BURN_IN)

        # Collect statistics
        run_stats = zone_sampler.statistics
        run_stats['true_zones_lls'] = [zone_sampler.log_likelihood(cz) for cz in contact_zones]
        run_stats['true_zones_priors'] = [zone_sampler.prior_zone(cz) for cz in contact_zones]
        run_stats['true_zones'] = contact_zones

        stats.append(run_stats)

        # Store the results
        path = TEST_SAMPLING_RESULTS_PATH.format(z=z, e=e, run=run)
        dump(run_stats, path)

    return samples, stats


if __name__ == '__main__':

    # Test ease
    with MyPool(1) as pool:
        all_stats = pool.map(evaluate_sampling_parameters, sampling_param_grid)

