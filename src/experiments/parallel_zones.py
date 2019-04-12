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
                               simulate_background_distribution,
                               simulate_contact, compute_feature_prob,
                               precompute_feature_likelihood,
                               define_contact_features,
                               estimate_random_walk_covariance)
from src.sampling.zone_sampling_particularity import ZoneMCMC


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


TEST_SAMPLING_DIRECTORY = 'data/results/test/parallel/{experiment}/'.format(experiment=now)
TEST_SAMPLING_RESULTS_PATH = TEST_SAMPLING_DIRECTORY + 'parallel_z{z}_e{e}_pz{pz}_{run}.pkl'
TEST_SAMPLING_LOG_PATH = TEST_SAMPLING_DIRECTORY + 'parallel.log'


# Make directory if it doesn't exist yet
if not os.path.exists(TEST_SAMPLING_DIRECTORY):
    os.mkdir(TEST_SAMPLING_DIRECTORY)


logging.basicConfig(filename=TEST_SAMPLING_LOG_PATH, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())


################################
# Simulation
################################
# l_z = (7, 5, 2, 3, 4) # large zones
# s_z = (1, 6, 8, 3, 9) # small zones

# Zones
z_1 = 2                 # one large zone
z_2 = (7, 5)            # two large zones
z_3 = (4, 3, 2)         # three large zones
z_4 = (4, 5, 7, 3)      # four large zones

z_5 = (4, 1)            # one small one large zone
z_6 = (4, 7, 8, 6)      # two small two large zones

z_7 = 6                 # one small zone
z_8 = (8, 3)            # two small zones
z_9 = (9, 1, 6)         # three small zones
z_10 = (8, 3, 6, 1)     # four small zones

zones = [z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8, z_9, z_10]
test_zone = range(0, len(zones))

# Intensity: proportion of sites, which are indicative of contact
i = [0.9] # [0.4, 0.7, 0.9]

# Features: proportion of features affected by contact
f = [0.9] #[0.4, 0.7, 0.9]

test_ease = range(0, len(f))

# [0] Hard: Unfavourable zones with low intensity and few features affected by contact
# [2] Easy: Favourable zones  with high intensity and many features affected by contact

# Feature probabilities
TOTAL_N_FEATURES = 30
P_SUCCESS_MIN = 0.05
P_SUCCESS_MAX = 0.95

# Geo-prior
GEO_PRIOR_NR_SAMPLES = 1000

#################################
# MCMC
#################################

# General
BURN_IN = 0
N_STEPS = 15000
N_SAMPLES = 1000
N_RUNS = 1


# Zone sampling
MIN_SIZE = 5
MAX_SIZE = 200
START_SIZE = 25
CONNECTED_ONLY = False

P_TRANSITION_MODE = {
    'swap': 0.5,
    'grow': 0.75,
    'shrink': 1.}

# Steepness of the likelihood function
LH_WEIGHT = 1

# Markov chain coupled MC (mc3)
N_CHAINS = 8                   # Number of independent chains
SWAP_PERIOD = 200
N_SWAPS = 1                   # Attempted inter-chain swaps after each SWAP_PERIOD

# At the moment not tested and set to default
MODEL = 'particularity'

# Parallel zones
N_ZONES = 6


sampling_param_grid = list(itertools.product(test_ease, test_zone))
print(sampling_param_grid)


def evaluate_sampling_parameters(params):
    e, z = params

    # Retrieve the network from the DB
    network = get_network(reevaluate=True)
    # Generate an empirical distribution for estimating the geo-likelihood
    ecdf_geo = generate_ecdf_geo_prior(net=network, min_n=MIN_SIZE, max_n=MAX_SIZE,
                                       nr_samples=GEO_PRIOR_NR_SAMPLES,
                                       reevaluate=False)

    stats = []
    samples = []

    contact_zones_idxs = get_contact_zones(zones[test_zone[z]])
    n_zones = len(contact_zones_idxs)
    contact_zones = np.zeros((n_zones, network['n']), bool)

    for k, cz_idxs in enumerate(contact_zones_idxs.values()):
        contact_zones[k, cz_idxs] = True

    for run in range(N_RUNS):

        # Initially, initial_zones for all chains and all zones are set to None
        initial_zones = [[None] * N_CHAINS] * N_ZONES

        # Simulation
        f_names, contact_f_names = define_contact_features(n_feat=TOTAL_N_FEATURES, r_contact_feat=f[e],
                                                           contact_zones=contact_zones_idxs)

        features_bg = simulate_background_distribution(features=f_names,
                                                       n_sites=len(network['vertices']),
                                                       contact_features=contact_f_names,
                                                       p_min=P_SUCCESS_MIN, p_max=P_SUCCESS_MAX,
                                                       p_max_contact=i[e],
                                                       reevaluate=True)

        features = simulate_contact(features=features_bg, contact_features=contact_f_names,
                                    p=i[e], contact_zones=contact_zones_idxs, reevaluate=True)

        feature_prob = compute_feature_prob(features, reevaluate=True)
        lh_lookup = precompute_feature_likelihood(MIN_SIZE, MAX_SIZE, feature_prob,
                                                  reevaluate=True)

        for N in range(N_ZONES):

            # Sampling
            zone_sampler = ZoneMCMC(network=network, features=features,
                                    min_size=MIN_SIZE, max_size=MAX_SIZE, start_size=START_SIZE,
                                    p_transition_mode=P_TRANSITION_MODE,  connected_only=CONNECTED_ONLY,
                                    feature_ll_mode=MODEL, geo_prior_mode=MODEL,
                                    lh_lookup=lh_lookup, lh_weight=LH_WEIGHT, ecdf_geo=ecdf_geo, ecdf_type="mst",
                                    n_chains=N_CHAINS, n_zones=N+1,
                                    swap_period=SWAP_PERIOD, initial_zones=initial_zones,
                                    chain_swaps=N_SWAPS, print_logs=False)

            zone_sampler.generate_samples(N_STEPS, N_SAMPLES, BURN_IN, return_steps=False)

            # Collect statistics
            run_stats = zone_sampler.statistics
            run_stats['true_zones_lls'] = [zone_sampler.log_likelihood(cz) for cz in contact_zones]
            run_stats['true_zones_priors'] = [zone_sampler.prior_zone(cz) for cz in contact_zones]
            run_stats['true_zones'] = contact_zones
            stats.append(run_stats)

            # Define initial zone for next run
            for iz in range(len(run_stats['last_sample'])):
                print('Number of zones', iz+1)
                print('Number of chains:', len(run_stats['last_sample'][iz]))
                initial_zones[iz] = run_stats['last_sample'][iz]

            # Store the results
            path = TEST_SAMPLING_RESULTS_PATH.format(z=z+1, e=2, pz=N+1, run=run)
            dump(run_stats, path)

    return samples, stats


if __name__ == '__main__':

    # Test ease
    with MyPool(4) as pool:
        all_stats = pool.map(evaluate_sampling_parameters, sampling_param_grid)



