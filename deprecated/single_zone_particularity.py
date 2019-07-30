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
from src.sampling.zone_sampling_particularity import ZoneMCMC_particularity


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

# Intensity: proportion of sites, which are indicative of contact
i = [0.5, 0.6, 0.7, 0.8, 0.9]

# Features: proportion of features affected by contact
f = [0.5, 0.6, 0.7, 0.8, 0.9]

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
START_SIZE = 25
CONNECTED_ONLY = False

P_TRANSITION_MODE = {
    'swap': 0.5,
    'grow': 0.75,
    'shrink': 1.}

# Markov chain coupled MC (mc3)
N_CHAINS = 6                  # Number of independent chains
SWAP_PERIOD = 200
N_SWAPS = 1   # Attempted inter-chain swaps after each SWAP_PERIOD

# Steepness of the likelihood function (for particularity model only)
LH_WEIGHT = 1
N_ZONES = 1

# Geo-prior (distance or gaussian)
GEO_PRIOR = 'none'
# If 'distance' set number of samples to generate empirical distribution
GEO_PRIOR_NR_SAMPLES = 1000


sampling_param_grid = list(itertools.product(test_ease, test_zone))
print(sampling_param_grid)


def evaluate_sampling_parameters(params):
    e, z = params

    # Retrieve the network from the DB
    network = get_network(reevaluate=True)

    # Generate an empirical distribution for estimating the geo-likelihood
    if GEO_PRIOR == 'distance':

        ecdf_geo = generate_ecdf_geo_prior(net=network, min_n=MIN_SIZE, max_n=MAX_SIZE,
                                           nr_samples=GEO_PRIOR_NR_SAMPLES,
                                           reevaluate=False)
    else:
        ecdf_geo = 0

    stats = []
    samples = []

    contact_zones_idxs = get_contact_zones(z)
    n_zones = len(contact_zones_idxs)
    contact_zones = np.zeros((n_zones, network['n']), bool)

    for k, cz_idxs in enumerate(contact_zones_idxs.values()):
        contact_zones[k, cz_idxs] = True

    for run in range(N_RUNS):

        # Single zones, hence there are no initial zones
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
        lh_lookup = precompute_feature_likelihood(MIN_SIZE, MAX_SIZE, feature_prob, log_surprise=False, reevaluate=True)

        # Sampling
        zone_sampler = ZoneMCMC_particularity(network=network, features=features,
                                              min_size=MIN_SIZE, max_size=MAX_SIZE, start_size=START_SIZE,
                                              p_transition_mode=P_TRANSITION_MODE, n_zones=N_ZONES,
                                              connected_only=CONNECTED_ONLY, geo_prior_mode=GEO_PRIOR,
                                              lh_lookup=lh_lookup,
                                              lh_weight=LH_WEIGHT, ecdf_geo=ecdf_geo, ecdf_type="mst",
                                              n_chains=N_CHAINS, initial_zones=initial_zones,
                                              swap_period=SWAP_PERIOD,
                                              chain_swaps=N_SWAPS, print_logs=False)

        zone_sampler.generate_samples(N_STEPS, N_SAMPLES, BURN_IN, return_steps=False)

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
    with MyPool(4) as pool:
        all_stats = pool.map(evaluate_sampling_parameters, sampling_param_grid)

