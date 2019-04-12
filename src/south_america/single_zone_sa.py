#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import logging
import itertools
import datetime
import multiprocessing.pool
import geopandas as gpd

import numpy as np

from src.util import dump, load_from
from src.preprocessing import (get_network,
                               generate_ecdf_geo_prior,
                               get_features, compute_feature_prob,
                               precompute_feature_likelihood)



from src.sampling.zone_sampling_particularity import ZoneMCMC


now = datetime.datetime.now().__str__().rsplit('.')[0]
now = now.replace(':', '-')
now = now.replace(' ', '_')


TEST_SAMPLING_DIRECTORY = 'src/south_america/{experiment}/'.format(experiment=now)
TEST_SAMPLING_RESULTS_PATH = TEST_SAMPLING_DIRECTORY + 'single_zone_run_{run}.pkl'
TEST_SAMPLING_LOG_PATH = TEST_SAMPLING_DIRECTORY + 'zone.log'


# Make directory if it doesn't exist yet
if not os.path.exists(TEST_SAMPLING_DIRECTORY):
    os.mkdir(TEST_SAMPLING_DIRECTORY)


logging.basicConfig(filename=TEST_SAMPLING_LOG_PATH, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())



#################################
# MCMC
#################################

# General
BURN_IN = 10000
N_STEPS = 100000
N_SAMPLES = 5000
N_RUNS = 5


# Zone sampling
MIN_SIZE = 5
MAX_SIZE = 80
START_SIZE = 15
CONNECTED_ONLY = False

P_TRANSITION_MODE = {
    'swap': 0.5,
    'grow': 0.75,
    'shrink': 1.}

# PARAMETERS
# Markov chain coupled MC (mc3)
N_CHAINS = 4                  # Number of independent chains
SWAP_PERIOD = 200
N_SWAPS = 1   # Attempted inter-chain swaps after each SWAP_PERIOD

# Model
MODEL = 'particularity'
# Steepness of the likelihood function (for particularity model only)
LH_WEIGHT = 1
N_ZONES = 1

# Geo-prior (distance or gaussian)
GEO_PRIOR = 'distance'
# If 'distance' set number of samples to generate empirical distribution
GEO_PRIOR_NR_SAMPLES = 1000

# Get Data
TABLE = 'sbayes_south_america.languages'

# Store data

NETWORK_PATH = 'src/south_america/network.pkl'
FEATURE_PATH = 'src/south_america/features.pkl'
FEATURE_PROB_PATH = 'src/south_america/feature_prob.pkl'
ECDF_GEO_PATH = 'src/south_america/ecdf_geo.pkl'
LOOKUP_PATH = 'src/south_america/lookup_table.pkl'
FEATURE_NAMES = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10',
                 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23']

if __name__ == '__main__':

    network = get_network(reevaluate=True, table=TABLE, path=NETWORK_PATH)
    # Generate an empirical distribution for estimating the geo-likelihood
    if GEO_PRIOR == 'distance':

        ecdf_geo = generate_ecdf_geo_prior(net=network, min_n=MIN_SIZE, max_n=MAX_SIZE,
                                           nr_samples=GEO_PRIOR_NR_SAMPLES, path=ECDF_GEO_PATH,
                                           reevaluate=False)
    else:
        ecdf_geo = 0

    stats = []
    samples = []
    features = get_features(table=TABLE, feature_names=FEATURE_NAMES, reevaluate=True, path=FEATURE_PATH)

    feature_prob = compute_feature_prob(features, reevaluate=True, path=FEATURE_PROB_PATH)

    if MODEL == "particularity":
        lh_lookup = precompute_feature_likelihood(MIN_SIZE, MAX_SIZE, feature_prob, log_surprise=False,
                                                  reevaluate=True, path=LOOKUP_PATH)
    else:
        lh_lookup = 0

    for run in range(N_RUNS):

        # Single zones, hence there are no initial zones
        initial_zones = [[None] * N_CHAINS] * N_ZONES

        # Sampling
        zone_sampler = ZoneMCMC(network=network, features=features,
                                min_size=MIN_SIZE, max_size=MAX_SIZE, start_size=START_SIZE,
                                p_transition_mode=P_TRANSITION_MODE, n_zones=N_ZONES,
                                connected_only=CONNECTED_ONLY,
                                feature_ll_mode=MODEL, geo_prior_mode=GEO_PRIOR,
                                lh_lookup=lh_lookup,
                                lh_weight=LH_WEIGHT, ecdf_geo=ecdf_geo, ecdf_type="mst",
                                n_chains=N_CHAINS, initial_zones=initial_zones,
                                swap_period=SWAP_PERIOD,
                                chain_swaps=N_SWAPS, print_logs=False)

        zone_sampler.generate_samples(N_STEPS, N_SAMPLES, BURN_IN, return_steps=False)

        # Collect statistics
        run_stats = zone_sampler.statistics
        stats.append(run_stats)

        # Store the results
        path = TEST_SAMPLING_RESULTS_PATH.format(run=run)
        dump(run_stats, path)

