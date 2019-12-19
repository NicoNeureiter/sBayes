#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import random
import logging
import itertools

import multiprocessing.pool
import numpy as np
import os

from src.util import (dump, set_experiment_name, read_features_from_csv)
from src.preprocessing import (get_p_global_prior, compute_network,
                               estimate_geo_prior_parameters)
from src.sampling.zone_sampling import ZoneMCMC_generative, Sample
from src.postprocessing import contribution_per_zone


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


experiment_name = set_experiment_name()

# File location
TEST_SAMPLING_DIRECTORY = 'results/shared_evolution/geo_prior/{experiment}/'.format(experiment=experiment_name)
TEST_SAMPLING_RESULTS_PATH = TEST_SAMPLING_DIRECTORY + 'bk_geo_prior_nz{nz}_{run}.pkl'

if not os.path.exists(TEST_SAMPLING_DIRECTORY):
    os.mkdir(TEST_SAMPLING_DIRECTORY)

# Logging
TEST_SAMPLING_LOG_PATH = TEST_SAMPLING_DIRECTORY + 'info.log'
logging.basicConfig(filename=TEST_SAMPLING_LOG_PATH, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("Experiment: %s", experiment_name)

#################################
# MCMC
#################################

# General
# In burn-in there are no weights-steps (find zones first, find weights later)
BURN_IN = 0
N_STEPS = 200000
N_SAMPLES = 1000
N_RUNS = 1
logging.info("MCMC with %s steps and %s samples (burn-in %s steps)", N_STEPS, N_SAMPLES, BURN_IN)

# Zone sampling
MIN_SIZE = 3
MAX_SIZE = 50
INITIAL_SIZE = 3
CONNECTED_ONLY = False
logging.info("Zones have a minimum size of %s and a maximum size of %s. The initial size is %s.",
             MIN_SIZE, MAX_SIZE, INITIAL_SIZE)

# Include inheritance (language families) in the model?
INHERITANCE = False
logging.info("Inheritance is considered: %s", INHERITANCE)

# Sample the probabilities for categories in zones and families? If False the maximum likelihood estimate is used.
SAMPLE_P = {"p_global": True,
            "p_zones":  True,
            "p_families": False}
logging.info("Sample p_global: %s", SAMPLE_P['p_global'])
logging.info("Sample p_zones: %s", SAMPLE_P['p_zones'])
logging.info("Sample p_families: %s", SAMPLE_P['p_families'])

# Define the operators in the MCMC and the frequency which with they are applied
ZONE_STEPS = 0.02

if SAMPLE_P['p_global']:
    P_GLOBAL_STEPS = 0.01
else:
    P_GLOBAL_STEPS = 0
if SAMPLE_P['p_zones']:
    P_ZONES_STEPS = 0.2
else:
    P_ZONES_STEPS = 0
if SAMPLE_P['p_families']:
    P_FAMILIES_STEPS = 0.05
else:
    P_FAMILIES_STEPS = 0

WEIGHT_STEPS = 1 - (ZONE_STEPS + P_GLOBAL_STEPS + P_FAMILIES_STEPS + P_ZONES_STEPS)
ALL_STEPS = ZONE_STEPS + WEIGHT_STEPS + P_GLOBAL_STEPS + P_ZONES_STEPS + P_FAMILIES_STEPS

if ALL_STEPS != 1.:
    print("steps must add to 1.")

OPERATORS = {'shrink_zone': ZONE_STEPS/4,
             'grow_zone': ZONE_STEPS/4,
             'swap_zone': ZONE_STEPS/2,
             'alter_weights': WEIGHT_STEPS,
             'alter_p_global': P_GLOBAL_STEPS,
             'alter_p_zones': P_ZONES_STEPS,
             'alter_p_families': P_FAMILIES_STEPS}

logging.info("Ratio of zone steps: %s", ZONE_STEPS)
logging.info("Ratio of weight steps: %s", WEIGHT_STEPS)
logging.info("Ratio of p_global steps: %s", P_GLOBAL_STEPS)
logging.info("Ratio of p_zones steps: %s", P_ZONES_STEPS)
logging.info("Ratio of p_families steps: %s", P_FAMILIES_STEPS)

# Markov chain coupled MC (mc3)
N_CHAINS = 6
SWAP_PERIOD = 1000
# Attempted inter-chain swaps after each SWAP_PERIOD
N_SWAPS = 3
logging.info("MCMC with %s chains and %s attempted swap(s) after %s steps.", N_CHAINS, N_SWAPS, SWAP_PERIOD)

# Number of zones
N_ZONES = 6
logging.info("Maximum number of sampled zones: %s", N_ZONES)


# Retrieve the features from the csv
sites, site_names, features, feature_names, category_names, families, family_names = \
    read_features_from_csv(file_location="data/features/", log=True)

network = compute_network(sites)

# Prior
# Geo-prior ("uniform", "distance" or "gaussian")
# Weights ("uniform")
# p_global ("uniform", "dirichlet")
# p_zones ("uniform")
# p_families ("uniform", "dirichlet")

PRIOR = {'geo': {'type': "uniform", 'parameters': {}},
         'weights': {'type': "uniform"},
         'p_global': {'type': "uniform", 'parameters': {}},
         'p_zones': {'type': "uniform"},
         'p_families': {'type': "uniform", 'parameters': {}}}

logging.info("Geo-prior: %s ", PRIOR['geo']['type'])
logging.info("Prior on weights: %s ", PRIOR['weights']['type'])
logging.info("Prior on p_global: %s ", PRIOR['p_global']['type'])
logging.info("Prior on p_zones: %s ", PRIOR['p_zones']['type'])
logging.info("Prior on p_families: %s ", PRIOR['p_families']['type'])


# Variance in the proposal distribution for weights, p_global, p_zones, p_families
VAR_PROPOSAL = {'weights': 0.1,
                'p_global': 0.1,
                'p_zones': 0.1,
                'p_families': 0.1}

logging.info("Variance of proposal distribution for weights: %s ", VAR_PROPOSAL['weights'])
logging.info("Variance of proposal distribution for p_global: %s ", VAR_PROPOSAL['p_global'])
logging.info("Variance of proposal distribution for p_zones: %s ", VAR_PROPOSAL['p_zones'])
logging.info("Variance of proposal distribution for p_families: %s ", VAR_PROPOSAL['p_families'])

# Estimate parameters for geo prior from the data
PRIOR['geo']['parameters'] = estimate_geo_prior_parameters(network, PRIOR['geo']['type'])


stats = []
samples = []

# Rerun experiment to check for consistency
for run in range(N_RUNS):
    for N in range(1, N_ZONES + 1):

        # Initial sample is empty
        initial_sample = Sample(zones=None, weights=None, p_global=None, p_zones=None, p_families=None)
        initial_sample.what_changed = {'lh': {'zones': True, 'weights': True,
                                              'p_global': True, 'p_zones': True, 'p_families': True},
                                       'prior': {'zones': True, 'weights': True,
                                                 'p_global': True, 'p_zones': True, 'p_families': True}}

        zone_sampler = ZoneMCMC_generative(network=network, features=features,
                                           min_size=MIN_SIZE, max_size=MAX_SIZE, initial_size=INITIAL_SIZE,
                                           n_zones=N, connected_only=CONNECTED_ONLY, n_chains=N_CHAINS,
                                           initial_sample=initial_sample, swap_period=SWAP_PERIOD,
                                           operators=OPERATORS, chain_swaps=N_SWAPS,
                                           families=None, inheritance=INHERITANCE,
                                           prior=PRIOR, sample_p=SAMPLE_P,
                                           var_proposal=VAR_PROPOSAL)
        zone_sampler.generate_samples(N_STEPS, N_SAMPLES, BURN_IN)

        # Evaluate likelihood and prior for each zone alone (makes it possible to rank zones)
        zone_sampler = contribution_per_zone(zone_sampler)

        run_stats = zone_sampler.statistics

        # The last sample is used as an initial sample
        initial_sample = Sample(zones=run_stats['last_sample'].zones, weights=run_stats['last_sample'].weights,
                                p_global=run_stats['last_sample'].p_global,
                                p_zones=run_stats['last_sample'].p_zones,
                                p_families=run_stats['last_sample'].p_families)

        initial_sample.what_changed = {'lh': {'zones': True, 'weights': True,
                                              'p_global': True, 'p_zones': True, 'p_families': True},
                                       'prior': {'zones': True, 'weights': True,
                                                 'p_global': True, 'p_zones': True, 'p_families': True}}
        # Save stats to file
        path = TEST_SAMPLING_RESULTS_PATH.format(nz=N, run=run)
        dump(run_stats, path)

