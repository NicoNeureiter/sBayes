#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import logging
import itertools
import datetime
import multiprocessing.pool

import numpy as np

from src.util import dump, transform_weights_from_log, transform_weights_to_log
from src.preprocessing import (get_network,
                               generate_ecdf_geo_prior,
                               simulate_zones,
                               simulate_assignment_probabilities,
                               sample_from_likelihood,
                               simulate_contact, assign_na)
from src.sampling.zone_sampling_generative import ZoneMCMC_generative, Sample


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

# Number of simulated features and categories
N_FEATURES_SIM = 30
# Most features are binary (70%), some have three and four categories (20 and 10%)
P_N_CATEGORIES_SIM = {'2': 0.7, '3': 0.2, '4': 0.1}

# Number and size of simulated families
N_FAMILIES_SIM = 0
MAX_FAMILY_SIZE_SIM = 80
MIN_FAMILY_SIZE_SIM = 20


# Intensity
# Intensity of contact, passed as alpha when drawing samples from dirichlet distribution
# lower values correspond to zones with more sites having similar features
I_GLOBAL_SIM = 3
I_CONTACT_SIM = [1.5, 1.25, 1, 0.75, 0.5, 0.25]
I_CONTACT_SIM = [1]
I_INHERITANCE_SIM = 0.0001

# Number of contact features /inherited features
# Number of features, passed as alpha when drawing samples from dirichlet distribution
# higher values correspond to more features for which the influence of contact/inheritance is strong
F_GLOBAL_SIM = 1.
F_CONTACT_SIM = [1.25, 1.5, 1.75, 2, 2.25, 2.5]
F_CONTACT_SIM = [1.5]
F_INHERITANCE_SIM = 0.00001


# Simulate contact for zones with different shapes and sizes (numbers correspond to IDs of hand-drawn zones)
TEST_ZONE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
TEST_ZONE = [6]
# Simulate ease (intensity, number of features affected by contact)
# [0] Hard: Unfavourable zones, i.e. low intensity and few features affected by contact
# [5] Easy: Favourable zones, i.e.e high intensity and many features affected by contact
TEST_EASE = range(0, len(F_CONTACT_SIM))


#################################
# MCMC
#################################

# General
BURN_IN = 0
N_STEPS = 400000
N_SAMPLES = 1000
N_RUNS = 1


# Zone sampling
MIN_SIZE = 3
MAX_SIZE = 200
INITIAL_SIZE = 25
CONNECTED_ONLY = False
OPERATORS = {'shrink_zone': 0.005,
             'grow_zone': 0.005,
             'swap_zone': 0.01,
             'alter_weights': 0.98}

# Markov chain coupled MC (mc3)

# Number of independent chains
N_CHAINS = 7
SWAP_PERIOD = 500
# Attempted inter-chain swaps after each SWAP_PERIOD
N_SWAPS = 1

# Number of zones
N_ZONES = 1

# Geo-prior (none, distance or gaussian)
GEO_PRIOR = 'none'

# Include inheritance in the model?
INHERITANCE = False

sampling_param_grid = list(itertools.product(TEST_EASE, TEST_ZONE))
print(sampling_param_grid)


def evaluate_sampling_parameters(params):
    e, z = params

    # Retrieve the network from the DB
    network_sim = get_network(reevaluate=True)

    stats = []
    samples = []

    # Simulate zones
    zones_sim = simulate_zones(z, network_sim)

    # Simulate families
    if not INHERITANCE:
        families_sim = None
    else:
        # Todo: Randomly assign sites to families
        families_sim = None

    #q_families = np.random.dirichlet([1.0] * N_FAMILIES_SIM, size=n_sites_sim)
    #families = (q_families == np.max(q_families, axis=-1)[:, None])
    # if N_families sim = 0, 0

    # Rerun experiment to check for consistency
    for run in range(N_RUNS):

        # Simulate probabilities for assigning features to categories in zones, globally and in families if available
        p_global_sim, p_contact_sim, p_inheritance_sim\
        = simulate_assignment_probabilities(n_features=N_FEATURES_SIM,
                                            p_number_categories=P_N_CATEGORIES_SIM,
                                            zones=zones_sim, families=families_sim,
                                            intensity_global=I_GLOBAL_SIM,
                                            intensity_contact=I_CONTACT_SIM[e],
                                            intensity_inheritance=I_INHERITANCE_SIM,
                                            inheritance=INHERITANCE)

        # Define weights which control the influence of contact (and inheritance if available) when simulating features
        if not INHERITANCE:
            alpha_sim = [F_GLOBAL_SIM, F_CONTACT_SIM[e]]
        else:
            alpha_sim = [F_GLOBAL_SIM, F_CONTACT_SIM[e], F_INHERITANCE_SIM]

        # columns in weights_sim: global, contact, (inheritance if available)
        weights_sim = np.random.dirichlet(alpha_sim, N_FEATURES_SIM)

        # Simulate features
        features = sample_from_likelihood(zones=zones_sim, families=families_sim,
                                          p_global=p_global_sim, p_contact=p_contact_sim,
                                          p_inheritance=p_inheritance_sim, weights=weights_sim,
                                          inheritance=INHERITANCE)
        # Todo: Inheritance in simulation should be ok, now check if it works in inference
        # Todo: Fix!
        # Assign some features to NA (makes testing more realistic)
        #features = assign_na(features=features, n_na=20)

        # Sampling
        # Intial sample is empty
        initial_sample = Sample(zones=None, weights=None)
        initial_sample.weights_changed = True
        initial_sample.zones_changed = True

        zone_sampler = ZoneMCMC_generative(network=network_sim, features=features,
                                           min_size=MIN_SIZE, max_size=MAX_SIZE, initial_size=INITIAL_SIZE,
                                           n_zones=N_ZONES, connected_only=CONNECTED_ONLY,
                                           n_chains=N_CHAINS, initial_sample=initial_sample,
                                           swap_period=SWAP_PERIOD, operators=OPERATORS,
                                           chain_swaps=N_SWAPS, inheritance=INHERITANCE)

        zone_sampler.generate_samples(N_STEPS, N_SAMPLES, BURN_IN)

        # Collect statistics
        run_stats = zone_sampler.statistics

        true_sample = Sample(zones=zones_sim, weights=transform_weights_to_log(weights_sim))
        true_sample.weights_changed=True
        true_sample.zones_changed=True

        true_sample.weights_changed = True
        true_sample.zones_changed = True
        run_stats['true_zones_ll'] = zone_sampler.likelihood(true_sample, 0)
        run_stats['true_zones_prior'] = zone_sampler.prior(true_sample)
        run_stats['true_zones'] = true_sample.zones
        run_stats['true_weights'] = true_sample.weights
        stats.append(run_stats)

        # Store the results
        path = TEST_SAMPLING_RESULTS_PATH.format(z=z, e=e, run=run)
        dump(run_stats, path)

    return samples, stats


if __name__ == '__main__':

    # Test ease
    with MyPool(4) as pool:
        all_stats = pool.map(evaluate_sampling_parameters, sampling_param_grid)
    # for params in sampling_param_grid:
    #     evaluate_sampling_parameters(params)

