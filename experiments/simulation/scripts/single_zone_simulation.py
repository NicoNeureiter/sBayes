#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import itertools
import multiprocessing.pool

import os

from src.util import (dump, transform_weights_to_log, transform_p_to_log, set_experiment_name)
from src.preprocessing import (get_sites, compute_network, simulate_assignment_probabilities,
                               estimate_geo_prior_parameters,
                               simulate_zones, simulate_families,
                               simulate_weights,
                               simulate_features, get_global_frequencies,
                               assign_na)
from src.sampling.zone_sampling import ZoneMCMC_generative, Sample


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

TEST_SAMPLING_DIRECTORY = 'results/single_zone/{experiment}/'.format(experiment=experiment_name)
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
# Some features are binary (20%), some have three and four categories (20 and 10%)
P_N_CATEGORIES_SIM = {'2': 0.1, '3': 0.3, '4': 0.6}

# Number and size of simulated families
N_FAMILIES_SIM = 1
MAX_FAMILY_SIZE_SIM = 50
MIN_FAMILY_SIZE_SIM = 45

# Intensity
# Intensity of contact, passed as alpha when drawing samples from dirichlet distribution
# lower values correspond to zones with more sites having similar features
I_GLOBAL_SIM = 0.5
# I_CONTACT_SIM = [1.5, 1.25, 1, 0.75, 0.5, 0.25]
I_CONTACT_SIM = [0.1]
I_INHERITANCE_SIM = 0.5

# Number of contact features /inherited features
# Number of features, passed as alpha when drawing samples from dirichlet distribution
# higher values correspond to more features for which the influence of contact/inheritance is strong
F_GLOBAL_SIM = 1.
# F_CONTACT_SIM = [1.25, 1.5, 1.75, 2, 2.25, 2.5]
F_CONTACT_SIM = [2]
F_INHERITANCE_SIM = 1.5


# Simulate contact for zones with different shapes and sizes (numbers correspond to IDs of hand-drawn zones)
# TEST_ZONE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
TEST_ZONE = [4]
# Simulate ease (intensity, number of features affected by contact)
# [0] Hard: Unfavourable zones, i.e. low intensity and few features affected by contact
# [5] Easy: Favourable zones, i.e.e high intensity and many features affected by contact
TEST_EASE = range(0, len(F_CONTACT_SIM))


#################################
# MCMC
#################################

# General
BURN_IN = 20
N_STEPS = 1000000
N_SAMPLES = 1000
N_RUNS = 1


# Zone sampling
MIN_SIZE = 1
MAX_SIZE = 200
INITIAL_SIZE = 25
CONNECTED_ONLY = False

# Include inheritance (language families) in the model?
INHERITANCE = True

# Sample the probabilities for categories in zones and families? If False the maximum likelihood estimate is used.
# For global, the probabilities for categories are the maximum likelihood estimates.
SAMPLE_P_ZONES = True
SAMPLE_P_FAMILIES = True
SAMPLE_P = {"zones": SAMPLE_P_ZONES,
            "families": SAMPLE_P_FAMILIES}

# Estimate the global frequencies of each category from data or load from file?
GLOBAL_FREQ_MODE = {'estimate_from_data': True, 'file': ""}

# Consistency check
if not INHERITANCE and SAMPLE_P_FAMILIES:
    raise ValueError('The model does not consider language families ("INHERITANCE" = False). '
                     'Set "SAMPLE_P_FAMILIES" to False.')

# Define the operators in the MCMC and the frequency which with they are applied

ZONE_STEPS = 0.02

if SAMPLE_P_ZONES:
    if SAMPLE_P_FAMILIES:
        P_FAMILIES_STEPS = 0.2
        P_ZONES_STEPS = 0.2
    else:
        P_FAMILIES_STEPS = 0.
        P_ZONES_STEPS = 0.2

    WEIGHT_STEPS = 1 - (ZONE_STEPS + P_FAMILIES_STEPS + P_ZONES_STEPS)

else:
    WEIGHT_STEPS = 1 - ZONE_STEPS
    P_ZONES_STEPS = 0
    P_FAMILIES_STEPS = 0

ALL_STEPS = ZONE_STEPS + WEIGHT_STEPS + P_ZONES_STEPS + P_FAMILIES_STEPS
if ALL_STEPS != 1.:
    print("steps must add to 1.")

print("Ratio of zone steps:", ZONE_STEPS,
      "\nRatio of weight steps", WEIGHT_STEPS,
      "\nRatio of p_zones steps", P_ZONES_STEPS,
      "\nRatio of p_families steps", P_FAMILIES_STEPS)

OPERATORS = {'shrink_zone': ZONE_STEPS/4,
             'grow_zone': ZONE_STEPS/4,
             'swap_zone': ZONE_STEPS/2,
             'alter_weights': WEIGHT_STEPS,
             'alter_p_zones': P_ZONES_STEPS,
             'alter_p_families': P_FAMILIES_STEPS}


# Markov chain coupled MC (mc3)
# Number of independent chains
# todo: change when testing is over
N_CHAINS = 15
SWAP_PERIOD = 500
# Attempted inter-chain swaps after each SWAP_PERIOD
N_SWAPS = 1

# Number of zones
N_ZONES = 1

# Prior
# Geo-prior ("uniform", "distance" or "gaussian")
GEO_PRIOR = "uniform"
# Weights ("uniform")
PRIOR_WEIGHTS = "uniform"
# Probabilities for categories in a zone ("uniform")
PRIOR_P_ZONES = "uniform"

# Probabilities for categories in a family ("uniform", "dirichlet")
PRIOR_P_FAMILIES = "uniform"
PRIOR_P_FAMILIES_FILES = [""]

# Combined
PRIOR = {'geo_prior': GEO_PRIOR,
         'geo_prior_parameters': {},
         'weights': PRIOR_WEIGHTS,
         'p_zones': PRIOR_P_ZONES,
         'p_families': PRIOR_P_FAMILIES,
         'p_families_parameters': {'files': PRIOR_P_FAMILIES_FILES}}


# Simulate missing data: How many features are NA?
NA_FEATURES = 1

sampling_param_grid = list(itertools.product(TEST_EASE, TEST_ZONE))
print(sampling_param_grid)


def evaluate_sampling_parameters(params):
    e, z = params

    # Retrieve the sites from the csv and transform into a network
    sites_sim, site_names = get_sites("data/sites_simulation.csv")
    network_sim = compute_network(sites_sim)

    stats = []
    samples = []

    # Simulate zones
    zones_sim = simulate_zones(zone_id=z, sites_sim=sites_sim)

    # Estimate parameters for geo prior from the data
    PRIOR['geo_prior_parameters'] = estimate_geo_prior_parameters(network_sim, GEO_PRIOR)

    # Rerun experiment to check for consistency
    for run in range(N_RUNS):
        # SIMULATION
        # Simulate families
        if not INHERITANCE:
            families_sim = None
        else:
            families_sim = simulate_families(sites_sim, n_families=N_FAMILIES_SIM,
                                             min_family_size=MIN_FAMILY_SIZE_SIM,
                                             max_family_size=MAX_FAMILY_SIZE_SIM)

        # Simulate weights, i.e. then influence of global bias, contact and inheritance on each feature
        weights_sim = simulate_weights(f_global=F_GLOBAL_SIM, f_contact=F_CONTACT_SIM[e],
                                       f_inheritance=F_INHERITANCE_SIM, inheritance=INHERITANCE,
                                       n_features=N_FEATURES_SIM)

        # Simulate probabilities for features to belong to categories, globally in zones (and in families if available)
        p_global_sim, p_zones_sim, p_families_sim \
            = simulate_assignment_probabilities(n_features=N_FEATURES_SIM, p_number_categories=P_N_CATEGORIES_SIM,
                                                zones=zones_sim, families=families_sim,
                                                intensity_global=I_GLOBAL_SIM,
                                                intensity_contact=I_CONTACT_SIM[e],
                                                intensity_inheritance=I_INHERITANCE_SIM, inheritance=INHERITANCE)

        # Simulate features
        features_sim, categories_sim = simulate_features(zones=zones_sim, families=families_sim,
                                                         p_global=p_global_sim, p_contact=p_zones_sim,
                                                         p_inheritance=p_families_sim,
                                                         weights=weights_sim, inheritance=INHERITANCE)

        # Assign some features to NA (makes testing more realistic)
        features_sim = assign_na(features=features_sim, n_na=NA_FEATURES)

        # COMPUTE GLOBAL FREQUENCY
        global_freq = get_global_frequencies(GLOBAL_FREQ_MODE, features=features_sim)

        # Sampling
        # Initial sample is empty
        initial_sample = Sample(zones=None, weights=None, p_zones=None, p_families=None)
        initial_sample.what_changed = {'lh': {'zones': True, 'weights': True, 'p_zones': True, 'p_families': True},
                                       'prior': {'zones': True, 'weights': True, 'p_zones': True, 'p_families': True}}

        zone_sampler = ZoneMCMC_generative(network=network_sim, features=features_sim,
                                           min_size=MIN_SIZE, max_size=MAX_SIZE, initial_size=INITIAL_SIZE,
                                           n_zones=N_ZONES, connected_only=CONNECTED_ONLY, n_chains=N_CHAINS,
                                           initial_sample=initial_sample, swap_period=SWAP_PERIOD, operators=OPERATORS,
                                           families=families_sim, global_freq=global_freq,
                                           chain_swaps=N_SWAPS, inheritance=INHERITANCE, prior=PRIOR,
                                           sample_p=SAMPLE_P)

        zone_sampler.generate_samples(N_STEPS, N_SAMPLES, BURN_IN)

        # Collect statistics
        run_stats = zone_sampler.statistics

        # Evaluate the likelihood of the true sample
        weights_sim_log = transform_weights_to_log(weights_sim)
        p_zones_sim_log = transform_p_to_log(p_zones_sim)

        if INHERITANCE:
            p_families_sim_log = transform_p_to_log(p_families_sim)
        else:
            p_families_sim_log = None

        true_sample = Sample(zones=zones_sim, weights=weights_sim_log,
                             p_zones=p_zones_sim_log, p_families=p_families_sim_log)

        true_sample.what_changed = {'lh': {'zones': True, 'weights': True, 'p_zones': True, 'p_families': True},
                                    'prior': {'zones': True, 'weights': True, 'p_zones': True, 'p_families': True}}

        # Save stats about true sample
        run_stats['true_zones'] = zones_sim
        run_stats['true_weights'] = weights_sim
        run_stats['true_p_global'] = global_freq
        run_stats['true_p_zones'] = p_zones_sim
        run_stats['true_p_families'] = p_families_sim
        run_stats['true_ll'] = zone_sampler.likelihood(true_sample, 0)
        run_stats['true_prior'] = zone_sampler.prior(true_sample, 0)
        run_stats["true_families"] = families_sim

        stats.append(run_stats)

        # Save stats to file
        path = TEST_SAMPLING_RESULTS_PATH.format(z=z, e=e, run=run)
        dump(run_stats, path)

    return samples, stats


if __name__ == '__main__':

    # Test ease
    with MyPool(4) as pool:
        all_stats = pool.map(evaluate_sampling_parameters, sampling_param_grid)
    # for params in sampling_param_grid:
    #     evaluate_sampling_parameters(params)

