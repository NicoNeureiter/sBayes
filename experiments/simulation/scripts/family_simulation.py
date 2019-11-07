#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import itertools
import multiprocessing.pool
import numpy as np
import os

from src.util import (dump, transform_weights_to_log, transform_p_to_log,
                      set_experiment_name)
from src.preprocessing import (get_sites, compute_network, simulate_assignment_probabilities,
                               estimate_geo_prior_parameters,
                               simulate_zones, simulate_families,
                               simulate_weights,
                               simulate_features, get_p_families_prior, get_p_global_prior,
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

################################
# Setup
################################

# Name of the experiment
experiment_name = set_experiment_name()

TEST_SAMPLING_DIRECTORY = 'results/family/{experiment}/'.format(experiment=experiment_name)
TEST_SAMPLING_RESULTS_PATH = TEST_SAMPLING_DIRECTORY + 'zone_z{z}_e{e}_i{i}_{run}.pkl'


# Make directory if it doesn't exist yet
if not os.path.exists(TEST_SAMPLING_DIRECTORY):
    os.mkdir(TEST_SAMPLING_DIRECTORY)

# Logging
TEST_SAMPLING_LOG_PATH = TEST_SAMPLING_DIRECTORY + 'info.log'
logging.basicConfig(filename=TEST_SAMPLING_LOG_PATH, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("Experiment: %s", experiment_name)


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
MIN_FAMILY_SIZE_SIM = 50
MAX_FAMILY_SIZE_SIM = 60

# Intensity
# Intensity of contact, passed as alpha when drawing samples from dirichlet distribution
# lower values correspond to zones with more sites having similar features
I_GLOBAL_SIM = 0.5
# I_CONTACT_SIM = [1.5, 1.25, 1, 0.75, 0.5, 0.25]
I_CONTACT_SIM = [0.5]
I_INHERITANCE_SIM = 0.25

# Number of contact features /inherited features
# Number of features, passed as alpha when drawing samples from dirichlet distribution
# higher values correspond to more features for which the influence of contact/inheritance is strong
F_GLOBAL_SIM = 1.
# F_CONTACT_SIM = [1.25, 1.5, 1.75, 2, 2.25, 2.5]
F_CONTACT_SIM = [1.5]
F_INHERITANCE_SIM = 1.75


# Simulate contact for zone (number corresponds to ID of hand-drawn zones)
TEST_ZONE = 10

# Simulate ease (intensity, number of features affected by contact)
# [0] Hard: Unfavourable zones, i.e. low intensity and few features affected by contact
# [5] Easy: Favourable zones, i.e.e high intensity and many features affected by contact
TEST_EASE = 0


#################################
# MCMC
#################################

# General
BURN_IN = 0
N_STEPS = 400000
N_SAMPLES = 2000
N_RUNS = 1


# Zone sampling
MIN_SIZE = 5
MAX_SIZE = 200
INITIAL_SIZE = 25
CONNECTED_ONLY = False

# Include inheritance (language families) in the simulation?
INHERITANCE_SIM = True

# Include inheritance in the inference?
TEST_INHERITANCE = [False, True]

# Markov chain coupled MC (mc3)
# Number of independent chains

N_CHAINS = 5
SWAP_PERIOD = 500
# Attempted inter-chain swaps after each SWAP_PERIOD
N_SWAPS = 1

# Number of zones
N_ZONES = 1

# Variance in the proposal distribution for weights, p_global, p_zones, p_families
VAR_PROPOSAL = {'weights': 0.1,
                'p_global': 0.1,
                'p_zones': 0.1,
                'p_families': 0.1}

# Prior
# Geo-prior ("uniform", "distance" or "gaussian")
GEO_PRIOR = "uniform"

# Weights ("uniform")
PRIOR_WEIGHTS = "uniform"

# Global prior probabilities for categories ("uniform", "dirichlet")
PRIOR_P_GLOBAL = "dirichlet"
PRIOR_P_GLOBAL_FILE = []

# Prior probabilities for categories in a zone ("uniform")
PRIOR_P_ZONES = "uniform"

# Probabilities for categories in a family ("uniform", "dirichlet")
PRIOR_P_FAMILIES = "uniform"

# Combined
PRIOR = {'geo': {'type': GEO_PRIOR, 'parameters': {}},
         'weights': {'type': PRIOR_WEIGHTS},
         'p_global': {'type': PRIOR_P_GLOBAL, 'parameters': {}},
         'p_zones': {'type': PRIOR_P_ZONES},
         'p_families': {'type': PRIOR_P_FAMILIES, 'parameters': {}}}

# SIMULATION
# Simulate zones
sites_sim, site_names = get_sites("data/sites_simulation.csv", retrieve_family=True)
network_sim = compute_network(sites_sim)
zones_sim = simulate_zones(zone_id=TEST_ZONE, sites_sim=sites_sim)


# Simulate families
if not INHERITANCE_SIM:
    families_sim = None
else:
    families_sim = simulate_families(fam_id=1, sites_sim=sites_sim)

# Simulate weights, i.e. the influence of global bias, contact and inheritance on each feature
weights_sim = simulate_weights(f_global=F_GLOBAL_SIM, f_contact=F_CONTACT_SIM[TEST_EASE],
                               f_inheritance=F_INHERITANCE_SIM, inheritance=INHERITANCE_SIM,
                               n_features=N_FEATURES_SIM)

# Simulate probabilities for features to belong to categories, globally in zones (and in families if available)
p_global_sim, p_zones_sim, p_families_sim \
    = simulate_assignment_probabilities(n_features=N_FEATURES_SIM, p_number_categories=P_N_CATEGORIES_SIM,
                                        zones=zones_sim, families=families_sim,
                                        intensity_global=I_GLOBAL_SIM,
                                        intensity_contact=I_CONTACT_SIM[TEST_EASE],
                                        intensity_inheritance=I_INHERITANCE_SIM, inheritance=INHERITANCE_SIM)

# Simulate features
features_sim, categories_sim = simulate_features(zones=zones_sim, families=families_sim,
                                                 p_global=p_global_sim, p_contact=p_zones_sim,
                                                 p_inheritance=p_families_sim,
                                                 weights=weights_sim, inheritance=INHERITANCE_SIM)

family_names = {'external': str(list(range(families_sim.shape[0]))),
                'internal': str(list(range(families_sim.shape[0])))}
feature_names = {'external': str(list(range(features_sim.shape[0]))),
                 'internal': str(list(range(features_sim.shape[0])))}
category_names = {'external': str(categories_sim),
                  'internal': str(categories_sim)}
stats = []
samples = []


if __name__ == '__main__':

    for i in TEST_INHERITANCE:

        # Sample the global probabilities, the prob. in zones and in families? If False use maximum likelihood estimate.
        # For global, the probabilities for categories are the maximum likelihood estimates.
        SAMPLE_P_GLOBAL = True
        SAMPLE_P_ZONES = True
        SAMPLE_P_FAMILIES = i

        SAMPLE_P = {"p_global": SAMPLE_P_GLOBAL,
                    "p_zones": SAMPLE_P_ZONES,
                    "p_families": SAMPLE_P_FAMILIES}

        # Define the operators in the MCMC and the frequency which with they are applied
        ZONE_STEPS = 0.02

        # todo: fix
        if SAMPLE_P_ZONES and SAMPLE_P_GLOBAL:
            if SAMPLE_P_FAMILIES:
                P_GLOBAL_STEPS = 0.01
                P_ZONES_STEPS = 0.3
                P_FAMILIES_STEPS = 0.05
            else:
                P_GLOBAL_STEPS = 0.02
                P_ZONES_STEPS = 0.5
                P_FAMILIES_STEPS = 0.

            WEIGHT_STEPS = 1 - (ZONE_STEPS + P_GLOBAL_STEPS + P_ZONES_STEPS + P_FAMILIES_STEPS)

        else:
            WEIGHT_STEPS = 1 - ZONE_STEPS
            P_GLOBAL_STEPS = 0
            P_ZONES_STEPS = 0
            P_FAMILIES_STEPS = 0

        ALL_STEPS = ZONE_STEPS + WEIGHT_STEPS + P_GLOBAL_STEPS + P_ZONES_STEPS + P_FAMILIES_STEPS
        if ALL_STEPS != 1.:
            print("steps must add to 1.")

        print("Ratio of zone steps:", ZONE_STEPS,
              "\nRatio of weight steps", WEIGHT_STEPS,
              "\nRatio of p_global steps", P_GLOBAL_STEPS,
              "\nRatio of p_zones steps", P_ZONES_STEPS,
              "\nRatio of p_families steps", P_FAMILIES_STEPS)

        OPERATORS = {'shrink_zone': ZONE_STEPS / 4,
                     'grow_zone': ZONE_STEPS / 4,
                     'swap_zone': ZONE_STEPS / 2,
                     'alter_weights': WEIGHT_STEPS,
                     'alter_p_global': P_GLOBAL_STEPS,
                     'alter_p_zones': P_ZONES_STEPS,
                     'alter_p_families': P_FAMILIES_STEPS}

        # Estimate parameters for geo prior from the data
        PRIOR['geo']['parameters'] = estimate_geo_prior_parameters(network_sim, PRIOR['geo']['type'])

        # Import the prior for p_global and p_families from file
        if SAMPLE_P_GLOBAL and PRIOR_P_GLOBAL == "dirichlet":
            p_global_dirichlet, p_global_categories = get_p_global_prior(feature_names=feature_names,
                                                                         category_names=category_names,
                                                                         file_location="data")
            PRIOR['p_global']['parameters']['dirichlet'] = p_global_dirichlet
            PRIOR['p_global']['parameters']['categories'] = p_global_categories

        if SAMPLE_P_FAMILIES and PRIOR_P_FAMILIES == "dirichlet" and i:
            family_dirichlet, family_categories = get_p_families_prior(family_names=family_names,
                                                                       feature_names=feature_names,
                                                                       category_names=category_names)
            PRIOR['p_families']['parameters']['dirichlet'] = family_dirichlet
            PRIOR['p_families']['parameters']['categories'] = family_categories

        # Rerun experiment to check for consistency
        for run in range(N_RUNS):

            # Sampling
            # Initial sample is empty
            initial_sample = Sample(zones=None, weights=None, p_zones=None, p_families=None, p_global=None)
            initial_sample.what_changed = {'lh': {'zones': True, 'weights': True,
                                                  'p_global': True, 'p_zones': True, 'p_families': True},
                                           'prior': {'zones': True, 'weights': True,
                                                     'p_global': True, 'p_zones': True, 'p_families': True}}

            zone_sampler = ZoneMCMC_generative(network=network_sim, features=features_sim,
                                               min_size=MIN_SIZE, max_size=MAX_SIZE, initial_size=INITIAL_SIZE,
                                               n_zones=N_ZONES, connected_only=CONNECTED_ONLY, n_chains=N_CHAINS,
                                               initial_sample=initial_sample, swap_period=SWAP_PERIOD,
                                               operators=OPERATORS, families=families_sim, chain_swaps=N_SWAPS,
                                               inheritance=i, prior=PRIOR, sample_p=SAMPLE_P, var_proposal=VAR_PROPOSAL)

            zone_sampler.generate_samples(N_STEPS, N_SAMPLES, BURN_IN)

            # Collect statistics
            run_stats = zone_sampler.statistics

            # Evaluate the likelihood of the true sample
            # If the model includes inheritance use all weights, if not use only the first two weights (global, zone)
            if i:
                weights_sim_log = transform_weights_to_log(weights_sim)
            else:
                weights_sim_log = transform_weights_to_log(weights_sim[:, 1:2])

            p_global_sim_log = transform_p_to_log(p_global_sim)
            p_zones_sim_log = transform_p_to_log(p_zones_sim)

            if INHERITANCE_SIM:
                p_families_sim_log = transform_p_to_log(p_families_sim)
            else:
                p_families_sim_log = None

            true_sample = Sample(zones=zones_sim, weights=weights_sim_log,
                                 p_global=p_global_sim_log, p_zones=p_zones_sim_log, p_families=p_families_sim_log)

            true_sample.what_changed = {'lh': {'zones': True, 'weights': True,
                                               'p_global': True, 'p_zones': True, 'p_families': True},
                                        'prior': {'zones': True, 'weights': True,
                                                  'p_global': True, 'p_zones': True, 'p_families': True}}

            # Save stats about true sample
            run_stats['true_zones'] = zones_sim
            run_stats['true_weights'] = weights_sim
            run_stats['true_p_global'] = p_global_sim
            run_stats['true_p_zones'] = p_zones_sim
            run_stats['true_p_families'] = p_families_sim
            run_stats['true_ll'] = zone_sampler.likelihood(true_sample, 0)
            run_stats['true_prior'] = zone_sampler.prior(true_sample, 0)
            run_stats["true_families"] = families_sim

            stats.append(run_stats)

            # Save stats to file
            path = TEST_SAMPLING_RESULTS_PATH.format(z=TEST_ZONE, e=TEST_EASE, i=int(i), run=run)
            dump(run_stats, path)