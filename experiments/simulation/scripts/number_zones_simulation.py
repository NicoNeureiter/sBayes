#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import itertools
import multiprocessing.pool
import numpy as np
import os

from src.util import dump, set_experiment_name
from src.preprocessing import (get_sites, compute_network, simulate_assignment_probabilities,
                               estimate_geo_prior_parameters,
                               simulate_zones,
                               simulate_weights,
                               simulate_features)
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


################################
# Setup
################################

# Name of the experiment
experiment_name = set_experiment_name()

TEST_SAMPLING_DIRECTORY = 'results/number_zones/{experiment}/'.format(experiment=experiment_name)
TEST_SAMPLING_RESULTS_PATH = TEST_SAMPLING_DIRECTORY + 'number_zones_nz{nz}_{run}.pkl'

# Make directory if it doesn't exist yet
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
# In burn-in there are no weights-step (find zones first, find weights later)
BURN_IN = 0
N_STEPS = 100000
N_STEPS_INCREASE = 50000
N_SAMPLES = 1000
N_RUNS = 1
logging.info("MCMC with %s steps (+ %s increase per extra zone)and %s samples (burn-in %s steps)",
             N_STEPS, N_SAMPLES, N_STEPS_INCREASE, BURN_IN)

# Zone sampling
MIN_SIZE = 3
MAX_SIZE = 300
INITIAL_SIZE = 40
CONNECTED_ONLY = False
logging.info("Zones have a minimum size of %s and a maximum size of %s. The initial size is %s.",
             MIN_SIZE, MAX_SIZE, INITIAL_SIZE)

# Include inheritance (language families) in the model?
INHERITANCE = False
logging.info("Inheritance is considered: %s", INHERITANCE)

# Sample the probabilities for categories in zones and families? If False the maximum likelihood estimate is used.
# For global, the probabilities for categories are the maximum likelihood estimates.
SAMPLE_P = {"p_global": True,
            "p_zones":  True,
            "p_families": False}
logging.info("Sample p_global: %s", SAMPLE_P['p_global'])
logging.info("Sample p_zones: %s", SAMPLE_P['p_zones'])
logging.info("Sample p_families: %s", SAMPLE_P['p_families'])

# Define the operators in the MCMC and the frequency with which they are applied
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
N_CHAINS = 10
SWAP_PERIOD = 1000
# Attempted inter-chain swaps after each SWAP_PERIOD
N_SWAPS = 3
logging.info("MCMC with %s chains and %s attempted swap(s) after %s steps.", N_CHAINS, N_SWAPS, SWAP_PERIOD)


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


################################
# Simulation
################################

# Number of simulated features and categories
N_FEATURES_SIM = 35
# Some features are binary (40%), some have three and four categories (30%%)
P_N_CATEGORIES_SIM = {'2': 0.2, '3': 0.3, '4': 0.5}
logging.info("Simulating %s features.", N_FEATURES_SIM)

# Intensity
# Intensity of contact, passed as alpha when drawing samples from dirichlet distribution
# lower values correspond to zones with more sites having similar features
I_GLOBAL_SIM = 3
I_CONTACT_SIM = 0.1
logging.info("Simulated global intensity: %s", I_GLOBAL_SIM)
logging.info("Simulated contact intensity: %s", I_CONTACT_SIM)

# Number of contact features /inherited features
# Number of features, passed as alpha when drawing samples from dirichlet distribution
# higher values correspond to more features for which the influence of contact/inheritance is strong
F_GLOBAL_SIM = 0.3
F_CONTACT_SIM = 1.5
logging.info("Simulated global exposition (number of similar features): %s", F_GLOBAL_SIM)
logging.info("Simulated exposition in zone (number of similar features): %s", F_CONTACT_SIM)

# Simulate contact for zones (number corresponds to ID of hand-drawn zones)
ZONES = (2, 5, 7, 9)
logging.info("Simulated zones: %s", ZONES)

# Simulate zones
sites_sim, site_names = get_sites("data/sites_simulation.csv")
network_sim = compute_network(sites_sim)
zones_sim = simulate_zones(zone_id=ZONES, sites_sim=sites_sim)

# Simulate weights, i.e. then influence of global bias, contact and inheritance on each feature
weights_sim = simulate_weights(f_global=F_GLOBAL_SIM, f_contact=F_CONTACT_SIM,
                               inheritance=INHERITANCE, n_features=N_FEATURES_SIM)

# Simulate probabilities for features to belong to categories, globally in zones (and in families if available)
p_global_sim, p_zones_sim, _ = simulate_assignment_probabilities(n_features=N_FEATURES_SIM,
                                                                 p_number_categories=P_N_CATEGORIES_SIM,
                                                                 zones=zones_sim,
                                                                 intensity_global=I_GLOBAL_SIM,
                                                                 intensity_contact=I_CONTACT_SIM,
                                                                 inheritance=INHERITANCE)

# Simulate features
features_sim, categories_sim = simulate_features(zones=zones_sim,
                                                 p_global=p_global_sim, p_contact=p_zones_sim,
                                                 weights=weights_sim, inheritance=INHERITANCE)


# Estimate parameters for geo prior from the data
PRIOR['geo']['parameters'] = estimate_geo_prior_parameters(network_sim, PRIOR['geo']['type'])

# Maximum number of zones used in the sampler
TEST_N_ZONES = range(1, 8)
stats = []
samples = []

if __name__ == '__main__':
    for run in range(N_RUNS):
        # In the beginning the initial sample is empty
        initial_sample = Sample(zones=None, weights=None,  p_global=None, p_zones=None, p_families=None,)
        initial_sample.what_changed = {'lh': {'zones': True, 'weights': True,
                                              'p_global': True, 'p_zones': True, 'p_families': True},
                                       'prior': {'zones': True, 'weights': True,
                                                 'p_global': True, 'p_zones': True, 'p_families': True}}

        for n in TEST_N_ZONES:

                # Sampling
                zone_sampler = ZoneMCMC_generative(network=network_sim, features=features_sim,
                                                   min_size=MIN_SIZE, max_size=MAX_SIZE, initial_size=INITIAL_SIZE,
                                                   n_zones=n, connected_only=CONNECTED_ONLY, n_chains=N_CHAINS,
                                                   initial_sample=initial_sample, swap_period=SWAP_PERIOD,
                                                   operators=OPERATORS, families=None, chain_swaps=N_SWAPS,
                                                   inheritance=INHERITANCE, prior=PRIOR, sample_p=SAMPLE_P,
                                                   var_proposal=VAR_PROPOSAL)

                zone_sampler.generate_samples(N_STEPS, N_SAMPLES, BURN_IN)

                # Evaluate likelihood and prior for each zone alone (makes it possible to rank zones)
                zone_sampler = contribution_per_zone(zone_sampler)

                # Evaluate the likelihood of the true sample
                weights_sim_log = weights_sim
                p_global_sim_log = p_global_sim[np.newaxis, ...]
                p_zones_sim_log = p_zones_sim

                true_sample = Sample(zones=zones_sim, weights=weights_sim_log,
                                     p_global=p_global_sim_log, p_zones=p_zones_sim_log, p_families=None)

                true_sample.what_changed = {'lh': {'zones': True, 'weights': True,
                                                   'p_global': True, 'p_zones': True, 'p_families': True},
                                            'prior': {'zones': True, 'weights': True,
                                                      'p_global': True, 'p_zones': True, 'p_families': True}}

                run_stats = zone_sampler.statistics

                # Save stats about true sample
                run_stats['true_zones'] = zones_sim
                run_stats['true_weights'] = weights_sim
                run_stats['true_p_global'] = p_global_sim
                run_stats['true_p_zones'] = p_zones_sim
                run_stats['true_ll'] = zone_sampler.likelihood(true_sample, 0)
                run_stats['true_prior'] = zone_sampler.prior(true_sample, 0)

                # Collect statistics
                stats.append(run_stats)

                # Save stats to file
                path = TEST_SAMPLING_RESULTS_PATH.format(nz=n, run=run)
                dump(run_stats, path)

                # Use last sample from the previous run as initial sample for next run
                initial_sample = Sample(zones=run_stats['last_sample'].zones, weights=run_stats['last_sample'].weights,
                                        p_global=run_stats['last_sample'].p_global,
                                        p_zones=run_stats['last_sample'].p_zones,
                                        p_families=run_stats['last_sample'].p_families)
                initial_sample.what_changed = {'lh': {'zones': True, 'weights': True,
                                                      'p_global': True, 'p_zones': True, 'p_families': True},
                                               'prior': {'zones': True, 'weights': True,
                                                         'p_global': True, 'p_zones': True, 'p_families': True}}

                # The problem becomes harder with each added zone -> increase step size
                N_STEPS += N_STEPS_INCREASE






