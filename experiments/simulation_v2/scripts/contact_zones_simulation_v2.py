#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import multiprocessing.pool
import numpy as np
import os

from src.postprocessing import print_operator_statistics, print_operator_statistics_header
from src.preprocessing import (compute_network, get_sites,
                               estimate_geo_prior_parameters,
                               simulate_assignment_probabilities,
                               simulate_families,
                               simulate_features,
                               simulate_weights,
                               simulate_zones)
from src.sampling.zone_sampling import Sample, ZoneMCMC_generative
from src.util import dump, normalize, set_experiment_name


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


class ContactZonesSimulator:
    def __init__(self):
        # General setup
        self.experiment_name = set_experiment_name()
        self.TEST_SAMPLING_DIRECTORY = '../results/contact_zones/{experiment}/'.\
            format(experiment=self.experiment_name)
        self.TEST_SAMPLING_RESULTS_PATH = self.TEST_SAMPLING_DIRECTORY + 'contact_zones_i{i}_{run}.pkl'
        self.TEST_SAMPLING_LOG_PATH = self.TEST_SAMPLING_DIRECTORY + 'info.log'
        if not os.path.exists(self.TEST_SAMPLING_DIRECTORY):
            os.mkdir(self.TEST_SAMPLING_DIRECTORY)

        # Get parameters from config.json
        self.config = {}
        self.get_parameters()

        # Simulation variables to be calculated
        self.network_sim = None
        self.zones_sim = None
        self.features_sim = None
        self.categories_sim = None
        self.families_sim = None
        self.weights_sim = None
        self.p_global_sim = None
        self.p_zones_sim = None
        self.p_families_sim = None

    def get_parameters(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)

    def logging_setup(self):
        logging.basicConfig(filename=self.TEST_SAMPLING_LOG_PATH, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Experiment: %s", self.experiment_name)

    def logging_simulation(self):
        logging.info("Simulating %s features.", self.config['simulation']['N_FEATURES_SIM'])
        logging.info("Simulated global intensity: %s", self.config['simulation']['I_GLOBAL_SIM'])
        logging.info("Simulated contact intensity: %s", self.config['simulation']['I_CONTACT_SIM'])
        logging.info("Simulated inherited intensity: %s", self.config['simulation']['I_CONTACT_SIM'])
        logging.info("Simulated global exposition (number of similar features): %s",
                     self.config['simulation']['F_GLOBAL_SIM'])
        logging.info("Simulated exposition in zone (number of similar features): %s",
                     self.config['simulation']['F_CONTACT_SIM'])
        logging.info("Simulated exposition in family (number of similar features): %s",
                     self.config['simulation']['F_CONTACT_SIM'])
        logging.info("Simulated zone: %s", self.config['simulation']['ZONE'])
        logging.info("Inheritance is simulated: %s", self.config['simulation']['INHERITANCE_SIM'])

    def logging_mcmc(self):
        logging.info("MCMC with %s steps and %s samples (burn-in %s steps)",
                     self.config['mcmc']['N_STEPS'], self.config['mcmc']['N_SAMPLES'],
                     self.config['mcmc']['BURN_IN'])
        logging.info("Zones have a minimum size of %s and a maximum size of %s. The initial size is %s.",
                     self.config['mcmc']['MIN_SIZE'], self.config['mcmc']['MAX_SIZE'],
                     self.config['mcmc']['INITIAL_SIZE'])
        logging.info("MCMC with %s chains and %s attempted swap(s) after %s steps.",
                     self.config['mcmc']['N_CHAINS'], self.config['mcmc']['N_SWAPS'],
                     self.config['mcmc']['SWAP_PERIOD'])
        logging.info("Number of sampled zones; %i", self.config['mcmc']['N_ZONES'])
        logging.info("Geo-prior: %s ", self.config['mcmc']['PRIOR']['geo']['type'])
        logging.info("Prior on weights: %s ", self.config['mcmc']['PRIOR']['weights']['type'])
        logging.info("Prior on p_global: %s ", self.config['mcmc']['PRIOR']['p_global']['type'])
        logging.info("Prior on p_zones: %s ", self.config['mcmc']['PRIOR']['p_zones']['type'])
        logging.info("Prior on p_families: %s ", self.config['mcmc']['PRIOR']['p_families']['type'])
        logging.info("Variance of proposal distribution for weights: %s ",
                     self.config['mcmc']['VAR_PROPOSAL']['weights'])
        logging.info("Variance of proposal distribution for p_global: %s ",
                     self.config['mcmc']['VAR_PROPOSAL']['p_global'])
        logging.info("Variance of proposal distribution for p_zones: %s ",
                     self.config['mcmc']['VAR_PROPOSAL']['p_zones'])
        logging.info("Variance of proposal distribution for p_families: %s ",
                     self.config['mcmc']['VAR_PROPOSAL']['p_families'])
        logging.info("Inheritance is considered for the inference: %s",
                     self.config['mcmc']['INHERITANCE_TEST'])

    def logging_sampling(self):
        logging.info("Sample p_global: %s", self.config['sampling']['SAMPLE_P']['p_global'])
        logging.info("Sample p_zones: %s", self.config['sampling']['SAMPLE_P']['p_zones'])
        logging.info("Sample p_families: %s", self.config['sampling']['SAMPLE_P']['p_families'])
        logging.info("Ratio of zone steps: %s", self.config['sampling']['ZONE_STEPS'])
        logging.info("Ratio of weight steps: %s", self.config['sampling']['WEIGHT_STEPS'])
        logging.info("Ratio of p_global steps: %s", self.config['sampling']['P_GLOBAL_STEPS'])
        logging.info("Ratio of p_zones steps: %s", self.config['sampling']['P_ZONES_STEPS'])
        logging.info("Ratio of p_families steps: %s", self.config['sampling']['P_FAMILIES_STEPS'])

    def logging_all(self):
        self.logging_setup()
        self.logging_simulation()
        self.logging_mcmc()

    def simulation(self):
        # Simulate zones
        sites_sim, site_names = get_sites("../data/sites_simulation.csv", retrieve_family=True)
        self.network_sim = compute_network(sites_sim)
        self.zones_sim = simulate_zones(zone_id=self.config['simulation']['ZONE'], sites_sim=sites_sim)

        # Simulate families
        self.families_sim = simulate_families(fam_id=1, sites_sim=sites_sim)

        # Simulate weights, i.e. the influence of global bias, contact and inheritance on each feature
        self.weights_sim = simulate_weights(f_global=self.config['simulation']['F_GLOBAL_SIM'],
                                            f_contact=self.config['simulation']['F_CONTACT_SIM'],
                                            f_inheritance=self.config['simulation']['F_INHERITANCE_SIM'],
                                            inheritance=self.config['simulation']['INHERITANCE_SIM'],
                                            n_features=self.config['simulation']['N_FEATURES_SIM'])

        # Simulate probabilities for features to belong to categories, globally in zones (and in families if available)
        self.p_global_sim, self.p_zones_sim, self.p_families_sim \
            = simulate_assignment_probabilities(n_features=self.config['simulation']['N_FEATURES_SIM'],
                                                p_number_categories=self.config['simulation']['P_N_CATEGORIES_SIM'],
                                                zones=self.zones_sim, families=self.families_sim,
                                                intensity_global=self.config['simulation']['I_GLOBAL_SIM'],
                                                intensity_contact=self.config['simulation']['I_CONTACT_SIM'],
                                                intensity_inheritance=self.config['simulation']['I_INHERITANCE_SIM'],
                                                inheritance=self.config['simulation']['INHERITANCE_SIM'])

        # Simulate features
        self.features_sim, self.categories_sim = \
            simulate_features(zones=self.zones_sim,
                              families=self.families_sim,
                              p_global=self.p_global_sim,
                              p_contact=self.p_zones_sim,
                              p_inheritance=self.p_families_sim,
                              weights=self.weights_sim,
                              inheritance=self.config['simulation']['INHERITANCE_SIM'])

        # Assign parameters for MCMC
        self.config['mcmc']['PRIOR']['geo']['parameters'] = \
            estimate_geo_prior_parameters(self.network_sim, self.config['mcmc']['PRIOR']['geo']['type'])

    def sampling_setup(self, inheritance_value):
        self.config['sampling']['SAMPLE_P']['p_families'] = inheritance_value

        # If p_global, p_zones and p_families are True -> use the values from the config.json
        # If they are False -> assign 0
        if not self.config['sampling']['SAMPLE_P']['p_global']:
            self.config['sampling']['P_GLOBAL_STEPS'] = 0
        if not self.config['sampling']['SAMPLE_P']['p_zones']:
            self.config['sampling']['P_ZONES_STEPS'] = 0
        if self.config['sampling']['SAMPLE_P']['p_families']:
            self.config['sampling']['P_FAMILIES_STEPS'] = 0

        # Assign WEIGHT_STEPS
        self.config['sampling']['WEIGHT_STEPS'] = \
            1 - (self.config['sampling']['ZONE_STEPS'] +
                 self.config['sampling']['P_GLOBAL_STEPS'] +
                 self.config['sampling']['P_FAMILIES_STEPS'] +
                 self.config['sampling']['P_ZONES_STEPS'])

        # Assign ALL_STEPS
        self.config['sampling']['ALL_STEPS'] = \
            self.config['sampling']['ZONE_STEPS'] + \
            self.config['sampling']['WEIGHT_STEPS'] + \
            self.config['sampling']['P_GLOBAL_STEPS'] + \
            self.config['sampling']['P_ZONES_STEPS'] + \
            self.config['sampling']['P_FAMILIES_STEPS']

        if self.config['sampling']['ALL_STEPS'] != 1:
            print("steps must add to 1.")

        # Assign operators
        ops = {'shrink_zone': self.config['sampling']['ZONE_STEPS'] / 4,
               'grow_zone': self.config['sampling']['ZONE_STEPS'] / 4,
               'swap_zone': self.config['sampling']['ZONE_STEPS'] / 2,
               'alter_weights': self.config['sampling']['WEIGHT_STEPS'],
               'alter_p_global': self.config['sampling']['P_GLOBAL_STEPS'],
               'alter_p_zones': self.config['sampling']['P_ZONES_STEPS'],
               'alter_p_families': self.config['sampling']['P_FAMILIES_STEPS']}

        return ops

    def sampling_run(self, inheritance_value, ops):
        # Sampling
        # Initial sample is empty
        initial_sample = Sample(zones=None, weights=None, p_zones=None, p_families=None, p_global=None)
        initial_sample.what_changed = {'lh': {'zones': True, 'weights': True,
                                              'p_global': True, 'p_zones': True, 'p_families': True},
                                       'prior': {'zones': True, 'weights': True,
                                                 'p_global': True, 'p_zones': True, 'p_families': True}}

        cur_zone_sampler = ZoneMCMC_generative(network=self.network_sim, features=self.features_sim,
                                               min_size=self.config['mcmc']['MIN_SIZE'],
                                               max_size=self.config['mcmc']['MAX_SIZE'],
                                               initial_size=self.config['mcmc']['INITIAL_SIZE'],
                                               n_zones=self.config['mcmc']['N_ZONES'],
                                               connected_only=self.config['mcmc']['CONNECTED_ONLY'],
                                               n_chains=self.config['mcmc']['N_CHAINS'],
                                               initial_sample=initial_sample,
                                               swap_period=self.config['mcmc']['SWAP_PERIOD'],
                                               operators=ops, families=self.families_sim,
                                               chain_swaps=self.config['mcmc']['N_SWAPS'],
                                               inheritance=inheritance_value,
                                               prior=self.config['mcmc']['PRIOR'],
                                               sample_p=self.config['sampling']['SAMPLE_P'],
                                               var_proposal=self.config['mcmc']['VAR_PROPOSAL'])

        cur_zone_sampler.generate_samples(self.config['mcmc']['N_STEPS'],
                                          self.config['mcmc']['N_SAMPLES'],
                                          self.config['mcmc']['BURN_IN'])

        return cur_zone_sampler

    @staticmethod
    def stats(cur_zone_sampler, ops):
        # Collect statistics
        cur_run_stats = cur_zone_sampler.statistics

        # Print operator stats
        print('')
        print_operator_statistics_header()
        for op_name in ops:
            print_operator_statistics(op_name, cur_run_stats)

        return cur_run_stats

    def true_sample_eval(self, inheritance_value):
        # Evaluate the likelihood of the true sample
        # If the model includes inheritance use all weights, if not use only the first two weights (global, zone)
        if inheritance_value:
            cur_weights_sim_normed = self.weights_sim.copy()
        else:
            cur_weights_sim_normed = normalize(self.weights_sim[:, :2])

        cur_p_global_sim_padded = self.p_global_sim[np.newaxis, ...]

        cur_true_sample = Sample(zones=self.zones_sim,
                                 weights=cur_weights_sim_normed,
                                 p_global=cur_p_global_sim_padded,
                                 p_zones=self.p_zones_sim,
                                 p_families=self.p_families_sim)

        cur_true_sample.what_changed = {'lh': {'zones': True,
                                               'weights': True,
                                               'p_global': True,
                                               'p_zones': True,
                                               'p_families': True},
                                        'prior': {'zones': True,
                                                  'weights': True,
                                                  'p_global': True,
                                                  'p_zones': True,
                                                  'p_families': True}}

        return cur_true_sample, cur_weights_sim_normed, cur_p_global_sim_padded

    def true_sample_stats(self, cur_zone_sampler, inheritance_value, cur_run, cur_true_sample, cur_run_stats,
                          cur_weights_sim_normed, cur_p_global_sim_padded):
        # Save stats about true sample
        run_stats['true_zones'] = self.zones_sim
        run_stats['true_weights'] = cur_weights_sim_normed
        run_stats['true_p_global'] = cur_p_global_sim_padded
        run_stats['true_p_zones'] = self.p_zones_sim
        run_stats['true_p_families'] = self.p_families_sim
        run_stats['true_ll'] = cur_zone_sampler.likelihood(cur_true_sample, 0)
        run_stats['true_prior'] = cur_zone_sampler.prior(cur_true_sample, 0)
        run_stats["true_families"] = self.families_sim

        # Save stats to file
        path = self.TEST_SAMPLING_RESULTS_PATH.format(i=int(inheritance_value), run=cur_run)
        dump(cur_run_stats, path)


if __name__ == '__main__':
    czs = ContactZonesSimulator()

    # Simulation
    czs.simulation()

    # Logging the setup, simulation and MCMC settings
    czs.logging_all()

    for inheritance_val in czs.config['mcmc']['INHERITANCE_TEST']:

        # Sampling setup, output operators
        operators = czs.sampling_setup(inheritance_val)

        # Logging sampling
        czs.logging_sampling()

        # Rerun experiment to check for consistency
        for run in range(czs.config['mcmc']['N_RUNS']):

            # Sampling run
            zone_sampler = czs.sampling_run(inheritance_val, operators)
            run_stats = czs.stats(zone_sampler, operators)

            # True sample
            true_sample, weights_sim_normed, p_global_sim_padded = czs.true_sample_eval(inheritance_val)
            czs.true_sample_stats(zone_sampler, inheritance_val, run,
                                  true_sample, run_stats,
                                  weights_sim_normed, p_global_sim_padded)
