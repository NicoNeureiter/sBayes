#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import numpy as np
import os

from src.postprocessing import print_operator_statistics, print_operator_statistics_header
from src.preprocessing import estimate_geo_prior_parameters
from src.sampling.zone_sampling import Sample, ZoneMCMC_generative
from src.util import dump, normalize, set_experiment_name


class MCMCSetup:
    def __init__(self, network_sim, zones_sim, features_sim, categories_sim, families_sim, weights_sim, p_global_sim,
                 p_zones_sim, p_families_sim):
        # General setup
        self.experiment_name = set_experiment_name()
        # we will have other directories and folders on line 43 (any changes required now?)
        self.TEST_SAMPLING_DIRECTORY = '../results/contact_zones/{experiment}/'. \
            format(experiment=self.experiment_name)
        self.TEST_SAMPLING_RESULTS_PATH = self.TEST_SAMPLING_DIRECTORY + 'contact_zones_i{i}_{run}.pkl'
        self.TEST_SAMPLING_LOG_PATH = self.TEST_SAMPLING_DIRECTORY + 'info.log'
        if not os.path.exists(self.TEST_SAMPLING_DIRECTORY):
            os.mkdir(self.TEST_SAMPLING_DIRECTORY)

        # Parameters from the simulation
        self.network_sim = network_sim
        self.zones_sim = zones_sim
        self.features_sim = features_sim
        self.categories_sim = categories_sim
        self.families_sim = families_sim
        self.weights_sim = weights_sim
        self.p_global_sim = p_global_sim
        self.p_zones_sim = p_zones_sim
        self.p_families_sim = p_families_sim

        # Get parameters from config_mcmc.json
        self.config = {}
        self.get_parameters()

    def get_parameters(self):
        with open('config_mcmc.json', 'r') as f:
            self.config = json.load(f)

        self.config['mcmc']['PRIOR']['geo']['parameters'] = \
            estimate_geo_prior_parameters(self.network_sim, self.config['mcmc']['PRIOR']['geo']['type'])

    def logging_setup(self):
        logging.basicConfig(filename=self.TEST_SAMPLING_LOG_PATH, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Experiment: %s", self.experiment_name)

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
        logging.info("Sample p_global: %s", self.config['mcmc']['SAMPLE_P']['p_global'])
        logging.info("Sample p_zones: %s", self.config['mcmc']['SAMPLE_P']['p_zones'])
        logging.info("Sample p_families: %s", self.config['mcmc']['SAMPLE_P']['p_families'])
        logging.info("Ratio of zone steps: %s", self.config['mcmc']['ZONE_STEPS'])
        logging.info("Ratio of weight steps: %s", self.config['mcmc']['WEIGHT_STEPS'])
        logging.info("Ratio of p_global steps: %s", self.config['mcmc']['P_GLOBAL_STEPS'])
        logging.info("Ratio of p_zones steps: %s", self.config['mcmc']['P_ZONES_STEPS'])
        logging.info("Ratio of p_families steps: %s", self.config['mcmc']['P_FAMILIES_STEPS'])
        
    def sampling_setup(self, inheritance_value):
        self.config['mcmc']['SAMPLE_P']['p_families'] = inheritance_value

        # If p_global, p_zones and p_families are True -> use the values from the config.json
        # If they are False -> assign 0
        if not self.config['mcmc']['SAMPLE_P']['p_global']:
            self.config['mcmc']['P_GLOBAL_STEPS'] = 0
        if not self.config['mcmc']['SAMPLE_P']['p_zones']:
            self.config['mcmc']['P_ZONES_STEPS'] = 0
        if self.config['mcmc']['SAMPLE_P']['p_families']:
            self.config['mcmc']['P_FAMILIES_STEPS'] = 0

        # Assign WEIGHT_STEPS
        self.config['mcmc']['WEIGHT_STEPS'] = \
            1 - (self.config['mcmc']['ZONE_STEPS'] +
                 self.config['mcmc']['P_GLOBAL_STEPS'] +
                 self.config['mcmc']['P_FAMILIES_STEPS'] +
                 self.config['mcmc']['P_ZONES_STEPS'])

        # Assign ALL_STEPS
        self.config['mcmc']['ALL_STEPS'] = \
            self.config['mcmc']['ZONE_STEPS'] + \
            self.config['mcmc']['WEIGHT_STEPS'] + \
            self.config['mcmc']['P_GLOBAL_STEPS'] + \
            self.config['mcmc']['P_ZONES_STEPS'] + \
            self.config['mcmc']['P_FAMILIES_STEPS']

        if self.config['mcmc']['ALL_STEPS'] != 1:
            print("steps must add to 1.")

        # Assign operators
        ops = {'shrink_zone': self.config['mcmc']['ZONE_STEPS'] / 4,
               'grow_zone': self.config['mcmc']['ZONE_STEPS'] / 4,
               'swap_zone': self.config['mcmc']['ZONE_STEPS'] / 2,
               'alter_weights': self.config['mcmc']['WEIGHT_STEPS'],
               'alter_p_global': self.config['mcmc']['P_GLOBAL_STEPS'],
               'alter_p_zones': self.config['mcmc']['P_ZONES_STEPS'],
               'alter_p_families': self.config['mcmc']['P_FAMILIES_STEPS']}

        return ops

    def sampling_run(self, inheritance_value, ops):
        # Sampling
        # Initial sample is empty

        # initial_sample on line 227 sometimes depends on previous results of the MCMC (any changes required now?)
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
                                               sample_p=self.config['mcmc']['SAMPLE_P'],
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
        cur_run_stats['true_zones'] = self.zones_sim
        cur_run_stats['true_weights'] = cur_weights_sim_normed
        cur_run_stats['true_p_global'] = cur_p_global_sim_padded
        cur_run_stats['true_p_zones'] = self.p_zones_sim
        cur_run_stats['true_p_families'] = self.p_families_sim
        cur_run_stats['true_ll'] = cur_zone_sampler.likelihood(cur_true_sample, 0)
        cur_run_stats['true_prior'] = cur_zone_sampler.prior(cur_true_sample, 0)
        cur_run_stats["true_families"] = self.families_sim

        # Save stats to file
        path = self.TEST_SAMPLING_RESULTS_PATH.format(i=int(inheritance_value), run=cur_run)
        dump(cur_run_stats, path)
