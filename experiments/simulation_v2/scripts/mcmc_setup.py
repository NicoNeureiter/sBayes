#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the MCMC process """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import numpy as np

from src.postprocessing import contribution_per_zone, print_operator_statistics, print_operator_statistics_header
from src.preprocessing import estimate_geo_prior_parameters
from src.sampling.zone_sampling import Sample, ZoneMCMC_generative
from src.util import dump, normalize


class MCMCSetup:
    def __init__(self, log_path, results_path, is_real, network, features, families):
        # Parameters from the simulation or real data
        self.network = network
        self.features = features
        self.families = families

        self.TEST_SAMPLING_RESULTS_PATH = results_path
        self.TEST_SAMPLING_LOG_PATH = log_path

        # Get parameters from config_mcmc.json
        self.config = {}
        self.get_parameters()

        # Flag for switching between real/simulated data
        self.is_real = is_real

    def get_parameters(self):
        with open('config_mcmc.json', 'r') as f:
            self.config = json.load(f)

        self.config['mcmc']['PRIOR']['geo']['parameters'] = \
            estimate_geo_prior_parameters(self.network, self.config['mcmc']['PRIOR']['geo']['type'])

    # This is used for South America
    def set_prior_parameters(self, p_global_dirichlet, p_global_categories,
                             p_families_dirichlet, p_families_categories):
        self.config['mcmc']['PRIOR']['p_global']['parameters']['dirichlet'] = p_global_dirichlet
        self.config['mcmc']['PRIOR']['p_global']['parameters']['categories'] = p_global_categories
        self.config['mcmc']['PRIOR']['p_families']['parameters']['dirichlet'] = p_families_dirichlet
        self.config['mcmc']['PRIOR']['p_families']['parameters']['categories'] = p_families_categories

    def logging_mcmc(self):

        if self.config['mcmc']['N_STEPS_INCREASE'] == 0:
            logging.info("MCMC with %s steps and %s samples (burn-in %s steps)",
                         self.config['mcmc']['N_STEPS'], self.config['mcmc']['N_SAMPLES'],
                         self.config['mcmc']['BURN_IN'])
        else:
            logging.info("MCMC with %s steps (+ %s increase per extra zone) and %s samples (burn-in %s steps)",
                         self.config['mcmc']['N_STEPS'], self.config['mcmc']['N_SAMPLES'],
                         self.config['mcmc']['N_STEPS_INCREASE'], self.config['mcmc']['BURN_IN'])

        logging.info("Zones have a minimum size of %s and a maximum size of %s. The initial size is %s.",
                     self.config['mcmc']['MIN_SIZE'], self.config['mcmc']['MAX_SIZE'],
                     self.config['mcmc']['INITIAL_SIZE'])
        logging.info("MCMC with %s chains and %s attempted swap(s) after %s steps.",
                     self.config['mcmc']['N_CHAINS'], self.config['mcmc']['N_SWAPS'],
                     self.config['mcmc']['SWAP_PERIOD'])

        # Just number or maximum number?
        logging.info("Number of sampled zones: %i", self.config['mcmc']['N_ZONES'])
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
        if not self.config['mcmc']['SAMPLE_P']['p_families']:
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

    def sampling_run(self, inheritance_value, ops, initial_sample):
        # Sampling
        cur_zone_sampler = ZoneMCMC_generative(network=self.network, features=self.features,
                                               min_size=self.config['mcmc']['MIN_SIZE'],
                                               max_size=self.config['mcmc']['MAX_SIZE'],
                                               initial_size=self.config['mcmc']['INITIAL_SIZE'],
                                               n_zones=self.config['mcmc']['N_ZONES'],
                                               connected_only=self.config['mcmc']['CONNECTED_ONLY'],
                                               n_chains=self.config['mcmc']['N_CHAINS'],
                                               initial_sample=initial_sample,
                                               swap_period=self.config['mcmc']['SWAP_PERIOD'],
                                               operators=ops, families=self.families,
                                               chain_swaps=self.config['mcmc']['N_SWAPS'],
                                               inheritance=inheritance_value,
                                               prior=self.config['mcmc']['PRIOR'],
                                               sample_p=self.config['mcmc']['SAMPLE_P'],
                                               var_proposal=self.config['mcmc']['VAR_PROPOSAL'])

        cur_zone_sampler.generate_samples(self.config['mcmc']['N_STEPS'],
                                          self.config['mcmc']['N_SAMPLES'],
                                          self.config['mcmc']['BURN_IN'])

        # Evaluate likelihood and prior for each zone alone (makes it possible to rank zones)
        if self.is_real:
            cur_zone_sampler = contribution_per_zone(cur_zone_sampler)

        return cur_zone_sampler

    @staticmethod
    def initialize_sample(init_zones, init_weights, init_p_global,
                          init_p_zones, init_p_families):
        initial_sample = Sample(zones=init_zones, weights=init_weights,
                                p_global=init_p_global, p_zones=init_p_zones, p_families=init_p_families)
        initial_sample.what_changed = {'lh': {'zones': True, 'weights': True,
                                              'p_global': True, 'p_zones': True, 'p_families': True},
                                       'prior': {'zones': True, 'weights': True,
                                                 'p_global': True, 'p_zones': True, 'p_families': True}}

        return initial_sample

    @staticmethod
    def print_op_stats(cur_run_stats, ops):
        # Print operator stats
        print('')
        print_operator_statistics_header()
        for op_name in ops:
            print_operator_statistics(op_name, cur_run_stats)

    @staticmethod
    def true_sample_eval(inheritance_value, zones, p_zones, weights, p_global, p_families):
        # Evaluate the likelihood of the true sample
        # If the model includes inheritance use all weights, if not use only the first two weights (global, zone)
        if inheritance_value:
            cur_weights_normed = weights.copy()
        else:
            cur_weights_normed = normalize(weights[:, :2])

        cur_p_global_padded = p_global[np.newaxis, ...]

        cur_true_sample = Sample(zones=zones,
                                 weights=cur_weights_normed,
                                 p_global=cur_p_global_padded,
                                 p_zones=p_zones,
                                 p_families=p_families)

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

        return cur_true_sample, cur_weights_normed, cur_p_global_padded

    def true_sample_stats(self, cur_zone_sampler, cur_true_sample, cur_run_stats,
                          cur_weights_normed, cur_p_global_padded, zones, p_zones, p_families):
        # Save stats about true sample
        cur_run_stats['true_zones'] = zones
        cur_run_stats['true_weights'] = cur_weights_normed
        cur_run_stats['true_p_global'] = cur_p_global_padded
        cur_run_stats['true_p_zones'] = p_zones
        cur_run_stats['true_p_families'] = p_families
        cur_run_stats['true_ll'] = cur_zone_sampler.likelihood(cur_true_sample, 0)
        cur_run_stats['true_prior'] = cur_zone_sampler.prior(cur_true_sample, 0)
        cur_run_stats["true_families"] = self.families

        return cur_run_stats

    def save_stats(self, cur_run, cur_run_stats,
                   n=0, inheritance_value=True):
        # Save stats to file
        if self.is_real:
            path = self.TEST_SAMPLING_RESULTS_PATH.format(nz=n, run=cur_run)
        else:
            path = self.TEST_SAMPLING_RESULTS_PATH.format(i=int(inheritance_value), run=cur_run)
        dump(cur_run_stats, path)
