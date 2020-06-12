#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the MCMC process """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np

from src.postprocessing import contribution_per_area, log_operator_statistics, log_operator_statistics_header
from src.sampling.zone_sampling import Sample, ZoneMCMC_generative
from src.util import dump, normalize, universal_counts_to_dirichlet, inheritance_counts_to_dirichlet


class MCMC:
    def __init__(self, data, experiment):

        # Retrieve the data
        self.data = data

        # Retrieve the configurations
        self.config = experiment.config

        # Paths
        self.path_log = experiment.path_results + 'experiment.log'
        self.path_results = experiment.path_results

        # Assign steps to operators
        self.ops = {}
        self.steps_per_operator()

        # Define prior distributions for all parameters
        self.define_priors()

        # Samples
        self.sampler = None
        self.samples = None

    def define_priors(self):
        # geo prior
        if self.config['mcmc']['PRIOR']['geo'] == "uniform":
            self.config['mcmc']['PRIOR']['geo'] = {'type': "uniform"}
        # weights
        if self.config['mcmc']['PRIOR']['weights'] == "uniform":
            self.config['mcmc']['PRIOR']['weights'] = {'type': "uniform"}
        if self.config['mcmc']['PRIOR']['contact'] == "uniform":
            self.config['mcmc']['PRIOR']['contact'] = {'type': "uniform"}

        # universal pressure
        if self.config['mcmc']['PRIOR']['universal'] == "uniform":
            self.config['mcmc']['PRIOR']['universal'] = {'type': "uniform"}

        elif self.config['mcmc']['PRIOR']['universal'] == "from_simulated_counts":
            dirichlet = universal_counts_to_dirichlet(self.data.prior_universal['counts'],
                                                      self.data.prior_universal['states'])
            self.config['mcmc']['PRIOR']['universal'] = {'type': "pseudocounts",
                                                         'dirichlet': dirichlet,
                                                         'states': self.data.prior_universal['states']}

        elif self.config['mcmc']['PRIOR']['universal'] == "from_counts":
            dirichlet = universal_counts_to_dirichlet(self.data.prior_universal['counts'],
                                                      self.data.prior_universal['states'])
            self.config['mcmc']['PRIOR']['universal'] = {'type': "pseudocounts",
                                                         'dirichlet': dirichlet,
                                                         'states': self.data.prior_universal['states']}

        # inheritance
        if self.config['mcmc']['PRIOR']['inheritance'] == "uniform":
            self.config['mcmc']['PRIOR']['inheritance'] = {'type': "uniform"}

        elif self.config['mcmc']['PRIOR']['inheritance'] is None:
            self.config['mcmc']['PRIOR']['inheritance'] = {'type': None}

        elif self.config['mcmc']['PRIOR']['inheritance'] == "from_counts":
            dirichlet = inheritance_counts_to_dirichlet(self.data.prior_inheritance['counts'],
                                                        self.data.prior_inheritance['states'])
            self.config['mcmc']['PRIOR']['inheritance'] = {'type': "pseudocounts",
                                                           'dirichlet': dirichlet,
                                                           'states': self.data.prior_inheritance['states']}
        print(self.config['mcmc']['PRIOR']['inheritance'])

    def log_setup(self):

        logging.basicConfig(format='%(message)s', filename=self.path_log, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("\n")
        logging.info("MCMC SETUP")
        logging.info("##########################################")

        logging.info("MCMC with %s steps and %s samples (burn-in %s steps)",
                     self.config['mcmc']['N_STEPS'], self.config['mcmc']['N_SAMPLES'],
                     self.config['mcmc']['BURN_IN'])

        logging.info("Areas have a minimum size of %s and a maximum size of %s.",
                     self.config['mcmc']['MIN_M'], self.config['mcmc']['MAX_M'])
        logging.info("MCMC with %s chains and %s attempted swap(s) after %s steps.",
                     self.config['mcmc']['N_CHAINS'], self.config['mcmc']['N_SWAPS'],
                     self.config['mcmc']['SWAP_PERIOD'])

        logging.info("Number of sampled areas: %i", self.config['mcmc']['N_AREAS'])
        logging.info("Geo-prior: %s ", self.config['mcmc']['PRIOR']['geo']['type'])
        logging.info("Prior on weights: %s ", self.config['mcmc']['PRIOR']['weights']['type'])
        logging.info("Prior on universal pressure (alpha): %s ", self.config['mcmc']['PRIOR']['universal']['type'])
        if self.config['mcmc']['PRIOR']['inheritance']['type'] is not None:
            logging.info("Prior on inheritance (beta): %s ", self.config['mcmc']['PRIOR']['inheritance']['type'])
        logging.info("Prior on contact (gamma): %s ", self.config['mcmc']['PRIOR']['contact']['type'])
        logging.info("Pseudocounts for tuning the width of the proposal distribution for weights: %s ",
                     self.config['mcmc']['PROPOSAL_PRECISION']['weights'])
        logging.info("Pseudocounts for tuning the width of the proposal distribution for "
                     "universal pressure (alpha): %s ",
                     self.config['mcmc']['PROPOSAL_PRECISION']['universal'])
        if self.config['mcmc']['PROPOSAL_PRECISION']['inheritance'] is not None:
            logging.info("Pseudocounts for tuning the width of the proposal distribution for inheritance (beta): %s ",
                         self.config['mcmc']['PROPOSAL_PRECISION']['inheritance'])
        logging.info("Pseudocounts for tuning the width of the proposal distribution for areas (gamma): %s ",
                     self.config['mcmc']['PROPOSAL_PRECISION']['contact'])
        logging.info("Inheritance is considered for inference: %s",
                     self.config['mcmc']['INHERITANCE'])

        logging.info("Ratio of areal steps (growing, shrinking, swapping areas): %s",
                     self.config['mcmc']['STEPS']['area'])
        logging.info("Ratio of weight steps (changing weights): %s", self.config['mcmc']['STEPS']['weights'])
        logging.info("Ratio of universal steps (changing alpha) : %s", self.config['mcmc']['STEPS']['universal'])
        logging.info("Ratio of inheritance steps (changing beta): %s", self.config['mcmc']['STEPS']['inheritance'])
        logging.info("Ratio of contact steps (changing gamma): %s", self.config['mcmc']['STEPS']['contact'])

    def steps_per_operator(self):
        # Assign steps per operator
        ops = {'shrink_zone': self.config['mcmc']['STEPS']['area'] / 4,
               'grow_zone': self.config['mcmc']['STEPS']['area'] / 4,
               'swap_zone': self.config['mcmc']['STEPS']['area'] / 2,
               'alter_weights': self.config['mcmc']['STEPS']['weights'],
               'alter_p_global': self.config['mcmc']['STEPS']['universal'],
               'alter_p_zones': self.config['mcmc']['STEPS']['contact'],
               'alter_p_families': self.config['mcmc']['STEPS']['inheritance']}
        self.ops = ops

    def sample(self, initial_sample=None, lh_per_area=False):
        initial_sample = self.initialize_sample(initial_sample)

        self.sampler = ZoneMCMC_generative(network=self.data.network, features=self.data.features,
                                           min_size=self.config['mcmc']['MIN_M'],
                                           max_size=self.config['mcmc']['MAX_M'],
                                           n_zones=self.config['mcmc']['N_AREAS'],
                                           n_chains=self.config['mcmc']['N_CHAINS'],
                                           initial_sample=initial_sample,
                                           swap_period=self.config['mcmc']['SWAP_PERIOD'],
                                           operators=self.ops, families=self.data.families,
                                           chain_swaps=self.config['mcmc']['N_SWAPS'],
                                           inheritance=self.config['mcmc']['INHERITANCE'],
                                           prior=self.config['mcmc']['PRIOR'],
                                           var_proposal=self.config['mcmc']['PROPOSAL_PRECISION'])

        self.sampler.generate_samples(self.config['mcmc']['N_STEPS'],
                                      self.config['mcmc']['N_SAMPLES'],
                                      self.config['mcmc']['BURN_IN'])

        # Evaluate likelihood and prior for each zone alone (makes it possible to rank zones)
        if lh_per_area:
            self.sampler = contribution_per_area(self.sampler)

        self.samples = self.sampler.statistics

    def log_statistics(self):
        logging.basicConfig(format='%(message)s', filename=self.path_log, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("\n")
        logging.info("MCMC STATISTICS")
        logging.info("##########################################")
        logging.info(log_operator_statistics_header())
        for op_name in self.ops:
            logging.info(log_operator_statistics(op_name, self.samples))

    @staticmethod
    def initialize_sample(initial_sample):
        if initial_sample is None:
            initial_sample = Sample(zones=None, weights=None,
                                    p_global=None, p_zones=None, p_families=None)

        initial_sample.what_changed = {'lh': {'zones': True, 'weights': True,
                                              'p_global': True, 'p_zones': True, 'p_families': True},
                                       'prior': {'zones': True, 'weights': True,
                                                 'p_global': True, 'p_zones': True, 'p_families': True}}

        return initial_sample

    def eval_ground_truth(self):
        # Evaluate the likelihood of the true sample in simulated data
        # If the model includes inheritance use all weights, if not use only the first two weights (global, zone)
        if self.config['mcmc']['INHERITANCE']:
            weights = self.data.weights.copy()
        else:
            weights = normalize(self.data.weights[:, :2])

        ground_truth_sample = Sample(zones=self.data.areas,
                                     weights=weights,
                                     p_global=self.data.p_universal[np.newaxis, ...],
                                     p_zones=self.data.p_contact,
                                     p_families=self.data.p_inheritance)

        ground_truth_sample.what_changed = {'lh': {'zones': True,
                                                   'weights': True,
                                                   'p_global': True,
                                                   'p_zones': True,
                                                   'p_families': True},
                                            'prior': {'zones': True,
                                                      'weights': True,
                                                      'p_global': True,
                                                      'p_zones': True,
                                                      'p_families': True}}

        self.samples['true_zones'] = self.data.areas
        self.samples['true_weights'] = weights
        self.samples['true_p_global'] = self.data.p_universal[np.newaxis, ...]
        self.samples['true_p_zones'] = self.data.p_contact
        self.samples['true_p_families'] = self.data.p_inheritance
        self.samples['true_ll'] = self.sampler.likelihood(ground_truth_sample, 0)
        self.samples['true_prior'] = self.sampler.prior(ground_truth_sample, 0)
        self.samples["true_families"] = self.data.families

    def save_samples(self, run=1):
        file_info = self.config['results']['FILE_INFO']

        if file_info == "n":
            outfile = self.path_results + 'contact_areas_n{n}_{run}.pkl'
            path = outfile.format(n=self.config['mcmc']['N_AREAS'], run=run)

        elif file_info == "s":
            outfile = self.path_results + 'contact_areas_s{s}_a{a}_{run}.pkl'
            path = outfile.format(s=self.config['simulation']['STRENGTH'],
                                  a=self.config['simulation']['AREA'], run=run)

        elif file_info == "i":
            outfile = self.path_results + 'contact_areas_i{i}_{run}.pkl'
            path = outfile.format(i=int(self.config['mcmc']['INHERITANCE']), run=run)

        elif file_info == "p":
            outfile = self.path_results + 'contact_areas_p{p}_{run}.pkl'
            p = 0 if self.config['mcmc']['PRIOR']['universal']['type'] == "uniform" else 1
            path = outfile.format(p=p, run=run)
        else:
            raise ValueError("file_info must be 'n', 's', 'i' or 'p'")
        dump(self.samples, path)
