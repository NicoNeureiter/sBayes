#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the MCMC process """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import os

from sbayes.postprocessing import (contribution_per_area, log_operator_statistics,
                                   log_operator_statistics_header, match_areas, rank_areas)
from sbayes.sampling.zone_sampling import Sample, ZoneMCMCGenerative, IndexSet
from sbayes.util import (normalize, counts_to_dirichlet,
                         inheritance_counts_to_dirichlet, samples2file)


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
        prior_config = self.config['model']['PRIOR']

        # geo prior
        if prior_config['geo'] == 'uniform':
            prior_config['geo'] = {'type': 'uniform'}
        else:
            raise ValueError('Currently only uniform geo-prior available.')
        # weights
        if prior_config['weights'] == 'uniform':
            prior_config['weights'] = {'type': 'uniform'}
        else:
            raise ValueError('Currently only uniform prior_weights are supported.')

        # universal pressure
        if prior_config['universal'] == 'uniform':
            prior_config['universal'] = {'type': 'uniform'}

        elif prior_config['universal'] == 'simulated_counts':
            dirichlet = counts_to_dirichlet(self.data.prior_universal['counts'],
                                            self.data.prior_universal['states'])
            prior_config['universal'] = {'type': 'counts',
                                         'dirichlet': dirichlet,
                                         'states': self.data.prior_universal['states']}

        elif prior_config['universal'] == 'counts':
            dirichlet = counts_to_dirichlet(self.data.prior_universal['counts'],
                                            self.data.prior_universal['states'])
            prior_config['universal'] = {'type': 'counts',
                                         'dirichlet': dirichlet,
                                         'states': self.data.prior_universal['states']}
        else:
            raise ValueError('Prior for universal must be uniform or counts.')

        # inheritance
        if prior_config['inheritance'] == 'uniform':
            prior_config['inheritance'] = {'type': 'uniform'}

        elif prior_config['inheritance'] is None:
            prior_config['inheritance'] = {'type': None}

        elif prior_config['inheritance'] == 'universal':
            prior_config['inheritance'] = {'type': 'universal',
                                           'strength': 10,
                                           'states': self.data.prior_inheritance['states']}

        elif prior_config['inheritance'] == 'counts':
            dirichlet = inheritance_counts_to_dirichlet(self.data.prior_inheritance['counts'],
                                                        self.data.prior_inheritance['states'])
            prior_config['inheritance'] = {'type': 'counts',
                                           'dirichlet': dirichlet,
                                           'states': self.data.prior_inheritance['states']}

        elif prior_config['inheritance'] == 'counts_and_universal':
            prior_config['inheritance'] = {'type': 'counts_and_universal',
                                           'counts': self.data.prior_inheritance['counts'],
                                           'strength': self.config['model']['UNIVERSAL_PRIOR_STRENGTH'],
                                           'states': self.data.prior_inheritance['states']}
        else:
            raise ValueError('Prior for inheritance must be uniform, counts or  counts_and_universal')

        # contact
        if prior_config['contact'] == 'uniform':
            prior_config['contact'] = {'type': 'uniform'}

        elif prior_config['contact'] == 'universal':
            prior_config['contact'] = {'type': 'universal',
                                       'strength': 10,
                                       'states': self.data.prior_inheritance['states']}
        else:
            raise ValueError('Prior for contact must be uniform or universal.')

    def log_setup(self):
        mcmc_config = self.config['mcmc']
        model_config = self.config['model']

        logging.basicConfig(format='%(message)s', filename=self.path_log, level=logging.DEBUG)

        logging.info('\n')

        logging.info("Model")
        logging.info("##########################################")
        logging.info("Number of inferred areas: %i", model_config['N_AREAS'])
        logging.info("Areas have a minimum size of %s and a maximum size of %s.",
                     model_config['MIN_M'], model_config['MAX_M'])
        logging.info("Inheritance is considered for inference: %s",
                     model_config['INHERITANCE'])

        logging.info("Geo-prior: %s ", model_config['PRIOR']['geo']['type'])
        logging.info("Prior on weights: %s ", model_config['PRIOR']['weights']['type'])
        logging.info("Prior on universal pressure (alpha): %s ", model_config['PRIOR']['universal']['type'])

        if model_config['PRIOR']['inheritance']['type'] is not None:

            logging.info("Prior on inheritance (beta): %s ", model_config['PRIOR']['inheritance']['type'])
        logging.info("Prior on contact (gamma): %s ", model_config['PRIOR']['contact']['type'])
        logging.info('\n')

        logging.info("MCMC SETUP")
        logging.info("##########################################")

        logging.info("MCMC with %s steps and %s samples (burn-in %s steps)",
                     mcmc_config['N_STEPS'], mcmc_config['N_SAMPLES'], mcmc_config['BURN_IN'])
        logging.info("MCMC with %s chains and %s attempted swap(s) after %s steps.",
                     mcmc_config['N_CHAINS'], mcmc_config['N_SWAPS'], mcmc_config['SWAP_PERIOD'])
        logging.info("Pseudocounts for tuning the width of the proposal distribution for weights: %s ",
                     mcmc_config['PROPOSAL_PRECISION']['weights'])
        logging.info("Pseudocounts for tuning the width of the proposal distribution for "
                     "universal pressure (alpha): %s ",
                     mcmc_config['PROPOSAL_PRECISION']['universal'])
        if mcmc_config['PROPOSAL_PRECISION']['inheritance'] is not None:
            logging.info("Pseudocounts for tuning the width of the proposal distribution for inheritance (beta): %s ",
                         mcmc_config['PROPOSAL_PRECISION']['inheritance'])
        logging.info("Pseudocounts for tuning the width of the proposal distribution for areas (gamma): %s ",
                     mcmc_config['PROPOSAL_PRECISION']['contact'])

        logging.info("Ratio of areal steps (growing, shrinking, swapping areas): %s",
                     mcmc_config['STEPS']['area'])
        logging.info("Ratio of weight steps (changing weights): %s", mcmc_config['STEPS']['weights'])
        logging.info("Ratio of universal steps (changing alpha) : %s", mcmc_config['STEPS']['universal'])
        logging.info("Ratio of inheritance steps (changing beta): %s", mcmc_config['STEPS']['inheritance'])
        logging.info("Ratio of contact steps (changing gamma): %s", mcmc_config['STEPS']['contact'])

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

    def sample(self, initial_sample=None, lh_per_area=True):
        initial_sample = self.initialize_sample(initial_sample)

        self.sampler = ZoneMCMCGenerative(network=self.data.network, features=self.data.features,
                                          min_size=self.config['model']['MIN_M'],
                                          max_size=self.config['model']['MAX_M'],
                                          n_zones=self.config['model']['N_AREAS'],
                                          prior=self.config['model']['PRIOR'],
                                          inheritance=self.config['model']['INHERITANCE'],
                                          n_chains=self.config['mcmc']['N_CHAINS'],
                                          initial_sample=initial_sample,
                                          swap_period=self.config['mcmc']['SWAP_PERIOD'],
                                          operators=self.ops, families=self.data.families,
                                          chain_swaps=self.config['mcmc']['N_SWAPS'],
                                          var_proposal=self.config['mcmc']['PROPOSAL_PRECISION'],
                                          p_grow_connected=self.config['mcmc']['P_GROW_CONNECTED'],
                                          sample_from_prior=self.config['model']['SAMPLE_FROM_PRIOR'])

        self.sampler.generate_samples(self.config['mcmc']['N_STEPS'],
                                      self.config['mcmc']['N_SAMPLES'],
                                      self.config['mcmc']['BURN_IN'])

        # Evaluate likelihood and prior for each zone alone (makes it possible to rank zones)
        if lh_per_area:
            self.sampler = contribution_per_area(self.sampler)

        self.samples = self.sampler.statistics

    def log_statistics(self):
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

        initial_sample.everything_changed()

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

        ground_truth_sample.everything_changed()

        self.samples['true_zones'] = self.data.areas
        self.samples['true_weights'] = weights
        self.samples['true_p_global'] = self.data.p_universal[np.newaxis, ...]
        self.samples['true_p_zones'] = self.data.p_contact
        self.samples['true_p_families'] = self.data.p_inheritance
        self.samples['true_ll'] = self.sampler.likelihood(ground_truth_sample, 0)
        self.samples['true_prior'] = self.sampler.prior(ground_truth_sample, 0)
        self.samples["true_families"] = self.data.families

    def save_samples(self, run=1):

        self.samples = match_areas(self.samples)
        self.samples = rank_areas(self.samples)

        file_info = self.config['results']['FILE_INFO']

        if file_info == "n":
            fi = 'n{n}'.format(n=self.config['model']['N_AREAS'])

        elif file_info == "s":
            fi = 's{s}_a{a}'.format(s=self.config['simulation']['STRENGTH'],
                                    a=self.config['simulation']['AREA'])

        elif file_info == "i":
            fi = 'i{i}'.format(i=int(self.config['model']['INHERITANCE']))

        elif file_info == "p":
            p = 0 if self.config['model']['PRIOR']['universal']['type'] == "uniform" else 1
            fi = 'p{p}'.format(p=p)

        else:
            raise ValueError("file_info must be 'n', 's', 'i' or 'p'")

        run = '_{run}'.format(run=run)
        pth = self.path_results + fi + '/'
        ext = '.txt'
        gt_pth = pth + 'ground_truth/'

        paths = {'parameters': pth + 'stats_' + fi + run + ext,
                 'areas': pth + 'areas_' + fi + run + ext,
                 'gt': gt_pth + 'stats' + ext,
                 'gt_areas': gt_pth + 'areas' + ext}

        if not os.path.exists(pth):
            os.makedirs(pth)

        if self.data.is_simulated:
            self.eval_ground_truth()
            if not os.path.exists(gt_pth):
                os.makedirs(gt_pth)

        samples2file(self.samples, self.data, self.config, paths)
