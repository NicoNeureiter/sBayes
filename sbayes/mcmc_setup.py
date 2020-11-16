#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the MCMC process """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import os
import random

from sbayes.postprocessing import (contribution_per_area, log_operator_statistics,
                                   log_operator_statistics_header, match_areas, rank_areas)
from sbayes.sampling.zone_sampling import Sample, ZoneMCMCGenerative, ZoneMCMCWarmup
from sbayes.util import (normalize, counts_to_dirichlet,
                         inheritance_counts_to_dirichlet, samples2file, scale_counts, get_max_size_list)


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
        self.prior_structured = None
        self.define_priors()

        # Samples
        self.sampler = None
        self.samples = None
        self.sample_from_warm_up = None

    def define_priors(self):
        self.prior_structured = dict.fromkeys(self.config['model']['PRIOR'])

        # geo prior
        if self.config['model']['PRIOR']['geo']['type'] == 'uniform':
            self.prior_structured['geo'] = {'type': 'uniform'}
        elif self.config['model']['PRIOR']['geo']['type'] == 'cost_based':
            # todo:  change prior if cost matrix is provided
            # todo: move config['model']['scale_geo_prior'] to config['model']['PRIOR']['geo']['scale']
            self.prior_structured['geo'] = {'type': 'cost_based',
                                            'scale': self.config['model']['scale_geo_prior']}
        else:
            raise ValueError('Geo prior not supported')
        # weights
        if self.config['model']['PRIOR']['weights']['type'] == 'uniform':
            self.prior_structured['weights'] = {'type': 'uniform'}
        else:
            raise ValueError('Currently only uniform prior_weights are supported.')

        # universal preference
        cfg_universal = self.config['model']['PRIOR']['universal']
        if cfg_universal['type'] == 'uniform':
            self.prior_structured['universal'] = {'type': 'uniform'}

        elif cfg_universal['type'] == 'simulated_counts':
            if cfg_universal['scale_counts'] is not None:
                self.data.prior_universal['counts'] = scale_counts(counts=self.data.prior_universal['counts'],
                                                                   scale_to=cfg_universal['scale_counts'])

            dirichlet = counts_to_dirichlet(self.data.prior_universal['counts'],
                                            self.data.prior_universal['states'])
            self.prior_structured['universal'] = {'type': 'counts',
                                                  'dirichlet': dirichlet,
                                                  'states': self.data.prior_universal['states']}

        elif cfg_universal['type'] == 'counts':
            if cfg_universal['scale_counts'] is not None:
                self.data.prior_universal['counts'] = scale_counts(counts=self.data.prior_universal['counts'],
                                                                   scale_to=cfg_universal['scale_counts'])

            dirichlet = counts_to_dirichlet(self.data.prior_universal['counts'],
                                            self.data.prior_universal['states'])
            self.prior_structured['universal'] = {'type': 'counts',
                                                  'dirichlet': dirichlet,
                                                  'states': self.data.prior_universal['states']}
        else:
            raise ValueError('Prior for universal must be uniform or counts.')

        # inheritance
        cfg_inheritance = self.config['model']['PRIOR']['inheritance']
        if self.config['model']['INHERITANCE']:
            if cfg_inheritance['type'] == 'uniform':
                self.prior_structured['inheritance'] = {'type': 'uniform'}

            elif cfg_inheritance['type'] is None:
                self.prior_structured['inheritance'] = {'type': None}

            elif cfg_inheritance['type'] == 'universal':
                self.prior_structured['inheritance'] = {'type': 'universal',
                                                        'strength': cfg_inheritance['scale_counts'],
                                                        'states': self.data.state_names['internal']}

            elif cfg_inheritance['type'] == 'counts':
                if cfg_inheritance['scale_counts'] is not None:
                    self.data.prior_inheritance['counts'] = scale_counts(
                        counts=self.data.prior_inheritance['counts'],
                        scale_to=cfg_inheritance['scale_counts'],
                        prior_inheritance=True
                    )
                dirichlet = inheritance_counts_to_dirichlet(self.data.prior_inheritance['counts'],
                                                            self.data.prior_inheritance['states'])
                self.prior_structured['inheritance'] = {'type': 'counts',
                                                        'dirichlet': dirichlet,
                                                        'states': self.data.prior_inheritance['states']}

            elif cfg_inheritance['type'] == 'counts_and_universal':
                if cfg_inheritance['scale_counts'] is not None:
                    self.data.prior_inheritance['counts'] = scale_counts(
                        counts=self.data.prior_inheritance['counts'],
                        scale_to=cfg_inheritance['scale_counts'],
                        prior_inheritance=True
                    )
                self.prior_structured['inheritance'] = {'type': 'counts_and_universal',
                                                        'counts': self.data.prior_inheritance['counts'],
                                                        'strength': cfg_inheritance['scale_counts'],
                                                        'states': self.data.prior_inheritance['states']}
            else:
                raise ValueError('Prior for inheritance must be uniform, counts or  counts_and_universal')
        else:
            self.prior_structured['inheritance'] = None

        # contact
        cfg_contact = self.config['model']['PRIOR']['contact']
        if cfg_contact['type'] == 'uniform':
            self.prior_structured['contact'] = {'type': 'uniform'}

        elif cfg_contact['type'] == 'universal':
            self.prior_structured['contact'] = {'type': 'universal',
                                                'strength': cfg_contact['scale_counts'],
                                                'states': self.data.state_names['internal']}
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

        logging.info("Geo-prior: %s ", self.prior_structured['geo']['type'])
        logging.info("Prior on weights: %s ", self.prior_structured['weights']['type'])
        logging.info("Prior on universal pressure (alpha): %s ", self.prior_structured['universal']['type'])
        if self.config['model']['INHERITANCE']:
            logging.info("Prior on inheritance (beta): %s ", self.prior_structured['inheritance']['type'])

        logging.info("Prior on contact (gamma): %s ", self.prior_structured['contact']['type'])
        logging.info('\n')

        logging.info("MCMC SETUP")
        logging.info("##########################################")

        logging.info("MCMC with %s steps and %s samples",
                     mcmc_config['N_STEPS'], mcmc_config['N_SAMPLES'])
        logging.info("Warm-up: %s chains exploring the parameter space in %s steps",
                     mcmc_config['WARM_UP']['N_WARM_UP_CHAINS'],  mcmc_config['WARM_UP']['N_WARM_UP_STEPS'])
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

    def sample(self, lh_per_area=True):

        if self.sample_from_warm_up is None:
            initial_sample = self.empty_sample()
        else:
            initial_sample = self.sample_from_warm_up

        self.sampler = ZoneMCMCGenerative(network=self.data.network, features=self.data.features,
                                          inheritance=self.config['model']['INHERITANCE'],
                                          prior=self.prior_structured,
                                          n_zones=self.config['model']['N_AREAS'],
                                          n_chains=self.config['mcmc']['N_CHAINS'],
                                          min_size=self.config['model']['MIN_M'],
                                          max_size=self.config['model']['MAX_M'],
                                          initial_sample=initial_sample,
                                          operators=self.ops, families=self.data.families,
                                          var_proposal=self.config['mcmc']['PROPOSAL_PRECISION'],
                                          p_grow_connected=self.config['mcmc']['P_GROW_CONNECTED'],
                                          initial_size=self.config['mcmc']['M_INITIAL'])

        self.sampler.generate_samples(self.config['mcmc']['N_STEPS'],
                                      self.config['mcmc']['N_SAMPLES'])

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
    def empty_sample():
        initial_sample = Sample(zones=None, weights=None,
                                p_global=None, p_zones=None, p_families=None)
        initial_sample.everything_changed()

        return initial_sample

    def eval_ground_truth(self, lh_per_area=True):
        # Evaluate the likelihood of the true sample in simulated data
        # If the model includes inheritance use all weights, if not use only the first two weights (global, zone)
        if self.config['model']['INHERITANCE']:
            weights = self.data.weights.copy()
        else:
            weights = normalize(self.data.weights[:, :2])

        ground_truth = Sample(zones=self.data.areas,
                              weights=weights,
                              p_global=self.data.p_universal[np.newaxis, ...],
                              p_zones=self.data.p_contact,
                              p_families=self.data.p_inheritance)

        ground_truth.everything_changed()

        self.samples['true_zones'] = self.data.areas
        self.samples['true_weights'] = weights
        self.samples['true_p_global'] = self.data.p_universal[np.newaxis, ...]
        self.samples['true_p_zones'] = self.data.p_contact
        self.samples['true_p_families'] = self.data.p_inheritance
        self.samples['true_ll'] = self.sampler.likelihood(ground_truth, 0)
        self.samples['true_prior'] = self.sampler.prior(ground_truth, 0)
        self.samples["true_families"] = self.data.families

        if lh_per_area:
            ground_truth_log_lh_single_area = []
            ground_truth_prior_single_area = []
            ground_truth_posterior_single_area = []

            for z in range(len(self.data.areas)):
                area = self.data.areas[np.newaxis, z]
                p_zone = self.data.p_contact[np.newaxis, z]

                # Define Model
                ground_truth_single_zone = Sample(zones=area, weights=weights,
                                                  p_global=self.data.p_universal[np.newaxis, ...],
                                                  p_zones=p_zone, p_families=self.data.p_inheritance)
                ground_truth_single_zone.everything_changed()

                # Evaluate Likelihood and Prior
                lh = self.sampler.likelihood(ground_truth_single_zone, 0)
                prior = self.sampler.prior(ground_truth_single_zone, 0)

                ground_truth_log_lh_single_area.append(lh)
                ground_truth_prior_single_area.append(prior)
                ground_truth_posterior_single_area.append(lh + prior)

                self.samples['true_lh_single_zones'] = ground_truth_log_lh_single_area
                self.samples['true_prior_single_zones'] = ground_truth_prior_single_area
                self.samples['true_posterior_single_zones'] = ground_truth_posterior_single_area

    def warm_up(self):
        initial_sample = self.empty_sample()

        # In warmup chains can have a different max_size for areas
        max_size_list = get_max_size_list((self.config['mcmc']['M_INITIAL'] + self.config['model']['MAX_M'])/4,
                                          self.config['model']['MAX_M'],
                                          self.config['mcmc']['WARM_UP']['N_WARM_UP_CHAINS'], 4)

        # Some chains only have connected steps, whereas others also have random steps

        p_grow_connected_list = \
            random.choices([1, self.config['mcmc']['P_GROW_CONNECTED']],
                           k=self.config['mcmc']['WARM_UP']['N_WARM_UP_CHAINS'])

        warmup = ZoneMCMCWarmup(network=self.data.network, features=self.data.features,
                                min_size=self.config['model']['MIN_M'],
                                max_size=max_size_list,
                                n_zones=self.config['model']['N_AREAS'],
                                prior=self.prior_structured,
                                inheritance=self.config['model']['INHERITANCE'],
                                n_chains=self.config['mcmc']['WARM_UP']['N_WARM_UP_CHAINS'],
                                operators=self.ops, families=self.data.families,
                                var_proposal=self.config['mcmc']['PROPOSAL_PRECISION'],
                                p_grow_connected=p_grow_connected_list,
                                initial_sample=initial_sample,
                                initial_size=self.config['mcmc']['M_INITIAL'])

        self.sample_from_warm_up = warmup.generate_samples(n_steps=0,
                                                           n_samples=0,
                                                           warm_up=True,
                                                           warm_up_steps=self.config['mcmc']['WARM_UP']['N_WARM_UP_STEPS'])

    def save_samples(self, run=1):

        self.samples = match_areas(self.samples)
        self.samples = rank_areas(self.samples)

        file_info = self.config['results']['FILE_INFO']

        if file_info == "n":
            fi = 'n{n}'.format(n=self.config['model']['N_AREAS'])

        elif file_info == "s":
            fi = 's{s}a{a}'.format(s=self.config['simulation']['STRENGTH'],
                                   a=self.config['simulation']['AREA'])

        elif file_info == "i":
            fi = 'i{i}'.format(i=int(self.config['model']['INHERITANCE']))

        elif file_info == "p":

            p = 0 if self.config['model']['PRIOR']['universal'] == "uniform" else 1
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
