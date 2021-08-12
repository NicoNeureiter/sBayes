#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the MCMC process """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import os
import random
import typing

from sbayes.postprocessing import (contribution_per_area, log_operator_statistics,
                                   log_operator_statistics_header, match_areas, rank_areas)
from sbayes.sampling.zone_sampling import Sample, ZoneMCMC, ZoneMCMCWarmup
from sbayes.util import normalize, samples2file, scale_counts, inheritance_counts_to_dirichlet, counts_to_dirichlet
from sbayes.model import Model


class MCMC:
    def __init__(self, data, experiment):

        # Retrieve the data
        self.data = data

        # Retrieve the configurations
        self.config = experiment.config

        # Paths
        self.path_log = experiment.path_results / 'experiment.log'
        self.path_results = experiment.path_results

        # Create the model to sample from
        self.model = Model(data=self.data, config=self.config['model'])

        # Assign steps to operators
        self.ops = {}
        self.steps_per_operator()

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

            self.prior_structured['geo'] = {'type': 'cost_based',
                                            'cost_matrix': self.data.geo_prior['cost_matrix'],
                                            'scale': self.config['model']['PRIOR']['geo']['scale']}

        else:
            raise ValueError('Geo prior not supported')

        # area_size prior
        VALID_SIZE_PRIOR_TYPES = ['none', 'uniform', 'quadratic']
        size_prior_type = self.config['model']['PRIOR']['area_size']['type']
        if size_prior_type in VALID_SIZE_PRIOR_TYPES:
            self.prior_structured['area_size'] = {'type': size_prior_type}
        else:
            raise ValueError(f'Area-size prior ´{size_prior_type}´ not supported' +
                             f'(valid types: {VALID_SIZE_PRIOR_TYPES}).')

        # weights prior
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

        logging.basicConfig(format='%(message)s', filename=self.path_log, level=logging.DEBUG)
        logging.info(self.model.get_setup_message())

        logging.info(f'''
MCMC SETUP
##########################################
MCMC with {mcmc_config['steps']} steps and {mcmc_config['samples']} samples
Warm-up: {mcmc_config['warmup']['warmup_chains']} chains exploring the parameter space in 
{mcmc_config['warmup']['warmup_steps']} steps
Ratio of areal steps (growing, shrinking, swapping areas): {mcmc_config['operators']['area']}
Ratio of weight steps (changing weights): {mcmc_config['operators']['weights']}
Ratio of universal steps (changing alpha) : {mcmc_config['operators']['universal']}
Ratio of inheritance steps (changing beta): {mcmc_config['operators']['inheritance']}
Ratio of contact steps (changing gamma): {mcmc_config['operators']['contact']}
''')

    def steps_per_operator(self):
        """Assign step frequency per operator."""
        steps_config = self.config['mcmc']['operators']
        ops = {'shrink_zone': steps_config['area'] * 0.4,
               'grow_zone': steps_config['area'] * 0.4,
               'swap_zone': steps_config['area'] * 0.2,
               # 'gibbsish_sample_zones': steps_config['area'] * 0.7
               }
        if self.model.sample_source:
            ops.update({
               'gibbs_sample_sources': steps_config['source'],
               'gibbs_sample_weights': steps_config['weights'],
               'gibbs_sample_p_global': steps_config['universal'],
               'gibbs_sample_p_zones': steps_config['contact'],
               'gibbs_sample_p_families': steps_config['inheritance'],
            })
        else:
            ops.update({
               'alter_weights': steps_config['weights'],
               'alter_p_global': steps_config['universal'],
               'alter_p_zones': steps_config['contact'],
               'alter_p_families': steps_config['inheritance'],
            })
        self.ops = ops

    def sample(self, lh_per_area=True, initial_sample: typing.Optional[typing.Any] = None):

        if initial_sample is None:
            if self.sample_from_warm_up is None:
                initial_sample = self.empty_sample()
            else:
                initial_sample = self.sample_from_warm_up

        self.sampler = ZoneMCMC(data=self.data,
                                model=self.model,
                                initial_sample=initial_sample,
                                operators=self.ops,
                                p_grow_connected=self.config['mcmc']['grow_to_adjacent'],
                                initial_size=self.config['mcmc']['init_lang_per_area'])

        self.sampler.generate_samples(self.config['mcmc']['steps'],
                                      self.config['mcmc']['samples'])

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

        warmup = ZoneMCMCWarmup(data=self.data,
                                model=self.model,
                                n_chains=self.config['mcmc']['warmup']['warmup_chains'],
                                operators=self.ops,
                                p_grow_connected=self.config['mcmc']['grow_to_adjacent'],
                                initial_sample=initial_sample,
                                initial_size=self.config['mcmc']['init_lang_per_area'])

        self.sample_from_warm_up = warmup.generate_samples(n_steps=0,
                                                           n_samples=0,
                                                           warm_up=True,
                                                           warm_up_steps=self.config['mcmc']['warmup']['warmup_steps'])

    def save_samples(self, run=1):

        self.samples = match_areas(self.samples)
        self.samples = rank_areas(self.samples)

        fi = 'K{K}'.format(K=self.config['model']['areas'])

        # Todo: could be relevant for simulation
        # file_info = "K"
        #
        # if file_info == "K":
        #     fi = 'K{K}'.format(K=self.config['model']['areas'])
        # elif file_info == "s":
        #     fi = 's{s}a{a}'.format(s=self.config['simulation']['STRENGTH'],
        #                            a=self.config['simulation']['AREA'])
        #
        # elif file_info == "i":
        #     fi = 'i{i}'.format(i=int(self.config['model']['INHERITANCE']))
        #
        # elif file_info == "p":
        #
        #     p = 0 if self.config['model']['PRIOR']['universal'] == "uniform" else 1
        #     fi = 'p{p}'.format(p=p)
        #
        # else:
        #     raise ValueError("file_info must be 'n', 's', 'i' or 'p'")

        run = '_{run}'.format(run=run)
        pth = self.path_results / fi
        ext = '.txt'
        gt_pth = pth / 'ground_truth'

        paths = {'parameters': pth / ('stats_' + fi + run + ext),
                 'areas': pth / ('areas_' + fi + run + ext),
                 'gt': gt_pth / ('stats' + ext),
                 'gt_areas': gt_pth / ('areas' + ext)}

        pth.mkdir(exist_ok=True)

        if self.data.is_simulated:
            self.eval_ground_truth()
            gt_pth.mkdir(exist_ok=True)

        samples2file(self.samples, self.data, self.config, paths)
