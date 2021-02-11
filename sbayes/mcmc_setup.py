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
from sbayes.util import normalize, samples2file, get_max_size_list
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

        # Assign steps to operators
        self.ops = {}
        self.steps_per_operator()

        # Samples
        self.sampler = None
        self.samples = None
        self.sample_from_warm_up = None

        # Create the model to sample from
        self.model = Model(data=self.data, config=self.config['model'])

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

        logging.info("Geo-prior: %s ", self.model.prior.config['geo']['type'])
        logging.info("Prior on weights: %s ", self.model.prior.config['weights']['type'])
        logging.info("Prior on universal pressure (alpha): %s ", self.model.prior.config['universal']['type'])
        if self.config['model']['INHERITANCE']:
            logging.info("Prior on inheritance (beta): %s ", self.model.prior.config['inheritance']['type'])

        logging.info("Prior on contact (gamma): %s ", self.model.prior.config['contact']['type'])
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
               'alter_p_families': self.config['mcmc']['STEPS']['inheritance'],
               'gibbs_sample_sources': self.config['mcmc']['STEPS']['source'],
               # 'gibbs_sample_weights': self.config['mcmc']['STEPS']['weights'],
               # 'gibbs_sample_p_global': self.config['mcmc']['STEPS']['universal'],
               # 'gibbs_sample_p_zones': self.config['mcmc']['STEPS']['contact'],
               # 'gibbs_sample_p_families': self.config['mcmc']['STEPS']['inheritance'],
               }
        self.ops = ops

    def sample(self, lh_per_area=True, initial_sample: typing.Optional[typing.Any] = None):

        if initial_sample is None:
            if self.sample_from_warm_up is None:
                initial_sample = self.empty_sample()
            else:
                initial_sample = self.sample_from_warm_up

        self.sampler = ZoneMCMC(data=self.data,
                                model=self.model,
                                n_chains=self.config['mcmc']['N_CHAINS'],
                                initial_sample=initial_sample,
                                operators=self.ops,
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

        warmup = ZoneMCMCWarmup(data=self.data,
                                model=self.model,
                                n_chains=self.config['mcmc']['WARM_UP']['N_WARM_UP_CHAINS'],
                                operators=self.ops,
                                var_proposal=self.config['mcmc']['PROPOSAL_PRECISION'],
                                p_grow_connected=self.config['mcmc']['P_GROW_CONNECTED'],
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
