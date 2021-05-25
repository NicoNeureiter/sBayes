#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the MCMC process """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import typing

from sbayes.postprocessing import (contribution_per_area, match_areas, rank_areas,
                                   annotate_results, evaluate_ground_truth)
from sbayes.sampling.zone_sampling import Sample, ZoneMCMC, ZoneMCMCWarmup
from sbayes.sampling.loggers import AreasLogger, ParametersLogger
from sbayes.util import normalize, samples2file
from sbayes.model import Model
from sbayes.simulation import Simulation


class MCMC:
    def __init__(self, data, experiment, i_run=0):

        # Retrieve the data
        self.data = data

        # Retrieve the configurations
        self.config = experiment.config

        # Paths
        self.path_log = experiment.path_results / 'experiment.log'
        self.path_results = experiment.path_results

        # Create the model to sample from
        self.model = Model(data=self.data, config=self.config['model'])

        # Samples
        self.sampler = None
        self.samples = None
        self.sample_from_warm_up = None

        # Experiment logger and results loggers
        self.logger = experiment.logger
        parameters_path, areas_path = experiment.set_up_result_paths(i_run=i_run)
        self.parameters_logger = ParametersLogger(path=parameters_path, data=data)
        self.areas_logger = AreasLogger(path=areas_path, data=data)

        # Remember the id of the current run
        self.i_run = i_run

    def log_setup(self):
        mcmc_config = self.config['mcmc']

        self.logger.info(self.model.get_setup_message())

        msg_inherit_precision = ''
        if mcmc_config['proposal_precision']['inheritance'] is not None:
            msg_inherit_precision = 'Pseudocounts for tuning the width of the proposal distribution for inheritance (beta): %s\n' % \
                 mcmc_config['proposal_precision']['inheritance']
        logging.info(f'''
MCMC SETUP
##########################################
MCMC with {mcmc_config['n_steps']} steps and {mcmc_config['n_samples']} samples
Warm-up: {mcmc_config['warm_up']['n_warm_up_chains']} chains exploring the parameter space in {mcmc_config['warm_up']['n_warm_up_steps']} steps
Pseudocounts for tuning the width of the proposal distribution for weights: {mcmc_config['proposal_precision']['weights']}
Pseudocounts for tuning the width of the proposal distribution for universal pressure (alpha): {mcmc_config['proposal_precision']['universal']}
{msg_inherit_precision}\
Pseudocounts for tuning the width of the proposal distribution for areas (gamma): {mcmc_config['proposal_precision']['contact']}
Ratio of areal steps (growing, shrinking, swapping areas): {mcmc_config['steps']['area']}
Ratio of weight steps (changing weights): {mcmc_config['steps']['weights']}
Ratio of universal steps (changing alpha) : {mcmc_config['steps']['universal']}
Ratio of inheritance steps (changing beta): {mcmc_config['steps']['inheritance']}
Ratio of contact steps (changing gamma): {mcmc_config['steps']['contact']}
        ''')

    def sample(self, lh_per_area=True, initial_sample: typing.Optional[typing.Any] = None):
        mcmc_config = self.config['mcmc']

        if initial_sample is None:
            initial_sample = self.sample_from_warm_up

        self.sampler = ZoneMCMC(
            data=self.data,
            model=self.model,
            n_chains=mcmc_config['N_CHAINS'],
            initial_sample=initial_sample,
            operators=mcmc_config['steps'],
            var_proposal=mcmc_config['proposal_precision'],
            p_grow_connected=mcmc_config['p_grow_connected'],
            initial_size=mcmc_config['m_initial'],
            logger=self.logger,
            sample_from_prior=mcmc_config['sample_from_prior'],
            results_loggers=[
                self.parameters_logger,
                self.areas_logger
            ]
        )

        self.sampler.generate_samples(mcmc_config['n_steps'],
                                      mcmc_config['n_samples'])

        # Evaluate likelihood and prior for each zone alone (makes it possible to rank zones)
        if lh_per_area:
            contribution_per_area(self.sampler)

        self.samples = self.sampler.statistics

        # TODO I put postprocessing here for backwards compatibility, but it should really
        # be separate from MCMC and done directly in cli.py (and custom run scripts)
        annotate_results(res_params_path=self.parameters_logger.path,
                         res_areas_path=self.areas_logger.path)

    def eval_ground_truth(self, lh_per_area=True):
        assert isinstance(self.data, Simulation)
        return evaluate_ground_truth(simulation=self.data, model=self.model,
                                     lh_per_area=lh_per_area)

    def warm_up(self):
        mcmc_config = self.config['mcmc']
        warmup = ZoneMCMCWarmup(
            data=self.data,
            model=self.model,
            n_chains=mcmc_config['warm_up']['n_warm_up_chains'],
            operators=mcmc_config['steps'],
            var_proposal=mcmc_config['proposal_precision'],
            p_grow_connected=mcmc_config['p_grow_connected'],
            initial_size=mcmc_config['m_initial'],
            logger=self.logger,
            sample_from_prior=mcmc_config['sample_from_prior']
        )

        self.sample_from_warm_up = warmup.generate_samples(
            n_steps=0,
            n_samples=0,
            warm_up=True,
            warm_up_steps=mcmc_config['warm_up']['n_warm_up_steps']
        )

    # def save_samples(self, run=1):
    #     self.samples = match_areas(self.samples)
    #     self.samples = rank_areas(self.samples)
    #
    #     run = '_{run}'.format(run=run)
    #     pth = self.path_results / fi
    #     ext = '.txt'
    #     gt_pth = pth / 'ground_truth'
    #
    #     paths = {'parameters': pth / ('stats_' + fi + run + ext),
    #              'areas': pth / ('areas_' + fi + run + ext),
    #              'gt': gt_pth / ('stats' + ext),
    #              'gt_areas': gt_pth / ('areas' + ext)}
    #
    #     pth.mkdir(exist_ok=True)

    def log_statistics(self):
        self.sampler.print_statistics(self.samples)
