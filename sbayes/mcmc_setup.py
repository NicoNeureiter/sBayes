#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the MCMC process """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import typing

from sbayes.postprocessing import contribution_per_area, match_areas, rank_areas
from sbayes.sampling.zone_sampling import Sample, ZoneMCMC, ZoneMCMCWarmup
from sbayes.util import normalize, samples2file
from sbayes.model import Model
from sbayes.simulation import Simulation


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

        # Samples
        self.sampler = None
        self.samples = None
        self.sample_from_warm_up = None

        self.logger = experiment.logger

    def log_setup(self):
        mcmc_config = self.config['mcmc']

        self.logger.info(self.model.get_setup_message())


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

    # def steps_per_operator(self):
    #     """Assign step frequency per operator."""
    #     steps_config = self.config['mcmc']['operators']
    #     ops = {'shrink_zone': steps_config['area'] * 0.4,
    #            'grow_zone': steps_config['area'] * 0.4,
    #            'swap_zone': steps_config['area'] * 0.2,
    #            # 'gibbsish_sample_zones': steps_config['area'] * 0.7
    #            }
    #     if self.model.sample_source:
    #         ops.update({
    #            'gibbs_sample_sources': steps_config['source'],
    #            'gibbs_sample_weights': steps_config['weights'],
    #            'gibbs_sample_p_global': steps_config['universal'],
    #            'gibbs_sample_p_zones': steps_config['contact'],
    #            'gibbs_sample_p_families': steps_config['inheritance'],
    #         })
    #     else:
    #         ops.update({
    #            'alter_weights': steps_config['weights'],
    #            'alter_p_global': steps_config['universal'],
    #            'alter_p_zones': steps_config['contact'],
    #            'alter_p_families': steps_config['inheritance'],
    #         })
    #     self.ops = ops

    def sample(self, lh_per_area=True, initial_sample: typing.Optional[typing.Any] = None):
        mcmc_config = self.config['mcmc']

        if initial_sample is None:
            initial_sample = self.sample_from_warm_up

        self.sampler = ZoneMCMC(data=self.data,
                                model=self.model,
                                initial_sample=initial_sample,
                                operators=mcmc_config['operators'],
                                p_grow_connected=mcmc_config['grow_to_adjacent'],
                                initial_size=mcmc_config['init_lang_per_area'])

        self.sampler.generate_samples(self.config['mcmc']['steps'],
                                      self.config['mcmc']['samples'])


        # Evaluate likelihood and prior for each zone alone (makes it possible to rank zones)
        if lh_per_area:
            contribution_per_area(self.sampler)

        self.samples = self.sampler.statistics

    def eval_ground_truth(self, lh_per_area=True):
        assert isinstance(self.data, Simulation)
        simulation = self.data

        # Evaluate the likelihood of the true sample in simulated data
        # If the model includes inheritance use all weights, if not use only the first two weights (global, zone)
        if self.config['model']['inheritance']:
            weights = simulation.weights.copy()
        else:
            weights = normalize(simulation.weights[:, :2])

        ground_truth = Sample(zones=simulation.areas,
                              weights=weights,
                              p_global=simulation.p_universal[np.newaxis, ...],
                              p_zones=simulation.p_contact,
                              p_families=simulation.p_inheritance)

        ground_truth.everything_changed()

        self.samples['true_zones'] = simulation.areas
        self.samples['true_weights'] = weights
        self.samples['true_p_global'] = simulation.p_universal[np.newaxis, ...]
        self.samples['true_p_zones'] = simulation.p_contact
        self.samples['true_p_families'] = simulation.p_inheritance
        self.samples['true_ll'] = self.sampler.likelihood(ground_truth, 0)
        self.samples['true_prior'] = self.sampler.prior(ground_truth, 0)
        self.samples["true_families"] = simulation.families

        if lh_per_area:
            ground_truth_log_lh_single_area = []
            ground_truth_prior_single_area = []
            ground_truth_posterior_single_area = []

            for z in range(len(simulation.areas)):
                area = simulation.areas[np.newaxis, z]
                p_zone = simulation.p_contact[np.newaxis, z]

                # Define Model
                ground_truth_single_zone = Sample(zones=area, weights=weights,
                                                  p_global=simulation.p_universal[np.newaxis, ...],
                                                  p_zones=p_zone, p_families=simulation.p_inheritance)
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
        mcmc_config = self.config['mcmc']

        warmup = ZoneMCMCWarmup(data=self.data,
                                model=self.model,
                                n_chains=mcmc_config['warmup']['warmup_chains'],
                                operators=mcmc_config['operators'],
                                p_grow_connected=mcmc_config['grow_to_adjacent'],
                                initial_sample=None,
                                initial_size=mcmc_config['init_lang_per_area'])

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

    def log_statistics(self):
        self.sampler.print_statistics(self.samples)
