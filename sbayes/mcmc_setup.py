#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the MCMC process """

import typing as tp

from sbayes.sampling.zone_sampling import Sample, ZoneMCMC, ZoneMCMCWarmup
from sbayes.model import Model
from sbayes.sampling.loggers import ResultsLogger, ParametersCSVLogger, AreasLogger, LikelihoodLogger


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
        self.logger.info(f'''
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

    def sample(
            self,
            lh_per_area=True,
            initial_sample: tp.Optional[Sample] = None,
            run: int = 1
    ):
        mcmc_config = self.config['mcmc']

        if initial_sample is None:
            initial_sample = self.sample_from_warm_up

        sample_loggers = self.get_sample_loggers(run=run)

        self.sampler = ZoneMCMC(
            data=self.data,
            model=self.model,
            sample_loggers=sample_loggers,
            initial_sample=initial_sample,
            operators=mcmc_config['operators'],
            p_grow_connected=mcmc_config['grow_to_adjacent'],
            initial_size=mcmc_config['init_lang_per_area'],
            sample_from_prior=mcmc_config['sample_from_prior'],
            logger=self.logger,
        )

        self.sampler.generate_samples(self.config['mcmc']['steps'],
                                      self.config['mcmc']['samples'])

        # # Evaluate likelihood and prior for each zone alone (makes it possible to rank zones)
        # if lh_per_area:
        #     contribution_per_area(self.sampler)

        self.samples = self.sampler.statistics
        self.sampler.print_statistics()

    def warm_up(self):
        mcmc_config = self.config['mcmc']
        warmup = ZoneMCMCWarmup(
            data=self.data,
            model=self.model,
            sample_loggers=[],
            n_chains=mcmc_config['warmup']['warmup_chains'],
            operators=mcmc_config['operators'],
            p_grow_connected=mcmc_config['grow_to_adjacent'],
            initial_sample=None,
            initial_size=mcmc_config['init_lang_per_area'],
            sample_from_prior=mcmc_config['sample_from_prior'],
            logger=self.logger,
        )

        self.sample_from_warm_up = warmup.generate_samples(n_steps=0,
                                                           n_samples=0,
                                                           warm_up=True,
                                                           warm_up_steps=self.config['mcmc']['warmup']['warmup_steps'])

    def get_sample_loggers(self, run=1) -> tp.List[ResultsLogger]:
        k = self.model.n_zones
        base_dir = self.path_results / f'K{k}'
        base_dir.mkdir(exist_ok=True)
        params_path = base_dir / f'stats_K{k}_{run}.txt'
        areas_path = base_dir / f'areas_K{k}_{run}.txt'
        likelihood_path = base_dir / f'likelihood_K{k}_{run}.h5'

        sample_loggers = [
            ParametersCSVLogger(params_path, self.data, self.model),
            AreasLogger(areas_path, self.data, self.model),
        ]
        if not self.config['mcmc']['sample_from_prior']:
            sample_loggers.append(LikelihoodLogger(likelihood_path, self.data, self.model))

        return sample_loggers

    """ OLD METHODS FOR COMPATIBILITY """

    def mc_eval_ground_truth(self):
        pass

    def save_samples(self, run=1):
        pass

    def log_statistics(self):
        pass
