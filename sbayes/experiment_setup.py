#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the Experiment"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

from sbayes.util import set_experiment_name, decompose_config_path, fix_relative_path, iter_items_recursive
from sbayes.config.config import SBayesConfig, BaseConfig, RelativeFilePath
from sbayes import config

try:
    import importlib.resources as pkg_resources     # PYTHON >= 3.7
except ImportError:
    import importlib_resources as pkg_resources     # PYTHON < 3.7

REQUIRED = '<REQUIRED>'
DEFAULT_CONFIG = json.loads(pkg_resources.read_text(config, 'default_config.json'))


class Experiment:

    """sBayes experiment class. Takes care of loading and verifying the config file,
    handling paths, setting up logging...

    Attributes:
        experiment_name (str): The name of the experiment run (= name of results folder)
        config_file (Path): The path to the config_file.
        config (dict): The config parsed into a python dictionary.
        base_directory (Path): The directory containing the config file.
        path_results (Path): The path to the results folder.
        logger (logging.Logger): The logger used throughout the run of the experiment.

    """

    def __init__(self,
                 experiment_name: str = None,
                 config_file: Optional[Path] = None,
                 custom_settings: Optional[dict] = None,
                 log: bool = True):

        # Naming and shaming
        if experiment_name is None:
            self.experiment_name = set_experiment_name()
        else:
            self.experiment_name = experiment_name

        self.config_file = None
        self.config = {}
        self.base_directory = None
        self.path_results = None

        self.logger = self.init_logger()

        if config_file is not None:
            self.load_config(config_file, custom_settings=custom_settings)

        if log:
            self.log_experiment()

    def load_config(self,
                    config_file: Path,
                    custom_settings: Optional[dict] = None):

        self.config = SBayesConfig.from_config_file(config_file, custom_settings)

        # Set results path
        self.path_results = self.config.results.path / self.experiment_name

        if not os.path.exists(self.path_results):
            os.makedirs(self.path_results)

        self.add_logger_file(self.path_results)

    def is_simulation(self):
        return 'simulation' in self.config

    def verify_priors(self, priors_cfg: dict):

        # Define which priors are required
        required_priors = ['objects_per_cluster', 'geo', 'weights', "cluster_effect", "confounding_effects"]

        # Check presence and validity of each required prior
        for key in required_priors:
            if key not in priors_cfg:
                NameError(f"Prior \'{key}\' is not defined in {self.config_file}.")

            prior = priors_cfg[key]

            # Cluster size
            if key == "objects_per_cluster":
                if 'type' not in prior:
                    raise NameError(f"`type` for prior \'{key}\' is not defined in {self.config_file}.")

            # Geo
            if key == "geo":
                if 'type' not in prior:
                    raise NameError(f"`type` for prior \'{key}\' is not defined in {self.config_file}.")

                if prior['type'] == "gaussian":
                    if 'covariance' not in prior:
                        raise NameError(
                            f"`covariance` for gaussian geo prior is not defined in {self.config_file}.")

                if prior['type'] == "cost_based":
                    if 'rate' not in prior:
                        raise NameError(
                            f"`rate` for cost based geo prior is not defined in {self.config_file}.")
                    if 'linkage' not in prior:
                        prior['linkage'] = "mst"
                    if 'costs' not in prior:
                        prior['costs'] = "from_data"
                    if prior['costs'] != "from_data":
                        prior['costs'] = fix_relative_path(prior['file'], self.base_directory)

            # Weights
            if key == "weights":
                if 'type' not in prior:
                    raise NameError(f"`type` for prior \'{key}\' is not defined in {self.config_file}.")

            # Cluster effects
            if key == "cluster_effect":
                if 'type' not in prior:
                    raise NameError(f"`type` for prior \'{key}\' is not defined in {self.config_file}.")

            # Confounding effects
            if key == "confounding_effects":
                for k, v in self.config['model']['confounders'].items():
                    if k not in self.config['model']['prior']['confounding_effects']:
                        raise NameError(f"Prior for \'{k}\' is not defined in {self.config_file}.")

                    for g in v:
                        if g not in self.config['model']['prior']['confounding_effects'][k]:
                            raise NameError(f"Prior for \'{g}\' is not defined in {self.config_file}.")

                        if 'type' not in self.config['model']['prior']['confounding_effects'][k][g]:
                            raise NameError(f"`type` for prior \'{g}\' is not defined in {self.config_file}.")

    def verify_config(self):
        # Check that all required fields are present
        for key, value, loc in iter_items_recursive(self.config):
            if value == REQUIRED:
                loc_string = ': '.join([f'"{k}"' for k in (loc + (key, REQUIRED))])
                raise NameError(f'The value for a required field is not defined in {self.config_file}:\n\t{loc_string}')\

        # Data
        if 'data' not in self.config:
            raise NameError(f'´data´ are not defined in {self.config_file}')

        if not self.config['data']['features']:
            raise NameError("`features` is empty. Provide path to features file (e.g. features.csv)")
        if not self.config['data']['feature_states']:
            raise NameError("`feature_states` is empty. Provide path to feature_states file (e.g. feature_states.csv)")

        self.config['data']['features'] = fix_relative_path(self.config['data']['features'], self.base_directory)
        self.config['data']['feature_states'] = fix_relative_path(self.config['data']['feature_states'],
                                                                  self.base_directory)

        # Model / Priors
        self.verify_priors(self.config['model']['prior'])

        # MCMC
        if 'mcmc' not in self.config:
            NameError(f'´mcmc´ is not defined in {self.config_file}')

        # Tracer does not like unevenly spaced samples
        spacing = self.config['mcmc']['steps'] % self.config['mcmc']['samples']

        if spacing != 0.:
            raise ValueError("Non-consistent spacing between samples. Set ´steps´ to be a multiple of ´samples´.")

        # Do not use source operators if sampling from source is disabled
        if not self.config['model']['sample_source']:
            if self.config['mcmc']['operators'].get('source', 0) != 0:
                logging.info('Operator for source was set to 0, because ´sample_source´ is disabled.')
            self.config['mcmc']['operators']['source'] = 0.0

        # Re-normalize weights for operators
        weights_sum = sum(self.config['mcmc']['operators'].values())

        for operator, weight in self.config['mcmc']['operators'].items():
            self.config['mcmc']['operators'][operator] = weight / weights_sum

    @staticmethod
    def init_logger():
        logger = logging.Logger('sbayesLogger', level=logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
        return logger

    def add_logger_file(self, path_results):
        log_path = path_results / 'experiment.log'
        self.logger.addHandler(logging.FileHandler(filename=log_path))

    def log_experiment(self):
        self.logger.info("Experiment: %s", self.experiment_name)
        self.logger.info("File location for results: %s", self.path_results)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
