#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the Experiment"""
import json
import logging
import os
import warnings

from pathlib import Path
from typing import Optional

try:
    import importlib.resources as pkg_resources     # PYTHON >= 3.7
except ImportError:
    import importlib_resources as pkg_resources     # PYTHON < 3.7

import typing
from pathlib import Path

import pycldf

from sbayes.util import set_experiment_name
from sbayes import config

REQUIRED = '<REQUIRED>'
DEFAULT_CONFIG = json.loads(pkg_resources.read_text(config, 'default_config.json'))
DEFAULT_CONFIG_SIMULATION = json.loads(pkg_resources.read_text(config, 'default_config_simulation.json'))


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

    def __init__(self, experiment_name: str = None,
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
        # Get parameters from config_file
        self.base_directory, self.config_file = self.decompose_config_path(config_file)

        # Read the user specified config file
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

        # Load defaults
        set_defaults(self.config, DEFAULT_CONFIG)
        if 'simulation' in self.config:
            self.config['data'].pop('features')
            self.config['data'].pop('feature_states')
            set_defaults(self.config['simulation'], DEFAULT_CONFIG_SIMULATION)

        if custom_settings is not None:
            update_recursive(self.config, custom_settings)

        # Verify config
        self.verify_config()

        # Set results path
        self.path_results = '{path}/{experiment}/'.format(
            path=self.config['results']['RESULTS_PATH'],
            experiment=self.experiment_name
        )

        # Compile relative paths, to be relative to config file
        self.path_results = self.fix_relative_path(self.path_results)

        if not os.path.exists(self.path_results):
            os.makedirs(self.path_results)

        self.add_logger_file(self.path_results)

    @staticmethod
    def decompose_config_path(config_path):
        abs_config_path = Path(config_path).absolute()
        base_directory = abs_config_path.parent
        return base_directory, abs_config_path

    def fix_relative_path(self, path):
        """Make sure that the provided path is either absolute or relative to
        the config file directory.

        Args:
            path (Path or str): The original path (absolute or relative).

        Returns:
            Path: The fixed path.
         """
        path = Path(path)
        if path.is_absolute():
            return path
        else:
            return self.base_directory / path

    def is_simulation(self):
        return 'simulation' in self.config

    def verify_priors(self, priors_cfg: dict, inheritance: bool):

        # Define which priors are required
        required_priors = ['geo', 'area_size', 'weights', 'universal', 'contact']
        if inheritance:
            required_priors.append('inheritance')
        else:
            if 'inheritance' in priors_cfg:
                warnings.warn("Inheritance is not considered in the model. prior for inheritance "
                              + f"defined in {self.config_file} will not be used.")
                priors_cfg['inheritance'] = None

        # Check presence and validity of each required prior
        for key in required_priors:
            if key not in priors_cfg:
                NameError(f"Prior \'{key}\' is not defined in {self.config_file}.")

            prior = priors_cfg[key]
            if 'type' not in prior:
                raise NameError(f"type for prior \'{key}\' is not defined in {self.config_file}.")
            if prior['type'] == 'cost_based':
                if 'scale' not in prior:
                    raise NameError(f"scale for geo prior is not defined in {self.config_file}.")
                if 'file' in prior:
                    prior['file'] = self.fix_relative_path(prior['file'])
            if prior['type'] == 'counts':
                if 'file_type' not in prior:
                    raise NameError(f"counts file for prior \'{key}\' is not defined in {self.config_file}.")
                if 'scale_counts' not in prior:
                    prior['scale_counts'] = None

                if key == 'universal':
                    if 'file' not in prior:
                        raise NameError(f"counts file for prior \'{key}\' is not "
                                        f"defined in {self.config_file}.")
                    prior['file'] = self.fix_relative_path(prior['file'])
                elif key == 'inheritance':
                    if 'files' not in prior:
                        raise NameError(f"counts files for prior \'{key}\' is not defined in {self.config_file}.")
                    for fam in prior['files']:
                        prior['files'][fam] = self.fix_relative_path(prior['files'][fam])

    def verify_config(self):
        for k, v in iter_items_recursive(self.config):
            if v == REQUIRED:
                raise NameError(f'{k} is not defined {self.config_file}')

        # Are priors complete and consistent?
        self.verify_priors(self.config['model']['prior'],
                           inheritance=self.config['model']['inheritance'])

        # SIMULATION
        if self.is_simulation():
            self.config['simulation']['sites'] = self.fix_relative_path(self.config['simulation']['sites'])
            if type(self.config['simulation']['area']) is list:
                self.config['simulation']['area'] = tuple(self.config['simulation']['area'])

        if 'NEIGHBOR_DIST' not in self.config['model']:
            self.config['model']['NEIGHBOR_DIST'] = 'euclidean'

        # MCMC
        # Is there an mcmc part in the config file?
        if 'mcmc' not in self.config:
            raise NameError('Information about the MCMC setup was not found in'
                            + self.config_file + '. Include mcmc as a key.')

        # todo: activate for MC3
        # Number of parallel Markov chains
        # if 'N_CHAINS' not in self.config['mcmc']:
        #     self.config['mcmc']['N_CHAINS'] = 5
        # # Steps between two attempted chain swaps
        # if 'SWAP_PERIOD' not in self.config['mcmc']:
        #     self.config['mcmc']['SWAP_PERIOD'] = 1000
        # # Number of attempted chain swaps
        # if 'N_SWAPS' not in self.config['mcmc']:
        #     self.config['mcmc']['N_SWAPS'] = 3
        if 'MC3' not in self.config['mcmc']:
            self.config['mcmc']['N_CHAINS'] = 1
        else:
            # todo: activate for MC3
            pass

        # Tracer does not like unevenly spaced samples
        spacing = self.config['mcmc']['n_steps'] % self.config['mcmc']['n_samples']

        if spacing != 0.:
            raise ValueError("Non-consistent spacing between samples. Set n_steps to be a multiple of n_samples. ")

        # Do not use inheritance steps if inheritance is disabled
        if not self.config['model']['inheritance']:
            if self.config['mcmc']['steps'].get('inheritance', 0) != 0:
                self.logger.warning('steps for inheritance was set to 0, because ´inheritance´ is disabled.')
            self.config['mcmc']['steps']['inheritance'] = 0.0

        if not self.config['model']['sample_source']:
            if self.config['mcmc']['steps'].get('source', 0) != 0:
                self.logger.warning('steps for source was set to 0, because ´sample_source´ is disabled.')
            self.config['mcmc']['steps']['source'] = 0.0

        # Normalize weights
        weights_sum = sum(self.config['mcmc']['steps'].values())
        for operator, weight in self.config['mcmc']['steps'].items():
            self.config['mcmc']['steps'][operator] = weight / weights_sum

        if 'results' in self.config:
            if 'RESULTS_PATH' not in self.config['results']:
                self.config['results']['RESULTS_PATH'] = "results"
            if 'FILE_INFO' not in self.config['results']:
                self.config['results']['FILE_INFO'] = "n"

        else:
            self.config['results'] = {}
            self.config['results']['RESULTS_PATH'] = "results"
            self.config['results']['FILE_INFO'] = "n"

        # Data
        if 'data' not in self.config:
            raise NameError("Provide file paths to data.")
        elif type(self.config['data']) == str:
            # TODO: type comparison is considered bad form in Python. What to
            # use instead?
            self.config['data'] = {
                'simulated': False,
                'cldf_dataset': pycldf.StructureDataset.from_metadata(
                    self.base_directory / self.config['data']
                ),
            }
        else:
            if 'simulated' not in self.config['data']:
                self.config['data']['simulated'] = False

            if not self.config['data']['simulated']:
                if 'cldf_dataset' in self.config['data']:
                    self.config['data']['cldf_dataset'] = pycldf.StructureDataset.from_metadata(
                        self.base_directory / self.config['data']
                    )
                else:
                    if not self.config['data']['features']:
                        raise NameError("features is empty. Provide file paths to features file (e.g. features.csv)")
                    else:
                        self.config['data']['features'] = self.fix_relative_path(self.config['data']['features'])
                    if not self.config['data']['feature_states']:
                        raise NameError("feature_states is empty. Provide file paths to feature_states file (e.g. feature_states.csv)")
                    else:
                        self.config['data']['feature_states'] = self.fix_relative_path(self.config['data']['feature_states'])

    def init_logger(self):
        logger = logging.Logger('sbayesLogger', level=logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
        return logger

    def add_logger_file(self, path_results):
        log_path = path_results / 'experiment.log'
        self.logger.addHandler(logging.FileHandler(filename=log_path))

    def log_experiment(self):
        self.logger.info("Experiment: %s", self.experiment_name)
        self.logger.info("File location for results: %s", self.path_results)


def set_defaults(cfg: dict, default_cfg: dict):
    """Iterate through a recursive config dictionary and set all fields that are not
    present in cfg to the default values from default_cfg.

    == Usage ===
    >>> set_defaults(cfg={0:0, 1:{1:0}, 2:{2:1}},
    ...              default_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 0}, 2: {2: 1, 1: 1}}
    >>> set_defaults(cfg={0:0, 1:1, 2:2},
    ...              default_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: 1, 2: 2}
    """
    for key in default_cfg:
        if key not in cfg:
            # Field ´key´ is not defined in cfg -> use default
            cfg[key] = default_cfg[key]

        else:
            # Field ´key´ is defined in cfg
            # -> update recursively if the field is a dictionary
            if isinstance(default_cfg[key], dict) and isinstance(cfg[key], dict):
                set_defaults(cfg[key], default_cfg[key])

    return cfg


def update_recursive(cfg: dict, new_cfg: dict):
    """Iterate through a recursive config dictionary and update cfg in all fields that are specified in new_cfg.

    == Usage ===
    >>> update_recursive(cfg={0:0, 1:{1:0}, 2:{2:1}},
    ...                  new_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 1}, 2: {2: 2, 1: 1}}
    >>> update_recursive(cfg={0:0, 1:1, 2:2},
    ...                  new_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 1}, 2: {1: 1, 2: 2}}
    """
    for key in new_cfg:
        if (key in cfg) and isinstance(new_cfg[key], dict) and isinstance(cfg[key], dict):
            # Both dictionaries have another layer -> update recursively
            update_recursive(cfg[key], new_cfg[key])
        else:
            cfg[key] = new_cfg[key]

    return cfg


def iter_items_recursive(cfg: dict):
    """Recursively iterate through all key-value pairs in ´cfg´ and sub-dictionaries.

    Args:
        cfg (dict): Config dictionary, potentially containing sub-dictionaries.

    Yields:
        tuple: key-value pairs of the bottom level dictionaries

    == Usage ===
    >>> list(iter_items_recursive({0: 0, 1: {1: 0}, 2: {2: 1, 1: 1}}))
    [(0, 0), (1, 0), (2, 1), (1, 1)]
    """
    for key, value in cfg.items():
        if isinstance(value, dict):
            yield from iter_items_recursive(value)
        else:
            yield key, value


if __name__ == '__main__':
    import doctest
    doctest.testmod()
