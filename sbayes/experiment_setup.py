#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the Experiment"""

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import os
import warnings

from sbayes.util import set_experiment_name


class Experiment:
    def __init__(self, experiment_name="default", config_file=None, log=False):

        # Naming and shaming
        if experiment_name == "default":
            self.experiment_name = set_experiment_name()
        else:
            self.experiment_name = experiment_name

        self.config_file = None
        self.config = {}
        self.base_directory = None
        self.path_results = None

        if config_file is not None:
            self.load_config(config_file)

        if log:
            self.log_experiment()

    def load_config(self, config_file):

        # Get parameters from config_file
        self.base_directory, self.config_file = self.decompose_config_path(config_file)
        self.read_config(path=self.config_file)

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

    @staticmethod
    def decompose_config_path(config_path):
        config_path = config_path.strip()
        if os.path.isabs(config_path):
            abs_config_path = config_path
        else:
            abs_config_path = os.path.abspath(config_path)

        base_directory = os.path.dirname(abs_config_path)

        return base_directory, abs_config_path.replace("\\", "/")

    def fix_relative_path(self, path):
        """Make sure that the provided path is either absolute or relative to
        the config file directory.

        Args:
            path (str): The original path (absolute or relative).

        Returns:
            str: The fixed path.
         """
        path = path.strip()
        if os.path.isabs(path):
            return path
        else:
            return os.path.join(self.base_directory, path).replace("\\", "/")

    def read_config(self, path):
        with open(path, 'r') as f:
            self.config = json.load(f)

    def is_simulation(self):
        return 'simulation' in self.config

    def verify_config(self):

        # SIMULATION
        if self.is_simulation():
            if 'SITES' not in self.config['simulation']:
                raise NameError("SITES is not defined in " + self.config_file)
            else:
                self.config['simulation']['SITES'] = self.fix_relative_path(self.config['simulation']['SITES'])
            # Does the simulation part of the config file provide all required simulation parameters?
            # Simulate inheritance?
            if 'INHERITANCE' not in self.config['simulation']:
                raise NameError("INHERITANCE is not defined in " + self.config_file)
            # Strength of the contact signal
            if 'E_CONTACT' not in self.config['simulation']:
                raise NameError("E_CONTACT is not defined in " + self.config_file)
            if 'I_CONTACT' not in self.config['simulation']:
                raise NameError("I_CONTACT is not defined in " + self.config_file)
            # Area for which contact is simulated
            if 'AREA' not in self.config['simulation']:
                raise NameError("AREA is not defined in " + self.config_file)
            if type(self.config['simulation']['AREA']) is list:
                self.config['simulation']['AREA'] = tuple(self.config['simulation']['AREA'])

            # Which optional parameters are provided in the config file?
            # Number of simulated features and states
            if 'N_FEATURES' not in self.config['simulation']:
                self.config['simulation']['N_FEATURES'] = 35
            if 'P_N_STATES' not in self.config['simulation']:
                self.config['simulation']['P_N_STATES'] = {"2": 0.4, "3": 0.3, "4": 0.3}
            # Strength of universal pressure
            if 'I_UNIVERSAL' not in self.config['simulation']:
                self.config['simulation']['I_UNIVERSAL'] = 1.0
            if 'E_UNIVERSAL' not in self.config['simulation']:
                self.config['simulation']['E_UNIVERSAL'] = 1.0
            # Use only a subset of the data for simulation?
            if 'SUBSET' not in self.config['simulation']:
                self.config['simulation']['SUBSET'] = False
            # Strength of inheritance
            if self.config['simulation']['INHERITANCE']:
                if 'I_INHERITANCE' not in self.config['simulation']:
                    self.config['simulation']['I_INHERITANCE'] = 0.2
                if 'E_INHERITANCE' not in self.config['simulation']:
                    self.config['simulation']['E_INHERITANCE'] = 2
            else:
                self.config['simulation']['I_INHERITANCE'] = None
                self.config['simulation']['E_INHERITANCE'] = None

        # Model
        # Does the config file define a model?
        if 'model' not in self.config:
            raise NameError("Information about the model was not found in"
                            + self.config_file + ". Include model as a key.")
        # Number of areas
        if 'N_AREAS' not in self.config['model']:
            raise NameError("N_AREAS is not defined in " + self.config_file)
        # Consider inheritance as a confounder?
        if 'INHERITANCE' not in self.config['model']:
            raise NameError("INHERITANCE is not defined in " + self.config_file)
        # Priors
        if 'PRIOR' not in self.config['model']:
            raise NameError("PRIOR is not defined in " + self.config_file)
        # Are priors complete and consistent?
        if 'geo' not in self.config['model']['PRIOR']:
            raise NameError("geo PRIOR is not defined in " + self.config_file)
        if 'weights' not in self.config['model']['PRIOR']:
            raise NameError("PRIOR for weights is not defined in " + self.config_file)
        if 'universal' not in self.config['model']['PRIOR']:
            raise NameError("PRIOR for universal pressure is not defined in " + self.config_file)
        if 'contact' not in self.config['model']['PRIOR']:
            raise NameError("PRIOR for contact is not defined in " + self.config_file)

        if self.config['model']['INHERITANCE']:
            if 'inheritance' not in self.config['model']['PRIOR']:
                raise NameError("PRIOR for inheritance (families) is not defined in " + self.config_file)
        else:
            if 'inheritance' in self.config['model']['PRIOR']:
                warnings.warn("Inheritance is not considered in the model. PRIOR for inheritance"
                              "defined in " + self.config_file + "will not be used.")
                self.config['model']['PRIOR']['inheritance'] = None

        if 'NEIGHBOR_DIST' not in self.config['model']:
            self.config['model']['NEIGHBOR_DIST'] = "euclidean"
        if 'LAMBDA_GEO_PRIOR' not in self.config['model']:
            self.config['model']['LAMBDA_GEO_PRIOR'] = "auto_tune"
        if 'SAMPLE_FROM_PRIOR' not in self.config['model']:
            self.config['model']['SAMPLE_FROM_PRIOR'] = False

        # Minimum, maximum size of areas
        if 'MIN_M' not in self.config['model']:
            self.config['model']['MIN_M'] = 3
        if 'MAX_M' not in self.config['model']:
            self.config['model']['MAX_M'] = 50

        # MCMC
        # Is there an mcmc part in the config file?
        if 'mcmc' not in self.config:
            raise NameError("Information about the MCMC setup was not found in"
                            + self.config_file + ". Include mcmc as a key.")

        # Which optional parameters are provided in the config file?
        # Number of steps
        if 'N_STEPS' not in self.config['mcmc']:
            self.config['mcmc']['N_STEPS'] = 30000
        # Number of samples
        if 'N_SAMPLES' not in self.config['mcmc']:
            self.config['mcmc']['N_SAMPLES'] = 1000
        # Number of runs
        if 'N_RUNS' not in self.config['mcmc']:
            self.config['mcmc']['N_RUNS'] = 1
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
        if 'P_GROW_CONNECTED' not in self.config['mcmc']:
            self.config['mcmc']['P_GROW_CONNECTED'] = 0.85
        if 'M_INITIAL' not in self.config['mcmc']:
            self.config['mcmc']['M_INITIAL'] = 5

        if 'MC3' not in self.config['mcmc']:
            self.config['mcmc']['N_CHAINS'] = 1
        else:
            # todo: activate for MC3
            pass
        # Tracer does not like unevenly spaced samples
        spacing = self.config['mcmc']['N_STEPS'] % self.config['mcmc']['N_SAMPLES']

        if spacing != 0.:
            raise ValueError("Non-consistent spacing between samples. Set N_STEPS to be a multiple of N_SAMPLES. ")

        # Precision of the proposal distribution
        # PROPOSAL_PRECISION is in config --> check for consistency
        if 'PROPOSAL_PRECISION' in self.config['mcmc']:
            if 'weights' not in self.config['mcmc']['PROPOSAL_PRECISION']:
                self.config['mcmc']['PROPOSAL_PRECISION']['weights'] = 30
            if 'universal' not in self.config['mcmc']['PROPOSAL_PRECISION']:
                self.config['mcmc']['PROPOSAL_PRECISION']['universal'] = 30
            if 'contact' not in self.config['mcmc']['PROPOSAL_PRECISION']:
                self.config['mcmc']['PROPOSAL_PRECISION']['contact'] = 30
            if self.config['model']['INHERITANCE']:
                if 'inheritance' not in self.config['mcmc']['PROPOSAL_PRECISION']:
                    self.config['mcmc']['PROPOSAL_PRECISION']['inheritance'] = 30
            else:
                self.config['mcmc']['PROPOSAL_PRECISION']['inheritance'] = None

        # PROPOSAL_PRECISION is not in config --> use default values
        else:
            if not self.config['model']['INHERITANCE']:
                self.config['mcmc']['PROPOSAL_PRECISION'] = {"weights": 15,
                                                             "universal": 40,
                                                             "contact": 20,
                                                             "inheritance": None}
            else:
                self.config['model']['PROPOSAL_PRECISION'] = {"weights": 15,
                                                              "universal": 40,
                                                              "contact": 20,
                                                              "inheritance": 20}

        # Steps per operator
        # STEPS is in config --> check for consistency
        steps_complete = True
        if 'STEPS' in self.config['mcmc']:
            if 'area' not in self.config['mcmc']['STEPS']:
                warnings.warn("STEPS for area are not defined in the config file, default STEPS will be used instead.")
                steps_complete = False

            if 'weights' not in self.config['mcmc']['STEPS']:
                warnings.warn("STEPS for weights are not defined in the config file, "
                              "default STEPS will be used instead.")
                steps_complete = False

            if 'universal' not in self.config['mcmc']['STEPS']:
                warnings.warn("STEPS for universal are not defined in the config file, "
                              "default STEPS will be used instead.")
                steps_complete = False

            if 'universal' not in self.config['mcmc']['STEPS']:
                warnings.warn("STEPS for contact are not defined in the config file, "
                              "default STEPS will be used instead.")
                steps_complete = False

            if self.config['model']['INHERITANCE']:
                if 'inheritance' not in self.config['mcmc']['STEPS']:
                    warnings.warn("Inheritance is modelled in the MCMC, but STEPS for inheritance are not defined "
                                  "in the config file, default STEPS will be used instead.")
                    steps_complete = False

            else:
                if 'inheritance' not in self.config['mcmc']['STEPS']:
                    self.config['mcmc']['STEPS']['inheritance'] = 0.0
                elif self.config['mcmc']['STEPS']['inheritance'] > 0.0:
                    warnings.warn("Inheritance is not modelled in the MCMC, but STEPS for inheritance are defined"
                                  "in the config file, default STEPS will be used instead.")
                    steps_complete = False

        # STEPS is not in config --> use default
        if 'STEPS' not in self.config['mcmc'] or not steps_complete:
            if self.config['model']['INHERITANCE']:
                self.config['mcmc']['STEPS'] = {"area": 0.05,
                                                "weights": 0.4,
                                                "universal": 0.05,
                                                "contact": 0.4,
                                                "inheritance": 0.1}
            else:
                self.config['model']['STEPS'] = {"area": 0.05,
                                                 "weights": 0.45,
                                                 "universal": 0.05,
                                                 "contact": 0.45,
                                                 "inheritance": 0.00}

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
        else:
            if 'simulated' not in self.config['data']:
                self.config['data']['simulated'] = False

            if not self.config['data']['simulated']:
                if not self.config['data']['FEATURES']:
                    raise NameError("FEATURES is empty. Provide file paths to features file (e.g. features.csv)")
                else:
                    self.config['data']['FEATURES'] = self.fix_relative_path(self.config['data']['FEATURES'])
                if self.config['data']['PRIOR']:
                    if self.config['data']['PRIOR']['universal']:
                        self.config['data']['PRIOR']['universal'] = \
                            self.fix_relative_path(self.config['data']['PRIOR']['universal'])
                    if self.config['data']['PRIOR']['inheritance']:
                        for key in self.config['data']['PRIOR']['inheritance']:
                            self.config['data']['PRIOR']['inheritance'][key] = \
                                self.fix_relative_path(self.config['data']['PRIOR']['inheritance'][key])

    def log_experiment(self):
        log_path = self.path_results + 'experiment.log'
        logging.basicConfig(format='%(message)s', filename=log_path, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Experiment: %s", self.experiment_name)
        logging.info("File location for results: %s", self.path_results)
