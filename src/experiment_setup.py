#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the Experiment"""

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import os
import warnings

from src.util import set_experiment_name


class InitializeExperiment:
    def __init__(self, experiment_name="default", path_results="default"):

        # Naming and shaming
        if experiment_name == "default":
            self.experiment_name = set_experiment_name()
        else:
            self.experiment_name = experiment_name

        # File location for results
        if path_results == "default":
            self.path_results = '../results/{experiment}/'. \
                format(experiment=self.experiment_name)
        else:
            self.path_results = '{loc}/{experiment}/'. \
                format(loc=path_results, experiment=self.experiment_name)

        if not os.path.exists(self.path_results):
            os.makedirs(self.path_results)

        self.config_file = None
        self.config = {}

    def load_config(self, config_file="default"):

        # Get parameters from config_file
        if config_file == "default":
            self.config_file = '../config/config.json'
        else:
            self.config_file = config_file

        # Read config file
        self.read_config()

        # Verify config
        self.verify_config()

    def read_config(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

    def verify_config(self):

        # SIMULATION
        if 'simulation' in self.config:

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

        # MCMC
        # Is there an mcmc part in the config file?
        if 'mcmc' not in self.config:
            raise NameError("Information about the MCMC setup was not found in"
                            + self.config_file + ". Use mcmc as a key.")

        # Does the config file provide all required MCMC parameters?
        # Number of inferred areas
        if 'N_AREAS' not in self.config['mcmc']:
            raise NameError("N_AREAS is not defined in " + self.config_file)
        # Consider inheritance as a confounder?
        if 'INHERITANCE' not in self.config['mcmc']:
            raise NameError("INHERITANCE is not defined in " + self.config_file)

        # Priors
        if 'PRIOR' not in self.config['mcmc']:
            raise NameError("PRIOR is not defined in " + self.config_file)
        # Are priors complete and consistent?
        if 'geo' not in self.config['mcmc']['PRIOR']:
            raise NameError("geo PRIOR is not defined in " + self.config_file)
        if 'weights' not in self.config['mcmc']['PRIOR']:
            raise NameError("PRIOR for weights is not defined in " + self.config_file)
        if 'universal' not in self.config['mcmc']['PRIOR']:
            raise NameError("PRIOR for universal pressure is not defined in " + self.config_file)
        if 'contact' not in self.config['mcmc']['PRIOR']:
            raise NameError("PRIOR for contact is not defined in " + self.config_file)
        if self.config['mcmc']['INHERITANCE']:
            if 'inheritance' not in self.config['mcmc']['PRIOR']:
                raise NameError("PRIOR for inheritance (families) is not defined in " + self.config_file)
        else:
            if 'inheritance' in self.config['mcmc']['PRIOR']:
                warnings.warn("Inheritance is not considered in the MCMC. PRIOR for inheritance"
                              "is defined in " + self.config_file + " will not be used.")
            self.config['mcmc']['PRIOR']['inheritance'] = None

        # Which optional parameters are provided in the config file?
        # Number of steps
        if 'N_STEPS' not in self.config['mcmc']:
            self.config['mcmc']['N_STEPS'] = 30000
        # Steps discarded as burn-in
        if 'BURN_IN' not in self.config['mcmc']:
            self.config['mcmc']['BURN_IN'] = 1000
        # Increase of steps per additional area
        if 'N_STEPS_INCREASE' not in self.config['mcmc']:
            self.config['mcmc']['N_STEPS_INCREASE'] = 0.
        # Number of samples
        if 'N_SAMPLES' not in self.config['mcmc']:
            self.config['mcmc']['N_SAMPLES'] = 1000
        # Number of runs
        if 'N_RUNS' not in self.config['mcmc']:
            self.config['mcmc']['N_RUNS'] = 1
        # Minimum, maximum and initial size of areas
        if 'MIN_SIZE_AREA' not in self.config['mcmc']:
            self.config['mcmc']['MIN_SIZE_AREA'] = 3
        if 'MAX_SIZE_AREA' not in self.config['mcmc']:
            self.config['mcmc']['MAX_SIZE_AREA'] = 200
        if 'INITIAL_SIZE_AREA' not in self.config['mcmc']:
            self.config['mcmc']['INITIAL_SIZE_AREA'] = 10
        # Number of parallel Markov chains
        if 'N_CHAINS' not in self.config['mcmc']:
            self.config['mcmc']['N_CHAINS'] = 5
        # Steps per attempted chain swap
        if 'SWAP_PERIOD' not in self.config['mcmc']:
            self.config['mcmc']['SWAP_PERIOD'] = 1000
        # Number of attempted chain swaps
        if 'N_SWAPS' not in self.config['mcmc']:
            self.config['mcmc']['N_SWAPS'] = 3
        # Compute the likelihood per area?
        if 'LH_PER_AREA' not in self.config['mcmc']:
            self.config['mcmc']['LH_PER_AREA'] = False
        # Areas must be connected in the Delaunay?
        if 'CONNECTED_ONLY' not in self.config['mcmc']:
            self.config['mcmc']['CONNECTED_ONLY'] = False

        # Precision of the proposal distribution
        # PROPOSAL_PRECISION is in config --> check for consistency
        if 'PROPOSAL_PRECISION' in self.config['mcmc']:
            if 'weights' not in self.config['mcmc']['PROPOSAL_PRECISION']:
                self.config['mcmc']['PROPOSAL_PRECISION']['weights'] = 30
            if 'universal' not in self.config['mcmc']['PROPOSAL_PRECISION']:
                self.config['mcmc']['PROPOSAL_PRECISION']['universal'] = 30
            if 'contact' not in self.config['mcmc']['PROPOSAL_PRECISION']:
                self.config['mcmc']['PROPOSAL_PRECISION']['contact'] = 30
            if self.config['mcmc']['INHERITANCE']:
                if 'inheritance' not in self.config['mcmc']['PROPOSAL_PRECISION']:
                    self.config['mcmc']['PROPOSAL_PRECISION']['inheritance'] = 30
            else:
                self.config['mcmc']['PROPOSAL_PRECISION']['inheritance'] = None

        # PROPOSAL_PRECISION is not in config --> use default values
        else:
            if not self.config['mcmc']['INHERITANCE']:
                self.config['mcmc']['PROPOSAL_PRECISION'] = {"weights": 30,
                                                             "universal": 30,
                                                             "contact": 30,
                                                             "inheritance": None}
            else:
                self.config['mcmc']['PROPOSAL_PRECISION'] = {"weights": 30,
                                                             "universal": 30,
                                                             "contact": 30,
                                                             "inheritance": 30}

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

            if self.config['mcmc']['INHERITANCE']:
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
            if self.config['mcmc']['INHERITANCE']:
                self.config['mcmc']['STEPS'] = {"area": 0.05,
                                                "weights": 0.65,
                                                "universal": 0.05,
                                                "contact": 0.2,
                                                "inheritance": 0.05}
            else:
                self.config['mcmc']['STEPS'] = {"area": 0.05,
                                                "weights": 0.7,
                                                "universal": 0.05,
                                                "contact": 0.2,
                                                "inheritance": 0.00}

    def log_experiment(self):
        log_path = self.path_results + 'experiment.log'
        logging.basicConfig(format='%(message)s', filename=log_path, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Experiment: %s", self.experiment_name)
        logging.info("File location for results: %s", self.path_results)
