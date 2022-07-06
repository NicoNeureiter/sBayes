#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Defines the class ContactAreasSimulator
    Outputs a simulated contact areas, together with network, features,
    states, families, weights, p_universal (alpha), p_inheritance(beta), p_contact(gamma) """

from __future__ import absolute_import, division, print_function, unicode_literals
from pathlib import Path

import json
import logging
import os
import csv
import itertools

from sbayes.util import set_defaults, iter_items_recursive, PathLike
from sbayes.util import decompose_config_path, fix_relative_path

from sbayes.preprocessing import (ComputeNetwork, load_canvas,
                                  simulate_assignment_probabilities,
                                  assign_to_confounders,
                                  assign_to_cluster,
                                  simulate_features,
                                  simulate_weights)
from sbayes import config

try:
    import importlib.resources as pkg_resources     # PYTHON >= 3.7
except ImportError:
    import importlib_resources as pkg_resources     # PYTHON < 3.7

REQUIRED = '<REQUIRED>'
DEFAULT_CONFIG_SIMULATION = json.loads(pkg_resources.read_text(config, 'default_config_simulation.json'))


# todo: fix logging
class Simulation:
    def __init__(self, log: bool = True):

        self.config_file = None
        self.config = {}

        self.base_directory = None
        self.path_results = None

        self.logger = self.init_logger()
        if log:
            self.log_experiment()

        # self.path_log = experiment.path_results / 'experiment.log'
        # self.config = experiment.config['simulation']
        # self.sites_file = experiment.config['simulation']['sites']

        self.log_load_canvas = None

        # Simulated parameters
        self.sites = None
        self.network = None
        self.clusters = None
        self.confounders = None

        self.weights = None
        self.probabilities = None

        self.features = None
        self.states = None

        self.prior_universal = None
        self.geo_prior = None

        self.feature_names = None
        self.state_names = None
        self.site_names = None

    def load_config_simulation(self, config_file: Path):
        # Get parameters from config_file
        self.base_directory, self.config_file = decompose_config_path(config_file)

        # Read the user specified config file for simulation
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

        # Load defaults
        set_defaults(self.config, DEFAULT_CONFIG_SIMULATION)

        # Verify config
        self.verify_config()

        # Set results path
        self.path_results = self.config['results']['path']

        # Compile relative paths, to be relative to config file
        self.path_results = fix_relative_path(self.path_results, self.base_directory)

        if not os.path.exists(self.path_results):
            os.makedirs(self.path_results)

        self.add_logger_file(self.path_results)

    def verify_config(self):
        # Check that all required fields are present
        for key, value, loc in iter_items_recursive(self.config):
            if value == REQUIRED:
                loc_string = ': '.join([f'"{k}"' for k in (loc + (key, REQUIRED))])
                raise NameError(f'The value for a required field is not defined in '
                                f'{self.config_file}:\n\t{loc_string}')
        # Canvas
        self.config['canvas'] = fix_relative_path(self.config['canvas'], self.base_directory)

        # todo: what's wrong with logging?
    def log_experiment(self):
        self.logger.info("Simulation")
        self.logger.info("File location for results: %s", self.path_results)

    @staticmethod
    def init_logger():
        logger = logging.Logger('sbayesLogger', level=logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
        return logger

    def add_logger_file(self, path_results):
        log_path = path_results / 'simulation.log'
        self.logger.addHandler(logging.FileHandler(filename=log_path))

    def run_simulation(self):

        # Get sites from canvas file
        self.sites, self.site_names = load_canvas(config=self.config, logger=self.logger)

        # Assign sites to clusters
        self.network = ComputeNetwork(self.sites)
        self.clusters = assign_to_cluster(sites_sim=self.sites)

        # Assign sites to confounders
        self.confounders = assign_to_confounders(sites_sim=self.sites)

        # Simulate weights, i.e. the influence of universal pressure, contact and inheritance on each feature
        self.weights = simulate_weights(config=self.config)

        # Simulate probabilities for features
        self.probabilities = simulate_assignment_probabilities(config=self.config, clusters=self.clusters,
                                                               confounders=self.confounders)

        # Simulate features
        self.features = simulate_features(clusters=self.clusters, confounders=self.confounders,
                                          probabilities=self.probabilities, weights=self.weights)

    def write_to_csv(self):
        col_names = ["id", "x", "y"]

        all_output = [self.sites['id'],
                      self.sites['locations'][:, 0].tolist(),
                      self.sites['locations'][:, 1].tolist()]
        for k, v in self.sites['confounders'].items():
            all_output.append(v)
            col_names.append(k)

        available_states = []
        feature_col_names = []

        i = 1
        for f in self.features.T:
            all_output.append(f.tolist())
            available_states.append(list(set(f.tolist())))

            f_name = "f" + str(i)
            col_names.append(f_name)
            feature_col_names.append(f_name)
            i += 1

        # Features to csv
        relative_path_features_csv = f"{self.config['results']['path'] + '/' + 'simulated_features.csv'}"
        path_features_csv = fix_relative_path(relative_path_features_csv, self.base_directory)

        with open(path_features_csv, 'w', newline='') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            # writing the fields
            csvwriter.writerow(col_names)

            # writing the data rows
            csvwriter.writerows(zip(*all_output))

        # Feature states to csv
        relative_path_feature_states_csv = f"{self.config['results']['path'] + '/' + 'simulated_feature_states.csv'}"
        path_feature_states_csv = fix_relative_path(relative_path_feature_states_csv, self.base_directory)

        with open(path_feature_states_csv, 'w', newline='') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            # writing the fields
            csvwriter.writerow(feature_col_names)

            # writing the data rows
            csvwriter.writerows(list(itertools.zip_longest(*available_states)))


def main(config_path: PathLike):
    """Main interface for sBayes simulation"""
    sim = Simulation()
    sim.load_config_simulation(config_file=config_path)

    # Simulate mobility behaviour
    sim.run_simulation()
    sim.write_to_csv()


def cli():
    """Interface allowing to run sBayes simulations from the command line using:
        python -m sbayes.simulation <path_to_config_file>
    """
    import argparse
    parser = argparse.ArgumentParser(description='Simulations for sBayes')
    parser.add_argument('config', type=Path, help='The JSON configuration file')
    args = parser.parse_args()

    main(config_path=args.config)


if __name__ == '__main__':
    cli()
