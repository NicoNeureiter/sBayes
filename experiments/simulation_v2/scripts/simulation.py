#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Defines the class ContactZonesSimulator 
    Outputs the parameters needed for the MCMC process:
    network, zones, features, 
    categories, families, weights, 
    p_global, p_zones, p_families """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import os

from src.preprocessing import (compute_network, get_sites,
                               simulate_assignment_probabilities,
                               simulate_families,
                               simulate_features,
                               simulate_weights,
                               simulate_zones)
from src.util import set_experiment_name


class ContactZonesSimulator:
    def __init__(self):
        # Get parameters from config_simulation.json
        self.config = {}
        self.get_parameters()

        # General setup
        self.experiment_name = set_experiment_name()
        # we will have other directories and folders on line 43
        self.TEST_SAMPLING_DIRECTORY = '../results/contact_zones/{experiment}/'.\
            format(experiment=self.experiment_name)
        self.TEST_SAMPLING_RESULTS_PATH = self.TEST_SAMPLING_DIRECTORY + 'contact_zones_i{i}_{run}.pkl'
        self.TEST_SAMPLING_LOG_PATH = self.TEST_SAMPLING_DIRECTORY + 'info.log'
        if not os.path.exists(self.TEST_SAMPLING_DIRECTORY):
            os.mkdir(self.TEST_SAMPLING_DIRECTORY)

        # Simulation variables to be calculated
        self.network = None
        self.zones = None
        self.features = None
        self.categories = None
        self.families = None
        self.weights = None
        self.p_global = None
        self.p_zones = None
        self.p_families = None

    def get_parameters(self):
        with open('config_simulation.json', 'r') as f:
            self.config = json.load(f)

    def logging_setup(self):
        logging.basicConfig(filename=self.TEST_SAMPLING_LOG_PATH, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Experiment: %s", self.experiment_name)
        logging.info("Inheritance is simulated: %s", self.config['simulation']['INHERITANCE'])

    def logging_simulation(self):
        logging.info("Simulating %s features.", self.config['simulation']['N_FEATURES'])
        logging.info("Simulated global intensity: %s", self.config['simulation']['I_GLOBAL'])
        logging.info("Simulated contact intensity: %s", self.config['simulation']['I_CONTACT'])
        logging.info("Simulated inherited intensity: %s", self.config['simulation']['I_CONTACT'])
        logging.info("Simulated global exposition (number of similar features): %s",
                     self.config['simulation']['F_GLOBAL'])
        logging.info("Simulated exposition in zone (number of similar features): %s",
                     self.config['simulation']['F_CONTACT'])
        logging.info("Simulated exposition in family (number of similar features): %s",
                     self.config['simulation']['F_CONTACT'])
        logging.info("Simulated zone: %s", self.config['simulation']['ZONE'])

    def simulation(self):
        # Simulate zones
        sites, site_names = get_sites("../data/sites_simulation.csv", retrieve_family=True)
        self.network = compute_network(sites)
        self.zones = simulate_zones(zone_id=self.config['simulation']['ZONE'], sites_sim=sites)

        # Simulate families
        self.families = simulate_families(fam_id=1, sites_sim=sites)

        # Simulate weights, i.e. the influence of global bias, contact and inheritance on each feature
        self.weights = simulate_weights(f_global=self.config['simulation']['F_GLOBAL'],
                                        f_contact=self.config['simulation']['F_CONTACT'],
                                        f_inheritance=self.config['simulation']['F_INHERITANCE'],
                                        inheritance=self.config['simulation']['INHERITANCE'],
                                        n_features=self.config['simulation']['N_FEATURES'])

        # Simulate probabilities for features to belong to categories, globally in zones (and in families if available)
        self.p_global, self.p_zones, self.p_families \
            = simulate_assignment_probabilities(n_features=self.config['simulation']['N_FEATURES'],
                                                p_number_categories=self.config['simulation']['P_N_CATEGORIES'],
                                                zones=self.zones, families=self.families,
                                                intensity_global=self.config['simulation']['I_GLOBAL'],
                                                intensity_contact=self.config['simulation']['I_CONTACT'],
                                                intensity_inheritance=self.config['simulation']['I_INHERITANCE'],
                                                inheritance=self.config['simulation']['INHERITANCE'])

        # Simulate features
        # Note: categories are not used in the further MCMC setup
        self.features, self.categories = \
            simulate_features(zones=self.zones,
                              families=self.families,
                              p_global=self.p_global,
                              p_contact=self.p_zones,
                              p_inheritance=self.p_families,
                              weights=self.weights,
                              inheritance=self.config['simulation']['INHERITANCE'])
