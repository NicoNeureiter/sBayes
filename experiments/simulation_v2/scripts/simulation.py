#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        self.network_sim = None
        self.zones_sim = None
        self.features_sim = None
        self.categories_sim = None
        self.families_sim = None
        self.weights_sim = None
        self.p_global_sim = None
        self.p_zones_sim = None
        self.p_families_sim = None

        # Get parameters from config_simulation.json
        self.config = {}
        self.get_parameters()

    def get_parameters(self):
        with open('config_simulation.json', 'r') as f:
            self.config = json.load(f)

    def logging_setup(self):
        logging.basicConfig(filename=self.TEST_SAMPLING_LOG_PATH, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Experiment: %s", self.experiment_name)

    def logging_simulation(self):
        logging.info("Simulating %s features.", self.config['simulation']['N_FEATURES_SIM'])
        logging.info("Simulated global intensity: %s", self.config['simulation']['I_GLOBAL_SIM'])
        logging.info("Simulated contact intensity: %s", self.config['simulation']['I_CONTACT_SIM'])
        logging.info("Simulated inherited intensity: %s", self.config['simulation']['I_CONTACT_SIM'])
        logging.info("Simulated global exposition (number of similar features): %s",
                     self.config['simulation']['F_GLOBAL_SIM'])
        logging.info("Simulated exposition in zone (number of similar features): %s",
                     self.config['simulation']['F_CONTACT_SIM'])
        logging.info("Simulated exposition in family (number of similar features): %s",
                     self.config['simulation']['F_CONTACT_SIM'])
        logging.info("Simulated zone: %s", self.config['simulation']['ZONE'])
        logging.info("Inheritance is simulated: %s", self.config['simulation']['INHERITANCE_SIM'])

    def simulation(self):
        # Simulate zones
        sites_sim, site_names = get_sites("../data/sites_simulation.csv", retrieve_family=True)
        self.network_sim = compute_network(sites_sim)
        self.zones_sim = simulate_zones(zone_id=self.config['simulation']['ZONE'], sites_sim=sites_sim)

        # Simulate families
        self.families_sim = simulate_families(fam_id=1, sites_sim=sites_sim)

        # Simulate weights, i.e. the influence of global bias, contact and inheritance on each feature
        self.weights_sim = simulate_weights(f_global=self.config['simulation']['F_GLOBAL_SIM'],
                                            f_contact=self.config['simulation']['F_CONTACT_SIM'],
                                            f_inheritance=self.config['simulation']['F_INHERITANCE_SIM'],
                                            inheritance=self.config['simulation']['INHERITANCE_SIM'],
                                            n_features=self.config['simulation']['N_FEATURES_SIM'])

        # Simulate probabilities for features to belong to categories, globally in zones (and in families if available)
        self.p_global_sim, self.p_zones_sim, self.p_families_sim \
            = simulate_assignment_probabilities(n_features=self.config['simulation']['N_FEATURES_SIM'],
                                                p_number_categories=self.config['simulation']['P_N_CATEGORIES_SIM'],
                                                zones=self.zones_sim, families=self.families_sim,
                                                intensity_global=self.config['simulation']['I_GLOBAL_SIM'],
                                                intensity_contact=self.config['simulation']['I_CONTACT_SIM'],
                                                intensity_inheritance=self.config['simulation']['I_INHERITANCE_SIM'],
                                                inheritance=self.config['simulation']['INHERITANCE_SIM'])

        # Simulate features
        self.features_sim, self.categories_sim = \
            simulate_features(zones=self.zones_sim,
                              families=self.families_sim,
                              p_global=self.p_global_sim,
                              p_contact=self.p_zones_sim,
                              p_inheritance=self.p_families_sim,
                              weights=self.weights_sim,
                              inheritance=self.config['simulation']['INHERITANCE_SIM'])
