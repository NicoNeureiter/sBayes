#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Imports the real world data """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import os

from src.preprocessing import compute_network, get_p_families_prior, get_p_global_prior
from src.util import read_features_from_csv, set_experiment_name


class DataImporter:
    def __init__(self):
        # Get parameters from config_simulation.json
        self.config = {}
        self.get_parameters()

        # General setup
        self.experiment_name = set_experiment_name()
        self.TEST_SAMPLING_DIRECTORY = '../results/contact_zones/{experiment}/'. \
            format(experiment=self.experiment_name)
        self.TEST_SAMPLING_RESULTS_PATH = \
            self.TEST_SAMPLING_DIRECTORY + self.config['data']['REGION_ABBR'] + '_contact_zones_nz{nz}_{run}.pkl'
        self.TEST_SAMPLING_LOG_PATH = self.TEST_SAMPLING_DIRECTORY + 'info.log'
        if not os.path.exists(self.TEST_SAMPLING_DIRECTORY):
            os.mkdir(self.TEST_SAMPLING_DIRECTORY)

        # Features to be extracted
        self.sites = None
        self.site_names = None
        self.features = None
        self.feature_names = None
        self.category_names = None
        self.families = None
        self.family_names = None
        self.network = None

        # Prior information
        self.p_global_dirichlet = None
        self.p_global_categories = None
        self.p_families_dirichlet = None
        self.p_families_categories = None

    def get_parameters(self):
        with open('config_data.json', 'r') as f:
            self.config = json.load(f)

    def get_data_features(self):
        self.sites, self.site_names, self.features, self.feature_names, \
            self.category_names, self.families, self.family_names = \
            read_features_from_csv(
                file_location="../../" + self.config['data']['REGION_FULL'] + "/data/features/", log=True)

        self.network = compute_network(self.sites)

    def get_prior_information(self):
        self.p_global_dirichlet, self.p_global_categories = \
            get_p_global_prior(feature_names=self.feature_names,
                               category_names=self.category_names,
                               file_location="../../" + self.config['data']['REGION_FULL'] + "/data/p_global/",
                               log=True)

        self.p_families_dirichlet, self.p_families_categories = \
            get_p_families_prior(family_names=self.family_names,
                                 feature_names=self.feature_names,
                                 category_names=self.category_names,
                                 file_location="../../" + self.config['data']['REGION_FULL'] + "/data/p_families",
                                 log=True)

    def logging_setup(self):
        logging.basicConfig(filename=self.TEST_SAMPLING_LOG_PATH, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Experiment: %s", self.experiment_name)
        logging.info("Inheritance is considered: %s", self.config['data']['INHERITANCE'])
