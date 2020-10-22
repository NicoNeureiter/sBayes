#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Imports the real world data """

from __future__ import absolute_import, division, print_function, unicode_literals

import pyproj

import logging

from sbayes.util import read_features_from_csv
from sbayes.preprocessing import (compute_network,
                                  read_inheritance_counts,
                                  read_universal_counts)


class Data:
    def __init__(self, experiment):

        self.path_results = experiment.path_results
        self.experiment_name = experiment.experiment_name

        # Config file
        self.config = experiment.config

        proj4_string = experiment.config['data'].get('CRS')
        if proj4_string is None:
            self.crs = None
        else:
            self.crs = pyproj.CRS(proj4_string)

        # Features to be imported
        self.sites = None
        self.site_names = None
        self.features = None
        self.feature_names = None
        self.states = None
        self.state_names = None
        self.families = None
        self.family_names = None
        self.network = None

        # Logs
        self.log_load_features = None
        self.log_load_universal_counts = None
        self.log_load_inheritance_counts = None

        # Priors to be imported
        self.prior_universal = {}
        self.prior_inheritance = {}

        # Not a simulation
        self.is_simulated = False

    def load_features(self):
        (self.sites, self.site_names, self.features, self.feature_names,
         self.state_names, self.states, self.families, self.family_names,
         self.log_load_features) = read_features_from_csv(file=self.config['data']['FEATURES'],
                                                          feature_states_file=self.config['data']['FEATURE_STATES'])
        self.network = compute_network(self.sites, crs=self.crs)

    def load_universal_counts(self):
        config_universal = self.config['model']['PRIOR']['universal']

        if config_universal['type'] != 'counts':
            return

        counts, self.log_load_universal_counts = \
            read_universal_counts(feature_names=self.feature_names,
                                  state_names=self.state_names,
                                  file=config_universal['file'],
                                  file_type=config_universal['file_type'],
                                  feature_states_file=self.config['data']['FEATURE_STATES'])

        self.prior_universal = {'counts': counts,
                                'states': self.states}

        import pandas as pd
        n_states = counts.shape[-1]
        df = pd.DataFrame(counts,
                          index=self.feature_names['external'])
        df.to_csv('universal_counts.csv')

    def load_inheritance_counts(self):
        if self.config['model']['INHERITANCE'] == False:
            # Inheritance is not modeled -> nothing to do
            return

        config_inheritance = self.config['model']['PRIOR']['inheritance']
        if config_inheritance['type'] != 'counts':
            # Inharitance prior does not use counts -> nothing to do
            return

        counts, self.log_load_inheritance_counts = \
            read_inheritance_counts(family_names=self.family_names,
                                    feature_names=self.feature_names,
                                    state_names=self.state_names,
                                    files=config_inheritance['files'],
                                    file_type=config_inheritance['file_type'],
                                    feature_states_file=self.config['data']['FEATURE_STATES'])

        self.prior_inheritance = {'counts': counts,
                                  'states': self.state_names['internal']}

    def log_loading(self):
        log_path = self.path_results + 'experiment.log'
        logging.basicConfig(format='%(message)s', filename=log_path, level=logging.DEBUG)
        logging.info("\n")
        logging.info("DATA IMPORT")
        logging.info("##########################################")
        logging.info(self.log_load_features)
        logging.info(self.log_load_universal_counts)
        logging.info(self.log_load_inheritance_counts)
