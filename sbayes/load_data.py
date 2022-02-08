#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Imports the real world data """

import pyproj
from dataclasses import dataclass
import typing as t
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import logging

import numpy

from sbayes.util import read_features_from_csv
from sbayes.preprocessing import (compute_network,
                                  read_inheritance_counts,
                                  read_universal_counts,
                                  read_geo_cost_matrix)


class Data:
    def __init__(self, experiment):

        self.path_results = experiment.path_results
        self.experiment_name = experiment.experiment_name

        # Config file
        self.config = experiment.config

        proj4_string = experiment.config['data'].get('projection')
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
        self.log_load_geo_cost_matrix = None

        # Priors to be imported
        self.prior_universal = {}
        self.prior_inheritance = {}
        self.geo_prior = {}

        # Not a simulation
        self.is_simulated = False

    def load_features(self):
        (self.sites, self.site_names, self.features, self.feature_names,
         self.state_names, self.states, self.families, self.family_names,
         self.log_load_features) = read_features_from_csv(file=self.config['data']['features'],
                                                          feature_states_file=self.config['data']['feature_states'])
        self.network = compute_network(self.sites, crs=self.crs)

    def load_universal_counts(self):
        config_universal = self.config['model']['prior']['universal']

        if config_universal['type'] == 'uniform':
            return

        if config_universal['type'] == 'dirichlet':
            if 'parameters' in config_universal:
                print("read parameters")
            if 'file' in config_universal:
                print("read file")
            # TODO: read JSON
            # counts, self.log_load_universal_counts = \
            #     read_universal_counts(feature_names=self.feature_names,
            #                           state_names=self.state_names,
            #                           file=config_universal['file'],
            #                           file_type=config_universal['file_type'],
            #                           feature_states_file=self.config['data']['FEATURE_STATES'])
            #
            # self.prior_universal = {'counts': counts,
            #                     'states': self.states}

        # import pandas as pd
        # n_states = counts.shape[-1]
        # df = pd.DataFrame(counts,
        #                   index=self.feature_names['external'])
        # df.to_csv('universal_counts.csv')


    def load_inheritance_counts(self):
        if not self.config['model']['inheritance']:
            # Inheritance is not modeled -> nothing to do
            return

        config_inheritance = self.config['model']['prior']['inheritance']

        for fam in config_inheritance:
            if config_inheritance[fam]['type'] == 'uniform':
                return
            if config_inheritance[fam]['type'] == 'dirichlet':
                if 'parameters' in config_inheritance[fam]:
                    print("read parameters")
                if 'file' in config_inheritance[fam]:
                    print("read file")

                # counts, self.log_load_inheritance_counts = \
                #     read_inheritance_counts(family_names=self.family_names,
                #                             feature_names=self.feature_names,
                #                             state_names=self.state_names,
                #                             files=config_inheritance['files'],
                #                             file_type=config_inheritance['file_type'],
                #                             feature_states_file=self.config['data']['FEATURE_STATES'])
                #
                # self.prior_inheritance = {'counts': counts,
                #                           'states': self.state_names['internal']}


    def load_geo_cost_matrix(self):
        geo_prior_cfg = self.config['model']['prior']['geo']
        if geo_prior_cfg['type'] != 'cost_based':
            # Geo prior is not cost-based -> nothing to do
            return

        if 'costs' not in geo_prior_cfg:
            geo_prior_cfg['costs'] = 'from_data'

        if geo_prior_cfg['costs'] == 'from_data':
            # No cost-matrix given. Use distance matrix as costs
            geo_cost_matrix = self.network['dist_mat']

        else:
            # Read cost matrix from data
            geo_cost_matrix, self.log_load_geo_cost_matrix =\
                read_geo_cost_matrix(site_names=self.site_names,
                                     file=geo_prior_cfg['costs'])

        self.geo_prior = {'cost_matrix': geo_cost_matrix}

    def log_loading(self):
        log_path = self.path_results / 'experiment.log'
        logging.basicConfig(format='%(message)s', filename=log_path, level=logging.DEBUG)
        logging.info("\n")
        logging.info("DATA IMPORT")
        logging.info("##########################################")
        logging.info(self.log_load_features)
        logging.info(self.log_load_universal_counts)
        logging.info(self.log_load_inheritance_counts)
        logging.info(self.log_load_geo_cost_matrix)


@dataclass
class Prior:
    counts: t.Any
    states: t.Any

    def __getitem__(self, item: Literal["counts", "states"]):
        if item == "counts":
            return self.counts
        elif item == "states":
            return self.states
        else:
            raise AttributeError(
                f"{item:} is no valid attribute of a count prior"
            )

    def __setitem__(self, item: Literal["counts", "states"], value: t.Any):
        if item == "counts":
            self.counts = value
        elif item == "states":
            self.states = value
        else:
            raise AttributeError(
                f"{item:} is no valid attribute of a count prior"
            )

@dataclass
class Sites:
    id: t.Any
    locations: t.Tuple[float, float]
    names: t.Any

    def __getitem__(self, item: Literal["id"]):
        if item == "id":
            return self.id
        elif item == "locations":
            return self.locations
        elif item == "names":
            return self.names
        else:
            raise AttributeError(
                f"{item:} is no valid attribute of Sites"
            )

    def __setitem__(self, item: Literal["id"], value: t.Any):
        if item == "id":
            self.id = value
        elif item == "locations":
            self.locations = value
        elif item == "names":
            self.names = value
        else:
            raise AttributeError(
                f"{item:} is no valid attribute of Sites"
            )


class CLDFData(Data):
    def __init__(self, experiment):
        super().__init__(experiment)
        self.ds = self.config['data']['cldf_dataset']

    def load_features(self):
        self.features = []
        c_id = self.ds["ParameterTable", 'id'].name
        for feature in self.ds["ParameterTable"]:
            self.features.append(feature[c_id])
        self.features = numpy.array([[1, 2]])

        c_id = self.ds["LanguageTable", 'id'].name
        c_name = self.ds["LanguageTable", 'name'].name
        c_lon = self.ds["LanguageTable", 'longitude'].name
        c_lat = self.ds["LanguageTable", 'latitude'].name
        self.sites = Sites(*zip(*
            [(site[c_id], (site[c_lon], site[c_lat]), site[c_name])
             for site in self.ds["LanguageTable"]]))
        self.network = compute_network(self.sites)

    def load_universal_counts(self):
        config_universal = self.config['model']['prior']['universal']

        if config_universal['type'] != 'counts':
            return

        counts, self.log_load_universal_counts = numpy.array([[0, 1]]), False

        self.prior_universal = Prior(counts=counts, states=numpy.array([[0, 1]]))

    def load_inheritance_counts(self):
        if not self.config['model']['inheritance']:
            # Inheritance is not modeled -> nothing to do
            return

        config_inheritance = self.config['model']['prior']['inheritance']
        if config_inheritance['type'] != 'counts':
            # Inharitance prior does not use counts -> nothing to do
            return

        counts, self.log_load_inheritance_counts = numpy.array([[[0, 1]]]), False

        self.prior_inheritance = Prior(counts=counts, states=numpy.array([[0, 1]]))
