#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Imports the real world data """

import pyproj
from dataclasses import dataclass
import typing as typ
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


from sbayes.util import read_features_from_csv
from sbayes.preprocessing import ComputeNetwork, read_geo_cost_matrix


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

        self.objects = None
        self.features = None
        self.confounders = None
        self.network = None

        # Priors to be imported
        self.prior_confounders = {}
        self.geo_cost_matrix = None

        self.logger = experiment.logger

    def load_features(self):
        self.objects, self.features, self.confounders = read_features_from_csv(config=self.config, logger=self.logger)
        self.network = ComputeNetwork(self.objects, crs=self.crs)

    def load_geo_cost_matrix(self):
        geo_prior_cfg = self.config['model']['prior']['geo']
        if geo_prior_cfg['type'] != 'cost_based':
            # Geo prior is not cost-based -> nothing to do
            return

        if 'costs' not in geo_prior_cfg:
            geo_prior_cfg['costs'] = 'from_data'

        if geo_prior_cfg['costs'] == 'from_data':
            # No cost-matrix given. Use distance matrix as costs
            self.geo_cost_matrix = self.network['dist_mat']

        else:
            # Read cost matrix from data
            self.geo_cost_matrix = read_geo_cost_matrix(site_names=self.objects,
                                                   file=geo_prior_cfg['costs'],
                                                   logger=self.logger)


@dataclass
class Prior:
    counts: typ.Any
    states: typ.Any

    def __getitem__(self, item: Literal["counts", "states"]):
        if item == "counts":
            return self.counts
        elif item == "states":
            return self.states
        else:
            raise AttributeError(
                f"{item:} is no valid attribute of a count prior"
            )

    def __setitem__(self, item: Literal["counts", "states"], value: typ.Any):
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
    id: typ.Any
    locations: typ.Tuple[float, float]
    names: typ.Any

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

    def __setitem__(self, item: Literal["id"], value: typ.Any):
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
