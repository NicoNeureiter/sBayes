#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Imports the real world data """
from __future__ import annotations
import pandas as pd
import pyproj
from dataclasses import dataclass
from logging import Logger
from collections import OrderedDict
from typing import List, Dict, Optional, Union, Sequence


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from numpy.typing import NDArray

from sbayes.preprocessing import ComputeNetwork, read_geo_cost_matrix
from sbayes.util import PathLike, read_data_csv, encode_states


@dataclass
class Objects:

    """Container class for a set of objects. Each object describes one sample (a language,
    person, state,...) which has an ID, name and location."""

    id: Sequence[int]
    locations: NDArray[float]  # shape: (n_sites, 2)
    names: Sequence[str]

    def __getitem__(self, key):
        return getattr(self, key)

    _indices: NDArray[int] = None

    @property
    def indices(self) -> NDArray[int]:
        if self._indices is None:
            self._indices = np.arange(len(self.id))
        return self._indices

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> "Objects":
        n_sites = data.shape[0]
        try:
            x = data["x"]
            y = data["y"]
            id_ext = data["id"].tolist()
            # id_ext = np.arange(n_sites)
        except KeyError:
            raise KeyError("The csv must contain columns `x`, `y` and `id`")

        locations = np.zeros((n_sites, 2))
        for i in range(n_sites):
            # Define location tuples
            locations[i, 0] = float(x[i])
            locations[i, 1] = float(y[i])

        objects_dict = {
            "locations": locations,
            "id": id_ext,
            "names": list(data.get("name", id_ext)),
        }
        return cls(**objects_dict)


@dataclass(frozen=True)
class Features:

    values: NDArray[bool]  # shape: (n_sites, n_features, n_states)
    names: NDArray[str]  # shape: (n_features,)
    states: NDArray[bool]  # shape: (n_features, n_states)
    state_names: List[List[str]]  # shape for each feature f: (n_states[f],)
    na_number: int

    def __getitem__(self, key: str) -> Union[NDArray, List, int]:
        return getattr(self, key)

    @property
    def n_sites(self) -> int:
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        return self.values.shape[1]

    @property
    def n_states_per_feature(self) -> List[int]:
        return [sum(applicable) for applicable in self.states]

    @classmethod
    def from_dataframes(
        cls: type,
        data: pd.DataFrame,
        feature_states: pd.DataFrame,
    ):
        feature_data = data.loc[:, feature_states.columns]
        features_dict, na_number = encode_states(feature_data, feature_states)
        features_dict["names"] = feature_states.columns.to_numpy()
        return cls(**features_dict, na_number=na_number)


@dataclass(frozen=True)
class Confounder:
    name: str
    group_assignment: NDArray[bool]  # shape: (n_groups, n_sites)
    group_names: NDArray[str]  # shape: (n_groups,)

    def __getitem__(self, key):
        if key == "names":
            return self.group_names
        elif key == "values":
            return self.group_assignment
        return getattr(self, key)

    def any_group(self) -> NDArray[bool]:  # shape: (n_groups,)
        return np.any(self.group_assignment, axis=0)

    @property
    def n_groups(self):
        return len(self.group_names)

    @classmethod
    def from_dataframe(
        cls: type,
        confounder_name: str,
        group_names: List[str],
        data: pd.DataFrame,
    ):
        n_sites = data.shape[0]

        if confounder_name not in data:
            if len(group_names) == 1 and group_names[0] == "<ALL>":
                # Special case: this effect applies to all objects in the same way and
                # does not require a separate column in the data file.
                group_assignment = np.ones((1, n_sites), dtype=bool)
                return cls(
                    name=confounder_name,
                    group_assignment=group_assignment,
                    group_names=group_names,
                )

            else:
                raise KeyError(
                    f"The config file lists '{confounder_name}' as a confounder. Remove "
                    f"confounder or include '{confounder_name}' in the features.csv file."
                )

        group_names_by_site = data[confounder_name]
        group_names = list(np.unique(group_names_by_site.dropna()))
        group_assignment = np.zeros((len(group_names), n_sites), dtype=bool)
        for g, g_name in enumerate(group_names):
            group_assignment[g, np.where(group_names_by_site == g_name)] = True

        return cls(
            name=confounder_name,
            group_assignment=group_assignment,
            group_names=group_names,
        )


class Data:

    """Container and loading functionality for different types of data involved in a
    sBayes analysis.
    """

    objects: Objects
    features: Features
    confounders: OrderedDict[str, Confounder]
    prior_confounders: ...
    crs: Optional[pyproj.CRS]
    geo_cost_matrix: Optional[NDArray[float]]
    network: ComputeNetwork
    logger: Logger

    def __init__(
        self,
        objects: Objects,
        features: Features,
        confounders: OrderedDict[str, Confounder],
        projection: Optional[str] = "epsg:4326",
        geo_costs: Union[Literal["from_data"], PathLike] = "from_data",
        logger: Logger = None,
    ):
        self.objects = objects
        self.features = features
        self.confounders = confounders
        self.logger = logger

        self.crs = pyproj.CRS(projection)
        self.network = ComputeNetwork(self.objects, crs=self.crs)

        if geo_costs == "from_data":
            self.geo_cost_matrix = self.network.dist_mat
        else:
            self.geo_cost_matrix = read_geo_cost_matrix(
                object_names=self.objects.names, file=geo_costs, logger=self.logger
            )

        self.prior_confounders = None  # TODO Load the confounder priors

    @classmethod
    def from_config(cls, config: "SBayesConfig", logger=None) -> "Data":
        if logger:
            cls.log_loading(logger)

        # Load objects, features, confounders
        objects, features, confounders = read_features_from_csv(
            data_path=config.data.features,
            feature_states_path=config.data.feature_states,
            groups_by_confounder=config.model.confounders,
            logger=logger,
        )

        # Create a Data object using __init__
        return cls(
            objects=objects,
            features=features,
            confounders=confounders,
            projection=config.data.projection,
            geo_costs=config.model.prior.geo.costs,
            logger=logger,
        )

    @classmethod
    def from_experiment(cls, experiment: "Experiment") -> "Data":
        return cls.from_config(experiment.config, logger=experiment.logger)

    def load_features(self):
        # Empty function for backwards compatibility. Loading now happens in initialization
        # TODO remove
        pass

    def load_geo_cost_matrix(self):
        # Empty function for backwards compatibility. Loading now happens in initialization
        # TODO remove
        pass

    @staticmethod
    def log_loading(logger):
        logger.info("\n")
        logger.info("DATA IMPORT")
        logger.info("##########################################")


@dataclass(frozen=True)
class Prior:
    counts: NDArray[int]
    states: List

    def __getitem__(self, key):
        return getattr(self, key)


def read_features_from_csv(
    data_path: PathLike,
    feature_states_path: PathLike,
    groups_by_confounder: Dict[str, list],
    logger: Optional[Logger] = None,
) -> (Objects, Features, OrderedDict[str, Confounder]):
    """This is a helper function to import data (objects, features, confounders) from a csv file
    Args:
        data_path: path to the data csv file.
        feature_states_path: path to the feature states csv file.
        groups_by_confounder: dict mapping confounder name to list of corresponding groups
        logger: A Logger instance for writing log messages.

    Returns:
        The parsed data objects (objects, features and confounders).
    """
    # Load the data and features-states
    data = read_data_csv(data_path)
    feature_states = read_data_csv(feature_states_path)

    features = Features.from_dataframes(data, feature_states)
    objects = Objects.from_dataframe(data)
    confounders = OrderedDict()
    for c, groups in groups_by_confounder.items():
        confounders[c] = Confounder.from_dataframe(confounder_name=c, group_names=groups, data=data)

    if logger:
        logger.info(
            f"{features.n_sites} sites with {features.n_features} features read from {data_path}."
        )
        logger.info(f"{features.na_number} NA value(s) found.")
        logger.info(
            f"The maximum number of states in a single feature was {feature_states.shape[0]}."
        )

    return objects, features, confounders
