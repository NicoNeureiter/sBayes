#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Imports the real world data """
from __future__ import annotations
import pyproj
from dataclasses import dataclass, field
from logging import Logger
from collections import OrderedDict
from typing import Optional, TypeVar, Type

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from sbayes.preprocessing import ComputeNetwork, read_geo_cost_matrix
from sbayes.util import PathLike, read_data_csv, encode_states
from sbayes.config.config import SBayesConfig
from sbayes.experiment_setup import Experiment

# Type variables and constants
S = TypeVar('S')  # Self type
ObjectName = TypeVar('ObjectName', bound=str)
ObjectID = TypeVar('ObjectID', bound=str)
FeatureName = TypeVar('FeatureName', bound=str)
StateName = TypeVar('StateName', bound=str)
ConfounderName = TypeVar('ConfounderName', bound=str)
GroupName = TypeVar('GroupName', bound=str)


@dataclass
class Objects:

    """Container class for a set of objects. Each object describes one sample (a language,
    person, state,...) which has an ID, name and location."""

    id: list[ObjectID]
    locations: NDArray[float]  # shape: (n_objects, 2)
    names: list[ObjectName]
    indices: NDArray[int] = field(init=False)  # shape: (n_objects,)

    def __post_init__(self):
        setattr(self, 'indices', np.arange(self.n_objects))

    def __getitem__(self, key) -> list | NDArray:
        return getattr(self, key)

    _indices: NDArray[int] = None

    @property
    def n_objects(self):
        return len(self.id)

    def __len__(self):
        return len(self.id)

    @classmethod
    def from_dataframe(cls: Type[S], data: pd.DataFrame) -> S:
        n_objects = data.shape[0]
        try:
            x = data["x"]
            y = data["y"]
            id_ext = data["id"].tolist()
        except KeyError:
            raise KeyError("The csv must contain columns `x`, `y` and `id`")

        locations = np.zeros((n_objects, 2))
        for i in range(n_objects):
            # Define location tuples
            locations[i, 0] = float(x[i])
            locations[i, 1] = float(y[i])

        objects_dict = {
            "locations": locations,
            "id": id_ext,
            "names": list(data.get("name", id_ext)),
        }
        return cls(**objects_dict)


@dataclass
class Features:

    values: NDArray[bool]  # shape: (n_objects, n_features, n_states)
    names: NDArray[FeatureName]  # shape: (n_features,)
    states: NDArray[bool]  # shape: (n_features, n_states)
    state_names: list[list[StateName]]  # shape for each feature f: (n_states[f],)
    na_number: int
    features_by_group = None

    feature_and_state_names: OrderedDict[FeatureName, list[StateName]] = field(init=False)
    # TODO This could replace names and state_names

    na_values: NDArray[bool] = field(init=False)  # shape: (n_objects, n_features)

    def __post_init__(self):
        object.__setattr__(self, 'feature_and_state_names', OrderedDict())
        for f, states_names_f in zip(self.names, self.state_names):
            self.feature_and_state_names[f] = states_names_f

        object.__setattr__(self, 'na_values', np.sum(self.values, axis=-1) == 0)

    def __getitem__(self, key: str) -> NDArray | list | int:
        return getattr(self, key)

    @property
    def n_objects(self) -> int:
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        return self.values.shape[1]

    @property
    def n_states(self) -> int:
        return self.values.shape[2]

    @property
    def n_states_per_feature(self) -> list[int]:
        return [sum(applicable) for applicable in self.states]

    @classmethod
    def from_dataframes(
        cls: Type[S],
        data: pd.DataFrame,
        feature_states: pd.DataFrame,
    ) -> S:
        feature_data = data.loc[:, feature_states.columns]
        features_dict, na_number = encode_states(feature_data, feature_states)
        features_dict["names"] = feature_states.columns.to_numpy()
        return cls(**features_dict, na_number=na_number)


@dataclass
class Confounder:

    name: str
    group_assignment: NDArray[bool]         # shape: (n_groups, n_objects)
    group_names: NDArray[GroupName]         # shape: (n_groups,)
    feature_counts: NDArray[int] = None    # shape: (n_groups, n_features, n_states)

    def __getitem__(self, key) -> str | NDArray:
        if key == "names":
            return self.group_names
        elif key == "values":
            return self.group_assignment
        return getattr(self, key)

    def any_group(self) -> NDArray[bool]:  # shape: (n_groups,)
        return np.any(self.group_assignment, axis=0)

    @property
    def n_groups(self) -> int:
        return len(self.group_names)

    @classmethod
    def from_dataframe(
        cls: Type[S],
        data: pd.DataFrame,
        confounder_name: ConfounderName,
        features: Features | None = None,
    ) -> S:
        n_objects = data.shape[0]

        if confounder_name not in data:
            # If there is no column specifying the group assignment for the confounder, it
            # is assumed to apply to all objects in the same way.
            group_assignment = np.ones((1, n_objects), dtype=bool)
            group_names = ["<ALL>"]
        else:
            group_names_by_site = data[confounder_name]
            group_names = list(np.unique(group_names_by_site.dropna()))
            group_assignment = np.zeros((len(group_names), n_objects), dtype=bool)
            for i_g, name_g in enumerate(group_names):
                group_assignment[i_g, np.where(group_names_by_site == name_g)] = True

        if features is not None:
            n_groups = len(group_names)
            feature_counts = np.zeros((n_groups, features.n_features, features.n_states), dtype=int)
            for i_g, g in enumerate(group_assignment):
                feature_counts[i_g] = np.sum(features.values[g], axis=0)

        else:
            feature_counts = None
            approximate_likelihood = None

        return cls(
            name=confounder_name,
            group_assignment=group_assignment,
            group_names=group_names,
            feature_counts=feature_counts,
        )


class Data:

    """Container and loading functionality for different types of data involved in a
    sBayes analysis.
    """

    objects: Objects
    features: Features
    confounders: OrderedDict[str, Confounder]
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
        geo_costs: Literal["from_data"] | PathLike = "from_data",
        logger: Logger = None,
    ):
        self.objects = objects
        self.features = features
        self.confounders = confounders
        self.logger = logger

        self.features_by_group = {
            conf_name: [features.values[g] for g in conf.group_assignment]
            for conf_name, conf in self.confounders.items()
        }
        self.features.features_by_group = self.features_by_group

        self.crs = pyproj.CRS(projection)
        self.network = ComputeNetwork(self.objects, crs=self.crs)

        if geo_costs == "from_data":
            self.geo_cost_matrix = self.network.dist_mat
        else:
            self.geo_cost_matrix = read_geo_cost_matrix(
                object_names=self.objects.id, file=geo_costs, logger=self.logger
            )

    @classmethod
    def from_config(cls: Type[S], config: SBayesConfig, logger=None) -> S:
        if logger:
            cls.log_loading(logger)

        # Load objects, features, confounders
        objects, features, confounders = read_features_from_csv(
            data_path=config.data.features,
            feature_states_path=config.data.feature_states,
            confounder_names=config.model.confounders,
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
    def from_experiment(cls: Type[S], experiment: Experiment) -> S:
        return cls.from_config(experiment.config, logger=experiment.logger)

    @staticmethod
    def log_loading(logger):
        logger.info("\n")
        logger.info("DATA IMPORT")
        logger.info("##########################################")


# @dataclass(frozen=True)
# class PriorCounts:
#     counts: NDArray[int]
#     states: list[...]
#
#     def __getitem__(self, key: str):
#         return getattr(self, key)
#
#
# def parse_prior_counts(
#     counts: dict[FeatureName, dict[StateName, int]],
#     features: Features,
# ) -> PriorCounts:
#     ...
#     return PriorCounts(
#         counts=...,
#         states=...,
#     )


def read_features_from_csv(
    data_path: PathLike,
    feature_states_path: PathLike,
    confounder_names: list[ConfounderName],
    logger: Optional[Logger] = None,
) -> (Objects, Features, dict[ConfounderName, Confounder]):
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
    for c in confounder_names:
        confounders[c] = Confounder.from_dataframe(data=data, confounder_name=c, features=features)

    if logger:
        logger.info(
            f"{features.n_objects} objects with {features.n_features} features read from {data_path}."
        )
        logger.info(f"{features.na_number} NA value(s) found.")
        logger.info(
            f"The maximum number of states in a single feature was {feature_states.shape[0]}."
        )

    return objects, features, confounders
