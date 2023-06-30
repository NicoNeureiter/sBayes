#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Imports the real world data """
from __future__ import annotations

import pyproj
import pandas as pd
import numpy as np
try:
    import ruamel.yaml as yaml
except ImportError:
    import ruamel_yaml as yaml
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from numpy.typing import NDArray
from dataclasses import dataclass, field
from logging import Logger
from collections import OrderedDict
from typing import Optional, TypeVar, Type

from sbayes.preprocessing import ComputeNetwork, read_geo_cost_matrix
from sbayes.util import PathLike, read_data_csv, encode_categorical_data
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
class CategoricalFeatures:
    # Binary representation of all categorical features
    values: NDArray[bool]                     # shape: (n_objects, n_features, n_states)
    states: NDArray[bool]                     # shape: (n_features, n_states)
    names: OrderedDict[FeatureName, list[StateName]] = field(init=False)
    feature_names: NDArray[FeatureName]  # shape: (n_features,)
    state_names: dict[list[StateName]]  # shape for each feature f: (n_states[f],)
    na_number: int
    na_values: NDArray[bool] = field(init=False)                # shape: (n_objects, n_features)

    def __post_init__(self):
        object.__setattr__(self, 'names', OrderedDict())
        for f, s in zip(self.feature_names, self.state_names):
            self.names[f] = s

        object.__setattr__(self, 'na_values', np.sum(self.values, axis=-1) == 0)

    @classmethod
    def from_dataframes(
        cls: Type[S],
        data: pd.DataFrame,
        feature_types: pd.DataFrame,
    ) -> S:

        # Retrieve all categorical features
        categorical_columns = [k for k, v in feature_types.items() if v['type'] == "categorical"]
        categorical_data = data.loc[:, categorical_columns]
        feature_states = dict((c, feature_types[c]['states']) for c in categorical_columns)

        if categorical_data.empty:
            return None
        else:
            categorical_features_dict = encode_categorical_data(categorical_data, feature_states)
            # return Feature class consisting of all binarised categorical features
            return cls(**categorical_features_dict)

    @property
    def n_features(self) -> int:
        return self.values.shape[1]

    @property
    def n_states(self) -> int:
        return self.values.shape[2]

    @property
    def n_states_per_feature(self) -> list[int]:
        return [sum(applicable) for applicable in self.states]


@dataclass
class GaussianFeatures:
    # All features that are continuous measurements
    values: NDArray[float]                          # shape: (n_objects, n_gaussian_features)
    names: NDArray[FeatureName]                     # shape: (n_gaussian_features,)
    na_number: int
    na_values: NDArray[bool] = field(init=False)    # shape: (n_objects, n_features)

    def __post_init__(self):
        object.__setattr__(self, 'na_values', np.sum(self.values, axis=-1) == 0)

    @classmethod
    def from_dataframes(
        cls: Type[S],
        data: pd.DataFrame,
        feature_types: pd.DataFrame,
    ) -> S:

        # Retrieve all gaussian features
        gaussian_columns = [k for k, v in feature_types.items() if v['type'] == "gaussian"]
        gaussian_data = data.loc[:, gaussian_columns]

        if gaussian_data.empty:
            return None
        else:
            gaussian_features_dict = dict(values=gaussian_data.to_numpy(dtype=float, na_value=np.nan),
                                          names=np.asarray(gaussian_columns),
                                          na_number=gaussian_data.isna().sum().sum())

            # return Feature class consisting of all gaussian features
            return cls(**gaussian_features_dict)

    @property
    def n_features(self) -> int:
        return self.values.shape[1]


@dataclass
class PoissonFeatures:
    # All features that are count variables
    values: NDArray[int]                            # shape: (n_objects, n_poisson_features)
    names: NDArray[FeatureName]                     # shape: (n_poisson_features,)
    na_number: int
    na_values: NDArray[bool] = field(init=False)    # shape: (n_objects, n_features)

    def __post_init__(self):
        object.__setattr__(self, 'na_values', np.sum(self.values, axis=-1) == 0)

    @classmethod
    def from_dataframes(
            cls: Type[S],
            data: pd.DataFrame,
            feature_types: pd.DataFrame,
    ) -> S:
        # Retrieve all Poisson features
        poisson_columns = [k for k, v in feature_types.items() if v['type'] == "poisson"]
        poisson_data = data.loc[:, poisson_columns]

        if poisson_data.empty:
            return None
        else:
            poisson_features_dict = dict(values=poisson_data.to_numpy(dtype=float, na_value=np.nan),
                                         names=np.asarray(poisson_columns),
                                         na_number=poisson_data.isna().sum().sum())

            # return Feature class consisting of all poisson features
            return cls(**poisson_features_dict)

    @property
    def n_features(self) -> int:
        return self.values.shape[1]


@dataclass
class LogitNormalFeatures:
    # All features that are percentages
    values: NDArray[int]                            # shape: (n_objects, n_logit_normal_features)
    names: NDArray[FeatureName]                     # shape: (n_logit_normal_features, )
    na_number: int
    na_values: NDArray[bool] = field(init=False)    # shape: (n_objects, n_features)

    def __post_init__(self):
        object.__setattr__(self, 'na_values', np.sum(self.values, axis=-1) == 0)

    @classmethod
    def from_dataframes(
            cls: Type[S],
            data: pd.DataFrame,
            feature_types: pd.DataFrame,
    ) -> S:
        # Retrieve all Poisson features
        logit_normal_columns = [k for k, v in feature_types.items() if v['type'] == "logit-normal"]
        logit_normal_data = data.loc[:, logit_normal_columns]

        if logit_normal_data.empty:
            return None
        else:
            logit_normal_features_dict = dict(values=logit_normal_data.to_numpy(dtype=float, na_value=np.nan),
                                              names=np.asarray(logit_normal_columns),
                                              na_number=logit_normal_data.isna().sum().sum())

            # return Feature class consisting of all logit-normal features
            return cls(**logit_normal_features_dict)

    @property
    def n_features(self) -> int:
        return self.values.shape[1]


@dataclass
class Features:
    categorical: CategoricalFeatures | None
    gaussian: GaussianFeatures | None
    poisson: PoissonFeatures | None
    logitnormal: LogitNormalFeatures | None

    def __getitem__(self, key: str) -> NDArray | list | int:
        return getattr(self, key)

    @property
    def n_features(self) -> int:
        if self.categorical is not None:
            n_categorical = self.categorical.n_features
        else:
            n_categorical = 0
        if self.gaussian is not None:
            n_gaussian = self.gaussian.n_features
        else:
            n_gaussian = 0
        if self.poisson is not None:
            n_poisson = self.poisson.n_features
        else:
            n_poisson = 0
        if self.logitnormal is not None:
            n_logitnormal = self.logitnormal.n_features
        else:
            n_logitnormal = 0

        return n_categorical + n_gaussian + n_poisson + n_logitnormal

    @classmethod
    def from_dataframes(
        cls: Type[S],
        data: pd.DataFrame,
        feature_types: pd.DataFrame,
    ) -> S:

        # Retrieve and one-hot encode all categorical features
        categorical_features = CategoricalFeatures.from_dataframes(data, feature_types)

        # Retrieve all Gaussian features
        gaussian_features = GaussianFeatures.from_dataframes(data, feature_types)

        # Retrieve all Poisson features
        poisson_features = PoissonFeatures.from_dataframes(data, feature_types)

        # Retrieve all logit-normal features
        logit_normal_features = LogitNormalFeatures.from_dataframes(data, feature_types)

        # return Feature class consisting of all different types of features
        return cls(categorical=categorical_features,
                   gaussian=gaussian_features,
                   poisson=poisson_features,
                   logitnormal=logit_normal_features)


@dataclass
class Confounder:

    name: str
    group_assignment: NDArray[bool]         # shape: (n_groups, n_objects)
    group_names: list[GroupName]            # shape: (n_groups,)

    def __getitem__(self, key) -> str | list | NDArray[bool]:
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
        confounder_name: ConfounderName
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

        # self.features_by_group = {
        #     conf_name: [features.values[g] for g in conf.group_assignment]
        #     for conf_name, conf in self.confounders.items()
        # }
        # self.features.features_by_group = self.features_by_group

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
        objects, features, confounders = read_features_from_file(
            data_path=config.data.features,
            feature_types_path=config.data.feature_types,
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


def read_features_from_file(
    data_path: PathLike,
    feature_types_path: PathLike,
    confounder_names: list[ConfounderName],
    logger: Optional[Logger] = None,
) -> (Objects, Features, dict[ConfounderName, Confounder]):
    """This is a helper function to import data (objects, features, confounders) from csv and yaml files
    Args:
        data_path: path to the data csv file.
        feature_types_path: path to the feature types yaml file
        confounder_names: dict mapping confounder name to list of corresponding groups
        logger: A Logger instance for writing log messages.

    Returns:
        The parsed data objects (objects, features and confounders).
    """
    # Load the data and feature types
    data = read_data_csv(data_path)

    with open(feature_types_path, "r") as f:
        yaml_loader = yaml.YAML(typ='safe')
        feature_types = yaml_loader.load(f)

    features = Features.from_dataframes(data, feature_types)
    objects = Objects.from_dataframe(data)
    confounders = OrderedDict()

    for c in confounder_names:
        confounders[c] = Confounder.from_dataframe(data=data, confounder_name=c)

    if logger:
        logger.info(
            f"{objects.n_objects} objects with {features.n_features} features read from {data_path}."
        )
        logger.info(f"Feature types:")
        try:
            logger.info(
                f"Categorical: {features.categorical.n_features} qualitative feature(s) with "
                f"{features.categorical.na_number} NA value(s)"
            )
        except AttributeError:
            pass
        try:
            logger.info(
                f"Gaussian: {features.gaussian.n_features} continuous feature(s) "
                f"{features.gaussian.na_number} NA value(s)"
            )
        except AttributeError:
            pass
        try:
            logger.info(
                f"Poisson: {features.poisson.n_features} count feature(s) with "
                f"{features.poisson.na_number} NA value(s)"
            )
        except AttributeError:
            pass
        try:
            logger.info(
                f"Logit-normal: {features.poisson.n_features} percentage features(s) with "
                f"{features.logitnormal.na_number} NA value(s)"
            )
        except AttributeError:
            pass

    return objects, features, confounders
