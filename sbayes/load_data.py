""" Imports the real world data """
from __future__ import annotations

from enum import Enum

import pyproj
from dataclasses import dataclass, field
from logging import Logger
from collections import OrderedDict
from typing import Literal, Optional, TypeVar, Type, Iterator

import pandas as pd
import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
from scipy.special import logit

try:
    import ruamel.yaml as yaml
except ImportError:
    import ruamel_yaml as yaml

from sbayes.preprocessing import ComputeNetwork, read_geo_cost_matrix
from sbayes.util import PathLike, read_data_csv, encode_states, EPS
from sbayes.config.config import SBayesConfig
from sbayes.experiment_setup import Experiment

# Type variables and constants for better readability
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


class GenericTypeFeatures:

    """Super class for features of a specific type."""

    values: jnp.array                               # shape: (n_objects, n_features)
    feature_indices: jnp.array                      # shape: (n_features,)
    names: NDArray[FeatureName]                     # shape: (n_features,)
    na_values: NDArray[bool]                        # shape: (n_objects, n_features)

    # For caching the feature values assigned to each confounder group
    _confounder_features: dict[ConfounderName, NDArray[bool]]

    def __init__(self, values: NDArray[bool], feature_indices: NDArray[bool], names: NDArray[FeatureName], na_values: NDArray[bool]):
        self.values = jnp.array(values)
        self.feature_indices = jnp.array(feature_indices)
        self.names = names
        self.na_values = na_values
        self._confounder_features = {}

    @property
    def n_objects(self) -> int:
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        return self.values.shape[1]

    @property
    def na_number(self) -> int:
        return np.sum(self.na_values)

    def get_confounder_features(self, confounder: Confounder) -> jnp.array:
        if confounder.name in self._confounder_features:
            return self._confounder_features[confounder.name]
        else:
            group_assignments = confounder.group_assignment
            conf_features = self.values[:, self.feature_indices]
            self._confounder_features[confounder.name] = conf_features
            return conf_features

    @property
    def name(self):
        raise NotImplementedError


class CategoricalFeatures(GenericTypeFeatures):

    """Integer representation of categorical features."""

    state_names: NDArray[StateName]         # (n_features, n_states)
    state_names_dict: dict[FeatureName, NDArray[StateName]]

    NA: int = -1

    def __init__(self, values: NDArray[bool], feature_indices: NDArray[bool], names: NDArray[FeatureName],
                 na_values: NDArray[bool], state_names: NDArray[StateName]):
        super().__init__(values, feature_indices, names, na_values)
        self.state_names = state_names
        self.state_names_dict = {f: state_names[i] for i, f in enumerate(self.names)}


    @classmethod
    def from_dataframes(
        cls: Type[S],
        data: pd.DataFrame,
        feature_types: dict[str, dict],
    ) -> S:

        # Retrieve all categorical features
        categorical_columns = [k for k, v in feature_types.items() if v['type'] == "categorical"]
        categorical_data = data.loc[:, categorical_columns]
        feature_states = dict((c, feature_types[c]['states']) for c in categorical_columns)

        if categorical_data.empty:
            return None
        else:
            categorical_features_dict = encode_states(categorical_data, feature_states)
            # return Feature class consisting of all binarised categorical features
            return cls(**categorical_features_dict)

    @classmethod
    def create_partitions_by_nstates(
        cls: Type[S],
        data: pd.DataFrame,
        feature_types: dict[str,
        dict],
        na_string: str = ''
    ) -> list[S]:
        features_by_states = {}
        names = data.columns.to_numpy()
        data = data.fillna(na_string)  # TODO: check whether this makes sense. Stop parsing NAs on read instead?
        data_int = np.empty(data.shape, dtype=int)
        na_values = np.zeros(data.shape, dtype=bool)
        for i_f, f_name in enumerate(data.columns):
            ft = feature_types[f_name]
            if ft['type'] == "categorical":
                n_states = len(ft['states'])
                if n_states not in features_by_states:
                    features_by_states[n_states] = []
                features_by_states[n_states].append(i_f)

                # Define a mapping from state names to integer indices
                state_mapping = {state: i for i, state in enumerate(ft['states']) if state != na_string}
                state_mapping[na_string] = cls.NA

                # Apply the mapping to the data of this feature
                data_int[:, i_f] = list(map(state_mapping.get, data.iloc[:, i_f]))

                # Collect NA values
                na_values[:, i_f] = data_int[:, i_f] == cls.NA

        partitions = []
        for n_states, feature_indices in features_by_states.items():
            names_partition = names[feature_indices]
            state_names = np.array([feature_types[f]['states'] for f in names_partition])
            partition_features = data_int[:, feature_indices]

            partition = cls(
                values=partition_features,
                feature_indices=np.array(feature_indices),
                names=names_partition,
                na_values=na_values[:, feature_indices],
                state_names=state_names
            )
            partitions.append(partition)

        return partitions

    @property
    def n_states(self) -> int:
        return self.state_names.shape[1]

    @property
    def name(self):
        return f"Categorical[{self.n_states}]"

    def to_binary(self):
        """Convert to binary one-hot encoding."""
        return np.eye(self.n_states)[self.values]

class GaussianFeatures(GenericTypeFeatures):
    """Features that are continuous measurements following a Gaussian distribution."""

    @classmethod
    def from_dataframes(
        cls: Type[S],
        data: pd.DataFrame,
        feature_types: dict[str, dict],
    ) -> S:

        # Retrieve all gaussian features
        gaussian_indices = np.array([
            i for i, f in enumerate(data.columns)
            if feature_types[f]['type'] == "gaussian"
        ])

        if len(gaussian_indices) == 0:
            # No gaussian features found
            return None

        gaussian_data = data.iloc[:, gaussian_indices]
        gaussian_names = gaussian_data.columns.to_numpy()
        gaussian_features = gaussian_data.to_numpy(dtype=float, na_value=np.nan)

        # return Feature class consisting of all gaussian features
        return cls(
            values=gaussian_features,
            feature_indices=gaussian_indices,
            names=gaussian_names,
            na_values=np.isnan(gaussian_features),
        )

    @property
    def name(self):
        return "Gaussian"


class PoissonFeatures(GenericTypeFeatures):
    """Features that are count variables following a Poisson distribution."""

    @classmethod
    def from_dataframes(
            cls: Type[S],
            data: pd.DataFrame,
            feature_types: dict[str, dict],
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
    def name(self):
        return "Poisson"


class LogitNormalFeatures(GenericTypeFeatures):

    """Features that are percentages following a logit-normal distribution."""

    @classmethod
    def from_dataframes(
            cls: Type[S],
            data: pd.DataFrame,
            feature_types: dict[str, dict],
    ) -> S:
        # Retrieve all Poisson features
        logit_normal_columns = [k for k, v in feature_types.items() if v['type'] == "logit-normal"]
        logit_normal_data = data.loc[:, logit_normal_columns]
        if logit_normal_data.empty:
            return None
        else:
            values = logit_normal_data.to_numpy(dtype=float, na_value=np.nan)

            # Adding machine epsilon to 0 and subtract it from 1 avoids -inf/inf in logit transform
            values = np.where(values == 0.0, EPS, values)
            values = np.where(values == 1.0, values-EPS, values)

            logit_values = logit(values)

            logit_normal_features_dict = dict(values=logit_values,
                                              names=np.asarray(logit_normal_columns),
                                              na_number=logit_normal_data.isna().sum().sum())

            # return Feature class consisting of all logit-normal features
            return cls(**logit_normal_features_dict)

    @property
    def name(self):
        return "LogitNormal"

class Features:

    all_features: pd.DataFrame  # shape: (n_objects, n_features)
    partitions: list[GenericTypeFeatures]

    def __init__(self, all_features: pd.DataFrame, partitions: list[GenericTypeFeatures]):
        self.all_features = all_features
        self.partitions = partitions

        # Derive feature names attribute
        self.names = np.array(all_features.columns)

        # Derive missing value attributes
        self.missing = self.all_features.isna().to_numpy()
        self.na_number = np.sum(self.missing)

        # Keep number of objects and features as attributes
        self.n_objects, self.n_features = self.all_features.shape

        # Some consistency checks
        assert all(p.n_objects == self.n_objects for p in self.partitions)
        assert sum(p.n_features for p in self.partitions) == self.n_features

    def categorical_partitions(self):
        return [p for p in self.partitions if isinstance(p, CategoricalFeatures)]

    @classmethod
    def from_dataframes(
        cls: Type[S],
        data: pd.DataFrame,
        feature_types: dict[str, dict],
    ) -> S:
        # Features are sorted by their order in the `data` CSV file. Use feature_types to exclude metadata columns.
        feature_names = [s for s in data.columns if s in feature_types]

        # Create a dataframe that excludes metadata columns
        all_features = data.loc[:, feature_names]

        # Collect partitions containing
        partitions = []
        # Retrieve and one-hot encode all categorical features
        categorical_partitions = CategoricalFeatures.create_partitions_by_nstates(all_features, feature_types)
        partitions += categorical_partitions

        # Retrieve all Gaussian features
        gaussian_features = GaussianFeatures.from_dataframes(all_features, feature_types)
        if gaussian_features:
            partitions.append(gaussian_features)

        # Retrieve all Poisson features
        poisson_features = PoissonFeatures.from_dataframes(all_features, feature_types)
        if poisson_features:
            partitions.append(poisson_features)

        # Retrieve all logit-normal features
        logit_normal_features = LogitNormalFeatures.from_dataframes(all_features, feature_types)
        if logit_normal_features:
            partitions.append(logit_normal_features)

        # return Feature class consisting of all different types of features
        return cls(all_features=all_features, partitions=partitions)

@dataclass
class Confounder:

    name: str
    group_assignment: NDArray[bool]         # shape: (n_groups, n_objects)
    group_names: list[GroupName]            # shape: (n_groups,)

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
    ) -> S:
        n_objects = data.shape[0]

        if confounder_name not in data:
            # If there is no column specifying the group assignment for the confounder, it
            # is assumed to apply to all objects in the same way.
            group_assignment = np.ones((1, n_objects), dtype=bool)
            group_names = ["<ALL>"]
        else:
            group_names_by_obj = data[confounder_name]
            group_names = list(np.unique(group_names_by_obj.dropna()))
            group_assignment = np.zeros((len(group_names), n_objects), dtype=bool)
            for i_g, name_g in enumerate(group_names):
                group_assignment[i_g, np.where(group_names_by_obj == name_g)] = True

        return cls(
            name=confounder_name,
            group_assignment=group_assignment,
            group_names=group_names,
        )


class FeatureType(str, Enum):

    categorical = "categorical"
    gaussian = "gaussian"
    poisson = "poisson"
    logitnormal = "logitnormal"

    @classmethod
    def values(cls) -> Iterator[FeatureType | str]:
        return iter(cls)


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
            feature_types_path=config.data.feature_types,
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
    confounder_names: list[ConfounderName],
    feature_types_path: PathLike = None,
    feature_states_path: PathLike = None,
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
    if feature_types_path:
        with open(feature_types_path, "r") as f:
            yaml_loader = yaml.YAML(typ='safe')
            feature_types = yaml_loader.load(f)

    elif feature_states_path:
        feature_types = {}
        feature_states = read_data_csv(feature_states_path)
        for f_name, f_states in feature_states.items():
            feature_types[f_name] = {
                "type": "categorical",
                "states": f_states.dropna().tolist(),
            }

    else:
        raise ValueError("Either `feature_types_path` or `feature_states_path` must be provided.")

    features = Features.from_dataframes(data, feature_types)
    objects = Objects.from_dataframe(data)
    confounders = OrderedDict()
    for c in confounder_names:
        confounders[c] = Confounder.from_dataframe(data=data, confounder_name=c)

    # Check whether all columns in the data
    for c in data.columns:
        if c not in ["id", "name", "x", "y", *confounder_names, *feature_types]:
            raise ValueError(
                f"Unused column '{c}' in the data CSV. Columns should be either id, name, x, y, a confounder "
                f"(specified in config: model > confounders) or a feature (specified in feature_types CSV)."
            )

    if logger:
        logger.info(f"{objects.n_objects} objects with {features.n_features} features read from {data_path}.")
        for p in features.partitions:
            logger.info(f"{p.name}: {p.n_features} feature(s) with {p.na_number} NA value(s).")

    return objects, features, confounders
