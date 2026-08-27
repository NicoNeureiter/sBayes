""" Imports the real world data """
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyproj

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from jax import Array
from logging import Logger
from numpy.typing import NDArray
from scipy.special import logit
from typing import Literal, Optional, Type, Iterator, Self

try:
    import ruamel.yaml as yaml
except ImportError:
    import ruamel_yaml as yaml

from sbayes.preprocessing import ComputeNetwork, read_geo_cost_matrix
from sbayes.util import PathLike, read_data_csv, EPS
from sbayes.config.config import SBayesConfig
from sbayes.experiment_setup import Experiment

# Type variables for better readability
ObjectName = str
ObjectID = str
FeatureName = str
StateName = str
ConfounderName = str
GroupName = str


@dataclass
class Objects:
    """A set of objects, each describing one sample (a language, person, state, ...)
    with an ID, name and location.

    Attributes:
        id: Object IDs, one per object.
        locations: Object coordinates, shape (n_objects, 2).
        names: Object names, one per object.
        indices: Integer index of each object (0, ..., n_objects-1), shape (n_objects,).
    """

    id: list[ObjectID]

    # todo: locations assumed here
    locations: NDArray[np.float64]           # shape: (n_objects, 2)
    names: list[ObjectName]
    indices: NDArray[np.int_] = field(init=False)  # shape: (n_objects,)

    def __post_init__(self) -> None:
        self.indices = np.arange(self.n_objects)

    @property
    def n_objects(self) -> int:
        return len(self.id)

    def __len__(self) -> int:
        return len(self.id)

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> Self:
        """Build an Objects instance from a data CSV DataFrame.

        Args:
            data: DataFrame with required columns `id`, `x`, `y`, and optional `name`.

        Returns:
            The parsed Objects.

        Raises:
            KeyError: If any of the required columns `id`, `x`, `y` is missing.
        """
        try:
            ids = data["id"].tolist()
            # todo: locations assumed here
            locations = data[["x", "y"]].to_numpy(dtype=float)
        except KeyError:
            raise KeyError("The data CSV must contain columns `id`, `x` and `y`.")

        names = list(data.get("name", ids))
        return cls(id=ids, locations=locations, names=names) # type: ignore[assignment]


class GenericTypeFeatures(ABC):

    """Super class for features of a specific type."""

    values: Array                                   # shape: (n_objects, n_features)
    feature_indices: Array                          # shape: (n_features,)
    names: NDArray                     # shape: (n_features,)
    na_values: NDArray[np.bool_]                        # shape: (n_objects, n_features)

    def __init__(self, values: NDArray,
                 feature_indices: NDArray[np.int_],
                 names: NDArray,
                 na_values: NDArray[np.bool_]):
        self.values = jnp.array(values)
        self.feature_indices = jnp.array(feature_indices)
        self.names = names
        self.na_values = na_values

    @property
    def n_objects(self) -> int:
        """The number of objects"""
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        """The number of features"""
        return self.values.shape[1]

    @property
    def na_number(self) -> np.int64:
        """The number of NA values"""
        return np.sum(self.na_values)

    @property
    @abstractmethod
    def name(self) -> str:
        """A short label identifying this feature type (e.g. 'Gaussian')."""
        ...


class CategoricalFeatures(GenericTypeFeatures):
    """Integer representation of categorical features.

    Each feature value is an integer index into that feature's list of states.
    Missing values are stored as the NA sentinel (`NA = -1`).
    """

    state_names: NDArray            # shape: (n_features, n_states)
    state_names_dict: dict[str, NDArray]
    NA: int = -1

    def __init__(
        self,
        values: NDArray[np.int_],
        feature_indices: NDArray[np.int_],
        names: NDArray,
        na_values: NDArray[np.bool_],
        state_names: NDArray,
    ):
        super().__init__(values, feature_indices, names, na_values)
        self.state_names = state_names
        self.state_names_dict = {f: state_names[i] for i, f in enumerate(self.names)}
        self._binarized: NDArray[np.bool_] | None = None


    @classmethod
    def create_partitions_by_nstates(
        cls,
        data: pd.DataFrame,
        feature_types: dict[str, dict],
        na_string: str = "",
    ) -> list[Self] | None:
        """Split categorical features into partitions grouped by number of states.

        Categorical features with the same number of states are collected into
        one partition (a single CategoricalFeatures instance), because features
        with different numbers of states cannot share a value array. State names
        are mapped to integer indices; missing values are mapped to the NA
        sentinel (`cls.NA`).

        Args:
            data: DataFrame of feature columns (metadata columns already excluded).
            feature_types: Mapping from feature name to its type and states.
            na_string: Placeholder that missing values are filled with before
                mapping. Must not collide with a real state name.

        Returns:
            One CategoricalFeatures partition per distinct number of states.
            Empty if there are no categorical features.
        """
        names = data.columns.to_numpy()
        data = data.fillna(na_string)  # TODO: revisit NA handling — stop parsing NAs on read instead?
        data_int = np.empty(data.shape, dtype=int)
        na_values = np.zeros(data.shape, dtype=bool)

        features_by_states: dict[int, list[int]] = {}
        for i_f, f_name in enumerate(data.columns):
            ft = feature_types[f_name]
            if ft["type"] != "categorical":
                continue

            states = ft["states"]
            n_states = len(states)
            features_by_states.setdefault(n_states, []).append(i_f)

            # Map state names to integer indices; the NA placeholder maps to cls.NA
            state_mapping = {state: i for i, state in enumerate(states) if state != na_string}
            state_mapping[na_string] = cls.NA

            column = data.iloc[:, i_f]
            unknown = set(column) - set(state_mapping)
            if unknown:
                raise ValueError(
                    f"Feature '{f_name}' contains values not declared in its states "
                    f"{states}: {sorted(unknown)}."
                )
            data_int[:, i_f] = column.map(state_mapping).to_numpy()
            na_values[:, i_f] = data_int[:, i_f] == cls.NA

        partitions = []
        for n_states, feature_indices in features_by_states.items():
            partitions.append(cls(
                values=data_int[:, feature_indices],
                feature_indices=np.array(feature_indices),
                names=names[feature_indices],
                na_values=na_values[:, feature_indices],
                state_names=np.array([feature_types[f]["states"] for f in names[feature_indices]]),
            ))

        return partitions

    @property
    def n_states(self) -> int:
        """The number of states shared by all features in this partition."""
        return self.state_names.shape[1]

    @property
    def name(self) -> str:
        """The name of the partition."""
        return f"Categorical[{self.n_states}]"

    def to_binary(self) -> NDArray[np.bool_]:
        """Return a one-hot (binary) encoding of the feature values.

        The result has shape (n_objects, n_features, n_states); NA positions are
        all-False across the state axis. The encoding is computed once and cached.

        Returns:
            Boolean one-hot array; do not mutate (it is cached and returned by reference).
        """
        binarized = self._binarized
        if binarized is None:
            binarized = np.eye(self.n_states, dtype=bool)[self.values]
            binarized[self.na_values, :] = False
            self._binarized = binarized
        return binarized


class GaussianFeatures(GenericTypeFeatures):
    """Features that are continuous measurements following a Gaussian distribution."""

    @classmethod
    def from_dataframes(
        cls,
        data: pd.DataFrame,
        feature_types: dict[str, dict],
    ) -> Self | None:
        """Build Gaussian features from the columns typed 'gaussian'.

        Args:
            data: DataFrame of feature columns (metadata columns already excluded).
            feature_types: Mapping from feature name to its type and states.

        Returns:
            A GaussianFeatures instance, or None if there are no Gaussian features.
        """
        indices, names = select_columns_of_type(data, feature_types, "gaussian")

        if len(indices) == 0:
            return None

        gaussian_data = data.iloc[:, indices]
        values = gaussian_data.to_numpy(dtype=float, na_value=np.nan)

        return cls(
            values=values,
            feature_indices=indices,
            names=gaussian_data.columns.to_numpy(),
            na_values=np.isnan(values),
        )

    @property
    def name(self) -> str:
        """The name of the feature type"""
        return "Gaussian"


class PoissonFeatures(GenericTypeFeatures):
    """Features that are count variables following a Poisson distribution."""

    @classmethod
    def from_dataframes(
        cls,
        data: pd.DataFrame,
        feature_types: dict[str, dict],
    ) -> Self | None:
        """Build Poisson features from the columns typed 'poisson'.

        Args:
            data: DataFrame of feature columns (metadata columns already excluded).
            feature_types: Mapping from feature name to its type and states.

        Returns:
            A PoissonFeatures instance, or None if there are no Poisson features.
        """
        indices, names = select_columns_of_type(data, feature_types, "poisson")
        if len(indices) == 0:
            return None

        values = data.iloc[:, indices].to_numpy(dtype=float)
        return cls(
            values=values,
            feature_indices=indices,
            names=names,
            na_values=np.isnan(values),
        )

    @property
    def name(self) -> str:
        """The name of the feature type"""
        return "Poisson"


# todo: logit normal was silently dead.
class LogitNormalFeatures(GaussianFeatures):
    """Features that are proportions in (0, 1), modelled as Gaussian after a
    logit transform.

    The raw proportions are logit-transformed at load time, mapping (0, 1) to the
    real line, after which they are handled exactly like Gaussian features.
    Values of exactly 0 or 1 are nudged by machine epsilon to avoid infinities.
    """

    @classmethod
    def from_dataframes(
        cls,
        data: pd.DataFrame,
        feature_types: dict[str, dict],
    ) -> Self | None:
        """Build logit-normal features from the columns typed 'logitnormal'.

        Args:
            data: DataFrame of feature columns (metadata columns already excluded).
            feature_types: Mapping from feature name to its type and states.

        Returns:
            A LogitNormalFeatures instance, or None if there are no logit-normal
            features.
        """
        indices, names = select_columns_of_type(data, feature_types, "logitnormal")
        if len(indices) == 0:
            return None
        values = data.iloc[:, indices].to_numpy(dtype=float, na_value=np.nan)
        values = np.where(values == 0.0, EPS, values)
        values = np.where(values == 1.0, values - EPS, values)
        return cls(
            values=logit(values),
            feature_indices=indices,
            names=names,
            na_values=np.isnan(values),   # NA from pre-transform values
        )

    @property
    def name(self) -> str:
        """The name of the feature type"""
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

    @classmethod
    def from_simulation(cls,
                        features_csv: pd.DataFrame,
                        feature_types: dict,
                        config: SBayesConfig,
                        logger=None) -> S:

        """Create Data directly from in-memory structures, without file I/O."""
        objects, features, confounders = parse_features(
            data=features_csv,
            feature_types=feature_types,
            confounder_names=config.model.confounders,
        )
        return cls(
            objects=objects,
            features=features,
            confounders=confounders,
            logger=logger
        )

    @staticmethod
    def log_loading(logger):
        logger.info("\n")
        logger.info("DATA IMPORT")
        logger.info("##########################################")


def select_columns_of_type(
    data: pd.DataFrame,
    feature_types: dict[str, dict],
    type_name: str,
) -> tuple[NDArray[np.int_], NDArray]:
    """Find the columns of a given feature type in the data.

    Args:
        data: DataFrame of feature columns.
        feature_types: Mapping from feature name to its type and states.
        type_name: The feature type to select (e.g. "gaussian", "poisson").

    Returns:
        A tuple of (indices, names): the integer positions of the matching
        columns in `data`, and their names. Both are empty if no column matches.
    """
    indices = np.array([
        i for i, f in enumerate(data.columns)
        if feature_types[f]["type"] == type_name
    ], dtype=int)
    names = data.columns.to_numpy()[indices]
    return indices, names


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

def parse_features(
    data: pd.DataFrame,
    feature_types: dict,
    confounder_names: list[ConfounderName],
    logger: Optional[Logger] = None,
) -> (Objects, Features, dict[ConfounderName, Confounder]):
    """Parse features, objects and confounders from in-memory structures.

    Core parsing logic shared between file-based loading and simulation.

    Args:
        data: DataFrame containing objects, features and confounders.
        feature_types: Dict mapping feature name to type and states.
        confounder_names: List of confounder names.
        logger: A Logger instance for writing log messages.

    Returns:
        The parsed data objects (objects, features and confounders).
    """
    features = Features.from_dataframes(data, feature_types)
    objects = Objects.from_dataframe(data)

    confounders = OrderedDict()
    for c in confounder_names:
        confounders[c] = Confounder.from_dataframe(data=data, confounder_name=c)

    for c in data.columns:
        if c not in ["id", "name", "x", "y", *confounder_names, *feature_types]:
            raise ValueError(
                f"Unused column '{c}' in the data CSV. Columns should be either id, name, x, y, a confounder "
                f"(specified in config: model > confounders) or a feature (specified in feature_types CSV)."
            )

    if logger:
        logger.info(f"{objects.n_objects} objects with {features.n_features} features.")
        for p in features.partitions:
            logger.info(f"{p.name}: {p.n_features} feature(s) with {p.na_number} NA value(s).")

    return objects, features, confounders


def read_features_from_csv(
    data_path: PathLike,
    confounder_names: list[ConfounderName],
    feature_types_path: PathLike = None,
    feature_states_path: PathLike = None,
    logger: Optional[Logger] = None,
) -> (Objects, Features, dict[ConfounderName, Confounder]):
    """Import data (objects, features, confounders) from a CSV file.

    Args:
        data_path: Path to the data CSV file.
        feature_types_path: Path to the feature types YAML file.
        feature_states_path: Path to the feature states CSV file.
        confounder_names: List of confounder names.
        logger: A Logger instance for writing log messages.

    Returns:
        The parsed data objects (objects, features and confounders).
    """
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

    return parse_features(data, feature_types, confounder_names, logger)
