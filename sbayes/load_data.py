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
from typing import ClassVar, Literal, Self

try:
    import ruamel.yaml as yaml
except ImportError:
    import ruamel_yaml as yaml

from sbayes.network import Network, parse_geo_cost_matrix
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


class FeatureType(str, Enum):
    """The feature types supported by sBayes.

    A string enum: each member equals its string value (e.g.
    ``FeatureType.gaussian == "gaussian"``), so members compare directly against
    the type strings read from a config file, while code referencing the members
    stays typo-proof.
    """

    categorical = "categorical"
    gaussian = "gaussian"
    poisson = "poisson"
    logitnormal = "logitnormal"

    def __str__(self) -> str:
        return self.value


class GenericTypeFeatures(ABC):

    """Super class for features of a specific type."""

    values: Array                                   # shape: (n_objects, n_features)
    feature_indices: Array                          # shape: (n_features,)
    names: NDArray                                  # shape: (n_features,)
    na_values: NDArray[np.bool_]                    # shape: (n_objects, n_features)

    FEATURE_TYPE: ClassVar[FeatureType]
    """The feature type this class handles. Must be set by every subclass."""

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

    FEATURE_TYPE = FeatureType.categorical
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
    ) -> list[Self]:
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
            if ft["type"] != cls.FEATURE_TYPE:
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
    FEATURE_TYPE = FeatureType.gaussian

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
        indices, names = select_columns_of_type(data, feature_types, cls.FEATURE_TYPE)

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
    FEATURE_TYPE = FeatureType.poisson

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
        indices, names = select_columns_of_type(data, feature_types, cls.FEATURE_TYPE)
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


class LogitNormalFeatures(GaussianFeatures):
    """Features that are proportions in (0, 1), modelled as Gaussian after a
    logit transform.

    The raw proportions are logit-transformed at load time, mapping (0, 1) to the
    real line, after which they are handled exactly like Gaussian features.
    Values of exactly 0 or 1 are nudged by machine epsilon to avoid infinities.
    """
    FEATURE_TYPE = FeatureType.logitnormal

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
        indices, names = select_columns_of_type(data, feature_types, cls.FEATURE_TYPE)
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
    """Container for all features of an analysis, grouped into type-specific partitions.

    Attributes:
        all_features: The full feature DataFrame, shape (n_objects, n_features).
        partitions: Type-specific feature partitions (categorical split by n_states).
        names: All feature names, shape (n_features,).
        missing: Boolean mask of missing values, shape (n_objects, n_features).
        na_number: Total count of missing values.
        n_objects: Number of objects.
        n_features: Total number of features across all partitions.
    """

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

        # Consistency checks
        if not all(p.n_objects == self.n_objects for p in self.partitions):
            raise ValueError("Not all partitions have the same number of objects.")

        n_partition_features = sum(p.n_features for p in self.partitions)
        if n_partition_features != self.n_features:
            raise ValueError(
                f"Partitions cover {n_partition_features} features but the data "
                f"has {self.n_features}."
            )

    def categorical_partitions(self):
        """Categorical partitions across all partitions."""
        return [p for p in self.partitions if isinstance(p, CategoricalFeatures)]

    @classmethod
    def from_dataframes(cls, data, feature_types) -> Self:
        """Build a Features container from a data DataFrame and feature-type spec.

        Metadata columns (those not in feature_types) are excluded. Categorical
        features are split into partitions by number of states; each continuous
        feature type forms at most one partition.

        Args:
            data: The full data DataFrame (features + metadata columns).
            feature_types: Mapping from feature name to its type and states.

        Returns:
            A Features container holding all typed partitions.
        """
        # Keep only feature columns, in their CSV order
        feature_names = [c for c in data.columns if c in feature_types]
        all_features = data.loc[:, feature_names]

        # Categorical features partition by number of states
        partitions: list[GenericTypeFeatures] = list(
            CategoricalFeatures.create_partitions_by_nstates(all_features, feature_types)
        )

        # Each continuous type forms at most one partition
        for feature_cls in (GaussianFeatures, PoissonFeatures, LogitNormalFeatures):
            features = feature_cls.from_dataframes(all_features, feature_types)
            if features is not None:
                partitions.append(features)

        return cls(all_features=all_features, partitions=partitions)


@dataclass
class Confounder:
    """A confounder assigning objects to groups (e.g. language families).

    Attributes:
        name: The confounder's name.
        group_assignment: Boolean membership matrix, shape (n_groups, n_objects).
        group_names: Names of the groups, shape (n_groups,).
    """

    name: str
    group_assignment: NDArray[np.bool_]     # shape: (n_groups, n_objects)
    group_names: list[str]                  # shape: (n_groups,)

    def any_group(self) -> NDArray[np.bool_]:
        """For each object, whether it belongs to any group of this confounder.

        Objects with a missing confounder value belong to no group (all False).

        Returns:
            Boolean array of shape (n_objects,).
        """
        return np.any(self.group_assignment, axis=0)

    @property
    def n_groups(self) -> int:
        """Number of groups in this confounder."""
        return len(self.group_names)

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, confounder_name: str) -> Self:
        """Build a Confounder from a data DataFrame.

        If the data has no column for this confounder, it is assumed to apply
        uniformly to all objects (a single group "<ALL>"). Objects with a
        missing value for the confounder are assigned to no group.

        Args:
            data: The data DataFrame.
            confounder_name: Name of the confounder (and of its column, if present).

        Returns:
            The parsed Confounder.
        """
        n_objects = data.shape[0]

        if confounder_name not in data.columns:
            group_assignment = np.ones((1, n_objects), dtype=bool)
            group_names = ["<ALL>"]
        else:
            group_by_object = data[confounder_name]
            group_names = list(np.unique(group_by_object.dropna()))
            group_assignment = np.zeros((len(group_names), n_objects), dtype=bool)
            for i_g, name_g in enumerate(group_names):
                group_assignment[i_g] = (group_by_object == name_g).to_numpy()

        return cls(
            name=confounder_name,
            group_assignment=group_assignment,
            group_names=group_names,
        )


class Data:
    """Container and loading logic for the data of an sBayes analysis.

    Attributes:
        objects: The objects (locations, ids, names).
        features: The features, grouped into type-specific partitions.
        confounders: Named confounders, each assigning objects to groups.
        crs: The coordinate reference system for object locations.
        geo_cost_matrix: Pairwise geographic costs between objects.
        network: The spatial network built from object locations.
        logger: Logger used during loading (likely cleared afterwards).
    """

    objects: Objects
    features: Features
    confounders: OrderedDict[str, Confounder]
    crs: pyproj.CRS | None
    geo_cost_matrix: NDArray[np.float64] | None
    network: Network
    logger: Logger | None

    def __init__(
        self,
        objects: Objects,
        features: Features,
        confounders: OrderedDict[str, Confounder],
        projection: str | None = "epsg:4326",
        geo_costs: Literal["from_data"] | PathLike = "from_data",
        logger: Logger | None = None,
    ):
        self.objects = objects
        self.features = features
        self.confounders = confounders
        self.validate_confounders()
        self.logger = logger

        # NOTE: location-dependent — for optional-locations feature, guard this block
        self.crs = pyproj.CRS(projection)
        self.network = Network.from_objects(self.objects, crs=self.crs)
        if geo_costs == "from_data":
            self.geo_cost_matrix = self.network.dist_mat
        else:
            self.geo_cost_matrix = parse_geo_cost_matrix(
                object_names=self.objects.id,
                file=geo_costs,
                logger=self.logger
            )

    @classmethod
    def from_config(cls, config: SBayesConfig, logger: Logger | None = None) -> Self:
        """Load Data from the files referenced in a config."""
        if logger:
            cls.log_loading(logger)
        objects, features, confounders = read_features_from_csv(
            data_path=config.data.features,
            feature_types_path=config.data.feature_types,
            feature_states_path=config.data.feature_states,
            confounder_names=config.model.confounders,
            logger=logger,
        )
        return cls(
            objects=objects,
            features=features,
            confounders=confounders,
            projection=config.data.projection,
            geo_costs=config.model.prior.geo.costs,
            logger=logger,
        )

    @classmethod
    def from_experiment(cls, experiment: Experiment) -> Self:
        """Load Data from an experiment's config and logger."""
        return cls.from_config(experiment.config, logger=experiment.logger)

    @classmethod
    def from_simulation(
        cls,
        features_csv: pd.DataFrame,
        feature_types: dict,
        config: SBayesConfig,
        logger: Logger | None = None,
    ) -> Self:
        """Create Data from in-memory structures, without file I/O."""
        objects, features, confounders = parse_features(
            data=features_csv,
            feature_types=feature_types,
            confounder_names=config.model.confounders,
        )
        return cls(
            objects=objects,
            features=features,
            confounders=confounders,
            projection=config.data.projection,   # respect config, not silent default
            geo_costs=config.model.prior.geo.costs,
            logger=logger,
        )

    @staticmethod
    def log_loading(logger: Logger) -> None:
        """Write the data-import header to the log."""
        logger.info("\n")
        logger.info("DATA IMPORT")
        logger.info("#" * 42)

    def validate_confounders(self) -> None:
        """Check that the confounders group the objects as required by the model.

        Every object must belong to a group of at least one confounder, since an object
        outside every group has no mixture component to draw its features from.
        """
        for name, conf in self.confounders.items():
            if not conf.any_group().any():
                raise ValueError(
                    f"No object is assigned to a group of confounder '{name}'. Remove "
                    f"the confounder, or omit the column entirely to define a single "
                    f"universal group."
                )

        in_any_group = np.any(
            [conf.any_group() for conf in self.confounders.values()], axis=0
        )
        if not in_any_group.all():
            missing = [self.objects.id[i] for i in np.flatnonzero(~in_any_group)]
            raise ValueError(
                f"Objects {missing} belong to no group of any confounder. Every object "
                f"must be assigned to a group of at least one confounder."
            )

def select_columns_of_type(
    data: pd.DataFrame,
    feature_types: dict[str, dict],
    feature_type: FeatureType,
) -> tuple[NDArray[np.int_], NDArray]:
    """Find the columns of a given feature type in the data.

    Args:
        data: DataFrame of feature columns.
        feature_types: Mapping from feature name to its type and states.
        feature_type: The feature type to select.

    Returns:
        A tuple of (indices, names): the integer positions of the matching
        columns in `data`, and their names. Both are empty if no column matches.
    """
    indices = np.array([
        i for i, f in enumerate(data.columns)
        if feature_types[f]["type"] == feature_type
    ], dtype=int)
    names = data.columns.to_numpy()[indices]
    return indices, names


def parse_features(
    data: pd.DataFrame,
    feature_types: dict,
    confounder_names: list[str],
    logger: Logger | None = None,
) -> tuple[Objects, Features, OrderedDict[str, Confounder]]:
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
    confounder_names: list[str],
    feature_types_path: PathLike | None = None,
    feature_states_path: PathLike | None = None,
    logger: Logger | None = None,
) -> tuple[Objects, Features, OrderedDict[str, Confounder]]:
    """Import data (objects, features, confounders) from a CSV file.

    Exactly one of `feature_types_path` or `feature_states_path` must be given.
    `feature_types_path` (YAML) supports all feature types; `feature_states_path`
    (CSV) is a categorical-only legacy format.

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

