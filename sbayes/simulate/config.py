from ruamel.yaml import YAML
import re

from pathlib import Path
from pydantic import model_validator, PositiveInt, Field
from typing import Self
from sbayes.config.config import (BaseConfig, RelativeFilePath, RelativePathType,
                                  RelativeDirectoryPath, SBayesConfig, PathLike)
from sbayes.load_data import FeatureType

CATEGORICAL_PATTERN = re.compile(r"^categorical_(\d+)$")
FIXED_FEATURE_TYPES = {str(t) for t in FeatureType if t is not FeatureType.categorical}

class OutputConfig(BaseConfig):
    """Where the simulated data and parameters are written."""

    path: RelativeDirectoryPath = Field(
        default_factory=lambda: RelativePathType.fix_path(".")
    )
    """Path to the output directory, relative to this config file."""


class SimulationConfig(BaseConfig):
    """Parameters of the simulated data sets."""

    objects: PositiveInt
    """Number of objects (e.g. languages) to simulate."""

    clusters: PositiveInt
    """Number of clusters in the simulated data. Overrides `clusters` in the sBayes
    config while simulating, so the ground truth may differ from the number of
    clusters used for inference."""

    datasets: PositiveInt
    """Number of data sets to draw from the prior."""

    seed: int = 0
    """Seed for the object locations and confounder groups."""

    x_min: float = 0.0
    """Lower bound of the simulated x coordinates."""

    x_max: float = 1.0
    """Upper bound of the simulated x coordinates."""

    y_min: float = 0.0
    """Lower bound of the simulated y coordinates."""

    y_max: float = 1.0
    """Upper bound of the simulated y coordinates."""

    features: dict[str, PositiveInt]
    """Number of features per feature type, e.g. {"categorical_2": 5, "gaussian": 2}.
    Categorical types are written as `categorical_<n_states>`."""

    confounders: dict[str, PositiveInt]
    """Number of groups per confounder, e.g. {"confounder_1": 4}. The names must match
    the confounders in the sBayes config used for inference."""


    output: OutputConfig = Field(default_factory=OutputConfig)
    """Where the simulated data and parameters are written."""

    @model_validator(mode="after")
    def validate_confounders(self) -> Self:
        """Check that every confounder has at least two groups."""
        for name, n_groups in self.confounders.items():
            if n_groups < 2:
                raise ValueError(
                    f"Confounder '{name}' has {n_groups} group(s); at least 2 are needed."
                )
        return self

    @model_validator(mode="after")
    def validate_feature_types(self) -> Self:
        """Check that every feature type is known and categorical types have 2+ states."""
        for key in self.features:
            if key in FIXED_FEATURE_TYPES:
                continue
            match = CATEGORICAL_PATTERN.fullmatch(key)
            if match is None:
                raise ValueError(
                    f"Unknown feature type '{key}'. Expected one of "
                    f"{sorted(FIXED_FEATURE_TYPES)} or `categorical_<n_states>`."
                )
            if int(match.group(1)) < 2:
                raise ValueError(f"'{key}': categorical features need at least 2 states.")
        return self


class SimulateConfigFile(BaseConfig):
    """The contents of a simulation config file.

    Holds the simulation parameters and the path to the sBayes config that defines the
    model to simulate from and to use for inference. Load it with `load_config`, which
    also reads that sBayes config into `inf_config`.
    """

    config_path: RelativeFilePath
    """Path to the sBayes config file, relative to this config file."""

    simulation: SimulationConfig
    """Parameters of the simulated data sets."""

    inf_config: SBayesConfig | None = None
    """The loaded sBayes config. Filled in by `load_config`; not set in the file."""

    @classmethod
    def from_config_file(cls, path: PathLike) -> Self:
        """Read a simulation config file, without loading the sBayes config.

        Paths in the file are resolved against the file's own directory. Use
        `load_config` to also load the sBayes config into `inf_config`.

        Args:
            path: the simulation config file (YAML)

        Returns:
            The parsed config.
        """
        RelativePathType.BASE_DIR = Path(path).absolute().parent
        with open(path, "r", encoding="utf-8") as f:
            config_dict = YAML(typ="safe").load(f)
        return cls(**config_dict)

    @model_validator(mode="after")
    def load_inference_config(self) -> Self:
        """Load the sBayes config that `config_path` points to."""
        if self.inf_config is None:
            self.inf_config = SBayesConfig.from_config_file(self.config_path)
        return self


def load_config(path: PathLike) -> SimulateConfigFile:
    """Load a simulation config together with the sBayes config it points to.

    Args:
        path: the simulation config file (YAML)

    Returns:
        The simulation config, with the sBayes config loaded into `inf_config`.
    """
    sim_config = SimulateConfigFile.from_config_file(path)
    sim_config.inf_config = SBayesConfig.from_config_file(sim_config.config_path)
    return sim_config


def feature_types_by_name(features: dict[str, PositiveInt]) -> dict[str, str]:
    """Assign a name to every simulated feature.

    Features are numbered consecutively across all types, in the order the types
    appear in the config. The names are the column headers of the simulated features
    CSV and the keys of the feature types YAML.

    Args:
        features: the number of features per type, e.g. {"categorical_2": 3, "gaussian": 2}

    Returns:
        The type per feature name, e.g. {"f_1": "categorical_2", "f_2": "categorical_2",
        "f_3": "categorical_2", "f_4": "gaussian", "f_5": "gaussian"}.
    """
    return {
        f"f_{i}": feature_type
        for i, feature_type in enumerate(
            (t for t, n in features.items() for _ in range(n)), start=1
        )
    }


def get_feature_types(features: dict[str, PositiveInt]) -> dict[str, dict]:
    """Build the feature types of the simulated features, in sBayes format.

    The result is written to `feature_types.yaml` and read back by the loader when
    the simulated data are used for inference.

    Args:
        features: the number of features per type, e.g. {"categorical_2": 3, "gaussian": 2}

    Returns:
        The type and states per feature name, e.g.
        {"f_1": {"type": "categorical", "states": ["s0", "s1"]}, "f_4": {"type": "gaussian"}}.

    Raises:
        ValueError: if a feature type is unknown. `SimulationConfig` validates the
            types, so this only triggers when the function is called directly.
    """
    feature_types = {}
    for name, feature_type in feature_types_by_name(features).items():
        match = CATEGORICAL_PATTERN.fullmatch(feature_type)
        if match is not None:
            feature_types[name] = {
                "type": FeatureType.categorical.value,
                "states": [f"s{s}" for s in range(int(match.group(1)))],
            }
        elif feature_type in FIXED_FEATURE_TYPES:
            feature_types[name] = {"type": feature_type}
        else:
            raise ValueError(f"Unknown feature type '{feature_type}'.")
    return feature_types
