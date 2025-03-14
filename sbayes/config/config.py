from __future__ import annotations

import os
from pathlib import Path
from enum import Enum
import warnings
import json
from typing import Union, List, Dict, Optional

try:
    from typing import Annotated, Literal
except ImportError:  # For python <= 3.8
    from typing_extensions import Annotated, Literal

try:
    import ruamel.yaml as yaml
except ImportError:
    import ruamel_yaml as yaml

from pydantic import model_validator, BaseModel, Field
from pydantic import ValidationError
from pydantic import DirectoryPath
from pydantic import PositiveInt, PositiveFloat, NonNegativeFloat
from pydantic.types import PathType
from pydantic_core import core_schema, PydanticCustomError

from sbayes.util import fix_relative_path, decompose_config_path, PathLike
from sbayes.util import update_recursive


class RelativePathType(PathType):

    BASE_DIR: DirectoryPath = "."

    @classmethod
    def fix_path(cls, value: PathLike) -> Path:
        return fix_relative_path(value, cls.BASE_DIR)

    @staticmethod
    def validate_file(path: Path, _: core_schema.ValidationInfo) -> Path:
        path = RelativePathType.fix_path(path)
        if path.is_file():
            return path
        else:
            raise PydanticCustomError('path_not_file', 'Path does not point to a file')

    @staticmethod
    def validate_directory(path: Path, _: core_schema.ValidationInfo) -> Path:
        path = RelativePathType.fix_path(path)
        os.makedirs(path, exist_ok=True)
        if path.is_dir():
            return path
        else:
            raise PydanticCustomError('path_not_directory', 'Path does not point to a directory')


RelativeFilePath = Annotated[Path, RelativePathType('file')]
"""A relative path that must point to a file."""

RelativeDirectoryPath = Annotated[Path, RelativePathType('dir')]
"""A relative path that must point to a directory."""


class BaseConfig(BaseModel, extra='forbid'):

    """The base class for all config classes. This inherits from pydantic.BaseModel and
    configures settings that should be shared across all setting classes."""

    def __getitem__(self, key):
        return self.__getattribute__(key)

    @classmethod
    def get_attr_doc(cls, attr: str) -> str:
        return cls.__attrdocs__.get(attr)

    @classmethod
    def annotations(cls, key: str) -> Union[str, None]:
        if key in cls.__annotations__:
            return cls.__annotations__[key]
        for base_cls in cls.__bases__:
            if issubclass(base_cls, BaseConfig):
                s = base_cls.annotations(key)
                if s is not None:
                    return s
        return None

    @classmethod
    def deprecated_attributes(cls) -> list:
        return []

    @model_validator(mode="before")
    def warn_about_deprecated_attributes(cls, values: dict):
        for key in cls.deprecated_attributes():
            if key in values:
                warnings.warn(f"The {key} key in {cls.__name__} is deprecated "
                              f"and will be removed in future versions of sBayes.")
                values.pop(key)
        return values


""" ===== PRIOR CONFIGS ===== """


class GaussianMeanPriorConfig(BaseConfig):
    """Configuration of the prior on the mean of a normal distribution"""

    class Types(str, Enum):
        IMPROPER_UNIFORM = "improper_uniform"
        GAUSSIAN = "gaussian"

    type: Types = Types.IMPROPER_UNIFORM
    """Type of prior distribution (`improper_uniform` or `gaussian`)."""

    file: Optional[RelativeFilePath] = None
    """Path to the parameters of the Gaussian distribution."""

    parameters: Optional[Dict[str, float]] = None
    """Parameters of the Gaussian distribution."""

    @model_validator(mode="before")
    def warn_when_using_default_type(cls, values):
        if "type" not in values:
            warnings.warn(
                f"No `type` defined for `{cls.__name__}`. Using `improper_uniform` as a default."
            )
        return values

    @model_validator(mode="before")
    def validate_gamma_parameters(cls, values):
        prior_type = values.get("type")

        if prior_type == "gaussian":
            if (values.get("file") is None) and (values.get("parameters") is None):
                raise ValidationError(
                    f"Provide `file` or `parameters` for `{cls.__name__}` of type `gaussian`."
                )
        return values

    def dict(self, *args, **kwargs):
        """A custom dict method to hide non-applicable attributes depending on prior type."""
        self_dict = super().dict(*args, **kwargs)
        if self.type is self.Types.IMPROPER_UNIFORM:
            self_dict.pop("file")
            self_dict.pop("parameters")
        else:
            if self.file is not None:
                self_dict.pop("parameters")
            elif self.parameters is not None:
                self_dict.pop("file")

        return self_dict

    @classmethod
    def get_attr_doc(cls, attr):
        doc = super().get_attr_doc(attr)
        if not doc:
            return GaussianMeanPriorConfig.__attrdocs__.get(attr)


class GaussianVariancePriorConfig(BaseConfig):
    """Configuration of the prior on the variance of a normal distribution"""

    class Types(str, Enum):
        JEFFREYS = "jeffreys"
        INV_GAMMA = "inv-gamma"
        FIXED = "fixed"
        EXPONENTIAL = "exponential"

    type: Types
    """Type of prior distribution (`improper_uniform` or `gaussian`)."""

    file: Optional[RelativeFilePath] = None
    """Path to the parameters of the Gaussian distribution."""

    parameters: Optional[Dict[str, float]] = None
    """Parameters of the Gaussian distribution."""

    @model_validator(mode="before")
    def validate_gamma_parameters(cls, values):
        prior_type = values.get("type")

        if prior_type == "inv-gamma":
            if (values.get("file") is None) and (values.get("parameters") is None):
                raise ValidationError(
                    f"Provide `file` or `parameters` for `{cls.__name__}` of type `inv-gamma`."
                )
        return values

    def dict(self, *args, **kwargs):
        """A custom dict method to hide non-applicable attributes depending on prior type."""
        self_dict = super().dict(*args, **kwargs)
        if self.type is self.Types.JEFFREYS:
            self_dict.pop("file")
            self_dict.pop("parameters")
        else:
            if self.file is not None:
                self_dict.pop("parameters")
            elif self.parameters is not None:
                self_dict.pop("file")

        return self_dict

    @classmethod
    def get_attr_doc(cls, attr):
        doc = super().get_attr_doc(attr)
        if not doc:
            return GaussianVariancePriorConfig.__attrdocs__.get(attr)


class GaussianPriorConfig(BaseConfig):
    """Configuration of the prior on the mean and variance of a normal distribution"""

    mean: GaussianMeanPriorConfig
    variance: GaussianVariancePriorConfig


class PoissonPriorConfig(BaseConfig):
    """Configuration of the prior on the rate parameter of a Poisson distribution"""

    class Types(str, Enum):
        JEFFREYS = "jeffreys"
        GAMMA = "gamma"

    type: Types = Types.JEFFREYS
    """Type of prior distribution (`jeffreys` or `gamma`)."""

    file: Optional[RelativeFilePath] = None
    """Path to the parameters of the Gamma distribution."""

    parameters: Optional[Dict[str, float]] = None
    """Parameters of the Gamma distribution."""

    @model_validator(mode="before")
    def warn_when_using_default_type(cls, values):
        if "type" not in values:
            warnings.warn(
                f"No `type` defined for `{cls.__name__}`. Using `jeffreys` as a default."
            )
        return values

    @model_validator(mode="before")
    def validate_gamma_parameters(cls, values):
        prior_type = values.get("type")

        if prior_type == "gamma":
            if (values.get("file") is None) and (values.get("parameters") is None):
                raise ValidationError(
                    f"Provide `file` or `parameters` for `{cls.__name__}` of type `gamma`."
                )
        return values

    def dict(self, *args, **kwargs):
        """A custom dict method to hide non-applicable attributes depending on prior type."""
        self_dict = super().dict(*args, **kwargs)
        if self.type is self.Types.JEFFREYS:
            self_dict.pop("file")
            self_dict.pop("parameters")
        else:
            if self.file is not None:
                self_dict.pop("parameters")
            elif self.parameters is not None:
                self_dict.pop("file")

        return self_dict

    @classmethod
    def get_attr_doc(cls, attr):
        doc = super().get_attr_doc(attr)
        if not doc:
            return PoissonPriorConfig.__attrdocs__.get(attr)


class DirichletPriorConfig(BaseConfig):

    class Types(str, Enum):
        UNIFORM = "uniform"
        DIRICHLET = "dirichlet"
        SYMMETRIC_DIRICHLET = "symmetric_dirichlet"

    type: Types = Types.UNIFORM
    """Type of prior distribution. Choose from: [uniform, dirichlet, symmetric_dirichlet]"""

    file: Optional[RelativeFilePath] = None
    """Path to parameters of the Dirichlet distribution (YAML or JSON format).
    This or `parameters` is required if type=dirichlet."""

    parameters: Optional[dict] = None
    """Parameters of the Dirichlet distribution. This or `file` is required if type=dirichlet."""

    prior_concentration: Optional[float] = None
    """The concentration of the prior distribution. Required if type=symmetric_dirichlet."""

    @model_validator(mode="before")
    @classmethod
    def warn_when_using_default_type(cls, values):
        if "type" not in values:
            warnings.warn(
                f"No `type` defined for `{cls.__name__}`. Using `uniform` as a default."
            )
        return values

    @model_validator(mode="after")
    def validate_dirichlet_parameters(self):
        cls_name = type(self).__name__
        if self.type == self.Types.DIRICHLET:
            if (self.file is None) and (self.parameters is None):
                raise ValidationError(
                    f"Provide `file` or `parameters` for `{cls_name}` of type `dirichlet`."
                )

        elif self.type == self.Types.SYMMETRIC_DIRICHLET:
            if self.prior_concentration is None:
                raise ValidationError(f"Provide `prior_concentration` for `{cls_name}` of type `{self.type}`.")

        return self

    def dict(self, *args, **kwargs):
        """A custom dict method to hide non-applicable attributes depending on prior type."""
        self_dict = super().dict(*args, **kwargs)
        if self.type is self.Types.UNIFORM:
            self_dict.pop("file")
            self_dict.pop("parameters")
        else:
            if self.file is not None:
                self_dict.pop("parameters")
            elif self.parameters is not None:
                self_dict.pop("file")

        return self_dict

    @classmethod
    def get_attr_doc(cls, attr):
        doc = super().get_attr_doc(attr)
        if not doc:
            return DirichletPriorConfig.__attrdocs__.get(attr)


class LogisticNormalPriorConfig(BaseConfig):
    loc: float = 0.0
    """The mean of the logistic normal prior."""

    scale: float = 1.0
    """The scale of the logistic normal prior."""


class ClusterPriorConfig(BaseConfig):
    """Configuration of the cluster assignment prior."""

    class Types(str, Enum):
        CATEGORICAL = "categorical"
        DIRICHLET = "dirichlet"
        LOGISTIC_NORMAL = "logistic_normal"

    type: Types
    """Type of prior distribution. Choose from: [categorical, dirichlet or logit_normal]."""

    dirichlet_config: Optional[DirichletPriorConfig] = None
    """Configuration of the Dirichlet prior for the cluster assignment."""

    logistic_normal_config: Optional[LogisticNormalPriorConfig] = None
    """Configuration of the Logistic Normal prior for the cluster assignment."""

    min: PositiveInt = 2
    """Minimum cluster size."""

    max: PositiveInt = 10000
    """Maximum cluster size."""

    # NN: min and max bounds are tricky to enforce with continuous assignments.
    # TODO: Discuss if there is demand and how it could be implemented.


class GeoPriorConfig(BaseConfig):

    """Configuration of the geo-prior."""

    class Types(str, Enum):
        UNIFORM = "uniform"
        COST_BASED = "cost_based"

    class AggregationStrategies(str, Enum):
        MEAN = "mean"
        SUM = "sum"
        MAX = "max"

    class ProbabilityFunction(str, Enum):
        EXPONENTIAL = "exponential"
        SIGMOID = "sigmoid"

    class Skeleton(str, Enum):
        MST = "mst"
        DELAUNAY = "delaunay"
        DIAMETER = "diameter"  # i.e. the longest shortest path between two nodes
        COMPLETE = "complete_graph"
        SPECTRAL = "spectral"

    type: Types = Types.UNIFORM
    """Type of prior distribution. Choose from: [uniform, cost_based, simulated]."""

    costs: Union[RelativeFilePath, Literal["from_data"]] = "from_data"
    # costs: FilePath = "from_data"
    """Source of the geographic costs used for cost_based geo-prior. Either `from_data`
    (derive geodesic distances from locations) or path to a CSV file."""

    aggregation: AggregationStrategies = AggregationStrategies.MEAN
    """Policy defining how costs of single edges are aggregated. Choose from: [mean, sum or max]."""

    probability_function: ProbabilityFunction = ProbabilityFunction.EXPONENTIAL
    """Monotonic function that defines how aggregated costs are mapped to prior probabilities."""

    rate: Optional[PositiveFloat] = None
    """Rate at which the prior probability decreases for a cost_based geo-prior. Required if type=cost_based."""

    inflection_point: Optional[float] = None
    """Value where the sigmoid probability function reaches 0.5. Required if type=cost_based
    and probability_function=sigmoid."""

    skeleton: Skeleton = Skeleton.MST
    """The graph along which the costs are aggregated. Per default, the cost of edges on the minimum
     spanning tree (mst) are aggregated. Choose from: [mst, delaunay, diameter, complete_graph]"""

    @model_validator(mode="before")
    @classmethod
    def validate_geo_prior_parameters(cls, values):
        if (values.get("type") == "cost_based") and (values.get("rate") is None):
            raise ValidationError(
                "Field `rate` is required for geo-prior of type `cost_based`."
            )
        return values


class WeightsPriorConfig(DirichletPriorConfig):
    """Configuration of the prion on the weights of the mixture components."""


class ConfoundingEffectConfig(BaseConfig):
    """Configuration of the prior on the parameters of the confounding-effects."""
    categorical: DirichletPriorConfig | None = None
    gaussian: GaussianPriorConfig | None = None
    poisson: PoissonPriorConfig | None = None


class ClusterEffectConfig(DirichletPriorConfig):
    """Configuration of the prior on the parameters of the cluster-effect."""
    categorical: DirichletPriorConfig | None = None
    gaussian: GaussianPriorConfig | None = None
    poisson: PoissonPriorConfig | None = None


class PriorConfig(BaseConfig):

    """Configuration of all priors of a sBayes model."""

    confounding_effects: Dict[str, Dict[str, ConfoundingEffectConfig]]
    """The priors for the confounding_effects in each group of each confounder."""

    cluster_effect: ClusterEffectConfig
    geo: GeoPriorConfig
    cluster_assignment: ClusterPriorConfig
    weights: WeightsPriorConfig

    @model_validator(mode="before")
    @classmethod
    def validate_confounder_priors(cls, values):
        """Ensure that priors are defined for each confounder."""
        if "objects_per_cluster" in values:
                raise NameError(f"The `objects_per_cluster` config has been changed to generally describe the "
                                f"distribution of cluster assignments and renamed to `cluster_assignment`.")
        return values


class ModelConfig(BaseConfig):

    """Configuration of the sBayes model."""

    clusters: Union[int, List[int]] = 1
    """The number of clusters to be inferred."""

    confounders: List[str] = Field(default_factory=list)
    """The list of confounder names."""

    prior: PriorConfig
    """The config section defining the priors of the model."""

    sample_from_prior: bool = False
    """If `true`, the data is ignored and parameters are sampled from the prior distribution."""

    @classmethod
    def deprecated_attributes(cls) -> list:
        return ["sample_source"]

    @model_validator(mode="before")
    @classmethod
    def validate_confounder_priors(cls, values):
        """Ensure that priors are defined for each confounder."""
        for conf in values['confounders']:
            if conf not in values['prior']['confounding_effects']:
                raise NameError(f"Prior for the confounder \'{conf}\' is not defined in the config file.")
        return values


class WarmupConfig(BaseConfig):

    """Configuration of the warm-up phase in the MCMC chain."""

    warmup_steps: PositiveInt = 50000
    """The number of steps performed in the warm-up phase."""

    warmup_chains: PositiveInt = 10
    """The number parallel chains used in the warm-up phase."""


class MC3Config(BaseConfig):

    """Configuration of Metropolis-Coupled Markov Chain Monte Carlo (MC3) parameters."""

    activate: bool = False
    """If `true`, use Metropolis-Coupled Markov Chain Monte Carlo sampling (MC3)."""

    chains: PositiveInt = 4
    """Number of MC3 chains."""

    swap_interval: PositiveInt = 1000
    """Number of MCMC steps between each MC3 chain swap attempt."""

    _swap_attempts: PositiveInt = 100
    """Number of chain pairs which are proposed to be swapped after each interval."""

    _only_swap_adjacent_chains: bool = False
    """Only swap chains that are next to each other in the temperature schedule."""

    temperature_diff: PositiveFloat = 0.05
    """Difference between temperatures of MC3 chains."""

    prior_temperature_diff: PositiveFloat = "temperature_diff"
    """Difference between prior-temperatures of MC3 chains. Defaults to the same values as `temperature_diff`."""

    exponential_temperatures: bool = False
    """If `true`, temperature increase exponentially ((1 + dt)**i), instead of linearly (1 + dt*i)."""

    log_swap_matrix: bool = True
    """If `True`, write a matrix containing the number of swaps between each pair of chains to an npy-file."""

    @classmethod
    def deprecated_attributes(cls) -> list:
        return ["only_heat_likelihood", "swap_attempts", "only_swap_adjacent_chains"]

    @model_validator(mode="after")
    def validate_mc3(self):
        if self.activate and self.chains < 2:
            self.activate = False
            warnings.warn(f"Deactivated MC3, as it is pointless with less than 2 chains.")

        # The number of swap attempts cannot exceed the number of valid chain pairs. The
        # number of valid chain pairs depends on whether we restrict swaps to adjacent
        # chains.
        if self._only_swap_adjacent_chains:
            valid_chain_pairs = self.chains - 1
        else:
            valid_chain_pairs = int(self.chains * (self.chains - 1) / 2)
        if self._swap_attempts > valid_chain_pairs:
            self._swap_attempts = valid_chain_pairs

        # Per default `prior_temperature_diff` is the same as `temperature_diff`.
        if self.prior_temperature_diff == "temperature_diff":
            self.prior_temperature_diff = self.temperature_diff

        return self


class MCMCConfig(BaseConfig):

    """Configuration of MCMC parameters."""

    steps: PositiveInt = 1000000
    """The total number of iterations in the MCMC chain."""

    samples: PositiveInt = 1000
    """The number of samples to be generated (more samples implies lower sampling interval)."""

    runs: PositiveInt = 1
    """The number of times the sampling is repeated (with new output files for each run)."""

    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    mc3: MC3Config = Field(default_factory=MC3Config)

    @classmethod
    def deprecated_attributes(cls) -> list:
        return ["sample_from_prior", "operators", "init_objects_per_cluster", "initialization", "grow_to_adjacent", "screen_log_interval"]

    @model_validator(mode="after")
    def validate_sample_spacing(self):
        # Tracer does not like unevenly spaced samples
        spacing = self.steps % self.samples
        if spacing != 0.:
            raise ValueError("Inconsistent spacing between samples. Set ´steps´ to be a multiple of ´samples´.")
        return self


class DataConfig(BaseConfig):

    """Information on the data for an sBayes analysis."""

    features: RelativeFilePath
    """Path to the CSV file with features used for the analysis."""

    feature_states: Optional[RelativeFilePath] = None
    """Path to the CSV file defining the possible states for each feature."""

    feature_types: Optional[RelativeFilePath] = None
    """Path to the YAML file defining the type and support of each feature."""

    projection: str = "epsg:4326"
    """String identifier of the projection in which locations are given."""

    @model_validator(mode="after")
    def validate_feature_types(self):
        """Ensure that either feature_types or feature_states file is provided."""
        if self.feature_types is None:
            if self.feature_states is None:
                raise ValidationError("Provide either `feature_types` or `feature_states` for the data.")
            else:
                warnings.warn("The `feature_states` field is deprecated. Please use `feature_types` instead.")

        return self


class ResultsConfig(BaseConfig):

    """Information on where and how results are written."""

    path: RelativeDirectoryPath = Field(
        default_factory=lambda: RelativePathType.fix_path("./results")
    )
    """Path to the results directory."""

    log_file: bool = True
    """Whether to write log-messages to a file."""

    log_likelihood: bool = True
    """Whether to log the likelihood of each observation in a .h5 file (used for model comparison)."""

    log_source: bool = False
    """Whether to log the proportion of objects assigned to each component in each feature."""

    log_hot_chains: bool = True
    """Whether to create log files (clusters, stats and operator_stats) for hot MC3 chains."""

    float_precision: PositiveInt = 8
    """The precision (number of decimal places) of real valued parameters in the stats file."""


class SettingsForLinguists(BaseConfig):

    """Optional settings that are only relevant for the analysis of linguistic areas."""

    isolates_as_universal: bool = False
    """If true, the inheritance distribution is replaced by the universal distribution for
     languages without a family. Otherwise, weights are renormalized and inheritance is 
     replaced by contact and universal (proportional to their corresponding weights)."""


class SBayesConfig(BaseConfig):

    data: DataConfig
    model: ModelConfig
    mcmc: MCMCConfig
    results: ResultsConfig = Field(default_factory=ResultsConfig)

    @classmethod
    def from_config_file(
        cls, path: PathLike, custom_settings: Optional[dict] = None
    ) -> "SBayesConfig":
        """Create an instance of SBayesConfig from a YAML or JSON config file."""

        # Prepare RelativePath class to allow paths relative to the config file location
        base_directory, config_file = decompose_config_path(path)
        RelativePathType.BASE_DIR = base_directory

        # Load a config dictionary from the json file
        with open(path, "r") as f:
            path_str = str(path).lower()
            if path_str.endswith(".yaml") or path_str.endswith("yml"):
                yaml_loader = yaml.YAML(typ='safe')
                config_dict = yaml_loader.load(f)
            else:
                config_dict = json.load(f)

        # Update the config dictionary with custom_settings
        if custom_settings:
            update_recursive(config_dict, custom_settings)

        # Create SBayesConfig instance from the dictionary
        return SBayesConfig(**config_dict)

    def update(self, other: dict):
        new_dict = update_recursive(self.dict(), other)
        return type(self)(**new_dict)
