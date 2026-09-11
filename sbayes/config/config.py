from __future__ import annotations

import secrets
import warnings
import json

from enum import Enum
from pathlib import Path
from pydantic import model_validator, BaseModel, Field, ConfigDict
from pydantic import PositiveInt, PositiveFloat, NonNegativeFloat, NonNegativeInt
from pydantic.types import PathType
from pydantic_core import core_schema, PydanticCustomError
from sbayes.util import fix_relative_path, decompose_config_path, PathLike
from sbayes.util import update_recursive
from typing import (
    Annotated, Any, ClassVar, Dict, List, Literal, Optional, Self, Union
)

try:
    import ruamel.yaml as yaml
except ImportError:
    import ruamel_yaml as yaml


class RelativePathType(PathType):

    """Pydantic path type that resolves relative paths against `BASE_DIR`.

    `BASE_DIR` is global class state, set once per config load in
    `SBayesConfig.from_config_file`, so that paths in a config file are
    interpreted relative to that config file's directory.
    """

    BASE_DIR: Path = Path(".")

    @classmethod
    def fix_path(cls, value: PathLike) -> Path:
        """Resolve `value` against the current `BASE_DIR`."""
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
        # Note: this validator has a side effect - it creates the directory if it does
        # not exist yet, so that output directories do not have to be prepared by hand.
        path = RelativePathType.fix_path(path)
        path.mkdir(parents=True, exist_ok=True)
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

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def __getitem__(self, key: str):
        """Allow dict-style access to config fields (raises AttributeError if unknown)."""
        return getattr(self, key)

    @classmethod
    def get_attr_doc(cls, attr: str) -> str | None:
        """Return the docstring of a config field, if it has been harvested.

        `__attrdocs__` is populated externally by `sbayes.config.generate_template`,
        so this returns None for classes that have not been through that process.
        """
        return getattr(cls, "__attrdocs__", {}).get(attr)

    @classmethod
    def annotations(cls, key: str) -> Union[str, None]:
        """Return the type annotation of `key` as a string, searching this class and
        its BaseConfig ancestors (annotations are strings due to `from __future__
        import annotations`)."""
        if key in cls.__annotations__:
            return cls.__annotations__[key]
        for base_cls in cls.__bases__:
            if issubclass(base_cls, BaseConfig):
                s = base_cls.annotations(key)
                if s is not None:
                    return s
        return None

    @classmethod
    def deprecated_attributes(cls) -> list[str]:
        """Config keys that are still accepted, but warned about and dropped."""
        return []

    @model_validator(mode="before")
    @classmethod
    def warn_about_deprecated_attributes(cls, values: Any) -> Any:
        """Warn about and remove deprecated keys before validation.

        Removing them is required: `extra='forbid'` would otherwise turn a
        deprecation warning into a hard validation error.
        """
        if not isinstance(values, dict):
            return values

        deprecated = [key for key in cls.deprecated_attributes() if key in values]
        if not deprecated:
            return values

        values = dict(values)  # don't mutate the caller's dict
        for key in deprecated:
            warnings.warn(f"The {key} key in {cls.__name__} is deprecated "
                          f"and will be removed in future versions of sBayes.")
            values.pop(key)
        return values


""" ===== PRIOR CONFIGS ===== """

class TypedPriorConfig(BaseConfig):

    """Base class for prior configs whose `type` field has a default value.

    Subclasses must declare a `type` field with a default and set
    `DEFAULT_TYPE_NAME` to the value of that default. Configs where `type` is
    required (no default) should inherit from `BaseConfig` instead and rely on
    pydantic's own required-field validation.
    """

    DEFAULT_TYPE_NAME: ClassVar[str]

    @model_validator(mode="before")
    @classmethod
    def warn_when_using_default_type(cls, values):
        if isinstance(values, dict) and "type" not in values:
            warnings.warn(
                f"No `type` defined for `{cls.__name__}`. "
                f"Using `{cls.DEFAULT_TYPE_NAME}` as a default."
            )
        return values


class GaussianMeanPriorConfig(TypedPriorConfig):
    """Configuration of the prior on the mean of a normal distribution"""

    class Types(str, Enum):
        GAUSSIAN = "gaussian"

    DEFAULT_TYPE_NAME: ClassVar[str] = Types.GAUSSIAN.value

    type: Types = Types.GAUSSIAN
    """Type of prior distribution (`gaussian`)."""

    file: Optional[RelativeFilePath] = None
    """Path to the parameters of the Gaussian distribution."""

    parameters: Optional[Dict[str, float]] = None
    """Parameters of the Gaussian distribution."""

    @model_validator(mode="before")
    @classmethod
    def validate_gaussian_parameters(cls, values: Any) -> Any:
        """A `gaussian` prior requires either a parameter file or explicit parameters."""
        if not isinstance(values, dict):
            return values

        if values.get("type") == cls.Types.GAUSSIAN:
            if (values.get("file") is None) and (values.get("parameters") is None):
                raise ValueError(
                    f"Provide `file` or `parameters` for `{cls.__name__}` of type "
                    f"`{cls.Types.GAUSSIAN.value}`."
                )
        return values


class GaussianVariancePriorConfig(BaseConfig):
    """Configuration of the prior on the variance of a normal distribution"""

    class Types(str, Enum):
        JEFFREYS = "jeffreys"
        INV_GAMMA = "inv-gamma"
        GAMMA = "gamma"
        FIXED = "fixed"
        EXPONENTIAL = "exponential"

    type: Types
    """Type of prior distribution (`jeffreys`, `inv-gamma`, `gamma`, `fixed` or
    `exponential`)."""

    file: Optional[RelativeFilePath] = None
    """Path to the parameters of the variance prior distribution."""

    parameters: Optional[Dict[str, float]] = None
    """Parameters of the variance prior distribution."""

    @model_validator(mode="before")
    @classmethod
    def validate_variance_parameters(cls, values: Any) -> Any:
        """An `inv-gamma` prior requires either a parameter file or explicit parameters.

        TODO: `gamma`, `fixed` and `exponential` most likely require parameters as
          well - confirm against `model/prior.py` and extend this check accordingly.
        """
        if not isinstance(values, dict):
            return values

        prior_type = values.get("type")
        needs_parameters = (cls.Types.INV_GAMMA, cls.Types.EXPONENTIAL,
                            cls.Types.GAMMA, cls.Types.FIXED)

        if prior_type in needs_parameters:
            if (values.get("file") is None) and (values.get("parameters") is None):
                raise ValueError(
                    f"Provide `file` or `parameters` for `{cls.__name__}` of type "
                    f"`{prior_type}`."
                )
        return values


class GaussianPriorConfig(BaseConfig):
    """Configuration of the prior on the mean and variance of a normal distribution"""

    mean: GaussianMeanPriorConfig
    variance: GaussianVariancePriorConfig


class PoissonPriorConfig(TypedPriorConfig):
    """Configuration of the prior on the rate parameter of a Poisson distribution"""

    class Types(str, Enum):
        GAMMA = "gamma"
        JEFFREYS = "jeffreys"


    DEFAULT_TYPE_NAME: ClassVar[str] = Types.JEFFREYS.value

    type: Types = Types.JEFFREYS
    """Type of prior distribution (`jeffreys` or `gamma`)."""

    file: Optional[RelativeFilePath] = None
    """Path to the parameters of the Gamma distribution."""

    parameters: Optional[Dict[str, float]] = None
    """Parameters of the Gamma distribution."""

    @model_validator(mode="before")
    @classmethod
    def validate_gamma_parameters(cls, values: Any) -> Any:
        """A `gamma` prior requires either a parameter file or explicit parameters."""
        if not isinstance(values, dict):
            return values

        if values.get("type") == cls.Types.GAMMA:
            if (values.get("file") is None) and (values.get("parameters") is None):
                raise ValueError(
                    f"Provide `file` or `parameters` for `{cls.__name__}` of type "
                    f"`{cls.Types.GAMMA.value}`."
                )
        return values


class CategoricalPriorConfig(TypedPriorConfig):
    """Configuration of the prior on the state probabilities of a categorical feature."""

    class Types(str, Enum):
        UNIFORM = "uniform"
        DIRICHLET = "dirichlet"
        SYMMETRIC_DIRICHLET = "symmetric_dirichlet"
        LOGISTIC_NORMAL = "logistic_normal"

    DEFAULT_TYPE_NAME: ClassVar[str] = Types.UNIFORM.value

    type: Types = Types.UNIFORM
    """Type of prior distribution. Choose from: [uniform, dirichlet, symmetric_dirichlet]"""

    file: Optional[RelativeFilePath] = None
    """Path to parameters of the Dirichlet distribution (YAML or JSON format).
    This or `parameters` is required if type=dirichlet."""

    parameters: Optional[dict] = None
    """Parameters of the Dirichlet distribution. This or `file` is required if type=dirichlet."""

    prior_concentration: Optional[PositiveFloat] = None
    """The concentration of the prior distribution. Required if type=symmetric_dirichlet."""

    logistic_normal_scale: Optional[PositiveFloat] = None
    """The scale of the logistic normal prior. Required if type=logistic_normal."""

    use_parameter_transformation: bool = True
    """If `true`, use a parameter transformation to improve mixing of the MCMC chain."""

    @model_validator(mode="after")
    def validate_type_specific_parameters(self) -> Self:
        """Ensure that the parameters required by the chosen prior type are present."""
        cls_name = type(self).__name__
        if self.type == self.Types.DIRICHLET:
            if (self.file is None) and (self.parameters is None):
                raise ValueError(
                    f"Provide `file` or `parameters` for `{cls_name}` of type "
                    f"`{self.type.value}`."
                )

        elif self.type == self.Types.SYMMETRIC_DIRICHLET:
            if self.prior_concentration is None:
                raise ValueError(
                    f"Provide `prior_concentration` for `{cls_name}` of type "
                    f"`{self.type.value}`."
                )

        elif self.type == self.Types.LOGISTIC_NORMAL:
            if self.logistic_normal_scale is None:
                raise ValueError(
                    f"Provide `logistic_normal_scale` for `{cls_name}` of type "
                    f"`{self.type.value}`."
                )

        return self


class LogisticNormalPriorConfig(BaseConfig):
    """Configuration of a logistic normal prior."""

    loc: float = 0.0
    """The mean of the logistic normal prior."""

    scale: PositiveFloat = 1.0
    """The scale of the logistic normal prior."""


class ClusterPriorConfig(BaseConfig):
    """Configuration of the cluster assignment prior."""

    class Types(str, Enum):
        CATEGORICAL = "categorical"
        DIRICHLET = "dirichlet"
        LOGISTIC_NORMAL = "logistic_normal"

    type: Types
    """Type of prior distribution. Choose from: [categorical, dirichlet or logit_normal]."""

    hierarchical: bool = False
    """If `true`, use a hierarchical Dirichlet prior for the cluster assignment."""

    estimate_no_cluster_concentration: bool = False
    """If `true`, estimate the probability for not being in a cluster using MCMC."""

    no_cluster_concentration: float | None = None
    """Concentration for the 'no cluster' component of the dirichlet distribution."""

    dirichlet_config: Optional[CategoricalPriorConfig] = None
    """Configuration of the Dirichlet prior for the cluster assignment."""

    logistic_normal_config: Optional[LogisticNormalPriorConfig] = None
    """Configuration of the Logistic Normal prior for the cluster assignment."""

    stretch_and_clip: bool = False
    """If `stretch_and_clip`, stretch the 'no cluster' component up by the `stretch_factor`
    and clip the resulting cluster assignment vector back to the probability simplex."""

    stretch_factor: PositiveFloat = 1.0
    """Factor by which to stretch the 'no cluster' component of the cluster prior."""

    cluster_mask: bool = False
    """If `true`, estimate a mask that fuzzily deactivates single clusters."""

    cluster_mask_concentration: PositiveFloat = 1.0
    """Concentration of the cluster mask (only used if `cluster_mask` is set)."""

    min: PositiveInt = 2
    """Minimum cluster size (currently not enforced, see TODO below)."""

    max: PositiveInt = 10000
    """Maximum cluster size (currently not enforced, see TODO below)."""

    @model_validator(mode="after")
    def validate_type_specific_config(self) -> Self:
        """Ensure that the config section required by the chosen prior type is present."""
        required = {
            self.Types.DIRICHLET: ("dirichlet_config", self.dirichlet_config),
            self.Types.LOGISTIC_NORMAL: ("logistic_normal_config", self.logistic_normal_config),
        }
        if self.type in required:
            name, value = required[self.type]
            if value is None:
                raise ValueError(
                    f"A `{self.type.value}` cluster prior requires a `{name}` section."
                )
        return self

    # NN: min and max bounds are tricky to enforce with continuous assignments and are
    # currently ignored by the model.
    # TODO: Discuss if there is demand and how it could be implemented.


class GeoPriorConfig(TypedPriorConfig):
    """Configuration of the geo-prior."""

    class Types(str, Enum):
        UNIFORM = "uniform"
        COST_BASED = "cost_based"

    class AggregationStrategies(str, Enum):
        MEAN = "mean"
        SUM = "sum"
        SUM_OF_MEAN = "sum_of_mean"
        MAX = "max"

    class ProbabilityFunction(str, Enum):
        EXPONENTIAL = "exponential"
        GAMMA_EXPONENTIAL = "gamma_exponential"
        SQUARED_EXPONENTIAL = "squared_exponential"
        SIGMOID = "sigmoid"

    class Skeleton(str, Enum):
        MST = "mst"
        DELAUNAY = "delaunay"
        DIAMETER = "diameter"  # i.e. the longest shortest path between two nodes
        COMPLETE = "complete_graph"
        SPECTRAL = "spectral"

    DEFAULT_TYPE_NAME: ClassVar[str] = Types.UNIFORM.value

    type: Types = Types.UNIFORM
    """Type of prior distribution. Choose from: [uniform, cost_based]."""

    costs: Union[RelativeFilePath, Literal["from_data"]] = "from_data"
    # costs: FilePath = "from_data"
    """Source of the geographic costs used for cost_based geo-prior. Either `from_data`
    (derive geodesic distances from locations) or path to a CSV file."""

    aggregation: AggregationStrategies = AggregationStrategies.SUM_OF_MEAN
    """Policy defining how costs of single edges are aggregated. Choose from: [mean, sum, sum_of_mean or max]."""

    probability_function: ProbabilityFunction = ProbabilityFunction.EXPONENTIAL
    """Monotonic function that defines how aggregated costs are mapped to prior probabilities."""

    rate: Optional[PositiveFloat] = None
    """Rate at which the prior probability decreases for a cost_based geo-prior. Required if type=cost_based."""

    inflection_point: Optional[float] = None
    """Value where the sigmoid probability function reaches 0.5. Required if type=cost_based
    and probability_function=sigmoid."""

    skeleton: Skeleton = Skeleton.COMPLETE
    """The graph along which the costs are aggregated. Per default, the cost of edges on the minimum
     spanning tree (mst) are aggregated. Choose from: [mst, delaunay, diameter, complete_graph]"""

    estimate_rate: bool = False
    """If `true`, estimate the rate parameter of the geo-prior using MCMC."""

    approx_norm_const: dict[str, int] = Field(
        default_factory=lambda: {"grid_size": 40, "steps_per_setting": 200}
    )
    """Settings for the numerical approximation of the geo-prior normalization constant:
    the resolution of the parameter grid (`grid_size`) and the number of Monte Carlo
    steps per grid point (`steps_per_setting`)."""

    @model_validator(mode="after")
    def validate_skeleton(self) -> Self:
        """Reject skeleton types that are not implemented."""
        if self.skeleton is self.Skeleton.MST:
            raise ValueError(
                "The `mst` skeleton is not supported: the minimum spanning tree cannot "
                "be computed on fuzzy cluster assignments."
                f"Use `{self.skeleton.COMPLETE.value}`, `{self.skeleton.SPECTRAL.value}` or "
                f"`{self.skeleton.DIAMETER.value}` instead."
            )
        return self
    
    @model_validator(mode="before")
    @classmethod
    def validate_geo_prior_parameters(cls, values: Any) -> Any:
        """Ensure that the parameters required by the chosen geo-prior are present."""
        if not isinstance(values, dict):
            return values

        if values.get("type") == cls.Types.COST_BASED:
            if values.get("rate") is None:
                raise ValueError(
                    f"Field `rate` is required for geo-prior of type "
                    f"`{cls.Types.COST_BASED.value}`."
                )
            if (values.get("probability_function") == cls.ProbabilityFunction.SIGMOID
                    and values.get("inflection_point") is None):
                raise ValueError(
                    f"Field `inflection_point` is required for geo-prior of type "
                    f"`{cls.Types.COST_BASED.value}` with probability function "
                    f"`{cls.ProbabilityFunction.SIGMOID.value}`."
                )
        return values


class GammaDistributionConfig(BaseConfig):
    """Shifted Gamma prior: offset + Gamma(shape, rate)."""

    shape: PositiveFloat
    """Shape parameter (α) of the Gamma distribution."""

    rate: PositiveFloat
    """Rate parameter (β) of the Gamma distribution. Higher values concentrate the distribution."""

    offset: NonNegativeFloat = 0.0
    """Lower bound / shift."""

    @property
    def mean(self) -> float:
        """Prior mean."""
        return self.offset + self.shape / self.rate


class WeightsPriorConfig(CategoricalPriorConfig):
    """Prior settings for mixture weights."""

    varying_cluster_weights: bool = False
    """Allow cluster weights to vary across clusters."""

    hierarchical: bool = False
    """Use a hierarchical prior on weights."""

    concentration_prior: GammaDistributionConfig = Field(
        default_factory=lambda: GammaDistributionConfig(shape=8.0, rate=8.0)
    )
    """Prior for the hierarchical concentration. May be given as a
    (shape, rate) or (shape, rate, offset) tuple."""

    cluster_weight_factor_concentration: GammaDistributionConfig = Field(
        default_factory=lambda: GammaDistributionConfig(shape=4.0, rate=8.0)
    )
    """Gamma prior on the concentration of the per-cluster weight scaling factor (if
    varying_cluster_weights). May be given as a (shape, rate) or (shape, rate, offset)
    tuple."""

    GAMMA_TUPLE_FIELDS: ClassVar[tuple[str, ...]] = (
        "concentration_prior",
        "cluster_weight_factor_concentration",
    )

    @model_validator(mode='before')
    @classmethod
    def convert_tuple_to_gamma_config(cls, values: Any) -> Any:
        """Allow (shape, rate) or (shape, rate, offset) tuples as Gamma parameters."""
        if not isinstance(values, dict):
            return values

        converted = {}
        for key in cls.GAMMA_TUPLE_FIELDS:
            prior = values.get(key)
            if isinstance(prior, (tuple, list)):
                if len(prior) == 2:
                    converted[key] = {'shape': prior[0], 'rate': prior[1], 'offset': 0.0}
                elif len(prior) == 3:
                    converted[key] = {'shape': prior[0], 'rate': prior[1], 'offset': prior[2]}
                else:
                    raise ValueError(
                        f"`{key}` in `{cls.__name__}` must be given as a "
                        f"(shape, rate) or (shape, rate, offset) tuple, "
                        f"but has {len(prior)} elements."
                    )

        if converted:
            values = {**values, **converted}  # don't mutate the caller's dict
        return values


class ConfoundingEffectConfig(BaseConfig):
    """Configuration of the prior on the parameters of the confounding-effects."""
    categorical: CategoricalPriorConfig | None = None
    gaussian: GaussianPriorConfig | None = None
    poisson: PoissonPriorConfig | None = None


class ClusterEffectConfig(BaseConfig):
    """Configuration of the prior on the parameters of the cluster-effect."""
    categorical: CategoricalPriorConfig | None = None
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
    def reject_renamed_objects_per_cluster(cls, values: Any) -> Any:
        """Reject the old `objects_per_cluster` key, which was renamed (not deprecated)."""
        if not isinstance(values, dict):
            return values

        if "objects_per_cluster" in values:
            raise ValueError(
                "The `objects_per_cluster` config has been changed to generally "
                "describe the distribution of cluster assignments and renamed to "
                "`cluster_assignment`."
            )
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
    def deprecated_attributes(cls) -> list[str]:
        return ["sample_source"]

    @model_validator(mode="before")
    @classmethod
    def validate_confounder_priors(cls, values: Any) -> Any:
        """Ensure that a prior is defined for each confounder."""
        if not isinstance(values, dict):
            return values

        prior = values.get('confounders', [])
        if not isinstance(prior, dict):
            # `prior` is missing or already a PriorConfig instance: leave the required-
            # field check to pydantic and the PriorConfig validators.
            return values

        confounding_effects = prior.get("confounding_effects") or {}
        for conf in values.get("confounders") or []:
            if conf not in confounding_effects:
                raise ValueError(
                    f"Prior for the confounder '{conf}' is not defined in the config file."
                )
        return values

    @model_validator(mode="after")
    def deactivate_dirichlet_transform_when_sampling_from_prior(self) -> Self:
        """Disable the categorical parameter transformation when sampling from the prior.

        The transformation is not supported in this mode, so an explicit setting is
        overridden with a warning.
        """
        if not self.sample_from_prior:
            return self

        categorical_priors = [self.prior.cluster_effect.categorical]
        for conf_eff in self.prior.confounding_effects.values():
            categorical_priors.extend(grp.categorical for grp in conf_eff.values())

        for prior in categorical_priors:
            if prior is not None and prior.use_parameter_transformation:
                warnings.warn(
                    "`use_parameter_transformation` is not supported when sampling "
                    "from the prior. Disabling it."
                )
                prior.use_parameter_transformation = False

        return self


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

    prior_temperature_diff: Optional[PositiveFloat] = None
    """Difference between prior-temperatures of MC3 chains. Defaults to the same value as
    `temperature_diff`."""

    exponential_temperatures: bool = False
    """If `true`, temperature increase exponentially ((1 + dt)**i), instead of linearly (1 + dt*i)."""

    log_swap_matrix: bool = True
    """If `True`, write a matrix containing the number of swaps between each pair of chains to an npy-file."""

    @classmethod
    def deprecated_attributes(cls) -> list[str]:
        return ["only_heat_likelihood", "swap_attempts", "only_swap_adjacent_chains"]

    @model_validator(mode="after")
    def validate_mc3(self) -> Self:
        """Deactivate MC3 for single chains, cap the swap attempts and fill in defaults."""
        if self.activate and self.chains < 2:
            self.activate = False
            warnings.warn("Deactivated MC3, as it is pointless with less than 2 chains.")

        # The number of swap attempts cannot exceed the number of valid chain pairs. The
        # number of valid chain pairs depends on whether we restrict swaps to adjacent
        # chains.
        if self._only_swap_adjacent_chains:
            valid_chain_pairs = self.chains - 1
        else:
            valid_chain_pairs = int(self.chains * (self.chains - 1) / 2)
        if self._swap_attempts > valid_chain_pairs:
            self._swap_attempts = valid_chain_pairs

        # Per default `prior_temperature_diff` is the same as `temperature_diff`. After
        # this validator it is always set, despite the Optional annotation.
        if self.prior_temperature_diff is None:
            self.prior_temperature_diff = self.temperature_diff

        return self


class MCMCConfig(BaseConfig):
    """Configuration of MCMC parameters."""

    class InferenceMode(str, Enum):
        MCMC = "MCMC"
        SVI = "SVI"

        def __str__(self) -> str:
            return self.value

    steps: PositiveInt = 1000000
    """The total number of iterations in the MCMC chain."""

    samples: PositiveInt = 1000
    """The number of samples to be generated (more samples implies lower sampling interval)."""

    runs: PositiveInt = 1
    """The number of times the sampling is repeated (with new output files for each run)."""

    initialization_strategy: Literal["SVI", "heuristic"] = "SVI"
    """How to generate an initial sample for the MCMC chain. Choose from: [SVI, heuristic]."""

    svi_guide: Literal["AutoDelta", "AutoNormal"] = "AutoDelta"
    """Guide family used for SVI-based initialization."""

    svi_steps: PositiveInt = 5_000
    """Number of optimization steps for SVI-based initialization."""

    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    mc3: MC3Config = Field(default_factory=MC3Config)

    inference_mode: InferenceMode = InferenceMode.MCMC
    """The inference algorithm to use (`mcmc` or `svi`)."""

    seed: NonNegativeInt = Field(default_factory=lambda: secrets.randbelow(2**31))
    """Random seed for reproducible runs. If not set, a random seed is drawn per run."""

    @classmethod
    def deprecated_attributes(cls) -> list[str]:
        return [
            "sample_from_prior",
            "operators",
            "init_objects_per_cluster",
            "initialization",
            "grow_to_adjacent",
            "screen_log_interval",
        ]

    @model_validator(mode="after")
    def validate_sample_spacing(self) -> Self:
        """Require `steps` to be a multiple of `samples` (Tracer dislikes uneven spacing)."""
        spacing = self.steps % self.samples
        if spacing != 0:
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
    def validate_feature_types(self) -> Self:
        """Ensure that either feature_types or feature_states file is provided."""
        if self.feature_types is None:
            if self.feature_states is None:
                raise ValueError(
                    "Provide either `feature_types` or `feature_states` for the data."
                )
            else:
                warnings.warn(
                    "The `feature_states` field is deprecated. Please use `feature_types` instead."
                )

        return self


class ResultsConfig(BaseConfig):

    """Information on where and how results are written."""

    # Note: the default is resolved against `RelativePathType.BASE_DIR` at instantiation
    # time, i.e. against the config file's directory when loaded via `from_config_file`
    # and against the current working directory otherwise.
    path: RelativeDirectoryPath = Field(
        default_factory=lambda: RelativePathType.fix_path("./results")
    )
    """Path to the results directory."""

    write_interval: PositiveInt = 1000
    """The number of MCMC steps between each write to the results file (`samples.h5`)."""

    samples_file_only: bool = False
    """If `true`, results are only written to a `samples.h5` file. If `false`, results are also transformed and stored 
    in various .txt files for clusters, stats and pointwise likelihood."""

    log_file: bool = True
    """Whether to write log-messages to a file."""

    log_likelihood: bool = True
    """Whether to log the likelihood of each observation in a .h5 file (used for model comparison)."""

    log_hot_chains: bool = True
    """Whether to create log files (clusters, stats and operator_stats) for hot MC3 chains."""

    float_precision: PositiveInt = 8
    """The precision (number of decimal places) of real valued parameters in the stats file."""


class SBayesConfig(BaseConfig):

    """Top-level configuration of an sBayes analysis."""

    data: Optional[DataConfig] = None
    """The config section defining the input data (required unless `simulation` is set)."""

    model: ModelConfig
    """The config section defining the model and its priors."""

    mcmc: MCMCConfig
    """The config section defining the MCMC sampling parameters."""

    results: ResultsConfig = Field(default_factory=ResultsConfig)
    """The config section defining where and how results are written."""

    simulation: bool = False
    """If `true`, the data is simulated instead of read from files."""

    @model_validator(mode="after")
    def validate_data(self) -> Self:
        """A `data` block is required for every analysis that is not a simulation."""
        if not self.simulation and self.data is None:
            raise ValueError("A `data` block is required for non-simulation analyses.")
        return self

    @classmethod
    def from_config_file(
        cls, path: PathLike, custom_settings: Optional[dict] = None
    ) -> Self:
        """Create an instance of SBayesConfig from a YAML or JSON config file."""

        # Prepare RelativePath class to allow paths relative to the config file location
        base_directory, _ = decompose_config_path(path)
        RelativePathType.BASE_DIR = base_directory

        # Load a config dictionary from the YAML or JSON file
        with open(path, "r") as f:
            if Path(path).suffix.lower() in (".yaml", ".yml"):
                yaml_loader = yaml.YAML(typ='safe')
                config_dict = yaml_loader.load(f)
            else:
                config_dict = json.load(f)

        # Update the config dictionary with custom_settings
        if custom_settings:
            update_recursive(config_dict, custom_settings)

        # Create SBayesConfig instance from the dictionary
        return cls(**config_dict)