import os
from pathlib import Path
from enum import Enum
import warnings
import json
import io
from typing import Union, List, Dict, Optional, Any

try:
    from typing import Annotated, Literal
except ImportError:
    from typing_extensions import Annotated, Literal
try:
    import ruamel.yaml as yaml
except ImportError:
    import ruamel_yaml as yaml

from pydantic import BaseModel, Field, model_validator
from pydantic import ValidationError
from pydantic import DirectoryPath
from pydantic import PositiveInt, PositiveFloat, confloat, NonNegativeFloat
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


class GeoPriorConfig(BaseConfig):

    """Configuration of the geo-prior."""

    class Types(str, Enum):
        UNIFORM = "uniform"
        COST_BASED = "cost_based"
        DIAMETER_BASED = "diameter_based"
        SIMULATED = "simulated"

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

    type: Types = Types.UNIFORM
    """Type of prior distribution (`uniform`, `cost_based` or `gaussian`)."""

    costs: Union[RelativeFilePath, Literal["from_data"]] = "from_data"
    """Source of the geographic costs used for cost_based geo-prior. Either `from_data`
    (derive geodesic distances from locations) or path to a CSV file."""

    aggregation: AggregationStrategies = AggregationStrategies.MEAN
    """Policy defining how costs of single edges are aggregated (`mean`, `sum` or `max`)."""

    probability_function: ProbabilityFunction = ProbabilityFunction.EXPONENTIAL
    """Monotonic function that defines how costs are mapped to prior probabilities."""

    rate: Optional[PositiveFloat] = None
    """Rate at which the prior probability decreases for a cost_based geo-prior."""

    inflection_point: Optional[float] = None
    """The point where a sigmoid probability function reaches 0.5."""

    skeleton: Skeleton = Skeleton.MST
    """The graph along which the costs are aggregated. Per default, the cost of edges on 
    the minimum spanning tree are aggregated."""

    @model_validator(mode="before")
    @classmethod
    def validate_dirichlet_parameters(cls, values):
        if (values.get("type") == "cost_based") and (values.get("rate") is None):
            raise ValidationError(
                "Field `rate` is required for geo-prior of type `cost_based`."
            )
        return values


class ClusterSizePriorConfig(BaseConfig):
    """Configuration of the area size prior."""

    class Types(str, Enum):
        UNIFORM_AREA = "uniform_area"
        UNIFORM_SIZE = "uniform_size"
        QUADRATIC_SIZE = "quadratic"

    type: Types
    """Type of prior distribution (`uniform_area`, `uniform_size` or `quadratic`)."""

    min: PositiveInt = 2
    """Minimum cluster size."""

    max: PositiveInt = 10000
    """Maximum cluster size."""


class GaussianMeanPriorConfig(BaseConfig):
    """Configuration of the prior on the mean of a normal distribution"""

    class Types(str, Enum):
        IMPROPER_UNIFORM = "improper_uniform"
        GAUSSIAN = "gaussian"

    type: Types = Types.IMPROPER_UNIFORM
    """Type of prior distribution (`improper_uniform` or `gaussian`)."""

    file: Optional[RelativeFilePath] = None
    """Path to the parameters of the Gaussian distribution."""

    parameters: Optional[dict] = None
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

    type: Types = Types.JEFFREYS
    """Type of prior distribution (`improper_uniform` or `gaussian`)."""

    file: Optional[RelativeFilePath] = None
    """Path to the parameters of the Gaussian distribution."""

    parameters: Optional[dict] = None
    """Parameters of the Gaussian distribution."""

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

    parameters: Optional[dict] = None
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
        JEFFREYS = "jeffreys"
        BBS = "BBS"
        UNIVERSAL = "universal"
        SYMMETRIC_DIRICHLET = "symmetric_dirichlet"

    type: Types = Types.UNIFORM
    """Type of prior distribution (`uniform` or `dirichlet`)."""

    file: Optional[RelativeFilePath] = None
    """Path to the parameters of the Dirichlet distribution."""

    parameters: Optional[dict] = None
    """Parameters of the Dirichlet distribution."""

    prior_confounder: Optional[str] = None
    """A string indicating which confounder should be used as the mean of the prior."""

    prior_concentration: Optional[float] = None
    """If another confounder is used as mean, we need to manually define the concentration of the Dirichlet prior."""

    @model_validator(mode="before")
    def warn_when_using_default_type(cls, values):
        if "type" not in values:
            warnings.warn(
                f"No `type` defined for `{cls.__name__}`. Using `uniform` as a default."
            )
        return values

    @model_validator(mode="before")
    def validate_dirichlet_parameters(cls, values):
        prior_type = values.get("type")

        if prior_type == "dirichlet":
            if (values.get("file") is None) and (values.get("parameters") is None):
                raise ValidationError(
                    f"Provide `file` or `parameters` for `{cls.__name__}` of type `dirichlet`."
                )

        elif prior_type in ["universal", "symmetric_dirichlet"]:
            if values.get("prior_concentration") is None:
                raise ValidationError(f"Provide `prior_concentration` for `{cls.__name__}` of type `{prior_type}`.")

        return values

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


class WeightsPriorConfig(DirichletPriorConfig):
    """Configuration of the prion on the weights of the mixture components."""


class ClusterEffectCategoricalPriorConfig(DirichletPriorConfig):
    """Configuration of the prior on the cluster-effect for categorical features."""


class ClusterEffectGaussianPriorConfig(GaussianPriorConfig):
    """Configuration of the prior on the cluster-effect of continuous features."""


class ClusterEffectPoissonPriorConfig(PoissonPriorConfig):
    """Configuration of the prior on the cluster-effect of count features."""


class ConfoundingEffectsCategoricalPriorConfig(DirichletPriorConfig):
    """Configuration of the prior on the confounding-effects of categorical features."""


class ConfoundingEffectsGaussianPriorConfig(GaussianPriorConfig):
    """Configuration of the prior on the confounding-effects of continuous features."""


class ConfoundingEffectsPoissonPriorConfig(PoissonPriorConfig):
    """Configuration of the prior on the confounding-effects of count features."""


class ConfoundingEffectsPriorConfig(BaseConfig):
    """Configuration of the prior on the parameters of the confounding-effects."""
    categorical: ConfoundingEffectsCategoricalPriorConfig
    gaussian: ConfoundingEffectsGaussianPriorConfig
    poisson: ConfoundingEffectsPoissonPriorConfig


class ClusterEffectPriorConfig(BaseConfig):
    """Configuration of the prior on the parameters of the cluster-effect."""
    categorical: ClusterEffectCategoricalPriorConfig
    gaussian: ClusterEffectGaussianPriorConfig
    poisson: ClusterEffectPoissonPriorConfig


class PriorConfig(BaseConfig):

    """Configuration of all priors of a sBayes model."""

    confounding_effects: Dict[str, Dict[str, ConfoundingEffectsPriorConfig]]
    """The priors for the confounding_effects in each group of each confounder."""

    cluster_effect: ClusterEffectPriorConfig
    geo: GeoPriorConfig
    objects_per_cluster: ClusterSizePriorConfig
    weights: WeightsPriorConfig


class ModelConfig(BaseConfig):

    """Configuration of the sBayes model."""

    clusters: Union[int, List[int]] = 1
    """The number of clusters to be inferred."""

    # confounders: OrderedDictType[str, List[str]] = Field(default_factory=OrderedDict)
    # """Dictionary with confounders as keys and lists of corresponding groups as values."""
    confounders: List[str] = Field(default_factory=list)
    """The list of confounder names."""

    sample_source: bool = True
    """Sample the source component for each observation (implicitly activates Gibbs sampling)."""

    prior: PriorConfig
    """The config section defining the priors of the model"""

    @model_validator(mode="before")
    def validate_confounder_priors(cls, values):
        """Ensure that priors are defined for each confounder."""
        for conf in values['confounders']:
            if conf not in values['prior']['confounding_effects']:
                raise NameError(f"Prior for the confounder \'{conf}\' is not defined in the config file.")
        return values


class OperatorsConfig(BaseConfig):

    """The frequency of each MCMC operator. Will be normalized to 1.0 at runtime."""

    clusters: NonNegativeFloat = 45.0
    """Frequency at which the assignment of objects to clusters is changed."""

    weights: NonNegativeFloat = 15.0
    """Frequency at which mixture weights are changed."""

    source: NonNegativeFloat = 10.0
    """Frequency at which the assignments of observations to mixture components are changed."""

    @classmethod
    def deprecated_attributes(cls) -> list:
        return ["cluster_effect", "confounding_effects"]


class WarmupConfig(BaseConfig):

    """Configuration of the warm-up phase in the MCMC chain."""

    warmup_steps: PositiveInt = 50000
    """The number of steps performed in the warm-up phase."""

    warmup_chains: PositiveInt = 10
    """The number parallel chains used in the warm-up phase."""


class MCMCConfig(BaseConfig):

    """Configuration of MCMC parameters."""

    steps: PositiveInt = 1000000
    """The total number of iterations in the MCMC chain."""

    samples: PositiveInt = 1000
    """The number of samples to be generated (more samples implies lower sampling interval)."""

    runs: PositiveInt = 1
    """The number of times the sampling is repeated (with new output files for each run)."""

    sample_from_prior: bool = False
    """If `true`, the MCMC ignores the data and samples parameters from the prior distribution."""

    init_objects_per_cluster: PositiveInt = 5
    """The number of objects in the initial clusters at the start of an MCMC run."""

    grow_to_adjacent: confloat(ge=0, le=1) = 0.8
    """The fraction of grow-steps that only propose adjacent languages as candidates to be added to an area."""

    operators: OperatorsConfig = Field(default_factory=OperatorsConfig)
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)

    @model_validator(mode="before")
    def validate_sample_spacing(cls, values):
        # Tracer does not like unevenly spaced samples
        spacing = values['steps'] % values['samples']
        if spacing != 0.:
            raise ValueError("Inconsistent spacing between samples. Set ´steps´ to be a multiple of ´samples´.")
        return values


class DataConfig(BaseConfig):

    """Information on the data for an sBayes analysis."""

    features: RelativeFilePath
    """Path to the CSV file with features used for the analysis."""

    feature_types: RelativeFilePath
    """Path to the YAML file defining the type of each feature including applicable states."""

    projection: str = "epsg:4326"
    """String identifier of the projection in which locations are given."""


class ResultsConfig(BaseConfig):

    """Information on where and how results are written."""

    path: RelativeDirectoryPath = Field(
        default_factory=lambda: RelativePathType.fix_path("./results")
    )
    """Path to the results directory."""

    log_file: bool = True
    """Whether to write log-messages to a file."""


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
    # settings_for_linguists: SettingsForLinguists = Field(default_factory=SettingsForLinguists)

    @model_validator(mode="before")
    def validate_operators(cls, values):
        # Do not use source operators if sampling from source is disabled
        if not values['model']['sample_source']:
            values['mcmc']['operators']['source'] = 0.0
        return values

    @classmethod
    def from_config_file(
        cls, path: PathLike, custom_settings: Optional[dict] = None
    ) -> "SBayesConfig":
        """Create an instance of SBayesConfig from a YAML or JSON config file."""

        # Prepare RelativePath class to allow paths relative to the config file location
        base_directory, config_file = decompose_config_path(path)
        RelativePathType.BASE_DIR = base_directory

        # Load a config dictionary from the json / YAML file
        with open(path, "r") as f:
            path_str = str(path).lower()
            if path_str.endswith(".yaml"):
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
