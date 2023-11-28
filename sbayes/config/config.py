import os
from pathlib import Path
from enum import Enum
import warnings
import json
from typing import Union, List, Dict, Optional

try:
    from typing import Annotated, Literal
except ImportError:
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


""" ===== PRIOR CONFIGS ===== """


class GeoPriorConfig(BaseConfig):

    """Configuration of the geo-prior."""

    class Types(str, Enum):
        UNIFORM = "uniform"
        COST_BASED = "cost_based"
        DIAMETER_BASED = "diameter_based"
        # GAUSSIAN = "gaussian"
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
    # costs: FilePath = "from_data"
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
    def validate_geo_prior_parameters(cls, values):
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
    @classmethod
    def warn_when_using_default_type(cls, values):
        if "type" not in values:
            warnings.warn(
                f"No `type` defined for `{cls.__name__}`. Using `uniform` as a default."
            )
        return values

    @model_validator(mode="before")
    @classmethod
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


class ConfoundingEffectPriorConfig(DirichletPriorConfig):
    """Configuration of the prior on the parameters of the confounding-effects."""


class ClusterEffectConfig(DirichletPriorConfig):
    """Configuration of the prior on the parameters of the cluster-effect."""


class PriorConfig(BaseConfig):

    """Configuration of all priors of a sBayes model."""

    confounding_effects: Dict[str, Dict[str, ConfoundingEffectPriorConfig]]
    """The priors for the confounding_effects in each group of each confounder."""

    cluster_effect: ClusterEffectConfig
    geo: GeoPriorConfig
    objects_per_cluster: ClusterSizePriorConfig
    weights: WeightsPriorConfig


class ModelConfig(BaseConfig):

    """Configuration of the sBayes model."""

    clusters: Union[int, List[int]] = 1
    """The number of clusters to be inferred."""

    confounders: List[str] = Field(default_factory=list)
    """The list of confounder names."""

    prior: PriorConfig
    """The config section defining the priors of the model"""

    @model_validator(mode="before")
    @classmethod
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

    cluster_effect: NonNegativeFloat = 5.0
    """Frequency at which cluster effect parameters are changed."""

    confounding_effects: NonNegativeFloat = 15.0
    """Frequency at which confounding effects parameters are changed."""

    source: NonNegativeFloat = 10.0
    """Frequency at which the assignments of observations to mixture components are changed."""


class WarmupConfig(BaseConfig):

    """Configuration of the warm-up phase in the MCMC chain."""

    warmup_steps: PositiveInt = 50000
    """The number of steps performed in the warm-up phase."""

    warmup_chains: PositiveInt = 10
    """The number parallel chains used in the warm-up phase."""


class InitializationConfig(BaseConfig):

    """Configuration for the initialization of the MCMC / each warm-up chain."""

    attempts: PositiveInt = 10
    """Number of initial samples for each warm-up chain. Only the one with highest posterior will be used."""

    initial_cluster_steps: bool = True
    """Whether to apply an initial cluster operator to each cluster before selecting the best sample."""

    em_steps: PositiveInt = 50
    """Number of steps in the expectation-maximization initializer."""


class MC3Config(BaseConfig):

    """Configuration of Metropolis-Coupled Markov Chain Monte Carlo (MC3) parameters."""

    activate: bool = False
    """Whether or not to use Metropolis-Coupled Markov Chain Monte Carlo sampling (MC3)."""

    chains: PositiveInt = 4
    """Number of MC3 chains."""

    swap_interval: PositiveInt = 1000
    """Number of MCMC steps between each MC3 chain swap attempt."""

    swap_attempts: PositiveInt = 2
    """Number of chain pairs which are proposed to be swapped after each interval."""

    only_swap_adjacent_chains: bool = False
    """Only swap chains that are next to each other in the temperature schedule."""

    temperature_diff: PositiveFloat = 0.05
    """Difference between temperatures of MC3 chains."""

    prior_temperature_diff: Optional[PositiveFloat] = None
    """Difference between prior-temperatures of MC3 chains. Defaults to the same values as 
    `temperature_diff` if `only_heat_likelihood == False`, and 0 otherwise."""

    only_heat_likelihood: bool = False
    """If `True`, only likelihood is affected by the MC3 temperature, i.e. all chains use the same prior."""

    exponential_temperatures: bool = False
    """If `True`, temperature increase exponentially ((1 + dt)**i), instead of linearly (1 + dt*i)."""

    log_swap_matrix: bool = True
    """If `True`, write a matrix containing the number of swaps between each pair of chains to an npy-file."""

    @model_validator(mode="after")
    def validate_mc3(self):
        if self.activate and self.chains < 2:
            self.activate = False
            warnings.warn(f"Deactivated MC3, as it is pointless with less than 2 chains.")

        # The number of swap attempts cannot exceed the number of valid chain pairs. The
        # number of valid chain pairs depends on whether we restrict swaps to adjacent
        # chains.
        if self.only_swap_adjacent_chains:
            valid_chain_pairs = self.chains - 1
        else:
            valid_chain_pairs = int(self.chains * (self.chains - 1) / 2)
        if self.swap_attempts > valid_chain_pairs:
            self.swap_attempts = valid_chain_pairs
            warnings.warn(
                f"With `only_swap_adjacent_chains={self.only_swap_adjacent_chains}` and "
                f"{self.chains} chains the number of swap attempts can not be more than "
                f"{valid_chain_pairs}. Adjusted swap_attempts={valid_chain_pairs}."
            )

        # Default behavior of `prior_temperature_diff` depends on the
        # `prior_temperature_diff` flag.
        if self.prior_temperature_diff is None:
            if self.only_heat_likelihood:
                # Prior is not heated at all -> prior_temperature_diff is 0
                self.prior_temperature_diff = 0.0
            else:
                # Prior is heated the same as the posterior
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

    sample_from_prior: bool = False
    """If `true`, the MCMC ignores the data and samples parameters from the prior distribution."""

    init_objects_per_cluster: PositiveInt = 5
    """The number of objects in the initial clusters at the start of an MCMC run."""

    grow_to_adjacent: Annotated[float, Field(ge=0, le=1)] = 0.8
    """The fraction of grow-steps that only propose adjacent languages as candidates to be added to an area."""

    screen_log_interval: PositiveInt = 1000
    """Frequency at which the step ID and log-likelihood are written to the screen logger (and log file)."""

    operators: OperatorsConfig = Field(default_factory=OperatorsConfig)
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    initialization: InitializationConfig = Field(default_factory=InitializationConfig)
    mc3: MC3Config = Field(default_factory=MC3Config)

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

    feature_states: RelativeFilePath
    """Path to the CSV file defining the possible states for each feature."""

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

    log_likelihood: bool = True
    """Whether to log the likelihood of each observation in a .h5 file (used for model comparison)."""

    log_source: bool = False
    """Whether to log the proportion of objects assigned to each component in each feature."""

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
