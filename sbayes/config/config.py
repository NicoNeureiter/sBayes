import types
from pathlib import Path
from enum import Enum
from typing import Union, List, Dict, Literal, Optional
import warnings

from pydantic import BaseModel, Extra, Field
from pydantic import validator, root_validator, ValidationError
from pydantic import FilePath, PositiveInt, PositiveFloat, confloat


class BaseConfig(BaseModel):
    """The base class for all config classes. This inherits from pydantic.BaseModel and
    configures settings that should be shared across all setting classes."""

    def __getitem__(self, key):
        return self.__getattribute__(key)

    class Config:

        extra = Extra.forbid
        """Don't allow unexpected keys to be defined in a config-file."""

        allow_mutation = False
        """Make config objects immutable."""


""" ===== PRIOR CONFIGS ===== """


class GeoPriorConfig(BaseConfig):
    """Config for the geo-prior."""

    class Types(str, Enum):
        UNIFORM = 'uniform'
        COST_BASED = 'cost_based'

    class AggregationStrategies(str, Enum):
        MEAN = 'mean'
        SUM = 'sum'
        MAX = 'max'

    type: Types = Types.UNIFORM
    rate: PositiveFloat = None
    aggregation: AggregationStrategies = AggregationStrategies.MEAN
    costs: Union[FilePath, Literal['from_data']] = 'from_data'

    @root_validator
    def validate_dirichlet_parameters(cls, values):
        if (values.get('type') == 'cost_based') and (values.get('rate') is None):
            raise ValidationError('Field `rate` is required for geo-prior of type `cost_based`.')
        return values


class ClusterSizePriorConfig(BaseConfig):
    """Config for area size prior."""

    class Types(str, Enum):
        UNIFORM_AREA = 'uniform_area'
        UNIFORM_SIZE = 'uniform_size'
        QUADRATIC_SIZE = 'quadratic'

    type: Types
    min: PositiveInt = 2
    max: PositiveInt = 10000


class DirichletPriorConfig(BaseConfig):

    class Types(str, Enum):
        UNIFORM = 'uniform'
        DIRICHLET = 'dirichlet'

    type: Types = Types.UNIFORM
    file: Optional[FilePath] = None
    parameters: Optional[dict] = None

    @root_validator(pre=True)
    def warn_when_using_default_type(cls, values):
        if 'type' not in values:
            warnings.warn(f'No `type` defined for `{cls.__name__}`. Using `uniform` as a default.')
        return values

    @root_validator(pre=True)
    def validate_dirichlet_parameters(cls, values):
        if values.get('type') == 'dirichlet':
            if (values.get('file') is None) and (values.get('parameters') is None):
                raise ValidationError(f'Provide `file` or `parameters` for `{cls.__name__}` of type `dirichlet`.')
        return values

    def dict(self, *args, **kwargs):
        """A custom dict method to hide non-applicable attributes depending on prior type."""
        self_dict = super().dict(*args, **kwargs)
        if self.type is self.Types.UNIFORM:
            self_dict.pop('file')
            self_dict.pop('parameters')
        else:
            if self.file is not None:
                self_dict.pop('parameters')
            elif self.parameters is not None:
                self_dict.pop('file')

        return self_dict


class WeightsPriorConfig(DirichletPriorConfig):
    """Config for prion on the weights of the mixture components."""


class ConfoundingEffectsPriorConfig(DirichletPriorConfig):
    """Config for prior on the parameters of the confounding-effects."""


class ClusterEffectConfig(DirichletPriorConfig):
    """Config for prior on the parameters of the cluster-effect."""


class PriorConfig(BaseConfig):
    """Config for all priors of a sBayes model."""

    geo: GeoPriorConfig
    objects_per_cluster: ClusterSizePriorConfig
    weights: WeightsPriorConfig
    confounding_effects: ConfoundingEffectsPriorConfig
    cluster_effect: ClusterEffectConfig


class ModelConfig(BaseConfig):

    clusters: Union[int, List[int]] = 1
    """The number of clusters to be inferred."""

    confounders: List[str] = Field(default_factory=list)
    """List of confounders to be modelled along the clusters."""

    sample_source: bool = True
    """Sample the source component for each observation (implicitly activates Gibbs sampling)."""

    prior: PriorConfig
    """The config section defining the priors of the model"""


class OperatorsConfig(BaseConfig):
    """The frequency of each MCMC operator. Will be normalized to 1.0 at runtime."""

    clusters: PositiveFloat = 45.0
    weights: PositiveFloat = 15.0
    cluster_effect: PositiveFloat = 5.0
    confounding_effects: PositiveFloat = 15.0
    source: PositiveFloat = 10.0


class WarmupConfig(BaseConfig):
    warmup_steps: PositiveInt = 50000
    """The number of steps performed in the warm-up phase."""
    warmup_chains: PositiveInt = 10
    """The number parallel chains used in the warm-up phase."""


class MCMCConfig(BaseConfig):
    steps: PositiveInt = 1000000
    samples: PositiveInt = 1000
    runs: PositiveInt = 1
    sample_from_prior: bool = False
    init_objects_per_cluster: PositiveInt = 5
    grow_to_adjacent: confloat(ge=0, le=1) = 0.8
    operators: OperatorsConfig = Field(default_factory=OperatorsConfig)
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)


class DataConfig(BaseConfig):
    """Config storing information on the data for an sBayes analysis."""

    features: FilePath
    """Path to the CSV file with features used for the analysis."""

    feature_states: FilePath
    """Path to the CSV file defining the possible states for each feature."""

    projection: str = "epsg:4326"
    """String identifies of the projection in which locations are given."""


class ResultsConfig(BaseConfig):

    path: Path = 'results'
    """Path to the results directory."""

    log_file: bool = True
    """Whether or not to write log-messages to a file."""


class SBayesConfig(BaseConfig):
    data: DataConfig
    model: ModelConfig
    mcmc: MCMCConfig
    results: ResultsConfig


def make_default_dict(cfg: type):
    if isinstance(cfg, type) and issubclass(cfg, BaseConfig):
        d = {}
        for key, field in cfg.__fields__.items():
            if field.default:
                d[key] = field.default
            else:
                d[key] = make_default_dict(field.type_)
        return d
    else:
        return '<REQUIRED>'


if __name__ == '__main__':
    cfg_1 = SBayesConfig(**{
        'data': {
            'features': 'config.py',
            'feature_states': 'config.py',
        },
        'model': {
            'prior': {
                'geo': {},
                'objects_per_cluster': {
                    'type': 'uniform_size',
                },
                'weights': {},
                'confounding_effects': {},
                'cluster_effect': {},
            },
        },
        'mcmc': {

        },
        'results': {
        },
    })

    print(cfg_1['model'])
