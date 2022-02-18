from pathlib import Path
from enum import Enum
from typing import Union, List, Dict, Literal, Optional

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


""" ----- PRIOR CONFIGS ----- """


class GeoPriorConfig(BaseConfig):
    """Config for the geo-prior."""

    class TYPES(Enum):
        UNIFORM = 'uniform'
        COST_BASED = 'cost_based'

    type: TYPES = TYPES.UNIFORM
    rate: PositiveFloat = None
    aggregation: Literal['mean', 'sum', 'max'] = 'mean'
    costs: Union[FilePath, Literal['from_data']] = 'from_data'

    @root_validator
    def validate_dirichlet_parameters(cls, values):
        if (values.get('type') == 'cost_based') and (values.get('rate') is None):
            raise ValidationError('Field `rate` is required for geo-prior of type `cost_based`.')
        return values


class ClusterSizePriorConfig(BaseConfig):
    """Config for area size prior."""

    class TYPES(Enum):
        UNIFORM_AREA = 'uniform_area'
        UNIFORM_SIZE = 'uniform_size'
        QUADRATIC_SIZE = 'quadratic'

    type: TYPES
    min: PositiveInt = 2
    max: PositiveInt = 10000


class WeightsPriorConfig(BaseConfig):
    """Config for prion on the weights of the mixture components."""

    class TYPES(Enum):
        UNIFORM = 'uniform'
        DIRICHLET = 'dirichlet'

    type: TYPES = TYPES.UNIFORM


class ConfoundingEffectsPriorConfig(BaseConfig):
    """Config for prior on the parameters of the confounding-effects."""

    class TYPES(Enum):
        UNIFORM = 'uniform'
        DIRICHLET = 'dirichlet'
        # UNIVERSAL = 'universal'

    type: TYPES
    file: Optional[FilePath] = None
    parameters: Optional[dict] = None

    @root_validator(pre=True)
    def validate_dirichlet_parameters(cls, values):
        if values.get('type') == 'dirichlet':
            if (values.get('file') is None) and (values.get('parameters') is None):
                raise ValidationError(f'Provide `file` or `parameters` for confounding-effects prior of type `dirichlet`.')
        return values


class ClusterEffectConfig(BaseConfig):
    """Config for prior on the parameters of the cluster-effect."""

    class TYPES(Enum):
        UNIFORM = 'uniform'
        DIRICHLET = 'dirichlet'
        # UNIVERSAL = 'universal'

    type: TYPES
    file: Optional[FilePath] = None
    parameters: Optional[dict] = None

    @root_validator(pre=True)
    def validate_dirichlet_parameters(cls, values):
        if values.get('type') == 'dirichlet':
            if (values.get('file') is None) and (values.get('parameters') is None):
                raise ValidationError(f'Provide `file` or `parameters` for cluster-effect prior of type `dirichlet`.')
        return values


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

    sample_source: bool = True
    """Sample the source component for each observation (implicitly activates gibbs sampling)."""

    confounders: List[str] = Field(default_factory=list)
    """List of confounders to be modelled along the clusters."""

    prior: PriorConfig
    """The config section defining the priors of the model"""


class OperatorsConfig(BaseConfig):
    clusters: PositiveFloat = 45.0
    weights: PositiveFloat = 15.0
    cluster_effect: PositiveFloat = 5.0
    confounding_effects: PositiveFloat = 15.0
    source: PositiveFloat = 10.0


class WarmupConfig(BaseConfig):
    warmup_steps: PositiveInt = 10
    warmup_chains: PositiveInt = 50000


class MCMCConfig(BaseConfig):
    steps: PositiveInt = 1000000
    samples: PositiveInt = 1000
    runs: PositiveInt = 1
    sample_from_prior: bool = False
    grow_to_adjacent: confloat(ge=0, le=1) = 0.8
    init_objects_per_cluster: PositiveInt = 5
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
                'weights': {'type': 'uniform'},
                'confounding_effects': {'type': 'uniform'},
                'cluster_effect': {'type': 'uniform'},
            },
        },
        'mcmc': {

        },
        'results': {
        },
    })

    print(cfg_1['model'])
