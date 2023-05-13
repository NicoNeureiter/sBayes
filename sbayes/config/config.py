import os
from collections import OrderedDict
from pathlib import Path
from enum import Enum
import warnings
import json
import io
from typing import Union, List, Dict, Optional, Any
from typing import OrderedDict as OrderedDictType

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
try:
    import ruamel.yaml as yaml
except ImportError:
    import ruamel_yaml as yaml

from pydantic import BaseModel, Extra, Field
from pydantic import root_validator, ValidationError
from pydantic import FilePath, DirectoryPath
from pydantic import PositiveInt, PositiveFloat, confloat, NonNegativeFloat

from sbayes.util import fix_relative_path, decompose_config_path, PathLike
from sbayes.util import update_recursive


class RelativePath:

    BASE_DIR: DirectoryPath = "."

    @classmethod
    def fix_path(cls, value: PathLike) -> Path:
        return fix_relative_path(value, cls.BASE_DIR)


class RelativeFilePath(FilePath, RelativePath):

    @classmethod
    def __get_validators__(cls):
        yield cls.fix_path
        yield from super(RelativeFilePath, cls).__get_validators__()


class RelativeDirectoryPath(DirectoryPath, RelativePath):

    @classmethod
    def __get_validators__(cls):
        yield cls.fix_path
        yield cls.initialize_directory
        yield from super(RelativeDirectoryPath, cls).__get_validators__()

    @staticmethod
    def initialize_directory(path: PathLike):
        os.makedirs(path, exist_ok=True)
        return path


class BaseConfig(BaseModel):

    """The base class for all config classes. This inherits from pydantic.BaseModel and
    configures settings that should be shared across all setting classes."""

    def __getitem__(self, key):
        return self.__getattribute__(key)

    @classmethod
    def get_attr_doc(cls, attr: str) -> str:
        return cls.__attrdocs__.get(attr)

    class Config:

        extra = Extra.forbid
        """Do not allow unexpected keys to be defined in a config-file."""

        allow_mutation = True
        """Make config objects immutable."""


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
    """Source of the geographic costs used for cost_based geo-prior. Either `from_data`
    (derive geodesic distances from locations) or path to a CSV file."""

    aggregation: AggregationStrategies = AggregationStrategies.MEAN
    """Policy defining how costs of single edges are aggregated (`mean`, `sum` or `max`)."""

    probability_function: ProbabilityFunction = ProbabilityFunction.EXPONENTIAL
    """Monotonic function that defines how costs are mapped to prior probabilities."""

    rate: Optional[PositiveFloat]
    """Rate at which the prior probability decreases for a cost_based geo-prior."""

    inflection_point: Optional[float]
    """The point where a sigmoid probability function reaches 0.5."""

    skeleton: Skeleton = Skeleton.MST
    """The graph along which the costs are aggregated. Per default, the cost of edges on 
    the minimum spanning tree are aggregated."""

    @root_validator
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

    @root_validator(pre=True)
    def warn_when_using_default_type(cls, values):
        if "type" not in values:
            warnings.warn(
                f"No `type` defined for `{cls.__name__}`. Using `uniform` as a default."
            )
        return values

    @root_validator(pre=True)
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

    # confounders: OrderedDictType[str, List[str]] = Field(default_factory=OrderedDict)
    # """Dictionary with confounders as keys and lists of corresponding groups as values."""
    confounders: List[str] = Field(default_factory=list)
    """The list of confounder names."""

    sample_source: bool = True
    """Sample the source component for each observation (implicitly activates Gibbs sampling)."""

    prior: PriorConfig
    """The config section defining the priors of the model"""

    @root_validator(pre=True)
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

    @root_validator
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

    feature_states: RelativeFilePath
    """Path to the CSV file defining the possible states for each feature."""

    projection: str = "epsg:4326"
    """String identifier of the projection in which locations are given."""


class ResultsConfig(BaseConfig):

    """Information on where and how results are written."""

    path: RelativeDirectoryPath = Field(
        default_factory=lambda: RelativeDirectoryPath.fix_path("./results")
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

    @root_validator(pre=True)
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
        RelativePath.BASE_DIR = base_directory

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


#

# ...BLACK MAGIC STARTS HERE...

# Automatically generating yaml files with comments from the config classes and attribute
# docstrings requires some code introspection, which is a bit obscure and at this point
# not well documented.

#

#


def ruamel_yaml_dumps(thing):
    y = yaml.YAML()
    y.indent(mapping=4, sequence=4, offset=4)
    out = io.StringIO()
    y.dump(thing, stream=out)
    out.seek(0)
    return out.read()


def generate_template():
    import ast
    import re

    def is_config_class(obj: Any) -> bool:
        """Check whether the given object is a subtype of BaseConfig"""
        return isinstance(obj, type) and issubclass(obj, BaseConfig)

    def could_be_a_docstring(node) -> bool:
        """Check whether the AST node is a string constant, i.e. could be a doc-string."""
        return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)

    def analyze_class_docstrings(modulefile: str) -> dict:
        """Collect all doc-strings of attributes in each class in a nested dictionary
        of the following structure:
            {class_name: {attribute_name: doc_string}}.

        Args:
            modulefile: the name of the python module to be analysed

        Returns:
            The nested dictionary with attribute doc-strings

        """
        with open(modulefile) as fp:
            root = ast.parse(fp.read())

        alldocs = {}
        for child in root.body:
            if not isinstance(child, ast.ClassDef):
                continue

            alldocs[child.name] = docs = {}
            last = None
            for childchild in child.body:
                if could_be_a_docstring(childchild):
                    if last:  # Skip class doc string
                        s = childchild.value.s
                        # replace multiple spaces and linebreaks by single space.
                        s = re.sub('\\s+', ' ', s)
                        docs[last] = s
                elif isinstance(childchild, ast.AnnAssign):
                    last = childchild.target.id
                else:
                    last = None

        return alldocs

    for class_name, docs in analyze_class_docstrings(__file__).items():
        cls: type = globals()[class_name]
        cls.__attrdocs__ = docs

    # all_docs = analyze_class_docstrings(__file__)
    # for cls in filter(globals().values, is_config_class):
    #     if issubclass(cls, DirichletPriorConfig):
    #         all_docs[cls].set_defaults(all_docs[DirichletPriorConfig])
    #
    # def get_docstring(cls: type, attr: str):
    #     if not issubclass(cls, BaseConfig):
    #         return None
    #     cls_docs = all_docs.get(cls.__name__, {})
    #     attr_doc = cls_docs.get(attr)
    #     return attr_doc

    schema = SBayesConfig.schema()
    definitions = schema.pop('definitions')
    definitions['SBayesConfig'] = schema

    def template_literal(field: Field, type_annotation: type):
        # If there is a default, use it:
        if field.default is not None:
            if isinstance(field.default, Enum):
                return field.default.value
            else:
                return field.default

        if field.default_factory and field.default_factory() is not None:
            factory = field.default_factory
            if isinstance(factory, type):
                if issubclass(factory, BaseConfig):
                    # if the default factory is itself a Config, return the template
                    return template(factory)
                if issubclass(factory, list) or issubclass(factory, dict):
                    return factory()
            else:
                return str(factory())
            # return field.default_factory()

        # Otherwise it may be optional or required:
        if 'NoneType' in str(type_annotation):
            assert not field.required
            return '<OPTIONAL>'
        else:
            assert field.required, field
            return '<REQUIRED>'

    def template(cfg: type(BaseConfig)) -> yaml.CommentedMap:
        d = yaml.CommentedMap()
        for key, field in cfg.__fields__.items():
            if is_config_class(field.type_):
                d[key] = template(field.type_)
            else:
                d[key] = template_literal(field, cfg.__annotations__[key])

                docstring = cfg.get_attr_doc(key)
                if docstring:
                    d.yaml_add_eol_comment(key=key, comment=docstring, column=40)

        if cfg.__doc__:
            d.yaml_set_start_comment(cfg.__doc__)

        return d

    def get_indent(line: str) -> int:
        return len(line) - len(str.lstrip(line))

    d = template(SBayesConfig)
    s = ruamel_yaml_dumps(d)
    lines = []
    for line in s.split('\n'):
        if line.startswith('#'):
            indent = ' ' * (4 + get_indent(lines[-1]))
            line = indent + line
        elif line.endswith(':'):
            lines.append('')

        lines.append(line)
        # lines.append('')

    yaml_str = '\n'.join(lines)
    return yaml_str


if __name__ == "__main__":
    template_str = generate_template()
    with open('config_template.yaml', 'w') as yaml_file:
        yaml_file.write(template_str)

