from __future__ import annotations
from collections import OrderedDict
from copy import copy, deepcopy
from contextlib import contextmanager
from enum import Enum
from functools import lru_cache
from typing import Optional, Generic, TypeVar, Type, Iterator

from numpy.typing import NDArray
import numpy as np

from sbayes.load_data import Confounder, Features
from sbayes.model import model_shapes
from sbayes.model.model_shapes import ModelShapes
from sbayes.util import get_along_axis

S = TypeVar('S')
Value = TypeVar('Value', NDArray, float)
DType = TypeVar('DType', bool, float, int)
VersionType = TypeVar('VersionType', tuple, int)


class FeatureType(str, Enum):
    categorical = "categorical"
    gaussian = "gaussian"
    poisson = "poisson"
    logitnormal = "logitnormal"

    @classmethod
    def values(cls) -> Iterator[str]:
        return iter(cls)

class Parameter(Generic[Value]):

    _value: Value
    version: int

    def __init__(self, value: Value):
        self._value = value
        self.version = 0

    @property
    def value(self) -> Value:
        return self._value

    def set_value(self, new_value: Value):
        self._value = new_value
        self.version += 1


class Weights:
    categorical: ArrayParameter | None
    gaussian: ArrayParameter | None
    poisson: ArrayParameter | None
    logitnormal: ArrayParameter | None


class ArrayParameter(Parameter[NDArray[DType]], Generic[DType]):

    def __init__(self, value: NDArray[DType], shared=False):
        super().__init__(value)
        self._value.flags.writeable = False
        self.shared = shared

    @property
    def shape(self) -> tuple[int]:
        return self.value.shape

    def set_value(self, new_value: NDArray[DType]):
        super().set_value(new_value)
        self._value.flags.writeable = False
        self.shared = False

    def set_items(self, keys, values):
        if self.shared:
            self.resolve_sharing()

        self._value.flags.writeable = True
        self._value[keys] = values
        self._value.flags.writeable = False
        self.version += 1

    @contextmanager
    def edit(self) -> NDArray[DType]:
        if self.shared:
            self.resolve_sharing()

        self._value.flags.writeable = True
        yield self.value
        self._value.flags.writeable = False
        self.version += 1
        self.tidy()

    def tidy(self):
        pass

    def copy(self: S) -> S:
        self.shared = True
        return copy(self)

    def resolve_sharing(self):
        self._value = self._value.copy()
        self.shared = False


class GroupedParameters(ArrayParameter):

    group_versions: NDArray[int]

    def __init__(self, value: NDArray[DType], group_dim: int = 0):
        super().__init__(value=value)
        self.group_dim = group_dim
        self.group_versions = np.zeros(self.n_groups)

    def set_items(self, keys, values):
        super().set_items(keys, values)

        # Update version of the changed group
        if isinstance(keys, int):
            self.group_versions[keys] = self.version
        elif isinstance(keys, tuple):
            self.group_versions[keys[self.group_dim]] = self.version
        else:
            raise RuntimeError('`set_items` is not implemented for GroupedParameters. '
                               'Use `GroupedParameters.edit()` instead.')

    def set_value(self, new_value: NDArray[DType]):
        super().set_value(new_value)

        # self.group_versions[:] = self.version
        self.group_versions = np.full_like(self.group_versions, self.version)

    def set_group(self, i: int, values: NDArray[Value]):
        with self.edit_group(i) as g:
            g[...] = values

    @property
    def n_groups(self):
        return self.shape[self.group_dim]

    @contextmanager
    def edit_group(self, i) -> NDArray[Value]:
        if self.shared:
            self.resolve_sharing()

        self._value.flags.writeable = True
        yield self.get_group(i)
        self._value.flags.writeable = False
        self.version += 1
        self.group_versions[i] = self.version

    @contextmanager
    def edit_groups(self, idxs) -> NDArray[Value]:
        if not self.group_dim == 0:
            raise NotImplementedError

        if self.shared:
            self.resolve_sharing()

        self._value.flags.writeable = True
        yield self.value[idxs]
        self._value.flags.writeable = False
        self.version += 1
        self.group_versions[idxs] = self.version
        self.tidy()

    def set_groups(self, group_idxs: NDArray[int], new_values: NDArray[DType]):
        if not self.group_dim == 0:
            raise NotImplementedError
        if self.shared:
            self.resolve_sharing()
        self._value.flags.writeable = True
        self.value[group_idxs] = new_values
        self._value.flags.writeable = False
        self.version += 1
        self.group_versions[group_idxs] = self.version
        # self.tidy()

    def tidy(self):
        self.group_versions = np.full_like(self.group_versions, self.version)

    # def edit(self) -> NDArray[DType]:
    #     raise NotImplementedError

    def resolve_sharing(self):
        self.group_versions = self.group_versions.copy()
        super().resolve_sharing()

    def get_group(self, i: int):
        return get_along_axis(a=self.value, index=i, axis=self.group_dim)


class Clusters(GroupedParameters):

    # alias for edit_group
    edit_cluster = GroupedParameters.edit_group

    @property
    def sizes(self):
        return np.count_nonzero(self._value, axis=1)

    @property
    def n_clusters(self):
        return self.shape[0]

    @property
    def n_objects(self):
        return self.shape[1]

    def any_cluster(self):
        return np.any(self._value, axis=0)

    def add_object(self, i_cluster, i_object):
        with self.edit_cluster(i_cluster) as c:
            c[i_object] = True

    def remove_object(self, i_cluster, i_object):
        with self.edit_cluster(i_cluster) as c:
            c[i_object] = False

    @contextmanager
    def value_for_cython(self) -> NDArray[DType]:
        self._value.flags.writeable = True
        yield self.value
        self._value.flags.writeable = False


@lru_cache(maxsize=128)
def outdated_group_version(shape: tuple[int]) -> NDArray[int]:
    """To manually mark the cache node as outdated we use a constant -1."""
    return -np.ones(shape)


class CacheNode(Generic[Value]):

    """Wrapper for cached calculated values (array or scalar). Keeps track of whether a
     cache node and its derived values are outdated."""

    _value: Value
    cached_version: VersionType
    inputs: OrderedDict[str, CacheNode | Parameter]
    cached_group_versions: dict[str, NDArray[int]]

    def __init__(self, value: Value):
        self._value = value
        self.inputs = OrderedDict()
        self.input_idx = OrderedDict()
        self.cached_version = self.outdated_version()
        self.cached_group_versions = {}

    def is_outdated(self) -> bool:
        return self.cached_version != self.version

    def ahead_of(self, input_key: str) -> bool:
        i = self.input_idx[input_key]
        return self.cached_version[i] != self.inputs[input_key].version

    def what_changed(self, input_key: str | list[str], caching=True) -> list[int]:
        if isinstance(input_key, list):
            return list(set.union(*(set(self.what_changed(k, caching=caching)) for k in input_key)))

        inpt = self.inputs[input_key]
        if isinstance(inpt, GroupedParameters):
            if caching:
                return list(np.flatnonzero(
                    self.cached_group_versions[input_key] != inpt.group_versions
                ))
            else:
                return list(range(inpt.n_groups))
        else:
            raise ValueError('Can only track what changed for GroupedParameters')

    @property
    def value(self) -> Value:
        return self._value

    def update_value(self, new_value: Value):
        self._value = new_value
        self.set_up_to_date()

    def set_up_to_date(self):
        self.cached_version = self.version
        for key, inpt in self.inputs.items():
            if isinstance(inpt, GroupedParameters):
                self.cached_group_versions[key] = inpt.group_versions.copy()
                self.cached_group_versions[key].flags.writeable = False

    @contextmanager
    def edit(self) -> Iterator[Value]:
        yield self.value
        self.set_up_to_date()

    def outdated_version(self) -> tuple[int]:
        """To manually mark the cache node as outdated we use a constant -1."""
        return (-1,) * len(self.inputs)

    def add_input(self, key: str, inpt: CacheNode | Parameter):
        """Add an input to this cache node."""
        self.input_idx[key] = len(self.inputs)
        self.inputs[key] = inpt
        self.cached_version = self.outdated_version()
        if isinstance(inpt, GroupedParameters):
            self.clear_group_version(key)

    def cached_version_by_input(self, input_key: str) -> tuple:
        """Get the cached version number for a specific input."""
        return self.cached_version[self.input_idx[input_key]]

    @property
    def version(self) -> tuple[VersionType]:
        """Calculate the current version number from the inputs."""
        return tuple(inpt.version for inpt in self.inputs.values())
        # return hash(tuple(i.version for i in self.inputs))

    @property
    def shape(self) -> tuple[int]:
        """Convenience property to directly access the shape of the value array."""
        return self._value.shape

    def clear(self):
        """Mark the cache node as outdated."""
        self.cached_version = self.outdated_version()
        for key in self.cached_group_versions:
            self.clear_group_version(key)

    def clear_group_version(self, key: str):
        """Mark the group versions of a specific input as outdated."""
        shape = self.inputs[key].group_versions.shape
        new_group_version = outdated_group_version(shape)
        self.cached_group_versions[key] = new_group_version
        new_group_version.flags.writeable = False

    def assign_from(self, other: CacheNode):
        """Assign the cache node's value and version nr from another calc node."""
        self._value = copy(other._value)
        self.cached_version = other.cached_version
        self.cached_group_versions = {k: v for k, v in other.cached_group_versions.items()}


class SufficientStatistics(GroupedParameters):
    """Parameters that describe the observations assigned to a component sufficiently for
    exact likelihood calculations."""

    @property
    def n_groups(self):
        return self.value.shape[0]

    @property
    def n_features(self):
        return self.value.shape[1]

    def add_changes(self, old: NDArray, new: NDArray):  # old/new shapes: (n_groups, n_features, ...)
        if self.shared:
            self.resolve_sharing()
        self.version += 1
        changed_groups = np.any(new != old, axis=new.shape[1:])
        self.group_versions[changed_groups] = self.version

    def mark_changes(self, changed_groups: NDArray[bool]):  # shape: (n_groups, n_features)
        if self.shared:
            self.resolve_sharing()
        self.version += 1
        self.group_versions[changed_groups] = self.version

    def make_dirty(self):
        if self.shared:
            self.resolve_sharing()
        self.version += 1
        self.group_versions[:] = self.version

    @classmethod
    def create_empty(cls, n_groups, n_features):
        return SufficientStatistics(np.empty((n_groups, n_features)))


class FeatureCounts(SufficientStatistics):
    """GroupedParameters for the feature counts of each group (or cluster) in a mixture
    component. Value shape: (n_groups, n_features, n_states)"""

    @property
    def n_states(self):
        return self.value.shape[2]

    # def add_changes(self, diff: NDArray[int]):  # diff shape: (n_groups, n_features, n_states)
    def add_changes(self, old: NDArray, new: NDArray):  # old/new shape: (n_groups, n_features, n_states)
        if self.shared:
            self.resolve_sharing()

        diff = new - old

        self._value.flags.writeable = True
        self._value += diff
        self._value.flags.writeable = False
        self.version += 1

        changed_groups = np.any(diff != 0, axis=(1, 2))
        self.group_versions[changed_groups] = self.version


class HasComponents(CacheNode[NDArray[bool]]):

    """
    Array cache node with shape (n_objects, n_components)
    """

    def __init__(self, clusters: Clusters, confounders: dict[str, Confounder]):
        # Set up value from clusters and confounders
        has_components = [clusters.any_cluster()]
        for conf in confounders.values():
            has_components.append(conf.any_group())
        super().__init__(value=np.array(has_components).T)

        self.clusters = clusters
        self.inputs['clusters'] = clusters

    @property
    def value(self) -> Value:
        if not self.is_outdated():
            return self._value
        else:
            self._value[:, 0] = self.inputs['clusters'].any_cluster()
            self.cached_version = self.version
            return self._value


class ModelCache:

    geo_prior: CacheNode[NDArray[float]]
    cluster_size_prior: CacheNode[float]
    categorical: CategoricalCache
    gaussian: GaussianCache
    poisson: PoissonCache
    logitnormal: LogitNormalCache

    def __init__(self, sample: Sample, ):
        # self.likelihood = CacheNode(0.)

        self.geo_prior = CacheNode(value=np.zeros(sample.n_clusters))
        self.cluster_size_prior = CacheNode(value=0.0)
        self.categorical = CategoricalCache(sample)
        self.gaussian = GaussianCache(sample)
        self.poisson = PoissonCache(sample)
        self.logitnormal = LogitNormalCache(sample)

        # Set up the dependencies in form of CacheNode inputs:
        self.cluster_size_prior.add_input('clusters', sample.clusters)
        self.geo_prior.add_input('clusters', sample.clusters)

        self.feature_type_cache = {
            "categorical": self.categorical,
            "gaussian": self.gaussian,
            "poisson": self.poisson,
            "logitnormal": self.logitnormal,
        }

    def clear(self):
        self.geo_prior.clear()
        self.cluster_size_prior.clear()
        self.categorical.clear()
        self.gaussian.clear()
        self.poisson.clear()
        self.logitnormal.clear()

    def copy(self, new_sample: Sample):
        new_cache = ModelCache(new_sample)
        new_cache.geo_prior.assign_from(self.geo_prior)
        new_cache.cluster_size_prior.assign_from(self.cluster_size_prior)
        self.categorical.copy(new_cache.categorical)
        self.gaussian.copy(new_cache.gaussian)
        self.poisson.copy(new_cache.poisson)
        self.logitnormal.copy(new_cache.logitnormal)

        return new_cache


class GenericTypeCache:

    # likelihood: CacheNode[float]
    component_likelihoods: CacheNode[NDArray[float]]
    group_likelihoods: dict[str, CacheNode[NDArray[float]]]
    weights_normalized: CacheNode[NDArray[float]]

    prior: CacheNode[float]
    source_prior: CacheNode[NDArray[float]]
    cluster_effect_prior: CacheNode[float]
    confounding_effects_prior: dict[str, CacheNode[float]]
    weights_prior: CacheNode[float]

    has_components: CacheNode[bool]

    def __init__(self, sample: Sample):
        # self.likelihood = CacheNode(0.)

        self.sample_type = self.get_typed_sample(sample)

        self.component_likelihoods = CacheNode(
            value=np.empty((sample.n_objects, self.sample_type.n_features, sample.n_components))
        )
        self.group_likelihoods = {
            conf: CacheNode(value=np.empty(sample.n_groups(conf)))
            for conf in sample.component_names
        }
        self.weights_normalized = CacheNode(
            value=np.empty((sample.n_objects, self.sample_type.n_features, sample.n_components))
        )
        self.cluster_effect_prior = CacheNode(value=0.0)
        self.confounding_effects_prior = {
            conf: CacheNode(value=np.ones(sample.n_groups(conf)))
            for conf in sample.confounders
        }
        self.weights_prior = CacheNode(value=0.0)
        self.has_components = HasComponents(sample.clusters, sample.confounders)

        # Set up the dependencies in form of CacheNode inputs:
        self.component_likelihoods.add_input('clusters', sample.clusters)
        self.weights_normalized.add_input('has_components', self.has_components)

        # # self.group_likelihoods['cluster'].add_input('cluster_effect', sample.cluster_effect)
        # # self.component_likelihoods.add_input('cluster_effect', sample.cluster_effect)
        # for conf, effect in self.sample_type.confounding_effects.items():
        #     self.group_likelihoods[conf].add_input(f'c_{conf}', effect)
        #     self.component_likelihoods.add_input(f'c_{conf}', effect)

        self.weights_prior.add_input('weights', self.sample_type.weights)
        self.weights_normalized.add_input('weights', self.sample_type.weights)

        self.source_prior = CacheNode(value=np.zeros(sample.n_objects))
        self.source_prior.add_input('weights_normalized', self.weights_normalized)
        self.source_prior.add_input('source', self.sample_type.source)
        self.source_prior.add_input('clusters', sample.clusters)
        self.source_prior.add_input('weights', self.sample_type.weights)
        self.component_likelihoods.add_input('source', self.sample_type.source)

        for comp, counts in self.sample_type.sufficient_statistics.items():
            self.group_likelihoods[comp].add_input('sufficient_stats', counts)
            self.component_likelihoods.add_input(f'{comp}_sufficient_stats', counts)

    @property
    def cluster_likelihoods(self) -> NDArray[float]:
        return self.component_likelihoods.value[0]

    @property
    def confounder_likelihoods(self) -> NDArray[float]:
        return self.component_likelihoods.value[1:]

    @staticmethod
    def get_typed_sample(sample: Sample):
        """Return the data-type specific part of the sample object"""
        raise NotImplementedError

    def clear(self):
        self.component_likelihoods.clear()
        self.weights_normalized.clear()
        self.source_prior.clear()
        self.cluster_effect_prior.clear()
        self.weights_prior.clear()
        self.has_components.clear()
        for conf_eff in self.confounding_effects_prior.values():
            conf_eff.clear()
        for group_lh in self.group_likelihoods.values():
            group_lh.clear()

    def copy(self, new_cache: GenericTypeCache):
        new_cache.component_likelihoods.assign_from(self.component_likelihoods)
        new_cache.weights_normalized.assign_from(self.weights_normalized)
        new_cache.source_prior.assign_from(self.source_prior)
        new_cache.cluster_effect_prior.assign_from(self.cluster_effect_prior)
        new_cache.weights_prior.assign_from(self.weights_prior)
        new_cache.has_components.assign_from(self.has_components)
        for conf, conf_eff_prior in new_cache.confounding_effects_prior.items():
            conf_eff_prior.assign_from(self.confounding_effects_prior[conf])
        for comp, group_lh in new_cache.group_likelihoods.items():
            group_lh.assign_from(self.group_likelihoods[comp])
        return new_cache


class CategoricalCache(GenericTypeCache):

    def __init__(self, sample: Sample):
        super().__init__(sample)
        categorical_sample = self.get_typed_sample(sample)

    @staticmethod
    def get_typed_sample(sample: Sample):
        """Return the data-type specific part of the sample object"""
        return sample.categorical


class GaussianCache(GenericTypeCache):

    @staticmethod
    def get_typed_sample(sample: Sample):
        """Return the data-type specific part of the sample object"""
        return sample.gaussian


class PoissonCache(GenericTypeCache):

    @staticmethod
    def get_typed_sample(sample: Sample):
        """Return the data-type specific part of the sample object"""
        return sample.poisson


class LogitNormalCache(GenericTypeCache):

    @staticmethod
    def get_typed_sample(sample: Sample):
        """Return the data-type specific part of the sample object"""
        return sample.logitnormal


class Source:

    def __init__(
        self,
        categorical: NDArray[bool] | None,
        gaussian: NDArray[bool] | None,
        poisson: NDArray[bool] | None,
        logitnormal: NDArray[bool] | None,
    ):
        self.categorical = categorical
        self.gaussian = gaussian
        self.poisson = poisson
        self.logitnormal = logitnormal

    def __getitem__(self, key: str) -> NDArray[bool]:
        if key in FeatureType.values():
            return getattr(self, key)
        else:
            raise KeyError(f"Unknown feature type `{key}`.")


class Sample:

    confounders: dict[str, Confounder]

    def __init__(
        self,
        clusters: Clusters,                                 # shape: (n_clusters, n_objects)
        confounders: dict[str, Confounder],
        categorical: CategoricalSample,                     # The categorical parameters
        gaussian: GaussianSample,                           # The Gaussian parameters
        poisson: PoissonSample,                             # The Poisson parameters
        logitnormal: LogitNormalSample,                     # The logit-normal parameters
        model_shapes: ModelShapes,
        chain: int = 0,
        _other_cache: ModelCache = None,
        _i_step: int = 0
    ):
        self.categorical = categorical
        self.gaussian = gaussian
        self.poisson = poisson
        self.logitnormal = logitnormal
        self.model_shapes = model_shapes
        self._clusters = clusters
        self.chain = chain
        self.confounders = confounders
        self.i_step = _i_step

        # Assign or initialize a ModelCache object
        if _other_cache is None:
            self.cache = ModelCache(sample=self)
        else:
            self.cache = _other_cache.copy(new_sample=self)

        # Store last likelihood and prior for logging
        self.last_lh = None
        self.last_prior = None

        # Store the likelihood of each observation (language and feature) for logging
        self.observation_lhs = None

        # Caching:
        self._groups_and_clusters = None

        self.feature_type_samples = {
            "categorical": self.categorical,
            "gaussian": self.gaussian,
            "poisson": self.poisson,
            "logitnormal": self.logitnormal,
        }

    @classmethod
    def from_numpy_arrays(
        cls: Type[S],
        clusters: NDArray[bool],
        weights: dict[str, NDArray[float]],
        # confounding_effects: dict[str, dict[str, NDArray[float]]],
        confounders: dict[str, Confounder],
        source: Source,
        feature_counts: dict[str, NDArray[int]],
        model_shapes: ModelShapes,
        chain: int = 0
    ) -> S:
        n_clusters = len(clusters)

        sample_categorical = None
        if weights.get('categorical') is not None:
            sample_categorical = CategoricalSample(
                weights=ArrayParameter(weights['categorical']),
                # confounding_effects={k: GroupedParameters(v['categorical']) for k, v in confounding_effects.items()},
                source=GroupedParameters(source['categorical']),
                sufficient_statistics={k: FeatureCounts(v) for k, v in feature_counts.items()},
                model_shapes=model_shapes,
            )

        sample_gaussian = None
        if weights.get('gaussian') is not None:
            n_features = source['gaussian'].shape[1]
            suff_stats = {'clusters': SufficientStatistics.create_empty(n_clusters, n_features)}
            for c, conf in confounders.items():
                suff_stats[c] = SufficientStatistics.create_empty(conf.n_groups, n_features)
            sample_gaussian = GaussianSample(
                weights=ArrayParameter(weights['gaussian']),
                # confounding_effects={k: GroupedParameters(v['gaussian']) for k, v in confounding_effects.items()},
                source=GroupedParameters(source['gaussian']),
                sufficient_statistics=suff_stats,
                model_shapes=model_shapes,
            )

        sample_poisson = None
        if weights.get('poisson') is not None:
            n_features = source['poisson'].shape[1]
            suff_stats = {'clusters': SufficientStatistics.create_empty(n_clusters, n_features)}
            for c, conf in confounders.items():
                suff_stats[c] = SufficientStatistics.create_empty(conf.n_groups, n_features)
            sample_poisson = PoissonSample(
                weights=ArrayParameter(weights['poisson']),
                # confounding_effects={k: GroupedParameters(v['poisson']) for k, v in confounding_effects.items()},
                source=GroupedParameters(source['poisson']),
                sufficient_statistics=suff_stats,
                model_shapes=model_shapes,
            )

        sample_logitnormal = None
        if weights.get('logitnormal') is not None:
            n_features = source['logitnormal'].shape[1]
            suff_stats = {'clusters': SufficientStatistics.create_empty(n_clusters, n_features)}
            for c, conf in confounders.items():
                suff_stats[c] = SufficientStatistics.create_empty(conf.n_groups, n_features)
            sample_logitnormal = LogitNormalSample(
                weights=ArrayParameter(weights['logitnormal']),
                # confounding_effects={k: GroupedParameters(v['logitnormal']) for k, v in confounding_effects.items()},
                source=GroupedParameters(source['logitnormal']),
                sufficient_statistics=suff_stats,
                model_shapes=model_shapes,
            )

        return cls(
            clusters=Clusters(clusters),
            categorical=sample_categorical,
            gaussian=sample_gaussian,
            poisson=sample_poisson,
            logitnormal=sample_logitnormal,
            confounders=confounders,
            chain=chain,
            model_shapes=model_shapes,
        )

    def copy(self: S) -> S:
        return Sample(
            chain=self.chain,
            clusters=self._clusters.copy(),
            categorical=self.categorical.copy(),
            gaussian=self.gaussian.copy(),
            poisson=self.poisson.copy(),
            logitnormal=self.logitnormal.copy(),
            confounders=self.confounders,
            model_shapes=self.model_shapes,
            _other_cache=self.cache,
            _i_step=self.i_step
        )

    def everything_changed(self):
        self.cache.clear()

    """ properties to make parameters read-only """

    @property
    def clusters(self) -> Clusters:
        return self._clusters

    """ shape properties """
    @property
    def n_clusters(self) -> int:
        return self._clusters.shape[0]

    @property
    def n_objects(self) -> int:
        return self._clusters.shape[1]

    @property
    def n_components(self) -> int:
        return len(self.confounders) + 1

    @property
    def component_names(self) -> list[str]:
        return ['clusters', *self.confounders.keys()]

    def n_groups(self, conf: str) -> int:
        if conf == 'clusters':
            return self.n_clusters
        else:
            return len(self.confounders[conf].group_names)

    def groups_and_clusters(self) -> dict[str, NDArray[bool]]:  # shape: (n_groups,) for each component
        # Confounder groups are constant, so only need to collect them once
        if self._groups_and_clusters is None:
            self._groups_and_clusters = {name: conf.group_assignment for name, conf in self.confounders.items()}

        # Update clusters in case they changed:
        self._groups_and_clusters['clusters'] = self.clusters.value

        return self._groups_and_clusters


class GenericTypeSample:

    def __init__(
        self,
        weights: ArrayParameter[float],                     # shape: (n_features, n_components)
        # confounding_effects: dict[str, GroupedParameters],  # shape per conf:  (n_groups, n_features, n_states)
        source: GroupedParameters[bool],                    # shape: (n_objects, n_features, n_components)
        sufficient_statistics: dict[str, SufficientStatistics],  # shape per conf: (n_groups, n_features)
        model_shapes: ModelShapes,
    ):
        self._weights = weights
        # self._confounding_effects = confounding_effects
        self._source = source
        self._sufficient_statistics = sufficient_statistics
        self.model_shapes = model_shapes

    def copy(self: S) -> S:
        return type(self)(
            weights=self.weights.copy(),
            # confounding_effects={k: v.copy() for k, v in self.confounding_effects.items()},
            source=self.source.copy(),
            sufficient_statistics={k: v.copy() for k, v in self.sufficient_statistics.items()},
            model_shapes=self.model_shapes,
        )

    """properties to make parameters read-only"""
    @property
    def weights(self) -> ArrayParameter:
        return self._weights

    # @property
    # def confounding_effects(self) -> dict[str, GroupedParameters]:
    #     return self._confounding_effects

    @property
    def source(self) -> GroupedParameters[bool]:  # shape: (n_objects, n_features, n_components)
        return self._source

    """shape properties """
    @property
    def n_features(self) -> int:
        return self._weights.shape[0]

    @property
    def sufficient_statistics(self) -> dict[str, SufficientStatistics]:
        return self._sufficient_statistics


class CategoricalSample(GenericTypeSample):

    def __init__(
        self,
        weights: ArrayParameter[float],                     # shape: (n_features, n_components)
        # confounding_effects: dict[str, GroupedParameters],  # shape per conf:  (n_groups, n_features, n_states)
        source: GroupedParameters[bool],                    # shape: (n_objects, n_features, n_components)
        sufficient_statistics: dict[str, FeatureCounts],           # shape per conf: (n_groups, n_features)
        model_shapes: ModelShapes,
    ):
        super().__init__(weights, source, sufficient_statistics, model_shapes)

    # def copy(self: S) -> S:
    #     return CategoricalSample(
    #         weights=self.weights.copy(),
    #         confounding_effects={k: v.copy() for k, v in self.confounding_effects.items()},
    #         source=self.source.copy(),
    #         sufficient_statistics={k: v.copy() for k, v in self.sufficient_statistics.items()},
    #     )

    @property
    def n_states(self) -> int:
        return next(iter(self.model_shapes.values())).shape[2]


class GaussianSample(GenericTypeSample):
    pass


class PoissonSample(GenericTypeSample):
    pass


class LogitNormalSample(GenericTypeSample):
    pass


if __name__ == '__main__':
    import doctest
    doctest.testmod()
