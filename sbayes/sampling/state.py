from __future__ import annotations
from collections import OrderedDict
from copy import copy, deepcopy
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional, Generic, TypeVar, Type, Iterator

from numpy.typing import NDArray
import numpy as np

from sbayes.load_data import Confounder
from sbayes.model.model_shapes import ModelShapes
from sbayes.util import get_along_axis, FLOAT_TYPE

S = TypeVar('S')
Value = TypeVar('Value', NDArray, float)
DType = TypeVar('DType', bool, float, int)
VersionType = TypeVar('VersionType', tuple, int)


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

    def __init__(self, value: NDArray[Value], group_dim: int = 0):
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
    def sizes(self) -> NDArray[int]:  # shape: (n_clusters,)
        return np.count_nonzero(self._value, axis=1)

    @property
    def n_clusters(self) -> int:
        return self.shape[0]

    @property
    def n_objects(self) -> int:
        return self.shape[1]

    def any_cluster(self) -> NDArray[bool]:
        return np.any(self._value, axis=0)

    def add_object(self, i_cluster: int, i_object: int):
        with self.edit_cluster(i_cluster) as c:
            c[i_object] = True

    def remove_object(self, i_cluster: int, i_object: int):
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

    def what_changed(self, input_key: str | list[str], caching=True) -> NDArray[int]:
        if isinstance(input_key, list):
            changes_by_input = [self.what_changed(k, caching=caching) for k in input_key]
            return np.unique(np.concatenate(changes_by_input))
            # changes_by_input = (set(self.what_changed(k, caching=caching)) for k in input_key)
            # return List(set.union(*changes_by_input))

        inpt = self.inputs[input_key]
        if isinstance(inpt, GroupedParameters):
            if caching:
                return np.flatnonzero(
                    self.cached_group_versions[input_key] != inpt.group_versions
                )
            else:
                return np.arange(inpt.n_groups)
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


class FeatureCounts(GroupedParameters):
    """GroupedParameters for the feature counts of each group (or cluster) in a mixture
    component. Value shape: (n_groups, n_features, n_states)"""

    @property
    def n_groups(self):
        return self.value.shape[0]

    @property
    def n_features(self):
        return self.value.shape[1]

    @property
    def n_states(self):
        return self.value.shape[2]

    def add_changes(self, diff: NDArray[int]):  # diff shape: (n_groups, n_features, n_states)
        if self.shared:
            self.resolve_sharing()

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
        super().__init__(value=np.array(has_components, dtype=bool).T)

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

    # likelihood: CacheNode[float]
    component_likelihoods: CacheNode[NDArray[float]]
    group_likelihoods: dict[str, CacheNode[NDArray[float]]]
    weights_normalized: CacheNode[NDArray[float]]

    prior: CacheNode[float]
    source_prior: CacheNode[NDArray[float]]
    geo_prior: CacheNode[NDArray[float]]
    cluster_size_prior: CacheNode[float]
    # cluster_effect_prior: CacheNode[float]
    # confounding_effects_prior: dict[str, CacheNode[float]]
    weights_prior: CacheNode[float]

    has_components: CacheNode[bool]

    def __init__(self, sample: Sample, ):
        # self.likelihood = CacheNode(0.)
        self.component_likelihoods = CacheNode(
            value=np.empty((sample.n_objects, sample.n_features, sample.n_components))
        )
        self.group_likelihoods = {
            conf: CacheNode(value=np.empty(sample.n_groups(conf)))
            for conf in sample.component_names
        }
        self.weights_normalized = CacheNode(
            value=np.empty((sample.n_objects, sample.n_features, sample.n_components))
        )
        self.geo_prior = CacheNode(value=np.zeros(sample.n_clusters))
        self.cluster_size_prior = CacheNode(value=0.0)
        # self.cluster_effect_prior = CacheNode(value=0.0)
        # self.confounding_effects_prior = {
        #     conf: CacheNode(value=np.ones(sample.n_groups(conf)))
        #     for conf in sample.confounders
        # }
        self.weights_prior = CacheNode(value=0.0)
        self.has_components = HasComponents(sample.clusters, sample.confounders)

        # Set up the dependencies in form of CacheNode inputs:

        self.component_likelihoods.add_input('clusters', sample.clusters)
        self.cluster_size_prior.add_input('clusters', sample.clusters)
        self.geo_prior.add_input('clusters', sample.clusters)
        self.weights_normalized.add_input('has_components', self.has_components)

        # # self.group_likelihoods['cluster'].add_input('cluster_effect', sample.cluster_effect)
        # # self.component_likelihoods.add_input('cluster_effect', sample.cluster_effect)
        # for conf, effect in sample.confounding_effects.items():
        #     self.group_likelihoods[conf].add_input(f'c_{conf}', effect)
        #     self.component_likelihoods.add_input(f'c_{conf}', effect)
        #     if sample.confounders[conf].has_universal_prior:
        #         self.group_likelihoods[conf].add_input(f'c_universal', sample.confounding_effects['universal'])

        self.weights_prior.add_input('weights', sample.weights)
        self.weights_normalized.add_input('weights', sample.weights)

        self.source_prior = CacheNode(value=np.zeros(sample.n_objects))
        self.source_prior.add_input('weights_normalized', self.weights_normalized)
        self.source_prior.add_input('source', sample.source)
        self.source_prior.add_input('clusters', sample.clusters)
        self.source_prior.add_input('weights', sample.weights)
        self.component_likelihoods.add_input('source', sample.source)

        for comp, counts in sample.feature_counts.items():
            self.group_likelihoods[comp].add_input('counts', counts)
            self.component_likelihoods.add_input(f'{comp}_counts', counts)
            # self.likelihood.add_input(f'counts_{comp}', self.group_likelihoods[comp])

            if comp != 'clusters' and sample.confounders[comp].has_universal_prior:
                self.group_likelihoods[comp].add_input(f'universal_counts', sample.feature_counts['universal'])

    @property
    def cluster_likelihoods(self) -> NDArray[float]:
        return self.component_likelihoods.value[0]

    @property
    def confounder_likelihoods(self) -> NDArray[float]:
        return self.component_likelihoods.value[1:]

    def clear(self):
        self.component_likelihoods.clear()
        self.weights_normalized.clear()
        self.geo_prior.clear()
        self.source_prior.clear()
        self.cluster_size_prior.clear()
        # self.cluster_effect_prior.clear()
        self.weights_prior.clear()
        self.has_components.clear()
        # for conf_eff in self.confounding_effects_prior.values():
        #     conf_eff.clear()
        for group_lh in self.group_likelihoods.values():
            group_lh.clear()

    def copy(self: S, new_sample: Sample) -> S:
        new_cache = ModelCache(new_sample)
        new_cache.component_likelihoods.assign_from(self.component_likelihoods)
        new_cache.weights_normalized.assign_from(self.weights_normalized)
        new_cache.geo_prior.assign_from(self.geo_prior)
        new_cache.source_prior.assign_from(self.source_prior)
        new_cache.cluster_size_prior.assign_from(self.cluster_size_prior)
        # new_cache.cluster_effect_prior.assign_from(self.cluster_effect_prior)
        new_cache.weights_prior.assign_from(self.weights_prior)
        new_cache.has_components.assign_from(self.has_components)
        # for conf, conf_eff_prior in new_cache.confounding_effects_prior.items():
        #     conf_eff_prior.assign_from(self.confounding_effects_prior[conf])
        for comp, group_lh in new_cache.group_likelihoods.items():
            group_lh.assign_from(self.group_likelihoods[comp])

        # new_cache.has_components
        return new_cache


class Sample:

    confounders: dict[str, Confounder]

    def __init__(
        self,
        clusters: Clusters,                                 # shape: (n_clusters, n_objects)
        weights: ArrayParameter[float],                     # shape: (n_features, n_components)
        confounders: dict[str, Confounder],
        source: GroupedParameters[bool],                       # shape: (n_objects, n_features, n_components)
        feature_counts: dict[str, FeatureCounts],           # shape per conf: (n_groups, n_features)
        model_shapes: ModelShapes,
        chain: int = 0,
        _other_cache: ModelCache = None,
        _i_step: int = 0,
    ):
        self._clusters = clusters
        self._weights = weights
        self._source = source
        self._feature_counts = feature_counts
        self.chain = chain
        self.confounders = confounders
        self.i_step = _i_step
        self.model_shapes = model_shapes

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

    @classmethod
    def from_numpy_arrays(
        cls: Type[S],
        clusters: NDArray[bool],
        weights: NDArray[float],
        confounders: dict[str, Confounder],
        source: NDArray[bool],
        feature_counts: dict[str, NDArray[int]],
        model_shapes: ModelShapes,
        chain: int = 0,
    ) -> S:
        return cls(
            clusters=Clusters(clusters),
            weights=ArrayParameter(weights.astype(FLOAT_TYPE)),
            confounders=confounders,
            source=GroupedParameters(source, group_dim=0),
            feature_counts={k: FeatureCounts(v) for k, v in feature_counts.items()},
            chain=chain,
            model_shapes=model_shapes,
        )

    def copy(self: S) -> S:
        return Sample(
            chain=self.chain,
            #
            # clusters=deepcopy(self._clusters),
            # weights=deepcopy(self.weights),
            # source=deepcopy(self.source),
            #
            clusters=self._clusters.copy(),
            weights=self.weights.copy(),
            source=self.source.copy(),
            feature_counts={k: v.copy() for k, v in self.feature_counts.items()},
            #
            confounders=self.confounders,
            _other_cache=self.cache,
            _i_step=self.i_step,
            model_shapes=self.model_shapes,
        )

    def everything_changed(self):
        self.cache.clear()

    """ properties to make parameters read-only """

    @property
    def clusters(self) -> Clusters:
        return self._clusters

    @property
    def weights(self) -> ArrayParameter:
        return self._weights

    @property
    def source(self) -> GroupedParameters[bool]:  # shape: (n_objects, n_features, n_components)
        return self._source

    @property
    def feature_counts(self) -> dict[str, FeatureCounts]:
        return self._feature_counts

    """ shape properties """

    @property
    def n_clusters(self) -> int:
        return self._clusters.shape[0]

    @property
    def n_objects(self) -> int:
        return self._clusters.shape[1]

    @property
    def n_features(self) -> int:
        return self._weights.shape[0]

    @property
    def n_components(self) -> int:
        return self._weights.shape[1]

    @property
    def n_states(self) -> int:
        return self.model_shapes.n_states

    @property
    def component_names(self) -> list[str]:
        return ['clusters', *self.confounders.keys()]

    def n_groups(self, conf: str) -> int:
        if conf == 'clusters':
            return self.n_clusters
        else:
            return self.model_shapes.n_groups[conf]

    def groups_and_clusters(self) -> dict[str, NDArray[bool]]:  # shape: (n_groups,) for each component
        # Confounder groups are constant, so only need to collect them once
        if self._groups_and_clusters is None:
            self._groups_and_clusters = {name: conf.group_assignment for name, conf in self.confounders.items()}

        # Update clusters in case they changed:
        self._groups_and_clusters['clusters'] = self.clusters.value

        return self._groups_and_clusters


if __name__ == '__main__':
    import doctest
    doctest.testmod()
