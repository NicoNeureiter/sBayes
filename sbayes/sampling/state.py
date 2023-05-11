from __future__ import annotations
from collections import OrderedDict
from copy import copy, deepcopy
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional, Generic, TypeVar, Type, Iterator

from numpy.typing import NDArray
import numpy as np

from sbayes.load_data import Confounder, Features
from sbayes.util import get_along_axis

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

    def resolve_sharing(self):
        self.group_versions = self.group_versions.copy()
        super().resolve_sharing()

    def get_group(self, i: int):
        return get_along_axis(a=self.value, index=i, axis=self.group_dim)


class Clusters(GroupedParameters):

    # alias for edit_group
    edit_cluster = GroupedParameters.edit_group

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
    """To manually mark the calculation node as outdated we use a constant -1."""
    return -np.ones(shape)


class CalculationNode(Generic[Value]):

    """Wrapper for cached calculated values (array or scalar). Keeps track of whether a
     calculation node and its derived values are outdated."""

    _value: Value
    cached_version: VersionType
    inputs: OrderedDict[str, CalculationNode | Parameter]
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

    def what_changed(self, input_key: str | list[str], caching=True) -> set[int]:
        if isinstance(input_key, list):
            return set.union(*(self.what_changed(k, caching=caching) for k in input_key))

        inpt = self.inputs[input_key]
        if isinstance(inpt, GroupedParameters):
            if caching:
                return set(np.nonzero(
                    self.cached_group_versions[input_key] != inpt.group_versions
                )[0])
            else:
                return set(range(inpt.n_groups))
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
        """To manually mark the calculation node as outdated we use a constant -1."""
        return (-1,) * len(self.inputs)

    def add_input(self, key: str, inpt: CalculationNode | Parameter):
        """Add an input to this calculation node."""
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
        """Mark the calculation node as outdated."""
        self.cached_version = self.outdated_version()
        for key in self.cached_group_versions:
            self.clear_group_version(key)

    def clear_group_version(self, key: str):
        """Mark the group versions of a specific input as outdated."""
        shape = self.inputs[key].group_versions.shape
        new_group_version = outdated_group_version(shape)
        self.cached_group_versions[key] = new_group_version
        new_group_version.flags.writeable = False

    def assign_from(self, other: CalculationNode):
        """Assign the calculation node's value and version nr from another calc node."""
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


class HasComponents(CalculationNode[NDArray[bool]]):

    """
    Array calculation node with shape (n_objects, n_components)
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

    # likelihood: CalculationNode[float]
    component_likelihoods: CalculationNode[NDArray[float]]
    group_likelihoods: dict[str, CalculationNode[NDArray[float]]]
    weights_normalized: CalculationNode[NDArray[float]]

    prior: CalculationNode[float]
    source_prior: CalculationNode[float]
    geo_prior: CalculationNode[float]
    cluster_size_prior: CalculationNode[float]
    cluster_effect_prior: CalculationNode[float]
    confounding_effects_prior: dict[str, CalculationNode[float]]
    weights_prior: CalculationNode[float]

    has_components: CalculationNode[bool]

    def __init__(self, sample: Sample, ):
        # self.likelihood = CalculationNode(0.)
        self.component_likelihoods = CalculationNode(
            value=np.empty((sample.n_objects, sample.n_features, sample.n_components))
        )
        self.group_likelihoods = {
            conf: CalculationNode(value=np.empty(sample.n_groups(conf)))
            for conf in sample.component_names
        }
        self.weights_normalized = CalculationNode(
            value=np.empty((sample.n_objects, sample.n_features, sample.n_components))
        )
        self.geo_prior = CalculationNode(value=0.0)
        self.cluster_size_prior = CalculationNode(value=0.0)
        self.cluster_effect_prior = CalculationNode(value=0.0)
        self.confounding_effects_prior = {
            conf: CalculationNode(value=np.ones(sample.n_groups(conf)))
            for conf in sample.confounders
        }
        self.weights_prior = CalculationNode(value=0.0)
        self.has_components = HasComponents(sample.clusters, sample.confounders)

        # Set up the dependencies in form of CalculationNode inputs:

        self.component_likelihoods.add_input('clusters', sample.clusters)
        self.cluster_size_prior.add_input('clusters', sample.clusters)
        self.geo_prior.add_input('clusters', sample.clusters)
        self.weights_normalized.add_input('has_components', self.has_components)

        # self.group_likelihoods['cluster'].add_input('cluster_effect', sample.cluster_effect)
        # self.component_likelihoods.add_input('cluster_effect', sample.cluster_effect)
        for conf, effect in sample.confounding_effects.items():
            self.group_likelihoods[conf].add_input(f'c_{conf}', effect)
            self.component_likelihoods.add_input(f'c_{conf}', effect)

        self.weights_prior.add_input('weights', sample.weights)
        self.weights_normalized.add_input('weights', sample.weights)

        # Differences between Gibbs/Non-Gibbs models:
        self.source_prior = CalculationNode(value=0.0)
        self.source_prior.add_input('weights_normalized', self.weights_normalized)
        self.source_prior.add_input('source', sample.source)
        self.component_likelihoods.add_input('source', sample.source)

        for comp, counts in sample.feature_counts.items():
            self.group_likelihoods[comp].add_input('counts', counts)
            self.component_likelihoods.add_input(f'{comp}_counts', counts)
            # self.likelihood.add_input(f'counts_{comp}', self.group_likelihoods[comp])

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
        self.cluster_size_prior.clear()
        self.cluster_effect_prior.clear()
        self.weights_prior.clear()
        self.has_components.clear()
        for conf_eff in self.confounding_effects_prior.values():
            conf_eff.clear()
        for group_lh in self.group_likelihoods.values():
            group_lh.clear()

    def copy(self: S, new_sample: Sample) -> S:
        new_cache = ModelCache(new_sample)
        new_cache.component_likelihoods.assign_from(self.component_likelihoods)
        new_cache.weights_normalized.assign_from(self.weights_normalized)
        new_cache.geo_prior.assign_from(self.geo_prior)
        new_cache.cluster_size_prior.assign_from(self.cluster_size_prior)
        new_cache.cluster_effect_prior.assign_from(self.cluster_effect_prior)
        new_cache.weights_prior.assign_from(self.weights_prior)
        for conf, conf_eff_prior in new_cache.confounding_effects_prior.items():
            conf_eff_prior.assign_from(self.confounding_effects_prior[conf])
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
        confounding_effects: dict[str, GroupedParameters],  # shape per conf:  (n_groups, n_features, n_states)
        confounders: dict[str, Confounder],
        source: GroupedParameters[bool],                       # shape: (n_objects, n_features, n_components)
        feature_counts: dict[str, FeatureCounts],           # shape per conf: (n_groups, n_features)
        chain: int = 0,
        _other_cache: ModelCache = None,
        _i_step: int = 0
    ):
        self._clusters = clusters
        self._weights = weights
        self._confounding_effects = confounding_effects
        self._source = source
        self._feature_counts = feature_counts
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

    @classmethod
    def from_numpy_arrays(
        cls: Type[S],
        clusters: NDArray[bool],
        weights: NDArray[float],
        confounding_effects: dict[str, NDArray[float]],
        confounders: dict[str, Confounder],
        source: NDArray[bool],
        feature_counts: dict[str, NDArray[int]],
        chain: int = 0,
    ) -> S:
        return cls(
            clusters=Clusters(clusters),
            weights=ArrayParameter(weights),
            confounding_effects={k: GroupedParameters(v) for k, v in confounding_effects.items()},
            confounders=confounders,
            source=GroupedParameters(source, group_dim=2),
            feature_counts={k: FeatureCounts(v) for k, v in feature_counts.items()},
            chain=chain,
        )

    def copy(self: S) -> S:
        return Sample(
            chain=self.chain,
            #
            # clusters=deepcopy(self._clusters),
            # weights=deepcopy(self.weights),
            # confounding_effects=deepcopy(self.confounding_effects),
            # source=deepcopy(self.source),
            #
            clusters=self._clusters.copy(),
            weights=self.weights.copy(),
            confounding_effects={k: v.copy() for k, v in self.confounding_effects.items()},
            source=self.source.copy(),
            feature_counts={k: v.copy() for k, v in self.feature_counts.items()},
            #
            confounders=self.confounders,
            _other_cache=self.cache,
            _i_step=self.i_step,
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
    def confounding_effects(self) -> dict[str, GroupedParameters]:
        return self._confounding_effects

    @property
    def source(self) -> ArrayParameter[bool]:  # shape: (n_objects, n_features, n_components)
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
        return next(iter(self._confounding_effects.values())).shape[2]

    @property
    def component_names(self) -> list[str]:
        return ['clusters', *self.confounders.keys()]

    def n_groups(self, conf: str) -> int:
        if conf == 'clusters':
            return self.n_clusters
        else:
            return self._confounding_effects[conf].shape[0]

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
