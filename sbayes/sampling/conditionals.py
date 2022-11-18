from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sbayes.load_data import Features, Confounder
from sbayes.model import Model, update_weights
from sbayes.model.likelihood import compute_component_likelihood
from sbayes.preprocessing import sample_categorical
from sbayes.sampling.state import Sample
from sbayes.util import normalize, timeit


def sample_dirichlet(alpha: NDArray[float]) -> NDArray[float]:
    ...


def sample_cluster_effect(
        features: Features | None,          # shape: (n_objects, n_features, n_states)
        source: NDArray[bool],              # shape: (n_objects, n_features, n_components)
        cluster: NDArray[bool],             # shape: (n_objects, )
        prior_counts: NDArray[float],       # shape: (n_features, n_states)
) -> NDArray[float]:
    _, n_features, n_states = features.values.shape

    if features is None:
        # To sample from prior we emulate an empty dataset
        counts = np.zeros((n_features, n_states))
    else:
        # Only consider observations that are attributed to the areal effect distribution
        from_cluster = source[..., 0] & cluster
        f = (from_cluster[..., np.newaxis] * features.values)
        counts = np.nansum(f, axis=0)

    # Resample cluster_effect according to these observations
    cluster_effect = np.zeros(...)
    for i_feat in range(n_features):
        s_idxs = features.states[i_feat]
        cluster_effect[i_feat, s_idxs] = np.random.dirichlet(
            alpha=1 + counts[i_feat, s_idxs]
        )

    return cluster_effect


def sample_group_effect(
        features: NDArray[bool],                # shape: (n_objects, n_features, n_states)
        from_group: NDArray[bool],              # shape: (n_objects, n_features)
        prior_counts: list[NDArray[float]],     # shape: (n_features, n_states_f)
) -> NDArray[float]:
    _, n_features, n_states = features.values.shape

    # Only consider observations that are attributed to the newly sampled group effect
    features_from_group = (from_group[..., np.newaxis] * features.values)

    # Sum up observations over objects to obtain feature state counts
    counts = np.nansum(features_from_group, axis=0)

    # Resample cluster_effect according to these observations
    cluster_effect = np.zeros((n_features, n_states))
    for i_f, states_f in enumerate(features.states):
        cluster_effect[i_f, states_f] = np.random.dirichlet(
            alpha=prior_counts[i_f] + counts[i_f, states_f]
        )

    return cluster_effect


# @timeit('ms')
def conditional_effect_concentration(
    features: NDArray[bool],                     # shape: (n_objects, n_features, n_states)
    is_source_group: NDArray[bool],              # shape: (n_groups, n_objects, n_features)
    applicable_states: NDArray[bool],            # shape: (n_features, n_states)
    prior_counts: NDArray[float] | float = 0.1,  # shape: (n_groups, n_features, n_states)
) -> NDArray[float]:                             # shape: (n_groups, n_features, n_states)
    # Only use features that are assigned to the given mixture component and group
    features_from_group = (is_source_group[..., np.newaxis] * features[np.newaxis, ...])
    # shape: (n_groups, n_objects, n_features, n_states)

    # Count occurrences of feature states in each group
    counts = np.sum(features_from_group, axis=1).astype(float)
    # shape: (n_groups, n_features, n_states)

    # _counts = np.einsum('hij,ijk->hjk', is_source_group.astype(float), features, optimize=True)
    # assert np.allclose(counts, _counts)

    # Add the prior counts to
    # if isinstance(prior_counts, np.ndarray):
        # If the prior counts are given by an array, simply add it to the observed counts
    counts += prior_counts
    # elif isinstance(prior_counts, list):
    #     # If the prior counts are given by a list of arrays, add counts for each feature in the list to the observed counts
    #     for i_f, states_f in enumerate(applicable_states):
    #         counts[:, i_f, states_f] += prior_counts[i_f].astype(float)
    # else:
    #     # If the prior is given by a single count, broadcast it to all applicable states
    #     for i_f, states_f in enumerate(applicable_states):
    #         counts[:, i_f, states_f] += prior_counts

    return counts


def get_features_from_group(
    i_object: int,
    features: NDArray[bool],                # shape: (n_objects, n_features, n_states)
    group_assignment: NDArray[bool],        # shape: (n_groups, n_objects)
    source_is_component: NDArray[bool],     # shape: (n_objects, n_features)
) -> NDArray[bool]:  # shape: (n_groups, n_features, n_states)
    features_from_group = (group_assignment[:, i_object, :, np.newaxis]         # (n_groups, n_features, 1)
                           * source_is_component[[i_object], :, np.newaxis]     # (1, n_features, 1)
                           * features[[i_object],...])                          # (1, n_features, n_states)
    return features_from_group


# @timeit('ms')
def compute_effect_counts(
    features: NDArray[bool],                # shape: (n_objects, n_features, n_states)
    group_assignment: NDArray[bool],        # shape: (n_groups, n_objects)
    source_is_component: NDArray[bool],     # shape: (n_objects, n_features)
    object_subset: slice | list[int] | NDArray[int] = slice(None),
        z=False,
) -> NDArray[int]:  # shape: (n_groups, n_features, n_states)
    group_assignment = group_assignment[:, object_subset]
    n_groups, n_objects = group_assignment.shape
    _, n_features, n_states = features.shape
    counts = np.zeros((n_groups, n_features, n_states))

    group_subset = np.any(group_assignment, axis=1)
    # if z:
    #     print(len(group_assignment), np.sum(group_subset), object_subset)
    if not np.any(group_subset):
        return counts

    features_from_group = (
            group_assignment[group_subset, :, None, None]   # (n_groups, n_objects, 1, 1)
            * source_is_component[None, object_subset, :, None]         # (1, n_objects, n_features, 1)
            * features[None, object_subset, ...]                        # (1, n_objects, n_features, n_states)
    )
    counts[group_subset] = np.sum(features_from_group, axis=1)
    return counts

    # return np.einsum('hij,ijk->hjk', is_source_group.astype(float), features)


def compute_feature_counts(
    features: NDArray[bool],
    sample: Sample,
) -> dict[str, NDArray[int]]:
    """Update the likelihood values for each of the mixture components"""
    clusters = sample.clusters.value
    confounders = sample.confounders
    source = sample.source.value
    counts = {'cluster': compute_effect_counts(features, clusters, source[..., 0])}
    for i, conf in enumerate(confounders.keys(), start=1):
        counts[f'c_{conf}'] = compute_effect_counts(features, confounders[conf].group_assignment, source[..., i])
    return counts

timed_compute_effect_counts = timeit('ms')(compute_effect_counts)

def update_feature_counts(
    sample_old: Sample,
    sample_new: Sample,
    features,
    object_subset,
):
    counts = sample_new.source.counts
    confounders = sample_new.confounders

    # Update cluster counts:
    old_cluster_counts = compute_effect_counts(
        features=features,
        group_assignment=sample_old.clusters.value,
        source_is_component=sample_old.source.value[..., 0],
        object_subset=object_subset
    )
    new_cluster_counts = compute_effect_counts(
        features=features,
        group_assignment=sample_new.clusters.value,
        source_is_component=sample_new.source.value[..., 0],
        object_subset=object_subset
    )
    counts['cluster'] += new_cluster_counts - old_cluster_counts

    for i, conf in enumerate(confounders.keys(), start=1):
        old_conf_counts = compute_effect_counts(
            features=features,
            group_assignment=sample_old.confounders[conf].group_assignment,
            source_is_component=sample_old.source.value[..., i],
            object_subset=object_subset
        )
        new_conf_counts = compute_effect_counts(
            features=features,
            group_assignment=sample_new.confounders[conf].group_assignment,
            source_is_component=sample_new.source.value[..., i],
            object_subset=object_subset
        )
        counts[f'c_{conf}'] += new_conf_counts - old_conf_counts

    return counts


def conditional_effect_mean(
    features: NDArray[bool],                     # shape: (n_objects, n_features, n_states)
    is_source_group: NDArray[bool],              # shape: (n_groups, n_objects, n_features)
    applicable_states: NDArray[bool],            # shape: (n_features, n_states)
    prior_counts: NDArray[float],                # shape: (n_groups, n_features, n_states)
    feature_counts: NDArray[int] = None,         # shape: (n_groups, n_features, n_states)
) -> NDArray[float]:                             # shape: (n_groups, n_features, n_states)
    if feature_counts is None:
        counts = conditional_effect_concentration(
            features=features,
            is_source_group=is_source_group,
            applicable_states=applicable_states,
            prior_counts=prior_counts
        )
    else:
        counts = feature_counts + prior_counts

    # The expected effect is given by the normalized counts
    return normalize(counts, axis=-1)


def conditional_effect_sample(
    features: NDArray[bool],                     # shape: (n_objects, n_features, n_states)
    is_source_group: NDArray[bool],              # shape: (n_groups, n_objects, n_features)
    applicable_states: NDArray[bool],            # shape: (n_features, n_states)
    prior_counts: NDArray[float] | float = 0.1,  # shape: (n_groups, n_features, n_states)
) -> NDArray[float]:                             # shape: (n_groups, n_features, n_states)
    concentration = conditional_effect_concentration(
        features=features,
        is_source_group=is_source_group,
        applicable_states=applicable_states,
        prior_counts=prior_counts
    )
    p = np.zeros_like(concentration)
    for i_g, conc_g in enumerate(concentration):
        for i_f, states_f in enumerate(applicable_states):
            p[i_g, i_f, states_f] = np.random.dirichlet(conc_g[i_f, states_f])
    return p


def sample_dirichlet_batched(
    concentration: NDArray[float] ,    # shape: (*batch_shape, n_features, n_states)
    applicable_states: NDArray[bool],  # shape: (n_features, n_states)
) -> NDArray[float]:  # shape: (*batch_shape, n_features, n_states)
    p = np.zeros_like(concentration)
    for i_f, states_f in enumerate(applicable_states):
        p[..., i_f, states_f] = np.random.dirichlet(concentration[..., i_f, states_f])
    return p


# def likelihood_by_group(
#         features: NDArray[bool],                # shape: (n_objects, n_features, n_states)
#         from_group: NDArray[bool],              # shape: (n_objects, n_features)
#         applicable_states: NDArray[bool],       # shape: (n_features, n_states)
#         prior_counts: NDArray
# ):
#     ...

timed_conditional_effect_mean = timeit('ms')(conditional_effect_mean)
timed_compute_component_likelihood = timeit('ms')(compute_component_likelihood)

def likelihood_per_component(
    model: Model,
    sample: Sample,
    source: NDArray[bool] = None,  # shape: (n_objects, n_features, n_components)
    caching=True
) -> NDArray[float]:  # shape: (n_objects, n_feature, n_components)
    """Update the likelihood values for each of the mixture components"""
    CHECK_CACHING = False
    features = model.data.features
    confounders = model.data.confounders
    if source is None:
        source = sample.source.value

    cache = sample.cache.component_likelihoods
    if caching and not cache.is_outdated():
        if CHECK_CACHING:
            assert np.all(cache.value == likelihood_per_component(model, sample, caching=False))
        return cache.value

    with cache.edit() as component_likelihood:
        cluster_effect = conditional_effect_mean(
            features=features.values,
            is_source_group=sample.clusters.value[..., np.newaxis] & source[np.newaxis, ..., 0],
            applicable_states=features.states,
            prior_counts=model.prior.prior_cluster_effect.concentration_array,
            feature_counts=sample.source.counts['cluster']
        )
        # Update component likelihood for cluster effects:
        compute_component_likelihood(
            features=features.values,
            probs=cluster_effect,
            groups=sample.clusters.value,
            changed_groups=cache.what_changed(['clusters'], caching=False),
            out=component_likelihood[..., 0],
        )

        # Update component likelihood for confounding effects:
        for i, conf in enumerate(confounders.keys(), start=1):
            groups = confounders[conf].group_assignment
            conf_effect = conditional_effect_mean(
                features=features.values,
                is_source_group=groups[..., np.newaxis] & source[np.newaxis, ..., i],
                applicable_states=features.states,
                prior_counts=model.prior.prior_confounding_effects[conf].concentration_array,
                feature_counts=sample.source.counts[f'c_{conf}'],
            )
            compute_component_likelihood(
                features=features.values,
                probs=conf_effect,
                groups=groups,
                changed_groups=cache.what_changed(f'c_{conf}', caching=False),
                # changed_groups=cache.what_changed(f'c_{conf}', caching=False),
                out=component_likelihood[..., i],
            )

        component_likelihood[features.na_values] = 1.

    return cache.value


def sample_source_from_prior(
    sample: Sample,
) -> NDArray:
    """Sample the source array from the prior, i.e. from the weights array"""
    p = update_weights(sample)
    return sample_categorical(p, binary_encoding=True)


def logprob_source_from_prior(
    sample: Sample,
) -> float:
    p = update_weights(sample)
    return np.log(p[sample.source]).sum()

