from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sbayes.sampling.state import Sample, FeatureCounts


def compute_effect_counts(
    features: NDArray[bool],                # shape: (n_objects, n_features, n_states)
    group_assignment: NDArray[bool],        # shape: (n_groups, n_objects)
    source_is_component: NDArray[bool],     # shape: (n_objects, n_features)
    object_subset: slice | list[int] | NDArray[int] = slice(None),
) -> NDArray[int]:  # shape: (n_groups, n_features, n_states)
    """"Computes the state counts of all features in all groups of a confounder."""
    group_assignment = group_assignment[:, object_subset]
    n_groups, n_objects = group_assignment.shape
    _, n_features, n_states = features.shape
    counts = np.zeros((n_groups, n_features, n_states))

    group_subset = np.any(group_assignment, axis=1)
    if not np.any(group_subset):
        return counts

    features_from_group = (
            group_assignment[group_subset, :, None, None]           # (n_groups, n_objects, 1, 1)
            * source_is_component[None, object_subset, :, None]     # (1, n_objects, n_features, 1)
            * features[None, object_subset, ...]                    # (1, n_objects, n_features, n_states)
    )
    counts[group_subset] = np.sum(features_from_group, axis=1)
    return counts


def recalculate_feature_counts(
    features: NDArray[bool],
    sample: Sample,
) -> dict[str, FeatureCounts]:
    """Update the likelihood values for each of the mixture components."""
    clusters = sample.clusters.value
    confounders = sample.confounders
    source = sample.source.value

    new_cluster_counts = compute_effect_counts(features, clusters, source[..., 0])
    sample.feature_counts['clusters'].set_value(new_cluster_counts)

    for i, conf in enumerate(confounders.keys(), start=1):
        groups = confounders[conf].group_assignment
        new_conf_counts = compute_effect_counts(features, groups, source[..., i])
        sample.feature_counts[conf].set_value(new_conf_counts)

    return sample.feature_counts


def update_feature_counts(
    sample_old: Sample,
    sample_new: Sample,
    features,
    object_subset,
):
    counts = sample_new.feature_counts
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
    counts['clusters'].add_changes(diff=new_cluster_counts - old_cluster_counts)

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

        counts[conf].add_changes(diff=new_conf_counts - old_conf_counts)

    return counts
