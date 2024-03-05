from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sbayes.load_data import FeatureType
from sbayes.sampling.state import Sample, FeatureCounts, CategoricalSample, SufficientStatistics


def compute_effect_counts(
    features: NDArray[bool],                # shape: (n_objects, n_features, n_states)
    group_assignment: NDArray[bool],        # shape: (n_groups, n_objects)
    source_is_component: NDArray[bool],     # shape: (n_objects, n_features)
    object_subset: slice | list[int] | NDArray[int] = slice(None),
) -> NDArray[int]:  # shape: (n_groups, n_features, n_states)
    """"Computes the state counts of all categorical features in all groups of a confounder."""

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
    categorical_features: NDArray[bool],
    sample: Sample,
) -> dict[str, SufficientStatistics]:
    """Update the likelihood values for each of the mixture components."""

    clusters = sample.clusters.value
    confounders = sample.confounders
    source = sample.categorical.source.value

    new_cluster_counts = compute_effect_counts(categorical_features, clusters, source[..., 0])
    sample.categorical.sufficient_statistics['clusters'].set_value(new_cluster_counts)

    for i, conf in enumerate(confounders.keys(), start=1):
        groups = confounders[conf].group_assignment
        new_conf_counts = compute_effect_counts(categorical_features, groups, source[..., i])
        sample.categorical.sufficient_statistics[conf].set_value(new_conf_counts)

    # Mark all other sufficient statistics as outdated
    for ft_sample in sample.feature_type_samples.values():
        for conf_suff_stats in ft_sample.sufficient_statistics.values():
            conf_suff_stats.make_dirty()

    return sample.categorical.sufficient_statistics


def update_sufficient_statistics(
    sample_old: Sample,
    sample_new: Sample,
    features: NDArray[bool],    # shape: (n_objects, n_features, feature_dim)
    object_subset: slice | list[int] | NDArray[int],
):
    confounders = sample_new.confounders
    clusters_old = sample_old.clusters.value
    clusters_new = sample_new.clusters.value
    cluster_changes = np.any(clusters_old != clusters_new, axis=1)

    # update_feature_counts(sample_old, sample_new, features, object_subset)

    for ft in sample_old.feature_type_samples.keys():
        ft_sample_old = sample_old.feature_type_samples[ft]
        ft_sample_new = sample_new.feature_type_samples[ft]
        if ft == FeatureType.categorical:
            continue  # Feature counts were already updated above
        else:
            source_old = ft_sample_old.source.value
            source_new = ft_sample_new.source.value
            source_changes: NDArray[bool] = (source_old != source_new)
            # shape: (n_objects, n_features, n_components)

            # Register changes in cluster effect observations
            source_changes_by_cluster = np.dot(clusters_new, source_changes[..., 0])
            cluster_changes = (cluster_changes | np.any(source_changes_by_cluster, axis=1))

            ft_sample_new.sufficient_statistics['clusters'].mark_changes(cluster_changes)

            # Register changes in confounding effect observations
            for i_comp, (conf_name, conf) in enumerate(confounders.items(), start=1):
                groups = conf.group_assignment
                group_changes = np.any(np.dot(groups, source_changes[..., i_comp]), axis=1)
                ft_sample_new.sufficient_statistics[conf_name].mark_changes(group_changes)
                # ft_sample_new.sufficient_statistics[conf_name].make_dirty()


def update_feature_counts(
    sample_old: Sample,
    sample_new: Sample,
    features: NDArray[bool],    # shape: (n_objects, n_features, n_states)
    object_subset: slice | list[int] | NDArray[int],
) -> dict[str, FeatureCounts]:
    counts = sample_new.categorical.sufficient_statistics
    confounders = sample_new.confounders

    for c in counts.values():
        assert np.all(c.value >= 0)

    # Update cluster counts:
    old_cluster_counts = compute_effect_counts(
        features=features,
        group_assignment=sample_old.clusters.value,
        source_is_component=sample_old.categorical.source.value[..., 0],
        object_subset=object_subset
    )
    new_cluster_counts = compute_effect_counts(
        features=features,
        group_assignment=sample_new.clusters.value,
        source_is_component=sample_new.categorical.source.value[..., 0],
        object_subset=object_subset
    )

    counts['clusters'].add_changes(old=old_cluster_counts, new=new_cluster_counts)
    assert np.all(counts['clusters'].value >= 0)

    for i, conf in enumerate(confounders.keys(), start=1):
        old_conf_counts = compute_effect_counts(
            features=features,
            group_assignment=sample_old.confounders[conf].group_assignment,
            source_is_component=sample_old.categorical.source.value[..., i],
            object_subset=object_subset
        )
        new_conf_counts = compute_effect_counts(
            features=features,
            group_assignment=sample_new.confounders[conf].group_assignment,
            source_is_component=sample_new.categorical.source.value[..., i],
            object_subset=object_subset
        )

        counts[conf].add_changes(old=old_conf_counts, new=new_conf_counts)

    for c in counts.values():
        assert np.all(c.value >= 0)

    return counts
