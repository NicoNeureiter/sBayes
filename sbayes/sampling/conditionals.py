from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sbayes.load_data import Features

from sbayes.preprocessing import sample_categorical
from sbayes.sampling.state import Sample, FeatureCounts
from sbayes.util import normalize

from sbayes.model.likelihood import update_weights, compute_component_likelihood
# from sbayes.cython.util import compute_component_likelihood


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

    counts += prior_counts

    return counts


def conditional_effect_mean_from_scratch(
    features: NDArray[bool],                     # shape: (n_objects, n_features, n_states)
    is_source_group: NDArray[bool],              # shape: (n_groups, n_objects, n_features)
    applicable_states: NDArray[bool],            # shape: (n_features, n_states)
    prior_counts: NDArray[float],                # shape: (n_groups, n_features, n_states)
) -> NDArray[float]:                             # shape: (n_groups, n_features, n_states)
    counts = conditional_effect_concentration(
        features=features,
        is_source_group=is_source_group,
        applicable_states=applicable_states,
        prior_counts=prior_counts
    )

    # The expected effect is given by the normalized counts
    return normalize(counts, axis=-1)


def conditional_effect_mean(
    prior_counts: NDArray[float],                # shape: (n_groups, n_features, n_states)
    feature_counts: NDArray[int] = None,         # shape: (n_groups, n_features, n_states)
) -> NDArray[float]:                             # shape: (n_groups, n_features, n_states)

    # Sum up the feature counts and prior counts to obtain posterior effect counts
    counts = feature_counts + prior_counts

    # The expected effect is given by the normalized posterior counts
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


def likelihood_per_component(
    model,
    sample: Sample,
    feature_counts: dict[str, FeatureCounts] = None,
    caching=True
) -> NDArray[float]:  # shape: (n_objects, n_feature, n_components)
    """Update the likelihood values for each of the mixture components"""
    CHECK_CACHING = False

    features = model.data.features
    confounders = model.data.confounders
    if feature_counts is None:
        feature_counts = sample.feature_counts

    cache = sample.cache.component_likelihoods
    if caching and not cache.is_outdated():
        if CHECK_CACHING:
            assert np.all(cache.value == likelihood_per_component(model, sample, caching=False))
        return cache.value

    with cache.edit() as component_likelihood:
        changed_clusters = cache.what_changed(input_key=['clusters', 'clusters_counts'], caching=caching)
        # print('cluster', changed_clusters)

        if len(changed_clusters) > 0:
            # The expected cluster effect is given by the normalized posterior counts
            cluster_effect_counts = (  # feature counts + prior counts
                feature_counts['clusters'].value +
                model.prior.prior_cluster_effect.concentration_array
            )
            cluster_effect = normalize(cluster_effect_counts, axis=-1)

            # Update component likelihood for cluster effects:
            compute_component_likelihood(
                features=features.values,
                probs=cluster_effect,
                groups=sample.clusters.value,
                changed_groups=changed_clusters,
                out=component_likelihood[..., 0],
            )

            # with sample.clusters.value_for_cython() as clusters:
            #     # Update component likelihood for cluster effects:
            #     compute_component_likelihood(
            #         features=features.values,
            #         features_by_group=[features.values[c] for c in clusters],
            #         probs=cluster_effect,
            #         groups=clusters,
            #         changed_groups=changed_clusters,
            #         out=component_likelihood[..., 0],
            #     )

        # Update component likelihood for confounding effects:
        for i, conf in enumerate(confounders.keys(), start=1):
            changed_groups = cache.what_changed(input_key=[f'c_{conf}', f'{conf}_counts'], caching=caching)
            # print(conf, changed_groups)

            if len(changed_groups) == 0:
                continue

            groups = confounders[conf].group_assignment

            # The expected confounding effect is given by the normalized posterior counts
            conf_effect_counts = (  # feature counts + prior counts
                feature_counts[conf].value +
                model.prior.prior_confounding_effects[conf].concentration_array(sample)
            )
            conf_effect = normalize(conf_effect_counts, axis=-1)

            compute_component_likelihood(
                features=features.values,
                probs=conf_effect,
                groups=groups,
                changed_groups=changed_groups,
                out=component_likelihood[..., i],
            )

        component_likelihood[features.na_values] = 1.

    if caching and CHECK_CACHING:
        cached = np.copy(cache.value)
        recomputed = likelihood_per_component(model, sample, caching=False)
        assert np.allclose(cached, recomputed)

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

