from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sbayes.load_data import Features, Confounder
from sbayes.model import Model

from sbayes.preprocessing import sample_categorical
from sbayes.sampling.counts import recalculate_feature_counts
from sbayes.sampling.state import Sample, Clusters
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
    caching=True
) -> NDArray[float]:  # shape: (n_objects, n_feature, n_components)
    """Update the likelihood values for each of the mixture components"""
    CHECK_CACHING = False

    features = model.data.features
    confounders = model.data.confounders
    feature_counts = sample.feature_counts

    cache = sample.cache.component_likelihoods
    if caching and not cache.is_outdated():
        if CHECK_CACHING:
            assert np.all(cache.value == likelihood_per_component(model, sample, caching=False))
        return cache.value

    with cache.edit() as component_likelihood:
        changed_clusters = cache.what_changed(input_key=['clusters', 'clusters_counts'], caching=caching)

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


def approx_likelihood_per_component(
    model: Model,
    sample: Sample,
    object_subset,
) -> NDArray[float]:  # shape: (n_objects, n_feature, n_components)
    features = model.data.features
    confounders = model.data.confounders

    likelihoods = np.zeros((sample.n_objects, sample.n_features, sample.n_components))
    likelihoods[..., 0] = approx_cluster_likelihood(
        sample.clusters, features, object_subset, sample.source.value, sample.feature_counts
    )

    # for i_conf, conf in enumerate(confounders.values(), start=1):
    #     likelihoods[..., i_conf] = approx_confounder_likelihood(conf, features)

    for i_conf, conf in enumerate(confounders.keys(), start=1):
        confounder = confounders[conf]
        groups = confounders[conf].group_assignment

        features_conf = features.values * sample.source.value[:, :, i_conf, None]

        changeable_counts = np.array([
            np.sum(features_conf[g] * object_subset[g, None, None], axis=0)
            for g in groups
        ])
        conf_effect_counts = (  # feature counts + prior counts
            sample.feature_counts[conf].value - changeable_counts  # We only use the counts from unchanged objects (to keep the transition symmetric)
            + model.prior.prior_confounding_effects[conf].concentration_array(sample)
        )

        conf_effect = normalize(conf_effect_counts, axis=-1)
        compute_component_likelihood(
            features=features.values,
            probs=conf_effect,
            groups=groups,
            changed_groups=set(np.arange(len(groups))),
            out=likelihoods[..., i_conf],
        )

    likelihoods[features.na_values] = 1.

    return likelihoods


def approx_cluster_likelihood(
    clusters: Clusters,
    features: Features,
    object_subset,
    source: NDArray[bool],  # (n_objects, n_features, n_components)
        feature_counts,
) -> NDArray[float]:  # (n_objects, n_features)
    lh = np.zeros((features.n_objects, features.n_features))
    unchanged_objects = ~object_subset
    cluster_features = features.values * source[:, :, 0, None]

    for i_c, c in enumerate(clusters.value):
        feature_counts_c = np.sum(cluster_features[c & unchanged_objects], axis=0)
        p = normalize(features.states + feature_counts_c, axis=-1)  # TODO: use prior counts, rather than 1+
        lh[c] = np.sum(p[None, ...] * features.values[c], axis=-1)

    return lh


def approx_confounder_likelihood(
    confounder: Confounder,
    features: Features,
    # prior: ConfoundingEffectsPrior
) -> NDArray[float]:  # (n_objects, n_features)
    lh = np.zeros((features.n_objects, features.n_features))
    for i_g, g in enumerate(confounder.group_assignment):
        p = normalize(features.states + confounder.feature_counts[i_g], axis=-1)  # TODO: use prior counts, rather than 1+
        lh[g] = np.sum(p[None,...] * features.values[g], axis=-1)

    return lh


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


def impute_source(sample: Sample, model: Model):
    na_features = model.data.features.na_values

    # Next iteration: sample source from prior (allows calculating feature counts)
    source = sample_source_from_prior(sample)
    source[na_features] = 0
    sample.source.set_value(source)
    recalculate_feature_counts(model.data.features.values, sample)

    # Next step: generate posterior sample of source
    lh_per_component = likelihood_per_component(model=model, sample=sample, caching=False)
    weights = update_weights(sample)
    p = normalize(lh_per_component * weights, axis=-1)

    # Sample the new source assignments
    with sample.source.edit() as source:
        source = sample_categorical(p=p, binary_encoding=True)
        source[na_features] = 0

    recalculate_feature_counts(model.data.features.values, sample)

