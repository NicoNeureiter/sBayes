#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from sbayes.sampling.counts import recalculate_feature_counts
from sbayes.util import dirichlet_categorical_logpdf
from numba import njit

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import numpy as np
from numpy.typing import NDArray

from sbayes.sampling.state import Sample
from sbayes.load_data import Data
from sbayes.model.model_shapes import ModelShapes


class Likelihood(object):

    """Likelihood of the sBayes model.

    Attributes:
        features (np.array): The values for all sites and features.
            shape: (n_objects, n_features, n_categories)
        confounders (dict): Assignment of objects to confounders. For each confounder (c) one np.array
            with shape: (n_groups(c), n_objects)
        shapes (ModelShapes): A dataclass with shape information for building the Likelihood and Prior objects
        na_features (np.array): A boolean array indicating missing observations
            shape: (n_objects, n_features)
    """

    def __init__(self, data: Data, shapes: ModelShapes, prior):
        self.features = data.features.values
        self.confounders = data.confounders
        self.shapes = shapes
        self.na_features = (np.sum(self.features, axis=-1) == 0)
        self.prior = prior

        self.source_index = {'clusters': 0}
        for i, conf in enumerate(self.confounders, start=1):
            self.source_index[conf] = i

    def __call__(self, sample: Sample, caching=True) -> float:
        """Compute the likelihood of all sites. The likelihood is defined as a mixture of areal and confounding effects.
            Args:
                sample: A Sample object consisting of clusters and weights
            Returns:
                The joint likelihood of the current sample
        """
        if not caching:
            recalculate_feature_counts(self.features, sample)

        # Sum up log-likelihood of each mixture component:
        log_lh = 0.0
        log_lh += self.compute_lh_clusters(sample, caching=caching)
        for i, conf in enumerate(self.confounders, start=1):
            log_lh += self.compute_lh_confounder(sample, conf, caching=caching)

        return log_lh

    def compute_lh_clusters(self, sample: Sample, caching=True) -> float:
        """Compute the log-likelihood of the observations that are assigned to cluster
        effects in the current sample."""

        cache = sample.cache.group_likelihoods['clusters']
        feature_counts = sample.feature_counts['clusters'].value

        with cache.edit() as lh:
            for i_cluster in cache.what_changed('counts', caching=caching):
                lh[i_cluster] = dirichlet_categorical_logpdf(
                    counts=feature_counts[i_cluster],
                    a=self.prior.prior_cluster_effect.concentration_array,
                ).sum()

        return cache.value.sum()

    def compute_lh_confounder(self, sample: Sample, conf: str, caching=True) -> float:
        """Compute the log-likelihood of the observations that are assigned to confounder
        `conf` in the current sample."""

        cache = sample.cache.group_likelihoods[conf]
        feature_counts = sample.feature_counts[conf].value
        conf_prior = self.prior.prior_confounding_effects[conf]

        with cache.edit() as lh:
            prior_concentration = conf_prior.concentration_array(sample)

            hyperprior_has_changed = conf_prior.any_dynamic_priors and cache.ahead_of('universal_counts')
            changed_groups = cache.what_changed('counts', caching=caching and not hyperprior_has_changed)

            for i_g in changed_groups:
                lh[i_g] = dirichlet_categorical_logpdf(
                    counts=feature_counts[i_g],
                    a=prior_concentration[i_g],
                ).sum()

        return cache.value.sum()


@njit
def compute_component_likelihood(
    features: NDArray[bool],        # shape: (n_objects, n_features, n_states)
    probs: NDArray[float],          # shape: (n_groups, n_features, n_states)
    groups: NDArray[bool],          # shape: (n_groups, n_objects)
    changed_groups: NDArray[int],   # shape: (n_changed_groups)
    out: NDArray[float]             # shape: (n_objects, n_features)
) -> NDArray[float]:                # shape: (n_objects, n_features)
    """Compute the likelihood of each observation in `features` according to one mixture
    components.
    Args:
        features: The feature observations in the dataset.
        probs: The probabilities defining the categorical distribution of the mixture component.
        groups: Boolean array assigning each object to a group in the mixture components.
        changed_groups: Array of indices indicating which groups need to be recalculated.
        out: The output array where the result will be stored.
    """
    # Assign likelihood 0 to objects that are in no group
    no_group = (groups.sum(axis=0) == 0)
    out[no_group, :] = 0.

    # Recalculate the likelihood of groups that changed since last updating the cache
    for i in changed_groups:
        g = groups[i]
        f_g = features[g, :, :]
        p_g = probs[i, :, :]
        out[g, :] = np.sum(f_g * p_g[np.newaxis, ...], axis=-1)
        # assert np.allclose(out[g, :], np.einsum('ijk,jk->ij', f_g, p_g))

    return out


def compute_component_likelihood_exact(
    features: NDArray[bool],        # shape: (n_objects, n_features, n_states)
    probs: list[NDArray[float]],    # shape: (n_objects_in_g, n_features, n_states) for each group
    groups: NDArray[bool],          # shape: (n_groups, n_objects)
    changed_groups: NDArray[int],
    out: NDArray[float]             # shape: (n_objects, n_features)
) -> NDArray[float]:                # shape: (n_objects, n_features)
    out[~groups.any(axis=0), :] = 0.
    for i in changed_groups:
        g = groups[i]
        f_g = features[g, :, :]
        p_g = probs[i]              # shape: (n_features, n_states)
        out[g, :] = np.einsum('ijk,ijk->ij', f_g, p_g)
        assert np.allclose(out[g, :], np.sum(f_g * p_g[np.newaxis, ...], axis=-1))
    return out


def update_weights(sample: Sample, caching: bool = True) -> NDArray[float]:
    """Compute the normalized mixture weights of each component at each object.
    Args:
        sample: the current MCMC sample.
        caching: ignore cache if set to false.
    Returns:
        np.array: normalized weights of each component at each object.
            shape: (n_objects, n_features, 1 + n_confounders)
    """
    cache = sample.cache.weights_normalized

    if (not caching) or cache.is_outdated():
        w_normed = normalize_weights(sample.weights.value, sample.cache.has_components.value)
        cache.update_value(w_normed)

    return cache.value


def normalize_weights(
    weights: NDArray[float],  # shape: (n_features, 1 + n_confounders)
    has_components: NDArray[bool]  # shape: (n_objects, 1 + n_confounders)
) -> NDArray[float]:  # shape: (n_objects, n_features, 1 + n_confounders)
    """This function assigns each site a weight if it has a likelihood and zero otherwise
    Args:
        weights: the weights to normalize
        has_components: indicators for which objects are affected by cluster and confounding effects
    Return:
        the weight_per site
    """
    # Find the unique patterns in `has_components` and remember the inverse mapping
    pattern, pattern_inv = np.unique(has_components, axis=0, return_inverse=True)

    # Calculate the normalized weights per pattern
    w_per_pattern = pattern[:, None, :] * weights[None, :, :]
    w_per_pattern /= np.sum(w_per_pattern, axis=-1, keepdims=True)

    # Broadcast the normalized weights per pattern to the objects where the patterns appeared using pattern_inv
    return w_per_pattern[pattern_inv]

    # # Broadcast weights to each site and mask with the has_components arrays
    # # Broadcasting:
    # #   `weights` does not know about sites -> add axis to broadcast to the sites-dimension of `has_component`
    # #   `has_components` does not know about features -> add axis to broadcast to the features-dimension of `weights`
    # weights_per_site = weights[np.newaxis, :, :] * has_components[:, np.newaxis, :]
    #
    # # Re-normalize the weights, where weights were masked
    # return weights_per_site / weights_per_site.sum(axis=2, keepdims=True)
