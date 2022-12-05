#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from numpy.core._umath_tests import inner1d

from sbayes.util import dirichlet_categorical_logpdf, timeit

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import numpy as np
from numpy.typing import NDArray

from sbayes.sampling.state import Sample
from sbayes.load_data import Data


class ModelShapes(Protocol):
    n_clusters: int
    n_sites: int
    n_features: int
    n_states: int
    states_per_feature: NDArray[bool]
    n_states_per_feature: list[int]
    n_confounders: int
    n_components: int


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

    def __init__(self, data: Data, shapes: ModelShapes):
        self.features = data.features.values
        self.confounders = data.confounders
        self.shapes = shapes
        self.na_features = (np.sum(self.features, axis=-1) == 0)

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

        # Sum up log-likelihood of each mixture component:
        log_lh = 0.0
        log_lh += self.compute_lh_clusters(sample)
        for i, conf in enumerate(self.confounders, start=1):
            log_lh += self.compute_lh_confounder(sample, conf)

        return log_lh

    def compute_lh_clusters(self, sample: Sample) -> float:
        """Compute the log-likelihood of the observations that are assigned to cluster
        effects in the current sample."""

        cache = sample.cache.group_likelihoods['clusters']
        feature_counts = sample.feature_counts['clusters'].value

        with cache.edit() as lh:
            for i_cluster in cache.what_changed('counts'):
                lh[i_cluster] = self.compute_lh_group_from_counts(feature_counts[i_cluster])

        return cache.value.sum()

    def compute_lh_confounder(self, sample: Sample, conf: str) -> float:
        """Compute the log-likelihood of the observations that are assigned to confounder
        `conf` in the current sample."""

        cache = sample.cache.group_likelihoods[conf]
        feature_counts = sample.feature_counts[conf].value

        with cache.edit() as lh:
            for i_g in cache.what_changed('counts'):
                lh[i_g] = self.compute_lh_group_from_counts(feature_counts[i_g])

        return cache.value.sum()

    def compute_lh_group_from_counts(
        self,
        state_counts: NDArray[int],         # shape: (n_features, n_states)
    ) -> float:
        """Compute the likelihood (in the form of a Dirichlet multinomial distribution)
        for one cluster or one group of a confounder."""

        # Use the applicable states to construct a uniform prior
        a = self.shapes.states_per_feature.astype(float)  # shape: (n_features, n_states)

        # Compute the dirichlet-multinomial likelihood
        log_lh = dirichlet_categorical_logpdf(counts=state_counts, a=a)

        # Using the boolean `states` array implicitly defines a concentration of 1 for
        # each applicable state (i.e. uniform distribution on the probability simplex)
        # TODO: Use concentration from config files

        return log_lh.sum()

    def compute_lh_group(
        self,
        group: NDArray[bool],               # shape: (n_objects,)
        component_source: NDArray[bool],    # shape: (n_objects, n_features)
    ) -> float:
        """Compute the likelihood (in the form of a Dirichlet multinomial distribution)
        for one cluster or one group of a confounder."""

        # Which observations are attributed to this component and group:
        source_group = group[:, np.newaxis] & component_source                  # shape: (n_objects, n_features)

        # Compute state counts for all observations assigned to this group and are not NA
        where = (source_group & ~self.na_features)[..., np.newaxis]             # shape: (n_objects, n_features, 1)
        state_counts = np.sum(self.features, where=where, axis=0)               # shape: (n_features, n_states)

        # Use the applicable states to construct a uniform prior
        a = self.shapes.states_per_feature.astype(float)                        # shape: (n_features, n_states)

        # Compute the dirichlet-multinomial likelihood
        log_lh = dirichlet_categorical_logpdf(counts=state_counts, a=a)
        # log_lh = dirichlet_multinomial_logpdf(counts=state_counts, a=a)
        # Using the boolean `states` array implicitly defines a concentration of 1 for
        # each applicable state (i.e. uniform distribution on the probability simplex)
        # TODO: Use concentration from config files

        return log_lh.sum()


@timeit('µs')
def compute_component_likelihood(
    features: NDArray[bool],    # shape: (n_objects, n_features, n_states)
    probs: NDArray[float],      # shape: (n_groups, n_features, n_states)
    groups: NDArray[bool],      # shape: (n_groups, n_objects)
    changed_groups: set[int],
    out: NDArray[float]         # shape: (n_objects, n_features)
) -> NDArray[float]:            # shape: (n_objects, n_features)

    # [NN] Idea: If the majority of groups is outdated, a full update (w/o caching) may be faster
    # [NN] ...does not seem like it -> deactivate for now.
    # if len(changed_groups) > 0.8 * groups.shape[0]:
    #     return np.einsum('ijk,hjk,hi->ij', features, probs, groups, optimize=True)

    out[~groups.any(axis=0), :] = 0.
    for i in changed_groups:
        g = groups[i]
        f_g = features[g, :, :]
        p_g = probs[i, :, :]
        # out[g, :] = np.einsum('ijk,jk->ij', f_g, p_g, optimize=True)
        # out[g, :] = np.sum(f_g * p_g[np.newaxis, ...], axis=-1)
        out[g, :] = inner1d(f_g, p_g[np.newaxis, ...])

    return out


def update_weights(sample: Sample, caching=True) -> NDArray[float]:
    """Compute the normalized weights of each component at each site.
    Args:
        sample: the current MCMC sample.
        caching: ignore cache if set to false.
    Returns:
        np.array: normalized weights of each component at each site.
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
    # Broadcast weights to each site and mask with the has_components arrays
    # Broadcasting:
    #   `weights` does not know about sites -> add axis to broadcast to the sites-dimension of `has_component`
    #   `has_components` does not know about features -> add axis to broadcast to the features-dimension of `weights`
    weights_per_site = weights[np.newaxis, :, :] * has_components[:, np.newaxis, :]

    # Re-normalize the weights, where weights were masked
    return weights_per_site / weights_per_site.sum(axis=2, keepdims=True)
