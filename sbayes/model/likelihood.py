#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from sbayes.util import normalize, log_multinom

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import numpy as np
from numpy.typing import NDArray
import scipy.stats as stats

from sbayes.sampling.state import Sample
from sbayes.load_data import Data


class ModelShapes(Protocol):
    n_clusters: int
    n_sites: int
    n_features: int
    n_states: int
    states_per_feature: NDArray[bool]
    n_states_per_feature: list[int]


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

        self.source_index = {}
        self.source_index['clusters'] = 0
        for i, conf in enumerate(self.confounders, start=1):
            self.source_index[conf] = i

    def __call__(self, sample, caching=True):
        """Compute the likelihood of all sites. The likelihood is defined as a mixture of areal and confounding effects.
            Args:
                sample(Sample): A Sample object consisting of clusters and weights
            Returns:
                float: The joint likelihood of the current sample
            """

        # Sum up log-likelihood of each mixture component:
        log_lh = 0.0
        log_lh += self.compute_lh_clusters(sample)
        for i, conf in enumerate(self.confounders, start=1):
            log_lh += self.compute_lh_confounder(sample, conf)

        return log_lh

    def compute_lh_clusters(self, sample: Sample) -> float:
        cluster_source = sample.source.value[..., 0]

        log_p = 0.0
        for i_cluster in range(sample.n_clusters):  # TODO use caching
            c = sample.clusters.value[i_cluster]
            p = sample.cluster_effect.value[i_cluster, :, :]
            log_p += self.compute_lh_group(c, cluster_source, p)
            assert not np.isnan(log_p)

        return log_p

    def compute_lh_confounder(self, sample: Sample, conf: str) -> float:
        conf_source = sample.source.value[..., self.source_index[conf]]
        confounder = sample.confounders[conf]
        confounding_effect = sample.confounding_effects[conf]

        log_p = 0.0
        for i_group in range(sample.n_groups(conf)):
            g = confounder.group_assignment[i_group]
            p = confounding_effect.value[i_group, :, :]
            log_p += self.compute_lh_group(g, conf_source, p)
            assert not np.isnan(log_p)

        return log_p

    def compute_lh_group(
        self,
        group: NDArray[bool],               # shape: (n_objects,)
        component_source: NDArray[bool],    # shape: (n_objects, n_features)
        p: NDArray[float],                  # shape: (n_features, n_states)
    ) -> float:
        # Which observations are attributed to this component and group:
        source_group = group[:, np.newaxis] & component_source                  # shape: (n_objects, n_features)

        # Compute state counts for all observations assigned to this group and are not NA
        where = (source_group & ~self.na_features)[..., np.newaxis]             # shape: (n_objects, n_features, 1)
        state_counts = np.sum(self.features, where=where, axis=0)               # shape: (n_features, n_states)

        # # Alternative way of computing likelihood using multinomial distribution:
        # n = np.sum(state_counts, axis=-1)
        # lh_per_feat = stats.multinomial._logpmf(x=state_counts, n=n, p=p)
        # for i in range(len(n)):
        #     lh_per_feat[i] -= log_multinom(n[i], state_counts[i])
        # lh = lh.sum(where=(n>0))

        # Directly computing likelihood through sum of powers:
        return np.log(p**state_counts).sum()

    ####################
    # FOR OPERATORS
    ##########
    def update_component_likelihoods(self, sample: Sample, caching=True) -> NDArray[float]:
        """Update the likelihood values for each of the mixture components"""
        component_likelihood = np.zeros((sample.n_objects, sample.n_features, sample.n_components))

        # Update component likelihood for cluster effects:
        compute_component_likelihood(
            features=self.features,
            probs=sample.cluster_effect.value,
            groups=sample.clusters.value,
            changed_groups=set(range(sample.n_clusters)),
            out=component_likelihood[..., 0],
        )

        # Update component likelihood for confounding effects:
        for i, conf in enumerate(self.confounders, start=1):
            compute_component_likelihood(
                features=self.features,
                probs=sample.confounding_effects[conf].value,
                groups=sample.confounders[conf].group_assignment,
                changed_groups=set(range(sample.n_groups(conf))),
                out=component_likelihood[..., i],
            )

        component_likelihood[self.na_features] = 1.
        sample.cache.component_likelihoods.set_up_to_date()

        return component_likelihood


def compute_component_likelihood(
    features: NDArray[bool],
    probs: NDArray[float],
    groups: NDArray[bool],  # (n_groups, n_sites)
    changed_groups: set[int],
    out: NDArray[float]
) -> NDArray[float]:  # shape: (n_sites, n_features)
    out[~groups.any(axis=0), :] = 0.
    for i in changed_groups:
        g = groups[i]
        f_g = features[g, :, :]
        p_g = probs[i, :, :]
        out[g, :] = np.einsum('ijk,jk->ij', f_g, p_g)
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
