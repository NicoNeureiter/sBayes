#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import numpy as np
from numpy.typing import NDArray

from sbayes.sampling.state import Sample, ArrayParameter
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

    def __call__(self, sample, caching=True):
        """Compute the likelihood of all sites. The likelihood is defined as a mixture of areal and confounding effects.
            Args:
                sample(Sample): A Sample object consisting of clusters and weights
            Returns:
                float: The joint likelihood of the current sample
            """
        if not caching:
            sample.cache.clear()

        # Compute the likelihood values per mixture component
        component_lhs = self.update_component_likelihoods(sample, caching=caching)

        # Compute the weights of the mixture component in each feature and site
        weights = update_weights(sample, caching=caching)

        # Compute the total log-likelihood
        observation_lhs = self.get_observation_lhs(component_lhs, weights, sample.source)
        sample.observation_lhs = observation_lhs
        log_lh = np.sum(np.log(observation_lhs))

        return log_lh

    @staticmethod
    def get_observation_lhs(
        all_lh: NDArray,                # shape: (n_objects, n_features, n_components)
        weights: NDArray[float],        # shape: (n_objects, n_features, n_components)
        source: ArrayParameter | None,  # shape: (n_objects, n_features, n_components)
    ) -> NDArray[float]:                # shape: (n_objects, n_features)
        """Combine likelihood from the selected source distributions."""
        if source is None:
            return np.sum(weights * all_lh, axis=2).ravel()
        else:
            is_source = np.where(source.value.ravel())
            return all_lh.ravel()[is_source]

    def update_component_likelihoods(
        self,
        sample: Sample,
        caching=True
    ) -> NDArray[float]:  # shape: (n_objects, n_features, n_components)
        """Update the likelihood values for each of the mixture components"""

        CHECK_CACHING = False

        cache = sample.cache.component_likelihoods
        if caching and not cache.is_outdated():
            x = sample.cache.component_likelihoods.value
            if CHECK_CACHING:
                assert np.all(x == self.update_component_likelihoods(sample, caching=False))
            return x

        with cache.edit() as component_likelihood:
            # TODO: Not sure whether a context manager is the best way to do this. Discuss!
            # Update component likelihood for cluster effects:
            compute_component_likelihood(
                features=self.features,
                probs=sample.cluster_effect.value,
                groups=sample.clusters.value,
                changed_groups=cache.what_changed(['cluster_effect', 'clusters'], caching),
                out=component_likelihood[..., 0],
            )
            if caching and CHECK_CACHING:
                x = component_likelihood[..., 0].copy()
                y = compute_component_likelihood(
                    features=self.features,
                    probs=sample.cluster_effect.value,
                    groups=sample.clusters.value,
                    changed_groups=cache.what_changed(['cluster_effect', 'clusters'], caching=False),
                    out=component_likelihood[..., 0],
                )
                assert np.all(x[~self.na_features] == y[~self.na_features])

            # Update component likelihood for confounding effects:
            for i, conf in enumerate(self.confounders, start=1):
                compute_component_likelihood(
                    features=self.features,
                    probs=sample.confounding_effects[conf].value,
                    groups=sample.confounders[conf].group_assignment,
                    changed_groups=cache.what_changed(f'c_{conf}', caching),
                    out=component_likelihood[..., i],
                )
                if caching and CHECK_CACHING:
                    x = component_likelihood[..., i].copy()
                    y: object = compute_component_likelihood(
                        features=self.features,
                        probs=sample.confounding_effects[conf].value,
                        groups=sample.confounders[conf].group_assignment,
                        changed_groups=cache.what_changed(f'c_{conf}', caching=False),
                        out=component_likelihood[..., i],
                    )
                    assert np.all(x[~self.na_features] == y[~self.na_features])

            component_likelihood[self.na_features] = 1.

        # cache.set_up_to_date()

        return cache.value


def compute_component_likelihood(
    features: NDArray[bool],
    probs: NDArray[float],
    groups: NDArray[bool],  # (n_groups, n_sites)
    changed_groups: set[int],
    out: NDArray[float]
) -> NDArray[float]:  # shape: (n_objects, n_features)
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
    #   `weights` doesnt know about sites -> add axis to broadcast to the sites-dimension of `has_component`
    #   `has_components` doesnt know about features -> add axis to broadcast to the features-dimension of `weights`
    weights_per_site = weights[np.newaxis, :, :] * has_components[:, np.newaxis, :]

    # Re-normalize the weights, where weights were masked
    return weights_per_site / weights_per_site.sum(axis=2, keepdims=True)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
