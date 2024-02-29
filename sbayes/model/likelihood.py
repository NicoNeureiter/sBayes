#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import nbinom

from sbayes.model.model_shapes import ModelShapes
from sbayes.sampling.state import Sample, ModelCache, GenericTypeCache
from sbayes.sampling.counts import recalculate_feature_counts
from sbayes.load_data import Data, Confounder, Features, FeatureType
from sbayes.util import dirichlet_categorical_logpdf, normalize, inner1d, \
    gaussian_mu_marginalised_logpdf, lh_poisson_lambda_marginalised_logpdf, gaussian_posterior_predictive_logpdf

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class Likelihood(object):

    categorical: LikelihoodCategorical
    gaussian: LikelihoodGaussian
    poisson: LikelihoodPoisson
    logitnormal: LikelihoodLogitNormal

    def __init__(self, data: Data, shapes: ModelShapes, prior):
        self.features = data.features
        self.confounders = data.confounders
        self.shapes = shapes
        self.prior = prior
        self.source_index = {'clusters': 0}
        for i, conf in enumerate(self.confounders, start=1):
            self.source_index[conf] = i

        self.feature_type_likelihoods = {}

        if data.features.categorical is not None:
            self.categorical = LikelihoodCategorical(features=self.features, confounders=self.confounders,
                                                     shapes=self.shapes, prior=self.prior, has_counts=True)
            self.feature_type_likelihoods[FeatureType.categorical] = self.categorical

        if data.features.gaussian is not None:
            self.gaussian = LikelihoodGaussian(features=self.features, confounders=self.confounders,
                                               shapes=self.shapes, prior=self.prior)
            self.feature_type_likelihoods[FeatureType.gaussian] = self.gaussian

        if data.features.poisson is not None:
            self.poisson = LikelihoodPoisson(features=self.features, confounders=self.confounders,
                                             shapes=self.shapes, prior=self.prior)
            self.feature_type_likelihoods[FeatureType.poisson] = self.poisson

        if data.features.logitnormal is not None:
            self.logitnormal = LikelihoodLogitNormal(features=self.features, confounders=self.confounders,
                                                     shapes=self.shapes, prior=self.prior)
            self.feature_type_likelihoods[FeatureType.logitnormal] = self.logitnormal

    def __call__(self, sample: Sample, caching=True) -> float:
        """Compute the likelihood of all sites. The likelihood is defined as a mixture of areal and confounding effects.
            Args:
                sample: A Sample object consisting of clusters and weights
            Returns:
                The joint likelihood of the current sample
        """
        return sum(
            ft_likelihood(sample=sample, caching=caching)
            for ft_likelihood in self.feature_type_likelihoods.values()
        )
        # log_lh = 0.0
        #
        # # Sum up log-likelihood of each mixture component and each type of feature
        # if sample.categorical is not None:
        #     log_lh += self.categorical(sample=sample, caching=caching)
        # if sample.gaussian is not None:
        #     log_lh += self.gaussian(sample=sample, caching=caching)
        # if sample.poisson is not None:
        #     log_lh += self.poisson(sample=sample, caching=caching)
        # if sample.logitnormal is not None:
        #     log_lh += self.logitnormal(sample=sample, caching=caching)
        #
        # return log_lh


class LikelihoodGenericType(object):

    """Likelihood of the sBayes model.

    Attributes:
        features (np.array): The values for all sites and features.
            shape: (n_objects, n_features, n_categories)
        confounders (dict): Assignment of objects to confounders. For each confounder (c) one np.array
            with shape: (n_groups(c), n_objects)
        shapes (ModelShapes): A dataclass with shape information for building the Likelihood and Prior objects
        has_counts: Does the likelihood use counts and should they be cached/recalculated? (currently only for
            categorical features)
    """
    def __init__(self, features: Features, confounders, shapes: ModelShapes, prior, has_counts=False):
        """"""
        self.features = features
        self.confounders = confounders
        self.shapes = shapes
        self.prior = prior
        self.has_counts = has_counts

    def is_used(self, sample: Sample):
        raise NotImplementedError

    def __call__(self, sample: Sample, caching: bool = True) -> float:
        """Compute the likelihood of all sites. The likelihood is defined as a mixture of areal and confounding effects.
            Args:
                sample: A Sample object consisting of clusters and weights
            Returns:
                The joint likelihood of the current sample
        """

        # Sum up log-likelihood of each mixture component:
        log_lh = 0.0

        log_lh += self.compute_lh_clusters(sample, caching=caching)
        for i, conf in enumerate(self.confounders, start=1):
            log_lh += self.compute_lh_confounder(sample, conf, caching=caching)

        return log_lh

    def compute_lh_clusters(self, sample: Sample, caching=True) -> float:
        """Compute the log-likelihood of the observations that are assigned to cluster
        effects in the current sample."""
        raise NotImplementedError

    def compute_lh_confounder(self, sample: Sample, conf: str, caching=True) -> float:
        """Compute the log-likelihood of the observations that are assigned to confounder
        `conf` in the current sample."""
        raise NotImplementedError

    def get_ft_cache(self, cache: ModelCache) -> GenericTypeCache:
        raise NotImplementedError

    def pointwise_likelihood(
        self,
        model,
        sample: Sample,
        caching=True
    ) -> NDArray[float]:  # shape: (n_objects, n_feature, n_components)
        """Update the likelihood values for each of the mixture components"""
        CHECK_CACHING = False

        lh_cache = self.get_ft_cache(sample.cache).component_likelihoods


        if caching and not lh_cache.is_outdated():
            if CHECK_CACHING:
                assert np.all(lh_cache.value == self.pointwise_likelihood(model, sample, caching=False))
            return lh_cache.value

        with lh_cache.edit() as component_likelihood:
            changed_clusters = lh_cache.what_changed(input_key=['clusters', 'clusters_sufficient_stats'],
                                                     caching=caching)
            if len(changed_clusters) > 0:
                # Update component likelihood for cluster effects:
                self.compute_pointwise_cluster_likelihood(
                    sample=sample,
                    changed_clusters=changed_clusters,
                    out=component_likelihood[..., 0],
                )

            # Update component likelihood for confounding effects:
            for i, conf in enumerate(self.confounders.values(), start=1):
                changed_groups = lh_cache.what_changed(input_key=f'{conf.name}_sufficient_stats', caching=caching)
                if len(changed_groups) == 0:
                    continue

                self.compute_pointwise_confounder_likelihood(
                    confounder=conf,
                    sample=sample,
                    changed_groups=changed_groups,
                    out=component_likelihood[..., i],
                )

            component_likelihood[self.na_values()] = 1.

        if caching and CHECK_CACHING:
            cached = np.copy(lh_cache.value)
            recomputed = self.pointwise_likelihood(model, sample, caching=False)
            assert np.allclose(cached, recomputed)

        return lh_cache.value

    def compute_pointwise_cluster_likelihood(
        self,
        sample: Sample,
        changed_clusters: list[int],
        out: NDArray[float],
    ):
        raise NotImplementedError

    def compute_pointwise_confounder_likelihood(
        self,
        confounder: Confounder,
        sample: Sample,
        changed_groups: list[int],
        out: NDArray[float],
    ):
        raise NotImplementedError

    def pointwise_conditional_cluster_lh(
        self,
        sample: Sample,
        i_cluster: int,
        available: NDArray[bool]
    ) -> NDArray[float]:
        raise NotImplementedError

    def na_values(self):
        raise NotImplementedError


class LikelihoodCategorical(LikelihoodGenericType):

    feature_type = FeatureType.categorical

    def __call__(self, sample: Sample, caching: bool = True):
        if not caching:
            recalculate_feature_counts(self.features.categorical.values, sample)
        return super().__call__(sample, caching=caching)

    def is_used(self, sample: Sample):
        return sample.categorical is not None

    def get_ft_cache(self, cache: ModelCache):
        return cache.categorical

    def compute_lh_clusters(self, sample: Sample, caching=True) -> float:
        """Compute the log-likelihood of the observations that are assigned to cluster
        effects in the current sample.
        This implementation of the likelihood integrates out the cluster-preference, i.e.
        probability vectors representing the distribution within clusters:
        P( X | c ) = ∫ P( X | γ ) P( γ | c ) dγ
        X... categorical observations assigned to this cluster
        γ... Cluster preference (probability vectors)
        c... prior concentration (parameter of a dirichlet-distribution)
        """

        cache = sample.cache.categorical.group_likelihoods['clusters']
        feature_counts = sample.categorical.sufficient_statistics['clusters'].value

        with cache.edit() as lh:
            for i_cluster in cache.what_changed('sufficient_stats', caching=caching):
                lh[i_cluster] = dirichlet_categorical_logpdf(
                    counts=feature_counts[i_cluster],
                    a=self.prior.prior_cluster_effect.categorical.concentration_array,
                ).sum()
        return cache.value.sum()

    def compute_lh_confounder(self, sample: Sample, conf: str, caching=True) -> float:
        """Compute the log-likelihood of the observations that are assigned to confounder
        `conf` in the current sample."""

        cache = sample.cache.categorical.group_likelihoods[conf]
        feature_counts = sample.categorical.sufficient_statistics[conf].value

        with cache.edit() as lh:
            prior_concentration = self.prior.prior_confounding_effects[conf].categorical.concentration_array(sample)
            for i_g in cache.what_changed('sufficient_stats', caching=caching):
                lh[i_g] = dirichlet_categorical_logpdf(
                    counts=feature_counts[i_g],
                    a=prior_concentration[i_g],
                ).sum()

        return cache.value.sum()

    def compute_pointwise_cluster_likelihood(
        self,
        sample: Sample,
        changed_clusters: list[int],
        out: NDArray[float],
    ) -> NDArray[float]:
        """Compute the likelihood distribution at each observation given all
        observations in the data-set.

        P( x' | X ) = ∫ P( x' | γ ) P( γ | X, c ) dγ
        x'... the categorical observation for which the likelihood is evaluated
        X...  the other categorical observations assigned to this cluster
        γ...  Cluster preference (probability vectors)
        c...  prior concentration (parameter of a dirichlet-distribution)
        """

        # The expected cluster effect is given by the normalized posterior counts
        cluster_effect_counts = (  # feature counts + prior counts
                sample.categorical.sufficient_statistics['clusters'].value +
                self.prior.prior_cluster_effect.categorical.concentration_array
        )

        cluster_effect = normalize(cluster_effect_counts, axis=-1)

        return compute_component_likelihood(
            features=self.features.categorical.values,
            probs=cluster_effect,
            groups=sample.clusters.value,
            changed_groups=changed_clusters,
            out=out,
        )

    def compute_pointwise_confounder_likelihood(
        self,
        confounder: Confounder,
        sample: Sample,
        changed_groups: list[int],
        out: NDArray[float],
    ) -> NDArray[float]:
        groups = confounder.group_assignment
        # The expected confounding effect is given by the normalized posterior counts
        prior = self.prior.prior_confounding_effects[confounder.name].categorical

        conf_effect_counts = (  # feature counts + prior counts
                sample.categorical.sufficient_statistics[confounder.name].value +
                prior.concentration_array(sample)
        )
        conf_effect = normalize(conf_effect_counts, axis=-1)

        return compute_component_likelihood(
            features=self.features.categorical.values,
            probs=conf_effect,
            groups=groups,
            changed_groups=changed_groups,
            out=out,
        )

    def pointwise_conditional_cluster_lh(
        self,
        sample: Sample,
        i_cluster: int,
        available: NDArray[bool]
    ) -> NDArray[float]:
        prior_counts = self.prior.prior_cluster_effect.categorical.concentration_array
        feature_counts = sample.categorical.sufficient_statistics['clusters'].value[[i_cluster]]
        p = normalize(prior_counts + feature_counts, axis=-1)
        return inner1d(self.features.categorical.values[available], p)

    def na_values(self):
        return self.features.categorical.na_values


class LikelihoodGaussian(LikelihoodGenericType):

    feature_type = FeatureType.gaussian

    def is_used(self, sample: Sample):
        return sample.gaussian is not None

    def get_ft_cache(self, cache: ModelCache):
        return cache.gaussian

    def compute_lh_clusters(self, sample: Sample, caching=True) -> float:
        """Compute the log-likelihood of the observations that are assigned to cluster
        effects in the current sample."""

        cache = sample.cache.gaussian.group_likelihoods['clusters']

        with cache.edit() as lh:
            for i_cluster in cache.what_changed('sufficient_stats', caching=caching):
                lh[i_cluster] = 0
                cluster = sample.clusters.value[i_cluster]

                # todo: Consider vectorization
                for f in range(self.features.gaussian.n_features):
                    x = self.features.gaussian.values[cluster][:, f]
                    mu_0 = self.prior.prior_cluster_effect.gaussian.mean.mu_0_array[f]
                    sigma_0 = self.prior.prior_cluster_effect.gaussian.mean.sigma_0_array[f]

                    # use the maximum likelihood value for sigma_fixed
                    sigma_fixed = np.nanstd(x)
                    lh[i_cluster] += gaussian_mu_marginalised_logpdf(
                        x=x, mu_0=mu_0, sigma_0=sigma_0, sigma_fixed=sigma_fixed,
                        in_component=sample.gaussian.source.value[cluster, f, 0],
                    )

        return cache.value.sum()

    def compute_lh_confounder(self, sample: Sample, conf: str, caching=True) -> float:
        """Compute the log-likelihood of the observations that are assigned to confounder
        `conf` in the current sample."""

        i_component = sample.model_shapes.get_component_index(conf)
        cache = sample.cache.gaussian.group_likelihoods[conf]
        groups = sample.confounders[conf].group_assignment

        mu_0 = self.prior.prior_confounding_effects[conf].gaussian.mean.mu_0_array
        sigma_0 = self.prior.prior_confounding_effects[conf].gaussian.mean.sigma_0_array

        with cache.edit() as lh:
            for i_g in cache.what_changed('sufficient_stats', caching=caching):
                lh[i_g] = 0
                g = groups[i_g]

                # todo: Consider vectorization
                for i_f in range(self.features.gaussian.n_features):
                    x = self.features.gaussian.values[g, i_f]

                    # use the maximum likelihood value for sigma_fixed
                    sigma_fixed = np.nanstd(x)
                    lh[i_g] += gaussian_mu_marginalised_logpdf(
                        x=x, mu_0=mu_0[i_g, i_f], sigma_0=sigma_0[i_g, i_f], sigma_fixed=sigma_fixed,
                        in_component=sample.gaussian.source.value[g, i_f, i_component],
                    )

        return cache.value.sum()

    def compute_pointwise_cluster_likelihood(
        self,
        sample: Sample,
        changed_clusters: list[int],
        out: NDArray[float],
    ):
        mu_0 = self.prior.prior_cluster_effect.gaussian.mean.mu_0_array
        sigma_0 = self.prior.prior_cluster_effect.gaussian.mean.sigma_0_array

        groups = sample.clusters.value
        features = self.features.gaussian.values

        out[~groups.any(axis=0), :] = 0.

        for i_g in changed_clusters:
            g = groups[i_g]

            for i_f in range(self.shapes.n_features_gaussian):
                f = features[g, i_f]
                sigma_fixed = np.nanstd(f)
                out[g, i_f] = np.exp(gaussian_posterior_predictive_logpdf(
                    x_new=f, x=f, sigma=sigma_fixed, mu_0=mu_0[i_f], sigma_0=sigma_0[i_f],
                    in_component=sample.gaussian.source.value[g, i_f, 0],
                ))

    def compute_pointwise_confounder_likelihood(
        self,
        confounder: Confounder,
        sample: Sample,
        changed_groups: list[int],
        out: NDArray[float],
    ) -> NDArray[float]:

        i_component = sample.model_shapes.get_component_index(confounder.name)
        mu_0 = self.prior.prior_confounding_effects[confounder.name].gaussian.mean.mu_0_array
        sigma_0 = self.prior.prior_confounding_effects[confounder.name].gaussian.mean.sigma_0_array

        groups = confounder.group_assignment
        features = self.features.gaussian.values

        out[~groups.any(axis=0), :] = 0.

        for i_g in changed_groups:
            g = groups[i_g]
            out[g, :] = np.exp(gaussian_posterior_predictive_logpdf(
                    x_new=features[g],
                    x=features[g],
                    sigma=np.nanstd(features[g], axis=0),
                    mu_0=mu_0[i_g, :],
                    sigma_0=sigma_0[i_g, :],
                    in_component=sample.gaussian.source.value[g, :, i_component],
            ))
        return out

    def pointwise_conditional_cluster_lh(
        self,
        sample: Sample,
        i_cluster: int,
        available: NDArray[bool]
    ) -> NDArray[float]:

        mu_0 = self.prior.prior_cluster_effect.gaussian.mean.mu_0_array
        sigma_0 = self.prior.prior_cluster_effect.gaussian.mean.sigma_0_array

        cluster = sample.clusters.value[i_cluster]
        features_candidates = self.features.gaussian.values[available]
        features_cluster = self.features.gaussian.values[cluster]
        source_cluster = sample.gaussian.source.value[cluster, :, 0]

        sigma_fixed = np.nanstd(features_cluster, axis=0)
        condition_lh = np.exp(gaussian_posterior_predictive_logpdf(
            x_new=features_candidates, x=features_cluster, sigma=sigma_fixed,
            mu_0=mu_0, sigma_0=sigma_0, in_component=source_cluster,
        ))
        # assert np.allclose(_condition_lh, condition_lh)

        return condition_lh

    def pointwise_conditional_likelihood_2(
        self,
        sample: Sample,
        out: NDArray[float],  # shape: (n_objects, n_features)
        condition_on: NDArray[bool] | None = None,  # shape: (n_objects,)
        evaluate_on: NDArray[bool] | None = None,  # shape: (n_objects,)
    ):
        if condition_on is None:
            condition_on = np.ones(sample.n_objects, dtype=bool)
        if evaluate_on is None:
            evaluate_on = np.ones(sample.n_objects, dtype=bool)

        mu_0 = self.prior.prior_cluster_effect.gaussian.mean.mu_0_array
        sigma_0 = self.prior.prior_cluster_effect.gaussian.mean.sigma_0_array

        groups = sample.clusters.value
        features = self.features.gaussian.values

        out[~groups.any(axis=0)[evaluate_on], :] = 0.

        for i_f in range(self.shapes.n_features_gaussian):
            f = features[:, i_f]
            sigma_fixed = np.nanstd(f[condition_on])
            out[:, i_f] = np.exp(gaussian_posterior_predictive_logpdf(
                x_new=f[evaluate_on], x=f[condition_on],
                sigma=sigma_fixed, mu_0=mu_0[i_f], sigma_0=sigma_0[i_f],
                in_component=sample.gaussian.source.value[condition_on, i_f, 0],
            ))

    def pointwise_conditional_cluster_likelihood_2(
        self,
        sample: Sample,
        out: NDArray[float],  # shape: (n_objects, n_features)
        changed_clusters: list[int] | None = None,
        condition_on: NDArray[bool] | None = None,    # shape: (n_objects,)
        evaluate_on: NDArray[bool] | None = None,     # shape: (n_objects,)
    ):
        if changed_clusters is None:
            changed_clusters = range(sample.n_clusters)
        if condition_on is None:
            condition_on = np.ones(sample.n_objects, dtype=bool)
        if evaluate_on is None:
            evaluate_on = np.ones(sample.n_objects, dtype=bool)

        mu_0 = self.prior.prior_cluster_effect.gaussian.mean.mu_0_array
        sigma_0 = self.prior.prior_cluster_effect.gaussian.mean.sigma_0_array

        groups = sample.clusters.value
        features = self.features.gaussian.values

        out[~groups.any(axis=0)[evaluate_on], :] = 0.

        for i_g in changed_clusters:
            g = groups[i_g]
            if not np.any(g & evaluate_on):
                continue

            for i_f in range(self.shapes.n_features_gaussian):
                f = features[:, i_f]
                f_cond = f[g & condition_on]
                f_eval = f[g & evaluate_on]
                sigma_fixed = np.nanstd(f_cond)
                out[g[evaluate_on], i_f] = np.exp(gaussian_posterior_predictive_logpdf(
                    x_new=f_eval, x=f_cond, sigma=sigma_fixed, mu_0=mu_0[i_f], sigma_0=sigma_0[i_f],
                    in_component=sample.gaussian.source.value[g & condition_on, i_f, 0],
                ))

    def pointwise_conditional_confounder_likelihood_2(
        self,
        confounder: Confounder,
        sample: Sample,
        out: NDArray[float],
        changed_groups: list[int] | None = None,
        condition_on: NDArray[bool] | None = None,    # shape: (n_objects,)
        evaluate_on: NDArray[bool] | None = None,     # shape: (n_objects,)
    ) -> NDArray[float]:
        if changed_groups is None:
            changed_groups = np.arange(confounder.n_groups)
        if condition_on is None:
            condition_on = np.ones(sample.n_objects, dtype=bool)
        if evaluate_on is None:
            evaluate_on = np.ones(sample.n_objects, dtype=bool)

        i_component = sample.model_shapes.get_component_index(confounder.name)
        mu_0 = self.prior.prior_confounding_effects[confounder.name].gaussian.mean.mu_0_array
        sigma_0 = self.prior.prior_confounding_effects[confounder.name].gaussian.mean.sigma_0_array

        groups = confounder.group_assignment
        features = self.features.gaussian.values

        out[~groups.any(axis=0)[evaluate_on], :] = 0.

        for i_g in changed_groups:
            g = groups[i_g]
            if not np.any(g & evaluate_on):
                continue

            f_cond = features[g & condition_on]
            f_eval = features[g & evaluate_on]
            out[g[evaluate_on], :] = np.exp(gaussian_posterior_predictive_logpdf(
                x_new=f_eval,
                x=f_cond,
                sigma=np.nanstd(f_cond, axis=0),
                mu_0=mu_0[i_g, :],
                sigma_0=sigma_0[i_g, :],
                in_component=sample.gaussian.source.value[g & condition_on, :, i_component],
            ))

        return out

    def na_values(self):
        return self.features.gaussian.na_values


class LikelihoodPoisson(LikelihoodGenericType):

    feature_type = FeatureType.poisson

    def is_used(self, sample: Sample):
        return sample.poisson is not None

    def get_ft_cache(self, cache: ModelCache):
        return cache.poisson

    def compute_lh_clusters(self, sample: Sample, caching=True) -> float:
        """Compute the log-likelihood of the observations that are assigned to cluster
        effects in the current sample."""

        cache = sample.cache.poisson.group_likelihoods['clusters']

        with cache.edit() as lh:

            for i_cluster in cache.what_changed('sufficient_stats', caching=caching):
                lh[i_cluster] = 0
                cluster = sample.clusters.value[i_cluster]

                # todo: Consider vectorization
                for f in range(self.features.poisson.n_features):
                    # TODO only use samples that have cluster as a source component
                    x = self.features.poisson.values[cluster, f]
                    alpha_0 = self.prior.prior_cluster_effect.poisson.alpha_0_array[f]
                    beta_0 = self.prior.prior_cluster_effect.poisson.beta_0_array[f]
                    lh[i_cluster] += lh_poisson_lambda_marginalised_logpdf(
                        x=x, alpha_0=alpha_0, beta_0=beta_0,
                        in_component=sample.poisson.source.value[cluster, f, 0],
                    )

        return cache.value.sum()

    def compute_lh_confounder(self, sample: Sample, conf: str, caching=True) -> float:
        """Compute the log-likelihood of the observations that are assigned to confounder
        `conf` in the current sample."""

        cache = sample.cache.poisson.group_likelihoods[conf]
        i_component = sample.model_shapes.get_component_index(conf)

        with cache.edit() as lh:
            for i_g in cache.what_changed('sufficient_stats', caching=caching):
                lh[i_g] = 0
                group = sample.confounders[conf].group_assignment[i_g]

                # todo: Consider vectorization
                for f in range(self.features.poisson.n_features):
                    x = self.features.poisson.values[group, f]
                    alpha_0 = self.prior.prior_confounding_effects[conf].poisson.alpha_0_array[i_g][f]
                    beta_0 = self.prior.prior_confounding_effects[conf].poisson.beta_0_array[i_g][f]

                    lh[i_g] += lh_poisson_lambda_marginalised_logpdf(
                        x=x, alpha_0=alpha_0, beta_0=beta_0,
                        in_component=sample.poisson.source.value[group, f, i_component],
                    )

        return cache.value.sum()

    def compute_pointwise_cluster_likelihood(
            self,
            sample: Sample,
            changed_clusters: list[int],
            out: NDArray[float],
    ):
        alpha_0 = self.prior.prior_cluster_effect.poisson.alpha_0_array
        beta_0 = self.prior.prior_cluster_effect.poisson.beta_0_array
        groups = sample.clusters.value
        features = self.features.poisson.values

        out[~groups.any(axis=0), :] = 0.

        for i in changed_clusters:
            g = groups[i]
            f_g = features[g, :]

            for j in range(f_g.shape[1]):

                f = f_g[:, j]
                n = len(f) - 1
                alpha_1 = alpha_0[j] + f.sum() - f
                beta_1 = beta_0[j] + n
                out[g, j] = nbinom.pmf(f, alpha_1, beta_1 / (1 + beta_1))

    def compute_pointwise_confounder_likelihood(
            self,
            confounder: Confounder,
            sample: Sample,
            changed_groups: list[int],
            out: NDArray[float],
    ) -> NDArray[float]:

        alpha_0 = self.prior.prior_confounding_effects[confounder.name].poisson.alpha_0_array
        beta_0 = self.prior.prior_confounding_effects[confounder.name].poisson.beta_0_array
        groups = confounder.group_assignment
        features = self.features.poisson.values

        out[~groups.any(axis=0), :] = 0.

        for i in changed_groups:
            g = groups[i]
            f_g = features[g, :]

            for j in range(f_g.shape[1]):
                f = f_g[:, j]
                n = len(f) - 1
                alpha_1 = alpha_0[i, j] + f.sum() - f
                beta_1 = beta_0[i, j] + n

                out[g, j] = nbinom.pmf(f, alpha_1, beta_1 / (1 + beta_1))

        return out

    def pointwise_conditional_cluster_lh(
        self,
        sample: Sample,
        i_cluster: int,
        available: NDArray[bool]
    ) -> NDArray[float]:
        alpha_0 = self.prior.prior_cluster_effect.poisson.alpha_0_array
        beta_0 = self.prior.prior_cluster_effect.poisson.beta_0_array

        features_candidates = self.features.poisson.values[available]
        features_cluster = self.features.poisson.values[sample.clusters.value[i_cluster]]
        condition_lh = np.zeros(features_candidates.shape)

        for j in range(features_cluster.shape[1]):
            f_c = features_cluster[:, j]
            n = len(f_c)
            alpha_1 = alpha_0[j] + f_c.sum()
            beta_1 = beta_0[j] + n
            condition_lh[:, j] = nbinom.pmf(features_candidates[:, j], alpha_1, beta_1 / (1 + beta_1))

        return condition_lh

    def na_values(self):
        return self.features.poisson.na_values


class LikelihoodLogitNormal(LikelihoodGenericType):

    feature_type = FeatureType.logitnormal

    def is_used(self, sample: Sample):
        return sample.logitnormal is not None

    def get_ft_cache(self, cache: ModelCache):
        return cache.logitnormal

    def compute_lh_clusters(self, sample: Sample, caching=True) -> float:
        """Compute the log-likelihood of the observations that are assigned to cluster
        effects in the current sample."""

        cache = sample.cache.logitnormal.group_likelihoods['clusters']

        with cache.edit() as lh:
            for i_cluster in cache.what_changed('sufficient_stats', caching=caching):
                lh[i_cluster] = 0
                cluster = sample.clusters.value[i_cluster]

                # todo: Consider vectorization
                for f in range(self.features.logitnormal.n_features):
                    # TODO only use samples that have cluster as a source component
                    x = self.features.logitnormal.values[cluster, f]
                    mu_0 = self.prior.prior_cluster_effect.logitnormal.mean.mu_0_array[f]
                    sigma_0 = self.prior.prior_cluster_effect.logitnormal.mean.sigma_0_array[f]
                    # use the maximum likelihood value for sigma_fixed
                    sigma_fixed = np.nanstd(x)
                    lh[i_cluster] += gaussian_mu_marginalised_logpdf(
                        x=x, mu_0=mu_0, sigma_0=sigma_0, sigma_fixed=sigma_fixed,
                        in_component=sample.logitnormal.source.value[cluster, f, 0],
                    )

        return cache.value.sum()

    def compute_lh_confounder(self, sample: Sample, conf: str, caching=True) -> float:
        """Compute the log-likelihood of the observations that are assigned to confounder
        `conf` in the current sample."""

        cache = sample.cache.logitnormal.group_likelihoods[conf]
        i_component = sample.model_shapes.get_component_index(conf)

        with cache.edit() as lh:
            for i_g in cache.what_changed('sufficient_stats', caching=caching):
                lh[i_g] = 0
                group = sample.confounders[conf].group_assignment[i_g]

                # todo: Consider vectorization
                for f in range(self.features.logitnormal.n_features):
                    x = self.features.logitnormal.values[group, f]
                    mu_0 = self.prior.prior_confounding_effects[conf].logitnormal.mean.mu_0_array[i_g][f]
                    sigma_0 = self.prior.prior_confounding_effects[conf].logitnormal.mean.sigma_0_array[i_g][f]

                    # use the maximum likelihood value for sigma_fixed
                    sigma_fixed = np.nanstd(x)
                    lh[i_g] += gaussian_mu_marginalised_logpdf(
                        x=x, mu_0=mu_0, sigma_0=sigma_0, sigma_fixed=sigma_fixed,
                        in_component=sample.logitnormal.source.value[group, f, i_component],
                    )

        return cache.value.sum()

    def compute_pointwise_cluster_likelihood(
        self,
        sample: Sample,
        changed_clusters: list[int],
        out: NDArray[float],
    ) -> NDArray[float]:

        mu_0 = self.prior.prior_cluster_effect.logitnormal.mean.mu_0_array
        sigma_0 = self.prior.prior_cluster_effect.logitnormal.mean.sigma_0_array
        # Jeffreys prior for variance

        groups = sample.clusters.value
        features = self.features.logitnormal.values

        out[~groups.any(axis=0), :] = 0.

        for i in changed_clusters:
            g = groups[i]
            f_g = features[g, :]

            for j in range(f_g.shape[1]):
                f = f_g[:, j]
                # todo: implement likelihood
                out[g, j] = np.repeat(0.5, len(f))

        return out

    def compute_pointwise_confounder_likelihood(
            self,
            confounder: Confounder,
            sample: Sample,
            changed_groups: list[int],
            out: NDArray[float],
    ):

        mu_0 = self.prior.prior_confounding_effects[confounder.name].logitnormal.mean.mu_0_array
        sigma_0 = self.prior.prior_confounding_effects[confounder.name].logitnormal.mean.sigma_0_array

        groups = confounder.group_assignment
        features = self.features.logitnormal.values

        out[~groups.any(axis=0), :] = 0.

        for i in changed_groups:
            g = groups[i]
            f_g = features[g, :]

            for j in range(f_g.shape[1]):
                f = f_g[:, j]
                # todo: implement likelihood
                out[g, j] = np.repeat(0.5, len(f))

    def pointwise_conditional_cluster_lh(
        self,
        sample: Sample,
        i_cluster: int,
        available: NDArray[bool]
    ) -> NDArray[float]:

        mu_0 = self.prior.prior_cluster_effect.logitnormal.mean.mu_0_array
        sigma_0 = self.prior.prior_cluster_effect.logitnormal.mean.sigma_0_array

        features_candidates = self.features.logitnormal.values[available]
        features_cluster = self.features.logitnormal.values[sample.clusters.value[i_cluster]]
        condition_lh = np.zeros(features_candidates.shape)

        # todo: implement likelihood
        for j in range(features_cluster.shape[1]):
            f_c = features_cluster[:, j]
            condition_lh[:, j] = np.repeat(0.5, len(features_candidates[:, j]))

        return condition_lh

    def na_values(self):
        return (self.features.logitnormal.na_values)


def compute_component_likelihood(
    features: NDArray[bool],    # shape: (n_objects, n_features, n_states)
    probs: NDArray[float],      # shape: (n_groups, n_features, n_states)
    groups: NDArray[bool],      # shape: (n_groups, n_objects)
    changed_groups: list[int],
    out: NDArray[float]         # shape: (n_objects, n_features)
) -> NDArray[float]:            # shape: (n_objects, n_features)

    out[~groups.any(axis=0), :] = 0.
    for i in changed_groups:
        g = groups[i]
        f_g = features[g, :, :]
        p_g = probs[i, :, :]
        out[g, :] = np.einsum('ijk,jk->ij', f_g, p_g)
        assert np.allclose(out[g, :], np.sum(f_g * p_g[np.newaxis, ...], axis=-1))
        assert np.allclose(out[g, :], inner1d(f_g, p_g[np.newaxis, ...]))

    return out




def update_categorical_weights(sample: Sample, caching=True) -> NDArray[float]:
    """Compute the normalized weights of each component at each site for categorical features
    Args:
        sample: the current MCMC sample.
        caching: ignore cache if set to false.
    Returns:
        np.array: normalized weights of each component at each site for categorical features
            shape: (n_objects, n_categorical_features, 1 + n_confounders)
    """

    cache = sample.cache.categorical.weights_normalized

    if (not caching) or cache.is_outdated():
        w_normed = normalize_weights(sample.categorical.weights.value, sample.cache.categorical.has_components.value)
        cache.update_value(w_normed)

    return cache.value


def update_gaussian_weights(sample: Sample, caching=True) -> NDArray[float]:
    """Compute the normalized weights of each component at each site for gaussian features
    Args:
        sample: the current MCMC sample.
        caching: ignore cache if set to false.
    Returns:
        np.array: normalized weights of each component at each site for gaussian features
            shape: (n_objects, n_gaussian_features, 1 + n_confounders)
    """

    cache = sample.cache.gaussian.weights_normalized

    if (not caching) or cache.is_outdated():
        w_normed = normalize_weights(sample.gaussian.weights.value, sample.cache.gaussian.has_components.value)
        cache.update_value(w_normed)

    return cache.value


def update_poisson_weights(sample: Sample, caching=True) -> NDArray[float]:
    """Compute the normalized weights of each component at each site for poisson features
    Args:
        sample: the current MCMC sample.
        caching: ignore cache if set to false.
    Returns:
        np.array: normalized weights of each component at each site for poisson features
            shape: (n_objects, n_poisson_features, 1 + n_confounders)
    """

    cache = sample.cache.poisson.weights_normalized

    if (not caching) or cache.is_outdated():
        w_normed = normalize_weights(sample.poisson.weights.value, sample.cache.poisson.has_components.value)
        cache.update_value(w_normed)

    return cache.value


def update_logitnormal_weights(sample: Sample, caching=True) -> NDArray[float]:
    """Compute the normalized weights of each component at each site for logitnormal features
    Args:
        sample: the current MCMC sample.
        caching: ignore cache if set to false.
    Returns:
        np.array: normalized weights of each component at each site for logitnormal features
            shape: (n_objects, n_logitnormal_features, 1 + n_confounders)
    """

    cache = sample.cache.logitnormal.weights_normalized

    if (not caching) or cache.is_outdated():
        w_normed = normalize_weights(sample.logitnormal.weights.value, sample.cache.logitnormal.has_components.value)
        cache.update_value(w_normed)

    return cache.value


update_weights = {
    FeatureType.categorical: update_categorical_weights,
    FeatureType.gaussian: update_gaussian_weights,
    FeatureType.poisson: update_poisson_weights,
    FeatureType.logitnormal: update_logitnormal_weights,
}


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
