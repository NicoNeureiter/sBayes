from __future__ import annotations
from copy import deepcopy
import random
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, Any

import numpy as np
from numpy.typing import NDArray
import scipy.stats as stats
from scipy.stats import nbinom

from sbayes.load_data import Features, CategoricalFeatures, FeatureType
from sbayes.sampling.conditionals import conditional_effect_mean
from sbayes.sampling.counts import recalculate_feature_counts, update_feature_counts, update_sufficient_statistics
from sbayes.sampling.state import Sample
from sbayes.util import dirichlet_logpdf, normalize, get_neighbours, RND_SEED, inner1d
from sbayes.model import Model, normalize_weights
from sbayes.preprocessing import sample_categorical
from sbayes.config.config import OperatorsConfig
from sbayes.model.likelihood import update_weights


DEBUG = 1
RNG = np.random.default_rng(seed=RND_SEED)


class Operator(ABC):

    """MCMC-operator base class"""

    weight: float
    """The relative frequency of this operator."""

    accepts: int
    rejects: int
    """Number of proposals accepted/rejected during the MCMC run (for operator stats)."""

    step_times: list[float]
    """Logs of time spent per MCMC step (including posterior calculation, accept/reject, ...) for this operator."""

    step_sizes: list[float]
    STEP_SIZE_UNIT: str = ""
    """Logs of size of each MCMC step for this operator."""

    # CLASS CONSTANTS

    Q_GIBBS = -np.inf
    Q_BACK_GIBBS = 0
    """Fixed transition probabilities for Gibbs operators (ensuring acceptance)."""

    Q_REJECT = 0
    Q_BACK_REJECT = -np.inf
    """Fixed transition probabilities for directly rejecting MCMC steps."""

    def __init__(self, weight: float, **kwargs):
        self.weight = weight
        self.accepts: int = 0
        self.rejects: int = 0
        self.step_times = []
        self.step_sizes = []

    def function(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        # TODO: potentially do some book-keeping
        return self._propose(sample, **kwargs)

    @abstractmethod
    def _propose(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        """Propose a new state from the given one."""
        pass

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Unknown attribute `{key}` for class `{type(self)}`")

    def __setitem__(self, key: str, value: Any):
        if key == "weight":
            self.weight = value
        elif key == "name":
            self.name = value
        else:
            raise ValueError(f"Attribute `{key}` cannot be set in class `{type(self)}`")

    def register_accept(self, step_time: float, sample_old: Sample, sample_new: Sample):
        self.accepts += 1
        self.step_times.append(step_time)
        step_size = self.get_step_size(sample_old, sample_new)
        self.step_sizes.append(step_size)

    def register_reject(self, step_time: float):
        self.rejects += 1
        self.step_times.append(step_time)
        # self.step_sizes.append(0)

    @property
    def total(self):
        return self.accepts + self.rejects

    @property
    def acceptance_rate(self):
        return self.accepts / self.total

    @property
    def operator_name(self) -> str:
        return self.__class__.__name__

    def get_parameters(self) -> dict[str, Any]:
        """Mapping from parameter name to parameter setting for an operator."""
        return {}

    @staticmethod
    def format_parameters(params: dict[str, Any]) -> str:
        params_str = ', '.join(f"{k}={v}" for k, v in params.items())
        return "[" + params_str + "]"

    def get_parameters_string(self) -> str:
        """Print the parameters of the current operator in the syntax defined in
         self.format_parameters(params):
            [param1_name=param1_value, param2_name=param2_value,...]
        """
        params = self.get_parameters()
        return self.format_parameters(params)

    def get_step_size(self, sample_old: Sample, sample_new: Sample) -> float:
        return 0.0


class DirichletOperator(Operator, ABC):

    """Base class for operators modifying probability vectors using a dirichlet proposal."""

    @staticmethod
    def dirichlet_proposal(
        w: NDArray[float], step_precision: float
    ) -> tuple[NDArray[float], float, float]:
        """Proposal distribution for normalized probability vectors (summing to 1).

        Args:
            w: The weight vector, which is being resampled.
                Shape: (n_states, 1 + n_confounders)
            step_precision: precision parameter controlling how narrow/wide the proposal
                distribution is. Low precision -> wide, high precision -> narrow.

        Returns:
            The newly proposed weights w_new (same shape as w).
            The transition probability q.
            The back probability q_back
        """
        alpha = 1 + step_precision * w
        w_new = RNG.dirichlet(alpha)
        log_q = dirichlet_logpdf(w_new, alpha)

        alpha_back = 1 + step_precision * w_new
        log_q_back = dirichlet_logpdf(w, alpha_back)

        if not np.all(np.isfinite(w_new)):
            logging.warning(f"Dirichlet step resulted in NaN or Inf:")
            logging.warning(f"\tOld sample: {w}")
            logging.warning(f"\tstep_precision: {step_precision}")
            logging.warning(f"\tNew sample: {w_new}")
            return w, Operator.Q_REJECT, Operator.Q_BACK_REJECT

        return w_new, log_q, log_q_back


class AlterWeights(DirichletOperator):

    STEP_PRECISION = 15

    def _propose(self, sample: Sample, **kwargs):
        """Modifies one weight of one feature in the current sample
        Args:
            sample: The current sample with clusters and parameters
        Returns:
            Sample: The modified sample
        """
        sample_new = sample.copy()

        # Randomly choose one of the features
        f_id = RNG.choice(range(sample.n_features))

        # Randomly choose two weights that will be changed, leave the others untouched
        weights_to_alter = random.sample(range(sample.n_components), 2)

        # Get the current weights and normalize
        w_curr = sample.weights.value[f_id, weights_to_alter]

        # Transform the weights such that they sum to 1
        w_curr_t = w_curr / w_curr.sum()

        # Propose new sample
        w_new_t, log_q, log_q_back = self.dirichlet_proposal(
            w_curr_t, step_precision=self.STEP_PRECISION
        )

        # Transform back
        w_new = w_new_t * w_curr.sum()

        # Update
        with sample_new.weights.edit() as w:
            w[f_id, weights_to_alter] = w_new

        return sample_new, log_q, log_q_back

    def get_step_size(self, sample_old: Sample, sample_new: Sample) -> float:
        return np.abs(sample_old.weights.value - sample_new.weights.value).sum()

    STEP_SIZE_UNIT: str = "sum of absolute weight changes"


class ObjectSelector(Enum):
    """Different ways to select a subset of object to change in the source operator."""
    ALL = 0
    GROUPS = 1
    RANDOM_SUBSET = 2


class GibbsSampleSource(Operator):

    def __init__(
        self,
        weight: float,
        model_by_chain: list[Model],
        feature_type: str = "",
        as_gibbs: bool = True,
        sample_from_prior: bool = False,
        object_selector: ObjectSelector = ObjectSelector.GROUPS,
        max_size: int = 50,
        **kwargs,
    ):
        super().__init__(weight=weight, **kwargs)
        self.model_by_chain = model_by_chain
        self.as_gibbs = as_gibbs
        self.sample_from_prior = sample_from_prior
        self.object_selector = object_selector
        self.min_size = 10
        self.max_size = min(max_size, self.model_by_chain[0].shapes.n_objects)

        # If the min_size includes all the sites we don't need to subsample at all
        if self.model_by_chain[0].shapes.n_objects <= self.min_size:
            self.object_selector = ObjectSelector.ALL

    def _propose(
            self,
            sample: Sample,
            object_subset: slice | list[int] | NDArray[bool] = slice(None),
            **kwargs,
    ) -> tuple[Sample, float, float]:
        """Resample the observations to mixture components (their source).

        Args:
            sample: The current sample with clusters and parameters
            object_subset: A subset of sites to be updated

        Returns:
            The modified sample and forward and backward transition log-probabilities
        """

        log_q = log_q_back = 0
        sample_new = sample.copy()
        model = self.model_by_chain[sample.chain]

        for ft, ft_sample in sample.feature_type_samples.items():
            features = model.data.features.get_ft_features(ft).values
            na_features = model.data.features.get_ft_features(ft).na_values

            assert np.all(getattr(sample, ft).source.value[na_features] == 0)

            object_subset = self.select_object_subset(sample)

            if self.sample_from_prior:
                update_ft_weights = update_weights[ft]
                p = update_ft_weights(sample)[object_subset]
            else:
                p = self.calculate_source_posterior(sample=sample, feature_type=ft, object_subset=object_subset)

            # Sample the new source assignments
            with ft_sample.source.edit() as source:
                source[object_subset] = sample_categorical(p=p, binary_encoding=True)
                source[na_features] = False

            if ft == FeatureType.categorical:
                update_feature_counts(sample, sample_new, features, object_subset)

                if DEBUG:
                    verify_counts(sample_new, features)
            else:
                update_sufficient_statistics(sample, sample_new, features, object_subset)

            # Transition probability forward:
            log_q += np.log(p[getattr(sample_new, ft).source.value[object_subset]]).sum()

            assert np.all(getattr(sample_new, ft).source.value[na_features] == 0)
            assert np.all(getattr(sample_new, ft).source.value[~na_features].sum(axis=-1) == 1)

            # Transition probability backward:
            if self.sample_from_prior:
                p_back = p
            else:
                p_back = self.calculate_source_posterior(sample=sample_new, feature_type=ft, object_subset=object_subset)

            log_q_back += np.log(p_back[getattr(sample, ft).source.value[object_subset]]).sum()

        return sample_new, log_q, log_q_back

    def get_parameters(self) -> dict[str, Any]:
        return {
            "object_selector": self.object_selector.name,
            "max_step_size": self.max_size,
        }

    @staticmethod
    def random_subset(n: int, k: int, population: Sequence[int] = None) -> NDArray[bool]:  # shape: (n,)
        if population is None:
            population = n
        subset_idxs = RNG.choice(population, size=k, replace=False)
        subset = np.zeros(n, dtype=bool)
        subset[subset_idxs] = True
        return subset

    def select_object_subset(self, sample) -> NDArray[bool] | slice:  # shape: (n_objects,)
        """Choose a subset of objects for which the source will be resampled"""

        if self.object_selector is ObjectSelector.ALL:
            # Resample source of all objects (may lead to low acceptance rate)
            object_subset = slice(None)

        elif self.object_selector is ObjectSelector.GROUPS:
            groups = sample.groups_and_clusters()
            components = list(groups.keys())

            conf = RNG.choice(components)
            object_subset = RNG.choice(groups[conf])

            # object_subset = np.zeros(sample.n_objects, dtype=bool)
            # while np.count_nonzero(object_subset) < self.min_size:
            #     # Select a random group/cluster in a random component to resample.
            #     conf = RNG.choice(components)
            #     object_subset |= RNG.choice(groups[conf])

            # If the group is too large (would lead to low acceptance rate), take a random subset
            if np.count_nonzero(object_subset) > self.max_size:
                object_subset = self.random_subset(n=sample.n_objects, k=self.max_size,
                                                   population=np.where(object_subset)[0])

        elif self.object_selector is ObjectSelector.RANDOM_SUBSET:
            # Choose a random subset for which the source is resampled
            object_subset = self.random_subset(n=sample.n_objects, k=self.max_size)
        else:
            raise ValueError(f"ObjectSelector '{self.object_selector}' not yet implemented.")

        return object_subset

    def calculate_source_posterior(
            self, sample: Sample, feature_type: str, object_subset: NDArray[bool] | list[int] | slice = slice(None),
    ) -> NDArray[float]:  # shape: (n_objects_in_subset, n_features, n_components)

        """Compute the posterior support for source assignments of every object and feature."""

        # 1. compute pointwise likelihood for each component
        model: Model = self.model_by_chain[sample.chain]
        lh_per_component = model.likelihood.feature_type_likelihoods[feature_type].pointwise_likelihood(model=model, sample=sample)

        # 2. multiply by weights and normalize over components to get the source posterior
        update_ft_weights = update_weights[feature_type]
        weights = update_ft_weights(sample)

        # 3. The posterior of the source for each observation is likelihood times prior
        # (normalized to sum up to one across source components):
        return normalize(lh_per_component[object_subset] * weights[object_subset], axis=-1)

    def get_step_size(self, sample_old: Sample, sample_new: Sample) -> float:
        count = 0
        for ft in sample_old.feature_type_samples:
                count += np.count_nonzero(sample_old.feature_type_samples[ft].source.value ^
                                          sample_new.feature_type_samples[ft].source.value)
        return count

    STEP_SIZE_UNIT: str = "observations reassigned"


class GibbsSampleWeights(Operator):

    def __init__(
        self,
        *args,
        model_by_chain: list[Model],
        sample_from_prior=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_by_chain = model_by_chain
        self.sample_from_prior = sample_from_prior

        self.last_accept_rate = 0

    def _propose(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:

        # The likelihood object contains relevant information on the areal and the confounding effect

        model = self.model_by_chain[sample.chain]
        feature_types = ["categorical", "gaussian", "poisson", "logitnormal"]

        for ft in feature_types:
            if getattr(sample, ft) is not None:
                na_features = model.likelihood.feature_type_likelihoods[ft].na_values()

                # Compute the old likelihood
                w_old = getattr(sample, ft).weights.value
                update_ft_weights = update_weights[ft]
                w_old_normalized = update_ft_weights(sample)

                log_lh_old = self.source_lh_by_feature(getattr(sample, ft).source.value,
                                                       w_old_normalized, na_features)
                log_prior_old = getattr(model.prior.prior_weights, ft).pointwise_prior(sample)

                # Resample the weights
                w_new, log_q, log_q_back = self.resample_weight_for_two_components(sample, ft)
                getattr(sample, ft).weights.set_value(w_new)

                # Compute new likelihood
                w_new_normalized = update_ft_weights(sample)
                log_lh_new = self.source_lh_by_feature(getattr(sample, ft).source.value, w_new_normalized, na_features)
                log_prior_new = getattr(model.prior.prior_weights, ft).pointwise_prior(sample)

                # Add the prior to get the weight posterior (for each feature)
                # log_prior_old = 0.0  # TODO add hyper prior on weights, when implemented
                # log_prior_new = 0.0  # TODO add hyper prior on weights, when implemented
                log_p_old = log_lh_old + log_prior_old
                log_p_new = log_lh_new + log_prior_new

                p_accept = np.exp(log_p_new - log_p_old + log_q_back - log_q)
                accept = RNG.random(p_accept.shape) < p_accept

                getattr(sample, ft).weights.set_value(np.where(accept[:, np.newaxis], w_new, w_old))

                assert ~np.any(np.isnan(getattr(sample, ft).weights.value))

                self.last_accept_rate = np.mean(accept)

        # We already accepted/rejected for each feature independently
        # The result should always be accepted in the MCMC
        return sample, self.Q_GIBBS, self.Q_BACK_GIBBS

    def resample_weight_for_two_components(self, sample: Sample, feature_type: str):

        model = self.model_by_chain[sample.chain]
        w = getattr(sample, feature_type).weights.value
        source = getattr(sample, feature_type).source.value
        has_components = getattr(sample.cache, feature_type).has_components.value
        prior_concentration = getattr(model.prior.prior_weights, feature_type).concentration_array

        # Fix weights for all but two random components
        i1, i2 = random.sample(range(sample.n_components), 2)

        # Select counts of the relevant languages
        has_both = np.logical_and(has_components[:, i1], has_components[:, i2])

        counts = (
                np.sum(source[has_both, :, :], axis=0)  # likelihood counts
                + prior_concentration   # prior counts
            )

        c1 = counts[..., i1]
        c2 = counts[..., i2]

        # Create proposal distribution based on the counts
        distr = stats.beta(1 + c2, 1 + c1)

        # Sample new relative weights
        a2 = distr.rvs()
        a1 = 1 - a2

        # Adapt w_new and renormalize
        w_02 = w[..., i1] + w[..., i2]
        w_new = w.copy()
        w_new[..., i1] = a1 * w_02
        w_new[..., i2] = a2 * w_02
        w_new = normalize(w_new, axis=-1)

        # Compute transition and back probability (for each feature)
        a2_old = w[..., i2] / w_02
        log_q = distr.logpdf(a2)
        log_q_back = distr.logpdf(a2_old)

        return w_new, log_q, log_q_back

    @staticmethod
    def source_lh_by_feature(source, weights, na_features):
        # multiply and sum to get likelihood per source observation
        p = np.sum(source * weights, axis=-1)
        p[na_features] = 1
        log_lh_per_observation = np.log(p)

        # sum over sites to obtain the total log-likelihood per feature
        return np.sum(log_lh_per_observation, axis=0)

    def get_step_size(self, sample_old: Sample, sample_new: Sample) -> float:
        return self.last_accept_rate
        # return np.abs(sample_old.weights.value - sample_new.weights.value).mean()

    STEP_SIZE_UNIT: str = "fraction of weights changed"
    # STEP_SIZE_UNIT: str = "average change per weight"


class ResampleSourceMode(str, Enum):
    GIBBS = "GIBBS"
    PRIOR = "PRIOR"
    UNIFORM = "UNIFORM"


class ClusterOperator(Operator):

    def __init__(
        self,
        *args,
        model_by_chain: list[Model],
        resample_source: bool,
        sample_from_prior: bool,
        p_grow: float = 0.5,
        n_changes: int = 1,
        resample_source_mode: ResampleSourceMode = ResampleSourceMode.GIBBS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_by_chain = model_by_chain
        self.resample_source = resample_source
        self.resample_source_mode = resample_source_mode
        self.sample_from_prior = sample_from_prior
        self.p_grow = p_grow
        self.n_changes = n_changes

    @staticmethod
    def available(sample: Sample, i_cluster: int) -> NDArray[bool]:
        return (~sample.clusters.any_cluster()) | sample.clusters.value[i_cluster]

    @staticmethod
    def get_removal_candidates(cluster: NDArray[bool]) -> NDArray[int]:
        """Finds objects which can be removed from the given zone.

        Args:
            cluster (np.array): The zone for which removal candidates are found.
                shape: (n_objects)
        Returns:
            Array of indices of objects that could be removed from the cluster.
        """
        return cluster.nonzero()[0]

    def propose_new_sources(
        self,
        sample_old: Sample,
        sample_new: Sample,
        i_cluster: int,
        object_subset: list[int] | NDArray[int],
        feature_type: FeatureType
    ) -> tuple[Sample, float, float]:
        n_features = sample_old[feature_type].n_features
        features = self.model_by_chain[sample_old.chain].data.features[feature_type].values
        na_features = self.model_by_chain[sample_old.chain].data.features[feature_type].na_values

        if self.resample_source_mode == ResampleSourceMode.GIBBS:
            sample_new, log_q, log_q_back = self.gibbs_sample_source(
                sample_new, sample_old, i_cluster, object_subset=object_subset,
                feature_type=feature_type
            )

        elif self.resample_source_mode == ResampleSourceMode.PRIOR:
            update_ft_weights = update_weights[feature_type]
            p = update_ft_weights(sample_new)[object_subset]
            p_back = update_ft_weights(sample_old)[object_subset]
            with getattr(sample_new, feature_type).source.edit() as source:
                source[object_subset, :, :] = sample_categorical(p, binary_encoding=True)
                source[na_features] = 0

                log_q = np.log(p[source[object_subset]]).sum()

            log_q_back = np.log(p_back[getattr(sample_old, feature_type).source.value[object_subset]]).sum()

            if feature_type == FeatureType.categorical:
                update_feature_counts(sample_old, sample_new, features, object_subset)
                if DEBUG:
                    verify_counts(sample_new, features)
            else:
                update_sufficient_statistics(sample_old, sample_new, features, object_subset)

        elif self.resample_source_mode == ResampleSourceMode.UNIFORM:
            has_components_new = getattr(sample_new.cache, feature_type).has_components.value
            p = normalize(
                np.tile(has_components_new[object_subset, None, :], (1, n_features, 1))
            )
            with sample_new.categorical.source.edit() as source:
                source[object_subset, :, :] = sample_categorical(
                    p, binary_encoding=True
                )
                source[na_features] = 0
                log_q = np.log(p[source[object_subset]]).sum()

            has_components_old = getattr(sample_old.cache, feature_type).has_components.value
            p_back = normalize(
                np.tile(has_components_old[object_subset, None, :], (1, n_features, 1))
            )
            log_q_back = np.log(p_back[getattr(sample_old, feature_type).source.value[object_subset]]).sum()

            if feature_type == FeatureType.categorical:
                update_feature_counts(sample_old, sample_new, features, object_subset)
                if DEBUG:
                    verify_counts(sample_new, features)
            else:
                update_sufficient_statistics(sample_old, sample_new, features, object_subset)
        else:
            raise ValueError(f"Invalid resample_source_mode `{self.resample_source_mode}` is not implemented.")

        return sample_new, log_q, log_q_back

    def gibbs_sample_source(
        self,
        sample_new: Sample,
        sample_old: Sample,
        i_cluster: int,
        feature_type: FeatureType,
        object_subset: slice | list[int] | NDArray[int] = slice(None)
    ) -> tuple[Sample, float, float]:
        """Resample the observations to mixture components (their source)."""
        model = self.model_by_chain[sample_old.chain]
        features = getattr(model.data.features, feature_type).values
        na_features = getattr(model.data.features, feature_type).na_values

        ft_sample_new = sample_new.feature_type_samples[feature_type]
        ft_sample_old = sample_old.feature_type_samples[feature_type]

        # Make sure object_subset is a boolean index array
        object_subset = np.isin(np.arange(sample_new.n_objects), object_subset)

        # todo: activate for other feature types (gaussian, poisson, logitnormal)
        lh_per_component = component_likelihood_given_unchanged(
            model, sample_new, object_subset, i_cluster, feature_type
        )

        update_ft_weights = update_weights[feature_type]
        if self.sample_from_prior:
            p = update_ft_weights(sample_new)[object_subset]
        else:
            w = update_ft_weights(sample_new)[object_subset]
            p = normalize(w * lh_per_component, axis=-1)
            p[na_features] = 1.0

        # Sample the new source assignments
        with ft_sample_new.source.edit() as source:
            source[object_subset] = sample_categorical(p=p, binary_encoding=True)
            source[na_features] = False

        if feature_type == FeatureType.categorical:
            update_feature_counts(sample_old, sample_new, features, slice(None))
            if DEBUG:
                verify_counts(sample_old, features)
                verify_counts(sample_new, features)
        else:
            update_sufficient_statistics(sample_old, sample_new, features, object_subset)

        # Transition probability forward:
        source_new = ft_sample_new.source.value[object_subset]
        log_q = np.log(p[source_new]).sum()

        # Transition probability backward:
        if self.sample_from_prior:
            p_back = update_ft_weights(sample_old)[object_subset]
        else:
            w = update_ft_weights(sample_old)[object_subset]
            p_back = normalize(w * lh_per_component, axis=-1)
            # p_back = self.calculate_source_posterior(sample_old, object_subset)

        source_old = ft_sample_old.source.value[object_subset]
        log_q_back = np.log(p_back[source_old]).sum()

        return sample_new, log_q, log_q_back

    def get_step_size(self, sample_old: Sample, sample_new: Sample) -> float:
        return np.count_nonzero(sample_old.clusters.value ^ sample_new.clusters.value)

    STEP_SIZE_UNIT: str = "objects reassigned"

    def get_parameters(self) -> dict[str, Any]:
        params = super().get_parameters()
        params["resample_source_mode"] = self.resample_source_mode.value
        return params


def component_likelihood_given_unchanged(
    model: Model,
    sample: Sample,
    object_subset: NDArray[bool],
    i_cluster: int,
    feature_type: FeatureType
) -> NDArray[float]:  # shape: (n_objects, n_feature, n_components)
    ft_sample = sample.feature_type_samples[feature_type]
    features = model.data.features.get_ft_features(feature_type)
    confounders = model.data.confounders
    cluster = sample.clusters.value[i_cluster]
    source = ft_sample.source.value
    subset_size = np.count_nonzero(object_subset)

    likelihoods = np.zeros((subset_size, ft_sample.n_features, sample.n_components))

    # todo: cluster likelihood given unchanged for poisson, gaussian and logitnormal
    if feature_type == FeatureType.categorical:
        likelihoods[..., 0] = cluster_likelihood_categorical_given_unchanged(cluster, features, object_subset, source)

        for i_conf, conf in enumerate(confounders, start=1):
            groups = confounders[conf].group_assignment

            features_conf = features.values * getattr(sample, feature_type).source.value[:, :, i_conf, None]
            changeable_counts = np.array([
                np.sum(features_conf[g & object_subset], axis=0)
                for g in groups
            ])

            unchangeable_feature_counts = ft_sample.sufficient_statistics[conf].value - changeable_counts
            prior_counts = getattr(model.prior.prior_confounding_effects[conf], feature_type).concentration_array(sample)
            conf_effect = normalize(unchangeable_feature_counts + prior_counts, axis=-1)

            # Calculate the likelihood of each observation in each group that is represented in object_subset
            subset_groups = groups[:, object_subset]
            group_in_subset = np.any(subset_groups, axis=1)
            features_subset = features.values[object_subset]
            for g, p_g in zip(subset_groups[group_in_subset], conf_effect[group_in_subset]):
                f_g = features_subset[g, :, :]
                likelihoods[g, :, i_conf] = np.einsum('ijk,jk->ij', f_g, p_g)

        # Fix likelihood of NA features to 1
        likelihoods[features.na_values[object_subset]] = 1.
    elif feature_type == FeatureType.gaussian:
        gaussian_lh = model.likelihood.gaussian
        gaussian_lh.pointwise_conditional_cluster_likelihood_2(
            sample=sample,
            out=likelihoods[..., 0],
            condition_on=object_subset,
        )
        for i, conf in enumerate(sample.confounders.values(), start=1):
            gaussian_lh.pointwise_conditional_confounder_likelihood_2(
                confounder=conf,
                sample=sample,
                out=likelihoods[..., i],
                condition_on=object_subset,
            )

        # Fix likelihood of NA features to 1
        likelihoods[features.na_values[object_subset]] = 1.
    else:
        raise NotImplementedError
        # likelihoods[:, :] = 1/likelihoods.shape[2]

        # Fix likelihood of NA features to 1
        likelihoods[features.na_values[object_subset]] = 1.

    assert np.all(likelihoods >= 0)
    return likelihoods


def cluster_likelihood_categorical_given_unchanged(
    cluster: NDArray[bool],  # (n_objects,)
    features: CategoricalFeatures,
    object_subset: NDArray[bool],  # (n_objects,)
    source: NDArray[bool],  # (n_objects, n_features, n_components)
) -> NDArray[float]:  # (n_objects, n_features)

    cluster_features = features.values * source[:, :, 0, None]
    feature_counts_c = np.sum(cluster_features[cluster & ~object_subset], axis=0)
    p = normalize(features.states + feature_counts_c, axis=-1)  # TODO: use prior counts, rather than 1+
    return np.sum(p[None, ...] * features.values[object_subset], axis=-1)


class AlterClusterGibbsish(ClusterOperator):
    def __init__(
        self,
        *args,
        adjacency_matrix,
        features: Features,
        consider_geo_prior: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adjacency_matrix = adjacency_matrix
        self.features = features
        self.consider_geo_prior = consider_geo_prior

    def _propose(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        log_q = 0.
        log_q_back = 0.

        for i in range(self.n_changes):
            if DEBUG and sample.categorical is not None:
                verify_counts(sample, self.model_by_chain[sample.chain].data.features.categorical.values)

            if random.random() < self.p_grow:
                sample_new, log_q_i, log_q_back_i = self.grow_cluster(sample)
                log_q_i += np.log(self.p_grow)
                log_q_back_i += np.log(1 - self.p_grow)
            else:
                sample_new, log_q_i, log_q_back_i = self.shrink_cluster(sample)
                log_q_i += np.log(1 - self.p_grow)
                log_q_back_i += np.log(self.p_grow)

            if log_q_back_i != self.Q_BACK_REJECT:
                sample = sample_new
                log_q += log_q_i
                log_q_back += log_q_back_i

        if DEBUG and sample.categorical is not None:
            verify_counts(sample, self.model_by_chain[sample.chain].data.features.categorical.values)
            verify_counts(sample_new, self.model_by_chain[sample.chain].data.features.categorical.values)

        return sample_new, log_q, log_q_back

    def compute_cluster_posterior(
        self,
        sample: Sample,
        i_cluster: int,
        candidates: NDArray[bool]
    ) -> NDArray[float]:  # shape: (n_available, )
        model = self.model_by_chain[sample.chain]

        cluster_posterior = np.zeros(sample.n_objects)
        cluster_posterior[candidates] = 1.

        if self.sample_from_prior:
            n_available = np.count_nonzero(candidates)
            return 0.5 * np.ones(n_available)

        for ft_likelihood in model.likelihood.feature_type_likelihoods:
            assert ft_likelihood.is_used(sample)
            cluster_lh_z = ft_likelihood.pointwise_conditional_cluster_lh(
                sample=sample, i_cluster=i_cluster, available=candidates
            )

            all_lh = deepcopy(
                ft_likelihood.pointwise_likelihood(model=model, sample=sample)[candidates, :]
            )
            all_lh[..., 0] = cluster_lh_z

            weights_z01 = self.compute_feature_weights_with_and_without(
                sample, candidates, feature_type=ft_likelihood.feature_type
            )
            feature_lh_z01 = inner1d(all_lh[np.newaxis, ...], weights_z01)
            marginal_lh_z01 = np.prod(feature_lh_z01, axis=-1)

            cluster_posterior[candidates] *= marginal_lh_z01[1] / (marginal_lh_z01[0] + marginal_lh_z01[1])

        if self.consider_geo_prior:
            cluster_posterior *= np.exp(model.prior.geo_prior.get_costs_per_object(sample, i_cluster)[candidates])

        return cluster_posterior

    @staticmethod
    def compute_feature_weights_with_and_without(
        sample: Sample,
        available: NDArray[bool],   # shape: (n_objects, )
        feature_type: FeatureType,
    ) -> NDArray[float]:            # shape: (2, n_objects, n_features, n_components)
        update_ft_weights = update_weights[feature_type]
        weights_current = update_ft_weights(sample, caching=True)[available]
        # weights = normalize_weights(sample.weights.value, has_components)

        has_components = deepcopy(sample.cache.feature_type_cache[feature_type].has_components.value[available, :])
        has_components[:, 0] = ~has_components[:, 0]
        weights_flipped = normalize_weights(sample.feature_type_samples[feature_type].weights.value, has_components)

        weights_z01 = np.empty((2, *weights_current.shape))
        weights_z01[1] = np.where(has_components[:, np.newaxis, [0]], weights_flipped, weights_current)
        weights_z01[0] = np.where(has_components[:, np.newaxis, [0]], weights_current, weights_flipped)

        return weights_z01

    @staticmethod
    def grow_candidates(sample: Sample) -> NDArray[bool]:
        return ~sample.clusters.any_cluster()

    @staticmethod
    def shrink_candidates(sample: Sample, i_cluster: int) -> NDArray[bool]:
        return sample.clusters.value[i_cluster]

    def grow_cluster(self, sample: Sample) -> tuple[Sample, float, float]:
        # Choose a cluster
        i_cluster = RNG.choice(range(sample.n_clusters))

        cluster = sample.clusters.value[i_cluster, :]

        # Load and precompute useful variables
        model = self.model_by_chain[sample.chain]
        available = self.available(sample, i_cluster)
        candidates = self.grow_candidates(sample)

        # If all the space is taken we can't grow
        if not np.any(candidates):
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        # If the cluster is already at max size, reject:
        if np.sum(cluster) == model.max_size:
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        # Otherwise create a new sample and continue with step:
        sample_new = sample.copy()

        # Assign probabilities for each unoccupied object
        # todo: change for gaussian, poisson and logitnormal features

        cluster_posterior = self.compute_cluster_posterior(sample, i_cluster, candidates)
        p_add = normalize(cluster_posterior)

        # # Draw new object according to posterior
        # object_new = RNG.choice(sample.n_objects, p=p_add, replace=False)
        # sample_new.clusters.add_object(i_cluster, object_new)
        #
        # # The removal probability of an inverse step
        # shrink_candidates = self.shrink_candidates(sample_new, i_cluster)
        # p_remove = normalize((1 - cluster_posterior) * shrink_candidates)

        # Draw new object according to posterior
        # n_add = min(self.n_changes, np.sum(candidates))
        n_add = min(1, np.sum(candidates))

        new_objects = RNG.choice(sample.n_objects, p=p_add, size=n_add, replace=False)
        for obj in new_objects:
            sample_new.clusters.add_object(i_cluster, obj)

        log_q_s = log_q_back_s = 0
        for ft in sample.feature_type_samples:
            if self.resample_source:
                if sample[ft].source is not None:
                    sample_new, log_q_s_ft, log_q_back_s_ft = self.propose_new_sources(
                        sample_old=sample,
                        sample_new=sample_new,
                        i_cluster=i_cluster,
                        object_subset=new_objects,
                        feature_type=ft
                    )
                    log_q_s += log_q_s_ft
                    log_q_back_s += log_q_back_s_ft

        # The removal probability of an inverse step

        shrink_candidates = self.shrink_candidates(sample_new, i_cluster)

        cluster_posterior_back = self.compute_cluster_posterior(sample_new, i_cluster, shrink_candidates)

        p_remove = normalize(1 - cluster_posterior_back)

        log_q = np.log(p_add[new_objects]).sum() + log_q_s
        log_q_back = np.log(p_remove[new_objects]).sum() + log_q_back_s

        return sample_new, log_q, log_q_back

    def shrink_cluster(self, sample: Sample) -> tuple[Sample, float, float]:
        # Choose a cluster
        i_cluster = RNG.choice(range(sample.n_clusters))

        # Load and precompute useful variables
        model = self.model_by_chain[sample.chain]
        available = self.available(sample, i_cluster)
        candidates = self.shrink_candidates(sample, i_cluster)
        n_candidates = candidates.sum()

        # If the cluster is already at min size, reject:
        assert n_candidates > 0
        if n_candidates == model.min_size:
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        # Otherwise create a new sample and continue with step:
        sample_new = sample.copy()

        # Assign probabilities for each unoccupied object
        cluster_posterior = np.zeros(sample.n_objects)
        cluster_posterior[available] = self.compute_cluster_posterior(
            sample, i_cluster, available,
        )

        x = (1 - cluster_posterior) * candidates
        if np.sum(x) == 0:
            return sample, self.Q_REJECT, self.Q_BACK_REJECT
        p_remove = normalize(x)

        n_removable = np.sum(p_remove > 0)
        # n_remove = min(self.n_changes, n_removable - model.min_size)
        n_remove = min(1, n_removable - model.min_size)

        # Draw object to be removed according to posterior
        removed_objects = RNG.choice(sample.n_objects, p=p_remove, size=n_remove, replace=False)
        for obj in removed_objects:
            sample_new.clusters.remove_object(i_cluster, obj)

        log_q_s = log_q_back_s = 0
        for ft in FeatureType.values():
            if self.resample_source and sample.feature_type_samples[ft] is not None:
                if sample[ft].source is not None:
                    sample_new, log_q_s_categorical, log_q_back_s_categorical = self.propose_new_sources(
                        sample_old=sample,
                        sample_new=sample_new,
                        i_cluster=i_cluster,
                        object_subset=removed_objects,
                        feature_type=ft,
                    )
                    log_q_s += log_q_s_categorical
                    log_q_back_s += log_q_back_s_categorical

        # The add probability of an inverse step
        cluster_posterior_back = np.zeros(sample_new.n_objects)
        cluster_posterior_back[available] = self.compute_cluster_posterior(sample_new, i_cluster, available)
        grow_candidates = self.grow_candidates(sample_new)
        p_add = normalize(cluster_posterior_back * grow_candidates)

        log_q = np.log(p_remove[removed_objects]).sum() + log_q_s
        log_q_back = np.log(p_add[removed_objects]).sum() + log_q_back_s

        return sample_new, log_q, log_q_back

    def get_parameters(self) -> dict[str, Any]:
        params = super().get_parameters()
        params["n_changes"] = self.n_changes
        if self.consider_geo_prior:
            params["consider_geo_prior"] = True
        return params


# todo: check where needed and update
class ClusterEffectProposals:

    @staticmethod
    def gibbs(model: Model, sample: Sample, i_cluster: int) -> NDArray[float]:
        # todo: include gaussian, poisson and logitnormal features
        return conditional_effect_mean(
            prior_counts=model.prior.categorical.prior_cluster_effect.concentration_array,
            feature_counts=sample.categorical.sufficient_statistics['clusters'].value[[i_cluster]]
        )

    def residual_counts(model: Model, sample: Sample, i_cluster: int) -> NDArray[float]:
        features = model.data.features.values

        cluster = sample.clusters.value[i_cluster]
        size = np.count_nonzero(cluster)
        free_objects = (~sample.clusters.any_cluster()) | cluster
        n_free = np.count_nonzero(free_objects)

        # Create counts in free objects
        prior_counts = model.prior.prior_cluster_effect.concentration_array
        exp_counts_conf = ClusterEffectProposals.expected_confounder_features(model, sample)
        residual_features = np.clip(features[free_objects] - exp_counts_conf[free_objects], 0, None)
        residual_counts = np.sum(residual_features, axis=0)

        # The expected effect is given by the normalized posterior counts
        p = normalize(residual_counts + prior_counts, axis=-1)

        # Only consider objects with likelihood contribution above median
        lh = np.sum(p * residual_features, axis=(1,2))
        relevant = lh >= np.quantile(lh, 1-size/n_free)
        residual_counts = np.sum(residual_features[relevant], axis=0)

        # The expected effect is given by the normalized posterior counts
        return normalize(residual_counts + prior_counts, axis=-1)

    @staticmethod
    def expected_confounder_features(model: Model, sample: Sample) -> NDArray[float]:
        expected_features = np.zeros((sample.n_objects, sample.n_features, sample.n_states))
        weights = update_weights[ft](sample, caching=False)
        confounders = model.data.confounders
        for i_comp, (name_conf, conf) in enumerate(confounders.items(), start=1):
            prior_counts = model.prior.prior_confounding_effects[name_conf].concentration_array(sample)

            p_conf = normalize(conf.sufficient_statistics + prior_counts, axis=-1)  # (n_features, n_states)
            for i_g, g in enumerate(conf.group_assignment):
                expected_features[g] += weights[g, :, [i_comp], None] * p_conf[np.newaxis, i_g, ...]

        return expected_features


class AlterClusterGibbsishWide(AlterClusterGibbsish):

    def __init__(
        self,
        *args,
        w_stay: float = 0.1,
        cluster_effect_proposal: callable = ClusterEffectProposals.gibbs,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.w_stay = w_stay
        self.eps = 0.0001
        self.cluster_effect_proposal = cluster_effect_proposal

    def get_cluster_membership_proposal(self, sample, i_cluster, available):
        cluster = sample.clusters.value[i_cluster]
        p = self.compute_cluster_probs(sample, i_cluster, available)

        # For more local steps: proposal is a mixture of posterior and current cluster
        p = (1 - self.w_stay) * normalize(p + self.eps) + self.w_stay * normalize(cluster[available])

        # Expected size should be the same as current size
        old_size = np.sum(cluster[available])
        new_expected_size = np.sum(p)
        for _ in range(10):
            p = p * old_size / new_expected_size
            p = p.clip(self.eps, 1-self.eps)

            new_expected_size = np.sum(p)
            if new_expected_size > 0.975 * old_size:
                break

        return p

    def compute_cluster_probs(
        self,
        sample: Sample,
        i_cluster: int,
        available: NDArray[bool],
    ) -> NDArray[float]:  # shape: (n_available, )
        model = self.model_by_chain[sample.chain]

        if self.sample_from_prior:
            n_available = np.count_nonzero(available)
            return 0.5*np.ones(n_available)

        cluster_posterior = np.zeros(sample.n_objects)
        cluster_posterior[available] = 1.

        for ft_likelihood in model.likelihood.feature_type_likelihoods:
            assert ft_likelihood.is_used(sample)
            cluster_lh_z = ft_likelihood.pointwise_conditional_cluster_lh(
                sample=sample, i_cluster=i_cluster, available=available
            )
            all_lh = deepcopy(
                ft_likelihood.pointwise_likelihood(model=model, sample=sample)[available, :]
            )
            all_lh[..., 0] = cluster_lh_z

            weights_z01 = self.compute_feature_weights_with_and_without(
                sample, available, feature_type=ft_likelihood.feature_type
            )
            feature_lh_z01 = inner1d(all_lh[np.newaxis, ...], weights_z01)
            marginal_lh_z01 = np.prod(feature_lh_z01, axis=-1)

            cluster_posterior *= marginal_lh_z01[1] / (marginal_lh_z01[0] + marginal_lh_z01[1])

        if self.consider_geo_prior:
            cluster_posterior *= np.exp(model.prior.geo_prior.get_costs_per_object(sample, i_cluster)[available])

        return cluster_posterior

    def _propose(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        sample_new = sample.copy()
        i_cluster = RNG.choice(range(sample.n_clusters))
        cluster_old = sample.clusters.value[i_cluster]
        available = self.available(sample, i_cluster)
        n_available = np.count_nonzero(available)

        model = self.model_by_chain[sample.chain]

        p = self.get_cluster_membership_proposal(sample, i_cluster, available)

        cluster_new = (RNG.random(n_available) < p)
        if not (model.min_size <= np.count_nonzero(cluster_new) <= model.max_size):
            # Reject if proposal goes out of cluster size bounds
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        if np.all(cluster_new == cluster_old[available]):
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        q_per_site = p * cluster_new + (1 - p) * (1 - cluster_new)
        log_q = np.log(q_per_site).sum()

        with sample_new.clusters.edit_cluster(i_cluster) as c:
            c[available] = cluster_new

        log_q_s = log_q_back_s = 0
        feature_types = ["categorical", "gaussian", "poisson", "logitnormal"]
        changed, = np.where(cluster_old != sample_new.clusters.value[i_cluster])

        for ft in feature_types:
            if getattr(sample, ft) is not None:
                sample_new, log_q_s_ft, log_q_back_s_ft = self.propose_new_sources(
                    sample_old=sample,
                    sample_new=sample_new,
                    i_cluster=i_cluster,
                    object_subset=changed,
                    feature_type=ft
                )
                log_q_s += log_q_s_ft
                log_q_back_s += log_q_back_s_ft

        p_back = self.get_cluster_membership_proposal(sample_new, i_cluster, available)
        q_back_per_site = p_back * cluster_old[available] + (1 - p_back) * (1 - cluster_old[available])
        log_q_back = np.log(q_back_per_site).sum()

        assert np.all(p_back > 0)
        assert np.all(q_back_per_site > 0), q_back_per_site

        log_q += log_q_s
        log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back

    def get_parameters(self) -> dict[str, Any]:
        params = super().get_parameters()
        params.pop("n_changes")
        params["w_stay"] = self.w_stay
        params["cluster_effect_proposal"] = self.cluster_effect_proposal.__name__
        return params


class AlterCluster(ClusterOperator):

    def __init__(
        self,
        *args,
        adjacency_matrix: NDArray[bool],
        p_grow_connected: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adjacency_matrix = adjacency_matrix
        self.p_grow_connected = p_grow_connected

    def _propose(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        log_q = 0.
        log_q_back = 0.
        for i in range(self.n_changes):
            if random.random() < self.p_grow:
                sample_new, log_q_i, log_q_back_i = self.grow_cluster(sample)
                log_q_i += np.log(self.p_grow)
                log_q_back_i += np.log(1 - self.p_grow)
            else:
                sample_new, log_q_i, log_q_back_i = self.shrink_cluster(sample)
                log_q_i += np.log(1 - self.p_grow)
                log_q_back_i += np.log(self.p_grow)

            if log_q_back_i != self.Q_BACK_REJECT:
                sample = sample_new
                log_q += log_q_i
                log_q_back += log_q_back_i

        if DEBUG and (FeatureType.categorical in sample.feature_type_samples):
            verify_counts(sample, self.model_by_chain[sample.chain].data.features.categorical.values)
            verify_counts(sample_new, self.model_by_chain[sample.chain].data.features.categorical.values)

        return sample_new, log_q, log_q_back

    def grow_cluster(self, sample: Sample) -> tuple[Sample, float, float]:
        """Grow a clusters in the current sample (i.e. add a new site to one cluster)."""
        sample_new = sample.copy()
        occupied = sample.clusters.any_cluster()

        # Randomly choose one of the clusters to modify
        i_cluster = RNG.choice(range(sample.clusters.n_clusters))
        cluster_current = sample.clusters.value[i_cluster, :]

        # Check if cluster is small enough to grow
        current_size = np.count_nonzero(cluster_current)

        if current_size >= self.model_by_chain[sample.chain].max_size:
            # Cluster too big to grow: don't modify the sample and reject the step (q_back = 0)
            return sample, 0, -np.inf

        neighbours = get_neighbours(cluster_current, occupied, self.adjacency_matrix)
        connected_step = random.random() < self.p_grow_connected
        if connected_step:
            # All neighboring sites that are not yet occupied by other clusters are candidates
            candidates = neighbours
        else:
            # All free sites are candidates
            candidates = ~occupied

        # When stuck (no candidates) return current sample and reject the step (q_back = 0)
        if not np.any(candidates):
            return sample, 0, -np.inf

        # Choose a random candidate and add it to the cluster
        object_add = RNG.choice(candidates.nonzero()[0])
        sample_new.clusters.add_object(i_cluster, object_add)

        # Transition probability when growing
        q_non_connected = 1 / np.count_nonzero(~occupied)
        q = (1 - self.p_grow_connected) * q_non_connected

        if neighbours[object_add]:
            q_connected = 1 / np.count_nonzero(neighbours)
            q += self.p_grow_connected * q_connected

        # Back-probability (shrinking)
        q_back = 1 / (current_size + 1)

        log_q = np.log(q)
        log_q_back = np.log(q_back)

        for ft in sample.feature_type_samples:
            if self.resample_source and sample[ft].source is not None:
                sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
                    sample_old=sample,
                    sample_new=sample_new,
                    i_cluster=i_cluster,
                    object_subset=[object_add],
                    feature_type=ft,
                )
                log_q += log_q_s
                log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back

    def shrink_cluster(self, sample: Sample) -> tuple[Sample, float, float]:
        """Shrink a cluster in the current sample (i.e. remove one object from one cluster)."""
        sample_new = sample.copy()

        # Randomly choose one of the clusters to modify
        i_cluster = RNG.choice(range(sample.clusters.n_clusters))
        cluster_current = sample.clusters.value[i_cluster, :]

        # Check if cluster is big enough to shrink
        current_size = np.count_nonzero(cluster_current)
        if current_size <= self.model_by_chain[sample.chain].min_size:
            # Cluster is too small to shrink: don't modify the sample and reject the step (q_back = 0)
            return sample, 0, -np.inf

        # Cluster is big enough: shrink
        removal_candidates = self.get_removal_candidates(cluster_current)
        object_remove = RNG.choice(removal_candidates)
        sample_new.clusters.remove_object(i_cluster, object_remove)

        # Transition probability when shrinking.
        q = 1 / len(removal_candidates)
        # Back-probability (growing)
        cluster_new = sample_new.clusters.value[i_cluster]
        occupied_new = sample_new.clusters.any_cluster()
        back_neighbours = get_neighbours(cluster_new, occupied_new, self.adjacency_matrix)

        # The back step could always be a non-connected grow step
        q_back_non_connected = 1 / np.count_nonzero(~occupied_new)
        q_back = (1 - self.p_grow_connected) * q_back_non_connected

        # If z is a neighbour of the new zone, the back step could also be a connected grow-step
        if back_neighbours[object_remove]:
            q_back_connected = 1 / np.count_nonzero(back_neighbours)
            q_back += self.p_grow_connected * q_back_connected

        log_q = np.log(q)
        log_q_back = np.log(q_back)

        for ft in sample.feature_type_samples:
            if self.resample_source and sample[ft].source is not None:
                sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
                    sample_old=sample,
                    sample_new=sample_new,
                    i_cluster=i_cluster,
                    object_subset=[object_remove],
                    feature_type=ft,
                )
                log_q += log_q_s
                log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back


# todo: update for continuous sBayes
class ClusterJump(ClusterOperator):

    def __init__(
        self,
        *args,
        gibbsish: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gibbsish = gibbsish

    def get_jump_lh(self, sample: Sample, i_source_cluster: int, i_target_cluster: int) -> NDArray[float]:
        model = self.model_by_chain[sample.chain]
        features = model.data.features.values
        source_cluster = sample.clusters.value[i_source_cluster]
        w_clust = update_weights[ft](sample)[source_cluster, :, 0]

        p_clust_source = conditional_effect_mean(
            prior_counts=model.prior.prior_cluster_effect.concentration_array,
            feature_counts=sample.feature_counts['clusters'].value[[i_source_cluster]]
        )
        p_clust_target = conditional_effect_mean(
            prior_counts=model.prior.prior_cluster_effect.concentration_array,
            feature_counts=sample.feature_counts['clusters'].value[[i_target_cluster]]
        )
        p_conf = ClusterEffectProposals.expected_confounder_features(model, sample)[source_cluster]

        p_total_source = p_conf + w_clust[..., np.newaxis] * p_clust_source
        p_total_target = p_conf + w_clust[..., np.newaxis] * p_clust_target

        lh_stay_per_feature = np.sum(features[source_cluster] * p_total_source, axis=-1)
        lh_stay = np.prod(lh_stay_per_feature, axis=-1,
                          where=~model.data.features.na_values[source_cluster])
        lh_jump_per_feature = np.sum(features[source_cluster] * p_total_target, axis=-1)
        lh_jump = np.prod(lh_jump_per_feature, axis=-1,
                          where=~model.data.features.na_values[source_cluster])

        return lh_jump / (lh_jump + lh_stay)

    def _propose(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        """Reassign an object from one cluster to another."""
        sample_new = sample.copy()
        model = self.model_by_chain[sample.chain]

        # Randomly choose a source and target cluster
        i_source_cluster, i_target_cluster = RNG.choice(range(sample.clusters.n_clusters), size=2, replace=False)
        source_cluster = sample.clusters.value[i_source_cluster, :]
        target_cluster = sample.clusters.value[i_target_cluster, :]

        # Check if clusters are too small/large for jump
        source_size = np.count_nonzero(source_cluster)
        target_size = np.count_nonzero(target_cluster)
        if source_size <= model.min_size or target_size >= model.max_size:
            # Directly reject the step
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        # Choose a random candidate and add it to the cluster
        if self.gibbsish:
            p_jump = normalize(self.get_jump_lh(sample, i_source_cluster, i_target_cluster))
        else:
            p_jump = np.ones(source_size) / source_size

        jumping_object = RNG.choice(np.flatnonzero(source_cluster), p=p_jump)
        sample_new.clusters.remove_object(i_source_cluster, jumping_object)
        sample_new.clusters.add_object(i_target_cluster, jumping_object)

        if self.resample_source:
            sample_new, log_q_s, log_q_back_s = self.gibbs_sample_source_jump(
                sample_new=sample_new,
                sample_old=sample,
                i_cluster_new=i_target_cluster,
                i_cluster_old=i_source_cluster,
                object_subset=[jumping_object],
            )
        else:
            update_feature_counts(sample, sample_new, model.data.features.values, [jumping_object])
            update_sufficient_statistics(sample, sample_new, model.data.features.values, [jumping_object])
            log_q_s = log_q_back_s = 0

        if self.gibbsish:
            p_jump_back = normalize(self.get_jump_lh(sample_new, i_target_cluster, i_source_cluster))
        else:
            p_jump_back = np.ones(target_size+1) / (target_size+1)

        # Transition probabilities
        new_source_cluster = sample_new.clusters.value[i_target_cluster, :]
        i_forward = np.flatnonzero(source_cluster).tolist().index(jumping_object)
        i_back = np.flatnonzero(new_source_cluster).tolist().index(jumping_object)
        log_q = np.log(p_jump[i_forward]) + log_q_s
        log_q_back = np.log(p_jump_back[i_back]) + log_q_back_s

        return sample_new, log_q, log_q_back

    def gibbs_sample_source_jump(
        self,
        sample_new: Sample,
        sample_old: Sample,
        i_cluster_new: int,
        i_cluster_old: int,
        object_subset: slice | list[int] | NDArray[int] = slice(None),
    ) -> tuple[Sample, float, float]:
        """Resample the observations to mixture components (their source)."""
        model = self.model_by_chain[sample_old.chain]
        features = model.data.features.values
        na_features = model.data.features.na_values

        # Make sure object_subset is a boolean index array
        object_subset = np.isin(np.arange(sample_new.n_objects), object_subset)
        w = update_weights[ft](sample_new)[object_subset]

        if self.sample_from_prior:
            p = w
        else:
            lh_per_component_new = component_likelihood_given_unchanged(
                model, sample_new, object_subset, i_cluster=i_cluster_new
            )
            p = normalize(w * lh_per_component_new, axis=-1)

        # Sample the new source assignments
        with sample_new.source.edit() as source:
            source[object_subset] = sample_categorical(p=p, binary_encoding=True)
            source[na_features] = 0

        update_feature_counts(sample_old, sample_new, features, object_subset)
        update_sufficient_statistics(sample_old, sample_new, features, object_subset)
        if DEBUG:
            verify_counts(sample_new, features)

        # Transition probability forward:
        source_new = sample_new.source.value[object_subset]
        log_q = np.log(p[source_new]).sum()

        # Transition probability backward:
        if self.sample_from_prior:
            p_back = w
        else:
            lh_per_component_old = component_likelihood_given_unchanged(
                model, sample_old, object_subset, i_cluster=i_cluster_old
            )
            p_back = normalize(w * lh_per_component_old, axis=-1)

        source_old = sample_old.source.value[object_subset]
        log_q_back = np.log(p_back[source_old]).sum()

        return sample_new, log_q, log_q_back


class OperatorSchedule:
    RW_OPERATORS_BY_NAMES = {
        "weights": AlterWeights,
        "...": "...",
    }
    GIBBS_OPERATORS_BY_NAMES = {
        "weights": GibbsSampleWeights,
        "...": "...",
    }

    def __init__(self, operators_config: OperatorsConfig, sampple_source: bool):
        self.operators_config = operators_config
        self.sample_source = sampple_source

        weights = []
        self.operators = []
        for name, w in self.operators_config:
            weights.append(w)
            self.operators.append(self.get_operator_by_name(name))
        self.weights = normalize(weights)

    def draw_operator(self) -> Operator:
        """Return a random operator with probability proportional to self.weights"""
        return RNG.choice(self.operators, 1, p=self.weights)[0]

    def get_operator_by_name(self, name):
        if self.sample_source:
            return self.GIBBS_OPERATORS_BY_NAMES[name]
        else:
            return self.RW_OPERATORS_BY_NAMES[name]


def verify_counts(sample, features: NDArray[bool]):
    # Copy the cached counts (otherwise they would be updated in the next line)
    cached_counts = deepcopy(sample.categorical.sufficient_statistics)

    # Recalculate the counts based on the features and current sample
    new_counts = recalculate_feature_counts(features, sample)

    # Assert that all keys and values are the same
    assert set(cached_counts.keys()) == set(new_counts.keys())
    for k in cached_counts.keys():
        assert np.allclose(cached_counts[k].value, new_counts[k].value), f'''Counts not matching in ´{k}´:
    Cached counts: {cached_counts[k].value.sum(axis=(1,2))}
    Recalculated counts: {new_counts[k].value.sum(axis=(1,2))}'''
