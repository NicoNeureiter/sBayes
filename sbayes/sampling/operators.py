from __future__ import annotations
from copy import deepcopy
import random
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, Any

import numpy as np
from numpy.typing import NDArray
from numpy.core.umath_tests import inner1d
import scipy.stats as stats

from sbayes.load_data import Features
from sbayes.sampling.conditionals import likelihood_per_component, conditional_effect_mean
from sbayes.sampling.counts import recalculate_feature_counts, update_feature_counts
from sbayes.sampling.state import Sample
from sbayes.util import dirichlet_logpdf, normalize, get_neighbours
from sbayes.model import Model, Likelihood, normalize_weights, update_weights
from sbayes.preprocessing import sample_categorical
from sbayes.config.config import OperatorsConfig


DEBUG = 0
RNG = np.random.default_rng()


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
        self.max_size = min(max_size, self.model_by_chain[0].shapes.n_sites)

        if self.model_by_chain[0].shapes.n_sites <= self.min_size:
            self.object_selector = ObjectSelector.ALL

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

    def select_object_subset(self, sample) -> NDArray[bool]:  # shape: (n_objects,)
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
                object_subset = self.random_subset(n=sample.n_objects, k=self.max_size, population=np.where(object_subset)[0])

        elif self.object_selector is ObjectSelector.RANDOM_SUBSET:
            # Choose a random subset for which the source is resampled
            # r = RNG.random(sample.n_objects)
            # object_subset = r < max(50 / sample.n_objects, np.min(r))
            # object_subset = r < max(0.3, np.min(r))
            object_subset = self.random_subset(n=sample.n_objects, k=self.max_size)
        else:
            raise ValueError(f"ObjectSelector '{self.object_selector}' not yet implemented.")

        return object_subset

    def _propose(
        self,
        sample: Sample,
        object_subset: slice | list[int] = slice(None),
        **kwargs,
    ) -> tuple[Sample, float, float]:
        """Resample the observations to mixture components (their source).

        Args:
            sample: The current sample with clusters and parameters
            object_subset: A subset of sites to be updated

        Returns:
            The modified sample and forward and backward transition log-probabilities
        """
        features = self.model_by_chain[sample.chain].data.features.values
        na_features = self.model_by_chain[sample.chain].data.features.na_values
        sample_new = sample.copy()

        assert np.all(sample.source.value[na_features] == 0)

        object_subset = self.select_object_subset(sample)

        if self.sample_from_prior:
            p = update_weights(sample)[object_subset]
        else:
            p = self.calculate_source_posterior(sample, object_subset)

        # Sample the new source assignments
        with sample_new.source.edit() as source:
            source[object_subset] = sample_categorical(p=p, binary_encoding=True)
            source[na_features] = 0

        update_feature_counts(sample, sample_new, features, object_subset)

        if DEBUG:
            verify_counts(sample_new, features)

        # Transition probability forward:
        log_q = np.log(p[sample_new.source.value[object_subset]]).sum()

        assert np.all(sample_new.source.value[na_features] == 0)
        assert np.all(sample_new.source.value[~na_features].sum(axis=-1) == 1)

        # Transition probability backward:
        if self.sample_from_prior:
            p_back = p
        else:
            p_back = self.calculate_source_posterior(sample_new, object_subset)

        log_q_back = np.log(p_back[sample.source.value[object_subset]]).sum()

        return sample_new, log_q, log_q_back

    def calculate_source_posterior(
        self, sample: Sample, object_subset: slice | list[int] = slice(None)
    ) -> NDArray[float]:  # shape: (n_objects_in_subset, n_features, n_components)
        """Compute the posterior support for source assignments of every object and feature."""

        # 1. compute likelihood for each component
        model: Model = self.model_by_chain[sample.chain]
        lh_per_component = likelihood_per_component(model=model, sample=sample, caching=True)

        # 2. multiply by weights and normalize over components to get the source posterior
        weights = update_weights(sample)

        # 3. The posterior of the source for each observation is likelihood times prior
        # (normalized to sum up to one across source components):
        return normalize(
            lh_per_component[object_subset] * weights[object_subset], axis=-1
        )

    def get_step_size(self, sample_old: Sample, sample_new: Sample) -> float:
        return np.count_nonzero(sample_old.source.value ^ sample_new.source.value)

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
        na_features = model.likelihood.na_features

        # Compute the old likelihood
        w = sample.weights.value
        w_normalized_old = update_weights(sample)
        log_lh_old = self.source_lh_by_feature(sample.source.value, w_normalized_old, na_features)
        log_prior_old = model.prior.prior_weights.pointwise_prior(sample)

        # Resample the weights
        w_new, log_q, log_q_back = self.resample_weight_for_two_components(
            sample, model.likelihood
        )
        sample.weights.set_value(w_new)

        # Compute new likelihood
        w_new_normalized = update_weights(sample)
        log_lh_new = self.source_lh_by_feature(sample.source.value, w_new_normalized, na_features)
        log_prior_new = model.prior.prior_weights.pointwise_prior(sample)

        # Add the prior to get the weight posterior (for each feature)
        # log_prior_old = 0.0  # TODO add hyper prior on weights, when implemented
        # log_prior_new = 0.0  # TODO add hyper prior on weights, when implemented
        log_p_old = log_lh_old + log_prior_old
        log_p_new = log_lh_new + log_prior_new

        # Compute hastings ratio for each feature and accept/reject independently
        p_accept = np.exp(log_p_new - log_p_old + log_q_back - log_q)
        accept = RNG.random(p_accept.shape) < p_accept
        sample.weights.set_value(np.where(accept[:, np.newaxis], w_new, w))

        assert ~np.any(np.isnan(sample.weights.value))

        self.last_accept_rate = np.mean(accept)

        # We already accepted/rejected for each feature independently
        # The result should always be accepted in the MCMC
        return sample, self.Q_GIBBS, self.Q_BACK_GIBBS

    def resample_weight_for_two_components(
        self, sample: Sample, likelihood: Likelihood
    ) -> NDArray[float]:
        model = self.model_by_chain[sample.chain]
        w = sample.weights.value
        source = sample.source.value
        has_components = sample.cache.has_components.value

        # Fix weights for all but two random components
        i1, i2 = random.sample(range(sample.n_components), 2)

        # Select counts of the relevant languages
        has_both = np.logical_and(has_components[:, i1], has_components[:, i2])
        counts = (
            np.sum(source[has_both, :, :], axis=0)           # likelihood counts
            + model.prior.prior_weights.concentration_array  # prior counts
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
    ) -> tuple[Sample, float, float]:
        n_features = sample_old.n_features
        features = self.model_by_chain[sample_old.chain].data.features.values
        na_features = self.model_by_chain[sample_old.chain].data.features.na_values

        MODE = self.resample_source_mode

        if MODE == ResampleSourceMode.GIBBS:
            sample_new, log_q, log_q_back = self.gibbs_sample_source(
                sample_new, sample_old, i_cluster, object_subset=object_subset
            )

        elif MODE == ResampleSourceMode.PRIOR:
            p = update_weights(sample_new)[object_subset]
            p_back = update_weights(sample_old)[object_subset]
            with sample_new.source.edit() as source:
                source[object_subset, :, :] = sample_categorical(p, binary_encoding=True)
                source[na_features] = 0

                log_q = np.log(p[source[object_subset]]).sum()


            log_q_back = np.log(p_back[sample_old.source.value[object_subset]]).sum()

            update_feature_counts(sample_old, sample_new, features, object_subset)
            if DEBUG:
                verify_counts(sample_new, features)

        elif MODE == ResampleSourceMode.UNIFORM:
            has_components_new = sample_new.cache.has_components.value
            p = normalize(
                np.tile(has_components_new[object_subset, None, :], (1, n_features, 1))
            )
            with sample_new.source.edit() as source:
                source[object_subset, :, :] = sample_categorical(
                    p, binary_encoding=True
                )
                source[na_features] = 0
                log_q = np.log(p[source[object_subset]]).sum()

            has_components_old = sample_old.cache.has_components.value
            p_back = normalize(
                np.tile(has_components_old[object_subset, None, :], (1, n_features, 1))
            )
            log_q_back = np.log(p_back[sample_old.source.value[object_subset]]).sum()

            update_feature_counts(sample_old, sample_new, features, object_subset)
            if DEBUG:
                verify_counts(sample_new, features)
        else:
            raise ValueError(f"Invalid mode `{MODE}`. Choose from (gibbs, prior and uniform)")

        return sample_new, log_q, log_q_back

    def gibbs_sample_source(
        self,
        sample_new: Sample,
        sample_old: Sample,
        i_cluster: int,
        object_subset: slice | list[int] | NDArray[int] = slice(None),
    ) -> tuple[Sample, float, float]:
        """Resample the observations to mixture components (their source)."""
        model = self.model_by_chain[sample_old.chain]
        features = model.data.features.values
        na_features = model.data.features.na_values

        # Make sure object_subset is a boolean index array
        object_subset = np.isin(np.arange(sample_new.n_objects), object_subset)
        lh_per_component = component_likelihood_given_unchanged(
            model, sample_new, object_subset, i_cluster=i_cluster
        )

        if self.sample_from_prior:
            p = update_weights(sample_new)[object_subset]
        else:
            w = update_weights(sample_new)[object_subset]
            p = normalize(w * lh_per_component, axis=-1)
            # p = self.calculate_source_posterior(sample_new, object_subset)

        # Sample the new source assignments
        with sample_new.source.edit() as source:
            source[object_subset] = sample_categorical(p=p, binary_encoding=True)
            source[na_features] = 0

        update_feature_counts(sample_old, sample_new, features, object_subset)
        if DEBUG:
            verify_counts(sample_new, features)

        # Transition probability forward:
        source_new = sample_new.source.value[object_subset]
        log_q = np.log(p[source_new]).sum()

        # Transition probability backward:
        if self.sample_from_prior:
            p_back = update_weights(sample_old)[object_subset]
        else:
            w = update_weights(sample_old)[object_subset]
            p_back = normalize(w * lh_per_component, axis=-1)
            # p_back = self.calculate_source_posterior(sample_old, object_subset)

        source_old = sample_old.source.value[object_subset]
        log_q_back = np.log(p_back[source_old]).sum()

        return sample_new, log_q, log_q_back

    # def gibbs_sample_source_shrink(
    #     self,
    #     sample_new: Sample,
    #     sample_old: Sample,
    #     object_subset: slice | list[int] | NDArray[int] = slice(None),
    # ) -> tuple[Sample, float, float]:
    #     """Resample the observations to mixture components (their source)."""
    #     features = self.model_by_chain[sample_old.chain].data.features.values
    #     na_features = self.model_by_chain[sample_old.chain].data.features.na_values
    #
    #     if self.sample_from_prior:
    #         p = update_weights(sample_new)[object_subset]
    #     else:
    #         p = self.calculate_source_posterior(sample_new, object_subset)
    #
    #     # Sample the new source assignments
    #     with sample_new.source.edit() as source:
    #         was_cluster = source[object_subset, :, 0]
    #
    #         # Resample cluster observations
    #         new_source = sample_categorical(p=p[was_cluster], binary_encoding=True)
    #
    #         for i, was_cluster_i, new_source_i in zip(object_subset, was_cluster, new_source):
    #             source[i, was_cluster_i] = new_source_i
    #         # source[object_subset, ...][was_cluster, ...] = new_source
    #
    #         source[na_features] = 0
    #
    #     update_feature_counts(sample_old, sample_new, features, object_subset)
    #     if DEBUG:
    #         verify_counts(sample_new, features)
    #
    #     # Transition probability forward:
    #     new_source = sample_new.source.value[object_subset]
    #     # log_q = np.log(p[new_source]).sum()
    #     log_q = np.log(p[was_cluster][new_source[was_cluster]]).sum()
    #
    #     # Transition probability backward:
    #     if self.sample_from_prior:
    #         p_back = update_weights(sample_old)[object_subset]
    #     else:
    #         p_back = self.calculate_source_posterior(sample_old, object_subset,
    #                                                  feature_counts=sample_new.feature_counts)
    #
    #     log_q_back = float(
    #         np.log(p_back[was_cluster, 0]).sum() +
    #         np.log(1 - p_back[~was_cluster, 0]).sum()
    #     )
    #
    #     return sample_new, log_q, log_q_back
    #
    # def gibbs_sample_source_grow(
    #     self,
    #     sample_new: Sample,
    #     sample_old: Sample,
    #     object_subset: slice | list[int] | NDArray[int] = slice(None),
    # ) -> tuple[Sample, float, float]:
    #     """Resample the observations to mixture components (their source)."""
    #     features = self.model_by_chain[sample_old.chain].data.features.values
    #     na_features = self.model_by_chain[sample_old.chain].data.features.na_values
    #
    #     w = update_weights(sample_old)[object_subset]
    #     s = sample_old.source.value[object_subset]
    #     assert np.all(s <= (w > 0)), np.max(w)
    #
    #     if self.sample_from_prior:
    #         p = update_weights(sample_new)[object_subset]
    #     else:
    #         p = self.calculate_source_posterior(sample_new, object_subset)
    #
    #     # Sample the new source assignments
    #     with sample_new.source.edit() as source:
    #         # Which non-cluster observations should become cluster source:
    #         p_cluster = p[..., 0]       # shape: (n_obj, n_feat)
    #         to_cluster = np.random.random(p_cluster.shape) < p_cluster
    #         for i, to_cluster_i in zip(object_subset, to_cluster):
    #             source[i, to_cluster_i, 0] = True
    #             source[i, to_cluster_i, 1:] = False
    #
    #         source[na_features] = 0
    #
    #     update_feature_counts(sample_old, sample_new, features, object_subset)
    #     if DEBUG:
    #         verify_counts(sample_new, features)
    #
    #     # Transition probability forward:
    #     # log_q = np.log(p[sample_new.source.value[object_subset]]).sum()
    #     log_q = float(
    #         np.log(p_cluster[to_cluster]).sum() +
    #         np.log(1 - p_cluster[~to_cluster]).sum()
    #     )
    #
    #     # Transition probability backward:
    #     if self.sample_from_prior:
    #         p_back = update_weights(sample_old)[object_subset]
    #     else:
    #         p_back = self.calculate_source_posterior(sample_old, object_subset,
    #                                                  feature_counts=sample_new.feature_counts)
    #
    #     old_source = sample_old.source.value[object_subset, ...]
    #     log_q_back = np.log(p_back[to_cluster][old_source[to_cluster]]).sum()
    #
    #     return sample_new, log_q, log_q_back

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
) -> NDArray[float]:  # shape: (n_objects, n_feature, n_components)
    features = model.data.features
    confounders = model.data.confounders
    cluster = sample.clusters.value[i_cluster]
    source = sample.source.value
    subset_size = np.count_nonzero(object_subset)

    likelihoods = np.zeros((subset_size, sample.n_features, sample.n_components))
    likelihoods[..., 0] = cluster_likelihood_given_unchanged(cluster, features, object_subset, source)

    for i_conf, conf in enumerate(confounders, start=1):
        groups = confounders[conf].group_assignment

        features_conf = features.values * sample.source.value[:, :, i_conf, None]
        changeable_counts = np.array([
            np.sum(features_conf[g & object_subset], axis=0)
            for g in groups
        ])
        unchangeable_feature_counts = sample.feature_counts[conf].value - changeable_counts
        prior_counts = model.prior.prior_confounding_effects[conf].concentration_array(sample)
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

    return likelihoods


def cluster_likelihood_given_unchanged(
    cluster: NDArray[bool],  # (n_objects,)
    features: Features,
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
        features: NDArray[bool],
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

        if DEBUG:
            verify_counts(sample, self.model_by_chain[sample.chain].data.features.values)
            verify_counts(sample_new, self.model_by_chain[sample.chain].data.features.values)

        return sample_new, log_q, log_q_back

    def compute_cluster_posterior(
        self,
        sample: Sample,
        i_cluster: int,
        available: NDArray[bool],
    ) -> NDArray[float]:  # shape: (n_available, )
        model = self.model_by_chain[sample.chain]

        if self.sample_from_prior:
            n_available = np.count_nonzero(available)
            return 0.5*np.ones(n_available)

        p = conditional_effect_mean(
            prior_counts=model.prior.prior_cluster_effect.concentration_array,
            feature_counts=sample.feature_counts['clusters'].value[[i_cluster]]
        )
        cluster_lh_z = inner1d(self.features[available], p)
        all_lh = deepcopy(likelihood_per_component(model, sample, caching=True)[available, :])
        all_lh[..., 0] = cluster_lh_z

        weights_z01 = self.compute_feature_weights_with_and_without(sample, available)
        feature_lh_z01 = inner1d(all_lh[np.newaxis, ...], weights_z01)
        marginal_lh_z01 = np.prod(feature_lh_z01, axis=-1)
        cluster_posterior = marginal_lh_z01[1] / (marginal_lh_z01[0] + marginal_lh_z01[1])

        if self.consider_geo_prior:
            cluster_posterior *= np.exp(model.prior.geo_prior.get_costs_per_object(sample, i_cluster)[available])

        return cluster_posterior

    @staticmethod
    def compute_feature_weights_with_and_without(
        sample: Sample,
        available: NDArray[bool],   # shape: (n_objects, )
    ) -> NDArray[float]:            # shape: (2, n_objects, n_features, n_components)
        weights_current = update_weights(sample, caching=True)[available]
        # weights = normalize_weights(sample.weights.value, has_components)

        has_components = deepcopy(sample.cache.has_components.value[available, :])
        has_components[:, 0] = ~has_components[:, 0]
        weights_flipped = normalize_weights(sample.weights.value, has_components)

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

        # If all the space is take we can't grow
        if not np.any(candidates):
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        # If the cluster is already at max size, reject:
        if np.sum(cluster) == model.max_size:
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        # Otherwise create a new sample and continue with step:
        sample_new = sample.copy()

        # Assign probabilities for each unoccupied object
        cluster_posterior = np.zeros(sample.n_objects)
        cluster_posterior[available] = self.compute_cluster_posterior(
            sample, i_cluster, available
        )
        p_add = normalize(cluster_posterior * candidates)

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

        if self.resample_source and sample.source is not None:
            sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
                sample_old=sample,
                sample_new=sample_new,
                i_cluster=i_cluster,
                object_subset=new_objects,
            )
        else:
            log_q_s = log_q_back_s = 0

        # The removal probability of an inverse step
        cluster_posterior_back = np.zeros(sample_new.n_objects)
        cluster_posterior_back[available] = self.compute_cluster_posterior(sample_new, i_cluster, available)
        shrink_candidates = self.shrink_candidates(sample_new, i_cluster)
        p_remove = normalize((1 - cluster_posterior_back) * shrink_candidates)

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
            sample, i_cluster, available
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

        if self.resample_source and sample.source is not None:
            sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
                sample_old=sample,
                sample_new=sample_new,
                i_cluster=i_cluster,
                object_subset=removed_objects,
            )
        else:
            log_q_s = log_q_back_s = 0

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


class ClusterEffectProposals:

    @staticmethod
    def gibbs(model: Model, sample: Sample, i_cluster: int) -> NDArray[float]:
        return conditional_effect_mean(
            prior_counts=model.prior.prior_cluster_effect.concentration_array,
            feature_counts=sample.feature_counts['clusters'].value[[i_cluster]]
        )

    @staticmethod
    def residual(model: Model, sample: Sample, i_cluster: int) -> NDArray[float]:
        features = model.data.features.values
        free_objects = ~sample.clusters.any_cluster()

        # Create counts in free objects
        prior_counts = model.prior.prior_cluster_effect.concentration_array
        feature_counts = features[free_objects].sum(axis=0)

        # The expected effect is given by the normalized posterior counts
        return normalize(feature_counts + prior_counts, axis=-1)

    @staticmethod
    def residual4(model: Model, sample: Sample, i_cluster: int) -> NDArray[float]:
        features = model.data.features.values
        free_objects = ~sample.clusters.any_cluster()

        # Create counts in free objects
        prior_counts = model.prior.prior_cluster_effect.concentration_array
        exp_counts_conf = ClusterEffectProposals.expected_confounder_features(model, sample)
        residual_features = features[free_objects] - exp_counts_conf[free_objects]
        residual_counts = residual_features.clip(0).sum(axis=0)

        # Create counts in free objects
        prior_counts = model.prior.prior_cluster_effect.concentration_array
        exp_counts_conf = ClusterEffectProposals.expected_confounder_features(model, sample)
        residual_features = features[free_objects] - exp_counts_conf[free_objects]
        residual_counts = residual_features.clip(0).sum(axis=0)

        # The expected effect is given by the normalized posterior counts
        return normalize(residual_counts + prior_counts, axis=-1)

    @staticmethod
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
        weights = update_weights(sample, caching=False)
        confounders = model.data.confounders
        for i_comp, (name_conf, conf) in enumerate(confounders.items(), start=1):
            prior_counts = model.prior.prior_confounding_effects[name_conf].concentration_array(sample)

            p_conf = normalize(conf.feature_counts + prior_counts, axis=-1)  # (n_features, n_states)
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
        self.eps = 0.001
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

        p = self.cluster_effect_proposal(model, sample, i_cluster)

        cluster_lh_z = inner1d(self.features[available], p)
        all_lh = deepcopy(likelihood_per_component(model, sample, caching=True)[available, :])
        all_lh[..., 0] = cluster_lh_z

        weights_z01 = self.compute_feature_weights_with_and_without(sample, available)
        feature_lh_z01 = inner1d(all_lh[np.newaxis, ...], weights_z01)
        marginal_lh_z01 = np.prod(feature_lh_z01, axis=-1)
        cluster_posterior = marginal_lh_z01[1] / (marginal_lh_z01[0] + marginal_lh_z01[1])

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

        # Resample the source assignment for the changed objects
        changed, = np.where(cluster_old != sample_new.clusters.value[i_cluster])
        sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
            sample_old=sample, sample_new=sample_new,
            i_cluster=i_cluster, object_subset=changed,
        )

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

        if DEBUG:
            verify_counts(sample, self.model_by_chain[sample.chain].data.features.values)
            verify_counts(sample_new, self.model_by_chain[sample.chain].data.features.values)

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

        if self.resample_source and sample.source is not None:
            sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
                sample_old=sample,
                sample_new=sample_new,
                i_cluster=i_cluster,
                object_subset=[object_add],
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

        if self.resample_source and sample.source is not None:
            sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
                sample_old=sample,
                sample_new=sample_new,
                i_cluster=i_cluster,
                object_subset=[object_remove],
            )
            log_q += log_q_s
            log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back


class ClusterJump(ClusterOperator):

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
        jumping_object = RNG.choice(source_cluster.nonzero()[0])
        sample_new.clusters.remove_object(i_source_cluster, jumping_object)
        sample_new.clusters.add_object(i_target_cluster, jumping_object)

        # Transition probabilities
        log_q = np.log(1 / source_size)
        log_q_back = np.log(1 / (1 + target_size))

        if self.resample_source and sample.source is not None:
            sample_new, log_q_s, log_q_back_s = self.gibbs_sample_source_jump(
                sample_new=sample_new,
                sample_old=sample,
                i_cluster_new=i_target_cluster,
                i_cluster_old=i_source_cluster,
                object_subset=[jumping_object],
            )
            log_q += log_q_s
            log_q_back += log_q_back_s
        else:
            update_feature_counts(sample, sample_new, model.data.features.values, [jumping_object])

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
        w = update_weights(sample_new)[object_subset]

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
    cached_counts = deepcopy(sample.feature_counts)
    new_counts = recalculate_feature_counts(features, sample)
    assert set(cached_counts.keys()) == set(new_counts.keys())
    for k in cached_counts.keys():
        assert np.allclose(cached_counts[k].value, new_counts[k].value), (f'counts not matching in ´{k}´.',
            cached_counts[k].value.sum(axis=(1,2)), new_counts[k].value.sum(axis=(1,2))
        )
