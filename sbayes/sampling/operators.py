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

from sbayes.load_data import Confounder
from sbayes.sampling.conditionals import likelihood_per_component, sample_source_from_prior, logprob_source_from_prior, \
    conditional_effect_mean
from sbayes.sampling.counts import compute_effect_counts, recalculate_feature_counts, update_feature_counts
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

    additional_parameters: dict
    """Potential parameters controlling properties of the proposal function."""

    # Class constants

    Q_GIBBS = -np.inf
    Q_BACK_GIBBS = 0
    """Fixed transition probabilities for Gibbs operators (ensuring acceptance)."""

    Q_REJECT = 0
    Q_BACK_REJECT = -np.inf
    """Fixed transition probabilities for directly rejecting MCMC steps."""

    REQUIRED_PARAMETERS: Sequence[str] = []
    """Parameters that need to be defined in `additional_parameters` for this operator type."""

    def __init__(self, weight: float, **kwargs):
        self.weight = weight
        self.additional_parameters = kwargs

        self.accepts: int = 0
        self.rejects: int = 0

        # Ensure that all required parameters are defined
        for req_param in self.REQUIRED_PARAMETERS:
            if req_param not in self.additional_parameters:
                raise ValueError(
                    f"Parameter `{req_param}` is required for operator `{type(self).__name__}`."
                )

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
        elif key in self.additional_parameters:
            return self.additional_parameters[key]
        else:
            raise KeyError(f"Unknown attribute `{key}` for class `{type(self)}`")

    def __setitem__(self, key: str, value: Any):
        if key == "weight":
            self.weight = value
        elif key == "name":
            self.name = value
        else:
            raise ValueError(f"Attribute `{key}` cannot be set in class `{type(self)}`")

    def register_accept(self):
        self.accepts += 1

    def register_reject(self):
        self.rejects += 1

    @property
    def total(self):
        return self.accepts + self.rejects

    @property
    def acceptance_rate(self):
        return self.accepts / self.total

    @property
    def operator_name(self) -> str:
        return self.__class__.__name__


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

    STEP_PRECISION = 30

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
        **kwargs,
    ):
        super().__init__(weight=weight, **kwargs)
        self.model_by_chain = model_by_chain
        self.as_gibbs = as_gibbs
        self.sample_from_prior = sample_from_prior
        self.object_selector = object_selector
        self.min_size = 10
        self.max_size = 50

        if self.model_by_chain[0].shapes.n_sites <= self.min_size:
            self.object_selector = ObjectSelector.ALL

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
        sample_new = sample.copy()

        object_subset = self.select_object_subset(sample)

        if self.sample_from_prior:
            p = update_weights(sample)[object_subset]
        else:
            p = self.calculate_source_posterior(sample, object_subset)

        # Sample the new source assignments
        with sample_new.source.edit() as source:
            source[object_subset] = sample_categorical(p=p, binary_encoding=True)
        update_feature_counts(sample, sample_new, features, object_subset)

        if DEBUG:
            verify_counts(sample_new, features)

        # Transition probability forward:
        log_q = np.log(p[sample_new.source.value[object_subset]]).sum()

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

    def _propose(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        # The likelihood object contains relevant information on the areal and the confounding effect
        likelihood = self.model_by_chain[sample.chain].likelihood

        # Compute the old likelihood
        w = sample.weights.value
        w_normalized_old = update_weights(sample)
        log_lh_old = self.source_lh_by_feature(sample.source.value, w_normalized_old)

        # Resample the weights
        w_new, log_q, log_q_back = self.resample_weight_for_two_components(
            sample, likelihood
        )
        sample.weights.set_value(w_new)

        # Compute new likelihood
        w_new_normalized = update_weights(sample)
        log_lh_new = self.source_lh_by_feature(sample.source.value, w_new_normalized)

        # Add the prior to get the weight posterior (for each feature)
        log_prior_old = 0.0  # TODO add hyper prior on weights, when implemented
        log_prior_new = 0.0  # TODO add hyper prior on weights, when implemented
        log_p_old = log_lh_old + log_prior_old
        log_p_new = log_lh_new + log_prior_new

        # Compute hastings ratio for each feature and accept/reject independently
        p_accept = np.exp(log_p_new - log_p_old + log_q_back - log_q)
        accept = RNG.random(p_accept.shape) < p_accept
        sample.weights.set_value(np.where(accept[:, np.newaxis], w_new, w))

        assert ~np.any(np.isnan(sample.weights.value))

        # We already accepted/rejected for each feature independently
        # The result should always be accepted in the MCMC
        return sample, self.Q_GIBBS, self.Q_BACK_GIBBS

    def resample_weight_for_two_components(
        self, sample: Sample, likelihood: Likelihood
    ) -> NDArray[float]:
        w = sample.weights.value
        source = sample.source.value
        has_components = sample.cache.has_components.value

        # Fix weights for all but two random components
        i1, i2 = random.sample(range(sample.n_components), 2)

        # Select counts of the relevant languages
        has_both = np.logical_and(has_components[:, i1], has_components[:, i2])
        counts = np.sum(source[has_both, :, :], axis=0)
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
    def source_lh_by_feature(source, weights):
        # multiply and sum to get likelihood per source observation
        log_lh_per_observation = np.log(np.sum(source * weights, axis=-1))

        # sum over sites to obtain the total log-likelihood per feature
        return np.sum(log_lh_per_observation, axis=0)


class _AlterCluster(Operator):

    def __init__(
        self,
        *args,
        model_by_chain: list[Model],
        resample_source: bool,
        sample_from_prior: bool,
        p_grow: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_by_chain = model_by_chain
        self.resample_source = resample_source
        self.sample_from_prior = sample_from_prior
        self.p_grow = p_grow

    def _propose(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        if random.random() < self.p_grow:
            sample_new, log_q, log_q_back = self.grow_cluster(sample)
            log_q += np.log(self.p_grow)
            log_q_back += np.log(1 - self.p_grow)
        else:
            sample_new, log_q, log_q_back = self.shrink_cluster(sample)
            log_q += np.log(1 - self.p_grow)
            log_q_back += np.log(self.p_grow)

        if DEBUG:
            verify_counts(sample, self.model_by_chain[sample.chain].data.features.values)
            verify_counts(sample_new, self.model_by_chain[sample.chain].data.features.values)

        return sample_new, log_q, log_q_back


    @abstractmethod
    def grow_cluster(self, sample: Sample) -> tuple[Sample, float, float]:
        """Grow a clusters in the current sample (i.e. add a new site to one cluster)."""

    @abstractmethod
    def shrink_cluster(self, sample: Sample) -> tuple[Sample, float, float]:
        """Shrink a cluster in the current sample (i.e. remove one object from one cluster)."""

    @staticmethod
    def available(sample: Sample, i_cluster: int):
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
        self, sample_old: Sample, sample_new: Sample, changed_objects: list[int] | NDArray[int]
    ) -> tuple[Sample, float, float]:
        n_features = sample_old.n_features
        features = self.model_by_chain[sample_old.chain].data.features.values

        MODE = "gibbs"
        # MODE = "prior"

        if MODE == "gibbs":
            sample_new, log_q, log_q_back = self.gibbs_sample_source(
                sample_new, sample_old, object_subset=changed_objects
            )

        elif MODE == "prior":
            p = update_weights(sample_new)[changed_objects]
            p_back = update_weights(sample_old)[changed_objects]
            with sample_new.source.edit() as source:
                source[changed_objects, :, :] = sample_categorical(p, binary_encoding=True)
                log_q = np.log(p[source[changed_objects]]).sum()
            log_q_back = np.log(p_back[sample_old.source.value[changed_objects]]).sum()

            update_feature_counts(sample_old, sample_new, features, changed_objects)
            if DEBUG:
                verify_counts(sample_new, features)

        elif MODE == "uniform":
            has_components_new = sample_new.cache.has_components.value
            p = normalize(
                np.tile(has_components_new[changed_objects, None, :], (1, n_features, 1))
            )
            with sample_new.source.edit() as source:
                source[changed_objects, :, :] = sample_categorical(
                    p, binary_encoding=True
                )
                log_q = np.log(p[source[changed_objects]]).sum()

            has_components_old = sample_old.cache.has_components.value
            p_back = normalize(
                np.tile(has_components_old[changed_objects, None, :], (1, n_features, 1))
            )
            log_q_back = np.log(p_back[sample_old.source.value[changed_objects]]).sum()

            update_feature_counts(sample_old, sample_new, features, changed_objects)
            if DEBUG:
                verify_counts(sample_new, features)
        else:
            raise ValueError(f"Invalid mode `{MODE}`. Choose from (gibbs, prior and uniform)")

        return sample_new, log_q, log_q_back

    def gibbs_sample_source(
        self,
        sample_new: Sample,
        sample_old: Sample,
        object_subset: slice | list[int] | NDArray[int] = slice(None),
    ) -> tuple[Sample, float, float]:
        """Resample the observations to mixture components (their source)."""
        features = self.model_by_chain[sample_old.chain].data.features.values

        w = update_weights(sample_old)[object_subset]
        s = sample_old.source.value[object_subset]
        assert np.all(s <= (w > 0)), np.max(w)

        if self.sample_from_prior:
            p = update_weights(sample_new)[object_subset]
        else:
            p = self.calculate_source_posterior(sample_new, object_subset)

        # Sample the new source assignments
        with sample_new.source.edit() as source:
            source[object_subset] = sample_categorical(p=p, binary_encoding=True)

        update_feature_counts(sample_old, sample_new, features, object_subset)
        if DEBUG:
            verify_counts(sample_new, features)

        # Transition probability forward:
        log_q = np.log(p[sample_new.source.value[object_subset]]).sum()

        # Transition probability backward:
        if self.sample_from_prior:
            p_back = update_weights(sample_old)[object_subset]
        else:
            p_back = self.calculate_source_posterior(sample_old, object_subset,
                                                     feature_counts=sample_new.feature_counts)

        log_q_back = np.log(p_back[sample_old.source.value[object_subset]]).sum()

        return sample_new, log_q, log_q_back

    def calculate_source_posterior(
        self,
        sample: Sample,
        object_subset: slice | list[int] | NDArray[int] = slice(None),
        feature_counts: dict[str, ...] = None,
    ) -> NDArray[float]:  # shape: (n_objects_in_subset, n_features, n_components)
        """Compute the posterior support for source assignments of every object and feature."""

        # 1. compute likelihood for each component
        model: Model = self.model_by_chain[sample.chain]
        lh_per_component = likelihood_per_component(model=model, sample=sample, feature_counts=feature_counts, caching=True)

        # 2. multiply by weights and normalize over components to get the source posterior
        weights = update_weights(sample)
        return normalize(
            lh_per_component[object_subset] * weights[object_subset], axis=-1
        )


class AlterClusterGibbsish(_AlterCluster):
    def __init__(
        self,
        *args,
        adjacency_matrix,
        features: NDArray[bool],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adjacency_matrix = adjacency_matrix
        self.features = features

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

        # New:
        weights_z01 = self.compute_feature_weights_with_and_without(sample, available)
        feature_lh_z01 = inner1d(all_lh[np.newaxis, ...], weights_z01)
        marginal_lh_z01 = np.prod(feature_lh_z01, axis=-1)
        cluster_posterior = marginal_lh_z01[1] / (marginal_lh_z01[0] + marginal_lh_z01[1])

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

        # Draw new object according to posterior
        object_new = RNG.choice(sample.n_objects, p=p_add, replace=False)
        sample_new.clusters.add_object(i_cluster, object_new)

        # The removal probability of an inverse step
        shrink_candidates = self.shrink_candidates(sample_new, i_cluster)
        p_remove = normalize((1 - cluster_posterior) * shrink_candidates)

        log_q = np.log(p_add[object_new])
        log_q_back = np.log(p_remove[object_new])

        if self.resample_source and sample.source is not None:
            sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
                sample, sample_new, [object_new]
            )
            log_q += log_q_s
            log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back

    def shrink_cluster(self, sample: Sample) -> tuple[Sample, float, float]:
        # Choose a cluster
        i_cluster = RNG.choice(range(sample.n_clusters))
        cluster = sample.clusters.value[i_cluster, :]

        # Load and precompute useful variables
        model = self.model_by_chain[sample.chain]
        available = self.available(sample, i_cluster)
        candidates = self.shrink_candidates(sample, i_cluster)

        # If the cluster is already at min size, reject:
        if np.sum(cluster) == model.min_size:
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        # Otherwise create a new sample and continue with step:
        sample_new = sample.copy()

        # Assign probabilities for each unoccupied object
        cluster_posterior = np.zeros(sample.n_objects)
        cluster_posterior[available] = self.compute_cluster_posterior(
            sample, i_cluster, available
        )
        p_remove = normalize((1 - cluster_posterior) * candidates)

        # Draw new object according to posterior
        object_remove = RNG.choice(sample.n_objects, p=p_remove, replace=False)
        sample_new.clusters.remove_object(i_cluster, object_remove)

        # The add probability of an inverse step
        grow_candidates = self.grow_candidates(sample_new)
        p_add = normalize(cluster_posterior * grow_candidates)

        log_q = np.log(p_remove[object_remove])
        log_q_back = np.log(p_add[object_remove])

        if self.resample_source and sample.source is not None:
            sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
                sample, sample_new, [object_remove]
            )
            log_q += log_q_s
            log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back


class AlterClusterGibbsish2(AlterClusterGibbsish):

    def _propose(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        sample_new = sample.copy()
        i_cluster = RNG.choice(range(sample.n_clusters))
        cluster_old = sample.clusters.value[i_cluster]
        available = self.available(sample, i_cluster)
        n_available = np.count_nonzero(available)
        model = self.model_by_chain[sample.chain]

        p = self.compute_cluster_posterior(sample, i_cluster, available)
        # shape: (n_available,)

        cluster_new = (RNG.random(n_available) < p)
        if not (model.min_size <= np.count_nonzero(cluster_new) <= model.max_size):
            # Reject if proposal goes out of cluster size bounds
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        if np.all(cluster_new == cluster_old[available]):
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        q_per_site = p * cluster_new + (1 - p) * (1 - cluster_new)
        log_q = np.log(q_per_site).sum()
        q_back_per_site = p * cluster_old[available] + (1 - p) * (1 - cluster_old[available])
        log_q_back = np.log(q_back_per_site).sum()

        with sample_new.clusters.edit_cluster(i_cluster) as c:
            c[available] = cluster_new

        changed = np.where(cluster_old != sample_new.clusters.value[i_cluster])[0]

        if self.resample_source and sample.source is not None:
            sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
                sample, sample_new, changed
            )
            log_q += log_q_s
            log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back


class AlterCluster(_AlterCluster):

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

    def grow_cluster(self, sample: Sample) -> tuple[Sample, float, float]:
        """Grow a clusters in the current sample (i.e. add a new site to one cluster)."""
        sample_new = sample.copy()
        occupied = sample.clusters.any_cluster()

        # Randomly choose one of the clusters to modify
        z_id = RNG.choice(range(sample.clusters.n_clusters))
        cluster_current = sample.clusters.value[z_id, :]

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
        object_add = random.choice(candidates.nonzero()[0])
        sample_new.clusters.add_object(z_id, object_add)

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

        if self.resample_source:
            assert sample.source is not None
            sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
                sample, sample_new, [object_add]
            )
            log_q += log_q_s
            log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back

    def shrink_cluster(self, sample: Sample) -> tuple[Sample, float, float]:
        """Shrink a cluster in the current sample (i.e. remove one object from one cluster)."""
        sample_new = sample.copy()

        # Randomly choose one of the clusters to modify
        z_id = RNG.choice(range(sample.clusters.n_clusters))
        cluster_current = sample.clusters.value[z_id, :]

        # Check if cluster is big enough to shrink
        current_size = np.count_nonzero(cluster_current)
        if current_size <= self.model_by_chain[sample.chain].min_size:
            # Cluster is too small to shrink: don't modify the sample and reject the step (q_back = 0)
            return sample, 0, -np.inf

        # Cluster is big enough: shrink
        removal_candidates = self.get_removal_candidates(cluster_current)
        object_remove = random.choice(removal_candidates)
        sample_new.clusters.remove_object(z_id, object_remove)

        # Transition probability when shrinking.
        q = 1 / len(removal_candidates)
        # Back-probability (growing)
        cluster_new = sample_new.clusters.value[z_id]
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

        if self.resample_source:
            assert sample.source is not None
            sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
                sample, sample_new, [object_remove]
            )
            log_q += log_q_s
            log_q_back += log_q_back_s

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
