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
from scipy.special import softmax

from sbayes.load_data import Features, Data
from sbayes.sampling.conditionals import likelihood_per_component, conditional_effect_mean
from sbayes.sampling.counts import recalculate_feature_counts, update_feature_counts
from sbayes.sampling.state import Sample
from sbayes.util import dirichlet_logpdf, normalize, get_neighbours, inner1d, heat_binary_probability
from sbayes.util import EPS, RNG, FLOAT_TYPE
from sbayes.model import Model, Likelihood, normalize_weights, update_weights
from sbayes.preprocessing import sample_categorical
from sbayes.config.config import OperatorsConfig


DEBUG = 0


def get_operator_schedule(
    operators_config: OperatorsConfig,
    model: Model,
    data: Data,
    temperature: float = 1.0,
    prior_temperature: float = 1.0,
    sample_from_prior: bool = True,
) -> dict[str, Operator]:
    """Get all relevant operator objects for proposing MCMC update steps and their probabilities.
    Args:
        operators_config: dictionary with names of all operators (keys) and their weights (values)
        model: The sBayes model object (for Gibbsish operators)
        data: The data (for Gibbsish operators)
        temperature: temperature to flatten the likelihood (for MC3)
        prior_temperature: temperature to flatten prior distribution (for MC3)
        sample_from_prior: sample from the prior (rather than posterior)?
    Returns:
        Dictionary mapping operator names to operator objects
    """
    geo_prior = model.prior.geo_prior
    consider_geo_prior = (geo_prior.prior_type == geo_prior.prior_type.COST_BASED)

    operators = {
        'cluster_naive_n1': AlterCluster(
            weight=0.05 * operators_config.clusters,
            adjacency_matrix=data.network.adj_mat,
            model=model,
            features=data.features.values,
            resample_source_mode=ResampleSourceMode.GIBBS,
            sample_from_prior=sample_from_prior,
            n_changes=1,
            consider_geo_prior=False,
            temperature=temperature,
            prior_temperature=prior_temperature,
            neighbourhood=Neighbourhood.direct,
            gibbsish=False,
        ),
        'cluster_naive_n1_geo': AlterCluster(
            weight=0.05 * operators_config.clusters,
            adjacency_matrix=data.network.adj_mat,
            model=model,
            features=data.features.values,
            resample_source_mode=ResampleSourceMode.GIBBS,
            sample_from_prior=sample_from_prior,
            n_changes=1,
            consider_geo_prior=consider_geo_prior,
            temperature=temperature,
            prior_temperature=prior_temperature,
            neighbourhood=Neighbourhood.direct,
            gibbsish=False,
        ),
        'cluster_naive_n2_geo': AlterCluster(
            weight=0.05 * operators_config.clusters,
            adjacency_matrix=data.network.adj_mat,
            model=model,
            features=data.features.values,
            resample_source_mode=ResampleSourceMode.GIBBS,
            sample_from_prior=sample_from_prior,
            n_changes=1,
            consider_geo_prior=consider_geo_prior,
            temperature=temperature,
            prior_temperature=prior_temperature,
            neighbourhood=Neighbourhood.twostep,
            gibbsish=False,
        ),
        'cluster_gibbsish': AlterCluster(
            weight=0.05 * operators_config.clusters,
            adjacency_matrix=data.network.adj_mat,
            model=model,
            features=data.features.values,
            resample_source_mode=ResampleSourceMode.GIBBS,
            sample_from_prior=sample_from_prior,
            n_changes=1,
            temperature=temperature,
            prior_temperature=prior_temperature,
        ),
        'cluster_gibbsish_geo': AlterCluster(
            weight=0.35 * operators_config.clusters,
            adjacency_matrix=data.network.adj_mat,
            model=model,
            features=data.features.values,
            resample_source_mode=ResampleSourceMode.GIBBS,
            sample_from_prior=sample_from_prior,
            n_changes=1,
            consider_geo_prior=consider_geo_prior,
            temperature=temperature,
            prior_temperature=prior_temperature,
        ),
        # 'gibbsish_sample_cluster_2_geo': AlterClusterGibbsish(
        #     weight=0.05 * operators_config.clusters,
        #     adjacency_matrix=data.network.adj_mat,
        #     model=model,
        #     features=data.features.values,
        #     resample_source_mode=ResampleSourceMode.GIBBS,
        #     sample_from_prior=sample_from_prior,
        #     consider_geo_prior=consider_geo_prior,
        #     n_changes=2,
        # ),
        'gibbsish_sample_cluster_wide_geo': AlterClusterWide(
            weight=0.1 * operators_config.clusters,
            adjacency_matrix=data.network.adj_mat,
            model=model,
            features=data.features.values,
            resample_source_mode=ResampleSourceMode.GIBBS,
            sample_from_prior=sample_from_prior,
            w_stay=0.15,
            consider_geo_prior=consider_geo_prior,
            eps=0.01 / model.shapes.n_sites,
            temperature=temperature,
            prior_temperature=prior_temperature,
        ),
        'gibbsish_sample_cluster_wide_residual': AlterClusterWide(
            weight=0.05 * operators_config.clusters,
            adjacency_matrix=data.network.adj_mat,
            model=model,
            features=data.features.values,
            resample_source_mode=ResampleSourceMode.GIBBS,
            sample_from_prior=sample_from_prior,
            w_stay=0.0,
            cluster_effect_proposal=ClusterEffectProposals.residual_counts,
            consider_geo_prior=consider_geo_prior,
            eps=0.01 / model.shapes.n_sites,
            temperature=temperature,
            prior_temperature=prior_temperature,
        ),
        # 'gibbsish_sample_cluster_em': AlterClusterEM(
        #     weight=0.05 * operators_config.clusters,
        #     adjacency_matrix=data.network.adj_mat,
        #     model=model,
        #     features=data.features.values,
        #     resample_source_mode=ResampleSourceMode.GIBBS,
        #     sample_from_prior=sample_from_prior,
        #     w_stay=0.0,
        #     cluster_effect_proposal=ClusterEffectProposals.residual_counts,
        # ),
        'cluster_jump_gibbsish': ClusterJump(
            weight=0.25 * operators_config.clusters if model.n_clusters > 1 else 0.0,
            model=model,
            sample_from_prior=sample_from_prior,
            gibbsish=True,
            temperature=temperature,
            prior_temperature=prior_temperature,
        ),
        # 'gibbs_sample_sources_single': GibbsSampleSource(
        #     weight=1.0*operators_config.source,
        #     model=model,
        #     sample_from_prior=sample_from_prior,
        #     object_selector=ObjectSelector.RANDOM_SUBSET,
        #     max_size=1,
        #     temperature=temperature,
        #     prior_temperature=prior_temperature,
        # ),
        'gibbs_sample_sources': GibbsSampleSource(
            weight=0.4*operators_config.source,
            model=model,
            sample_from_prior=sample_from_prior,
            object_selector=ObjectSelector.RANDOM_SUBSET,
            max_size=20,
            temperature=temperature,
            prior_temperature=prior_temperature,
        ),
        'gibbs_sample_sources_groups': GibbsSampleSource(
            weight=0.6*operators_config.source,
            model=model,
            sample_from_prior=sample_from_prior,
            object_selector=ObjectSelector.GROUPS,
            max_size=30,
            temperature=temperature,
            prior_temperature=prior_temperature,
        ),
        'gibbs_sample_weights': GibbsSampleWeights(
            weight=operators_config.weights,
            model=model,
            sample_from_prior=sample_from_prior,
            temperature=temperature,
            prior_temperature=prior_temperature,
        ),
    }

    normalize_operator_weights(operators)

    return operators


def normalize_operator_weights(operators: dict[str, Operator]):
    total = sum(op.weight for op in operators.values())
    for op in operators.values():
        op.weight /= total





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

    def __init__(
        self,
        weight: float,
        temperature: float = 1.0,
        prior_temperature: float = 1.0
    ):
        # Operator weight
        self.weight = weight

        # Operator statistics
        self.accepts: int = 0
        self.rejects: int = 0
        self.step_times = []
        self.step_sizes = []
        self.next_step_times = []

        # Operator parameters
        self.temperature = temperature
        self.prior_temperature = prior_temperature

    def function(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        # ... potentially do some bookkeeping
        return self._propose(sample, **kwargs)

    @abstractmethod
    def _propose(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        """Propose a new state from the given one."""
        pass

    def register_accept(self, step_time: float, sample_old: Sample, sample_new: Sample, prev_operator: Operator = None):
        self.accepts += 1

        step_size = self.get_step_size(sample_old, sample_new)
        self.step_sizes.append(step_size)

        self.step_times.append(step_time)
        if prev_operator:
            prev_operator.next_step_times.append(step_time)

    def register_reject(self, step_time: float, prev_operator: Operator = None):
        self.rejects += 1
        self.step_times.append(step_time)
        # self.step_sizes.append(0)
        if prev_operator:
            prev_operator.next_step_times.append(step_time)

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
        params = {}
        if self.temperature != 1.0:
            params["temperature"] = self.temperature
        if self.prior_temperature != 1.0:
            params["prior_temperature"] = self.prior_temperature
        return params

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
        w_new = RNG.dirichlet(alpha).astype(FLOAT_TYPE)
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
        model: Model,
        as_gibbs: bool = True,
        sample_from_prior: bool = False,
        object_selector: ObjectSelector = ObjectSelector.GROUPS,
        max_size: int = 50,
        **kwargs,
    ):
        super().__init__(weight=weight, **kwargs)
        self.model = model
        self.as_gibbs = as_gibbs
        self.sample_from_prior = sample_from_prior
        self.object_selector = object_selector
        self.min_size = 10
        self.max_size = min(max_size, self.model.shapes.n_sites)

        if self.model.shapes.n_sites <= self.min_size:
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
        features = self.model.data.features.values
        na_features = self.model.data.features.na_values
        sample_new = sample.copy()

        assert np.all(sample.source.value[na_features] == 0)

        object_subset = self.select_object_subset(sample)

        if self.sample_from_prior:
            weights = update_weights(sample)[object_subset]
            p = normalize(weights ** (1 / self.prior_temperature), axis=-1)
        else:
            p = self.calculate_source_posterior(sample, object_subset)

        # Sample the new source assignments
        x = sample_categorical(p=p, binary_encoding=True)
        x[na_features[object_subset]] = False
        sample_new.source.set_groups(object_subset, x)
        # with sample_new.source.edit() as source:
        #     source[object_subset] = sample_categorical(p=p, binary_encoding=True)
        #     source[na_features] = 0

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

        # Compute likelihood for each component
        # lh_per_component = likelihood_per_component_exact(model=self.model, sample=sample)
        lh_per_component = likelihood_per_component(model=self.model, sample=sample, caching=True)

        # Multiply by weights and normalize over components to get the source posterior
        weights = update_weights(sample)

        # The posterior of the source for each observation is likelihood times prior
        # Apply temperature for likelihood and prior separately (for MC3)
        source_posterior = (
                lh_per_component[object_subset] ** (1 / self.temperature) *
                weights[object_subset] ** (1 / self.prior_temperature)
        )

        # Normalize to sum up to one across components
        return normalize(source_posterior, axis=-1)

    def get_step_size(self, sample_old: Sample, sample_new: Sample) -> float:
        return np.count_nonzero(sample_old.source.value ^ sample_new.source.value)

    STEP_SIZE_UNIT: str = "observations reassigned"


class GibbsSampleWeights(Operator):

    def __init__(
        self,
        *args,
        model: Model,
        sample_from_prior=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.sample_from_prior = sample_from_prior

        self.last_accept_rate = 0

    def _propose(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        # The likelihood object contains relevant information on the areal and the confounding effect
        model = self.model
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
        log_p_old = log_lh_old + log_prior_old
        log_p_new = log_lh_new + log_prior_new

        # Compute hastings ratio for each feature and accept/reject independently
        # TODO use temperature earlier in the proposal
        p_accept = np.exp((log_p_new - log_p_old + log_q_back - log_q) / self.prior_temperature)
        accept = RNG.random(p_accept.shape, dtype=FLOAT_TYPE) < p_accept
        sample.weights.set_value(np.where(accept[:, np.newaxis], w_new, w))

        assert ~np.any(np.isnan(sample.weights.value))

        self.last_accept_rate = np.mean(accept)

        # We already accepted/rejected for each feature independently
        # The result should always be accepted in the MCMC
        return sample, self.Q_GIBBS, self.Q_BACK_GIBBS

    def resample_weight_for_two_components(
        self, sample: Sample, likelihood: Likelihood
    ) -> NDArray[float]:
        model = self.model
        w = sample.weights.value
        source = sample.source.value
        has_components = sample.cache.has_components.value

        # Fix weights for all but two random components
        i1, i2 = random.sample(range(sample.n_components), 2)

        # Select counts of the relevant objects
        has_both = np.logical_and(has_components[:, i1], has_components[:, i2])
        counts = (
            np.sum(source[has_both, :, :], axis=0)           # likelihood counts
            + model.prior.prior_weights.concentration_array  # prior counts
        )
        c1 = counts[..., i1] / self.prior_temperature
        c2 = counts[..., i2] / self.prior_temperature

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
        model: Model,
        sample_from_prior: bool = False,
        p_grow: float = 0.5,
        n_changes: int = 1,
        p_grow_connected: float = 0.85,
        resample_source_mode: ResampleSourceMode = ResampleSourceMode.GIBBS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.resample_source_mode = resample_source_mode
        self.sample_from_prior = sample_from_prior
        self.p_grow = p_grow
        self.n_changes = n_changes
        self.p_grow_connected = p_grow_connected

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
        features = self.model.data.features.values
        na_features = self.model.data.features.na_values

        if self.resample_source_mode == ResampleSourceMode.GIBBS:
            sample_new, log_q, log_q_back = self.gibbs_sample_source(
                sample_new, sample_old, i_cluster, object_subset=object_subset
            )

        elif self.resample_source_mode == ResampleSourceMode.PRIOR:
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

        elif self.resample_source_mode == ResampleSourceMode.UNIFORM:
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
            raise ValueError(f"Invalid resample_source_mode `{self.resample_source_mode}` is not implemented.")

        return sample_new, log_q, log_q_back

    def gibbs_sample_source(
        self,
        sample_new: Sample,
        sample_old: Sample,
        i_cluster: int,
        object_subset: slice | list[int] | NDArray[int] = slice(None),
    ) -> tuple[Sample, float, float]:
        """Resample the observations to mixture components (their source)."""
        model = self.model
        features = model.data.features.values
        na_features = model.data.features.na_values

        # Make sure object_subset is a boolean index array
        object_subset = np.isin(np.arange(sample_new.n_objects), object_subset)

        # Compute the likelihood for each mixture component
        lh_per_component = component_likelihood_given_unchanged(
            model, sample_new, object_subset, i_cluster=i_cluster,
            temperature=self.temperature, prior_temperature=self.prior_temperature,
        )

        # Compute the source-prior for each mixture component
        if self.sample_from_prior:
            p = update_weights(sample_new)[object_subset] ** (1 / self.prior_temperature)
        else:
            w = update_weights(sample_new)[object_subset] ** (1 / self.prior_temperature)
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
            p_back = update_weights(sample_old)[object_subset] ** (1 / self.prior_temperature)
        else:
            w = update_weights(sample_old)[object_subset] ** (1 / self.prior_temperature)
            p_back = normalize(w * lh_per_component, axis=-1)
            # p_back = self.calculate_source_posterior(sample_old, object_subset)

        source_old = sample_old.source.value[object_subset]
        log_q_back = np.log(p_back[source_old]).sum()

        return sample_new, log_q, log_q_back

    def get_step_size(self, sample_old: Sample, sample_new: Sample) -> float:
        return np.count_nonzero(sample_old.clusters.value ^ sample_new.clusters.value)

    STEP_SIZE_UNIT: str = "objects reassigned"

    def get_parameters(self) -> dict[str, Any]:
        params = super().get_parameters()
        if self.resample_source_mode != ResampleSourceMode.GIBBS:
            params["resample_source_mode"] = self.resample_source_mode.value
        return params


def component_likelihood_given_unchanged(
    model: Model,
    sample: Sample,
    object_subset: NDArray[bool],
    i_cluster: int,
    temperature: float = 1.0,
    prior_temperature: float = 1.0,
) -> NDArray[float]:  # shape: (n_objects, n_feature, n_components)
    features = model.data.features
    confounders = model.data.confounders
    cluster = sample.clusters.value[i_cluster]
    source = sample.source.value
    subset_size = np.count_nonzero(object_subset)

    likelihoods = np.zeros((subset_size, sample.n_features, sample.n_components), dtype=FLOAT_TYPE)

    cluster_features = features.values * source[:, :, 0, None]
    # feature_counts_c = np.sum(cluster_features[cluster & ~object_subset], axis=0)
    cluster_effect = conditional_effect_mean(
        prior_counts=model.prior.prior_cluster_effect.concentration_array,
        feature_counts=np.sum(cluster_features[cluster & ~object_subset], axis=0),
        unif_counts=model.prior.prior_cluster_effect.uniform_concentration_array,
        prior_temperature=prior_temperature, temperature=temperature,
    )
    likelihoods[..., 0] = np.sum(cluster_effect[None, ...] * features.values[object_subset], axis=-1)

    # likelihoods[..., 0] = cluster_likelihood_given_unchanged(
    #     cluster, features, object_subset, source,
    #     prior_concentration=model.prior.prior_cluster_effect.concentration_array
    # )

    for i_conf, conf in enumerate(confounders, start=1):
        conf_prior = model.prior.prior_confounding_effects[conf]
        groups = confounders[conf].group_assignment

        features_conf = features.values * sample.source.value[:, :, i_conf, None]
        changeable_counts = np.array([
            np.sum(features_conf[g & object_subset], axis=0)
            for g in groups
        ])
        unchangeable_feature_counts = sample.feature_counts[conf].value - changeable_counts
        if conf_prior.any_dynamic_priors:
            prior_counts = conf_prior.concentration_array_given_unchanged(sample, changed_objects=object_subset)
        else:
            prior_counts = conf_prior.concentration_array(sample)

        conf_effect = conditional_effect_mean(
            prior_counts=prior_counts,
            feature_counts=unchangeable_feature_counts,
            unif_counts=conf_prior.uniform_concentration_array,
            prior_temperature=prior_temperature, temperature=temperature,
        )
        # conf_effect = normalize(unchangeable_feature_counts + prior_counts, axis=-1)

        # Calculate the likelihood of each observation in each group that is represented in object_subset
        subset_groups = groups[:, object_subset]
        group_in_subset = np.any(subset_groups, axis=1)
        features_subset = features.values[object_subset]
        for g, p_g in zip(subset_groups[group_in_subset], conf_effect[group_in_subset]):
            f_g = features_subset[g, :, :]
            likelihoods[g, :, i_conf] = np.einsum('ijk,jk->ij', f_g, p_g)

    # Fix likelihood of NA features to 1
    likelihoods[features.na_values[object_subset]] = 1.

    return likelihoods ** (1 / temperature)

# @njit(fastmath=True)
# def group_likelihood_given_unchanged(
#     features: NDArray[bool],
#     probs: NDArray,
#     object_subset: NDArray[bool],
#     groups: NDArray[bool],
#     out: NDArray[float],
# ):
#     # Calculate the likelihood of each observation in each group that is represented in object_subset
#     subset_groups = groups[:, object_subset]
#     group_in_subset = (np.sum(subset_groups, axis=1) > 0)
#     features_subset = features[object_subset]
#     for i in range(len(group_in_subset)):
#         g = subset_groups[i]
#         p_g = probs[i]
#         f_g = features_subset[g, :, :]
#         out[g, :] = np.sum(f_g * p_g[np.newaxis, ...], axis=-1)
#         # assert np.allclose(out[g, :], np.einsum('ijk,jk->ij', f_g, p_g))


def cluster_likelihood_given_unchanged(
    cluster: NDArray[bool],                 # (n_objects,)
    features: Features,
    object_subset: NDArray[bool],           # (n_objects,)
    source: NDArray[bool],                  # (n_objects, n_features, n_components)
    prior_concentration: NDArray[float],    # (n_features, n_states)
) -> NDArray[float]:  # (n_objects, n_features)
    cluster_features = features.values * source[:, :, 0, None]
    feature_counts_c = np.sum(cluster_features[cluster & ~object_subset], axis=0)
    p = normalize(prior_concentration + feature_counts_c, axis=-1)
    return np.sum(p[None, ...] * features.values[object_subset], axis=-1)


class Neighbourhood(Enum):
    direct = "direct"
    twostep = "twostep"
    everywhere = "everywhere"


class AlterCluster(ClusterOperator):

    def __init__(
        self,
        *args,
        adjacency_matrix,
        features: NDArray[bool],
        consider_geo_prior: bool = False,
        neighbourhood: Neighbourhood = Neighbourhood.everywhere,
        additive_smoothing: float = 1E-6,
        gibbsish: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adjacency_matrix = adjacency_matrix
        self.features = features
        self.consider_geo_prior = consider_geo_prior
        self.neighbourhood = neighbourhood
        self.additive_smoothing = additive_smoothing
        self.gibbsish = gibbsish

    def _propose(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        model = self.model

        log_q = 0.
        log_q_back = 0.
        all_direct_rejects = True

        for i in range(self.n_changes):
            i_cluster = RNG.choice(range(sample.n_clusters))
            size = sample.clusters.sizes[i_cluster]
            if size == model.min_size:
                grow = True
                log_q_back -= np.log(2)
            elif size == model.max_size:
                grow = False
                log_q_back -= np.log(2)
            else:
                grow = random.random() < self.p_grow

            if grow:
                sample_new, log_q_i, log_q_back_i = self.grow_cluster(sample, i_cluster=i_cluster)
                log_q_i += np.log(self.p_grow)
                log_q_back_i += np.log(1 - self.p_grow)
            else:
                sample_new, log_q_i, log_q_back_i = self.shrink_cluster(sample, i_cluster=i_cluster)
                log_q_i += np.log(1 - self.p_grow)
                log_q_back_i += np.log(self.p_grow)

            if log_q_back_i != self.Q_BACK_REJECT:
                all_direct_rejects = False
                sample = sample_new
                log_q += log_q_i
                log_q_back += log_q_back_i

        if DEBUG:
            verify_counts(sample, self.features)
            verify_counts(sample_new, self.features)

        if all_direct_rejects:
            # This avoids logging empty steps as accepted (more interpretable operator
            # stats and possibly caching advantages)
            return sample, self.Q_REJECT, self.Q_BACK_REJECT
        else:
            return sample_new, log_q, log_q_back

    def compute_cluster_posterior(
        self,
        sample: Sample,
        i_cluster: int,
        available: NDArray[bool],
    ) -> NDArray[float]:  # shape: (n_available, )
        model = self.model
        NAs = model.data.features.na_values[available, :]

        if self.sample_from_prior or not self.gibbsish:
            n_available = np.count_nonzero(available)
            return 0.5*np.ones(n_available)

        p = conditional_effect_mean(
            prior_counts=model.prior.prior_cluster_effect.concentration_array,
            feature_counts=sample.feature_counts['clusters'].value[[i_cluster]],
            unif_counts=model.prior.prior_cluster_effect.uniform_concentration_array,
            prior_temperature=self.prior_temperature, temperature=self.temperature,
        )
        cluster_lh_z = inner1d(self.features[available], p)
        all_lh = deepcopy(likelihood_per_component(model, sample, caching=True)[available, :])
        all_lh[..., 0] = cluster_lh_z
        all_lh[NAs, 0] = 1.

        weights_z01 = self.compute_feature_weights_with_and_without(sample, available)
        feature_lh_z01 = inner1d(all_lh[np.newaxis, ...], weights_z01)
        marginal_lh_z01 = np.prod(feature_lh_z01, axis=-1) ** (1 / self.temperature)
        cluster_posterior = marginal_lh_z01[1] / (marginal_lh_z01[0] + marginal_lh_z01[1])

        if self.consider_geo_prior:
            cluster_posterior *= np.exp(model.prior.geo_prior.get_costs_per_object(sample, i_cluster)[available] / self.prior_temperature)

        if self.additive_smoothing > 0:
            # Add the additive smoothing constant and renormalize
            a = self.additive_smoothing
            cluster_posterior = (cluster_posterior + a) / (1 + 2 * a)

        return cluster_posterior

    def compute_feature_weights_with_and_without(
        self,
        sample: Sample,
        available: NDArray[bool],   # shape: (n_objects, )
    ) -> NDArray[float]:            # shape: (2, n_objects, n_features, n_components)
        weights_current = update_weights(sample, caching=True)[available]
        weights_current = normalize(weights_current ** (1 / self.prior_temperature), axis=-1)
        # weights = normalize_weights(sample.weights.value, has_components)

        has_components = deepcopy(sample.cache.has_components.value[available, :])
        has_components[:, 0] = ~has_components[:, 0]
        weights_flipped = normalize_weights(
            weights=sample.weights.value ** (1 / self.prior_temperature),
            has_components=has_components
        )

        weights_z01 = np.empty((2, *weights_current.shape))
        weights_z01[1] = np.where(has_components[:, np.newaxis, [0]], weights_flipped, weights_current)
        weights_z01[0] = np.where(has_components[:, np.newaxis, [0]], weights_current, weights_flipped)

        return weights_z01

    def grow_candidates(self, sample: Sample, i_cluster: int) -> NDArray[bool]:
        cluster_current = sample.clusters.value[i_cluster]
        occupied = sample.clusters.any_cluster()
        if self.neighbourhood == Neighbourhood.everywhere:
            return ~occupied
        elif self.neighbourhood == Neighbourhood.direct:
            return get_neighbours(cluster_current, occupied, self.adjacency_matrix)
        elif self.neighbourhood == Neighbourhood.twostep:
            return get_neighbours(cluster_current, occupied, self.adjacency_matrix, indirection=1)
        else:
            raise ValueError(f"Unknown neighborhood {self.neighbourhood}")

    @staticmethod
    def shrink_candidates(sample: Sample, i_cluster: int) -> NDArray[bool]:
        return sample.clusters.value[i_cluster]

    def grow_cluster(
        self,
        sample: Sample,
        i_cluster: int = None
    ) -> tuple[Sample, float, float]:

        if i_cluster is None:
            # Choose a cluster
            i_cluster = RNG.choice(range(sample.n_clusters))

        cluster = sample.clusters.value[i_cluster, :]

        # Load and precompute useful variables
        candidates = self.grow_candidates(sample, i_cluster)

        # If all the space is take we can't grow
        if not np.any(candidates):
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        # If the cluster is already at max size, reject:
        if np.sum(cluster) == self.model.max_size:
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        # Otherwise create a new sample and continue with step:
        sample_new = sample.copy()

        # Assign probabilities for each unoccupied object
        cluster_posterior = np.zeros(sample.n_objects, dtype=FLOAT_TYPE)
        cluster_posterior[candidates] = heat_binary_probability(
            self.compute_cluster_posterior(sample, i_cluster, candidates),
            self.temperature
        )
        p_add = normalize(cluster_posterior)

        if np.sum(p_add) == 0:
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        # Draw new object according to posterior
        object_add = RNG.choice(sample.n_objects, p=p_add, replace=False)
        sample_new.clusters.add_object(i_cluster, object_add)

        sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
            sample_old=sample,
            sample_new=sample_new,
            i_cluster=i_cluster,
            object_subset=[object_add],
        )

        # The removal probability of an inverse step
        shrink_candidates = self.shrink_candidates(sample_new, i_cluster)
        cluster_posterior_back = self.compute_cluster_posterior(sample_new, i_cluster, shrink_candidates)
        cluster_posterior_back = heat_binary_probability(cluster_posterior_back, self.temperature)
        assert np.all(cluster_posterior_back > 0) and np.all(cluster_posterior_back < 1)

        p_remove = np.empty(sample_new.n_objects, dtype=FLOAT_TYPE)
        p_remove[shrink_candidates] = normalize(1 - cluster_posterior_back)

        log_q = np.log(p_add[object_add]).sum() + log_q_s
        log_q_back = np.log(p_remove[object_add]).sum() + log_q_back_s

        return sample_new, log_q, log_q_back

    def shrink_cluster(
        self,
        sample: Sample,
        i_cluster: int = None
    ) -> tuple[Sample, float, float]:

        if i_cluster is None:
            # Choose a cluster
            i_cluster = RNG.choice(range(sample.n_clusters))

        # Load and precompute useful variables
        available = self.available(sample, i_cluster)
        candidates = self.shrink_candidates(sample, i_cluster)
        n_candidates = candidates.sum()

        # If the cluster is already at min size, reject:
        assert n_candidates > 0
        if n_candidates == self.model.min_size:
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        # Otherwise create a new sample and continue with step:
        sample_new = sample.copy()

        # Assign probabilities for each unoccupied object
        cluster_posterior = np.zeros(sample.n_objects, dtype=FLOAT_TYPE)
        cluster_posterior[candidates] = heat_binary_probability(
            self.compute_cluster_posterior(sample, i_cluster, candidates),
            self.temperature
        )

        x = (1 - cluster_posterior) * candidates
        if np.sum(x) == 0:
            return sample, self.Q_REJECT, self.Q_BACK_REJECT
        p_remove = normalize(x)

        # Draw object to be removed according to posterior
        removed_objects = RNG.choice(sample.n_objects, p=p_remove, size=1, replace=False)
        for obj in removed_objects:
            sample_new.clusters.remove_object(i_cluster, obj)

        sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
            sample_old=sample,
            sample_new=sample_new,
            i_cluster=i_cluster,
            object_subset=removed_objects,
        )

        # The add probability of an inverse step
        grow_candidates = self.grow_candidates(sample_new, i_cluster)
        if np.count_nonzero(grow_candidates) == 0:
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        cluster_posterior_back = self.compute_cluster_posterior(sample_new, i_cluster, grow_candidates)
        cluster_posterior_back = heat_binary_probability(cluster_posterior_back, self.temperature)

        p_add = np.empty(sample_new.n_objects, dtype=FLOAT_TYPE)
        p_add[grow_candidates] = normalize(cluster_posterior_back)

        log_q = np.log(p_remove[removed_objects]).sum() + log_q_s
        log_q_back = np.log(p_add[removed_objects]).sum() + log_q_back_s

        return sample_new, log_q, log_q_back

    def get_parameters(self) -> dict[str, Any]:
        params = super().get_parameters()
        if self.n_changes > 1:
            params["n_changes"] = self.n_changes
        if self.consider_geo_prior:
            params["geo"] = self.consider_geo_prior
        if self.neighbourhood != Neighbourhood.everywhere:
            params["neighbours"] = self.neighbourhood.value
        if self.additive_smoothing > 1E-6:
            params["additive_smoothing"] = self.additive_smoothing
        if not self.gibbsish:
            params["gibbsish"] = self.gibbsish
        return params


class ClusterEffectProposals:

    @staticmethod
    def posterior_counts(unif_counts, prior_counts, feature_counts, temperature, prior_temperature):
        """Compute the posterior counts (concentration parameter) for a Dirichlet
        distribution from prior counts and feature counts, but apply a different
        temperature for prior and likelihood."""
        # print(unif_counts.shape, prior_counts.shape, feature_counts.shape, temperature, prior_temperature)
        return (unif_counts + (prior_counts - unif_counts) / prior_temperature + feature_counts / temperature)

    @staticmethod
    def gibbs(
        model: Model,
        sample: Sample,
        i_cluster: int,
        temperature: float = 1.0,
        prior_temperature: float = 1.0
    ) -> NDArray[float]:
        c = ClusterEffectProposals.posterior_counts(
            unif_counts=model.prior.prior_cluster_effect.uniform_concentration_array,
            prior_counts=model.prior.prior_cluster_effect.concentration_array,
            feature_counts=sample.feature_counts['clusters'].value[[i_cluster]],
            temperature=temperature,
            prior_temperature=prior_temperature,
        )
        return normalize(c, axis=-1)

    @staticmethod
    def residual(
        model: Model,
        sample: Sample,
        i_cluster: int,
        temperature: float = 1.0,
        prior_temperature: float = 1.0
    ) -> NDArray[float]:
        features = model.data.features.values
        free_objects = ~sample.clusters.any_cluster()

        # Create posterior counts for free objects
        c = ClusterEffectProposals.posterior_counts(
            unif_counts=model.prior.prior_cluster_effect.uniform_concentration_array,
            prior_counts=model.prior.prior_cluster_effect.concentration_array,
            feature_counts=features[free_objects].sum(axis=0),
            temperature=temperature,
            prior_temperature=prior_temperature,
        )
        return normalize(c, axis=-1)

    @staticmethod
    def residual_counts(
        model: Model,
        sample: Sample,
        i_cluster: int,
        temperature: float = 1.0,
        prior_temperature: float = 1.0
    ) -> NDArray[float]:
        features = model.data.features.values

        cluster = sample.clusters.value[i_cluster]
        size = np.count_nonzero(cluster)
        free_objects = (~sample.clusters.any_cluster()) | cluster
        n_free = np.count_nonzero(free_objects)

        # Create counts in free objects
        unif_counts = model.prior.prior_cluster_effect.uniform_concentration_array
        prior_counts = model.prior.prior_cluster_effect.concentration_array
        exp_features_conf = ClusterEffectProposals.expected_confounder_features(
            model, sample, temperature, prior_temperature
        )
        residual_features = np.clip(features[free_objects] - exp_features_conf[free_objects], 0, None)
        residual_counts = np.sum(residual_features, axis=0)

        # The expected effect is given by the normalized posterior counts
        c = ClusterEffectProposals.posterior_counts(
            unif_counts, prior_counts, residual_counts, temperature, prior_temperature
        )
        p = normalize(c, axis=-1)
        # p = normalize(residual_counts + prior_counts, axis=-1)

        # Only consider objects with likelihood contribution above median
        lh = np.sum(p * residual_features, axis=(1,2))
        relevant = lh >= np.quantile(lh, 1-size/n_free)
        residual_counts = np.sum(residual_features[relevant], axis=0)

        # The expected effect is given by the normalized posterior counts
        c = ClusterEffectProposals.posterior_counts(
            unif_counts, prior_counts, residual_counts, temperature, prior_temperature
        )
        return normalize(c, axis=-1)
        # return normalize(residual_counts + prior_counts, axis=-1)

    @staticmethod
    def expected_confounder_features(
            model: Model, sample: Sample, temperature: float, prior_temperature: float
        ) -> NDArray[float]:
        """Compute the expected value for each feature according to the mixture of confounders."""
        expected_features = np.zeros((sample.n_objects, sample.n_features, sample.n_states), dtype=FLOAT_TYPE)
        weights = update_weights(sample, caching=False)
        weights_heated = normalize(weights ** (1 / prior_temperature), axis=-1)

        confounders = model.data.confounders
        unif_counts = model.prior.prior_cluster_effect.uniform_concentration_array

        # Iterate over mixture components
        for i_comp, (name_conf, conf) in enumerate(confounders.items(), start=1):
            conf_prior = model.prior.prior_confounding_effects[name_conf]

            # if conf_prior.any_dynamic_priors:
            #     prior_counts = ...
            # else:
            prior_counts = conf_prior.concentration_array(sample)

            # The expected confounding effect is given by the prior counts and feature
            # counts normalized to 1 over the different states.
            c = ClusterEffectProposals.posterior_counts(
                unif_counts=unif_counts,
                prior_counts=prior_counts,
                feature_counts=sample.feature_counts[name_conf].value,
                temperature=temperature,
                prior_temperature=prior_temperature,
            )
            p_conf = normalize(c, axis=-1)  # (n_features, n_states)
            # p_conf = normalize(feature_counts + prior_counts, axis=-1)  # (n_features, n_states)

            # The weighted sum of the confounding effects defines the expected features
            for i_g, g in enumerate(conf.group_assignment):
                expected_features[g] += weights_heated[g, :, [i_comp], None] * p_conf[np.newaxis, i_g, ...]

        return expected_features


class AlterClusterWide(AlterCluster):

    def __init__(
        self,
        *args,
        w_stay: float = 0.1,
        eps: float = 0.000001,
        cluster_effect_proposal: callable = ClusterEffectProposals.gibbs,
        geo_scaler: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.w_stay = w_stay
        self.eps = eps
        self.cluster_effect_proposal = cluster_effect_proposal
        self.geo_scaler = geo_scaler

    def compute_cluster_probs(self, sample, i_cluster, available):
        cluster = sample.clusters.value[i_cluster]
        p = self.compute_raw_cluster_probs(sample, i_cluster, available)
        p = normalize(p + EPS)

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

    def compute_raw_cluster_probs(
        self,
        sample: Sample,
        i_cluster: int,
        available: NDArray[bool],   # Shape: (n_objects,)
    ) -> NDArray[float]:            # shape: (n_available,)
        """Compute the proposal probability of each objects to be in cluster `i_cluster`
        (conditioned on other parameters in `sample`).

        Args:
            sample: The current MCMC Samples to be modified in this operator.
            i_cluster: The ID of the cluster to be changed in this operator.
            available: Boolean array indicating which objects can be added/removed.
        """
        model = self.model
        NAs = model.data.features.na_values[available]
        n_available = np.count_nonzero(available)

        if self.sample_from_prior:
            return 0.5 * np.ones(n_available)

        p = self.cluster_effect_proposal(
            model, sample, i_cluster, self.temperature, self.prior_temperature
        )
        cluster_lh_z = inner1d(self.features[available], p) ** (1 / self.temperature)
        all_lh = deepcopy(likelihood_per_component(model, sample, caching=True)[available, :])
        all_lh[..., 0] = cluster_lh_z
        all_lh[NAs, 0] = 1.

        weights_z01 = self.compute_feature_weights_with_and_without(sample, available)
        feature_lh_z01 = inner1d(all_lh[np.newaxis, ...], weights_z01)
        marginal_lh_z01 = np.prod(feature_lh_z01, axis=-1) ** (1 / self.temperature)
        cluster_posterior = marginal_lh_z01[1] / (marginal_lh_z01[0] + marginal_lh_z01[1] + EPS)

        if self.consider_geo_prior:
            if self.cluster_effect_proposal is ClusterEffectProposals.residual_counts:
                distances = model.data.geo_cost_matrix[available][:, available]
                z = normalize(cluster_posterior)
                avg_dist_to_cluster = z.dot(distances)
                geo_likelihoods = np.exp(-avg_dist_to_cluster / model.prior.geo_prior.scale / self.prior_temperature / self.geo_scaler)
                cluster_posterior = normalize(geo_likelihoods * z)
            else:
                cluster_posterior *= np.exp(model.prior.geo_prior.get_costs_per_object(sample, i_cluster)[available] / self.geo_scaler)

        return cluster_posterior

    def ml_step(self, sample: Sample, i_cluster=None) -> Sample:
        sample_new = sample.copy()
        model = self.model
        if i_cluster is None:
            i_cluster = RNG.choice(range(sample.n_clusters))

        available = self.available(sample, i_cluster)
        cluster_old = sample.clusters.value[i_cluster]
        p = self.compute_cluster_probs(sample, i_cluster, available)

        size = np.count_nonzero(cluster_old)
        size = np.clip(size, model.min_size, model.max_size)
        threshold = np.sort(p)[-size]
        cluster_new = p >= threshold

        if not (model.min_size <= np.count_nonzero(cluster_new) <= model.max_size):
            # Reject if proposal goes out of cluster size bounds
            return sample

        with sample_new.clusters.edit_cluster(i_cluster) as c:
            c[available] = cluster_new

        changed, = np.where(sample.clusters.value[i_cluster] != sample_new.clusters.value[i_cluster])
        sample_new, _, _ = self.propose_new_sources(
            sample_old=sample, sample_new=sample_new,
            i_cluster=i_cluster, object_subset=changed,
        )

        return sample_new

    def _propose(self, sample: Sample, i_cluster=None, **kwargs) -> tuple[Sample, float, float]:
        sample_new = sample.copy()
        if i_cluster is None:
            i_cluster = RNG.choice(range(sample.n_clusters))
        cluster_old = sample.clusters.value[i_cluster]
        available = self.available(sample, i_cluster)
        n_available = np.count_nonzero(available)
        model = self.model

        p = self.compute_cluster_probs(sample, i_cluster, available)

        cluster_new = (RNG.random(n_available, dtype=FLOAT_TYPE) < p)
        while np.all(cluster_new == cluster_old[available]):
            cluster_new = (RNG.random(n_available, dtype=FLOAT_TYPE) < p)

        if not (model.min_size <= np.count_nonzero(cluster_new) <= model.max_size):
            # Reject if proposal goes out of cluster size bounds
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        q_per_site = np.where(cluster_new, p, 1-p)
        log_q = np.log(q_per_site).sum()

        # Adjust log_q to reflect the fact that we don't allow stand-stills in the operator
        p_standstill = np.where(cluster_old[available], p, 1-p).prod()
        log_q -= np.log(1 - p_standstill)

        with sample_new.clusters.edit_cluster(i_cluster) as c:
            c[available] = cluster_new

        # Resample the source assignment for the changed objects
        changed, = np.where(cluster_old != sample_new.clusters.value[i_cluster])
        sample_new, log_q_s, log_q_back_s = self.propose_new_sources(
            sample_old=sample, sample_new=sample_new,
            i_cluster=i_cluster, object_subset=changed,
        )

        p_back = self.compute_cluster_probs(sample_new, i_cluster, available)
        q_back_per_site = np.where(cluster_old[available], p_back, 1 - p_back)
        log_q_back = np.log(q_back_per_site).sum()

        # Adjust log_q_back to reflect the fact that we don't allow stand-stills in the operator
        p_standstill_back = np.where(cluster_new, p_back, 1-p_back).prod()
        log_q_back -= np.log(1 - p_standstill_back)

        assert np.all(p_back > 0)
        assert np.all(q_back_per_site > 0), q_back_per_site

        log_q += log_q_s
        log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back

    def get_parameters(self) -> dict[str, Any]:
        params = super().get_parameters()

        if self.w_stay != 0.0:
            params["w_stay"] = self.w_stay
        if self.cluster_effect_proposal != ClusterEffectProposals.gibbs:
            params["cluster_effect_proposal"] = self.cluster_effect_proposal.__name__
        if self.eps != 0.000001:
            params["eps"] = self.eps
        if self.geo_scaler != 1.0:
            params["geo_scaler"] = self.geo_scaler

        return params


class AlterClusterEM(AlterClusterWide):

    def compute_cluster_probs(self, sample, i_cluster, available, n_steps=10):
        model = self.model
        cluster = sample.clusters.value[i_cluster]

        features = model.data.features.values
        valid_observations = ~model.data.features.na_values
        n_objects, n_features, n_states = features.shape
        n_clusters = model.n_clusters

        if self.sample_from_prior:
            n_available = np.count_nonzero(available)
            return 0.5 * np.ones(n_available)

        p_clust = self.cluster_effect_proposal(
            model, sample, i_cluster, self.temperature, self.prior_temperature
        )

        groups = [f"a{c}" for c in range(n_clusters)]
        for conf_name, conf in model.data.confounders.items():
            groups += [f"{conf_name}_{grp_name}" for grp_name in conf.group_names]
        n_groups = len(groups)

        groups_available = np.zeros((n_groups, n_objects), dtype=bool)
        groups_available[:n_clusters, :] = True
        i = n_clusters
        for conf_name, conf in model.data.confounders.items():
            groups_available[i:i+conf.n_groups, :] = conf.group_assignment
            i += conf.n_groups

        prior_counts = 0.5 * model.data.features.states

        z = np.ones((n_groups, n_objects)) * groups_available
        z[:n_clusters, :] = sample.clusters.value.astype(float)
        z[i_cluster, available] = 1.0
        z = normalize(z, axis=0)
        # z = np.mean(sample.source.value, axis=1)

        distances = model.data.geo_cost_matrix

        _features = np.copy(features)
        _features[~valid_observations, :] = 1

        for i_step in range(n_steps):
            state_counts = np.einsum("ij,jkl->ikl", z, features, optimize='optimal')
            # shape: (n_groups, n_features, n_states)

            p = normalize(state_counts + prior_counts, axis=-1)
            # shape: (n_groups, n_features, n_states)

            if i_step == 0:
                p[i_cluster] = p_clust

            # How likely would each feature observation be if it was explained by each group
            pointwise_likelihood_by_group = np.einsum("ikl,jkl->ijk", p, _features, optimize='optimal')
            # shape: (n_groups, n_objects, n_features)

            pointwise_likelihood_by_group[:, ~valid_observations] = 1
            group_likelihoods = np.prod(pointwise_likelihood_by_group, axis=-1)
            # shape: (n_groups, n_objects)

            if self.consider_geo_prior:
                z_peaky = softmax(n_objects * z, axis=1)
                avg_dist_to_cluster = z_peaky.dot(distances)
                geo_likelihoods = np.exp(-avg_dist_to_cluster / model.prior.geo_prior.scale / 2)
                geo_likelihoods[n_clusters:] = np.mean(geo_likelihoods[:n_clusters])
            else:
                geo_likelihoods = 1

            temperature = (n_steps / (1+i_step)) ** 2
            lh = (geo_likelihoods * group_likelihoods ** (1/temperature))
            z_unnoralized = lh * groups_available
            z_unnoralized[i_cluster, ~available] = 0
            z = normalize(z_unnoralized, axis=0)
            # shape: (n_groups, n_objects)

        # Normalize cluster probabilities across objects to ensure that adding eps has a predictable effect
        z_cluster = normalize(z[i_cluster, available])

        # For more local steps: proposal is a mixture of posterior and current cluster
        z_cluster = (1 - self.w_stay) * normalize(z_cluster + self.eps) + self.w_stay * normalize(cluster[available])

        # Expected size should be the same as current size
        old_size = np.sum(cluster[available])
        new_expected_size = np.sum(z)
        for _ in range(10):
            z_cluster = z_cluster * old_size / new_expected_size
            z_cluster = z_cluster.clip(self.eps, 1-self.eps)

            new_expected_size = np.sum(z_cluster)
            if new_expected_size > 0.975 * old_size:
                break

        return z_cluster


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
        model = self.model
        features = model.data.features.values
        NAs = model.data.features.na_values
        source_cluster = sample.clusters.value[i_source_cluster]
        weights = update_weights(sample)
        weights_heated = normalize(weights ** (1 / self.prior_temperature), axis=-1)
        w_clust = weights_heated[source_cluster, :, 0]

        p_clust_source = conditional_effect_mean(
            prior_counts=model.prior.prior_cluster_effect.concentration_array,
            feature_counts=sample.feature_counts['clusters'].value[[i_source_cluster]],
            unif_counts=model.prior.prior_cluster_effect.uniform_concentration_array,
            temperature=self.temperature, prior_temperature=self.prior_temperature,
        )
        p_clust_target = conditional_effect_mean(
            prior_counts=model.prior.prior_cluster_effect.concentration_array,
            feature_counts=sample.feature_counts['clusters'].value[[i_target_cluster]],
            unif_counts=model.prior.prior_cluster_effect.uniform_concentration_array,
            temperature=self.temperature, prior_temperature=self.prior_temperature,
        )
        p_conf = ClusterEffectProposals.expected_confounder_features(
            model, sample, temperature=self.temperature, prior_temperature=self.prior_temperature
        )[source_cluster]

        p_total_source = p_conf + w_clust[..., np.newaxis] * p_clust_source
        p_total_target = p_conf + w_clust[..., np.newaxis] * p_clust_target

        lh_stay_per_feature = np.sum(features[source_cluster] * p_total_source, axis=-1)
        lh_stay = np.prod(lh_stay_per_feature, axis=-1, where=~NAs[source_cluster])
        lh_jump_per_feature = np.sum(features[source_cluster] * p_total_target, axis=-1)
        lh_jump = np.prod(lh_jump_per_feature, axis=-1, where=~NAs[source_cluster])

        # Apply temperature (for MC3)
        lh_stay **= (1 / self.temperature)
        lh_jump **= (1 / self.temperature)

        # Avoid NaN values by adding EPS to both outcomes
        lh_stay += EPS
        lh_jump += EPS

        # TODO include geo prior?

        return lh_jump / (lh_jump + lh_stay)

    def _propose(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        """Reassign an object from one cluster to another."""
        sample_new = sample.copy()

        # Randomly choose a source and target cluster
        i_source_cluster, i_target_cluster = RNG.choice(range(sample.clusters.n_clusters), size=2, replace=False)
        source_cluster = sample.clusters.value[i_source_cluster, :]
        target_cluster = sample.clusters.value[i_target_cluster, :]

        # Check if clusters are too small/large for jump
        source_size = np.count_nonzero(source_cluster)
        target_size = np.count_nonzero(target_cluster)
        if source_size <= self.model.min_size or target_size >= self.model.max_size:
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

        sample_new, log_q_s, log_q_back_s = self.gibbs_sample_source_jump(
            sample_new=sample_new,
            sample_old=sample,
            i_cluster_new=i_target_cluster,
            i_cluster_old=i_source_cluster,
            object_subset=[jumping_object],
        )

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
        model = self.model
        features = model.data.features.values
        na_features = model.data.features.na_values

        # Make sure object_subset is a boolean index array
        object_subset = np.isin(np.arange(sample_new.n_objects), object_subset)
        w = update_weights(sample_new)[object_subset]

        if self.sample_from_prior:
            p = w
        else:
            lh_per_component_new = component_likelihood_given_unchanged(
                model, sample_new, object_subset, i_cluster=i_cluster_new,
                temperature=self.temperature, prior_temperature=self.prior_temperature,
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
                model, sample_old, object_subset, i_cluster=i_cluster_old,
                temperature=self.temperature, prior_temperature=self.prior_temperature,
            )
            p_back = normalize(w * lh_per_component_old, axis=-1)

        source_old = sample_old.source.value[object_subset]
        log_q_back = np.log(p_back[source_old]).sum()

        return sample_new, log_q, log_q_back


class ClusterJump2(ClusterOperator):

    def __init__(
        self,
        *args,
        gibbsish: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gibbsish = gibbsish

    def get_jump_lh(self, sample: Sample, i_source_cluster: int, i_target_cluster: int) -> NDArray[float]:
        model = self.model
        features = model.data.features.values
        source_cluster = sample.clusters.value[i_source_cluster]
        w_clust = update_weights(sample)[source_cluster, :, 0]

        p_clust_source = conditional_effect_mean(
            prior_counts=model.prior.prior_cluster_effect.concentration_array,
            feature_counts=sample.feature_counts['clusters'].value[[i_source_cluster]],
            unif_counts=model.prior.prior_cluster_effect.uniform_concentration_array,
            temperature=self.temperature, prior_temperature=self.prior_temperature,
        )
        p_clust_target = conditional_effect_mean(
            prior_counts=model.prior.prior_cluster_effect.concentration_array,
            feature_counts=sample.feature_counts['clusters'].value[[i_target_cluster]],
            unif_counts=model.prior.prior_cluster_effect.uniform_concentration_array,
            temperature=self.temperature, prior_temperature=self.prior_temperature,
        )
        p_conf = ClusterEffectProposals.expected_confounder_features(
            model, sample, temperature=self.temperature, prior_temperature=self.prior_temperature
        )[source_cluster]

        p_total_source = p_conf + w_clust[..., np.newaxis] * p_clust_source
        p_total_target = p_conf + w_clust[..., np.newaxis] * p_clust_target

        lh_stay_per_feature = np.sum(features[source_cluster] * p_total_source, axis=-1)
        lh_stay = np.prod(lh_stay_per_feature, axis=-1,
                          where=~model.data.features.na_values[source_cluster])
        lh_jump_per_feature = np.sum(features[source_cluster] * p_total_target, axis=-1)
        lh_jump = np.prod(lh_jump_per_feature, axis=-1,
                          where=~model.data.features.na_values[source_cluster])

        return lh_jump / (lh_jump + lh_stay)

    def get_q_target_cluster(self, sample: Sample, i_source_cluster: int) -> NDArray[float]:
        # Calculate transition probability based on
        return


    def _propose(self, sample: Sample, **kwargs) -> tuple[Sample, float, float]:
        """Reassign an object from one cluster to another."""
        sample_new = sample.copy()
        model = self.model

        # Randomly choose a source and target cluster
        i_source_cluster = RNG.choice(sample.clusters.n_clusters)
        p_clust_source = conditional_effect_mean(
            prior_counts=model.prior.prior_cluster_effect.concentration_array,
            feature_counts=sample.feature_counts['clusters'].value[[i_source_cluster]]
        )

        q_target_clust = self.get_q_target_cluster(sample, i_source_cluster=i_source_cluster)
        q_target_clust[i_source_cluster] = 0.0
        i_target_cluster = RNG.choice(sample.clusters.n_clusters, p=q_target_clust)

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

        sample_new, log_q_s, log_q_back_s = self.gibbs_sample_source_jump(
            sample_new=sample_new,
            sample_old=sample,
            i_cluster_new=i_target_cluster,
            i_cluster_old=i_source_cluster,
            object_subset=[jumping_object],
        )

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
        features = self.model.data.features.values
        na_features = self.model.data.features.na_values

        # Make sure object_subset is a boolean index array
        object_subset = np.isin(np.arange(sample_new.n_objects), object_subset)
        w = update_weights(sample_new)[object_subset]

        if self.sample_from_prior:
            p = w
        else:
            lh_per_component_new = component_likelihood_given_unchanged(
                self.model, sample_new, object_subset, i_cluster=i_cluster_new,
                temperature=self.temperature, prior_temperature=self.prior_temperature,
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
                self.model, sample_old, object_subset, i_cluster=i_cluster_old,
                temperature=self.temperature, prior_temperature=self.prior_temperature,
            )
            p_back = normalize(w * lh_per_component_old, axis=-1)

        source_old = sample_old.source.value[object_subset]
        log_q_back = np.log(p_back[source_old]).sum()

        return sample_new, log_q, log_q_back


class OperatorSchedule:

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
        return self.GIBBS_OPERATORS_BY_NAMES[name]


def verify_counts(sample: Sample, features: NDArray[bool]):
    cached_counts = deepcopy(sample.feature_counts)
    new_counts = recalculate_feature_counts(features, sample)
    assert set(cached_counts.keys()) == set(new_counts.keys())
    for k in cached_counts.keys():
        c = cached_counts[k].value
        c_new = new_counts[k].value
        assert np.allclose(c, c_new), (f'counts not matching in ´{k}´.', c.sum(axis=(1, 2)), c_new.sum(axis=(1, 2)))
