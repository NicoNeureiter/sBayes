#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import random as _random
from copy import deepcopy

import numpy as np
import scipy.stats as stats

from sbayes.sampling.mcmc_generative import MCMCGenerative
from sbayes.model import normalize_weights, ConfoundingEffectsPrior
from sbayes.util import get_neighbours, normalize, dirichlet_logpdf, get_max_size_list
from sbayes.preprocessing import sample_categorical


class IndexSet(set):

    def __init__(self, all_i=True):
        super().__init__()
        self.all = all_i

    def add(self, element):
        super(IndexSet, self).add(element)

    def clear(self):
        super(IndexSet, self).clear()
        self.all = False

    def __bool__(self):
        return self.all or (len(self) > 0)

    def __copy__(self):
        # other = deepcopy(super(IndexSet, self))
        other = IndexSet(all_i=self.all)
        for element in self:
            other.add(element)

        return other

    def __deepcopy__(self, memo):
        # other = deepcopy(super(IndexSet, self))
        other = IndexSet(all_i=self.all)
        for element in self:
            other.add(deepcopy(element))

        return other


class Sample(object):
    """
    Attributes:
        chain (int): index of the the chain for parallel sampling.
        clusters (np.array): Assignment of sites to clusters.
            shape: (n_clusters, n_sites)
        weights (np.array): Weights of the areal effect and each of the i confounding effects
            shape: (n_features, 1 + n_confounders)
        areal_effect (np.array): Probabilities of states in the clusters
            shape: (n_clusters, n_features, n_states)
        confounding_effects (dict): Probabilities of states for each of the i confounding effects,
            each confounding effect [i] with shape: (n_groups, n_features, n_states)
        source (np.array): Assignment of single observations (a feature in an object) to be
                           the result of the areal effect or one of the i confounding effects
            shape: (n_sites, n_features, 1 + n_confounders)
    """

    def __init__(self, clusters, weights, areal_effect, confounding_effects, source=None, chain=0):
        self.clusters = clusters
        self.weights = weights
        self.areal_effect = areal_effect
        self.confounding_effects = confounding_effects
        self.source = source
        self.chain = chain

        # The sample contains information about which of its parameters was changed in the last MCMC step
        self.what_changed = {}
        self.everything_changed()

    @classmethod
    def empty_sample(cls, conf):

        initial_sample = cls(clusters=None, weights=None, areal_effect=None,
                             confounding_effects={k: None for k in conf})
        initial_sample.everything_changed()
        return initial_sample

    def everything_changed(self):

        self.what_changed = {
            'lh': {'clusters': IndexSet(), 'weights': True, 'areal_effect': IndexSet(),
                   'confounding_effects': {k: IndexSet() for k in self.confounding_effects}},
            'prior': {'clusters': IndexSet(), 'weights': True, 'areal_effect': IndexSet(),
                      'confounding_effects': {k: IndexSet() for k in self.confounding_effects}}}

    def copy(self):
        clusters_copied = deepcopy(self.clusters)
        weights_copied = deepcopy(self.weights)
        what_changed_copied = deepcopy(self.what_changed)

        def maybe_copy(obj):
            if obj is not None:
                return obj.copy()

        areal_effect_copied = maybe_copy(self.areal_effect)
        confounding_effects_copied = dict()

        for k, v in self.confounding_effects.items():
            confounding_effects_copied[k] = maybe_copy(v)

        source_copied = maybe_copy(self.source)

        new_sample = Sample(chain=self.chain, clusters=clusters_copied, weights=weights_copied,
                            areal_effect=areal_effect_copied, confounding_effects=confounding_effects_copied,
                            source=source_copied)

        new_sample.what_changed = what_changed_copied

        return new_sample


class ZoneMCMC(MCMCGenerative):

    def __init__(self, p_grow_connected,
                 initial_sample, initial_size,
                 **kwargs):
        """
        Args:
            p_grow_connected (float): Probability at which grow operator only considers neighbours to add to the cluster
            initial_sample (Sample): The starting sample
            initial_size (int): The initial size of a cluster
            **kwargs: Other arguments that are passed on to MCMCGenerative
        """

        super(ZoneMCMC, self).__init__(**kwargs)

        # Data
        self.features = self.data.features['values']
        self.applicable_states = self.data.features['states']
        self.n_states_by_feature = np.sum(self.data.features['states'], axis=-1)
        self.n_features = self.features.shape[1]
        self.n_sites = self.features.shape[0]

        # Locations and network
        self.locations = self.data.network['locations']
        self.adj_mat = self.data.network['adj_mat']

        # Sampling
        self.p_grow_connected = p_grow_connected

        # Clustering
        self.n_clusters = self.model.n_clusters
        self.min_size = self.model.min_size
        self.max_size = self.model.max_size
        self.initial_size = initial_size

        # Confounders and sources
        self.confounders = self.model.confounders
        self.n_sources = 1 + len(self.confounders)
        self.source_index = self.get_source_index()
        self.n_groups = self.get_groups_per_confounder()

        # Initial Sample
        if initial_sample is None:
            self.initial_sample = Sample.empty_sample(self.confounders)
        else:
            self.initial_sample = initial_sample

        # Variance of the proposal distribution
        self.var_proposal_weight = 10
        self.var_proposal_areal_effect = 20
        self.var_proposal_confounding_effects = 10

        # todo remove after testing
        self.q_clusters_stats = {'q_grow': [],
                                 'q_back_grow': [],
                                 'q_shrink': [],
                                 'q_back_shrink': []
                                 }

    def get_groups_per_confounder(self):
        n_groups = dict()
        for k, v in self.confounders.items():
            n_groups[k] = len(v)
        return n_groups

    def get_source_index(self):
        source_index = {'areal_effect': 0, 'confounding_effects': dict()}

        for i, k, in enumerate(self.confounders):
            source_index['confounding_effects'][k] = i+1

        return source_index

    def gibbs_sample_sources(self, sample: Sample, as_gibbs=True,
                             site_subset=slice(None), **kwargs):
        """Resample the observations to mixture components (their source).

        Args:
            sample (Sample): The current sample with clusters and parameters
            as_gibbs (bool): Flag indicating whether this is a pure Gibbs operator (the
                default) or whether it is used as part of a non-Gibbs operator
            site_subset (slice or np.array[bool]): A subset of sites to be updated

        Returns:
            Sample: The modified sample
        """
        likelihood = self.posterior_per_chain[sample.chain].likelihood

        if self.sample_from_prior:
            lh_per_component = np.ones((self.n_sites, self.n_features, self.n_sources))
        else:
            # The likelihood of each component in each feature and languages
            lh_per_component = likelihood.update_component_likelihoods(sample=sample, caching=False)

        # Weights (in each feature and object) are the priors on the source assignments
        weights = likelihood.update_weights(sample=sample)

        # The source posterior is defined by the (normalized) product of weights and lh
        source_posterior = normalize(lh_per_component[site_subset] * weights[site_subset], axis=-1)

        # Sample the new source assignments
        sample.source[site_subset] = sample_categorical(p=source_posterior, binary_encoding=True)

        # # Some validity checks
        # lh = likelihood.update_component_likelihoods(sample=sample, caching=False)
        # assert np.min(np.sum(lh, axis=-1)) > 0
        # assert np.min(np.sum(lh * sample.source, axis=-1)) > 0
        # assert np.min(np.sum(sample.source * weights, axis=-1)) > 0

        if as_gibbs:
            # This is a Gibbs operator, which should always be accepted
            return sample, self.Q_GIBBS, self.Q_BACK_GIBBS
        else:
            # If part of another (non-Gibbs) operator, we need the correct hastings factor:
            is_source = np.where(sample.source[site_subset].ravel())
            log_q = np.sum(np.log(source_posterior.ravel()[is_source]))
            return sample, log_q, 0

    def gibbs_sample_weights(self, sample: Sample, **kwargs):
        sample_new = sample.copy()
        w = sample.weights
        w_new = sample_new.weights

        # Weights are always changed in this operator
        sample.what_changed['lh']['weights'] = True
        sample.what_changed['prior']['weights'] = True
        sample_new.what_changed['lh']['weights'] = True
        sample_new.what_changed['prior']['weights'] = True

        # The likelihood object contains relevant information on the areal and the confounding effect
        likelihood = self.posterior_per_chain[sample.chain].likelihood

        # If ´inheritance´ is off, we can exactly resample the weights, based on the
        # source counts of ´universal´ and ´contact´.
        if not self.inheritance:
            has_area = likelihood.get_zone_assignment(sample)
            counts = np.sum(sample.source[has_area], axis=0)

            for i_feat in range(self.n_features):
                sample_new.weights[i_feat, :] = np.random.dirichlet(1 + counts[i_feat])

            return sample_new, self.Q_GIBBS, self.Q_BACK_GIBBS

        # Otherwise we can compute the approximate posterior for a pair of weights.
        # We always keep one weight fixed (relative to the universal weight)
        fixed = _random.choice(['inheritance', 'contact'])

        if fixed == 'inheritance':
            # Fix w_inheritance, resample contribution of w_contact and w_universal.

            # Select counts of the relevant languages
            has_area = likelihood.get_zone_assignment(sample)
            counts = np.sum(sample.source[has_area], axis=0)
            c_univ = counts[..., 0]
            c_contact = counts[..., 1]

            # Create proposal distribution based on the counts
            distr = stats.beta(1 + c_contact, 1 + c_univ)

            # Sample new relative weights
            a_contact = distr.rvs()
            a_univ = 1 - a_contact

            # Adapt w_new and renormalize
            w_01 = w[..., 0] + w[..., 1]
            w_new[..., 0] = a_univ * w_01
            w_new[..., 1] = a_contact * w_01
            w_new = normalize(w_new, axis=-1)

            # Compute transition and back probability (for each feature)
            a_contact_old = w[..., 1] / w_01
            log_q = distr.logpdf(a_contact)
            log_q_back = distr.logpdf(a_contact_old)

        else:
            # Fix w_contact, resample contribution of w_inheritance and w_universal.

            # Select counts of the relevant languages
            has_family = likelihood.has_family
            counts = np.sum(sample.source[has_family], axis=0)
            c_univ = counts[..., 0]
            c_inherit = counts[..., 2]

            # Create proposal distribution based on the counts
            distr = stats.beta(1 + c_inherit, 1 + c_univ)

            # Sample new relative weights
            a_inherit = distr.rvs()
            a_univ = 1 - a_inherit

            # Adapt w_new and renormalize
            w_02 = w[..., 0] + w[..., 2]
            w_new[..., 0] = a_univ * w_02
            w_new[..., 2] = a_inherit * w_02
            w_new = normalize(w_new, axis=-1)

            # Compute transition and back probability (for each feature)
            a_inherit_old = w[..., 2] / w_02
            log_q = distr.logpdf(a_inherit)
            log_q_back = distr.logpdf(a_inherit_old)

        sample_new.weights = w_new

        w_normalized = likelihood.update_weights(sample)
        w_new_normalized = likelihood.update_weights(sample_new)

        # Compute old and new weight likelihoods (for each feature)
        log_lh_old_per_site = np.log(np.sum(sample.source * w_normalized, axis=-1))
        log_lh_old = np.sum(log_lh_old_per_site, axis=0)
        log_lh_new_per_site = np.log(np.sum(sample_new.source * w_new_normalized, axis=-1))
        log_lh_new = np.sum(log_lh_new_per_site, axis=0)

        # Add the prior to get the weight posterior (for each feature)
        log_prior_old = 0.   # TODO add hyper prior on weights, when implemented
        log_prior_new = 0.  # TODO add hyper prior on weights, when implemented
        log_p_old = log_lh_old + log_prior_old
        log_p_new = log_lh_new + log_prior_new

        # Compute hastings ratio for each feature and accept/reject independently
        p_accept = np.exp(log_p_new - log_p_old + log_q_back - log_q)
        accept = np.random.random(p_accept.shape) < p_accept
        sample_new.weights = np.where(accept[:, np.newaxis], w_new, w)

        # We already accepted/rejected for each feature independently
        # The result should always be accepted in the MCMC
        return sample_new, self.Q_GIBBS, self.Q_BACK_GIBBS

    def gibbs_sample_areal_effect(self, sample: Sample, i_cluster=None, **kwargs):
        if i_cluster is None:
            i_cluster = np.random.randint(0, self.n_clusters)

        i_source = self.source_index['areal_effect']

        if self.sample_from_prior:
            # To sample from prior we emulate an empty dataset
            n_states = self.applicable_states.shape[-1]
            features = np.zeros((1, self.n_features, n_states))
        else:
            # Only consider observations that are attributed to the areal effect distribution
            from_cluster = (sample.source[:, :, i_source] & sample.clusters[i_cluster, :, np.newaxis])
            features = from_cluster[..., np.newaxis] * self.features

        # Resample areal_effect according to these observations
        for i_feat in range(self.n_features):
            s_idxs = self.applicable_states[i_feat]
            feature_counts = np.nansum(features[:, i_feat, s_idxs], axis=0)
            sample.areal_effect[i_cluster, i_feat, s_idxs] = np.random.dirichlet(alpha=1 + feature_counts)

            # The step changed the areal effect (which has an influence on how the lh and the prior look like)
            sample.what_changed['lh']['areal_effect'].add((i_cluster, i_feat))
            sample.what_changed['prior']['areal_effect'].add((i_cluster, i_feat))

        return sample, self.Q_GIBBS, self.Q_BACK_GIBBS

    def gibbs_sample_confounding_effects(self, sample: Sample, i_group=None,
                                         fraction_of_features=0.4, **kwargs):

        conf = kwargs['additional_parameters']['confounder']
        if i_group is None:
            i_group = np.random.randint(0, self.n_groups[conf])

        source_i = self.source_index['confounding_effects'][conf]

        feature_subset = np.random.random(self.n_features) < fraction_of_features
        n_features_subset = np.sum(feature_subset)
        n_states = self.applicable_states.shape[-1]

        if self.sample_from_prior:
            # To sample from prior we emulate an empty dataset
            features = np.zeros((1, n_features_subset, n_states))
        else:
            # Select subset of features
            features = self.features[:, feature_subset, :]

            # Only consider observations that are attributed to the relevant confounding effect and group
            from_group = (sample.source[:, feature_subset, source_i] &
                          self.data.confounders[conf]['values'][i_group, :, np.newaxis])
            features = from_group[..., np.newaxis] * features

        # Get the prior pseudo-counts
        prior = self.posterior_per_chain[sample.chain].prior
        prior_counts = prior.prior_confounding_effects[conf].concentration[i_group]

        # Resample confounding effect according to these observations
        for i_feat_subset, i_feat in enumerate(np.argwhere(feature_subset)):
            i_feat = i_feat[0]
            s_idxs = self.applicable_states[i_feat]
            feature_counts = np.nansum(features[:, i_feat_subset, s_idxs], axis=0)
            sample.confounding_effects[conf][i_group, i_feat, s_idxs] = \
                np.random.dirichlet(prior_counts[i_feat] + feature_counts)

            sample.what_changed['lh']['confounding_effects'][conf].add((i_group, i_feat))
            sample.what_changed['prior']['confounding_effects'][conf].add((i_group, i_feat))

        return sample, self.Q_GIBBS, self.Q_BACK_GIBBS

    def alter_weights(self, sample: Sample, **kwargs):
        """Modifies one weight of one feature in the current sample
        Args:
            sample(Sample): The current sample with clusters and parameters
        Returns:
            Sample: The modified sample
        """
        sample_new = sample.copy()

        # Randomly choose one of the features
        f_id = np.random.choice(range(self.n_features))

        # Randomly choose two weights that will be changed, leave the others untouched

        weights_to_alter = _random.sample(range(self.n_sources), 2)

        # Get the current weights
        weights_current = sample.weights[f_id, weights_to_alter]

        # Transform the weights such that they sum to 1
        weights_current_t = weights_current / weights_current.sum()

        # Propose new sample
        weights_new_t, log_q, log_q_back = self.dirichlet_proposal(weights_current_t, self.var_proposal_weight)

        # Transform back
        weights_new = weights_new_t * weights_current.sum()

        # Update
        sample_new.weights[f_id, weights_to_alter] = weights_new

        # The step changed the weights (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['weights'] = True
        sample_new.what_changed['prior']['weights'] = True
        sample.what_changed['lh']['weights'] = True
        sample.what_changed['prior']['weights'] = True

        return sample_new, log_q, log_q_back

    def alter_areal_effect(self, sample, **kwargs):
        """Modifies the areal effect of one state, feature and cluster in the current sample
        Args:
            sample(Sample): The current sample with clusters and parameters
        Returns:
            Sample: The modified sample
                """
        sample_new = sample.copy()

        # Randomly choose one of the clusters, one of the features and one of the states
        z_id = np.random.choice(range(self.n_clusters))
        f_id = np.random.choice(range(self.n_features))

        # Different features have different applicable states
        f_states = np.nonzero(self.applicable_states[f_id])[0]

        # Randomly choose two applicable states for which the probabilities will be changed, leave the others untouched
        states_to_alter = _random.sample(list(f_states), 2)

        # Get the current probabilities
        p_current = sample.areal_effect[z_id, f_id, states_to_alter]

        # Transform the probabilities such that they sum to 1
        p_current_t = p_current / p_current.sum()

        # Sample new p from dirichlet distribution with given precision
        p_new_t, log_q, log_q_back = self.dirichlet_proposal(p_current_t, step_precision=self.var_proposal_areal_effect)

        # Transform back
        p_new = p_new_t * p_current.sum()

        # Update sample
        sample_new.areal_effect[z_id, f_id, states_to_alter] = p_new

        # The step changed p_zones (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['areal_effect'].add((z_id, f_id))
        sample_new.what_changed['prior']['areal_effect'].add((z_id, f_id))
        sample.what_changed['lh']['areal_effect'].add((z_id, f_id))
        sample.what_changed['prior']['areal_effect'].add((z_id, f_id))

        return sample_new, log_q, log_q_back

    @staticmethod
    def dirichlet_proposal(w, step_precision):
        """ A proposal distribution for normalized weight and probability vectors (summing to 1).
        Args:
            w (np.array): The weight vector, which is being resampled.
                Shape: (n_states, 1 + n_confounders)
            step_precision (float): The precision parameter controlling how narrow/wide the proposal
                distribution is. Low precision -> wide, high precision -> narrow.

        Returns:
            np.array: The newly proposed weights w_new (same shape as w).
            float: The transition probability q.
            float: The back probability q_back
        """
        alpha = 1 + step_precision * w
        w_new = np.random.dirichlet(alpha)
        log_q = dirichlet_logpdf(w_new, alpha)

        alpha_back = 1 + step_precision * w_new
        log_q_back = dirichlet_logpdf(w, alpha_back)

        if not np.all(np.isfinite(w_new)):
            logging.warning(f'Dirichlet step resulted in NaN or Inf:')
            logging.warning(f'\tOld sample: {w}')
            logging.warning(f'\tstep_precision: {step_precision}')
            logging.warning(f'\tNew sample: {w_new}')
            # return w, 0., -np.inf

        return w_new, log_q, log_q_back

    def alter_confounding_effects(self, sample, **kwargs):
        """This function modifies confounding effect [i] of one state and one feature in the current sample
            Args:
                 sample(Sample): The current sample with clusters and parameters
            Returns:
                 Sample: The modified sample
        """

        sample_new = sample.copy()
        conf = kwargs['additional_parameters']['confounder']

        # Randomly choose one of the families and one of the features
        group_id = np.random.randint(0, self.n_groups[conf])
        f_id = np.random.choice(range(self.n_features))

        # Different features have different applicable states
        f_states = np.nonzero(self.applicable_states[f_id])[0]

        # Randomly choose two applicable states for which the probabilities will be changed, leave the others untouched
        states_to_alter = _random.sample(list(f_states), 2)

        # Get the current probabilities
        p_current = sample.confounding_effects[conf][group_id, f_id, states_to_alter]

        # Transform the probabilities such that they sum to 1
        p_current_t = p_current / p_current.sum()

        # Sample new p from dirichlet distribution with given precision
        p_new_t, log_q, log_q_back = self.dirichlet_proposal(
            p_current_t, step_precision=self.var_proposal_confounding_effects)

        # Transform back
        p_new = p_new_t * p_current.sum()

        # Update sample
        sample_new.confounding_effects[conf][group_id, f_id, states_to_alter] = p_new

        # The step changed confounding effect [i] (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['confounding_effects'][conf].add((group_id, f_id))
        sample_new.what_changed['prior']['confounding_effects'][conf].add((group_id, f_id))
        sample.what_changed['lh']['confounding_effects'][conf].add((group_id, f_id))
        sample.what_changed['prior']['confounding_effects'][conf].add((group_id, f_id))

        return sample_new, log_q, log_q_back

    def gibbsish_sample_clusters_local(self, sample, resample_source=True):
        return self.gibbsish_sample_clusters(sample,
                                          resample_source=resample_source,
                                          site_subset=get_neighbours)

    # todo: fix
    def gibbsish_sample_clusters(self, sample, c=0, resample_source=True):

        sample_new = sample.copy()
        likelihood = self.posterior_per_chain[sample.chain].likelihood
        occupied = np.any(sample.zones, axis=0)

        # Randomly choose one of the zones to modify
        z_id = np.random.choice(range(sample.zones.shape[0]))
        zone = sample.zones[z_id, :]
        available = ~occupied | zone

        # if site_subset is not None:
        #     new_candidates &= (site_subset(zone, occupied, self.adj_mat) | zone)
        n_available = np.count_nonzero(available)
        if n_available > 100:
            available[available] &= np.random.random(n_available) < (100/n_available)
            n_available = np.count_nonzero(available)

        if n_available == 0:
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        if self.model.sample_source and resample_source:
            likelihood = self.posterior_per_chain[sample.chain].likelihood
            lh_per_component = likelihood.update_component_likelihoods(sample=sample, caching=False)
            weights = likelihood.update_weights(sample=sample)
            source_posterior = normalize(lh_per_component[available] * weights[available], axis=-1)
            is_source = np.where(sample.source[available].ravel())
            log_q_back_s = np.sum(np.log(source_posterior.ravel()[is_source]))

        # Compute lh per language with and without zone
        global_lh = likelihood.get_global_lh(sample)[available, :]
        zone_lh = np.einsum('ijk,jk->ij', self.features[available], sample.p_zones[z_id])
        if self.inheritance:
            family_lh = likelihood.get_family_lh(sample)[available, :]
            all_lh = np.array([global_lh, zone_lh, family_lh]).transpose((1, 2, 0))
        else:
            all_lh = np.array([global_lh, zone_lh]).transpose((1, 2, 0))
        all_lh[likelihood.na_features[available]] = 1.

        if self.inheritance:
            has_components = np.ones((n_available , 3), dtype=bool)
            has_components[:, -1] = likelihood.has_family[available]
        else:
            has_components = np.ones((n_available, 2), dtype=bool)

        universe_for_orphans = self.model.likelihood.missing_family_as_universal
        weights_with_z = normalize_weights(sample.weights, has_components,
                                           missing_family_as_universal=universe_for_orphans)
        has_components[:, 1] = False
        weights_without_z = normalize_weights(sample.weights, has_components,
                                              missing_family_as_universal=universe_for_orphans)

        feature_lh_with_z = np.sum(all_lh * weights_with_z, axis=-1)
        feature_lh_without_z = np.sum(all_lh * weights_without_z, axis=-1)

        marginal_lh_with_z = np.exp(np.sum(np.log(feature_lh_with_z), axis=-1))
        marginal_lh_without_z = np.exp(np.sum(np.log(feature_lh_without_z), axis=-1))

        posterior_zone = marginal_lh_with_z / (marginal_lh_with_z + marginal_lh_without_z)
        new_zone = (np.random.random(n_available) < posterior_zone)

        sample_new.zones[z_id, available] = new_zone

        # Reject when an area outside the valid size range is proposed
        new_area_size = np.sum(sample_new.zones[z_id])
        max_size = self.max_size[c] if self.IS_WARMUP else self.max_size
        if not (self.min_size <= new_area_size <= max_size):
            return sample, self.Q_REJECT, self.Q_BACK_REJECT

        q_per_site = posterior_zone * new_zone + (1 - posterior_zone) * (1 - new_zone)
        log_q = np.sum(np.log(q_per_site))
        q_back_per_site = posterior_zone * zone[available] + (1 - posterior_zone) * (1 - zone[available])
        if np.any(q_back_per_site == 0):
            return sample, self.Q_REJECT, self.Q_BACK_REJECT
        log_q_back = np.sum(np.log(q_back_per_site))

        sample_new.what_changed['lh']['zones'].add(z_id)
        sample.what_changed['lh']['zones'].add(z_id)
        sample_new.what_changed['prior']['zones'].add(z_id)
        sample.what_changed['prior']['zones'].add(z_id)

        if self.model.sample_source and resample_source:
            sample_new, log_q_s, _ = self.gibbs_sample_sources(sample_new, as_gibbs=False,
                                                               site_subset=available)
            log_q += log_q_s
            log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back

    def swap_cluster(self, sample, c=0, resample_source=True, **kwargs):
        """ This functions swaps sites in one of the clusters of the current sample
        (i.e. in of the clusters a site is removed and another one added)
        Args:
            sample(Sample): The current sample with clusters and parameters
            c(int): The current warmup chain
            resample_source(bool): Resample the source?
        Returns:
            Sample: The modified sample.
         """
        sample_new = sample.copy()
        clusters_current = sample.clusters
        occupied = np.any(clusters_current, axis=0)

        if self.model.sample_source and resample_source:
            likelihood = self.posterior_per_chain[sample.chain].likelihood
            lh_per_component = likelihood.update_component_likelihoods(sample=sample, caching=False)
            weights = likelihood.update_weights(sample=sample)
            source_posterior = normalize(lh_per_component * weights, axis=-1)
            # q_per_observation = np.sum(sample.source * source_posterior, axis=2)
            # log_q_back_s = np.sum(np.log(q_per_observation))
            is_source = np.where(sample.source.ravel())
            log_q_back_s = np.sum(np.log(source_posterior.ravel()[is_source]))

        # Randomly choose one of the clusters to modify
        z_id = np.random.choice(range(clusters_current.shape[0]))
        cluster_current = clusters_current[z_id, :]

        neighbours = get_neighbours(cluster_current, occupied, self.adj_mat)
        connected_step = (_random.random() < self.p_grow_connected)
        if connected_step:
            # All neighboring sites that are not yet occupied by other clusters are candidates
            candidates = neighbours
        else:
            # All free sites are candidates
            candidates = ~occupied

        # When stuck (all neighbors occupied) return current sample and reject the step (q_back = 0)
        if not np.any(candidates):
            return sample, 0, -np.inf

        # Add a site to the zone
        site_new = _random.choice(candidates.nonzero()[0])
        sample_new.clusters[z_id, site_new] = 1

        # Remove a site from the zone
        removal_candidates = self.get_removal_candidates(cluster_current)
        site_removed = _random.choice(removal_candidates)
        sample_new.clusters[z_id, site_removed] = 0

        # # Compute transition probabilities
        back_neighbours = get_neighbours(cluster_current, occupied, self.adj_mat)
        # q = 1. / np.count_nonzero(candidates)
        # q_back = 1. / np.count_nonzero(back_neighbours)

        # Transition probability growing to the new cluster
        q_non_connected = 1 / np.count_nonzero(~occupied)

        q = (1 - self.p_grow_connected) * q_non_connected
        if neighbours[site_new]:
            q_connected = 1 / np.count_nonzero(neighbours)
            q += self.p_grow_connected * q_connected

        # Transition probability of growing back to the original zone
        q_back_non_connected = 1 / np.count_nonzero(~occupied)
        q_back = (1 - self.p_grow_connected) * q_back_non_connected

        # If z is a neighbour of the new zone, the back step could also be a connected grow step
        if back_neighbours[site_removed]:
            q_back_connected = 1 / np.count_nonzero(back_neighbours)
            q_back += self.p_grow_connected * q_back_connected

        # The step changed the zone (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['clusters'].add(z_id)
        sample.what_changed['lh']['clusters'].add(z_id)
        sample_new.what_changed['prior']['clusters'].add(z_id)
        sample.what_changed['prior']['clusters'].add(z_id)

        assert 0 < q <= 1
        assert 0 < q_back <= 1, q_back

        log_q = np.log(q)
        log_q_back = np.log(q_back)

        if self.model.sample_source and resample_source:
            sample_new, log_q_s, _ = self.gibbs_sample_sources(sample_new, as_gibbs=False)
            log_q += log_q_s
            log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back

    def grow_cluster(self, sample, c=0, resample_source=True, **kwargs):
        """ This functions grows one of the clusters in the current sample (i.e. it adds a new site to one cluster)
        Args:
            sample(Sample): The current sample with clusters and parameters
            c(int): The current warmup chain
            resample_source(bool): Resample the source?
        Returns:
            (Sample): The modified sample.
        """
        sample_new = sample.copy()
        clusters_current = sample.clusters
        occupied = np.any(clusters_current, axis=0)

        if self.model.sample_source and resample_source:
            likelihood = self.posterior_per_chain[sample.chain].likelihood
            lh_per_component = likelihood.update_component_likelihoods(sample=sample, caching=False)
            weights = likelihood.update_weights(sample=sample)
            source_posterior = normalize(lh_per_component * weights, axis=-1)
            # q_per_observation = np.sum(sample.source * source_posterior, axis=2)
            # log_q_back_s = np.sum(np.log(q_per_observation))
            is_source = np.where(sample.source.ravel())
            log_q_back_s = np.sum(np.log(source_posterior.ravel()[is_source]))

        # Randomly choose one of the clusters to modify
        z_id = np.random.choice(range(clusters_current.shape[0]))
        cluster_current = clusters_current[z_id, :]

        # Check if cluster is small enough to grow
        current_size = np.count_nonzero(cluster_current)

        if current_size >= self.max_size:
            # Cluster too big to grow: don't modify the sample and reject the step (q_back = 0)
            return sample, 0, -np.inf

        neighbours = get_neighbours(cluster_current, occupied, self.adj_mat)
        connected_step = (_random.random() < self.p_grow_connected)
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
        site_new = _random.choice(candidates.nonzero()[0])
        sample_new.clusters[z_id, site_new] = 1

        # Transition probability when growing
        q_non_connected = 1 / np.count_nonzero(~occupied)
        q = (1 - self.p_grow_connected) * q_non_connected

        if neighbours[site_new]:
            q_connected = 1 / np.count_nonzero(neighbours)
            q += self.p_grow_connected * q_connected

        # Back-probability (shrinking)
        q_back = 1 / (current_size + 1)

        # The step changed the zone (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['clusters'].add(z_id)
        sample.what_changed['lh']['clusters'].add(z_id)
        sample_new.what_changed['prior']['clusters'].add(z_id)
        sample.what_changed['prior']['clusters'].add(z_id)

        assert 0 < q <= 1
        assert 0 < q_back <= 1

        log_q = np.log(q)
        log_q_back = np.log(q_back)

        if self.model.sample_source and resample_source:
            sample_new, log_q_s, _ = self.gibbs_sample_sources(sample_new, as_gibbs=False)
            log_q += log_q_s
            log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back

    def shrink_cluster(self, sample, c=0, resample_source=True, **kwargs):
        """ This functions shrinks one of the clusters in the current sample (i.e. it removes one site from one cluster)
        Args:
            sample(Sample): The current sample with clusters and parameters
            c(int): The current warmup chain
            resample_source(bool): Resample the source?
        Returns:
            (Sample): The modified sample.
        """
        sample_new = sample.copy()
        clusters_current = sample.clusters

        if self.model.sample_source and resample_source:
            likelihood = self.posterior_per_chain[sample.chain].likelihood
            lh_per_component = likelihood.update_component_likelihoods(sample=sample, caching=False)
            weights = likelihood.update_weights(sample=sample)
            source_posterior = normalize(lh_per_component * weights, axis=-1)
            # q_per_observation = np.sum(sample.source * source_posterior, axis=2)
            # log_q_back_s = np.sum(np.log(q_per_observation))
            is_source = np.where(sample.source.ravel())
            log_q_back_s = np.sum(np.log(source_posterior.ravel()[is_source]))

        # Randomly choose one of the clusters to modify
        z_id = np.random.choice(range(clusters_current.shape[0]))
        cluster_current = clusters_current[z_id, :]

        # Check if cluster is big enough to shrink
        current_size = np.count_nonzero(cluster_current)
        if current_size <= self.min_size:
            # Cluster is too small to shrink: don't modify the sample and reject the step (q_back = 0)
            return sample, 0, -np.inf

        # Cluster is big enough: shrink
        removal_candidates = self.get_removal_candidates(cluster_current)
        site_removed = _random.choice(removal_candidates)
        sample_new.clusters[z_id, site_removed] = 0

        # Transition probability when shrinking.
        q = 1 / len(removal_candidates)
        # Back-probability (growing)
        cluster_new = sample_new.clusters[z_id]
        occupied_new = np.any(sample_new.clusters, axis=0)
        back_neighbours = get_neighbours(cluster_new, occupied_new, self.adj_mat)

        # The back step could always be a non-connected grow step
        q_back_non_connected = 1 / np.count_nonzero(~occupied_new)
        q_back = (1 - self.p_grow_connected) * q_back_non_connected

        # If z is a neighbour of the new zone, the back step could also be a connected grow step
        if back_neighbours[site_removed]:
            q_back_connected = 1 / np.count_nonzero(back_neighbours)
            q_back += self.p_grow_connected * q_back_connected

        # Back-probability (shrinking)
        q_back = 1 / (current_size + 1)

        # self.q_areas_stats['q_shrink'].append(q)
        # self.q_areas_stats['q_back_shrink'].append(q_back)

        # The step changed the zone (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['clusters'].add(z_id)
        sample.what_changed['lh']['clusters'].add(z_id)
        sample_new.what_changed['prior']['clusters'].add(z_id)
        sample.what_changed['prior']['clusters'].add(z_id)

        assert 0 < q <= 1
        assert 0 < q_back <= 1

        log_q = np.log(q)
        log_q_back = np.log(q_back)

        if self.model.sample_source and resample_source:
            sample_new, log_q_s, _ = self.gibbs_sample_sources(sample_new, as_gibbs=False)
            log_q += log_q_s
            log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back

    def generate_initial_clusters(self):
        """For each chain (c) generate initial clusters by
        A) growing through random grow-steps up to self.min_size,
        B) using the last sample of a previous run of the MCMC

        Returns:
            np.array: The generated initial clusters.
                shape(n_clusters, n_sites)
        """

        # If there are no clusters in the model, return empty matrix
        if self.n_clusters == 0:
            return np.zeros((self.n_clusters, self.n_sites), bool)

        occupied = np.zeros(self.n_sites, bool)
        initial_clusters = np.zeros((self.n_clusters, self.n_sites), bool)
        n_generated = 0

        # B: When clusters from a previous run exist use them as the initial sample
        if self.initial_sample.clusters is not None:
            for i in range(len(self.initial_sample.clusters)):
                initial_clusters[i, :] = self.initial_sample.clusters[i]
                occupied += self.initial_sample.clusters[i]
                n_generated += 1

        not_initialized = range(n_generated, self.n_clusters)

        # A: Grow the remaining clusters
        # With many clusters new ones can get stuck due to unfavourable seeds.
        # We perform several attempts to initialize the clusters.
        attempts = 0
        max_attempts = 1000

        while True:
            for i in not_initialized:
                try:
                    initial_size = self.initial_size
                    cl, in_cl = self.grow_cluster_of_size_k(k=initial_size, already_in_cluster=occupied)

                except self.ClusterError:
                    # Rerun: Error might be due to an unfavourable seed
                    if attempts < max_attempts:
                        attempts += 1
                        not_initialized = range(n_generated, self.n_clusters)
                        break
                    # Seems there is not enough sites to grow n_clusters of size k
                    else:
                        raise ValueError(f"Failed to add additional cluster. Try fewer clusters "
                                         f"or set initial_sample to None.")

                n_generated += 1
                initial_clusters[i, :] = cl
                occupied = in_cl

            if n_generated == self.n_clusters:
                return initial_clusters

    def grow_cluster_of_size_k(self, k, already_in_cluster=None):
        """ This function grows a cluster of size k excluding any of the sites in <already_in_cluster>.
        Args:
            k (int): The size of the cluster, i.e. the number of sites in the cluster
            already_in_cluster (np.array): All sites already assigned to a cluster (boolean)

        Returns:
            np.array: The newly grown cluster (boolean).
            np.array: all sites already assigned to a cluster (boolean).

        """
        if already_in_cluster is None:
            already_in_cluster = np.zeros(self.n_sites, bool)

        # Initialize the cluster
        cluster = np.zeros(self.n_sites, bool)

        # Find all sites that are occupied by a cluster and those that are still free
        sites_occupied = np.nonzero(already_in_cluster)[0]
        sites_free = set(range(self.n_sites)) - set(sites_occupied)

        # Take a random free site and use it as seed for the new cluster
        try:
            i = _random.sample(sites_free, 1)[0]
            cluster[i] = already_in_cluster[i] = 1
        except ValueError:
            raise self.ClusterError

        # Grow the cluster if possible
        for _ in range(k - 1):
            neighbours = get_neighbours(cluster, already_in_cluster, self.adj_mat)
            if not np.any(neighbours):
                raise self.ClusterError

            # Add a neighbour to the cluster
            site_new = _random.choice(neighbours.nonzero()[0])
            cluster[site_new] = already_in_cluster[site_new] = 1

        return cluster, already_in_cluster

    def generate_initial_weights(self):
        """This function generates initial weights for the Bayesian additive mixture model, either by
        A) proposing them (randomly) from scratch
        B) using the last sample of a previous run of the MCMC
        Weights are in log-space and not normalized.

        Returns:
            np.array: weights for areal_effect and each of the i confounding_effects
            """

        # B: Use weights from a previous run
        if self.initial_sample.weights is not None:
            initial_weights = self.initial_sample.weights

        # A: Initialize new weights
        else:
            initial_weights = np.full((self.n_features, self.n_sources), 1.)

        return normalize(initial_weights)

    def set_p_zones_to_mle(self, updated_zone):
        """This function sets the p_zones to the MLE of the current zone
        Probabilities are in log-space and not normalized.
        Args:
            updated_zone (np.array): The currently updated zone
            (n_sites)
        Returns:
            np.array: probabilities for categories in each zones
                shape (n_zones, n_features, max(n_categories))
        """

        idx = updated_zone.nonzero()[0]
        features_zone = self.features[idx, :, :]
        l_per_cat = np.sum(features_zone, axis=0)
        p_zones = normalize(l_per_cat)

        return p_zones

    def generate_initial_areal_effect(self, initial_clusters):
        """This function generates initial state probabilities for each of the clusters, either by
        A) proposing them (randomly) from scratch
        B) using the last sample of a previous run of the MCMC
        Probabilities are in log-space and not normalized.
        Args:
            initial_clusters: The assignment of sites to clusters
            (n_clusters, n_sites)
        Returns:
            np.array: probabilities for categories in each cluster
                shape (n_clusters, n_features, max(n_states))
        """
        # We place the areal_effect of all features in one array, even though not all have the same number of states
        initial_areal_effect = np.zeros((self.n_clusters, self.n_features, self.features.shape[2]))
        n_generated = 0

        # B: Use areal_effect from a previous run
        if self.initial_sample.areal_effect is not None:

            for i in range(len(self.initial_sample.areal_effect)):
                initial_areal_effect[i, :] = self.initial_sample.areal_effect[i]
                n_generated += 1

        not_initialized = range(n_generated, self.n_clusters)

        # A: Initialize a new areal_effect using a value close to the MLE of the current cluster
        for i in not_initialized:
            idx = initial_clusters[i].nonzero()[0]
            features_cluster = self.features[idx, :, :]

            sites_per_state = np.nansum(features_cluster, axis=0)

            # Some clusters have nan for all states, resulting in a non-defined MLE
            # other clusters have only a single state, resulting in an MLE including 1.
            # to avoid both, we add 1 to all applicable states of each feature,
            # which gives a well-defined initial areal_effect slightly nudged away from the MLE

            sites_per_state[np.isnan(sites_per_state)] = 0
            sites_per_state[self.applicable_states] += 1

            site_sums = np.sum(sites_per_state, axis=1)
            areal_effect = sites_per_state / site_sums[:, np.newaxis]

            initial_areal_effect[i, :, :] = areal_effect

        return initial_areal_effect

    def generate_initial_confounding_effect(self, conf):
        """This function generates initial state probabilities for each group in confounding effect [i], either by
        A) using the MLE
        B) using the last sample of a previous run of the MCMC
        Probabilities are in log-space and not normalized.
        Args:
            conf(dict): The confounding effect [i]
        Returns:
            np.array: probabilities for states in each group of confounding effect [i]
                shape (n_groups, n_features, max(n_states))
        """

        n_groups = self.n_groups[conf]
        groups = self.data.confounders[conf]['values']

        initial_confounding_effect = np.zeros((n_groups, self.n_features, self.features.shape[2]))

        # B: Use confounding_effect from a previous run
        if self.initial_sample.confounding_effects[conf] is not None:
            for i in range(len(self.initial_sample.confounding_effects[conf])):
                initial_confounding_effect[i, :] = self.initial_sample.confounding_effects[conf][i]

        # A: Initialize new confounding_effect using the MLE
        else:
            for g in range(n_groups):

                idx = groups[g].nonzero()[0]
                features_group = self.features[idx, :, :]

                sites_per_state = np.nansum(features_group, axis=0)

                # Compute the MLE for each state and each group in the confounding effect
                # Some groups have only NAs for some features, resulting in a non-defined MLE
                # other groups have only a single state, resulting in an MLE including 1.
                # To avoid both, we add 1 to all applicable states of each feature,
                # which gives a well-defined initial confounding effect, slightly nudged away from the MLE

                sites_per_state[np.isnan(sites_per_state)] = 0
                sites_per_state[self.applicable_states] += 1

                state_sums = np.sum(sites_per_state, axis=1)
                p_group = sites_per_state / state_sums[:, np.newaxis]
                initial_confounding_effect[g, :, :] = p_group

        return initial_confounding_effect

    def generate_initial_sample(self, c=0):
        """Generate initial Sample object (clusters, weights, areal_effect, confounding_effects)
        Kwargs:
            c (int): index of the MCMC chain
        Returns:
            Sample: The generated initial Sample
        """
        # Clusters
        initial_clusters = self.generate_initial_clusters()

        # Weights
        initial_weights = self.generate_initial_weights()

        # Areal effect
        initial_areal_effect = self.generate_initial_areal_effect(initial_clusters)

        # Confounding effects
        initial_confounding_effects = dict()
        for k, v in self.confounders.items():
            initial_confounding_effects[k] = self.generate_initial_confounding_effect(k)

        sample = Sample(clusters=initial_clusters, weights=initial_weights,
                        areal_effect=initial_areal_effect,
                        confounding_effects=initial_confounding_effects,
                        chain=c)

        # Generate the initial source using a Gibbs sampling step
        if self.model.sample_source:
            sample.source = np.empty((self.n_sites, self.n_features, self.n_sources),
                                     dtype=bool)
            sample, _, _ = self.gibbs_sample_sources(sample)

        return sample

    @staticmethod
    def get_removal_candidates(zone):
        """Finds sites which can be removed from the given zone.

        Args:
            zone (np.array): The zone for which removal candidates are found.
                shape(n_sites)
        Returns:
            (list): Index-list of removal candidates.
        """
        return zone.nonzero()[0]

    class ClusterError(Exception):
        pass

    def get_operators(self, operators_raw):
        """Get all relevant operator functions for proposing MCMC update steps and their probabilities
        Args:
            operators_raw(dict): dictionary with names of all operators (keys) and their weights (values)
        Returns:
            dict: for each operator: functions (callable), weights (float) and if applicable additional parameters
        """
        a_test = []

        op_weights = {
            'shrink_cluster': {'weight': operators_raw['clusters'] * 0.4,
                               'function': getattr(self, "shrink_cluster"),
                               'additional_parameters': None},
            'grow_cluster': {'weight': operators_raw['clusters'] * 0.4,
                             'function': getattr(self, "grow_cluster"),
                             'additional_parameters': None},
            'swap_cluster': {'weight': operators_raw['clusters'] * 0.2,
                             'function': getattr(self, "swap_cluster"),
                             'additional_parameters': None},
            'gibbsish_sample_clusters': {'weight': operators_raw['clusters'] * 0.0,
                                         'function': getattr(self, "gibbsish_sample_clusters"),
                                         'additional_parameters': None}
        }

        if self.model.sample_source:
            op_weights.update({
                'gibbs_sample_sources': {'weight': operators_raw['source'],
                                         'function': getattr(self, "gibbs_sample_sources"),
                                         'additional_parameters': None},
                'gibbs_sample_weights': {'weight': operators_raw['weights'],
                                         'function': getattr(self, 'gibbs_sample_weights'),
                                         'additional_parameters': None},
                'gibbs_sample_areal_effect': {'weight': operators_raw['areal_effect'],
                                              'function': getattr(self, "gibbs_sample_areal_effect"),
                                              'additional_parameters': None}
            })

            r = float(1 / len(self.model.confounders))
            for k in self.model.confounders:
                op_name = "gibbs_sample_confounding_effects_" + str(k)
                op_weights.update({
                    op_name: {'weight': operators_raw['confounding_effects'] * r,
                              'function': getattr(self, "gibbs_sample_confounding_effects"),
                              'additional_parameters': {'confounder': k}}
                })

        else:
            op_weights.update({
                'alter_weights': {'weight': operators_raw['weights'],
                                  'function': getattr(self, "alter_weights"),
                                  'additional_parameters': None},
                'alter_areal_effect': {'weight': operators_raw['areal_effect'],
                                       'function': getattr(self, "alter_areal_effect"),
                                       'additional_parameters': None}
            })

            r = float(1 / len(self.model.confounders))
            for k in self.model.confounders:
                op_name = "alter_confounding_effects_" + str(k)
                op_weights.update({
                    op_name: {'weight': operators_raw['confounding_effects'] * r,
                              'function': getattr(self, "alter_confounding_effects"),
                              'confounder': k, 'additional_parameters': {'confounder': k}}
                })

        return op_weights

    def log_sample_statistics(self, sample, c, sample_id):
        super(ZoneMCMC, self).log_sample_statistics(sample, c, sample_id)


class ZoneMCMCWarmup(ZoneMCMC):

    IS_WARMUP = True

    def __init__(self, **kwargs):
        super(ZoneMCMCWarmup, self).__init__(**kwargs)

        # In warmup chains can have a different max_size for clusters
        self.max_size = get_max_size_list(
            start=(self.initial_size + self.max_size)/4,
            end=self.max_size,
            n_total=self.n_chains,
            k_groups=4
        )

        # Some chains only have connected steps, whereas others also have random steps
        self.p_grow_connected = _random.choices(
            population=[0.95, self.p_grow_connected],
            k=self.n_chains
        )

    def gibbs_sample_sources(self, sample, c=0, as_gibbs=True, site_subset=slice(None), **kwargs):
        return super(ZoneMCMCWarmup, self).gibbs_sample_sources(
            sample, as_gibbs=True, site_subset=site_subset)

    def gibbs_sample_weights(self, sample, c=0, **kwargs):
        return super(ZoneMCMCWarmup, self).gibbs_sample_weights(
            sample, additional_parameters=kwargs['additional_parameters'])

    def gibbs_sample_areal_effect(self, sample, i_zone=None, c=0, **kwargs):
        return super(ZoneMCMCWarmup, self).gibbs_sample_areal_effect(
            sample, i_zone=i_zone, additional_parameters=kwargs['additional_parameters'])

    def gibbs_sample_confounding_effects(self, sample, i_family=None, fraction_of_features=0.4, c=0, **kwargs):
        return super(ZoneMCMCWarmup, self).gibbs_sample_confounding_effects(
            sample, i_family=i_family, fraction_of_features=fraction_of_features,
            additional_parameters=kwargs['additional_parameters'])

    def alter_weights(self, sample, c=0, **kwargs):
        return super(ZoneMCMCWarmup, self).alter_weights(
            sample, additional_parameters=kwargs['additional_parameters'])

    def alter_areal_effect(self, sample, c=0, **kwargs):
        return super(ZoneMCMCWarmup, self).alter_areal_effect(
            sample, additional_parameters=kwargs['additional_parameters'])

    def alter_confounding_effects(self, sample, c=0, **kwargs):
        return super(ZoneMCMCWarmup, self).alter_confounding_effects(
            sample, additional_parameters=kwargs['additional_parameters'])

    def gibbsish_sample_clusters(self, sample, resample_source=True, site_subset=None, c=0):
        return super(ZoneMCMCWarmup, self).gibbsish_sample_clusters(
            sample, resample_source=resample_source, c=c, site_subset=site_subset
        )

    def swap_cluster(self, sample, c=0, resample_source=True, **kwargs):
        """ This functions swaps sites in one of the clusters of the current sample
        (i.e. in of the clusters a site is removed and another one added)
        Args:
            sample(Sample): The current sample with clusters and parameters
            c(int): The current warmup chain
            resample_source(bool): Resample the source?
        Returns:
            Sample: The modified sample.
         """
        sample_new = sample.copy()
        clusters_current = sample.clusters
        occupied = np.any(clusters_current, axis=0)

        if self.model.sample_source and resample_source:
            likelihood = self.posterior_per_chain[sample.chain].likelihood
            lh_per_component = likelihood.update_component_likelihoods(sample=sample, caching=False)
            weights = likelihood.update_weights(sample=sample)
            source_posterior = normalize(lh_per_component * weights, axis=-1)
            # q_per_observation = np.sum(sample.source * source_posterior, axis=2)
            # log_q_back_s = np.sum(np.log(q_per_observation))
            is_source = np.where(sample.source.ravel())
            log_q_back_s = np.sum(np.log(source_posterior.ravel()[is_source]))

        # Randomly choose one of the clusters to modify
        z_id = np.random.choice(range(clusters_current.shape[0]))
        cluster_current = clusters_current[z_id, :]

        neighbours = get_neighbours(cluster_current, occupied, self.adj_mat)
        connected_step = (_random.random() < self.p_grow_connected[c])
        if connected_step:
            # All neighboring sites that are not yet occupied by other clusters are candidates
            candidates = neighbours
        else:
            # All free sites are candidates
            candidates = ~occupied

        # When stuck (all neighbors occupied) return current sample and reject the step (q_back = 0)
        if not np.any(candidates):
            return sample, 0, -np.inf

        # Add a site to the zone
        site_new = _random.choice(candidates.nonzero()[0])
        sample_new.clusters[z_id, site_new] = 1

        # Remove a site from the zone
        removal_candidates = self.get_removal_candidates(cluster_current)
        site_removed = _random.choice(removal_candidates)
        sample_new.clusters[z_id, site_removed] = 0

        # # Compute transition probabilities
        back_neighbours = get_neighbours(cluster_current, occupied, self.adj_mat)
        # q = 1. / np.count_nonzero(candidates)
        # q_back = 1. / np.count_nonzero(back_neighbours)

        # Transition probability growing to the new cluster
        q_non_connected = 1 / np.count_nonzero(~occupied)

        q = (1 - self.p_grow_connected[c]) * q_non_connected
        if neighbours[site_new]:
            q_connected = 1 / np.count_nonzero(neighbours)
            q += self.p_grow_connected[c] * q_connected

        # Transition probability of growing back to the original zone
        q_back_non_connected = 1 / np.count_nonzero(~occupied)
        q_back = (1 - self.p_grow_connected[c]) * q_back_non_connected

        # If z is a neighbour of the new zone, the back step could also be a connected grow step
        if back_neighbours[site_removed]:
            q_back_connected = 1 / np.count_nonzero(back_neighbours)
            q_back += self.p_grow_connected[c] * q_back_connected

        # The step changed the zone (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['clusters'].add(z_id)
        sample.what_changed['lh']['clusters'].add(z_id)
        sample_new.what_changed['prior']['clusters'].add(z_id)
        sample.what_changed['prior']['clusters'].add(z_id)

        assert 0 < q <= 1
        assert 0 < q_back <= 1, q_back

        log_q = np.log(q)
        log_q_back = np.log(q_back)

        if self.model.sample_source and resample_source:
            sample_new, log_q_s, _ = self.gibbs_sample_sources(sample_new, as_gibbs=False)
            log_q += log_q_s
            log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back

    def grow_cluster(self, sample, c=0, resample_source=True, **kwargs):
        """ This functions grows one of the clusters in the current sample (i.e. it adds a new site to one cluster)
        Args:
            sample(Sample): The current sample with clusters and parameters
            c(int): The current warmup chain
            resample_source(bool): Resample the source?
        Returns:
            (Sample): The modified sample.
        """
        sample_new = sample.copy()
        clusters_current = sample.clusters
        occupied = np.any(clusters_current, axis=0)

        if self.model.sample_source and resample_source:
            likelihood = self.posterior_per_chain[sample.chain].likelihood
            lh_per_component = likelihood.update_component_likelihoods(sample=sample, caching=False)
            weights = likelihood.update_weights(sample=sample)
            source_posterior = normalize(lh_per_component * weights, axis=-1)
            # q_per_observation = np.sum(sample.source * source_posterior, axis=2)
            # log_q_back_s = np.sum(np.log(q_per_observation))
            is_source = np.where(sample.source.ravel())
            log_q_back_s = np.sum(np.log(source_posterior.ravel()[is_source]))

        # Randomly choose one of the clusters to modify
        z_id = np.random.choice(range(clusters_current.shape[0]))
        cluster_current = clusters_current[z_id, :]

        # Check if cluster is small enough to grow
        current_size = np.count_nonzero(cluster_current)

        if current_size >= self.max_size[c]:
            # Cluster too big to grow: don't modify the sample and reject the step (q_back = 0)
            return sample, 0, -np.inf

        neighbours = get_neighbours(cluster_current, occupied, self.adj_mat)
        connected_step = (_random.random() < self.p_grow_connected[c])
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
        site_new = _random.choice(candidates.nonzero()[0])
        sample_new.clusters[z_id, site_new] = 1

        # Transition probability when growing
        q_non_connected = 1 / np.count_nonzero(~occupied)
        q = (1 - self.p_grow_connected[c]) * q_non_connected

        if neighbours[site_new]:
            q_connected = 1 / np.count_nonzero(neighbours)
            q += self.p_grow_connected[c] * q_connected

        # Back-probability (shrinking)
        q_back = 1 / (current_size + 1)

        # The step changed the zone (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['clusters'].add(z_id)
        sample.what_changed['lh']['clusters'].add(z_id)
        sample_new.what_changed['prior']['clusters'].add(z_id)
        sample.what_changed['prior']['clusters'].add(z_id)

        assert 0 < q <= 1
        assert 0 < q_back <= 1

        log_q = np.log(q)
        log_q_back = np.log(q_back)

        if self.model.sample_source and resample_source:
            sample_new, log_q_s, _ = self.gibbs_sample_sources(sample_new, as_gibbs=False)
            log_q += log_q_s
            log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back

    def shrink_cluster(self, sample, c=0, resample_source=True, **kwargs):
        """ This functions shrinks one of the clusters in the current sample (i.e. it removes one site from one cluster)
        Args:
            sample(Sample): The current sample with clusters and parameters
            c(int): The current warmup chain
            resample_source(bool): Resample the source?
        Returns:
            (Sample): The modified sample.
        """
        sample_new = sample.copy()
        clusters_current = sample.clusters

        if self.model.sample_source and resample_source:
            likelihood = self.posterior_per_chain[sample.chain].likelihood
            lh_per_component = likelihood.update_component_likelihoods(sample=sample, caching=False)
            weights = likelihood.update_weights(sample=sample)
            source_posterior = normalize(lh_per_component * weights, axis=-1)
            # q_per_observation = np.sum(sample.source * source_posterior, axis=2)
            # log_q_back_s = np.sum(np.log(q_per_observation))
            is_source = np.where(sample.source.ravel())
            log_q_back_s = np.sum(np.log(source_posterior.ravel()[is_source]))

        # Randomly choose one of the clusters to modify
        z_id = np.random.choice(range(clusters_current.shape[0]))
        cluster_current = clusters_current[z_id, :]

        # Check if cluster is big enough to shrink
        current_size = np.count_nonzero(cluster_current)
        if current_size <= self.min_size:
            # Cluster is too small to shrink: don't modify the sample and reject the step (q_back = 0)
            return sample, 0, -np.inf

        # Cluster is big enough: shrink
        removal_candidates = self.get_removal_candidates(cluster_current)
        site_removed = _random.choice(removal_candidates)
        sample_new.clusters[z_id, site_removed] = 0

        # Transition probability when shrinking.
        q = 1 / len(removal_candidates)
        # Back-probability (growing)
        cluster_new = sample_new.clusters[z_id]
        occupied_new = np.any(sample_new.clusters, axis=0)
        back_neighbours = get_neighbours(cluster_new, occupied_new, self.adj_mat)

        # The back step could always be a non-connected grow step
        q_back_non_connected = 1 / np.count_nonzero(~occupied_new)
        q_back = (1 - self.p_grow_connected[c]) * q_back_non_connected

        # If z is a neighbour of the new zone, the back step could also be a connected grow step
        if back_neighbours[site_removed]:
            q_back_connected = 1 / np.count_nonzero(back_neighbours)
            q_back += self.p_grow_connected[c] * q_back_connected

        # Back-probability (shrinking)
        q_back = 1 / (current_size + 1)

        # self.q_areas_stats['q_shrink'].append(q)
        # self.q_areas_stats['q_back_shrink'].append(q_back)

        # The step changed the zone (which has an influence on how the lh and the prior look like)
        sample_new.what_changed['lh']['clusters'].add(z_id)
        sample.what_changed['lh']['clusters'].add(z_id)
        sample_new.what_changed['prior']['clusters'].add(z_id)
        sample.what_changed['prior']['clusters'].add(z_id)

        assert 0 < q <= 1
        assert 0 < q_back <= 1

        log_q = np.log(q)
        log_q_back = np.log(q_back)

        if self.model.sample_source and resample_source:
            sample_new, log_q_s, _ = self.gibbs_sample_sources(sample_new, as_gibbs=False)
            log_q += log_q_s
            log_q_back += log_q_back_s

        return sample_new, log_q, log_q_back
