from copy import deepcopy


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
        other = IndexSet(all_i=self.all)
        for element in self:
            other.add(element)

        return other

    def __deepcopy__(self, memo):
        other = IndexSet(all_i=self.all)
        for element in self:
            other.add(deepcopy(element))

        return other


class Sample(object):
    """
    Attributes:
        chain (int): index of the MC3 chain.
        clusters (np.array): Assignment of sites to clusters.
            shape: (n_clusters, n_sites)
        weights (np.array): Weights of the areal effect and each of the i confounding effects
            shape: (n_features, 1 + n_confounders)
        cluster_effect (np.array): Probabilities of states in the clusters
            shape: (n_clusters, n_features, n_states)
        confounding_effects (dict): Probabilities of states for each of the i confounding effects,
            each confounding effect [i] with shape: (n_groups[i], n_features, n_states)
        source (np.array): Assignment of single observations (a feature in an object) to be
                           the result of the areal effect or one of the i confounding effects
            shape: (n_sites, n_features, 1 + n_confounders)
        what_changed (dict): Flags to indicate which parts of the state changed since
                             the last likelihood/prior evaluation.
        last_lh (float): The last log-likelihood value of this sample (for logging).
        last_prior (float): The last log-prior value of this sample (for logging).
        observation_lhs (np.array): The likelihood of each observation, given the state of
                                    the sample.
            shape: (n_sites, n_features)
    """

    def __init__(self, clusters, weights, cluster_effect, confounding_effects, source=None, chain=0):
        self.clusters = clusters
        self.weights = weights
        self.cluster_effect = cluster_effect
        self.confounding_effects = confounding_effects
        self.source = source
        self.chain = chain

        # The sample contains information about which of its parameters was changed in the last MCMC step
        self.what_changed = {}
        self.everything_changed()

        # Store last likelihood and prior for logging
        self.last_lh = None
        self.last_prior = None

        # Store the likelihood of each observation (language and feature) for logging
        self.observation_lhs = None

        self.i_step = 0

    @property
    def n_clusters(self):
        return self.clusters.shape[0]

    @property
    def n_sites(self):
        return self.clusters.shape[1]

    @property
    def n_features(self):
        return self.weights.shape[0]

    @property
    def n_components(self):
        return self.weights.shape[1]

    @property
    def n_states(self):
        return self.cluster_effect.shape[2]

    @classmethod
    def empty_sample(cls, conf):
        initial_sample = cls(clusters=None, weights=None, cluster_effect=None,
                             confounding_effects={k: None for k in conf})
        initial_sample.everything_changed()
        return initial_sample

    def everything_changed(self):

        self.what_changed = {
            'lh': {'clusters': IndexSet(), 'weights': True, 'cluster_effect': IndexSet(),
                   'confounding_effects': {k: IndexSet() for k in self.confounding_effects}},
            'prior': {'clusters': IndexSet(), 'weights': True, 'cluster_effect': IndexSet(),
                      'confounding_effects': {k: IndexSet() for k in self.confounding_effects}}}

    def copy(self):
        clusters_copied = deepcopy(self.clusters)
        weights_copied = deepcopy(self.weights)
        what_changed_copied = deepcopy(self.what_changed)

        def maybe_copy(obj):
            if obj is not None:
                return obj.copy()

        cluster_effect_copied = maybe_copy(self.cluster_effect)
        confounding_effects_copied = dict()

        for k, v in self.confounding_effects.items():
            confounding_effects_copied[k] = maybe_copy(v)

        source_copied = maybe_copy(self.source)

        new_sample = Sample(chain=self.chain, clusters=clusters_copied, weights=weights_copied,
                            cluster_effect=cluster_effect_copied, confounding_effects=confounding_effects_copied,
                            source=source_copied)

        new_sample.what_changed = what_changed_copied

        return new_sample
