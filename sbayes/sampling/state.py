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
        zones (np.array): Assignment of sites to zones.
            shape: (n_zones, n_sites)
        weights (np.array): Weights of zone, family and global likelihood for different features.
            shape: (n_features, n_components)
        p_global (np.array): Global probabilities of categories
            shape(1, n_features, n_states)
        p_zones (np.array): Probabilities of categories in zones
            shape: (n_zones, n_features, n_states)
        p_families (np.array): Probabilities of categories in families
            shape: (n_families, n_features, n_states)
        source (np.array): Assignment of single observations (a feature in a language) to be
                           the result of the global, zone or family distribution.
            shape: (n_sites, n_features, n_components)

        what_changed (dict): Flags to indicate which parts of the state changed since
                             the last likelihood/prior evaluation.
        last_lh (float): The last log-likelihood value of this sample (for logging).
        last_prior (float): The last log-prior value of this sample (for logging).
        observation_lhs (np.array): The likelihood of each observation, given the state of
                                    the sample.
            shape: (n_sites, n_features)

    """

    def __init__(self, zones, weights, p_global, p_zones, p_families, source=None, chain=0):
        self.zones = zones
        self.weights = weights
        self.p_global = p_global
        self.p_zones = p_zones
        self.p_families = p_families
        self.source = source
        self.chain = chain

        # The sample contains information about which of its parameters was changed in the last MCMC step
        self.what_changed = {}
        self.everything_changed()

        # Store last likelihood and prior for logging
        self.last_lh = 0.0
        self.last_prior = 0.0

        # Store the likelihood of each observation (language and feature) for logging
        self.observation_lhs = None

        self.i_step = 0

    @property
    def n_areas(self):
        return self.zones.shape[0]

    @property
    def n_sites(self):
        return self.zones.shape[1]

    @property
    def n_features(self):
        return self.weights.shape[0]

    @property
    def n_components(self):
        return self.weights.shape[1]

    @property
    def n_families(self):
        return self.p_families.shape[0]

    @property
    def n_states(self):
        return self.p_families.shape[2]

    @property
    def inheritance(self):
        return self.n_components == 3

    @classmethod
    def empty_sample(cls):
        initial_sample = cls(zones=None, weights=None, p_global=None, p_zones=None, p_families=None)
        initial_sample.everything_changed()
        return initial_sample

    def everything_changed(self):
        self.what_changed = {
            'lh': {'zones': IndexSet(), 'weights': True,
                   'p_global': IndexSet(), 'p_zones': IndexSet(), 'p_families': IndexSet()},
            'prior': {'zones': IndexSet(), 'weights': True,
                      'p_global': IndexSet(), 'p_zones': IndexSet(), 'p_families': IndexSet()}
        }

    def copy(self):
        zone_copied = deepcopy(self.zones)
        weights_copied = deepcopy(self.weights)
        what_changed_copied = deepcopy(self.what_changed)

        def maybe_copy(obj):
            if obj is not None:
                return obj.copy()

        p_global_copied = maybe_copy(self.p_global)
        p_zones_copied = maybe_copy(self.p_zones)
        p_families_copied = maybe_copy(self.p_families)
        source_copied = maybe_copy(self.source)

        new_sample = Sample(chain=self.chain, zones=zone_copied, weights=weights_copied,
                            p_global=p_global_copied, p_zones=p_zones_copied, p_families=p_families_copied,
                            source=source_copied)

        new_sample.what_changed = what_changed_copied

        return new_sample
