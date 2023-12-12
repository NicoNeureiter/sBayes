import logging
import random

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax, logsumexp
from scipy.stats import truncnorm

from sbayes.load_data import Data
from sbayes.model import Model, update_weights
from sbayes.sampling.conditionals import sample_source_from_prior
from sbayes.sampling.counts import recalculate_feature_counts
from sbayes.sampling.loggers import ClustersLogger
from sbayes.sampling.operators import ObjectSelector, GibbsSampleSource, AlterCluster, AlterClusterWide
from sbayes.sampling.state import Sample, Clusters
from sbayes.util import get_neighbours, normalize

EPS = 1E-10
RNG = np.random.default_rng()


class ClusterError(Exception):
    pass


class DummySample(Sample):
    def __init__(self, clusters):
        self._clusters = Clusters(clusters)


class SbayesInitializer:

    def __init__(
        self,
        model: Model,
        data: Data,
        initial_size: int,
        attempts: int,
        initial_cluster_steps: bool,
        n_em_steps: int = 50,
        logger: logging.Logger = None,
    ):
        self.model = model
        self.data = data

        self.initialization_attempts = attempts
        self.initial_cluster_steps = initial_cluster_steps
        self.n_em_steps = n_em_steps

        # Data
        self.features = data.features.values
        self.applicable_states = data.features.states
        self.n_states_by_feature = np.sum(data.features.states, axis=-1)
        self.n_features = self.features.shape[1]
        self.n_states = self.features.shape[2]
        self.n_objects = self.features.shape[0]

        # Locations and network
        self.adj_mat = data.network.adj_mat

        # Clustering
        self.n_clusters = model.n_clusters
        self.min_size = model.min_size
        self.max_size = model.max_size
        self.initial_size = initial_size

        # Confounders and sources
        self.confounders = model.confounders
        self.n_sources = 1 + len(self.confounders)
        self.n_groups = self.get_groups_per_confounder()

        if logger is None:
            import logging
            self.logger = logging.getLogger()
        else:
            self.logger = logger

        self.init_cluster_logger: ClustersLogger = None

    def get_groups_per_confounder(self):
        n_groups = dict()
        for k, v in self.confounders.items():
            n_groups[k] = v.n_groups
        return n_groups

    def sample_n_objects_in_all_clusters(self, mid, lower_bound, upper_bound):
        """Draw the total number of objects in all clusters from a truncated normal distribution."""
        scale = max(20, mid - lower_bound)
        a = (lower_bound - mid) / scale
        b = (upper_bound - mid) / scale
        return int(truncnorm(a, b, loc=mid, scale=scale).rvs())

    def generate_clusters_em(self):
        """EM-style inference of a continuous approximation of the sBayes model.
        We estimate distributions for each cluster/confounder group and a flat continuous
        assignment of each object to clusters and confounder groups.
        """
        features = self.data.features.values
        valid_observations = ~self.data.features.na_values
        n_objects, n_features, n_states = features.shape
        n_clusters = self.model.n_clusters

        # Decide how many objects to assign to clusters in total
        total_size = self.sample_n_objects_in_all_clusters(
            mid=n_clusters * self.initial_size,
            lower_bound=n_clusters * self.model.min_size,
            upper_bound=min(n_objects, n_clusters * self.model.max_size),
        )

        groups = [f"a{c}" for c in range(n_clusters)]
        for conf_name, conf in self.data.confounders.items():
            groups += [f"{conf_name}_{grp_name}" for grp_name in conf.group_names]

        n_groups = len(groups)

        groups_available = np.zeros((n_groups, n_objects), dtype=bool)
        groups_available[:n_clusters, :] = True
        i = n_clusters
        for conf_name, conf in self.data.confounders.items():
            groups_available[i:i+conf.n_groups, :] = conf.group_assignment
            i += conf.n_groups

        prior_counts = 0.5 * self.data.features.states

        z = normalize(np.random.random((n_groups, n_objects)) * groups_available, axis=0)

        # Decide whether to use geo-prior and, if yes, extract the cost_matrix
        geo_prior = self.model.prior.geo_prior
        consider_geo_prior = geo_prior.prior_type is geo_prior.PriorTypes.COST_BASED
        log_geo_priors = 0.0

        _features = np.copy(features)
        _features[~valid_observations, :] = True

        for i_step in range(self.n_em_steps):

            state_counts = np.einsum("ij,jkl->ikl", z, features, optimize='optimal')
            # shape: (n_groups, n_features, n_states)

            p = normalize(state_counts + prior_counts, axis=-1)
            # shape: (n_groups, n_features, n_states)

            # How likely would each feature observation be if it was explained by each group
            pointwise_ll_by_group = np.log(np.einsum("ikl,jkl->ijk", p, _features, optimize='optimal'))
            # shape: (n_groups, n_objects, n_features)

            group_lls = np.sum(pointwise_ll_by_group, axis=-1)
            # shape: (n_groups, n_objects)

            if consider_geo_prior:
                z_peaky = softmax(n_objects * z, axis=1)
                avg_dist_to_cluster = z_peaky.dot(geo_prior.cost_matrix)
                log_geo_priors = -avg_dist_to_cluster / geo_prior.scale / 2

                # Non-cluster groups get the average geo-prior value of the clusters
                log_geo_priors[n_clusters:] = logsumexp(log_geo_priors[:n_clusters]) - np.log(log_geo_priors[:n_clusters].size)

            temperature = (self.n_em_steps / (1+i_step)) ** 3
            lh = log_geo_priors + group_lls / temperature

            lh = np.where(groups_available, lh, -np.inf)
            z = softmax(lh, axis=0)

            if (self.init_cluster_logger is not None) and (i_step % 5 == 0):
                clusters = self.discretize_fuzzy_cluster_2(z, total_size=total_size)
                self.init_cluster_logger.write_sample(sample=DummySample(clusters))

        clusters = self.discretize_fuzzy_cluster_2(z, total_size=total_size)
        return clusters

    def discretize_fuzzy_cluster(self, z):
        # z represents a probabilistic assignment to clusters and confounder groups
        # z[:n_clusters] is a soft assignment of objects to clusters:
        fuzzy_clusters = np.copy(z[:self.model.n_clusters + 1])
        fuzzy_clusters[-1] = 0.9 * np.sum(fuzzy_clusters[:-1], axis=0)

        # Select the highest fuzzy value
        best = np.argmax(fuzzy_clusters, axis=0)
        clusters = np.eye(self.model.n_clusters+1, dtype=bool)[best].T

        # Kick out the "no cluster" assignments
        clusters = clusters[:-1]

        return clusters

    def discretize_fuzzy_cluster_2(self, z: NDArray[float], total_size: int):
        n_clusters = self.model.n_clusters
        n_objects = self.model.shapes.n_sites

        # z represents a probabilistic assignment to clusters and confounder groups
        # z[:n_clusters] is a soft assignment of objects to clusters:
        fuzzy_clusters = np.copy(z[:n_clusters])

        # Make sure every cluster get at least min_size objects assigned
        for i_c in range(n_clusters):
            best_ids = np.argsort(fuzzy_clusters[i_c])[-self.model.min_size:]
            fuzzy_clusters[:, best_ids] = 0
            fuzzy_clusters[i_c, best_ids] = 1

        best = np.argmax(fuzzy_clusters, axis=0)
        best_value = np.max(fuzzy_clusters, axis=0)

        threshold = np.sort(best_value)[-total_size]
        best[best_value < threshold] = n_clusters
        clusters = np.eye(n_clusters+1, dtype=bool)[best].T

        # Kick out the "no cluster" assignments
        clusters = clusters[:-1]
        return clusters

    def generate_sample(self, c: int = 0, log_initialization: bool = False) -> Sample:
        """Generate initial Sample object (clusters, weights, cluster_effect, confounding_effects)
        Kwargs:
            c: index of the MCMC chain
        Returns:
            The generated initial Sample
        """

        best_sample = None
        best_lh = -np.inf
        assert self.initialization_attempts > 0
        for i_attempt in range(self.initialization_attempts):

            if log_initialization:
                self.init_cluster_logger = ClustersLogger(
                    f"./cluster_initialization_steps_{c}_{i_attempt}.txt",
                    self.data, self.model, resume=False
                )

            sample = self.generate_sample_attempt(c, i_attempt)
            lh = self.model.likelihood(sample, caching=False)
            # lh = self.model(sample, caching=False)
            if lh > best_lh:
                best_sample = sample
                best_lh = lh

        best_sample.i_step = 0
        return best_sample

    def generate_sample_attempt(self, c: int = 0, i_attempt: int = 0) -> Sample:

        initial_clusters = self.generate_clusters_em()
        initial_weights = self.generate_initial_weights()
        initial_source = np.empty((self.n_objects, self.n_features, self.n_sources), dtype=bool)
        sample = Sample.from_numpy_arrays(
            clusters=initial_clusters,
            weights=initial_weights,
            confounders=self.data.confounders,
            source=initial_source,
            feature_counts={'clusters': np.zeros((self.n_clusters, self.n_features, self.n_states)),
                            **{conf: np.zeros((n_groups, self.n_features, self.n_states))
                               for conf, n_groups in self.n_groups.items()}},
            chain=c,
            model_shapes=self.model.shapes,
        )

        # Generate the initial source using a Gibbs sampling step
        sample.everything_changed()
        source = sample_source_from_prior(sample)
        source[self.data.features.na_values] = 0
        sample.source.set_value(source)
        recalculate_feature_counts(self.features, sample)

        assert np.isfinite(self.model(sample))

        w = update_weights(sample, caching=False)
        s = sample.source.value
        assert np.all(s <= (w > 0)), np.max(w)

        full_source_operator = GibbsSampleSource(
            weight=1,
            model=self.model,
            sample_from_prior=False,
            object_selector=ObjectSelector.ALL,
        )
        cluster_operator = AlterClusterWide(
            weight=0,
            adjacency_matrix=self.data.network.adj_mat,
            model=self.model,
            features=self.data.features.values,
            consider_geo_prior=True,
        )
        source = full_source_operator.function(sample)[0].source.value
        sample.source.set_value(source)
        recalculate_feature_counts(self.features, sample)

        # for _ in range(1, self.initial_size):
        #     for i_c in range(self.n_clusters):
        #         sample = self.grow(sample, cluster_operator, i_c)

        if self.initial_cluster_steps:
            # Apply cluster operator to each cluster
            for i_c in range(self.n_clusters):
                sample = cluster_operator.ml_step(sample, i_cluster=i_c)

            if self.init_cluster_logger is not None:
                self.init_cluster_logger.write_sample(sample)

            # Estimate weights by counting source observations in each feature
            s = sample.source.value
            S = sample.cache.has_components.value
            s_ratio = np.sum(s, axis=0) / np.sum(S, axis=0, keepdims=True)
            sample.weights.set_value(
                normalize(1 + s_ratio, axis=-1)
            )

            # Propose a new source array
            source = full_source_operator.function(sample)[0].source.value
            sample.source.set_value(source)
            recalculate_feature_counts(self.features, sample)

            # Apply cluster operator to each cluster again
            for i_c in range(self.n_clusters):
                sample = cluster_operator.ml_step(sample, i_cluster=i_c)

            if self.init_cluster_logger is not None:
                self.init_cluster_logger.write_sample(sample)

        sample.everything_changed()
        return sample

    @staticmethod
    def grow(sample: Sample, cluster_operator: AlterCluster, i_cluster: int) -> Sample:
        for _ in range(20):
            sample, q, q_back = cluster_operator.grow_cluster(sample, i_cluster=i_cluster)
            if q != cluster_operator.Q_REJECT:
                return sample

        # Give up and raise exception after 20 attempts
        raise ClusterError

    def generate_initial_clusters(self, c : int = 0):
        return self.initialize_clusters()
        # return self.grow_random_clusters()

    def initialize_clusters(self) -> NDArray[bool]:
        """Grow clusters in parallel, each starting from a random point, and
        incrementally adding similar objects."""

        # Start with empty clusters
        clusters = np.zeros((self.n_clusters, self.n_objects), bool)
        free_objs = list(range(self.n_objects))

        # Choose a random starting object for each cluster
        for i_c in range(self.n_clusters):
            # Take a random free site and use it as seed for the new cluster
            j = random.sample(free_objs, 1)[0]
            clusters[i_c, j] = True
            free_objs.remove(j)

        return clusters

    def grow_random_clusters(self) -> NDArray[bool]:  # (n_clusters, n_objects)
        """Generate initial clusters by growing through random grow-steps up to self.min_size."""

        # If there are no clusters in the model, return empty matrix
        if self.n_clusters == 0:
            return np.zeros((self.n_clusters, self.n_objects), bool)

        occupied = np.zeros(self.n_objects, bool)
        initial_clusters = np.zeros((self.n_clusters, self.n_objects), bool)

        # A: Grow the remaining clusters
        # With many clusters new ones can get stuck due to unfavourable seeds.
        # We perform several attempts to initialize the clusters.
        attempts = 0
        max_attempts = 1000
        n_initialized = 0
        restart = False

        while True:
            if restart:
                occupied = np.zeros(self.n_objects, bool)
                initial_clusters = np.zeros((self.n_clusters, self.n_objects), bool)
                n_initialized = 0
                restart = False

            for i in range(n_initialized, self.n_clusters):
                try:
                    initial_size = self.initial_size
                    cl, in_cl = self.grow_cluster_of_size_k(k=initial_size, already_in_cluster=occupied)

                except ClusterError:
                    # Rerun: Error might be due to an unfavourable seed
                    if attempts < max_attempts:
                        attempts += 1
                        if attempts % 20 == 0 and self.initial_size > 3:
                            print(np.sum(initial_clusters, axis=-1))
                            self.initial_size -= 1
                            restart = True
                            self.logger.warning(f"Reduced 'initial_size' to {self.initial_size} after "
                                                f"{attempts} unsuccessful initialization attempts.")
                        break
                    # Seems there is not enough sites to grow n_clusters of size k
                    else:
                        raise ValueError(f"Failed to add additional cluster. Try fewer clusters "
                                         f"or set initial_sample to None.")

                initial_clusters[i, :] = cl
                # n_initialized += 1
            else:  # No break -> no cluster exception
                return initial_clusters

    def grow_cluster_of_size_k(self, k, already_in_cluster=None, ):
        """ This function grows a cluster of size k excluding any of the sites in <already_in_cluster>.
        Args:
            k (int): The size of the cluster, i.e. the number of sites in the cluster
            already_in_cluster (np.array): All sites already assigned to a cluster (boolean)

        Returns:
            np.array: The newly grown cluster (boolean).
            np.array: all sites already assigned to a cluster (boolean).

        """
        if already_in_cluster is None:
            already_in_cluster = np.zeros(self.n_objects, bool)

        # Initialize the cluster
        cluster = np.zeros(self.n_objects, bool)

        # Find all sites that are occupied by a cluster and those that are still free
        sites_occupied = np.nonzero(already_in_cluster)[0]
        sites_free = list(set(range(self.n_objects)) - set(sites_occupied))

        # Take a random free site and use it as seed for the new cluster
        try:
            i = random.sample(sites_free, 1)[0]
            cluster[i] = already_in_cluster[i] = 1
        except ValueError:
            raise ClusterError

        # Grow the cluster if possible
        for _ in range(k - 1):
            neighbours = get_neighbours(cluster, already_in_cluster, self.adj_mat)
            if not np.any(neighbours):
                raise ClusterError

            # Add a neighbour to the cluster
            site_new = random.choice(list(neighbours.nonzero()[0]))
            cluster[site_new] = already_in_cluster[site_new] = 1

        return cluster, already_in_cluster

    def generate_initial_weights(self):
        """This function generates initial weights for the Bayesian additive mixture model, either by
        A) proposing them (randomly) from scratch
        B) using the last sample of a previous run of the MCMC
        Weights are in log-space and not normalized.

        Returns:
            np.array: weights for cluster_effect and each of the i confounding_effects
        """
        return normalize(np.ones((self.n_features, self.n_sources)))

    def generate_initial_confounding_effect(self, conf: str):
        """This function generates initial state probabilities for each group in confounding effect [i], either by
        A) using the MLE
        B) using the last sample of a previous run of the MCMC
        Probabilities are in log-space and not normalized.
        Args:
            conf: The confounding effect [i]
        Returns:
            np.array: probabilities for states in each group of confounding effect [i]
                shape (n_groups, n_features, max(n_states))
        """

        n_groups = self.n_groups[conf]
        groups = self.confounders[conf].group_assignment

        initial_confounding_effect = np.zeros((n_groups, self.n_features, self.features.shape[2]))

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
