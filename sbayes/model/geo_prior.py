from __future__ import annotations


from sbayes.config.config import GeoPriorConfig
from numpy.typing import NDArray

from sbayes.model.thermodynamic_integration import estimate_marginal_log_likelihood_curve
from sbayes.network import Network

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from typing import Callable, TYPE_CHECKING

from sbayes.util import log_expit

if TYPE_CHECKING:
    from sbayes.model.prior import ClusterPrior


class GeoPrior:

    """Prior on the geographic spread of the clusters."""

    PriorTypes = GeoPriorConfig.Types
    AggrStrats = GeoPriorConfig.AggregationStrategies

    def __init__(
        self,
        config: GeoPriorConfig,
        cost_matrix: NDArray[np.float64] | None = None,
        network: Network | None = None,
    ) -> None:
        """
        Args:
            config: the geo-prior configuration
            cost_matrix: the pairwise costs between objects
            network: the network over the objects
        """

        self.config = config
        self.cost_matrix = None if cost_matrix is None else jnp.asarray(cost_matrix)
        self.network = network
        self.prior_type = config.type

        self.aggregation_policy = None
        self.prob_func_type = None
        self.rate = None
        self.inflection_point = None

        # Set by `calibrate`, only used when `estimate_rate` is set
        self._log_norm_const_interpolator: Callable | None = None
        self._norm_const_grid_min: float | None = None
        self._norm_const_grid_max: float | None = None

        if self.prior_type is self.PriorTypes.COST_BASED:
            if self.cost_matrix is None:
                raise ValueError("A `cost_based` geo-prior requires a cost matrix.")
            self.rate = config.rate
            self.aggregation_policy = config.aggregation
            self.prob_func_type = config.probability_function
            self.inflection_point = config.inflection_point


    def same_cluster_prob(self, clusters: jnp.ndarray) -> jnp.ndarray:
        """Probability that each pair of objects is in the same cluster.

        Args:
            clusters: the fuzzy cluster assignments. shape: (n_objects, n_clusters)

        Returns:
            shape: (n_objects, n_objects, n_clusters)
        """
        cluster_size = jnp.sum(clusters, axis=-2)
        clusters_normed = clusters / cluster_size[None, :]

        if self.aggregation_policy is self.AggrStrats.MEAN:
            return jnp.einsum("ik,jk->ijk", clusters_normed, clusters_normed)
        elif self.aggregation_policy is self.AggrStrats.SUM:
            return jnp.einsum("ik,jk->ijk", clusters, clusters)
        elif self.aggregation_policy is self.AggrStrats.SUM_OF_MEAN:
            return jnp.einsum("ik,jk->ijk", clusters_normed, clusters)
        else:
            raise ValueError(
                f"Unknown aggregation policy `{self.aggregation_policy}`."
            )

    def calibrate(self, cluster_prior: ClusterPrior) -> None:
        """Estimate the normalization constant of the geo-prior over a grid of rates.

        The cost-based geo-prior is unnormalized: its normalization constant depends on
        the rate, so a sampled rate requires knowing that constant as a function of the
        rate. It is estimated once by thermodynamic integration and stored as an
        interpolator over a grid around the configured rate.

        Args:
            cluster_prior: the prior on the cluster assignments, used as the base measure
        """
        if not self.config.estimate_rate:
            return

        def cost(z: jax.Array, rate: float) -> jax.Array:
            clusters = z[..., :-1]
            same_cluster_prob = self.same_cluster_prob(clusters) # (n_objects, n_clusters)

            total_dist_per_cluster = jnp.sum(
                same_cluster_prob * self.cost_matrix[..., None], axis=(0, 1)
            )
            return self.log_prob(total_dist_per_cluster, rate)

        # Grid around the configured rate, denser towards small rates
        grid_size = self.config.approx_norm_const["grid_size"]
        grid_min = self.rate / 8.0
        grid_max = self.rate * 4.0
        r_grid = jnp.linspace(grid_min ** 0.5, grid_max ** 0.5, grid_size) ** 2

        self._log_norm_const_interpolator, _, _ = estimate_marginal_log_likelihood_curve(
            base_prior=lambda: cluster_prior.get_numpyro_distr(
                allow_reparameterization=False
            ),
            log_g_fn=cost,
            r_grid=r_grid,
            num_samples=self.config.approx_norm_const["steps_per_setting"],
        )

        self._norm_const_grid_min = float(jnp.min(r_grid))
        self._norm_const_grid_max = float(jnp.max(r_grid))


    def log_norm_const(self, rate: jnp.ndarray | float) -> jnp.ndarray:
        """Look up the log normalization constant of the geo-prior for a given rate.

        Requires `calibrate` to have been called, which estimates the constant over a
        grid of rates. Rates outside that grid are clamped to its bounds, since
        extrapolating the interpolator yields NaN.

        Args:
            rate: the rate of the geo-prior

        Returns:
            The log normalization constant at `rate`.
        """
        if self._log_norm_const_interpolator is None:
            raise ValueError(
                "The geo-prior normalization constant has not been estimated. "
                "`calibrate` must be called before sampling with `estimate_rate`."
            )

        rate_clamped = jnp.clip(
            jnp.atleast_1d(rate),
            self._norm_const_grid_min,
            self._norm_const_grid_max,
        )
        return self._log_norm_const_interpolator(rate_clamped)

    def log_prob(self, x: jnp.ndarray, rate: jnp.ndarray | float) -> jnp.ndarray:
        """Map an aggregated distance to a log-probability.

        The functional form is set by the `probability_function` config field.

        Args:
            x: the aggregated distance within each cluster
            rate: the rate (scale) of the probability function

        Returns:
            The log-probability of the aggregated distance.
        """
        prob_funcs = GeoPriorConfig.ProbabilityFunction
        x_agg = jnp.sum(x)

        if self.prob_func_type is prob_funcs.EXPONENTIAL:
            return -x_agg / rate                # == log(e**(-x/rate))

        elif self.prob_func_type is prob_funcs.SQUARED_EXPONENTIAL:
            return -(x_agg / rate) ** 2         # == log(e**(-(x/rate)**2))

        elif self.prob_func_type is prob_funcs.SIGMOID:
            if self.inflection_point is None:
                raise ValueError("A `sigmoid` probability function requires an `inflection_point`.")
            x0 = self.inflection_point
            return log_expit(-(x_agg - x0) / rate) - log_expit(x0 / rate)

        elif self.prob_func_type is prob_funcs.GAMMA_EXPONENTIAL:
            # Hierarchical model:
            #   x_i | lambda_i ~ Exponential(rate=lambda_i)
            #   lambda_i ~ Gamma(shape=alpha, rate=beta)
            # Marginal for each x_i: p(x_i) = alpha * beta**alpha / (x_i + beta)**(alpha+1)
            # Note: this sums over the elements of `x` instead of aggregating them first.
            alpha = 5.0
            beta = rate * (alpha - 1.0)
            return jnp.sum(
                jnp.log(alpha)
                + alpha * jnp.log(beta)
                - (alpha + 1.0) * jnp.log(x + beta)
            )

        else:
            raise ValueError(
                f"Unknown probability function `{self.prob_func_type}`."
            )

    def add_geo_prior(self, clusters: jnp.ndarray) -> jnp.ndarray | float:
        """Add the geo-prior of the fuzzy clusters to the model.

        The aggregated within-cluster distance is computed according to the configured
        skeleton and mapped to a log-probability, which is added to the model as a
        factor. If the rate is estimated, it is sampled here and the log-probability is
        normalized by the calibrated normalization constant.

        Args:
            clusters: current sample of the fuzzy cluster assignments.
                shape: (n_objects, n_clusters)

        Returns:
            The log-probability of the aggregated distance.
        """
        if self.prior_type is self.PriorTypes.UNIFORM:
            return 0.0

        if self.rate is None:
            raise ValueError("A `cost_based` geo-prior requires a `rate`.")

        aggregated_distance = self.aggregated_distance(clusters)

        if self.config.estimate_rate:
            # Sample the rate around the configured value, on a log scale
            sigma = 1.0
            mu = jnp.log(self.rate)
            log_rate = numpyro.sample("geoprior_log_scale", dist.Normal(mu, sigma))
            rate = jnp.exp(log_rate)
            numpyro.deterministic("geoprior_scale", rate)
            log_norm_const = self.log_norm_const(rate)
        else:
            rate = self.rate
            log_norm_const = 0.0

        log_geo_prior = self.log_prob(aggregated_distance, rate)
        numpyro.factor("geoprior", log_geo_prior - log_norm_const)
        numpyro.deterministic("geoprior_total_dist", aggregated_distance)

        return log_geo_prior

    def aggregated_distance(self, clusters: jnp.ndarray) -> jnp.ndarray:
        """Aggregate the within-cluster distances according to the configured skeleton.

        Args:
            clusters: the fuzzy cluster assignments. shape: (n_objects, n_clusters)

        Returns:
            The aggregated distance, per cluster or summed over clusters depending on
            the skeleton.
        """
        skeletons = GeoPriorConfig.Skeleton

        if self.cost_matrix is None:
            raise ValueError("A `cost_based` geo-prior requires a cost matrix.")


        if self.config.skeleton is skeletons.MST:
            raise NotImplementedError(
                "The `mst` skeleton is not supported: the minimum spanning tree cannot "
                "be computed on fuzzy cluster assignments. Use `complete`, `spectral` "
                "or `diameter` instead."
            )

        elif self.config.skeleton is skeletons.COMPLETE:
            same_cluster_prob = self.same_cluster_prob(clusters)
            return jnp.sum(
                same_cluster_prob * self.cost_matrix[..., None], axis=(0, 1)
            )

        elif self.config.skeleton is skeletons.SPECTRAL:
            def get_spectrum(c: jnp.ndarray) -> jnp.ndarray:
                laplacian = jnp.fill_diagonal(c, jnp.sum(c, axis=-1), inplace=False)
                return jnp.sum(jnp.real(jnp.linalg.eigvals(laplacian)))

            connectivities = clusters.T[:, :, None] * clusters.T[:, None, :]
            # shape: (n_clusters, n_objects, n_objects)
            eigvals = jax.vmap(get_spectrum)(connectivities * self.cost_matrix)
            return jnp.sum(jnp.real(eigvals))

        elif self.config.skeleton is skeletons.DIAMETER:
            return jnp.sum(average_max_distance(clusters, self.cost_matrix))

        else:
            raise ValueError(f"Unknown skeleton type `{self.config.skeleton}`.")

    def get_setup_message(self) -> str:
        """Compile a set-up message for logging."""
        msg = f"Geo-prior: {self.prior_type.value}\n"

        if self.prior_type is self.PriorTypes.COST_BASED:
            msg += f"\tProbability function: {self.prob_func_type.value}\n"
            msg += f"\tAggregation policy: {self.aggregation_policy.value}\n"
            msg += f"\tRate: {self.rate}\n"
            msg += f"\tSkeleton: {self.config.skeleton.value}\n"

            if self.prob_func_type is GeoPriorConfig.ProbabilityFunction.SIGMOID:
                msg += f"\tInflection point: {self.inflection_point}\n"

            if self.config.costs == "from_data":
                msg += "\tCost-matrix inferred from geo-locations\n"
            else:
                msg += f"\tCost-matrix file: {self.config.costs}\n"

        if self.config.estimate_rate:
            msg += f"\tEstimating geo-prior rate: {self.config.approx_norm_const}\n"

        return msg

def average_max_distance(
    clusters: jnp.ndarray,      # shape: (n_objects, n_clusters)
    cost_matrix: jnp.ndarray,   # shape: (n_objects, n_objects)
) -> jnp.ndarray:
    """Compute the expected distance from each object to the farthest object in a cluster.

    For each object, the distance to the farthest cluster member is the cost at which
    the first unit of probability mass is reached when going from the most distant
    object inwards. These distances are then averaged over the objects of the cluster,
    weighted by cluster membership.

    Args:
        clusters: the fuzzy cluster assignments
        cost_matrix: the cost matrix between objects

    Returns:
        One expected maximum distance per cluster. shape: (n_clusters,)
    """
    n_objects, n_clusters = clusters.shape

    max_cost_order = jnp.argsort(cost_matrix, axis=-1, descending=True)
    sorted_cost = jnp.take_along_axis(cost_matrix, max_cost_order, axis=-1)

    max_distances = []
    for i in range(n_clusters):
        c = clusters[:, i]
        other_probs = jnp.repeat(c[None, :], n_objects, axis=0)
        sorted_probs = jnp.take_along_axis(other_probs, max_cost_order, axis=-1)
        max_cost_distr = first_k_continuous(sorted_probs, k=1, axis=-1)
        d = jnp.sum(max_cost_distr * sorted_cost, axis=-1)
        max_distances.append(jnp.dot(c, d))

    return jnp.array(max_distances)


def first_k_continuous(probs: jnp.ndarray, k: float, axis: int = -1) -> jnp.ndarray:
    """Clip the probabilities in `probs` to only contain the first `k` probability mass.

    Going along `axis`, probabilities are kept until their cumulative sum reaches `k`;
    the entry that crosses `k` is truncated and the rest are set to zero.

    Args:
        probs: the probabilities to clip
        k: the amount of probability mass to keep
        axis: the axis to accumulate along

    Returns:
        The clipped probabilities, summing to `min(k, probs.sum(axis))` along `axis`.
    """
    cum_probs = jnp.cumsum(probs, axis=axis)
    cum_probs_to_k = cum_probs.clip(0, k)
    return jnp.diff(cum_probs_to_k, prepend=0, axis=axis)
