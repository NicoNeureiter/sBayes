import math

import numpy as np

from sbayes.sampling.state import Sample
from sbayes.util import get_best_permutation


def compute_dic(lh, burn_in):
    """This function computes the deviance information criterion
    (see for example Celeux et al. 2006) using the posterior mode as a point estimate
    Args:
        lh: (dict): log-likelihood of samples in hte posterior
        burn_in(float): percentage of samples, which are discarded as burn-in
        """
    end_bi = math.ceil(len(lh) * burn_in)
    lh = lh[end_bi:]

    # max(ll) likelihood evaluated at the posterior mode
    d_phi_pm = -2 * np.max(lh)
    mean_d_phi = -4 * (np.mean(lh))

    dic = mean_d_phi + d_phi_pm

    return dic


def match_clusters(samples):
    """Align clusters from (possibly) different chains.
    Args:
        samples (dict): samples from the MCMC
    Returns:
        matched_samples(list): Resulting matching.
    """
    cluster_samples = np.array([np.array(s) for s in samples['sample_clusters']])
    n_samples, n_clusters, n_sites = cluster_samples.shape
    s_sum = np.zeros((n_clusters, n_sites))

    # All potential permutations of cluster labels
    matching_list = []
    for s in cluster_samples:
        permutation = get_best_permutation(s, s_sum)
        s_sum += s[permutation, :]

    # Reorder chains according to matching
    reordered_clusters = []
    reordered_cluster_effect = []
    reordered_lh = []
    reordered_prior = []
    reordered_posterior = []

    print("Matching cluster ...")
    for s in range(len(samples['sample_clusters'])):
        reordered_clusters.append(samples['sample_clusters'][s][:][matching_list[s]])
    samples['sample_clusters'] = reordered_clusters

    print("Matching cluster effect ...")
    for s in range(len(samples['sample_cluster_effect'])):
        reordered_cluster_effect.append(samples['sample_cluster_effect'][s][matching_list[s]])
    samples['sample_cluster_effect'] = reordered_cluster_effect

    print("Matching cluster lh ...")
    for s in range(len(samples['sample_lh_single_cluster'])):
        reordered_lh.append([samples['sample_lh_single_cluster'][s][i] for i in matching_list[s]])
    samples['sample_lh_single_cluster'] = reordered_lh

    print("Matching cluster prior ...")
    for s in range(len(samples['sample_prior_single_cluster'])):
        reordered_prior.append([samples['sample_prior_single_cluster'][s][i] for i in matching_list[s]])
    samples['sample_prior_single_cluster'] = reordered_prior

    print("Matching cluster posterior...")
    for s in range(len(samples['sample_posterior_single_cluster'])):
        reordered_posterior.append([samples['sample_posterior_single_cluster'][s][i] for i in matching_list[s]])
    samples['sample_posterior_single_cluster'] = reordered_posterior

    return samples


# TODO: update to generalized sBayes
def contribution_per_cluster(mcmc_sampler):
    """Evaluate the contribution of each cluster to the lh and the posterior in each sample.
    Args:
        mcmc_sampler (MCMC_generative): MCMC sampler for generative model (including samples)
    Returns:
        MCMC_generative: MCMC sampler including statistics on the likelihood and prior per cluster
    """
    stats = mcmc_sampler.statistics
    stats['sample_lh_single_cluster'] = []
    stats['sample_prior_single_cluster'] = []
    stats['sample_posterior_single_cluster'] = []

    # Iterate over all samples
    n_samples = len(stats['sample_clusters'])

    # todo: finish adapting this function

    for s in range(n_samples):
        weights = stats['sample_weights'][s]
        p_global = stats['sample_p_global'][s]
        p_families = stats['sample_p_families'][s]

        log_lh = []
        log_prior = []
        log_posterior = []

        for z in range(len(stats['sample_clusters'][s])):
            cluster = stats['sample_clusters'][s][np.newaxis, z]
            cluster_effect = stats['sample_cluster_effect'][s][np.newaxis, z]

            single_cluster = Sample.from_numpy_arrays(
                clusters=cluster,
                weights=weights,
                p_global=p_global,
                cluster_effect=cluster_effect, p_families=p_families
            )

            lh = mcmc_sampler.likelihood(single_cluster, 0)
            prior = mcmc_sampler.prior(single_cluster, 0)

            log_lh.append(lh)
            log_prior.append(prior)
            log_posterior.append(lh+prior)

        # Save stats about single clusters
        stats['sample_lh_single_cluster'].append(log_lh)
        stats['sample_prior_single_cluster'].append(log_prior)
        stats['sample_posterior_single_cluster'].append(log_posterior)

    mcmc_sampler.statistics = stats


def rank_clusters(samples):
    """ Rank the contribution of each cluster to the posterior
        Args:
            samples (dict): samples from the MCMC

        Returns:
            dict: the ordered samples

            """

    post_per_cluster = np.asarray(samples['sample_posterior_single_cluster'])
    to_rank = np.mean(post_per_cluster, axis=0)
    ranked = np.argsort(-to_rank)

    # probability per area in log-space
    # p_total = logsumexp(to_rank)
    # p = to_rank[np.argsort(-to_rank)] - p_total

    ranked_clusters = []
    ranked_lh = []
    ranked_prior = []
    ranked_posterior = []
    ranked_cluster_effect = []

    print("Ranking clusters ...")
    for s in range(len(samples['sample_clusters'])):
        ranked_clusters.append(samples['sample_clusters'][s][ranked])
    samples['sample_clusters'] = ranked_clusters

    print("Ranking lh clusters ...")
    for s in range(len(samples['sample_lh_single_cluster'])):
        ranked_lh.append([samples['sample_lh_single_cluster'][s][r] for r in ranked])
    samples['sample_lh_single_cluster'] = ranked_lh

    print("Ranking prior clusters ...")
    for s in range(len(samples['sample_prior_single_cluster'])):
        ranked_prior.append([samples['sample_prior_single_cluster'][s][r] for r in ranked])
    samples['sample_prior_single_cluster'] = ranked_prior

    print("Ranking posterior clusters ...")
    for s in range(len(samples['sample_posterior_single_cluster'])):
        ranked_posterior.append([samples['sample_posterior_single_cluster'][s][r] for r in ranked])
    samples['sample_posterior_single_cluster'] = ranked_posterior

    print("Ranking cluster effect ...")
    for s in range(len(samples['sample_cluster_effect'])):
        ranked_cluster_effect.append(samples['sample_cluster_effect'][s][ranked])
    samples['sample_cluster_effect'] = ranked_cluster_effect

    return samples
