from itertools import permutations
import numpy as np
import math
from scipy.special import logsumexp
from sbayes.sampling.zone_sampling import ZoneMCMC, Sample


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


def compute_model_quality(mcmc_results, mode):
    """This function computes an estimator of the relative quality of a model ( either AIC, BIC or MLE)
    Args:
        mcmc_results (dict): the samples from the MCMC
        mode (char): either  'AIC' (Akaike Information Criterion),
                             'BIC' (Bayesian Information Criterion),
                             'MLE' (Maximum likelihood estimation)
    Returns:
        float: either the AIC, BIC or MLE
        """
    valid_modes = ['AIC', 'BIC', 'MLE']
    if mode not in valid_modes:
        raise ValueError("mode must be AIC, BIC or MLE")

    mle = max(mcmc_results['stats']['sample_likelihoods'])

    if mode == 'MLE':
        return mle
    k = mcmc_results['n_zones']

    if mode == 'AIC':
        aic = 2*k - 2 * math.log(mle)
        return aic

    elif mode == 'BIC':
        bic = math.log(mcmc_results['n_features']) * k - 2 * math.log(mle)
        return bic


def compute_marginal_likelihood(mcmc, samples, mode, n_temp=100):
    """ This function estimates the marginal likelihood of a model either using the power posterior approach or the
    stepping stone sample
    Args:
        mcmc(ZoneMCMC): a model, for which the marginal likelihood is computed, provided as a ZoneMCMC object
        samples(int): the number of samples generated per temperature
        mode (char):  either "stepping stones" or "power posterior"
        n_temp (int): the number of temperatures for which samples are generated
    Returns:
        float: the marginal likelihood of the model
    """
    lh = temp = []

    # Iterate over temperatures
    for t in np.nditer(np.linspace(0, 1, n_temp)):
        # TODO if this is used again: adapt to ZoneMCMC changes
        sampler = ZoneMCMC(
            network=mcmc['network'], features=mcmc['features'], n_steps=mcmc['n_steps'],
            min_size=mcmc['min_size'], max_size=mcmc['max_size'], p_transition_mode=['p_transition_mode'],
            geo_weight=mcmc['geo_weight'] / mcmc['features'].shape[1],
            lh_lookup=mcmc['lh_lookup'], n_zones=mcmc['n_zones'],
            ecdf_geo=mcmc['ecdf_geo'], restart_interval=mcmc['restart_interval'],
            simulated_annealing=False, plot_samples=False)

        sampler.generate_samples(samples)
        lh.append(sampler.statistics['sample_likelihoods'])
        temp.append(t)

    # Perform Stepping stone sampling
    if mode == 'stepping stones':
        raise NotImplementedError

    # Perform Power Posterior Sampling
    elif mode == "power posterior":
        return power_posterior(lh, temp)

    else:
        raise ValueError("mode must be `power posterior` or `stepping stones`")


def power_posterior (lh, temp):
    """ This function estimates the marginal likelihood of a model using the
    power posterior method
    Args:
        lh (list): A list of all the likelihood values for a specific temperature
        temp (list): A list of all temperature values
    Returns:
        (float): The marginal likelihood of the model
    """
    n = len(temp)
    res = 0

    for i in range(n-1):
        w_tz = temp[i+1] - temp[i]
        h_tz = (np.mean(lh[i+1]) + np.mean(lh[i])) / 2
        res += w_tz * h_tz
    return res


def stepping_stone_sampler(samples):
    """ This function estimates the marginal likelihood of a model using the stepping stone sampler
    Args:
        samples(dict): samples returned from the marginal lh sampler
    Returns:
        float: the marginal likelihood of the model
    """
    n = len(samples)
    temp = []
    lh = []
    for t in sorted(samples):
        temp.append(t)
        lh.append(samples[t][0]['sample_likelihoods'][0])

    res = 0
    for i in range(n-1):
        temp_diff = temp[i+1] - temp[i]
        max_ll = np.max(lh[i+1])
        old_ll = lh[i]

        res += temp_diff*max_ll
        res += np.log((1 / len(old_ll)) * sum(np.exp(temp_diff * (old_ll - max_ll))))
    return res


def compute_bayes_factor(m_lh_1, m_lh_2):
    """ This function computes the Bayes' factor between two models.
    Args:
        m_lh_1 (list): the marginal likelihood of model 1 (for different zone sizes)
        m_lh_2 (list): the marginal likelihood of model 2 (for different zone sizes)
    Returns:
        float: the Bayes' factor of model 2 compared to model 1
    """
    # Compute geometric mean
    m1 = sum(m_lh_1) / len(m_lh_1)
    m2 = sum(m_lh_2) / len(m_lh_2)

    print('m1:',  m1, 'm2:', m2)
    log_bf = m1 - m2
    return np.exp(log_bf)


def samples_list_to_array(samples_in, samples_type="zone"):
    """ Helper function to change the data structure
    Args:
        samples_in (list): list of zone assignments as returned by the MCMC
        samples_type (str): type of the input, either "zone", "lh" or "posterior"
    Returns
        samples_out (ndarray): samples, as needed for matching
         """
    if samples_type == "zone":
        samples_1 = np.array([np.array(s) for s in samples_in])

        # Swap axes (twice)
        samples_2 = np.swapaxes(samples_1, 0, 1)
        samples_out = np.swapaxes(samples_2, 1, 2)

    elif samples_type in ["lh", "posterior"]:
        samples_1 = np.array([np.array(s) for s in samples_in])
        samples_out = np.swapaxes(samples_1, 0, 1)
    else:
        raise ValueError('samples_type must be "zone" or "lh" or "posterior')
    return samples_out


def samples_array_to_list(samples_in, samples_type):
    """ Helper function to change the data structure from samples
    Args:
        samples_in (list): Samples, as returned by the matching function
        samples_type (str): type of the input, either "zone", "lh", or "posterior'
    Returns
        chains_out (ndaray): Chains with samples
         """
    if samples_type == "zone":
        samples_np = np.array([np.array(s) for s in samples_in])

        # Swap axes
        samples_out = np.swapaxes(samples_np, 1, 0)

    elif samples_type in ["lh", "posterior"]:
        samples_np = np.array([np.array(s) for s in samples_in])

        # Swap axes
        samples_out = np.swapaxes(samples_np, 1, 0)

    else:
        raise ValueError('samples_type must be "zone" or "lh" or "posterior')
    return samples_out


def match_areas(samples):
    """Align clusters from (possibly) different chains.
    Args:
        samples (dict): samples from the MCMC
    Returns:
        matched_samples(list): Resulting matching.
    """

    cluster_samples = np.array([np.array(s) for s in samples['sample_clusters']])
    cluster_samples = np.swapaxes(cluster_samples, 1, 2)

    n_samples, n_sites, n_clusters = cluster_samples.shape
    s_sum = np.zeros((n_sites, n_clusters))

    # All potential permutations of cluster labels
    perm = list(permutations(range(n_clusters)))
    matching_list = []
    for s in cluster_samples:

        def clustering_agreement(p):

            """In how many sites does the permutation 'p'
            match the previous sample?
            """

            return np.sum(s_sum * s[:, p])

        best_match = max(perm, key=clustering_agreement)
        matching_list.append(list(best_match))
        s_sum += s[:, best_match]

    # Reorder chains according to matching
    reordered_clusters = []
    reordered_areal_effect = []
    reordered_lh = []
    reordered_prior = []
    reordered_posterior = []

    print("Matching areas ...")
    for s in range(len(samples['sample_zones'])):
        reordered_zones.append(samples['sample_zones'][s][:][matching_list[s]])
    samples['sample_zones'] = reordered_zones

    print("Matching areal effect ...")
    for s in range(len(samples['sample_p_zones'])):
        reordered_p_zones.append(samples['sample_p_zones'][s][matching_list[s]])
    samples['sample_p_zones'] = reordered_p_zones

    print("Matching areal lh ...")
    for s in range(len(samples['sample_lh_single_zones'])):
        reordered_lh.append([samples['sample_lh_single_zones'][s][i] for i in matching_list[s]])
    samples['sample_lh_single_zones'] = reordered_lh

    print("Matching areal prior ...")
    for s in range(len(samples['sample_prior_single_zones'])):
        reordered_prior.append([samples['sample_prior_single_zones'][s][i] for i in matching_list[s]])
    samples['sample_prior_single_zones'] = reordered_prior

    print("Matching areal posterior...")
    for s in range(len(samples['sample_posterior_single_zones'])):
        reordered_posterior.append([samples['sample_posterior_single_zones'][s][i] for i in matching_list[s]])
    samples['sample_posterior_single_zones'] = reordered_posterior

    return samples


def contribution_per_cluster(mcmc_sampler):
    """Evaluate the contribution of each cluster to the lh and the posterior in each sample.
    Args:
        mcmc_sampler (MCMC_generative): MCMC sampler for generative model (including samples)
    Returns:
        MCMC_generative: MCMC sampler including statistics on the likelihood and prior per zone
    """
    stats = mcmc_sampler.statistics
    stats['sample_lh_single_clusters'] = []
    stats['sample_prior_single_clusters'] = []
    stats['sample_posterior_single_clusters'] = []

    # Iterate over all samples
    n_samples = len(stats['sample_clusters'])
    # todo: stop
    for s in range(n_samples):
        weights = stats['sample_weights'][s]
        p_global = stats['sample_p_global'][s]
        p_families = stats['sample_p_families'][s]

        log_lh = []
        log_prior = []
        log_posterior = []

        for z in range(len(stats['sample_zones'][s])):
            zone = stats['sample_zones'][s][np.newaxis, z]
            p_zone = stats['sample_p_zones'][s][np.newaxis, z]

            single_zone = Sample(zones=zone, weights=weights,
                                 p_global=p_global, p_zones=p_zone, p_families=p_families)

            lh = mcmc_sampler.likelihood(single_zone, 0)
            prior = mcmc_sampler.prior(single_zone, 0)

            log_lh.append(lh)
            log_prior.append(prior)
            log_posterior.append(lh+prior)

        # Save stats about single zones
        stats['sample_lh_single_zones'].append(log_lh)
        stats['sample_prior_single_zones'].append(log_prior)
        stats['sample_posterior_single_zones'].append(log_posterior)

    mcmc_sampler.statistics = stats


def rank_areas(samples):
    """ Rank the contribution of each area to the posterior
        Args:
            samples (dict): samples from the MCMC

        Returns:
            dict: the ordered samples

            """

    post_per_area = np.asarray(samples['sample_posterior_single_zones'])
    to_rank = np.mean(post_per_area, axis=0)
    ranked = np.argsort(-to_rank)

    # probability per area in log-space
    # p_total = logsumexp(to_rank)
    # p = to_rank[np.argsort(-to_rank)] - p_total

    ranked_areas = []
    ranked_lh = []
    ranked_prior = []
    ranked_posterior = []
    ranked_p_areas = []

    print("Ranking areas ...")
    for s in range(len(samples['sample_zones'])):
        ranked_areas.append(samples['sample_zones'][s][ranked])
    samples['sample_zones'] = ranked_areas

    print("Ranking lh areas ...")
    for s in range(len(samples['sample_lh_single_zones'])):
        ranked_lh.append([samples['sample_lh_single_zones'][s][r] for r in ranked])
    samples['sample_lh_single_zones'] = ranked_lh

    print("Ranking prior areas ...")
    for s in range(len(samples['sample_prior_single_zones'])):
        ranked_prior.append([samples['sample_prior_single_zones'][s][r] for r in ranked])
    samples['sample_prior_single_zones'] = ranked_prior

    print("Ranking posterior areas ...")
    for s in range(len(samples['sample_posterior_single_zones'])):
        ranked_posterior.append([samples['sample_posterior_single_zones'][s][r] for r in ranked])
    samples['sample_posterior_single_zones'] = ranked_posterior

    print("Ranking p areas ...")
    for s in range(len(samples['sample_p_zones'])):
        ranked_p_areas.append(samples['sample_p_zones'][s][ranked])
    samples['sample_p_zones'] = ranked_p_areas

    return samples


# # deprecated
# def apply_matching(samples_in, matching):
#     """iterates through a list of samples and reorders the chains in each sample according to the matching schema
#         Args:
#             samples_in (list): samples that need to be reordered
#             matching (list): matching according to which the samples are ordered
#         Returns:
#             list: samples, reordered according to matching
#         """
#
#     # Change data structure to perform the reordering
#     samples = chains_to_samples(samples_in)
#     print('After chains_to_samples', samples.shape)
#     # Reorder chains according to matching
#     reordered = []
#     for s in range(len(samples)):
#
#         if not len(samples) == len(matching):
#             raise ValueError("number of samples and number of matches differ")
#
#         reordered.append(samples[s][matching[s]])
#
#     return samples_to_chains(reordered)


def unnest_marginal_lh(samples_in):
    """Un-nest marginal likelihood samples
    Args:
        samples_in: samples to un-nest
    Returns:
        list: un-nested samples
    """
    # Some un-nesting is necessary
    samples_out = []

    for s in samples_in:
        if len(s) == 1:

            samples_out.append(s[0])

        else:
            raise ValueError("Parallel zones detected! Rerun without parallel zones.")

    return samples_out


