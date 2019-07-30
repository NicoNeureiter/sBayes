from itertools import permutations
import numpy as np
import math


def compute_dic(mcmc_res, burn_in):
    """This function computes the deviance information criterion
    (see for example Celeux et al. 2006) using the posterior mode as a point estimate

    Args:
        mcmc_res: (dict): the samples from the MCMC
        burn_in(float): percentage of samples, which are discarded as burn-in
        """
    end_bi = math.ceil(len(mcmc_res['lh']) * burn_in)
    ll = mcmc_res['lh'][end_bi:]

    # max(ll) likelihood evaluated at the posterior mode
    d_phi_pm = -2 * np.max(ll)
    mean_d_phi = -4 * (np.mean(ll))

    dic = mean_d_phi + d_phi_pm

    return dic


def compute_model_quality(mcmc_results, mode):
    """This function computes an estimator of the relative quality of a model ( either AIC, BIC or MLE)

    Args:
        results (dict): the samples from the MCMC
        mode (char): either  'AIC' (Akaike Information Criterion),
                             'BIC' (Bayesian Information Criterion),
                             'MLE' (Maximum likelihood estimation)
    Returns:
        float: either the AIC, BIC or MLE
        """
    valid_modes = ['AIC', 'BIC', 'MLE']
    if mode not in valid_modes:
        raise ValueError("mode must be AIC, BIC or MLE")

    mle = max(results['stats']['sample_likelihoods'])

    if mode == 'MLE':
        return mle
    k = results['n_zones']

    if mode == 'AIC':
        aic = 2*k - 2 * math.log(mle)
        return aic

    elif mode== 'BIC':
        bic = math.log(results['n_features'])*k - 2 * math.log(mle)
        return bic


def compute_marginal_likelihood(model, samples, mode, n_temp=100):

    """ This function estimates the marginal likelihood of a model either using the power posterior approach or the
    stepping stone sample
    Args:
        model(ZoneMCMC): a model, for which the marginal likelihood is computed, provided as a ZoneMCMC object
        samples(int): the number of samples generated per temperature
        mode (char):  either "stepping stones" or "power posterior"
        n_temp (int): the number of temperatures for which samples are generated

    Returns:
        float: the marginal likelihood of the model
    """
    lh = temp = []

    # Iterate over temperatures
    for t in np.nditer(np.linspace(0, 1, n_temp)):

        sampler = ZoneMCMC(network=model['network'], features=model['features'], n_steps=model['n_steps'],
                           min_size=model['min_size'], max_size=model['max_size'], p_transition_mode=['p_transition_mode'],
                           geo_weight=model['geo_weight']/model['features'].shape[1],
                           lh_lookup=model['lh_lookup'], n_zones=model['n_zones'],
                           ecdf_geo=model['ecdf_geo'], restart_interval=model['restart_interval'],
                           simulated_annealing=False, plot_samples=False)

        sampler.generate_samples(samples)
        lh.append(sampler.statistics['sample_likelihoods'])
        temp.append(t)

    # Perform Stepping stone sampling
    if mode == 'stepping stones':
        raise NotImplementedError

    # Perform Power Posterior Sampling
    elif mode == "power posterior":
        return power_posterior (lh, temp)

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


def samples_list_to_array(samples_in):
    """ Helper function to change the data structure

    Args:
        samples_in (list): list of zone assignments as returned by the MCMC

    Returns
        samples_out (ndarray): samples, as needed for matching
         """

    samples_1 = np.array([np.array(s) for s in samples_in])

    # Swap axes (twice)
    samples_2 = np.swapaxes(samples_1, 0, 1)
    samples_out = np.swapaxes(samples_2, 1, 2)

    return samples_out


def samples_array_to_list(samples_in):
    """ Helper function to change the data structure from samples

    Args:
        samples_in (list): Samples, as returned by the matching function

    Returns
        chains_out (ndaray): Chains with samples
         """

    samples_np = np.array([np.array(s) for s in samples_in])

    # Swap axes
    samples_out = np.swapaxes(samples_np, 1, 0)

    return samples_out


def match_zones(mcmc_res):
    """Align zones in (possibly) different chains.

    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict

    Returns:
        perm_list(list): Resulting matching.

    """
    # Change structure
    zone_samples = samples_list_to_array(mcmc_res['zones'])
    n_samples, n_sites, n_zones = zone_samples.shape

    s_sum = np.zeros((n_sites, n_zones))

    # All potential permutations of cluster labels
    perm = list(permutations(range(n_zones)))
    i = 1
    matching_list = []
    for s in zone_samples:
        i += 1
        print(i)

        def clustering_agreement(p):

            """In how many sites does the permutation 'p'
            match the previous sample?
            """
            return np.sum(s_sum * s[:, p])

        best_match = max(perm, key=clustering_agreement)
        matching_list.append(list(best_match))
        s_sum += s[:, best_match]
    # Reorder chains according to matching
    reordered = []

    zone_samples = np.swapaxes(zone_samples, 1, 2)

    for z in range(len(zone_samples)):

        reordered.append(zone_samples[z][:][matching_list[z]])

    mcmc_res['zones'] = samples_array_to_list(reordered)

    return mcmc_res


def apply_matching(samples_in, matching):
    """iterates through a list of samples and reorders the chains in each sample according to the matching schema

        Args:
            samples_in (list): samples that need to be reordered
            matching (list): matching according to which the samples are ordered
        Returns:
            list: samples, reordered according to matching

        """

    # Change data structure to perform the reordering
    samples = chains_to_samples(samples_in)
    print('After chains_to_samples', samples.shape)
    # Reorder chains according to matching
    reordered = []
    for s in range(len(samples)):

        if not len(samples) == len(matching):
            raise ValueError("number of samples and number of matches differ")

        reordered.append(samples[s][matching[s]])

    return samples_to_chains(reordered)


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
