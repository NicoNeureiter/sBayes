from src.util import load_from
from src.config import *
from src.sampling.zone_sampling import ZoneMCMC
import numpy as np
import math


def compute_model_quality(results, mode):
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

        # MISSING: Perform zone sampling at a fixed temperature (ask Nico)
        # MISSING ASSIGN NUMBER OF SAMPLES TO MCMC and MODE (PM or GM)

        sampler = ZoneMCMC(network=model['network'], features=model['features'], n_steps=model['n_steps'],
                           min_size=model['min_size'], max_size=model['max_size'], p_transition_mode=['p_transition_mode'],
                           geo_weight=model['geo_weight']/model['features'].shape[1],
                           lh_lookup=model['lh_lookup'], n_zones=model['n_zones'],
                           ecdf_geo=model['ecdf_geo'], restart_chain=model['restart_chain'],
                           simulated_annealing=False, plot_samples=False)

        sampler.generate_samples(samples)
        lh.append(sampler.statistics['sample_likelihoods'])
        temp.append(t)

    # Perform Stepping stone sampling
    if mode == 'stepping stones':

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

def stepping_stone_sampler(lh, temp):
    """ This function estimates the marginal likelihood of a model using the stepping stone sampler

    Args:
        lh (list): a list of all the likelihood values for a specific temperature
        temp (): a list of all temperature values

    Returns:
        float: the marginal likelihood of the model
    """
    n = len(temp)
    res = 0
    for i in range(n-1):
        temp_diff = temp[i+1] - temp[i]
        max_ll = np.max(lh[i+1])
        old_ll =  lh[i]

        res += temp_diff*max_ll
        res += np.log((1 / len(old_ll)) * sum(np.exp(temp_diff * (old_ll - max_ll))))
    return res

def compute_bayes_factor(m_lh_1, m_lh_2):
    """ This function computes the Bayes' factor between two models.

    Args:
        m_lh_1 (float): the marginal likelihood of model 1
        m_lh_2 (float): the marginal likelihood of model 2

    Returns:
        float: the Bayes' factor of model 2 compared to model 1
    """
    log_bf = m_lh_2 - m_lh_1
    return np.exp(log_bf)


mcmc_res = load_from(MCMC_RESULTS_PATH)

aic = compute_model_quality(mcmc_res, 'AIC')
bic = compute_model_quality(mcmc_res, 'BIC')
mle = compute_model_quality(mcmc_res, 'MLE')

print(aic, bic, mle)
