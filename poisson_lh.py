# Likelihood for Poisson model

import numpy as np
from numpy import log
from scipy.stats import gamma as gamma_distribution, poisson, nbinom
from scipy.special import factorial
from scipy.special import gamma as gamma_function, loggamma
from scipy.special import logsumexp
from scipy.stats import nbinom


def lh_poisson_lambda_marginalised(x: np.array, alpha_0: float, beta_0: float, log_lh: bool = True) -> float:
    """
    Computes the marginal (log)-likelihood for a Poisson model with the rate (lambda) marginalised out where the
    prior on lambda follows a gamma distribution: gamma(alpha_0, beta_0).
    :param x: the data, measurements following a Poisson distribution
    :param alpha_0: shape of the gamma prior on lambda
    :param beta_0: rate of the gamma prior on lambda
    :param log_lh: Compute the marginal log likelihood (True) or marginal likelihood (False)?
    :return: the marginal (log)-likelihood of the data
    """
    n = len(x)
    x_bar = x.mean()

    if log_lh:
        return alpha_0 * log(beta_0) - loggamma(x + 1).sum() - loggamma(alpha_0) + \
             loggamma(n * x_bar + alpha_0) - log(n + beta_0) * (n * x_bar + alpha_0)
    else:
        return beta_0 ** alpha_0 / factorial(x, exact=True).prod() / gamma_function(alpha_0) * \
               gamma_function(n * x_bar + alpha_0) / (n + beta_0) ** (n * x_bar + alpha_0)


# Test analytical marginal likelihood against grid approximation

# Grid approximate the marginal likelihood, rate (lambda) marginalized out
start_grid, end_grid, grid_size = 0.001, 100, 1000000
lambda_grid = np.linspace(start_grid, end_grid, grid_size)
delta = (end_grid - start_grid) / grid_size


# Gamma prior on the rate (lambda)
alpha_0 = 3
beta_0 = 2

prior_lambda = gamma_distribution.pdf(lambda_grid, alpha_0, scale=1/beta_0)
log_prior_lambda = gamma_distribution.logpdf(lambda_grid, alpha_0, scale=1/beta_0)


# Data
d = np.array([4, 5])

# Likelihood
lh = poisson.pmf(d[:, None], lambda_grid[None, :])
log_lh = poisson.logpmf(d[:, None], lambda_grid[None, :])

# Grid approximation
marginal_lh = sum(lh.prod(axis=0) * prior_lambda * delta)
marginal_log_lh = logsumexp(log_lh.sum(axis=0) + log_prior_lambda + np.log(delta))

# Analytical likelihood
marginal_lh_analytical = lh_poisson_lambda_marginalised(x=d, alpha_0=alpha_0, beta_0=beta_0, log_lh=False)
marginal_log_lh_analytical = lh_poisson_lambda_marginalised(x=d, alpha_0=alpha_0, beta_0=beta_0, log_lh=True)

print(marginal_lh, marginal_lh_analytical, "likelihood")
print(marginal_log_lh, marginal_log_lh_analytical, "log likelihood")


# Conditional likelihood for observing a new data point given the data in the cluster and the prior

old_data_point = 4
new_data_point = 5

alpha_1 = alpha_0 + old_data_point
beta_1 = beta_0 + 1

# beta_1 = beta_0 + len(d)
# alpha_1 = alpha_0 + d.sum()

lh1 = nbinom.logpmf(old_data_point, alpha_0, beta_0/(1+beta_0))
lh2 = nbinom.logpmf(new_data_point, alpha_1, beta_1/(1+beta_1))

print(lh1 + lh2, "Negative binomial")

data_in_cluster = np.array([3, 4, 3, 4])
alpha_1 = alpha_0 + data_in_cluster.sum()
beta_1 = beta_0 + len(data_in_cluster)

new_data_point = 4
conditional_lh_nb = nbinom.pmf(new_data_point, alpha_1, beta_1/(1+beta_1))
print(conditional_lh_nb, "Conditional lh")

