# Likelihood for normal model

import numpy as np
from numpy import log, sqrt, pi, exp
from scipy.stats import norm
from scipy.special import logsumexp, logit


def lh_normal_mu_marginalised(x: np.array, sigma_fixed: float, mu_0: float,
                              sigma_0: float, log_lh: bool = True) -> float:
    """
    Computes the marginal likelihood for a normal model with the mean (mu) marginalised out and standard deviation
    (sigma) fixed where the prior on mu is a normal distribution with N(mu_0, sigma_0**2).
    :param x: the data, measurements following a normal distribution
    :param mu_0: mean of the prior on mu
    :param sigma_0: standard deviation of the prior on mu
    :param sigma_fixed: known standard deviation of the normal distribution
    :param log_lh: Compute the marginal log likelihood (True) or the marginal likelihood (False)?
    :return: the marginal (log)-likelihood of the data
    """
    n = len(x)
    x_bar = x.mean()
    x_2_bar = (x**2).mean()

    if log_lh:

        loga = -log(sigma_0) - 1 / 2 * log(2 * pi) + - n * (log(sigma_fixed) + 1 / 2 * log(2 * pi))
        logb = (-mu_0 ** 2 / (2 * sigma_0 ** 2) - n * x_2_bar / (2 * sigma_fixed ** 2))
        c = (sigma_fixed ** 2 + sigma_0 ** 2 * n) / (2 * sigma_0 ** 2 * sigma_fixed ** 2)
        f = (mu_0 * sigma_fixed ** 2 + n * x_bar * sigma_0 ** 2) / (sigma_0 ** 2 * sigma_fixed ** 2)

        return loga + logb + 1/2 * log(pi) - log(sqrt(c)) + (f ** 2 / (4 * c))

    else:

        a = (1 / (sigma_0 * sqrt(2 * pi))) * (1 / (sigma_fixed * sqrt(2 * pi))) ** n
        b = exp(-mu_0 ** 2 / (2 * sigma_0 ** 2) - n * x_2_bar / (2 * sigma_fixed ** 2))
        c = (sigma_fixed ** 2 + sigma_0 ** 2 * n) / (2 * sigma_0 ** 2 * sigma_fixed ** 2)
        f = (mu_0 * sigma_fixed ** 2 + n * x_bar * sigma_0 ** 2) / (sigma_0 ** 2 * sigma_fixed ** 2)

        return a * b * sqrt(pi) / sqrt(c) * exp(f ** 2 / (4 * c))


# Test analytical marginal likelihood against grid approximation

# Grid approximate the marginal likelihood, mean (mu) marginalized out, variance (sigma) fixed
start_grid, end_grid, grid_size = -20, 20, 100000
mu_grid = np.linspace(start_grid, end_grid, grid_size)
delta = (end_grid - start_grid) / grid_size

# Prior on the mean
mu_0 = 3
sigma_0 = 4
prior_mu = norm.pdf(mu_grid, mu_0, sigma_0)
log_prior_mu = norm.logpdf(mu_grid, mu_0, sigma_0)

# Data
d = np.array([3.56, 14, 3.57])

# Maximum likelihood estimate for sigma
sigma = d.std()

# (log)-Likelihood
lh = norm.pdf(d[:, None], mu_grid[None, :], sigma)
log_lh = norm.logpdf(d[:, None], mu_grid[None, :], sigma)

# Grid approximation
marginal_lh = sum(lh.prod(axis=0) * prior_mu * delta)
marginal_log_lh = logsumexp((log_lh.sum(axis=0) + log_prior_mu) + log(delta))

# Analytical likelihood
marginal_lh_analytical = lh_normal_mu_marginalised(x=d, sigma_fixed=sigma,
                                                   mu_0=mu_0, sigma_0=sigma_0, log_lh=False)

marginal_log_lh_analytical = lh_normal_mu_marginalised(x=d, sigma_fixed=sigma,
                                                       mu_0=mu_0, sigma_0=sigma_0, log_lh=True)

# print(marginal_lh, marginal_lh_analytical)
print(marginal_log_lh, marginal_log_lh_analytical)


# With logit-transformed data
d = np.array([0.9, 0.6, 0.91, 0.9, 0.6])
sigma = 0.2


marginal_log_lh_analytical_logit = lh_normal_mu_marginalised(x=logit(d), sigma_fixed=sigma,
                                                             mu_0=logit(0.9), sigma_0=2, log_lh=True)

print(marginal_log_lh_analytical_logit, "logit")
