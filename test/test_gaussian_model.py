#!/usr/bin/env python3
import unittest

import numpy as np
import numpy.testing
from numpy.typing import NDArray
from scipy.stats import norm as gaussian
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from sbayes.util import log_multinom, gaussian_mu_marginalised_logpdf, gaussian_mu_posterior_logpdf, \
    gaussian_posterior_predictive_logpdf, normalize


class TestGaussian(unittest.TestCase):

    """Test marginal distributions in a hierarchical Gaussian model."""

    def setUp(self):
        """Generate samples from the hierarchical model."""

        self.M = np.array([3, 2])
        self.S = np.array([4, 2])
        self.SIGMA = np.array([1.2, 0.1])
        self.X = np.array([[3.56, 14.1, 3.82],
                           [2.21, 1.37, 5.92]]).T

    @staticmethod
    def approximate_marginal_lh(
        data: NDArray[float],   # (n_objects, n_features)
        sigma: NDArray[float],  # (n_features,)
        m0: NDArray[float],     # (n_features,)
        s0: NDArray[float],     # (n_features,)
        grid_size: int = 10_000
    ) -> NDArray[float]:
        start_grid, end_grid = -20, 20
        mu = np.linspace(start_grid, end_grid, grid_size)
        delta = (end_grid - start_grid) / grid_size

        log_prior_mu = gaussian.logpdf(mu[None, :], m0[:, None], s0[:, None])
        # shape: (n_features, grid_size)

        log_lh = gaussian.logpdf(data[:, :, None], mu[None, None, :], sigma[None, :, None])
        # shape: (n_objects, n_features, grid_size)

        log_posterior = (log_lh.sum(axis=0) + log_prior_mu)
        # shape: (n_features, grid_size)

        # Grid approximation
        return logsumexp(log_posterior + np.log(delta), axis=1)

    def approximate_posterior_predictive_logpdf(
        self,
        data: NDArray[float],   # (n_objects, n_features)
        mu: NDArray[float],     # (n_features,)
        sigma: NDArray[float],  # (n_features,)
        m0: NDArray[float],     # (n_features,)
        s0: NDArray[float],     # (n_features,)
        grid_size: int = 10_000,
    ) -> NDArray[float]:        # (n_features,)
        pass

    @staticmethod
    def approximate_posterior_predictive(
        x_new: NDArray[float],  # (n_features,)
        data: NDArray[float],   # (n_objects, n_features)
        sigma: NDArray[float],  # (n_features,)
        m0: NDArray[float],     # (n_features,)
        s0: NDArray[float],     # (n_features,)
        grid_size: int = 10_000
    ) -> NDArray[float]:
        start_grid, end_grid = -20, 20
        mu = np.linspace(start_grid, end_grid, grid_size)
        delta = (end_grid - start_grid) / grid_size

        log_prior_mu = gaussian.logpdf(mu[None, :], m0[:, None], s0[:, None])
        # shape: (n_features, grid_size)

        log_lh = gaussian.logpdf(data[:, :, None], mu[None, None, :], sigma[None, :, None])
        # shape: (n_objects, n_features, grid_size)

        posterior = normalize(np.exp(log_lh.sum(axis=0) + log_prior_mu), axis=1)
        # shape: (n_features, grid_size)

        lh_new = gaussian.pdf(x=x_new[:, None], loc=mu[None, :], scale=sigma[:, None])
        # shape: (n_features, grid_size)

        # Grid approximation
        return np.log(np.sum(lh_new * posterior, axis=1))

    def test_marginal_lh(self):
        lh_analytical = gaussian_mu_marginalised_logpdf(
            x=self.X,
            sigma_fixed=self.SIGMA,
            mu_0=self.M,
            sigma_0=self.S,
            in_component=np.ones_like(self.X, dtype=bool),
        )
        lh_approx = self.approximate_marginal_lh(self.X, self.SIGMA, self.M, self.S)

        np.testing.assert_almost_equal(lh_analytical, lh_approx, decimal=4)

    def test_posterior(self):
        """Evaluate the posterior density of an arbitrary value for mu."""
        mu = np.array([4.72, 1.19])

        lh_analytical = gaussian_mu_posterior_logpdf(
            x=self.X,
            mu=mu,
            sigma=self.SIGMA,
            mu_0=self.M,
            sigma_0=self.S,
            in_component=np.ones_like(self.X, dtype=bool),
        )
        lh_analytical_unnormalized = (
            gaussian.logpdf(self.X, loc=mu, scale=self.SIGMA).sum(axis=0) +
            gaussian.logpdf(mu, loc=self.M, scale=self.S)
        )
        marginal_lh = gaussian_mu_marginalised_logpdf(
            x=self.X,
            sigma_fixed=self.SIGMA,
            mu_0=self.M,
            sigma_0=self.S,
            in_component=np.ones_like(self.X, dtype=bool),
        )
        lh_analytical_2 = lh_analytical_unnormalized - marginal_lh

        np.testing.assert_almost_equal(lh_analytical, lh_analytical_2)

    def test_posterior_predictive(self):
        pp_analytical_1 = []
        pp_approx_1 = []
        pp_analytical_2 = []
        pp_approx_2 = []
        x_grid = np.linspace(2, 9, 40)
        for x in x_grid:
            x_new = np.array([x, x])
            pp_analytical = gaussian_posterior_predictive_logpdf(
                x_new=x_new,
                x=self.X,
                sigma=self.SIGMA,
                mu_0=self.M,
                sigma_0=self.S,
                in_component=np.ones_like(self.X, dtype=bool),
            )
            pp_approx = self.approximate_posterior_predictive(x_new, self.X, self.SIGMA, self.M, self.S)

            pp_analytical_1.append(np.exp(pp_analytical[0]))
            pp_approx_1.append(np.exp(pp_approx[0]))
            pp_analytical_2.append(np.exp(pp_analytical[1]))
            pp_approx_2.append(np.exp(pp_approx[1]))

        np.testing.assert_almost_equal(pp_analytical_1, pp_approx_1, decimal=4)
        np.testing.assert_almost_equal(pp_analytical_2, pp_approx_2, decimal=4)

        # print()
        # print(x_grid[np.argmax(pp_approx_1)], x_grid[np.argmax(pp_approx_2)])
        # print(x_grid[np.argmax(pp_analytical_1)], x_grid[np.argmax(pp_analytical_2)])
        # plt.plot(x_grid, pp_analytical_1, label="pp_analytical_1")
        # plt.plot(x_grid, pp_approx_1, label="pp_approx_1", ls="--")
        # plt.plot(x_grid, pp_analytical_2, label="pp_analytical_2")
        # plt.plot(x_grid, pp_approx_2, label="pp_approx_2", ls="--")
        # plt.legend()
        # plt.show()


if __name__ == "__main__":
    unittest.main()
