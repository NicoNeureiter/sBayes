#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sbayes.util import log_multinom
from scipy.special import binom

log_binom = lambda n, k: np.log(binom(n, k))

class TestUtil(unittest.TestCase):

    def test_multinom_symmetry(self):
        for _ in range(100):
            n = 15
            k1 = np.random.randint(1, n-1)
            k2 = np.random.randint(1, n-k1)
            k3 = np.random.randint(1, n-k1-k2+1)

            x1 = np.exp(log_multinom(n, [k1, k2, k3]))
            x2 = np.exp(log_multinom(n, [k1, k3, k2]))
            x3 = np.exp(log_multinom(n, [k3, k2, k1]))

            np.testing.assert_almost_equal(x1, x2)
            np.testing.assert_almost_equal(x2, x3)

            # n0 = n
            # n1 = k1 + k2 + k3
            # n2 = k2 + k3
            # n3 = k3
            #
            # y = log_binom(n0, n1) + log_binom(n1, n2) + log_binom(n2, n3)
            # print(n1, n2, n3)
            # print(log_binom(n0, n1), log_binom(n1, n2), log_binom(n2, n3))
            # print(x1, y)
            #
            # np.testing.assert_almost_equal(x1, y)

    def test_multinom_vs_binom(self):
        n = 500
        for k in np.random.choice(n, 50, replace=False):
            log_bnm = log_binom(n, k)
            log_mnm = log_multinom(n, [k])
            log_mnm_0 = log_multinom(n, [k, 0])

            np.testing.assert_almost_equal(log_bnm, log_mnm)
            np.testing.assert_almost_equal(log_bnm, log_mnm_0)

    def test_multinom_vs_brutreforce(self):
        # Enumerate all element assignments + compare counts to log_multinom (for some small example)
        n = 5
        K = 3
        ...


if __name__ == '__main__':
    unittest.main()
