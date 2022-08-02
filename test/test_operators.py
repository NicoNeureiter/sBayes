#!/usr/bin/env python3
from __future__ import annotations

import unittest
import random
import math
from copy import deepcopy
from abc import abstractmethod, ABC
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
import scipy.stats as stats
from scipy.stats import kstest

from sbayes.model import Model
from sbayes.sampling.operators import Operator, AlterCluster
from sbayes.sampling.state import Sample, Clusters

Value = TypeVar("Value")


class DummySample(Sample):
    """This dummy subclass of Sample allows initializing without setting all parameters."""

    chain = 0

    def __init__(self):
        pass

    def copy(self):
        return deepcopy(self)


class DummyModel(Model):

    def __init__(self, max_size):
        self.min_size = 0
        self.max_size = max_size


class AbstractOperatorTest(ABC, Generic[Value]):

    """Test whether repeated application of an MCMC operator reaches the desired
    stationary distribution."""

    @abstractmethod
    def get_operator(self) -> Operator:
        ...

    @abstractmethod
    def set_value_in_sample(self, value: Value, sample: Sample):
        ...

    @abstractmethod
    def get_value_from_sample(self, value: Value) -> Sample:
        ...

    @abstractmethod
    def generate_initial_value(self) -> Value:
        ...

    @abstractmethod
    def sample_stationary_distribution(self, n_samples: int) -> list[Value]:
        ...

    # @abstractmethod
    # def get_stationary_distribution_cdf(self) -> callable:
    #     ...
    #
    # @abstractmethod
    # def iterate_comparable_statistics(self, samples: list[Value]) -> Iterator[list[float]]:
    #     ...

    @abstractmethod
    def compare_sampled_values(self, mcmc: list[Value], exact: list[Value]):
        ...

    def test_operator(self):

        n_samples = 100
        n_steps_per_sample = 30

        mcmc_samples = []
        for i_sample in range(n_samples):
            # Create a dummy sample object (required for sBayes operators to work)
            sample = DummySample()

            # Generate an intial value and set it in the Sample object
            v = self.generate_initial_value()
            self.set_value_in_sample(v, sample)

            # Repeatedly apply the operator
            operator = self.get_operator()
            for i_step in range(n_steps_per_sample):
                sample = self.mcmc_step(sample, operator)

            # Remember the final value
            mcmc_samples.append(self.get_value_from_sample(sample))

        exact_samples = self.sample_stationary_distribution(n_samples)

        self.compare_sampled_values(mcmc_samples, exact_samples)

        # for mcmc_sample_stats, exact_sample_stats in zip(
        #     self.iterate_comparable_statistics(mcmc_samples),
        #     self.iterate_comparable_statistics(exact_samples),
        # ):
        #     # plt.hist([mcmc_sample_stats, exact_sample_stats], bins=20, density=True)
        #     # t = np.arange(0, 11)
        #     # plt.plot(t, 2*stats.binom(10, 0.5).pmf(t))
        #     # plt.show()
        #
        #     ks_stat, p_value = kstest(mcmc_sample_stats, exact_sample_stats)
        #     print(p_value, stats.binom_test(sum(mcmc_sample_stats), n_samples, 0.5))
        #     # print(kstest(mcmc_sample_stats, cdf))
        #     # print(kstest(exact_sample_stats, cdf))
        #     assert p_value > 0.01

    @staticmethod
    def mcmc_step(sample: Sample, operator: Operator) -> Sample:
        new_sample, log_q, log_q_back = operator.function(sample)
        p_accept = math.exp(log_q_back - log_q)
        if random.random() < p_accept:
            return new_sample
        else:
            return sample


class ClusterOperatorTest(AbstractOperatorTest[NDArray[bool]], unittest.TestCase):

    N_OBJECTS = 30
    STATIONARY_DISTRIBUTION = stats.binom(N_OBJECTS, 0.5)

    def get_operator(self) -> Operator:
        return AlterCluster(
            weight=0.0,
            adjacency_matrix=np.ones((self.N_OBJECTS, self.N_OBJECTS), dtype=bool),
            p_grow_connected=0.8,
            model_by_chain={0: DummyModel(self.N_OBJECTS)},
            resample_source=False,
            sample_from_prior=False,
        )

    def set_value_in_sample(self, value: Value, sample: Sample):
        sample._clusters = Clusters(value)

    def get_value_from_sample(self, sample: Sample) -> Value:
        return sample.clusters.value

    def generate_initial_value(self) -> Value:
        return np.random.random((1, self.N_OBJECTS)) < 0.5

    def sample_stationary_distribution(self, n_samples: int) -> list[Value]:
        return [np.random.random((1, self.N_OBJECTS)) < 0.5 for _ in range(n_samples)]

    # def get_stat_extractors(self) -> Iterator[callable]:
    #     yield lambda value: value.sum()  # cluster size
    #     for i in range(self.N_OBJECTS):
    #         yield lambda value: value[0, i]
    #
    # def get_stationary_distribution_cdf(self) -> callable:
    #     return stats.binom(self.N_OBJECTS, 0.5).cdf
    #
    # def iterate_comparable_statistics(
    #     self, samples: list[Value]
    # ) -> Iterator[list[float]]:
    #     for extractor in self.get_stat_extractors():
    #         yield [extractor(s) for s in samples]

    def compare_sampled_values(self, mcmc: list[Value], exact: list[Value]):
        n_samples = len(mcmc)
        mcmc = np.asarray(mcmc)  # shape = (n_samples, 1, n_objects)

        for i in range(self.N_OBJECTS):
            p_value_i = stats.binom_test(
                x=np.sum(mcmc[:, 0, i]),
                n=n_samples,
                p=0.5
            )
            assert p_value_i > 0.001

        p_value_flat = stats.binom_test(
            x=np.sum(mcmc),
            n=n_samples * self.N_OBJECTS,
            p=0.5
        )
        assert p_value_flat > 0.01, p_value_flat


if __name__ == "__main__":
    unittest.main()
