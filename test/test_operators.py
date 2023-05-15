#!/usr/bin/env python3
from __future__ import annotations

import unittest
import random
import math
from copy import deepcopy
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
import scipy.stats as stats
from scipy.stats import kstest, binom_test
import matplotlib.pyplot as plt

from sbayes.cli import main as sbayes_main
from sbayes.experiment_setup import Experiment
from sbayes.mcmc_setup import MCMCSetup
from sbayes.model import Model
from sbayes.results import Results
from sbayes.sampling.counts import update_feature_counts, recalculate_feature_counts
from sbayes.sampling.operators import Operator, AlterCluster, AlterClusterGibbsish
from sbayes.sampling.state import Sample, Clusters
from sbayes.load_data import Data
from sbayes.util import normalize
from .test_model import (
    dummy_features_from_values,
    dummy_universal_confounder,
    dummy_objects,
)

Value = TypeVar("Value")

#
# class DummySample(Sample):
#     """This dummy subclass of Sample allows initializing without setting all parameters."""
#
#     chain = 0
#
#     def __init__(self):
#         pass
#
#     def copy(self):
#         return deepcopy(self)
#
#
# class DummyModel(Model):
#
#     def __init__(self, max_size):
#         self.min_size = 0
#         self.max_size = max_size
#
#     def likelihood(self, *args, **kwargs):
#         return 0.0
#
#     def prior(self, *args, **kwargs):
#         return 0.0
#
#
# class AbstractOperatorTest(ABC, Generic[Value]):
#
#     """Test whether repeated application of an MCMC operator reaches the desired
#     stationary distribution."""
#
#     @abstractmethod
#     def get_operator(self) -> Operator:
#         ...
#
#     @abstractmethod
#     def set_value_in_sample(self, value: Value, sample: Sample):
#         ...
#
#     @abstractmethod
#     def get_value_from_sample(self, value: Value) -> Sample:
#         ...
#
#     @abstractmethod
#     def generate_initial_value(self) -> Value:
#         ...
#
#     @abstractmethod
#     def sample_stationary_distribution(self, n_samples: int) -> list[Value]:
#         ...
#
#     # @abstractmethod
#     # def get_stationary_distribution_cdf(self) -> callable:
#     #     ...
#     #
#     # @abstractmethod
#     # def iterate_comparable_statistics(self, samples: list[Value]) -> Iterator[list[float]]:
#     #     ...
#
#     @abstractmethod
#     def compare_sampled_values(self, mcmc: list[Value], exact: list[Value]):
#         ...
#
#     def test_operator(self):
#
#         n_samples = 100
#         n_steps_per_sample = 30
#
#         mcmc_samples = []
#         for i_sample in range(n_samples):
#             # Create a dummy sample object (required for sBayes operators to work)
#             sample = DummySample()
#
#             # Generate an intial value and set it in the Sample object
#             v = self.generate_initial_value()
#             self.set_value_in_sample(v, sample)
#
#             # Repeatedly apply the operator
#             operator = self.get_operator()
#             for i_step in range(n_steps_per_sample):
#                 sample = self.mcmc_step(sample, operator)
#
#             # Remember the final value
#             mcmc_samples.append(self.get_value_from_sample(sample))
#
#         exact_samples = self.sample_stationary_distribution(n_samples)
#
#         self.compare_sampled_values(mcmc_samples, exact_samples)
#
#         # for mcmc_sample_stats, exact_sample_stats in zip(
#         #     self.iterate_comparable_statistics(mcmc_samples),
#         #     self.iterate_comparable_statistics(exact_samples),
#         # ):
#         #     # plt.hist([mcmc_sample_stats, exact_sample_stats], bins=20, density=True)
#         #     # t = np.arange(0, 11)
#         #     # plt.plot(t, 2*stats.binom(10, 0.5).pmf(t))
#         #     # plt.show()
#         #
#         #     ks_stat, p_value = kstest(mcmc_sample_stats, exact_sample_stats)
#         #     print(p_value, stats.binom_test(sum(mcmc_sample_stats), n_samples, 0.5))
#         #     # print(kstest(mcmc_sample_stats, cdf))
#         #     # print(kstest(exact_sample_stats, cdf))
#         #     assert p_value > 0.01
#
#     @staticmethod
#     def mcmc_step(sample: Sample, operator: Operator) -> Sample:
#         new_sample, log_q, log_q_back = operator.function(sample)
#         p_accept = math.exp(log_q_back - log_q)
#         if random.random() < p_accept:
#             return new_sample
#         else:
#             return sample
#
#
# class ClusterOperatorTest(AbstractOperatorTest[NDArray[bool]]):
#
#     N_OBJECTS = 30
#     STATIONARY_DISTRIBUTION = stats.binom(N_OBJECTS, 0.5)
#
#     def set_value_in_sample(self, value: Value, sample: Sample):
#         sample._clusters = Clusters(value)
#
#     def get_value_from_sample(self, sample: Sample) -> Value:
#         return sample.clusters.value
#
#     def generate_initial_value(self) -> Value:
#         return np.random.random((1, self.N_OBJECTS)) < 0.5
#
#     def sample_stationary_distribution(self, n_samples: int) -> list[Value]:
#         return [np.random.random((1, self.N_OBJECTS)) < 0.5 for _ in range(n_samples)]
#
#     # def get_stat_extractors(self) -> Iterator[callable]:
#     #     yield lambda value: value.sum()  # cluster size
#     #     for i in range(self.N_OBJECTS):
#     #         yield lambda value: value[0, i]
#     #
#     # def get_stationary_distribution_cdf(self) -> callable:
#     #     return stats.binom(self.N_OBJECTS, 0.5).cdf
#     #
#     # def iterate_comparable_statistics(
#     #     self, samples: list[Value]
#     # ) -> Iterator[list[float]]:
#     #     for extractor in self.get_stat_extractors():
#     #         yield [extractor(s) for s in samples]
#
#     def compare_sampled_values(self, mcmc: list[Value], exact: list[Value]):
#         n_samples = len(mcmc)
#         mcmc = np.asarray(mcmc)  # shape = (n_samples, 1, n_objects)
#
#         for i in range(self.N_OBJECTS):
#             p_value_i = stats.binom_test(
#                 x=np.sum(mcmc[:, 0, i]),
#                 n=n_samples,
#                 p=0.5,
#             )
#             assert p_value_i > 0.001
#
#         p_value_flat = stats.binom_test(
#             x=np.sum(mcmc),
#             n=n_samples * self.N_OBJECTS,
#             p=0.5,
#         )
#         print(p_value_flat)
#         assert p_value_flat > 0.01, p_value_flat
#
#     def get_model(self):
#         objects = dummy_objects(self.N_OBJECTS)
#         feature_values = np.zeros((self.N_OBJECTS, 0, 0))
#         features = dummy_features_from_values(feature_values)
#         data = Data(objects=objects, features=features, confounders={})
#         config = {} # TODO fix this
#         return Model(data, config=config)
#
#
# class AlterClusterTest(ClusterOperatorTest, unittest.TestCase):
#
#     def get_operator(self) -> Operator:
#         return AlterCluster(
#             weight=0.0,
#             adjacency_matrix=np.ones((self.N_OBJECTS, self.N_OBJECTS), dtype=bool),
#             p_grow_connected=0.8,
#             model_by_chain={0: DummyModel(self.N_OBJECTS)},
#             resample_source=False,
#             sample_from_prior=False,
#         )
#
# # class AlterClusterGibbsishTest(ClusterOperatorTest, unittest.TestCase):
# #
# #     def get_operator(self) -> Operator:
# #         return AlterClusterGibbsish(
# #             weight=0.0,
# #             adjacency_matrix=np.ones((self.N_OBJECTS, self.N_OBJECTS), dtype=bool),
# #             p_grow_connected=0.8,
# #             model_by_chain={0: DummyModel(self.N_OBJECTS)},
# #             resample_source=False,
# #             sample_from_prior=False,
# #             features=np.zeros((self.N_OBJECTS, 1, 1))
# #         )


class OperatorsTest(unittest.TestCase):

    data: Data
    results_path: Path
    results: Results

    CONFIG_PATH = Path("test/test_files/config.yaml")
    EXPERIMENT_NAME = "operator_test"
    N_REFERENCE_SAMPLES = 50_000

    def run_sbayes(self):
        """Run a sbayes analysis which generates the samples to be evaluated."""
        experiment = Experiment(config_file=self.CONFIG_PATH, experiment_name=self.EXPERIMENT_NAME, log=True)
        data = Data.from_experiment(experiment)
        mcmc = MCMCSetup(data=data, experiment=experiment)
        mcmc.sample()

        self.data = data
        self.results_path = experiment.path_results

        experiment.close()

    def generate_reference_samples(self):
        """Generate reference samples using importance sampling"""
        experiment = Experiment(config_file=self.CONFIG_PATH, experiment_name=self.EXPERIMENT_NAME, log=True)
        data = Data.from_experiment(experiment)
        model = Model(data, experiment.config.model)

        # Generate reference samples from prior distribution
        self.reference_samples = model.prior.generate_samples(self.N_REFERENCE_SAMPLES)

        # Assign importance weights according to the likelihood of each sample
        for sample in self.reference_samples:
            recalculate_feature_counts(data.features.values, sample)
            sample.lh = model.likelihood(sample, caching=False)
        self.ref_lh = np.array([sample.lh for sample in self.reference_samples])
        self.ref_importance = normalize(np.exp(self.ref_lh))

        self.ref_weights = {}
        for i_feat, feat in enumerate(data.features.names):
            self.ref_weights[feat] = np.array([
                s.weights.value[i_feat] for s in self.reference_samples
            ])

        # self.ref_conf_effects = {}
        # for fam in data.confounders['family'].group_names:
        #     self.ref_conf_effects[fam] = np.array([
        #         s.confounding_effects[fam].value for s in self.reference_samples
        #     ])

        self.ref_clusters = np.array([
            s.clusters.value
            for s in self.reference_samples
        ]).transpose((1, 0, 2))

        experiment.close()

    def setUp(self) -> None:
        self.run_sbayes()
        self.results = Results.from_csv_files(
            clusters_path=self.results_path / "K1" / "clusters_K1_1.txt",
            parameters_path=self.results_path / "K1" / "stats_K1_1.txt"
        )
        self.generate_reference_samples()

    def test_everything(self):
        # for conf_name, conf_eff in self.results.confounding_effects.items():
        #     for group_name, group_eff in conf_eff.items():
        #         for feat_name, probs in group_eff.items():
        #             print(f'pref {conf_name} {group_name} {feat_name}', kstest(probs[:, 0], self.ref_weights[k][:, 0]))
        #             plt.hist([v[:, 0], self.ref_weights[k][:, 0]], bins=30, density=True)
        #             plt.show()

        n_samples = self.results.n_samples
        for i_clust, cluster in enumerate(self.results.clusters):
            for i_obj in range(self.results.n_objects):
                print(np.mean(self.ref_clusters[i_clust][:, i_obj]),
                      np.mean(cluster[:, i_obj]))
                p_value = binom_test(
                    x=np.sum(cluster[:, i_obj]),
                    n=n_samples,
                    # p=self.ref_clusters[i_clust][:, i_obj].mean()
                    p=self.ref_importance.dot(self.ref_clusters[i_clust][:, i_obj])
                )
                print(f'p-value for cluster {i_clust} object {i_obj}:    {p_value:.3f}')

            p_value_size = binom_test(
                x=np.sum(cluster),
                n=n_samples * self.results.n_objects,
                p=self.ref_clusters[i_clust].mean()
            )
            print(f'p-value for cluster {i_clust} size:    {p_value_size:.3f}')



if __name__ == "__main__":
    unittest.main()



"""

IDEALES TEST SET-UP


1. END-TO-END

    Generate sBayes samples:
        - Model and data like in real analysis (load from config.yaml and features.csv)
        - Run sBayes
    Generate reference samples:
        - Importance sampling with custom prior sampler as a proposal distribution and 
          model.likelihood for importance weights 
        - Use Model object for posterior
    Tests:
        - 2 sample KS test


2. CUSTOM: 
    Generate samples:
        - Model and data like in real analysis
        - Model set-up in config file
    - Daten in csv file


"""
