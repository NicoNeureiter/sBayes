from __future__ import annotations
from typing import TextIO, Optional
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import tables

from sbayes.load_data import Data
from sbayes.sampling.operators import Operator
from sbayes.util import format_cluster_columns, get_best_permutation
from sbayes.model import Model
from sbayes.sampling.state import Sample, ModelCache


class ResultsLogger(ABC):

    def __init__(
        self,
        path: str,
        data: Data,
        model: Model,
    ):
        self.path: str = path
        self.data: Data = data
        self.model: Model = model.__copy__()

        self.file: Optional[TextIO] = None
        self.column_names: Optional[list] = None

    @abstractmethod
    def write_header(self, sample: Sample):
        pass

    @abstractmethod
    def _write_sample(self, sample: Sample):
        pass

    def write_sample(self, sample: Sample):
        if self.file is None:
            self.open()
            self.write_header(sample)
        self._write_sample(sample)

    def open(self):
        self.file = open(self.path, "w", buffering=1)
        # ´buffering=1´ activates line-buffering, i.e. flushing to file after each line

    def close(self):
        self.file.close()
        self.file = None


class ParametersCSVLogger(ResultsLogger):

    """The ParametersCSVLogger collects all real-valued parameters (weights, alpha, beta,
    gamma) and some statistics (cluster size, likelihood, prior, posterior) and continually
    writes them to a tab-separated text-file."""

    def __init__(
        self,
        *args,
        log_contribution_per_cluster: bool = True,
        float_format: str = "%.12g",
        match_clusters: bool = True,
    ):
        super().__init__(*args)
        self.float_format = float_format
        self.log_contribution_per_cluster = log_contribution_per_cluster
        self.match_clusters = match_clusters
        self.cluster_sum: Optional[npt.NDArray[int]] = None

        # For logging single cluster likelihood values we do not want to use the sampled
        # source arrays
        self.model.sample_source = False
        self.model.prior.sample_source = False

    def write_header(self, sample: Sample):
        feature_names = self.data.features.names
        state_names = self.data.features.state_names

        column_names = ["Sample", "posterior", "likelihood", "prior"]

        # No need for matching if only 1 cluster (or no clusters at all)
        if sample.n_clusters <= 1:
            self.match_clusters = False

        # Initialize cluster_sum array for matching
        self.cluster_sum = np.zeros((sample.n_clusters, sample.n_objects), dtype=int)

        # Cluster sizes
        for i in range(sample.n_clusters):
            column_names.append(f"size_a{i}")

        # weights
        for i_f, f in enumerate(feature_names):
            # Areal effect
            w_areal = f"w_areal_{f}"
            column_names += [w_areal]
            # index of areal = 0
            # todo: use source_index instead of remembering the order

            # Confounding effects
            for i_conf, conf in enumerate(self.data.confounders.values()):
                w_conf = f"w_{conf.name}_{f}"
                column_names += [w_conf]
                # todo: use source_index instead of remembering the order
                # index of confounding effect starts with 1

        # Areal effect
        for i_a in range(sample.n_clusters):
            for i_f, f in enumerate(feature_names):
                for i_s, s in enumerate(state_names[i_f]):
                    col_name = f"areal_a{i_a+1}_{f}_{s}"
                    column_names += [col_name]

        # Confounding effects
        for conf in self.data.confounders.values():
            for i_g, g in enumerate(conf.group_names):
                for i_f, f in enumerate(feature_names):
                    for i_s, s in enumerate(state_names[i_f]):
                        feature_name = f"{conf.name}_{g}_{f}_{s}"
                        column_names += [feature_name]

        # lh, prior, posteriors
        if self.log_contribution_per_cluster:
            for i in range(sample.n_clusters):
                column_names += [f"post_a{i}", f"lh_a{i}", f"prior_a{i}"]

        # Store the column names in an attribute (important to keep order consistent)
        self.column_names = column_names

        # Write the column names to the logger file
        self.file.write("\t".join(column_names) + "\n")

    def _write_sample(self, sample: Sample):
        feature_names = self.data.features.names
        state_names = self.data.features.state_names

        if self.match_clusters:
            # Compute the best matching permutation
            permutation = get_best_permutation(sample.clusters.value, self.cluster_sum)

            # Permute parameters
            cluster_effect = sample.cluster_effect.value[permutation, :, :]
            clusters = sample.clusters.value[permutation, :]

            # Update cluster_sum for matching future samples
            self.cluster_sum += clusters
        else:
            # Unpermuted parameters
            cluster_effect = sample.cluster_effect.value
            clusters = sample.clusters.value

        row = {
            "Sample": sample.i_step,
            "posterior": sample.last_lh + sample.last_prior,
            "likelihood": sample.last_lh,
            "prior": sample.last_prior,
        }

        # Cluster sizes
        for i, cluster in enumerate(clusters):
            col_name = f"size_a{i}"
            row[col_name] = np.count_nonzero(cluster)

        # Weights
        for i_f, f in enumerate(feature_names):

            # Areal effect weights
            w_areal = f"w_areal_{f}"
            # index of areal = 0
            # todo: use source_index instead of remembering the order
            row[w_areal] = sample.weights.value[i_f, 0]

            # Confounding effects weights
            for i_conf, conf in enumerate(self.data.confounders.values(), start=1):
                w_conf = f"w_{conf.name}_{f}"
                # todo: use source_index instead of remembering the order
                # index of confounding effect starts with 1
                row[w_conf] = sample.weights.value[i_f, i_conf]

        # Areal effect
        for i_a in range(sample.n_clusters):
            for i_f, f in enumerate(feature_names):
                for i_s, s in enumerate(state_names[i_f]):
                    col_name = f"areal_a{i_a+1}_{f}_{s}"
                    row[col_name] = cluster_effect[i_a, i_f, i_s]

        # Confounding effects
        for conf in self.data.confounders.values():
            for i_g, g in enumerate(conf.group_names):
                for i_f, f in enumerate(feature_names):
                    for i_s, s in enumerate(state_names[i_f]):
                        col_name = f"{conf.name}_{g}_{f}_{s}"
                        row[col_name] = sample.confounding_effects[conf.name].value[i_g, i_f, i_s]

        # lh, prior, posteriors
        if self.log_contribution_per_cluster:
            sample_single_cluster = sample.copy()
            sample_single_cluster._source = None

            for i in range(sample.n_clusters):
                sample_single_cluster.clusters._value = clusters[[i], :]
                sample_single_cluster.cluster_effect._value = cluster_effect[[i],...]
                sample_single_cluster.cache = ModelCache(sample_single_cluster)
                lh = self.model.likelihood(sample_single_cluster, caching=False)
                prior = self.model.prior(sample_single_cluster, caching=False)
                row[f"lh_a{i}"] = lh
                row[f"prior_a{i}"] = prior
                row[f"post_a{i}"] = lh + prior

        row_str = "\t".join([self.float_format % row[k] for k in self.column_names])
        self.file.write(row_str + "\n")


class ClustersLogger(ResultsLogger):

    """The ClustersLogger encodes each cluster in a bit-string and continually writes multiple
    clusters to a tab-separated text file."""

    def __init__(
        self,
        *args,
        match_clusters: bool = True,
    ):
        super().__init__(*args)
        self.match_clusters = match_clusters
        self.cluster_sum: Optional[npt.NDArray[int]] = None

    def write_header(self, sample: Sample):
        if sample.n_clusters <= 1:
            # Nothing to match
            self.match_clusters = False

        self.cluster_sum = np.zeros((sample.n_clusters, sample.n_objects), dtype=int)

    def _write_sample(self, sample):
        if self.match_clusters:
            # Compute best matching perm
            permutation = get_best_permutation(sample.clusters.value, self.cluster_sum)

            # Permute clusters
            clusters = sample.clusters.value[permutation, :]

            # Update cluster_sum for matching future samples
            self.cluster_sum += clusters
        else:
            clusters = sample.clusters.value

        row = format_cluster_columns(clusters)
        self.file.write(row + "\n")


class LikelihoodLogger(ResultsLogger):

    """The LikelihoodLogger continually writes the likelihood of each observation (one per
     site and feature) as a flattened array to a pytables file (.h5)."""

    def __init__(self, *args, **kwargs):
        self.logged_likelihood_array = None
        super().__init__(*args, **kwargs)

    def open(self):
        self.file = tables.open_file(self.path, mode="w")

    def write_header(self, sample: Sample):
        # Create the likelihood array
        self.logged_likelihood_array = self.file.create_earray(
            where=self.file.root,
            name="likelihood",
            atom=tables.Float32Col(),
            filters=tables.Filters(
                complevel=9, complib="blosc:zlib", bitshuffle=True, fletcher32=True
            ),
            shape=(0, sample.n_objects * sample.n_features),
        )

    def _write_sample(self, sample: Sample):
        self.logged_likelihood_array.append(sample.observation_lhs[None, ...])


class OperatorStatsLogger(ResultsLogger):

    """The OperatorStatsLogger keeps an operator_stats.txt file which contains statistics
     on each MCMC operator. The file is updated at each logged sample."""

    COL_WIDTHS = [20, 8, 8, 8, 10]

    def __init__(self, *args, operators: list[Operator], **kwargs):
        super().__init__(*args, **kwargs)
        self.operators = operators

    def write_sample(self, sample: Sample):
        with open(self.path, 'w') as self.file:
            self.write_to(self.file)

    def write_to(self, out: TextIO):
        out.write(self.get_log_message_header() + '\n')
        for operator in self.operators:
            out.write(self.get_log_message_row(operator) + '\n')

    @classmethod
    def get_log_message_header(cls) -> str:
        name_header = str.ljust('OPERATOR', cls.COL_WIDTHS[0])
        acc_header = str.ljust('ACCEPTS', cls.COL_WIDTHS[1])
        rej_header = str.ljust('REJECTS', cls.COL_WIDTHS[2])
        total_header = str.ljust('TOTAL', cls.COL_WIDTHS[3])
        acc_rate_header = 'ACC. RATE'

        return '\t'.join([name_header, acc_header, rej_header, total_header, acc_rate_header])

    @classmethod
    def get_log_message_row(cls, operator: Operator) -> str:
        if operator.total == 0:
            row_strings = [operator.operator_name, '-', '-', '-', '-']
            return '\t'.join([str.ljust(x, cls.COL_WIDTHS[i]) for i, x in enumerate(row_strings)])

        name_str = str.ljust(operator.operator_name, cls.COL_WIDTHS[0])
        acc_str = str.ljust(str(operator.accepts), cls.COL_WIDTHS[1])
        rej_str = str.ljust(str(operator.rejects), cls.COL_WIDTHS[2])
        total_str = str.ljust(str(operator.total), cls.COL_WIDTHS[3])
        acc_rate_str = '%.2f%%' % (100 * operator.acceptance_rate)

        return '\t'.join([name_str, acc_str, rej_str, total_str, acc_rate_str])

    def write_header(self, sample: Sample):
        pass

    def _write_sample(self, sample: Sample):
        pass
