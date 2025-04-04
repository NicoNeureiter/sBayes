from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TextIO, Optional
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import tables

from sbayes.load_data import Data
from sbayes.sampling.conditionals import conditional_effect_sample, likelihood_per_component_exact
from sbayes.sampling.operators import Operator
from sbayes.util import format_cluster_columns, get_best_permutation
from sbayes.model import Model
from sbayes.model.likelihood import update_weights
from sbayes.sampling.state import Sample, ModelCache


class ResultsLogger(ABC):

    def __init__(
        self,
        path: str,
        data: Data,
        model: Model,
        resume: bool,
    ):
        self.path: Path = Path(path)
        self.data: Data = data
        self.model: Model = model.__copy__()

        self.file: Optional[TextIO] = None
        self.column_names: Optional[list] = None

        self.resume = resume

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
        self.file = open(self.path, "a" if self.resume else "w", buffering=1)
        # ´buffering=1´ activates line-buffering, i.e. flushing to file after each line

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


class ParametersCSVLogger(ResultsLogger):

    """The ParametersCSVLogger collects all real-valued parameters (weights, alpha, beta,
    gamma) and some statistics (cluster size, likelihood, prior, posterior) and continually
    writes them to a tab-separated text-file."""

    def __init__(
        self,
        *args,
        log_contribution_per_cluster: bool = False,
        float_format: str = "%.8g",
        match_clusters: bool = True,
        log_source: bool = False,
        log_sample_id: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.float_format = float_format
        self.log_contribution_per_cluster = log_contribution_per_cluster
        self.match_clusters = match_clusters
        self.log_source = log_source
        self.log_sample_id = log_sample_id

        self.cluster_sum = np.zeros((self.model.shapes.n_clusters, self.model.shapes.n_sites), dtype=int)

    def write_header(self, sample: Sample):
        feature_names = self.data.features.names
        state_names = self.data.features.state_names

        column_names = ["Sample", "posterior", "likelihood", "prior"]

        # No need for matching if only 1 cluster (or no clusters at all)
        if sample.n_clusters <= 1:
            self.match_clusters = False

        # # Initialize cluster_sum array for matching
        # self.cluster_sum = np.zeros((sample.n_clusters, sample.n_objects), dtype=int)

        # Cluster sizes
        for i in range(sample.n_clusters):
            column_names.append(f"size_a{i}")

        # weights
        for i_f, f in enumerate(feature_names):
            # Areal effect
            column_names += [f"w_areal_{f}"]
            # index of areal = 0
            # todo: use source_index instead of remembering the order

            # Confounding effects
            for i_conf, conf in enumerate(self.data.confounders.values()):
                column_names += [f"w_{conf.name}_{f}"]
                # todo: use source_index instead of remembering the order
                # index of confounding effect starts with 1

        # Areal effect
        for i_a in range(sample.n_clusters):
            for i_f, f in enumerate(feature_names):
                for i_s, s in enumerate(state_names[i_f]):
                    column_names += [f"areal_a{i_a}_{f}_{s}"]

        # Confounding effects
        for conf in self.data.confounders.values():
            for i_g, g in enumerate(conf.group_names):
                for i_f, f in enumerate(feature_names):
                    for i_s, s in enumerate(state_names[i_f]):
                        column_names += [f"{conf.name}_{g}_{f}_{s}"]

        # Contribution of each component to each feature (mean source assignment across objects)
        if self.log_source:
            for i_f, f in enumerate(feature_names):
                for i_s, source in enumerate(sample.component_names):
                    column_names += [f"source_{source}_{f}"]


        # lh, prior, posteriors
        if self.log_contribution_per_cluster:
            for i in range(sample.n_clusters):
                column_names += [f"post_a{i}", f"lh_a{i}", f"prior_a{i}"]

        # Prior column
        column_names += ["cluster_size_prior", "geo_prior", "source_prior", "weights_prior"]

        if self.log_sample_id:
            column_names += ["sample_id"]

        # Store the column names in an attribute (important to keep order consistent)
        self.column_names = column_names

        # Write the column names to the logger file
        if not self.resume:
            self.file.write("\t".join(column_names) + "\n")

    def _write_sample(self, sample: Sample):
        features = self.data.features

        clusters = sample.clusters.value
        # cluster_effect = sample.cluster_effect.value
        cluster_effect = conditional_effect_sample(
            features=features.values,
            is_source_group=sample.clusters.value[..., np.newaxis] & sample.source.value[np.newaxis, ..., 0],
            applicable_states=features.states,
            prior_counts=self.model.prior.prior_cluster_effect.concentration_array,
        )

        if self.match_clusters:
            # Compute the best matching permutation
            permutation = get_best_permutation(sample.clusters.value, self.cluster_sum)

            # Permute parameters
            cluster_effect = cluster_effect[permutation, :, :]
            clusters = clusters[permutation, :]

            # Update cluster_sum for matching future samples
            self.cluster_sum += clusters

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
        for i_f, f in enumerate(features.names):

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
            for i_f, f in enumerate(features.names):
                for i_s, s in enumerate(features.state_names[i_f]):
                    col_name = f"areal_a{i_a}_{f}_{s}"
                    row[col_name] = cluster_effect[i_a, i_f, i_s]

        # Confounding effects
        for i_conf, conf in enumerate(self.data.confounders.values(), start=1):
            conf_effect = conditional_effect_sample(
                features=features.values,
                is_source_group=conf.group_assignment[..., np.newaxis] & sample.source.value[np.newaxis, ..., i_conf],
                applicable_states=features.states,
                prior_counts=self.model.prior.prior_confounding_effects[conf.name].concentration_array(sample),
            )

            for i_g, g in enumerate(conf.group_names):
                for i_f, f in enumerate(features.names):
                    for i_s, s in enumerate(features.state_names[i_f]):
                        col_name = f"{conf.name}_{g}_{f}_{s}"
                        row[col_name] = conf_effect[i_g, i_f, i_s]

        # Contribution of each component to each feature (mean source assignment across objects)
        if self.log_source:
            mean_source = np.mean(sample.source.value, axis=0)
            for i_f, f in enumerate(features.names):
                for i_s, source in enumerate(sample.component_names):
                    col_name = f"source_{source}_{f}"
                    row[col_name] = mean_source[i_f, i_s]

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

        row["cluster_size_prior"] = sample.cache.cluster_size_prior.value
        row["geo_prior"] = sample.cache.geo_prior.value.sum()
        row["source_prior"] = sample.cache.source_prior.value.sum()
        row["weights_prior"] = sample.cache.weights_prior.value

        if self.log_sample_id:
            row["sample_id"] = sample.chain

        row_str = "\t".join([self.float_format % row[k] for k in self.column_names])
        self.file.write(row_str + "\n")


class ClustersLogger(ResultsLogger):

    """The ClustersLogger encodes each cluster in a bit-string and continually writes multiple
    clusters to a tab-separated text file."""

    def __init__(
        self,
        *args,
        match_clusters: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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
        if self.resume:
            try:
                self.file = tables.open_file(self.path, mode="a")

            except tables.exceptions.HDF5ExtError as e:
                logging.warning(f"Could not append to existing likelihood file '{self.path.name}' ({type(e).__name__})."
                                f" Overwriting previously likelihood values instead.")
                # Set resume to False and open again
                self.resume = False
                self.open()

        else:
            self.file = tables.open_file(self.path, mode="w")

    def write_header(self, sample: Sample):
        if self.resume:
            self.logged_likelihood_array = self.file.root.likelihood
        else:
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

            na_values = self.model.data.features.na_values
            na_array = self.file.create_carray(
                where=self.file.root,
                name="na_values",
                obj=na_values.ravel(),
                atom=tables.BoolCol(),
                filters=tables.Filters(complevel=9, fletcher32=True),
                shape=(sample.n_objects * sample.n_features, ),
            )
            na_array.close()

    def _write_sample(self, sample: Sample):
        weights = update_weights(sample)
        lh_per_comp_exact = likelihood_per_component_exact(model=self.model, sample=sample)
        lh = np.sum(weights * lh_per_comp_exact, axis=2).ravel()
        self.logged_likelihood_array.append(lh[None, ...])
        self.file.flush()


class OperatorStatsLogger(ResultsLogger):

    """The OperatorStatsLogger keeps an operator_stats.txt file which contains statistics
     on each MCMC operator. The file is updated at each logged sample."""

    COLUMNS: dict[str, int] = {
        "OPERATOR": 27,
        "ACCEPTS": 8,
        "REJECTS": 8,
        "TOTAL": 8,
        "ACCEPT-RATE": 11,
        "STEP-SIZE": 11,
        "STEP-TIME": 11,
        # "NEXT STEP-TIME": 15,
        "PARAMETERS": 0,
    }

    N_COLUMNS: int = len(COLUMNS)

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
        headers = [column.ljust(width) for column, width in cls.COLUMNS.items()]
        return ' '.join(headers)

    @classmethod
    def get_log_message_row(cls, operator: Operator) -> str:
        if operator.total == 0:
            row_strings = [operator.operator_name] + ['-'] * cls.N_COLUMNS
            return '\t'.join([str.ljust(x, width) for x, width in zip(row_strings, cls.COLUMNS.values())])

        name_str = operator.operator_name.ljust(cls.COLUMNS["OPERATOR"])
        acc_str = str(operator.accepts).ljust(cls.COLUMNS["ACCEPTS"])
        rej_str = str(operator.rejects).ljust(cls.COLUMNS["REJECTS"])
        total_str = str(operator.total).ljust(cls.COLUMNS["TOTAL"])
        acc_rate_str = f"{operator.acceptance_rate:.2%}".ljust(cls.COLUMNS["ACCEPT-RATE"])
        step_size_str_raw = f"{np.mean(operator.step_sizes):.2f}" if operator.step_sizes else "-"
        step_size_str = step_size_str_raw.ljust(cls.COLUMNS["STEP-SIZE"])
        step_time_str = f"{1000 * np.mean(operator.step_times):.2f} ms".ljust(cls.COLUMNS["STEP-TIME"])
        # next_step_time_str = f"{1000 * np.mean(operator.next_step_times):.2f} ms".ljust(cls.COLUMNS["NEXT STEP-TIME"])
        paramters_str = operator.get_parameters_string().ljust(cls.COLUMNS["PARAMETERS"])

        return ' '.join([name_str, acc_str, rej_str, total_str, acc_rate_str, step_size_str, step_time_str, paramters_str])
        # return ' '.join([name_str, acc_str, rej_str, total_str, acc_rate_str, step_size_str, step_time_str, next_step_time_str, paramters_str])

    def write_header(self, sample: Sample):
        pass

    def _write_sample(self, sample: Sample):
        pass


class StateDumper(ResultsLogger):

    def write_header(self, sample: Sample):
        pass

    def _write_sample(self, sample: Sample):
        pass

    def open(self):
        pass

    def close(self):
        pass

    def write_sample(self, sample: Sample):
        with open(self.path, "wb") as pickle_file:
            pickle.dump(sample, pickle_file)
