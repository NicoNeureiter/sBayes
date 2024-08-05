from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, Optional, Callable
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import tables

from sbayes.load_data import Data, FeatureType
from sbayes.sampling.conditionals import conditional_effect_sample
from sbayes.sampling.operators import Operator
from sbayes.util import format_cluster_columns, get_best_permutation, gaussian_mu_posterior_sample, \
    poisson_lambda_posterior_sample, EPS
from sbayes.model import Model
from sbayes.model.likelihood import update_categorical_weights, Likelihood, get_fixed_sigma
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
        float_format: str = "%.12g",
        match_clusters: bool = True,
        log_source: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.float_format = float_format
        self.log_contribution_per_cluster = log_contribution_per_cluster
        self.match_clusters = match_clusters
        self.log_source = log_source

        self.cluster_sum = np.zeros((self.model.shapes.n_clusters, self.model.shapes.n_objects), dtype=int)

    def write_header(self, sample: Sample):

        feature_names = {}
        for ft in sample.feature_type_samples:
            if ft is FeatureType.categorical:
                feature_names[ft] = [k for k in self.data.features.categorical.names]
                state_names = {k: v for k, v in self.data.features.categorical.names.items()}
            else:
                feature_names[ft] = self.data.features.get_ft_features(ft).names.tolist()

        column_names = ["Sample", "posterior", "likelihood", "prior"]

        # No need for matching if only 1 cluster (or no clusters at all)
        if sample.n_clusters <= 1:
            self.match_clusters = False

        # # Initialize cluster_sum array for matching
        # self.cluster_sum = np.zeros((sample.n_clusters, sample.n_objects), dtype=int)

        # Cluster sizes
        for i in range(sample.n_clusters):
            column_names.append(f"size_cluster_{i}")

        # weights
        for ft in sample.feature_type_samples:

            for i_f, f in enumerate(feature_names[ft]):
                # Cluster effect
                column_names += [f"w_cluster_{f}"]
                # index of cluster = 0
                # todo: use source_index instead of remembering the order

                # Confounding effects
                for i_conf, conf in enumerate(self.data.confounders.values()):
                    column_names += [f"w_{conf.name}_{f}"]
                    # todo: use source_index instead of remembering the order
                    # index of confounding effect starts with 1

                # Cluster effect
                for i_c in range(sample.n_clusters):
                    if ft == "categorical":
                        for i_s, s in enumerate(state_names[f]):
                            column_names += [f"cluster_{i_c}_{f}_{s}"]
                    elif ft == "gaussian":
                        column_names += [f"cluster_{i_c}_{f}_mu"]
                        column_names += [f"cluster_{i_c}_{f}_sigma"]
                    elif ft == "poisson":
                        column_names += [f"cluster_{i_c}_{f}_lambda"]
                    elif ft == "logitnormal":
                        column_names += [f"cluster_{i_c}_{f}_mu_logit"]
                        column_names += [f"cluster_{i_c}_{f}_sigma_logit"]

                # Confounding effects
                for conf in self.data.confounders.values():
                    for i_g, g in enumerate(conf.group_names):
                        if ft == "categorical":
                            for i_s, s in enumerate(state_names[f]):
                                column_names += [f"{conf.name}_{g}_{f}_{s}"]
                        elif ft == "gaussian":
                            column_names += [f"{conf.name}_{g}_{f}_mu"]
                            column_names += [f"{conf.name}_{g}_{f}_sigma"]
                        elif ft == "poisson":
                            column_names += [f"{conf.name}_{g}_{f}_lambda"]
                        elif ft == "logitnormal":
                            column_names += [f"{conf.name}_{g}_{f}_mu_logit"]
                            column_names += [f"{conf.name}_{g}_{f}_sigma_logit"]

                # Contribution of each component to each feature (mean source assignment across objects)
                if self.log_source:
                    for i_s, source in enumerate(sample.component_names):
                        column_names += [f"source_{source}_{f}"]

        # lh, prior, posteriors
        if self.log_contribution_per_cluster:
            for i in range(sample.n_clusters):
                column_names += [f"post_a{i}", f"lh_a{i}", f"prior_a{i}"]

        # Prior column
        # todo: do we have to change things for logging the source and weights prior?
        # column_names += ["cluster_size_prior", "geo_prior", "source_prior", "weights_prior"]

        # Store the column names in an attribute (important to keep order consistent)
        self.column_names = column_names

        # Write the column names to the logger file
        if not self.resume:
            self.file.write("\t".join(column_names) + "\n")

    def _write_sample(self, sample: Sample):
        features = self.data.features
        clusters = sample.clusters.value
        prior = self.model.prior

        # Compute cluster effects

        cluster_effect = {}
        if FeatureType.categorical in sample.feature_type_samples:
            is_source_cluster = sample.clusters.value[..., np.newaxis] & sample.categorical.source.value[np.newaxis, ..., 0]
            cluster_effect[FeatureType.categorical] = conditional_effect_sample(
                features=features.categorical.values,
                is_source_group=is_source_cluster,
                applicable_states=features.categorical.states,
                prior_counts=prior.prior_cluster_effect.categorical.concentration_array,
            )
        if FeatureType.gaussian in sample.feature_type_samples:
            cluster_effect[FeatureType.gaussian] = np.empty((sample.n_clusters, sample.gaussian.n_features, 2))
            for i_c, cluster in enumerate(sample.clusters.value):
                sigma = get_fixed_sigma(self.data.features.gaussian.values, axis=0)

                is_source_cluster = cluster[:, np.newaxis] & sample.gaussian.source.value[..., 0]
                cluster_effect[FeatureType.gaussian][i_c, :, 0] = gaussian_mu_posterior_sample(
                    x=self.data.features.gaussian.values,
                    sigma=sigma,
                    mu_0=prior.prior_cluster_effect.gaussian.mean.mu_0_array,
                    sigma_0=prior.prior_cluster_effect.gaussian.mean.sigma_0_array,
                    in_component=is_source_cluster,
                )

                cluster_effect[FeatureType.gaussian][i_c, :, 1] = sigma

        if FeatureType.poisson in sample.feature_type_samples:
            cluster_effect[FeatureType.poisson] = np.empty((sample.n_clusters, sample.poisson.n_features,))
            for i_c, cluster in enumerate(sample.clusters.value):
                is_source_cluster = cluster[:, np.newaxis] & sample.poisson.source.value[..., 0]
                cluster_effect[FeatureType.poisson][i_c] = poisson_lambda_posterior_sample(
                    x=self.data.features.poisson.values,
                    alpha_0=prior.prior_cluster_effect.poisson.mean.mu_0_array,
                    beta_0=prior.prior_cluster_effect.poisson.mean.sigma_0_array,
                    in_component=is_source_cluster,
                )

        if self.match_clusters:
            # Compute the best matching permutation
            permutation = get_best_permutation(sample.clusters.value, self.cluster_sum)

            # Permute parameters
            for ft in sample.feature_type_samples.keys():
                cluster_effect[ft] = cluster_effect[ft][permutation, ...]
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
            col_name = f"size_cluster_{i}"
            row[col_name] = np.count_nonzero(cluster)

        for ft in sample.feature_type_samples:

            # Weights
            for i_f, f in enumerate(features[ft].names):

                # cluster effect weights
                w_cluster = f"w_cluster_{f}"
                # index of cluster = 0
                # todo: use source_index instead of remembering the order
                row[w_cluster] = getattr(sample, ft).weights.value[i_f, 0]

                # Confounding effects weights
                for i_conf, conf in enumerate(self.data.confounders.values(), start=1):
                    w_conf = f"w_{conf.name}_{f}"
                    # todo: use source_index instead of remembering the order
                    # index of confounding effect starts with 1
                    row[w_conf] = getattr(sample, ft).weights.value[i_f, i_conf]

            # cluster effect
            for i_c in range(sample.n_clusters):
                for i_f, f in enumerate(features[ft].names):
                    if ft == FeatureType.categorical:
                        for i_s, s in enumerate(features.categorical.names[f]):
                            col_name = f"cluster_{i_c}_{f}_{s}"
                            row[col_name] = cluster_effect[ft][i_c, i_f, i_s]
                    if ft == FeatureType.gaussian:
                        col_name = f"cluster_{i_c}_{f}_mu"
                        row[col_name] = cluster_effect[ft][i_c, i_f, 0]

                        col_name = f"cluster_{i_c}_{f}_sigma"
                        row[col_name] = cluster_effect[ft][i_c, i_f, 1]

                    if ft == FeatureType.poisson:
                        col_name = f"cluster_{i_c}_{f}_lambda"
                        row[col_name] = cluster_effect[ft][i_c, i_f]

                    if ft == FeatureType.logitnormal:
                        col_name = f"cluster_{i_c}_{f}_mu_logit"
                        row[col_name] = cluster_effect[ft][i_c, i_f, 0]

                        col_name = f"cluster_{i_c}_{f}_sigma_logit"
                        row[col_name] = cluster_effect[ft][i_c, i_f, 1]

            # Confounding effects
            for i_conf, conf in enumerate(self.data.confounders.values(), start=1):
                conf_prior = prior.prior_confounding_effects[conf.name]

                if ft == FeatureType.categorical:
                    conf_effect = conditional_effect_sample(
                        features=features.categorical.values,
                        is_source_group=conf.group_assignment[..., None] & sample.categorical.source.value[None, ..., i_conf],
                        applicable_states=features.categorical.states,
                        prior_counts=conf_prior.categorical.concentration_array(sample),
                    )
                    for i_g, g in enumerate(conf.group_names):
                        for i_f, f in enumerate(features.categorical.names):
                            for i_s, s in enumerate(features.categorical.names[f]):
                                col_name = f"{conf.name}_{g}_{f}_{s}"
                                row[col_name] = conf_effect[i_g, i_f, i_s]

                if ft == FeatureType.gaussian:

                    for i_g, g in enumerate(conf.group_names):
                        # Sample gaussian parameters (mu and sigma) from conditional posterior
                        is_source_group = conf.group_assignment[i_g, :, np.newaxis] & sample.gaussian.source.value[..., 0]
                        sigma = get_fixed_sigma(features.gaussian.values, axis=0)
                        mu = gaussian_mu_posterior_sample(
                            x=features.gaussian.values,
                            sigma=sigma,
                            mu_0=conf_prior.gaussian.mean.mu_0_array[i_g],
                            sigma_0=conf_prior.gaussian.mean.sigma_0_array[i_g],
                            in_component=is_source_group,
                        )

                        # Write to row dictionary
                        for i_f, f in enumerate(features.gaussian.names):
                            col_name = f"{conf.name}_{g}_{f}_mu"
                            row[col_name] = mu[i_f]

                            col_name = f"{conf.name}_{g}_{f}_sigma"
                            row[col_name] = sigma[i_f]

                if ft == FeatureType.poisson:
                    for i_g, g in enumerate(conf.group_names):
                        # Sample poisson mean from conditional posterior distribution
                        is_source_group = conf.group_assignment[i_g, :, None] & sample.poisson.source.value[..., 0]
                        poisson_mean = poisson_lambda_posterior_sample(
                            x=features.poisson.values,
                            alpha_0=conf_prior.poisson.mean.mu_0_array,
                            beta_0=conf_prior.poisson.mean.sigma_0_array,
                            in_component=is_source_group,
                        )

                        # Write to row dictionary
                        for i_f, f in enumerate(features.poisson.names):
                            col_name = f"{conf.name}_{g}_{f}_lambda"
                            row[col_name] = poisson_mean[i_f]

                if ft == FeatureType.logitnormal:
                    for i_g, g in enumerate(conf.group_names):
                        for i_f, f in enumerate(features[ft].names):
                            col_name = f"{conf.name}_{g}_{f}_mu_logit"
                            row[col_name] = 13

                            col_name = f"{conf.name}_{g}_{f}_sigma_logit"
                            row[col_name] = 13

            # Contribution of each component to each feature (mean source assignment across objects)
            if self.log_source:
                mean_source = np.mean(getattr(sample, ft).source.value, axis=0)
                for i_f, f in enumerate(features[ft].names):
                    for i_s, source in enumerate(sample.component_names):
                        col_name = f"source_{source}_{f}"
                        row[col_name] = mean_source[i_f, i_s]

        row["cluster_size_prior"] = sample.cache.cluster_size_prior.value
        row["geo_prior"] = sample.cache.geo_prior.value.sum()
        # todo: log source and weights prior
        # row["source_prior"] = sample.cache.source_prior.value.sum()
        # row["weights_prior"] = sample.cache.weights_prior.value

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
            n_features = 0
            for ft in sample.feature_type_samples:
                n_features += self.data.features.get_ft_features(ft).n_features

            self.logged_likelihood_array = self.file.create_earray(
                where=self.file.root,
                name="likelihood",
                atom=tables.Float32Col(),
                filters=tables.Filters(
                    complevel=9, complib="blosc:zlib", bitshuffle=True, fletcher32=True
                ),
                # activate for all features
                shape=(0, sample.n_objects * n_features),
            )

    def _write_sample(self, sample: Sample):
        for ft in sample.feature_type_samples:
            continue
            # todo: generalise for all features and activate
            # lh_per_comp = 13
            # lh_per_comp = likelihood_per_component(model=self.model, sample=sample)
            # weights = update_categorical_weights(sample)
            # lh = np.sum(weights * lh_per_comp, axis=2).ravel()
            # self.logged_likelihood_array.append(lh[None, ...])


class OperatorStatsLogger(ResultsLogger):

    """The OperatorStatsLogger keeps an operator_stats.txt file which contains statistics
     on each MCMC operator. The file is updated at each logged sample."""

    COLUMNS: dict[str, int] = {
        "OPERATOR": 30,
        "ACCEPTS": 8,
        "REJECTS": 8,
        "TOTAL": 8,
        "ACCEPT-RATE": 11,
        "STEP-TIME": 11,
        "STEP-SIZE": 11,
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
        return '\t'.join(headers)

        # name_header = str.ljust('OPERATOR', cls.COL_WIDTHS[0])
        # acc_header = str.ljust('ACCEPTS', cls.COL_WIDTHS[1])
        # rej_header = str.ljust('REJECTS', cls.COL_WIDTHS[2])
        # total_header = str.ljust('TOTAL', cls.COL_WIDTHS[3])
        # acc_rate_header = str.ljust('ACC. RATE', cls.COL_WIDTHS[4])
        # parameters_header = str.ljust('PARAMETERS', cls.COL_WIDTHS[5])
        # return '\t'.join([name_header, acc_header, rej_header, total_header, acc_rate_header])

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
        step_time_str = f"{1000 * np.mean(operator.step_times):.2f} ms".ljust(cls.COLUMNS["STEP-TIME"])
        if operator.step_sizes:
            step_size_str = f"{np.mean(operator.step_sizes):.2f}".ljust(cls.COLUMNS["STEP-SIZE"])
        else:
            step_size_str = "-"
        paramters_str = operator.get_parameters_string().ljust(cls.COLUMNS["PARAMETERS"])

        return '\t'.join([name_str, acc_str, rej_str, total_str, acc_rate_str, step_time_str, step_size_str, paramters_str])

    def write_header(self, sample: Sample):
        pass

    def _write_sample(self, sample: Sample):
        pass
