from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TextIO, Optional
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import numpy.typing as npt
import pandas as pd
import tables

from sbayes.load_data import Data
from sbayes.preprocessing import sample_categorical
from sbayes.sampling.conditionals import conditional_effect_sample, likelihood_per_component_exact
from sbayes.sampling.operators import Operator
from sbayes.util import format_cluster_columns, get_best_permutation, normalize
from sbayes.model import Model
from sbayes.model.likelihood import update_weights
from sbayes.sampling.state import Sample, ModelCache


class ResultsWriter:

    def __init__(
        self,
        results_dir: Path,
        run: int,
        data: Data,
        model: Model,
        resume: bool,
        chains: int = 1,
    ):
        self.results_dir = results_dir
        self.data: Data = data
        self.model: Model = model.__copy__()

        self.logger = None
        self.resume = resume

        # Define results file paths
        k = self.model.shapes.n_clusters
        chain_str = '' if chains == 1 else f'.chain{0}'
        self.params_path = self.results_dir / f'stats_K{k}_{run}{chain_str}.txt'
        self.clusters_path = self.results_dir / f'clusters_K{k}_{run}{chain_str}.txt'
        self.likelihood_path = self.results_dir / f'likelihood_K{k}_{run}{chain_str}.h5'

    def write_numpyro_results(self, samples: dict):
        # self.likelihood_file = tables.open_file(self.likelihood_path, mode="a" if self.resume else "w")
        write_mode = "a" if self.resume else "w"
        with (open(self.clusters_path, write_mode) as self.clusters_file,
              open(self.params_path, write_mode) as self.params_file):

            cluster_permutations = self.write_clusters(samples)
            self.write_params(samples, cluster_permutations)

    def write_clusters(self, samples: dict) -> list:
        """
        Writes the cluster assignments for each sample to the clusters file.

        Args:
            samples: A dictionary containing the samples with cluster assignments.

        Returns:
            A list of permutations used to match cluster labels across samples.
        """
        clusters_sum = np.zeros((self.model.n_clusters, self.model.shapes.n_sites), dtype=int)
        cluster_permutations = []

        for sample in samples:
            clusters = sample['z']

            # Find the best permutation to match previous cluster labels
            permutation = get_best_permutation(clusters, clusters_sum)
            clusters = clusters[permutation]

            # Update the sum of clusters for future iterations
            clusters_sum += clusters

            # Write the clusters to the file
            clusters_binary = sample_categorical(clusters, binary_encoding=True)[:, :-1]
            clusters_string = format_cluster_columns(clusters_binary.T)
            self.clusters_file.write(clusters_string + '\n')

            # Store the permutation for later use
            cluster_permutations.append(permutation)

        return cluster_permutations

    # def write_params_header(self, sample: dict):
    #     feature_names = self.data.features.names
    #     state_names = self.data.features.state_names
    #     n_clusters = self.model.n_clusters
    #
    #     column_names = ["Sample", "posterior"]
    #
    #     # No need for matching if only 1 cluster (or no clusters at all)
    #     if n_clusters <= 1:
    #         self.match_clusters = False
    #
    #     # Cluster sizes
    #     for i in range(n_clusters):
    #         column_names.append(f"size_a{i}")
    #
    #     # weights
    #     for i_f, f in enumerate(feature_names):
    #         # Areal effect
    #         column_names += [f"w_areal_{f}"]
    #         # index of areal = 0
    #         # todo: use source_index instead of remembering the order
    #
    #         # Confounding effects
    #         for i_conf, conf in enumerate(self.data.confounders.values()):
    #             column_names += [f"w_{conf.name}_{f}"]
    #             # todo: use source_index instead of remembering the order
    #             # index of confounding effect starts with 1
    #
    #     # Areal effect
    #     for i_a in range(n_clusters):
    #         for i_f, f in enumerate(feature_names):
    #             for i_s, s in enumerate(state_names[i_f]):
    #                 column_names += [f"areal_a{i_a}_{f}_{s}"]
    #
    #     # Confounding effects
    #     for conf in self.data.confounders.values():
    #         for i_g, g in enumerate(conf.group_names):
    #             for i_f, f in enumerate(feature_names):
    #                 for i_s, s in enumerate(state_names[i_f]):
    #                     column_names += [f"{conf.name}_{g}_{f}_{s}"]
    #
    #     # Store the column names in an attribute (important to keep order consistent)
    #     self.column_names = column_names
    #
    #     # Write the column names to the logger file
    #     if not self.resume:
    #         self.params_file.write("\t".join(column_names) + "\n")

    def write_params(self, sample: dict, cluster_permutations: list):
        features = self.data.features

        # TODO: create a modified dictionary that contains the desired columns

        sample_df = pd.DataFrame.from_dict(sample)
        sample_df.to_csv(self.params_path, sep='\t', index=False)


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
    def write_header(self, sample: dict):
        pass

    @abstractmethod
    def _write_sample(self, sample: dict):
        pass

    def write_sample(self, sample: dict):
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

    def write_header(self, sample: dict):
        feature_names = self.data.features.names
        state_names = self.data.features.state_names
        n_clusters = self.model.n_clusters

        column_names = ["Sample", "posterior"]

        # No need for matching if only 1 cluster (or no clusters at all)
        if n_clusters <= 1:
            self.match_clusters = False

        # # Initialize cluster_sum array for matching
        # self.cluster_sum = np.zeros((sample.n_clusters, sample.n_objects), dtype=int)

        # Cluster sizes
        for i in range(n_clusters):
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
        for i_a in range(n_clusters):
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

    def _write_sample(self, sample: dict):
        features = self.data.features
        n_clusters = self.model.n_clusters

        clusters_continuous = sample['z']

        clusters = sample_categorical(clusters_continuous, binary_encoding=True)[:, :-1]

        cluster_effect = np.zeros((n_clusters, features.n_features, features.n_states))
        for partition in self.data.partitions:
            cluster_effect[:, partition.feature_indices, :partition.n_states] = sample[f'cluster_effect_{partition.name}']


        if self.match_clusters:
            # Compute the best matching permutation
            permutation = get_best_permutation(clusters, self.cluster_sum)

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


def get_cluster_names(n_clusters: int) -> list[str]:
    return [f"a{i}" for i in range(n_clusters)]


def write_samples(
    samples: dict,
    clusters_path: Path,
    params_path: Path,
    data: Data,
    model: Model,
    match_clusters: bool = True
):
    clusters_samples_cont = np.array(samples['z'])
    clusters_samples = sample_categorical(clusters_samples_cont, binary_encoding=True)[:, :, :-1].transpose(0, 2, 1)
    n_samples, n_clusters, n_objects = clusters_samples.shape

    # cluster_eff_samples = np.array(samples['cluster_effect'])
    cluster_eff_samples = np.zeros((n_samples, n_clusters, data.features.n_features, data.features.n_states))
    for partition in data.partitions:
        cluster_eff_samples[:, :, partition.feature_indices, :partition.n_states] = samples[f'cluster_effect_{partition.name}']

    if match_clusters:
        clusters_sum = np.zeros(clusters_samples.shape[1:], dtype=int)

        for i, clusters in enumerate(clusters_samples):
            # Compute best matching perm
            permutation = get_best_permutation(clusters, clusters_sum)

            # Permute clusters
            clusters = clusters[permutation]
            cluster_eff_samples[i] = cluster_eff_samples[i, permutation]

            # Update cluster_sum for matching future samples
            clusters_sum += clusters
            clusters_samples[i] = clusters

    cluster_names = get_cluster_names(n_clusters)
    feature_names = data.features.names
    state_names = [f"s{s}" for s in range(data.features.n_states)]

    cluster_sizes = np.sum(clusters_samples, axis=-1)

    sample_id_df = pd.DataFrame(data=np.arange(n_samples), columns=['Sample'])
    cluster_sizes_df = pd.DataFrame(data=cluster_sizes, columns=[f'size_{c}' for c in cluster_names])
    cluster_effects_df = samples_array_to_df(cluster_eff_samples, [cluster_names, feature_names, state_names], prefix='areal')

    valid_states = data.features.states
    all_conf_eff_dfs = []
    for i_c, conf in enumerate(data.confounders.values()):
        group_names = conf.group_names

        for partition in data.partitions:


            cluster_eff_samples[:, :, partition.feature_indices, :partition.n_states] = samples[f'cluster_effect_{partition.name}']

            conf_eff_samples = np.array(samples[f"conf_eff_{i_c}_{partition.name}"])[:, :-1, :, :]
            # conf_eff_raw_samples = np.array(samples[f"conf_eff_{i_c}_raw"])[:, :-1, :, :]
            # conf_eff_samples = normalize(np.where(valid_states[None, None, :, :], conf_eff_raw_samples, 0.0))

            conf_eff_df = samples_array_to_df(
                param_samples=conf_eff_samples,
                names=[group_names, feature_names[partition.feature_indices], state_names[:partition.n_states]],
                prefix=conf.name)
            all_conf_eff_dfs.append(conf_eff_df)

        # conf_eff_samples = np.array(samples[f"conf_eff_{i_c}"])[:, :-1, :, :]
        # # conf_eff_raw_samples = np.array(samples[f"conf_eff_{i_c}_raw"])[:, :-1, :, :]
        # # conf_eff_samples = normalize(np.where(valid_states[None, None, :, :], conf_eff_raw_samples, 0.0))
        # group_names = conf.group_names
        # conf_eff_df = samples_array_to_df(conf_eff_samples, [group_names, feature_names, state_names], prefix=conf.name)
        # all_conf_eff_dfs.append(conf_eff_df)

    params_df = pd.concat([sample_id_df, cluster_sizes_df, cluster_effects_df] + all_conf_eff_dfs, axis=1)

    # Write clusters file
    with open(clusters_path, "w") as clusters_file:
        for clusters in clusters_samples:
            row = format_cluster_columns(clusters)
            clusters_file.write(row + "\n")

    # Write params file
    with open(params_path, "w") as params_file:
        params_df.to_csv(params_file, sep='\t', index=False)


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


def samples_array_to_df(
    param_samples: np.ndarray,
    names: list[list[str]],
    prefix: str = "",
) -> pd.DataFrame:
    """
    Flatten an n-dimensional numpy array into a pandas DataFrame, with column names
    based on the provided names for each dimension.

    Parameters:
        param_samples (np.ndarray): An n-dimensional numpy array, where the first dimension
                              corresponds to the number of samples.
        names (list[list[str]]): A list of lists where each sublist provides names
                                 for the respective dimension of the array.

    Returns:
        pd.DataFrame: A pandas DataFrame with flattened data and appropriately named columns.
    """
    # Ensure the number of dimensions matches the names provided
    if len(param_samples.shape[1:]) != len(names):
        raise ValueError("Number of dimensions in samples does not match the number of name groups provided.")

    # Create column names by combining names for each dimension
    column_names = ['_'.join(combo) for combo in product(*names)]
    if prefix:
        column_names = [f"{prefix}_{col}" for col in column_names]

    # Reshape the samples array to have shape (num_samples, -1)
    flattened_samples = param_samples.reshape(param_samples.shape[0], -1)

    # Create a DataFrame using the flattened data and generated column names
    return pd.DataFrame(flattened_samples, columns=column_names)

def combine_param_samples(
    samples: dict[str, np.ndarray],
    param_names: dict[str, list[list[str]]]
) -> pd.DataFrame:
    """
    Combine multiple parameter samples stored in a dictionary into a single DataFrame.

    Parameters:

        samples (dict): A dictionary where keys are parameter names and values
                              are n-dimensional numpy arrays of samples.
        param_names (dict): A dictionary where keys are parameter names and values
                            are lists of lists for column naming for the respective parameter.

    Returns:
        pd.DataFrame: A combined DataFrame with all parameters, with prefixed column names.
    """
    combined_df = pd.DataFrame()

    for param, samples in (
            samples.items()):
        if param not in param_names:
            raise ValueError(f"No names provided for parameter '{param}'.")

        # Flatten the samples for this parameter
        param_df = samples_array_to_df(samples, param_names[param])

        # Prefix column names with the parameter name
        param_df.columns = [f"{param}_{col}" for col in param_df.columns]

        # Concatenate with the combined DataFrame
        combined_df = pd.concat([combined_df, param_df], axis=1)

    return combined_df


if __name__ == '__main__':
    # Example usage
    param_samples = {
        "param1": np.array([
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [8, 9, 10],
                [11, 12, 13]
            ],
        ]),
        "param2": np.array([
            [
                [14, 15],
                [16, 17]
            ],
            [
                [18, 19],
                [20, 21]
            ],
        ])
    }

    param_names = {
        "param1": [
            ['a', 'b'],
            ['x', 'y', 'z']
        ],
        "param2": [
            ['c', 'd'],
            ['u', 'v']
        ]
    }

    combined_df = combine_param_samples(param_samples, param_names)
    print(combined_df)
