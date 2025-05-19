from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from itertools import product

import numpy as np
import numpyro.handlers
import pandas as pd
from numpyro.infer import log_likelihood
import pickle
import tables

from sbayes.load_data import Data, CategoricalFeatures, GaussianFeatures, PoissonFeatures, GenericTypeFeatures
from sbayes.preprocessing import sample_categorical
from sbayes.util import format_cluster_columns, get_best_permutation, normalize
from sbayes.model import Model

import warnings

# Suppress warnings from PyTables about natural names
warnings.simplefilter("ignore", category=tables.NaturalNameWarning)


def get_cluster_names(n_clusters: int) -> list[str]:
    """Create a list of names for the clusters based on the number of clusters."""
    return [f"a{i}" for i in range(n_clusters)]


def get_cluster_effect_names(partitions: list[GenericTypeFeatures]) -> list[str]:
    """Create a list of names for the cluster effects based on the partitions."""
    names = []
    for p in partitions:
        if isinstance(p, CategoricalFeatures):
            names.append(f"cluster_effect_{p.name}")
        elif isinstance(p, GaussianFeatures):
            names.append(f"cluster_effect_{p.name}_mean")
            names.append(f"cluster_effect_{p.name}_variance")
        elif isinstance(p, PoissonFeatures):
            names.append(f"cluster_effect_{p.name}_rate")
        else:
            raise ValueError("Only categorical partitions are currently supported.")

    return []


def write_samples(
    run: int,
    base_path: Path,
    samples: dict,
    data: Data,
    model: Model,
    match_clusters: bool = True
):
    """Write samples to files for post-analysis and visualisation."""
    clusters_samples_cont = np.array(samples['z'])
    clusters_samples = sample_categorical(clusters_samples_cont, binary_encoding=True)[:, :, :-1].transpose(0, 2, 1)
    n_samples, n_clusters, n_objects = clusters_samples.shape

    # cluster_eff_samples = np.array(samples['cluster_effect'])
    partitions = data.features.partitions

    # Cast all parameters from jax.numpy to numpy
    samples = {k: np.array(v) for k, v in samples.items()}

    if match_clusters:
        clusters_sum = np.zeros(clusters_samples.shape[1:], dtype=int)

        for i, clusters in enumerate(clusters_samples):
            # Compute best matching perm
            permutation = get_best_permutation(clusters, clusters_sum)

            # Permute clusters
            clusters = clusters[permutation]
            for param_name in get_cluster_effect_names(partitions):
                samples[param_name][i] = samples[param_name][i, permutation]

            # Update cluster_sum for matching future samples
            clusters_sum += clusters
            clusters_samples[i] = clusters

    cluster_names = get_cluster_names(n_clusters)
    feature_names = data.features.names
    n_states = max((p.n_states for p in data.features.categorical_partitions()), default=0)
    state_names = [f"s{s}" for s in range(n_states)]

    param_dfs_list = []

    cluster_sizes = np.sum(clusters_samples, axis=-1)

    sample_id_df = pd.DataFrame(data=np.arange(n_samples), columns=['Sample'])
    cluster_sizes_df = pd.DataFrame(data=cluster_sizes, columns=[f'size_{c}' for c in cluster_names])
    param_dfs_list += [sample_id_df, cluster_sizes_df]

    # Transform the weights samples to a data frame
    w_samples = np.array(samples['w'])  # shape: (n_samples, n_features, n_components)
    component_names = ['areal', *data.confounders.keys()]
    weights_df = samples_array_to_df(
        param_samples=w_samples.transpose((0, 2, 1)),
        names=[component_names, feature_names],
        prefix='w',
    )
    param_dfs_list.append(weights_df)

    # If using varying weights per cluster, add them to the data frame list
    if "w_cluster" in samples:
        w_cluster_samples = np.array(samples['w_cluster'])
        w_cluster_df = samples_array_to_df(
            param_samples=w_cluster_samples,
            names=[cluster_names, feature_names],
            prefix='w_cluster'
        )
        param_dfs_list.append(w_cluster_df)

    all_cluster_eff_dfs = []
    all_conf_eff_dfs = []
    for partition in data.features.partitions:
        if isinstance(partition, CategoricalFeatures):
            cluster_eff_df = samples_array_to_df(
                param_samples=samples[f'cluster_effect_{partition.name}'],
                names=[cluster_names, partition.names, state_names[:partition.n_states]],
                prefix='areal'
            )
            all_cluster_eff_dfs.append(cluster_eff_df)

            # Subset states for all confounding effects and place in data frames
            for i_c, conf in enumerate(data.confounders.values()):
                conf_eff_df = samples_array_to_df(
                    param_samples=samples[f"conf_effect_{i_c}_{partition.name}"],
                    names=[conf.group_names, partition.names, state_names[:partition.n_states]],
                    prefix=conf.name)
                all_conf_eff_dfs.append(conf_eff_df)

        elif isinstance(partition, GaussianFeatures):
            cluster_eff_mean_df = samples_array_to_df(
                param_samples=samples[f'cluster_effect_{partition.name}_mean'],
                names=[cluster_names, partition.names],
                prefix='areal',
                suffix='mean',
            )
            cluster_eff_variance_df = samples_array_to_df(
                param_samples=samples[f'cluster_effect_{partition.name}_variance'],
                names=[cluster_names, partition.names],
                prefix='areal',
                suffix='variance',
            )
            all_cluster_eff_dfs.append(cluster_eff_mean_df)
            all_cluster_eff_dfs.append(cluster_eff_variance_df)

            # Collect, reshape and place all confounding effects in data frames
            for i_c, conf in enumerate(data.confounders.values()):
                conf_eff_mean_df = samples_array_to_df(
                    param_samples=samples[f"conf_effect_{i_c}_{partition.name}_mean"],
                    names=[conf.group_names, partition.names],
                    prefix=conf.name,
                    suffix='mean'
                )
                all_conf_eff_dfs.append(conf_eff_mean_df)

                conf_eff_variance_df = samples_array_to_df(
                    param_samples=samples[f"conf_effect_{i_c}_{partition.name}_variance"],
                    names=[conf.group_names, partition.names],
                    prefix=conf.name,
                    suffix='variance'
                )
                all_conf_eff_dfs.append(conf_eff_variance_df)

        else:
            print(partition)
            raise NotImplementedError("Only categorical features are currently supported.")


    # Add cluster and confounding effects to the data frame list
    param_dfs_list += all_cluster_eff_dfs + all_conf_eff_dfs

    params_df = pd.concat(param_dfs_list, axis=1)


    if "potential_energy" in samples:
        params_df["potential_energy"] = samples["potential_energy"]

    if "z_concentration" in samples:
        params_df["z_concentration"] = samples["z_concentration"]

    if "w_cluster_concentration_0" in samples:
        params_df["w_cluster_concentration_0"] = samples["w_cluster_concentration_0"]
        params_df["w_cluster_concentration_1"] = samples["w_cluster_concentration_1"]

    clusters_path = base_path / f'clusters_K{n_clusters}_{run}.txt'
    clusters_continuous_path = base_path / f'clusters_K{n_clusters}_{run}.npy'
    params_path = base_path / f'stats_K{n_clusters}_{run}.txt'

    # Write clusters file
    with open(clusters_path, "w") as clusters_file:
        for clusters in clusters_samples:
            row = format_cluster_columns(clusters)
            clusters_file.write(row + "\n")

    # Write continuous cluster assignment file
    np.save(clusters_continuous_path, clusters_samples_cont)

    # Write params file
    with open(params_path, "w") as params_file:
        params_df.to_csv(params_file, sep='\t', index=False)

    # Write pointwise likelihoods to file
    if not model.config.sample_from_prior:
        likelihoods_by_partition = log_likelihood(model.get_model, samples)
        likelihoods_flat = np.empty((n_samples,) + data.features.all_features.shape)
        for p in data.features.partitions:
            likelihoods_flat[:, :, p.feature_indices] = likelihoods_by_partition[f"x_{p.name}"]

        # Create the likelihood array
        with tables.open_file(base_path / f'likelihood_K{n_clusters}_{run}.h5', mode="w") as lh_file:
            logged_likelihood_array = lh_file.create_earray(
                where=lh_file.root,
                name="likelihood",
                obj=likelihoods_flat.reshape(n_samples, -1),
                atom=tables.Float32Col(),
                filters=tables.Filters(
                    complevel=9, complib="blosc:zlib", bitshuffle=True, fletcher32=True
                ),
                shape=(0, n_objects * data.features.n_features),
            )
            logged_likelihood_array.close()

            na_array = lh_file.create_carray(
                where=lh_file.root,
                name="na_values",
                obj=data.features.missing.ravel(),
                atom=tables.BoolCol(),
                filters=tables.Filters(complevel=9, fletcher32=True),
                shape=(n_objects * data.features.n_features,),
            )
            na_array.close()


def samples_array_to_df(
    param_samples: np.ndarray,
    names: list[list[str]],
    prefix: str = "",
    suffix: str = "",
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
    if suffix:
        column_names = [f"{col}_{suffix}" for col in column_names]

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


class OnlineLogger:

    def __init__(
        self,
        path: str,
        data: Data,
        model: Model,
        resume: bool,
    ):
        self.path: Path = Path(path)
        self.state_path = self.path.parent / "state.pkl"
        self.data: Data = data
        self.model: Model = model.__copy__()

        self.file = None
        self.column_names = None

        self.resume = resume

    @abstractmethod
    def write_header(self, sample: dict):
        pass

    @abstractmethod
    def _write_sample(self, sample: dict, **write_args):
        pass

    def write_sample(self, sample: dict, **write_args):
        if self.file is None:
            self.open()
            self.write_header(sample)
        self._write_sample(sample, **write_args)

    def dump_state(self, state):
        with open(self.state_path, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self):
        with open(self.state_path, 'rb') as f:
            return pickle.load(f)

    def open(self):
        self.file = open(self.path, "a" if self.resume else "w", buffering=1)
        # ´buffering=1´ activates line-buffering, i.e. flushing to file after each line

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

def numpy_to_tables_dtype(dtype):
    """Convert numpy dtype to tables dtype."""
    if np.issubdtype(dtype, np.integer):
        return tables.Int32Col()
    elif np.issubdtype(dtype, np.floating):
        return tables.Float32Col()
    elif np.issubdtype(dtype, np.bool_):
        return tables.BoolCol()
    else:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")


class OnlineSampleLogger(OnlineLogger):

    """The OnlineSampleLogger continually writes the samples to a pytables file (.h5)."""

    def __init__(self, *args, **kwargs):
        self.logged_likelihood_array = None
        super().__init__(*args, **kwargs)

    def open(self):
        if self.resume:
            try:
                self.file = tables.open_file(self.path, mode="a")

            except tables.exceptions.HDF5ExtError as e:
                logging.warning(f"Could not append to existing sample file '{self.path.name}' ({type(e).__name__})."
                                f" Overwriting previous samples.")
                # Set resume to False and open again
                self.resume = False
                self.open()

        else:
            self.file = tables.open_file(self.path, mode="w")

    def write_header(self, sample: dict):
        if self.resume:
            return

        for param_name, param_value in sample.items():
            # Create the likelihood array
            n_chains, n_samples, *param_dims = param_value.shape
            self.file.create_earray(
                where=self.file.root,
                name=param_name,
                atom=numpy_to_tables_dtype(np.array(param_value).dtype),
                filters=tables.Filters(complevel=5),
                shape=(n_chains, 0, *param_dims),
            )

    def _write_sample(self, sample: dict, **write_args):
        # n_new_samples = write_args.get("n_new_samples", 1)
        for param_name, param_values in sample.items():
            # Write the new samples to the file
            self.file.root[param_name].append(np.array(param_values))
        self.file.flush()

    def read_samples(self) -> dict[str, np.ndarray]:
        """Read the samples from the HDF file and return them as a dictionary."""
        samples = {}
        for param in self.file.root:
            # Swap axes from (n_samples, n_chains,...) to (n_chains, n_samples, ...)
            param_array = np.array(param).swapaxes(0, 1)

            # Place the paramter samples in the dictionary
            samples[param._v_name] = param_array

        return samples

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
