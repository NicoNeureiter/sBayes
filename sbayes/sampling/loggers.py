from __future__ import annotations

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from sbayes.load_data import Data
from sbayes.preprocessing import sample_categorical
from sbayes.util import format_cluster_columns, get_best_permutation, normalize
from sbayes.model import Model

def get_cluster_names(n_clusters: int) -> list[str]:
    """Create a list of names for the clusters based on the number of clusters."""
    return [f"a{i}" for i in range(n_clusters)]


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
    for partition in data.partitions:
        feat_idxs = partition.feature_indices

        # Subset states for cluster effects and place in data frame
        cluster_eff_df = samples_array_to_df(
            param_samples=cluster_eff_samples[:, :, feat_idxs, :partition.n_states],
            names=[cluster_names, feature_names[feat_idxs], state_names[:partition.n_states]],
            prefix='areal'
        )
        all_cluster_eff_dfs.append(cluster_eff_df)

        # Subset states for all confounding effects and place in data frames
        for i_c, conf in enumerate(data.confounders.values()):
            conf_eff_samples = np.array(samples[f"conf_eff_{i_c}_{partition.name}"])[:, :-1, :, :]

            conf_eff_df = samples_array_to_df(
                param_samples=conf_eff_samples,
                names=[conf.group_names, feature_names[feat_idxs], state_names[:partition.n_states]],
                prefix=conf.name)
            all_conf_eff_dfs.append(conf_eff_df)

    # Add cluster and confounding effects to the data frame list
    param_dfs_list += all_cluster_eff_dfs + all_conf_eff_dfs

    params_df = pd.concat(param_dfs_list, axis=1)


    if "potential_energy" in samples:
        params_df["potential_energy"] = samples["potential_energy"]

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
