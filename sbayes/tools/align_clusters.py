from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from sbayes.results import Results
from sbayes.util import parse_cluster_columns, normalize, format_cluster_columns


def load_clusters(filename=None) -> NDArray[int]:  # shape: (n_samples, n_clusters, n_objects)
    if filename is None:
        # Import tkinter only when needed (CLI works without it)
        import tkinter as tk
        from tkinter import filedialog

        tk.Tk().withdraw()
        filename = filedialog.askopenfilename(title='Select clusters file.', initialdir='../experiments/',
                                              filetypes=(('txt', '*.txt'),('all', '*.*')))
        print('Loading aras from selected file at:', filename)

    with open(filename, 'r') as clusters_file:
        clusters = []
        for line in clusters_file:
            clusters.append(parse_cluster_columns(line.strip()))

    return np.array(clusters, dtype=int)


def write_clusters(filename, cluster_samples):
    with open(filename, 'w') as clusters_file:
        clusters_file.writelines(
            format_cluster_columns(sample) + "\n" for sample in cluster_samples
        )


def cluster_agreement(a1, a2):
    return np.matmul(a1, a2.T)


def get_permuted_params(results: Results, permutation: list) -> pd.DataFrame:
    params: pd.DataFrame = results.parameters
    cluster_names = np.array(results.cluster_names)
    remap = {}
    for clust_i, clust_j in zip(cluster_names, cluster_names[permutation]):
        # Fix areal effects columns
        prefix_i = f"areal_{clust_i}_"
        prefix_j = f"areal_{clust_j}_"
        for k in params.columns:
            if k.startswith(prefix_i):
                k_j = prefix_j + k[len(prefix_i):]
                remap[k] = params[k_j]

    for i, j in enumerate(permutation):
        # Fix cluster size columns
        remap[f"size_a{i}"] = params[f"size_a{j}"]

    for k_old, params_k_new in remap.items():
        params[k_old] = params_k_new

    return params


def main():
    from shutil import copyfile
    import argparse

    parser = argparse.ArgumentParser(description="Align clusters in logs of two sBayes runs.")
    parser.add_argument("-k", type=int)
    parser.add_argument("path1", type=Path)
    parser.add_argument("run1", type=int, nargs="?", default=0)
    parser.add_argument("path2", type=Path, nargs="?", default=None)
    parser.add_argument("run2", type=int, nargs="?", default=1)
    args = parser.parse_args()
    K = args.k

    # Define paths
    clusters_path_1 = args.path1 / f'K{K}' / f'clusters_K{K}_{args.run1}.txt'
    parameters_path_1 = args.path1 / f'K{K}' / f'stats_K{K}_{args.run1}.txt'
    path2 = args.path2 if args.path2 is not None else args.path1
    clusters_path_2 = path2 / f'K{K}' / f'clusters_K{K}_{args.run2}.txt'
    parameters_path_2 = path2 / f'K{K}' / f'stats_K{K}_{args.run2}.txt'
    clusters_path_2_out = path2 / f'K{K}' / f'clusters_K{K}_{args.run2}.aligned.txt'
    parameters_path_2_out = path2 / f'K{K}' / f'clusters_K{K}_{args.run2}.aligned.txt'

    # Load results
    results_1 = Results.from_csv_files(clusters_path_1, parameters_path_1, burn_in=0)
    results_2 = Results.from_csv_files(clusters_path_2, parameters_path_2, burn_in=0)

    # Compute the best permutation
    mean_clusters_1 = np.mean(results_1.clusters, axis=1)
    mean_clusters_2 = np.mean(results_2.clusters, axis=1)
    d = cluster_agreement(mean_clusters_1, mean_clusters_2)
    perm = linear_sum_assignment(d, maximize=True)[1]

    # Permute the clusters and parameters
    clusters_2_aligned = results_2.clusters[perm].transpose((1,0,2))
    params_2_aligned = get_permuted_params(results_2, perm)

    write_clusters(clusters_path_2_out, clusters_2_aligned)
    params_2_aligned.to_csv(parameters_path_2_out, index=False, sep="\t")


if __name__ == '__main__':
    main()