from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from sbayes.results import Results
from sbayes.util import parse_cluster_columns, format_cluster_columns
from sbayes.tools.align_clusters import load_clusters, write_clusters, cluster_agreement


def permute_cluster_params(params: pd.DataFrame, cluster_names: list, permutation: NDArray[int]) -> pd.DataFrame:
    cluster_names = np.array(cluster_names)
    remap = {}
    for clust_i, clust_j in zip(cluster_names, cluster_names[permutation]):
        # Fix areal effects columns
        prefix_i = f"areal_{clust_i}_"
        prefix_j = f"areal_{clust_j}_"
        for k in params.columns:
            if k.startswith(prefix_i):
                k_j = prefix_j + k[len(prefix_i):]
                remap[k] = params[k_j].copy()

    for i, j in enumerate(permutation):
        # Fix cluster size columns
        remap[f"size_a{i}"] = params[f"size_a{j}"]

    for k_old, params_k_new in remap.items():
        params[k_old] = params_k_new

    return params


def align_clusters(clusters, params, cluster_names: list[str]):
    sum_clusters = np.mean(clusters[:, :20, :], axis=1)
    for i_s in range(clusters.shape[1]):
        d = cluster_agreement(sum_clusters, clusters[:, i_s])
        perm = linear_sum_assignment(d, maximize=True)[1]
        if not np.all(np.diff(perm) == 1):
            # Rearrange clusters
            clusters[:, i_s:] = clusters[perm, i_s:]

            # Rearrange parameters
            permuted_params = permute_cluster_params(params.copy(), cluster_names, perm)
            params = pd.concat([
                params.iloc[:i_s, :],
                permuted_params.iloc[i_s:, :],
            ], axis=0)

        sum_clusters += clusters[:, i_s]

    return clusters, params


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Align clusters in logs of two sBayes runs.")
    parser.add_argument("path", type=Path)
    parser.add_argument("k", type=int)
    parser.add_argument("run", type=int, nargs="?", default=0)
    args = parser.parse_args()
    K = args.k

    # Define paths
    clusters_path = args.path / f'K{K}' / f'clusters_K{K}_{args.run}.txt'
    parameters_path = args.path / f'K{K}' / f'stats_K{K}_{args.run}.txt'
    clusters_path_out = args.path / f'K{K}' / f'clusters_K{K}_{args.run}.aligned.txt'
    parameters_path_out = args.path / f'K{K}' / f'stats_K{K}_{args.run}.aligned.txt'

    # Load results
    results = Results.from_csv_files(clusters_path, parameters_path, burn_in=0)
    clusters, params = align_clusters(results.clusters, results.parameters, results.cluster_names)

    write_clusters(clusters_path_out, clusters.transpose((1,0,2)))
    params.to_csv(parameters_path_out, index=False, sep="\t")


if __name__ == '__main__':
    main()