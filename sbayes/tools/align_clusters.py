import re
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


def find_k_values(path: Path) -> list[int]:
    """Find all K values where K{K} is a subdirectory of path."""
    k_values = []
    for d in sorted(path.iterdir()):
        m = re.fullmatch(r'K(\d+)', d.name)
        if d.is_dir() and m:
            k_values.append(int(m.group(1)))
    return k_values


def find_run_ids(path: Path, K: int) -> list[int]:
    """Find all run IDs present in the K{K} directory."""
    k_dir = path / f'K{K}'
    run_ids = []
    for f in sorted(k_dir.iterdir()):
        m = re.fullmatch(rf'clusters_K{K}_(\d+)\.txt', f.name)
        if m:
            run_ids.append(int(m.group(1)))
    return run_ids


def align_run(path1: Path, path2: Path, K: int, run1: int, run2: int):
    """Align clusters of run2 to run1 for a given K."""
    clusters_path_1 = path1 / f'K{K}' / f'clusters_K{K}_{run1}.txt'
    parameters_path_1 = path1 / f'K{K}' / f'stats_K{K}_{run1}.txt'
    clusters_path_2 = path2 / f'K{K}' / f'clusters_K{K}_{run2}.txt'
    parameters_path_2 = path2 / f'K{K}' / f'stats_K{K}_{run2}.txt'
    clusters_path_2_out = path2 / f'K{K}' / f'clusters_K{K}_{run2}.aligned.txt'
    parameters_path_2_out = path2 / f'K{K}' / f'stats_K{K}_{run2}.aligned.txt'

    results_1 = Results.from_csv_files(clusters_path_1, parameters_path_1, burn_in=0)
    results_2 = Results.from_csv_files(clusters_path_2, parameters_path_2, burn_in=0)

    mean_clusters_1 = np.mean(results_1.clusters, axis=1)
    mean_clusters_2 = np.mean(results_2.clusters, axis=1)
    d = cluster_agreement(mean_clusters_1, mean_clusters_2)
    perm = linear_sum_assignment(d, maximize=True)[1]

    # Permute clusters along axis 0 (n_clusters): (n_clusters, n_samples, n_sites)
    clusters_2_aligned = results_2.clusters[perm]
    params_2_aligned = get_permuted_params(results_2, perm)

    # Save aligned .npy if the source was loaded from .npy
    npy_path_2 = clusters_path_2.with_suffix('.npy')
    if npy_path_2.exists():
        # Convert (n_clusters, n_samples, n_sites) → (n_samples, n_sites, n_clusters+1)
        clusters_npy = np.transpose(clusters_2_aligned, (1, 2, 0))
        not_assigned = 1.0 - clusters_npy.sum(axis=-1, keepdims=True)
        clusters_npy = np.concatenate([clusters_npy, not_assigned], axis=-1)
        np.save(clusters_path_2_out.with_suffix('.npy'), clusters_npy)

    # Save aligned .txt (convert to binary if data is continuous)
    clusters_2_txt = clusters_2_aligned.transpose((1, 0, 2))
    if np.issubdtype(clusters_2_txt.dtype, np.floating):
        clusters_2_txt = (clusters_2_txt > 0.5).astype(bool)
    write_clusters(clusters_path_2_out, clusters_2_txt)

    params_2_aligned.to_csv(parameters_path_2_out, index=False, sep="\t")


def merge_runs(path1: Path, path2: Path, K: int, run1: int, run2_values: list[int]):
    """Merge clusters and parameters from run1 and all aligned run2s into single files."""
    k_dir1 = path1 / f'K{K}'
    k_dir2 = path2 / f'K{K}'

    # Merge clusters (.txt: binary, one sample per line)
    all_cluster_lines = []
    with open(k_dir1 / f'clusters_K{K}_{run1}.txt', 'r') as f:
        all_cluster_lines.extend(f.readlines())
    for run2 in run2_values:
        with open(k_dir2 / f'clusters_K{K}_{run2}.aligned.txt', 'r') as f:
            all_cluster_lines.extend(f.readlines())
    with open(k_dir2 / f'clusters_K{K}_merged.aligned.txt', 'w') as f:
        f.writelines(all_cluster_lines)

    # Merge clusters (.npy: continuous assignments, if available)
    npy_run1 = k_dir1 / f'clusters_K{K}_{run1}.npy'
    if npy_run1.exists():
        npy_arrays = [np.load(npy_run1)]
        for run2 in run2_values:
            npy_arrays.append(np.load(k_dir2 / f'clusters_K{K}_{run2}.aligned.npy'))
        merged_npy = np.concatenate(npy_arrays, axis=0)
        np.save(k_dir2 / f'clusters_K{K}_merged.aligned.npy', merged_npy)

    # Merge parameters (TSV with header)
    params_dfs = [pd.read_csv(k_dir1 / f'stats_K{K}_{run1}.txt', sep='\t')]
    for run2 in run2_values:
        params_dfs.append(pd.read_csv(k_dir2 / f'stats_K{K}_{run2}.aligned.txt', sep='\t'))
    merged_params = pd.concat(params_dfs, ignore_index=True)
    merged_params.to_csv(k_dir2 / f'stats_K{K}_merged.aligned.txt', index=False, sep='\t')


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Align clusters in logs of two sBayes runs.")
    parser.add_argument("-k", type=int, default=None)
    parser.add_argument("--merge", action="store_true",
                        help="After aligning, merge all runs into single results files.")
    parser.add_argument("path1", type=Path)
    parser.add_argument("run1", type=int, nargs="?", default=0)
    parser.add_argument("path2", type=Path, nargs="?", default=None)
    parser.add_argument("run2", type=int, nargs="?", default=None)
    args = parser.parse_args()

    path2 = args.path2 if args.path2 is not None else args.path1

    # Determine K values to process
    if args.k is not None:
        k_values = [args.k]
    else:
        k_values = find_k_values(args.path1)
        if not k_values:
            print(f"No K* subdirectories found in {args.path1}")
            return

    for K in k_values:
        # Determine run2 values to process
        if args.run2 is not None:
            run2_values = [args.run2]
        else:
            all_runs = find_run_ids(path2, K)
            run2_values = [r for r in all_runs if r != args.run1]
            if not run2_values:
                print(f"K{K}: No other runs found to align (run1={args.run1})")
                continue

        for run2 in run2_values:
            print(f"Aligning K={K}, run {run2} to run {args.run1}...")
            align_run(args.path1, path2, K, args.run1, run2)

        if args.merge:
            print(f"Merging K={K}, runs [{args.run1}, {', '.join(str(r) for r in run2_values)}]...")
            merge_runs(args.path1, path2, K, args.run1, run2_values)


if __name__ == '__main__':
    main()