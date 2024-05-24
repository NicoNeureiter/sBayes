# Read the simulated parameters
# Read the estimated parameters
# Plot diagonal line
# 100 times, out of which 91–99 should yield a credible interval (grey/red bars)
# covering the simulated value (diagonal) in a well-calibrated mode
import numpy as np
import yaml
import pandas as pd
import os
import matplotlib.pyplot as plt
from sbayes.load_data import Objects, Confounder
from typing import Union, Sequence
from pathlib import Path
PathLike = Union[str, Path]


def get_relevant_parameter(d, keys):
    for key in keys:
        d = d[key]
    return d


def sort_by_number(name):
    return int(name.split('_')[-1])


def get_folders(sim_path: str, results_path: str, model_n: str):
    """Get all folders with simulated and inferred parameters

     Args:
         sim_path: The folder name with the simulated parameters
         results_path: The results folder name with the inferred parameters
         model_n: relevant folder in the results folder
     Returns:
         dictionary mapping feature names to corresponding weights arrays
     """

    param_path = "/".join([sim_path, "parameters"])
    stats_files = dict(simulated=[],
                       inferred=[])
    # Iterate over all items in the sims folder
    sims = os.listdir(param_path)

    # Open the items in the sims folder and the results in the results folder
    for i in sorted(sims, key=sort_by_number):

        sim_path_folder = os.path.join(param_path, i)
        results_path_folder = os.path.join(sim_path, "results", i, results_path, model_n)
        print(results_path_folder, "res")
        # Check if it's a directory
        if os.path.isdir(sim_path_folder):
            sim_path_file = os.path.join(sim_path_folder, "stats_sim.txt")
            if os.path.exists(sim_path_file):
                stats_files['simulated'].append(sim_path_file)

        if os.path.isdir(results_path_folder):

            results_path_file = os.path.join(results_path_folder, "_".join(["stats", model_n, "0.txt"]))

            if os.path.exists(results_path_file):
                stats_files['inferred'].append(results_path_file)

    return stats_files


def read_meta(path):
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def parse_parameters(params_raw: dict, feat: dict, clust: list, conf: dict) -> dict:
    """Parse weights array for each feature in a dictionary from the parameters
    data-frame.

    Args:
        params_raw: The simulated and inferred parameters of the model in raw format
        feat: The feature names
        clust: The name of the clusters
        conf: The name of the confounders and groups per confounder
    Returns:
        dictionary mapping feature names to corresponding weights arrays
    """
    # The components include the cluster effect and all confounding effects and define
    # the dimensions of weights for each feature.

    components = ["cluster"] + list(conf.keys())
    # components = list(conf.keys())

    # Collect parameters across all simulations by feature
    params = dict(simulated=dict(weights={},
                                 cluster_effect={},
                                 confounding_effects={}),
                  inferred=dict(weights={},
                                cluster_effect={},
                                confounding_effects={}))
    # Weights
    for ft in feat.keys():
        params['simulated']['weights'][ft] = {}
        params['inferred']['weights'][ft] = {}
        for f in feat[ft]:
            params['simulated']['weights'][ft][f] = {}
            params['inferred']['weights'][ft][f] = {}
            for c in components:
                params['simulated']['weights'][ft][f][c] = np.column_stack(
                    [params_raw['simulated'][p][f"w_{c}_{f}"].to_numpy(dtype=float)
                     for p in range(len(params_raw['simulated']))]
                ).flatten()

                params['inferred']['weights'][ft][f][c] = np.row_stack(
                    [params_raw['inferred'][p][f"w_{c}_{f}"].to_numpy(dtype=float)
                     for p in range(len(params_raw['inferred']))]
                )

    # Cluster effect
    for ft in feat.keys():
        params['simulated']['cluster_effect'][ft] = {}
        params['inferred']['cluster_effect'][ft] = {}

        for cl in clust:
            params['simulated']['cluster_effect'][ft][cl] = {}
            params['inferred']['cluster_effect'][ft][cl] = {}
            if ft == 'categorical':
                for f in feat[ft].keys():
                    params['simulated']['cluster_effect'][ft][cl][f] = {}
                    params['inferred']['cluster_effect'][ft][cl][f] = {}
                    for s in feat[ft][f]:
                        params['simulated']['cluster_effect'][ft][cl][f][s] = np.column_stack(
                            [params_raw['simulated'][p][f"cluster_{cl}_{f}_{s}"].to_numpy(dtype=float)
                             for p in range(len(params_raw['simulated']))]
                        ).flatten()
                        params['inferred']['cluster_effect'][ft][cl][f][s] = np.row_stack(
                            [params_raw['inferred'][p][f"cluster_{cl}_{f}_{s}"].to_numpy(dtype=float)
                             for p in range(len(params_raw['inferred']))])

            if ft == 'gaussian':
                for f in feat[ft]:
                    params['simulated']['cluster_effect'][ft][cl][f] = {}
                    params['inferred']['cluster_effect'][ft][cl][f] = {}
                    params['simulated']['cluster_effect'][ft][cl][f]['mu'] = np.column_stack(
                            [params_raw['simulated'][p][f"cluster_{cl}_{f}_mu"].to_numpy(dtype=float)
                             for p in range(len(params_raw['simulated']))]
                        ).flatten()
                    params['simulated']['cluster_effect'][ft][cl][f]['sigma'] = np.row_stack(
                        [params_raw['inferred'][p][f"cluster_{cl}_{f}_sigma"].to_numpy(dtype=float)
                         for p in range(len(params_raw['inferred']))]
                    ).flatten()

                    params['inferred']['cluster_effect'][ft][cl][f]['mu'] = np.row_stack(
                            [params_raw['inferred'][p][f"cluster_{cl}_{f}_mu"].to_numpy(dtype=float)
                             for p in range(len(params_raw['inferred']))])

                    params['inferred']['cluster_effect'][ft][cl][f]['sigma'] = np.row_stack(
                            [params_raw['inferred'][p][f"cluster_{cl}_{f}_sigma"].to_numpy(dtype=float)
                             for p in range(len(params_raw['inferred']))])

            if ft == 'poisson':
                for f in feat[ft]:
                    params['simulated']['cluster_effect'][ft][cl][f] = {}
                    params['simulated']['cluster_effect'][ft][cl][f]['rate'] = np.column_stack(
                        [params_raw['simulated'][p][f"cluster_{cl}_{f}_rate"].to_numpy(dtype=float)
                         for p in range(len(params_raw['simulated']))]
                    ).flatten()

                    # todo: read the actual results
                    params['inferred']['cluster_effect'][ft][cl][f]['rate'] = \
                        params['simulated']['cluster_effect'][ft][cl][f]['rate']

    # Collect confounding effects by feature
    for ft in feat.keys():
        params['simulated']['confounding_effects'][ft] = {}
        params['inferred']['confounding_effects'][ft] = {}
        for c in conf.keys():
            params['simulated']['confounding_effects'][ft][c] = {}
            params['inferred']['confounding_effects'][ft][c] = {}
            for g in conf[c]:
                params['simulated']['confounding_effects'][ft][c][g] = {}
                params['inferred']['confounding_effects'][ft][c][g] = {}
                if ft == 'categorical':
                    for f in feat[ft].keys():
                        params['simulated']['confounding_effects'][ft][c][g][f] = {}
                        params['inferred']['confounding_effects'][ft][c][g][f] = {}
                        for s in feat[ft][f]:
                            params['simulated']['confounding_effects'][ft][c][g][f][s] = np.column_stack(
                                [params_raw['simulated'][p][f"{c}_{g}_{f}_{s}"].to_numpy(dtype=float)
                                 for p in range(len(params_raw['simulated']))]
                            ).flatten()
                            params['inferred']['confounding_effects'][ft][c][g][f][s] = np.row_stack(
                                [params_raw['inferred'][p][f"{c}_{g}_{f}_{s}"].to_numpy(dtype=float)
                                 for p in range(len(params_raw['inferred']))]
                            )

                if ft == 'gaussian':
                    for f in feat[ft]:
                        params['simulated']['confounding_effects'][ft][c][g][f] = {}
                        params['inferred']['confounding_effects'][ft][c][g][f] = {}

                        params['simulated']['confounding_effects'][ft][c][g][f]['mu'] = np.column_stack(
                            [params_raw['simulated'][p][f"{c}_{g}_{f}_mu"].to_numpy(dtype=float)
                             for p in range(len(params_raw['simulated']))]
                        ).flatten()

                        params['simulated']['confounding_effects'][ft][c][g][f]['sigma'] = np.column_stack(
                            [params_raw['simulated'][p][f"{c}_{g}_{f}_sigma"].to_numpy(dtype=float)
                             for p in range(len(params_raw['simulated']))]
                        ).flatten()

                        # todo: read the actual data
                        params['inferred']['confounding_effects'][ft][c][g][f]['mu'] = np.row_stack(
                            [params_raw['inferred'][p][f"{c}_{g}_{f}_mu"].to_numpy(dtype=float)
                             for p in range(len(params_raw['inferred']))]
                        )
                        params['inferred']['confounding_effects'][ft][c][g][f]['sigma'] = np.row_stack(
                            [params_raw['inferred'][p][f"{c}_{g}_{f}_sigma"].to_numpy(dtype=float)
                             for p in range(len(params_raw['inferred']))]
                        )

                if ft == 'poisson':
                    for f in feat[ft]:
                        params['simulated']['confounding_effects'][ft][c][g][f] = {}
                        params['inferred']['confounding_effects'][ft][c][g][f] = {}
                        params['simulated']['confounding_effects'][ft][c][g][f]['rate'] = np.column_stack(
                            [params_raw['simulated'][p][f"{c}_{g}_{f}_rate"].to_numpy(dtype=float)
                             for p in range(len(params_raw['simulated']))]
                        ).flatten()
                        params['inferred']['confounding_effects'][ft][c][g][f]['rate'] = \
                            params['simulated']['confounding_effects'][ft][c][g][f]['rate']
    return params


def plot_simulated_against_inferred(sim, inf, title):
    """
    Plots elements of sim on the x-axis against all elements of the inf on the y-axis.

    Parameters:
        sim (numpy.ndarray): Array with shape (n,)
        inf (numpy.ndarray): Array with shape (n, m)
        title (str): Plot title
    Returns:
        None
    """
    plt.figure(figsize=(8, 6))

    low_perc = [np.percentile(i, 5) for i in inf]
    high_perc = [np.percentile(i, 95) for i in inf]

    in_perc = [low_perc[s] < sim[s] < high_perc[s] for s in range(len(sim))]

    min_val = min(np.concatenate((inf.flatten(), sim)))
    max_val = max(np.concatenate((inf.flatten(), sim)))

    for i in range(len(sim)):
        color = 'grey' if in_perc[i] else 'red'
        plt.plot([sim[i]] * inf.shape[1], inf[i], 'o', markersize=3, color=color)

    success_text = " / ".join([str(sum(in_perc)), str(len(in_perc))])

    plt.axline((0, 0), slope=1, color='black')
    plt.text(0.92, 0.05, success_text,
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes)

    plt.xlabel('simulated')
    plt.ylabel('estimated')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(False)
    plt.title(title)
    plt.show()


sim_path = "../../experiments/simulation/2024-05-24_13-27_gaussian"
results_path = "2024-05-24_13-41"
meta = read_meta("/".join([sim_path, "meta/meta_sim.yaml"]))
n_clusters = len(meta['clusters'].keys())


parameter_files = get_folders(sim_path=sim_path,
                              results_path=results_path,
                              model_n="".join(["K", str(n_clusters)]))

parameters_raw = dict(simulated=[pd.read_csv(f, **{"delimiter": "\t"}) for f in parameter_files['simulated']],
                      inferred=[pd.read_csv(f, **{"delimiter": "\t"}) for f in parameter_files['inferred']])

clusters = [str(i) for i in range(len(list(meta['clusters'].keys())))]

confounders = {conf: list(groups.keys()) for conf, groups in meta['confounders'].items()}
features = meta['names']

parameters = parse_parameters(parameters_raw, features, clusters, confounders)

parameter_keys = ['cluster_effect', 'gaussian', '0', 'f1', 'mu']

plot_simulated_against_inferred(get_relevant_parameter(parameters['simulated'], parameter_keys),
                                get_relevant_parameter(parameters['inferred'], parameter_keys),
                                "Mean of Gaussian feature f1 in cluster 0")
