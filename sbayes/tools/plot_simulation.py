# Read the simulated parameters
# Read the estimated parameters
# Plot diagonal line
# 100 times, out of which 91–99 should yield a credible interval (grey/red bars)
# covering the simulated value (diagonal) in a well-calibrated mode
import dataclasses

import numpy as np
import yaml
import pandas as pd
import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from numpy._typing import NDArray

from sbayes.results import Results


def get_relevant_parameter(d, keys):
    for key in keys:
        d = d[key]
    return d


def sort_by_number(name):
    return int(name.split('_')[-1])


def get_folders(sim_path: Path, model_n: str):
    """Get all folders with simulated and inferred parameters

     Args:
         sim_path: The folder name with the simulated parameters
         model_n: relevant folder in the results folder
     Returns:
         dictionary mapping feature names to corresponding weights arrays
     """

    param_path = sim_path / "parameters"
    stats_files = dict(simulated=[],
                       inferred=[])
    # Iterate over all items in the sims folder
    sims = os.listdir(param_path)

    # Open the items in the sims folder and the results in the results folder
    for i in sorted(sims, key=sort_by_number):

        sim_path_folder = param_path / i
        results_path_folder = sim_path / "results" / i / model_n

        # Check if it's a directory
        if os.path.isdir(sim_path_folder):
            sim_path_file = sim_path_folder / "stats_sim.txt"
            if os.path.exists(sim_path_file):
                stats_files['simulated'].append(sim_path_file)

        if os.path.isdir(results_path_folder):

            results_path_file = results_path_folder / f"stats_{model_n}_0.txt"

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


def plot_simulated_against_inferred(
    sim: NDArray,
    inf: NDArray,
    title: str | None = None,
    ax = None):
    """
    Plots elements of sim on the x-axis against all elements of the inf on the y-axis.

    Parameters:
        sim: Array with shape (n,)
        inf: Array with shape (n, m)
        title: Plot title
    Returns:
        None
    """
    ax = ax or plt.gca()

    low_perc = [np.percentile(i, 5) for i in inf]
    high_perc = [np.percentile(i, 95) for i in inf]

    in_perc = [low_perc[s] < sim[s] < high_perc[s] for s in range(len(sim))]

    min_val = min(np.concatenate((inf.flatten(), sim)))
    max_val = max(np.concatenate((inf.flatten(), sim)))

    for i in range(len(sim)):
        color = 'grey' if in_perc[i] else 'red'
        ax.plot([sim[i]] * inf.shape[1], inf[i], 'o', markersize=2, alpha=0.3, color=color)

    success_text = f"{sum(in_perc)} / {len(in_perc)}"

    ax.axline((0, 0), slope=1, color='black')
    ax.text(0.99, 0.01, success_text,
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=ax.transAxes)

    ax.set_xlabel('simulated')
    ax.set_ylabel('estimated')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(False)
    if title:
        ax.set_title(title)

def my_tight_layout(fig, axes):
    bottom = 0.2 / axes.shape[0]
    left = 0.2 / axes.shape[1]
    fig.subplots_adjust(
        left=left,
        bottom=bottom,
        right=1 - left,
        top=1 - bottom,
        wspace=0.25,
        hspace=0.4
    )


def plot_simulated_against_inferred_grid(
    simulated: list[Results],
    inferred: list[Results],
    parameter: str,
    save_to_path: Path | None = None,
):
    """
    Plots elements of sim on the x-axis against all elements of the inf on the y-axis.

    Parameters:
        simulated: List of `n` Results objects for the simulated areas and parameters
        inferred: List of `n` Results objects for `m` samples of inferred areas and parameters
        parameter: Which parameter to plot (weights, confounding_effects, cluster_effects)
        save_to_path: Optional path to save the figure (otherwise it will be shown on the screen immediately)
    """
    n_features = simulated[0].n_features
    n_components = simulated[0].n_confounders + 1
    if parameter == 'weights':
        fig, axes = plt.subplots(nrows=n_features, ncols=n_components,
                                 figsize=(6 * n_components, 5 * n_features))
        for i, f in enumerate(simulated[0].weights.keys()):
            for j, component in enumerate(simulated[0].component_names):
                sim = np.array([r.weights[f][0, j] for r in simulated])
                inf = np.array([r.weights[f][:, j] for r in inferred])
                plot_simulated_against_inferred(sim, inf, ax=axes[i, j],
                                                  title=f'Mixture weight of `{component}` for feature `{f}`')

        my_tight_layout(fig, axes)
        plt.savefig(save_to_path)

    elif parameter == 'cluster_effects':
        structure = simulated[0].cluster_effect
        feature_names = simulated[0].feature_names
        feature_states = simulated[0].feature_states

        cluster_names = list(structure.keys())
        n_clusters = len(cluster_names)

        fig, axes = plt.subplots(nrows=n_features, ncols=n_clusters,
                                 figsize=(6 * n_clusters, 5 * n_features), squeeze=False)

        for i_g, g in enumerate(cluster_names):
            for i_f, f in enumerate(feature_names):
                # For now we just always plot the distribution of the first state
                # TODO Include state as a dimension in the grid?
                s = feature_states[i_f][0]
                sim = np.array([r.cluster_effect[g][f][0, 0] for r in simulated])
                inf = np.array([r.cluster_effect[g][f][:, 0] for r in inferred])
                plot_simulated_against_inferred(
                    sim, inf, ax=axes[i_f, i_g],
                    title=f'Cluster effect of cluster {g}, feature {f}, state {s}'
                )

        my_tight_layout(fig, axes)
        plt.savefig(save_to_path)

    elif parameter == 'confounding_effects':
        structure = simulated[0].confounding_effects
        feature_names = simulated[0].feature_names
        feature_states = simulated[0].feature_states
        confounder_names = list(structure.keys())
        for c in confounder_names:
            group_names = structure[c].keys()
            n_groups = len(group_names)

            # n_states_max = max(len(states) for states in feature_states)
            fig, axes = plt.subplots(nrows=n_features, ncols=n_groups,
                                     figsize=(6 * n_groups, 5 * n_features), squeeze=False)

            for i_g, g in enumerate(group_names):
                for i_f, f in enumerate(feature_names):
                    # For now we just always plot the distribution of the first state
                    # TODO Include state as a dimension in the grid?
                    s = feature_states[i_f][0]
                    sim = np.array([r.confounding_effects[c][g][f][0, 0] for r in simulated])
                    inf = np.array([r.confounding_effects[c][g][f][:, 0] for r in inferred])
                    plot_simulated_against_inferred(
                        sim, inf, ax=axes[i_f, i_g],
                        title=f'Confounding effect of group {g}, feature {f}, state {s}'
                    )

            my_tight_layout(fig, axes)
            plt.savefig(save_to_path.with_suffix(f'.{c}.pdf'))

    elif parameter == 'clusters':
        n_sims = len(simulated)
        n_clusters = simulated[0].n_clusters
        cluster_posterior = np.array([
            r.cluster_posterior() for r in simulated
        ])  # shape: (n_sims, n_clusters, n_objects)
        cluster_truth = np.array([
            r.clusters[:, 0, :] for r in inferred
        ])  # shape: (n_sims, n_clusters, n_objects)
        TP = (cluster_posterior * cluster_truth).sum(axis=2)
        FP = (cluster_posterior * ~cluster_truth).sum(axis=2)
        FN = ((1 - cluster_posterior) * cluster_truth).sum(axis=2)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        fig, axes = plt.subplots(nrows=n_clusters, figsize=(6, 5 * n_clusters), squeeze=False)
        for i_clust, ax in enumerate(axes):
            t = np.linspace(0, 1, n_sims)
            # p = cluster_posterior[:, i_clust, :][cluster_truth[:, i_clust]]
            # ax[0].hist(p, bins=30)
            ax[0].scatter(precision[:, i_clust], recall[:, i_clust])
            for i in range(n_sims):
                ax[0].annotate(inferred[i].run_name, (precision[i], recall[i]),
                               xytext=(0.3, 0.3), textcoords='offset fontsize')
            ax[0].set_xlabel('Precision')
            ax[0].set_ylabel('Recall')
            ax[0].set_xlim(0, 1)
            ax[0].set_ylim(0, 1)

    elif parameter == 'cluster_precision_recall':
        n_sims = len(simulated)
        n_clusters = simulated[0].n_clusters
        cluster_posterior = np.array([
            r.cluster_posterior() for r in simulated
        ])  # shape: (n_sims, n_clusters, n_objects)
        cluster_truth = np.array([
            r.clusters[:, 0, :] for r in inferred
        ])  # shape: (n_sims, n_clusters, n_objects)
        TP = (cluster_posterior * cluster_truth).sum(axis=2)
        FP = (cluster_posterior * ~cluster_truth).sum(axis=2)
        FN = ((1-cluster_posterior) * cluster_truth).sum(axis=2)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        fig, axes = plt.subplots(nrows=n_clusters, figsize=(6, 5 * n_clusters), squeeze=False)
        for i_clust, ax in enumerate(axes):
            t = np.linspace(0, 1, n_sims)
            # p = cluster_posterior[:, i_clust, :][cluster_truth[:, i_clust]]
            # ax[0].hist(p, bins=30)
            ax[0].scatter(precision[:, i_clust], recall[:, i_clust])
            for i in range(n_sims):
                ax[0].annotate(inferred[i].run_name, (precision[i], recall[i]),
                               xytext=(0.3, 0.3), textcoords='offset fontsize')
            ax[0].set_xlabel('Precision')
            ax[0].set_ylabel('Recall')
            ax[0].set_xlim(0, 1)
            ax[0].set_ylim(0, 1)

        my_tight_layout(fig, axes)
        plt.savefig(save_to_path)
    else:
        raise ValueError(f'Parameter `{parameter}` not recognized')

def plot_area_match():
    ...

def main():
    parser = argparse.ArgumentParser(description="Plot sBayes simulation results")
    parser.add_argument("input", type=Path,
                        help="Path to the simulation directory (containing config, data and results). ")
    args = parser.parse_args()
    base_path = args.input
    sim_names = [f.name for f in (base_path / 'parameters').iterdir()]
    meta = read_meta(base_path / "meta" / "meta_sim.yaml")
    n_clusters = meta['n_clusters']

    results_simulated = []
    results_inferred = []
    for s in sim_names:
        simulated_path = base_path / 'parameters' / s
        inferred_path = base_path / 'results' / s / f'K{n_clusters}'

        # Load simulated clusters and parameters into Results object
        results_simulated.append(
            Results.from_csv_files(clusters_path=simulated_path / 'clusters_sim.txt',
                                   parameters_path=simulated_path / 'stats_sim.txt',
                                   burn_in=0, sampling_info_missing=True, run_name=s)
        )

        # Load inferred clusters and parameters samples into Results object
        results_inferred.append(
            Results.from_csv_files(clusters_path=inferred_path / f'clusters_K{n_clusters}_0.txt',
                                   parameters_path=inferred_path / f'stats_K{n_clusters}_0.txt',
                                   burn_in=0, run_name=s)
        )

    plots_path = base_path / 'plots'
    plots_path.mkdir(exist_ok=True)
    plot_simulated_against_inferred_grid(results_simulated, results_inferred, parameter='weights',
                                    save_to_path=plots_path / 'weights.pdf')
    plot_simulated_against_inferred_grid(results_simulated, results_inferred, parameter='confounding_effects',
                                    save_to_path=plots_path / 'confounding_effects.pdf')
    plot_simulated_against_inferred_grid(results_simulated, results_inferred, parameter='cluster_effects',
                                         save_to_path=plots_path / 'cluster_effects.pdf')
    # plot_simulated_against_inferred_grid(results_simulated, results_inferred, parameter='clusters',
    #                                      save_to_path=plots_path / 'clusters.pdf')
    plot_simulated_against_inferred_grid(results_simulated, results_inferred, parameter='cluster_precision_recall',
                                         save_to_path=plots_path / 'precision_recall.pdf')

    # parameter_files = get_folders(sim_path=base_path, model_n=f"K{n_clusters}")
    # parameters_raw = dict(simulated=[pd.read_csv(f, delimiter="\t") for f in parameter_files['simulated']],
    #                       inferred=[pd.read_csv(f, delimiter="\t") for f in parameter_files['inferred']])
    #
    # clusters_names = [str(i) for i in range(n_clusters)]
    #
    # confounders = {conf: list(groups.keys()) for conf, groups in meta['confounders'].items()}
    # features = meta['names']
    #
    #
    # parameters = parse_parameters(parameters_raw, features, clusters_names, confounders)
    #
    # param = 'weights'
    # # param = 'cluster_effect'
    # ft = 'categorical'
    # c = '2'
    # f = 'f4'
    # s = 'A'
    #
    # if param == 'cluster_effect':
    #     parameter_keys = ['cluster_effect', ft, c, f, s]
    # if param == 'weights':
    #     parameter_keys = ['weights', ft, f, 'cluster']
    #
    # plot_simulated_against_inferred(get_relevant_parameter(parameters['simulated'], parameter_keys),
    #                                 get_relevant_parameter(parameters['inferred'], parameter_keys),
    #                                 f'{param} of {ft} feature {f} in cluster {c}')


# @dataclasses.dataclass
# class Parameters:
#     weights: dict
#     cluster_effect: dict
#     confounding_effects: dict


if __name__ == '__main__':
    main()