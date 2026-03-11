""" Helper functions for performing simulation experiments """
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional,NamedTuple
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from sbayes.sampling.loggers import CategoricalFeatures
from sbayes.results import Results

class PathGroup(NamedTuple):
    """Represents a group of related file paths (parameters and clusters)."""
    parameters: Path
    clusters: Path


class Paths(NamedTuple):
    """Represents all file paths for both simulated and inferred results."""
    simulated: PathGroup
    inferred: PathGroup

def prepare_folder(
        path: Path,
        idx: int
) -> Tuple[Path, Path]:
    """
    Creates a folder structure for a simulation run.
    Args:
        path (Path): Base directory where the simulation folders will be created
        idx (int): Index of the simulation sample (used to name the folder)

    Returns:
        Tuple[Path, Path]: Paths to the parameters folder, and data folder
    """
    i_folder = path / f"sim_{idx}"
    i_folder.mkdir(parents=False, exist_ok=True)

    p_folder = i_folder / "sim_params"
    p_folder.mkdir(parents=False, exist_ok=True)

    d_folder = i_folder / "sim_data"
    d_folder.mkdir(parents=False, exist_ok=True)

    return p_folder, d_folder

def write_data(
    partitions: list,
    sample: Dict[str, np.ndarray],
    features_csv: pd.DataFrame,
    base_path: Path
) -> None:
    """
    Updates features CSV file with simulation data for a given sample index and saves it

    Args:
        partitions (Any): List containing the different partitions, e.g., Categorical[2]
        sample (Dict[str, np.ndarray]): Dictionary of prior sample keyed by feature name
        features_csv (pd.DataFrame): DataFrame containing the non-simulated data, otherwise empty
        base_path (Path): Path to the folder where the updated features CSV will be saved

    this function:
        - Updates the features_csv DataFrame with the new simulated data
        - Writes the updated DataFrame to `features.csv` in the data folder
    """
    for ft in partitions:
        # index 0 is correct, the ith sample was sliced earlier while retaining dimensionality
        sim = sample[f"x_{ft.name}"][0]

        # Rename categorical indices to state names
        if isinstance(ft, CategoricalFeatures):
            col_indices = np.arange(sim.shape[1])[None, :]
            sim = ft.state_names[col_indices, sim]

        df = pd.DataFrame(sim, columns=ft.names)
        features_csv[ft.names] = features_csv[ft.names].astype(str)
        features_csv.update(df)

    features_csv.to_csv(base_path / "features.csv", index=False)


def sort_by_number(name: str) -> int:
    """
    Extracts and returns the numeric suffix from a string separated by underscores.
    Used to sort folder or file 'sim_1', 'sim_10', etc.,
    by extracting the number at the end for proper numerical ordering.

    Args:
        name (str): A string that ends with an underscore followed by a number.

    Returns:
        int: The numeric suffix extracted from the string.
    """
    return int(name.split('_')[-1])

def read_parameters(base_path: Path, k: int = None,
                    feature_names: List[str] = None,
                    confounder_names: Dict = None) -> Path:
    """Read the simulated and inferred parameters

     Args:
         base_path: The folder name with the simulated parameters
         k: number of clusters in the model, if None it will be inferred
         feature_names: names of the features, if None it will be inferred
         confounder_names: names of the confounder and its groups, if None it will be inferred
     Returns:
         simulated and inferred parameters
     """

    sims = os.listdir(base_path)

    params = dict()

    # Open the simulation runs in the sims folder one by one
    for i in sorted(sims, key=sort_by_number):
        params[i] = dict()
        if k:
            # Read clusters and parameters
            result_paths = build_result_paths(base_path, i, k)
            simulated = Results.from_csv_files(result_paths.simulated.clusters,
                                               result_paths.simulated.parameters, burn_in=0,
                                               feature_names=feature_names,
                                               confounder_names=confounder_names)
            inferred = Results.from_csv_files(result_paths.inferred.clusters,
                                              result_paths.inferred.parameters, burn_in=0,
                                              feature_names=feature_names,
                                              confounder_names=confounder_names)
            inferred_mean_cluster = np.mean(inferred.clusters, axis=1)
            d = cluster_agreement(simulated.clusters[:, 0, :].astype(float), inferred_mean_cluster)

            perm = linear_sum_assignment(d, maximize=True)[1]

            params[i]['simulated'] = simulated.parameters
            params[i]['inferred'] = get_permuted_params(inferred, perm)

    return params


def build_result_paths(base_path: Path, sim: str, k: int) -> Paths:
    """
    Builds file paths for simulation and inference results, grouped by category.

    Args:
        base_path: The root directory where simulations are stored.
        sim: The simulation name or identifier.
        k: The cluster size parameter used in the file names.

    Returns:
        A `Paths` NamedTuple with two attributes:
            - simulated: A PathGroup containing simulation parameter and cluster files.
            - inferred: A PathGroup containing inference parameter and cluster files.
    """

    return Paths(
        simulated=PathGroup(
            parameters=base_path / sim / "sim_params" / f"stats_K{k}_0.txt",
            clusters=base_path / sim / "sim_params" / f"clusters_K{k}_0.txt"),
        inferred=PathGroup(
            parameters=base_path / sim / f"results/K{k}/" / f"stats_K{k}_0.txt",
            clusters=base_path / sim / f"results/K{k}/" / f"clusters_K{k}_0.txt")
    )

def cluster_agreement(a1: NDArray[np.float64], a2: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Computes the agreement between two cluster assignment matrices.

    The agreement is calculated as the matrix product of `a1` and the transpose of `a2`.

    Args:
        a1: A 2D NumPy array representing the first cluster assignment matrix.
        a2: A 2D NumPy array representing the second cluster assignment matrix.

    Returns:
        A 2D NumPy array where each entry (i, j) represents the agreement score
        between row i of `a1` and row j of `a2`.
    """
    return np.matmul(a1, a2.T)

def find_conf_prefix(name: str, confounders: List) -> Tuple[str, str]:
    """
    Extracts the confounder prefix from the given name.

    Args:
        name: The input string expected to start with a confounder prefix.
        confounders: A list of known confounder name strings.

    Returns:
        A tuple containing:
            - The matching confounder string.
            - The remaining portion of the name after removing the prefix.

    Raises:
        ValueError: If no confounder prefix matches the start of the name.
    """
    for conf in confounders:
        prefix = f"{conf}_"
        if name.startswith(prefix):
            return conf, name.removeprefix(prefix)
    raise ValueError(f"No matching confounder found in '{name}'")

def find_group_prefix(name: str, groups: List[str]) -> Tuple[str, str]:
    """
    Extracts the group prefix from the given name.

    Args:
        name: The input string expected to start with a group prefix.
        groups: A list of known group name strings.

    Returns:
        A tuple containing:
            - The matching group string.
            - The remaining portion of the name after removing the prefix.

    Raises:
        ValueError: If no group prefix matches the start of the name.
    """
    for g in groups:
        prefix = f"{g}_"
        if name.startswith(prefix):
            return g, name.removeprefix(prefix)
    raise ValueError(f"No matching group found in '{name}'")

def find_feature_prefix(name, features):
    """
    Extracts the feature prefix from the given name.

    Args:
        name: The input string expected to start with a feature prefix.
        features: A list of known feature name strings.

    Returns:
        A tuple containing:
            - The matching feature string.
            - The remaining portion of the name after removing the prefix.

    Raises:
        ValueError: If no feature prefix matches the start of the name.
    """
    for f in features:
        prefix = f"{f}_"
        if name.startswith(prefix):
            return f, name.removeprefix(prefix)
    raise ValueError(f"No matching feature found in '{name}'")

def find_areal_prefix(name: str) -> Tuple[int, str]:
    """
    Extracts the areal cluster number and remaining string from the given name.

    The function expects the name to start with a prefix matching the pattern "areal_a<digits>_".
    It extracts the number after 'a', increments it by 1, and returns it along with the
    remaining part of the name after removing the matched prefix.

    Args:
        name: The input string expected to start with an areal prefix, e.g. "areal_a12_..."


    Returns:
        A tuple containing:
            - The incremented areal cluster number (int).
            - The remaining portion of the name after removing the areal prefix.

    Raises:
        ValueError: If the input string does not match the expected pattern.
    """
    match_obj = re.match(r"(areal_a\d+_)", name)
    if not match_obj:
        raise ValueError(f"Name '{name}' does not start with a valid areal prefix")

    match = match_obj.group(1)

    number_match = re.search(r"a(\d+)", match)
    if not number_match:
        raise ValueError(f"Could not extract number from areal prefix '{match}'")

    number = number_match.group(1)

    return int(number) + 1, name.removeprefix(match)

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


def find_title(
    name: str,
    confounders: Dict[str, Any],
    feature_names: List[str]
) -> Optional[str]:
    """
    Generates a descriptive title based on the given parameter name.

    The function parses the input name string to identify confounder prefixes, groups,
    features, and components, or areal cluster prefixes, returning a human-readable
    description.

    Args:
        name: The parameter name string to parse.
        confounders: A dictionary mapping confounder names to confounder objects.
                     Each confounder object must have a 'group_names' attribute,
                     which is a list of group name strings.
        feature_names: A list of valid feature name strings.

    Returns:
        A descriptive string summarizing the feature/component/group information,
        or None if the name does not match any expected pattern.

    """
    confounder_names = list(confounders.keys())
    confounder_prefixes = tuple(f"{conf}_" for conf in confounder_names)
    w_confounder_prefixes = tuple(f"w_{conf}_" for conf in confounder_names)

    if name.startswith(confounder_prefixes):

        conf, remaining = find_conf_prefix(name, confounder_names)
        group, remaining = find_group_prefix(remaining,
                                             confounders[conf].group_names)
        feature, component = find_feature_prefix(remaining, feature_names)
        return f"Feature {feature}, component {component} in {conf}, group {group}"

    elif name.startswith("areal_"):
        cluster, remaining = find_areal_prefix(name)
        feature, component = find_feature_prefix(remaining, feature_names)
        return f"Feature {feature}, component {component} in cluster {cluster}"

    elif name.startswith(w_confounder_prefixes):
        conf, feature = find_conf_prefix(name.removeprefix("w_"), confounder_names)
        return f"Weights for feature {feature} and {conf}"

    elif name.startswith("w_areal_"):
        feature = name.removeprefix("w_areal_")
        return f"Cluster weights for feature {feature}"
    else:
        return None

def plot_simulated_against_inferred(
    simulated: NDArray[np.floating],
    inferred: NDArray[np.floating],
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    return_failed_sims: bool = False,
) -> None | list:
    """
    Plot simulated values on the x-axis against inferred distributions on the y-axis.

    Each row in `inf` corresponds to one simulated value in `sim`. The function checks
    whether each simulated value falls within the 5th–95th percentile of its corresponding
    inferred distribution and color-codes the points accordingly.

    Parameters:
        simulated (NDArray[np.floating]): Array of shape (n,), containing simulated values
        inferred (NDArray[np.floating]): Array of shape (n, m), containing inferred distributions
        title (Optional[str]): Optional title for the plot
        ax (Optional[plt.Axes]): Matplotlib Axes to plot on. Defaults to current axis
        return_failed_sims (bool): Whether to return the ids of failed simulations
    Returns:
        None | failed simulations (list)
    """
    if ax is None:
        ax = plt.gca()

    low_perc = np.percentile(inferred, 2.5, axis=1)
    high_perc = np.percentile(inferred, 97.5, axis=1)
    in_perc = (low_perc < simulated) & (simulated < high_perc)

    min_val = np.min([inferred.min(), simulated.min()])
    max_val = np.max([inferred.max(), simulated.max()])
    failed_sims = []
    for i, sim in enumerate(simulated):
        color = 'lightgrey' if in_perc[i] else 'red'
        if not in_perc[i]:
            failed_sims.append(i)
        ax.plot([sim] * inferred.shape[1], inferred[i],
                'o', markersize=1, alpha=0.1, color=color)

    ax.axline((0, 0), slope=1, color='black')
    ax.text(
        0.99, 0.01,
        f"{np.sum(in_perc)} / {len(simulated)}",
        ha='right',
        va='bottom',
        transform=ax.transAxes
    )

    ax.set_xlabel('Simulated')
    ax.set_ylabel('Estimated')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(False)

    if title:
        ax.set_title(title)

    if return_failed_sims:
        return failed_sims
    else:
        return None


