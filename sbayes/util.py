#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime
import sys
import time
import csv
import os
import traceback
import warnings
from pathlib import Path
from math import sqrt, floor, ceil
from itertools import combinations, permutations
from typing import Sequence, Union, Iterator

import psutil
from numpy._typing import NDArray
from unidecode import unidecode
from math import lgamma

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
import pandas as pd
import jax.scipy as scipy
import scipy.spatial as spatial
from jax.scipy.special import betaln, expit
import scipy.stats as stats
from scipy.sparse import csr_matrix
from numba import jit, njit, float32, float64, int64, boolean, vectorize
import jax.numpy as jnp


FLOAT_TYPE = np.float32
INT_TYPE = np.int32
EPS = np.finfo(FLOAT_TYPE).eps
LOG_EPS = np.finfo(FLOAT_TYPE).min
RNG = np.random.default_rng()


PathLike = Union[str, Path]
"""Convenience type for cases where `str` or `Path` are acceptable types."""


class FamilyError(Exception):
    pass


def encode_cluster(cluster: NDArray[bool]) -> str:
    """Format the given cluster as a compact bit-string."""
    cluster_s = cluster.astype(int).astype(str)
    return ''.join(cluster_s)


def decode_cluster(cluster_str: str) -> NDArray[bool]:
    """Read a bit-string and parse it into an area array."""
    return np.array(list(cluster_str)).astype(int).astype(bool)


def format_cluster_columns(clusters: NDArray[bool]) -> str:
    """Format the given array of clusters as tab separated strings."""
    clusters_encoded = map(encode_cluster, clusters)
    return '\t'.join(clusters_encoded)


def parse_cluster_columns(clusters_encoded: str) -> NDArray[bool]:
    """Read tab-separated area encodings into a two-dimensional area array."""
    clusters_decoded = map(decode_cluster, clusters_encoded.split('\t'))
    return np.array(list(clusters_decoded))


def compute_distance(a, b):
    """ This function computes the Euclidean distance between two points a and b

    Args:
        a (list): The x and y coordinates of a point in a metric CRS.
        b (list): The x and y coordinates of a point in a metric CRS.

    Returns:
        float: Distance between a and b
    """

    a = np.asarray(a)
    b = np.asarray(b)
    ab = b-a
    dist = sqrt(ab[0]**2 + ab[1]**2)

    return dist


def bounding_box(points):
    """ This function retrieves the bounding box for a set of 2-dimensional input points

    Args:
        points (numpy.array): Point tuples (x,y) for which the bounding box is computed
    Returns:
        (dict): the bounding box of the points
    """
    x = [x[0] for x in points]
    y = [x[1] for x in points]
    box = {'x_max': max(x),
           'y_max': max(y),
           'x_min': min(x),
           'y_min': min(y)}

    return box


def get_neighbours(cluster, already_in_cluster, adjacency_matrix, indirection=0):
    """This function returns the neighbourhood of a cluster as given in the adjacency_matrix, excluding
    objects already belonging to this or any other cluster.

    Args:
        cluster (np.array): The current cluster (boolean array)
        already_in_cluster (np.array): All objects already assigned to a cluster (boolean array)
        adjacency_matrix (np.array): The adjacency matrix of the objects (boolean)
        indirection (int): Number of inbetween steps allowed for transitive neighborhood.

    Returns:
        np.array: The neighborhood of the cluster (boolean array)
    """

    # Get all neighbors of the current zone
    reachable = adjacency_matrix.dot(cluster)

    # Get neighbors of neighbors for each level of indirection
    for i in range(indirection):
        reachable = adjacency_matrix.dot(reachable)

    # Exclude all vertices that are already in a zone
    return np.logical_and(reachable, ~already_in_cluster)





def gabriel_graph_from_delaunay(delaunay, locations):
    delaunay = delaunay.toarray()
    # converting delaunay graph to boolean array denoting whether points are connected
    delaunay = delaunay > 0

    # Delaunay indices and locations
    delaunay_connections = []
    delaunay_locations = []

    for index, connected in np.ndenumerate(delaunay):
        if connected:
            # getting indices of points in area
            i1, i2 = index[0], index[1]
            if [i2, i1] not in delaunay_connections:
                delaunay_connections.append([i1, i2])
                delaunay_locations.append(locations[[*[i1, i2]]])
    delaunay_connections = np.sort(np.asarray(delaunay_connections), axis=1)
    delaunay_locations = np.asarray(delaunay_locations)

    # Find the midpoint on all Delaunay edges
    m = (delaunay_locations[:, 0, :] + delaunay_locations[:, 1, :]) / 2

    # Find the radius sphere between each pair of nodes
    r = np.sqrt(np.sum((delaunay_locations[:, 0, :] - delaunay_locations[:, 1, :]) ** 2, axis=1)) / 2

    # Use the kd-tree function in Scipy's spatial module
    tree = spatial.cKDTree(locations)
    # Find the nearest point for each midpoint
    n = tree.query(x=m, k=1)[0]
    # If nearest point to m is at a distance r, then the edge is a Gabriel edge
    g = n >= r * 0.999  # The factor is to avoid precision errors in the distances

    return delaunay_connections[g]


def gabriel(distances):
    """Directly compute the adjacency matrix for the Gabriel graph from a distance matrix."""
    n = len(distances)
    adj = np.empty((n, n), dtype=bool)
    d_squared = distances ** 2
    for i in range(n):
        # An edge is included if the squared distance between the node is smaller
        # than the sum of squared distances of any detour via a third node.
        detour = np.min(d_squared[i, :] + d_squared[:, :], axis=-1)
        adj[i, :] = (d_squared[i] <= detour)
    return adj


def n_smallest_distances(a, n, return_idx: bool):
    """ This function finds the n smallest distances in a distance matrix

    >>> n_smallest_distances([
    ... [0, 2, 3, 4],
    ... [2, 0, 5, 6],
    ... [3, 5, 0, 7],
    ... [4, 6, 7, 0]], 3, return_idx=False)
    array([2, 3, 4])

    >>> n_smallest_distances([
    ... [0, 2, 3, 4],
    ... [2, 0, 5, 6],
    ... [3, 5, 0, 7],
    ... [4, 6, 7, 0]], 3, return_idx=True)
    (array([1, 2, 3]), array([0, 0, 0]))

    Args:
        a (np.array): The distane matrix
        n (int): The number of distances to return
        return_idx (bool): return the indices of the points (True) or rather the distances (False)

    Returns:
        (np.array): the n_smallest distances
    or
        (np.array, np.array): the indices between which the distances are smallest
    """
    a_tril = np.tril(a)
    a_nn = a_tril[np.nonzero(a_tril)]
    smallest_n = np.sort(a_nn)[: n]
    a_idx = np.isin(a_tril, smallest_n)

    if return_idx:
        return np.where(a_idx)
    else:
        return smallest_n


def set_experiment_name():
    """Get the current time and use it to name the current experiment
    Returns:
         (str): the name of the current experiment
    """
    now = datetime.datetime.now().__str__().rsplit('.')[0]
    now = now[:-3]
    now = now.replace(':', '-')
    now = now.replace(' ', '_')

    return now


def clusters_autosimilarity(cluster, t):
    """
    This function computes the similarity of consecutive cluster in a chain
    Args:
        cluster (list): cluster
        t (integer): lag between consecutive cluster in the chain

    Returns:
        (float) : mean similarity between cluster in the chain with lag t
    """
    z = np.asarray(cluster)
    z = z[:, 0, :]
    unions = np.maximum(z[t:], z[:-t])
    intersections = np.minimum(z[t:], z[:-t])
    sim_norm = np.sum(intersections, axis=1) / np.sum(unions, axis=1)

    return np.mean(sim_norm)


def range_like(a):
    """Return a list of incrementing integers (range) with same length as `a`."""
    return list(range(len(a)))



def normalize_str(s: str) -> str:
    if pd.isna(s):
        return s
    return str.strip(unidecode(s))


def read_data_csv(csv_path: PathLike) -> pd.DataFrame:
    na_values = ["", " ", "\t", "  "]
    data: pd.DataFrame = pd.read_csv(csv_path, na_values=na_values, keep_default_na=False, dtype=str)
    data.columns = [unidecode(c) for c in data.columns]

    if pd.__version__ >= '2.1.0':  # Handle Pandas deprecation warning
        return data.map(normalize_str)
    else:
        return data.applymap(normalize_str)





def scale_counts(counts, scale_to, prior_inheritance=False):
    """Scales the counts for parametrizing the prior on universal probabilities (or inheritance in a family)

        Args:
            counts (np.array): the counts of categorical data.
                shape: (n_features, n_states) or (n_families, n_features, n_states)
            scale_to (float): the counts are scaled to this value
            prior_inheritance (bool): are these inheritance counts?
        Returns:
            np.array: the rescaled counts
                shape: same as counts.shape
    """
    counts_sum = np.sum(counts, axis=-1)
    counts_sum = np.where(counts_sum == 0, EPS, counts_sum)
    scale_factor = scale_to / counts_sum

    scale_factor = np.where(scale_factor < 1, scale_factor, 1)
    return counts * scale_factor[..., None]


def touch(fname):
    """Create an empty file at path `fname`."""
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()


def mkpath(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isdir(path):
        touch(path)


def normalize(x, axis=-1):
    """Normalize ´x´ s.t. the last axis sums up to 1.

    Args:
        x (np.array): Array to be normalized.
        axis (int): The axis to be normalized (will sum up to 1).

    Returns:
         np.array: x with normalized s.t. the last axis sums to 1.

    == Usage ===
    >>> normalize(np.ones((2, 4))).tolist()
    [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
    >>> normalize(np.ones((2, 4)), axis=0).tolist()
    [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]
    """
    return (x / jnp.sum(x, axis=axis, keepdims=True))


def assess_correlation_probabilities(p_universal, p_contact, p_inheritance, corr_th, include_universal=False):
    """Asses the correlation of probabilities in simulated data

        Args:
            p_universal (np.array): universal state probabilities
                shape (n_features, n_states)
            p_contact (np.array): state probabilities in areas
                shape: (n_areas, n_features, n_states)
            p_inheritance (np.array): state probabilities in families
                shape: (n_families, n_features, n_states)
            corr_th (float): correlation threshold
            include_universal (bool): Should p_universal also be checked for independence?

        """
    if include_universal:
        if p_inheritance is not None:
            samples = np.vstack((p_universal[np.newaxis, :, :], p_contact, p_inheritance))
        else:
            samples = np.vstack((p_universal[np.newaxis, :, :], p_contact))
    else:
        if p_inheritance is not None:
            samples = np.vstack((p_contact, p_inheritance))
        else:
            samples = p_contact

    n_samples = samples.shape[0]
    n_features = samples.shape[1]

    comb = list(combinations(list(range(n_samples)), 2))
    correlated_probability_vectors = 0

    for f in range(n_features):
        for c in comb:
            p0 = samples[c[0], f]
            p1 = samples[c[1], f]
            p_same_state = np.dot(p0, p1)
            if p_same_state > corr_th:
                correlated_probability_vectors += 1
    return correlated_probability_vectors


def log_multinom(n: int, ks: Sequence[int]) -> float:
    """Compute the logarithm of (n choose k1,k2,...), i.e. the multinomial coefficient of
    `n` and the integers in the list `ks`. The sum of the sample sizes (the numbers in
     `ks`) may not exceed the population size (`n`).

    Args:
        n: Population size.
        ks: Sample sizes

    Returns:
        The log multinomial coefficient: log(n choose k1,k2,...)

    == Usage ===
    >>> log_multinom(5, [1,1,1,1])  # == log(5!)
    4.787491742782046
    >>> log_multinom(13, [4])  # == log_binom(13, 4)
    6.572282542694008
    >>> log_multinom(13, [3, 2])  # == log_binom(13, 3) + log_binom(10, 2)
    9.462654300590172
    """
    ks = np.asarray(ks)
    # assert np.all(ks >= 0)
    # assert np.sum(ks) <= n

    # Simple special case
    if np.sum(ks) == 0:
        return 0.

    # Filter out 0-samples
    ks = ks[ks > 0]

    log_i = np.log(1 + np.arange(n))
    log_i_cumsum = np.cumsum(log_i)

    # Count all permutations of the total population
    m = np.sum(log_i)

    # Subtract all permutation within the samples (with sample sizes specified in `ks`).
    m -= np.sum(log_i_cumsum[ks-1])

    # If there are is a remainder in the population, that was not assigned to any of the
    # samples, subtract all permutations of the remainder population.
    rest = n - np.sum(ks)
    # assert rest >= 0
    if rest > 0:
        m -= log_i_cumsum[rest-1]

    # assert m >= 0, m
    return m


def decompose_config_path(config_path: PathLike) -> (Path, Path):
    """Extract the base directory of `config_path` and return the path itself as an
    absolute path."""
    abs_config_path = Path(config_path).absolute()
    base_directory = abs_config_path.parent
    return base_directory, abs_config_path


def fix_relative_path(path: PathLike, base_directory: PathLike) -> Path:
    """Make sure that the provided path is either absolute or relative to the config file directory.

    Args:
        path: The original path (absolute or relative).
        base_directory: The base directory

    Returns:
        The fixed path.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    else:
        return base_directory / path


def timeit(units='s'):
    SECONDS_PER_UNIT = {
        'h': 3600.,
        'm': 60.,
        's': 1.,
        'ms': 1E-3,
        'µs': 1E-6,
        'ns': 1E-9
    }
    unit_scaler = SECONDS_PER_UNIT[units]

    def timeit_decorator(func):

        def timed_func(*args, **kwargs):


            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            passed = (end - start) / unit_scaler

            print(f'Runtime {func.__name__}: {passed:.2f}{units}')

            return result

        return timed_func

    return timeit_decorator


def get_best_permutation(
        areas: NDArray[bool],  # shape = (n_areas, n_objects)
        prev_area_sum: NDArray[int],  # shape = (n_areas, n_objects)
) -> NDArray[int]:
    """Return a permutation of areas that would align the areas in the new sample with previous ones."""
    cluster_agreement_matrix = np.matmul(prev_area_sum, areas.T)
    return linear_sum_assignment(cluster_agreement_matrix, maximize=True)[1]


def cluster_agreement(a1, a2):
    return np.matmul(a1, a2.T)


# if scipy.__version__ >= '1.8.0':
#     log_expit = scipy.special.log_expit
# else:
def log_expit(*args, **kwargs):
    return jnp.log(expit(*args, **kwargs))


def set_defaults(cfg: dict, default_cfg: dict):
    """Iterate through a recursive config dictionary and set all fields that are not
    present in cfg to the default values from default_cfg.

    == Usage ===
    >>> set_defaults(cfg={0:0, 1:{1:0}, 2:{2:1}},
    ...              default_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 0}, 2: {2: 1, 1: 1}}
    >>> set_defaults(cfg={0:0, 1:1, 2:2},
    ...              default_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: 1, 2: 2}
    """
    for key in default_cfg:
        if key not in cfg:
            # Field ´key´ is not defined in cfg -> use default
            cfg[key] = default_cfg[key]

        else:
            # Field ´key´ is defined in cfg
            # -> update recursively if the field is a dictionary
            if isinstance(default_cfg[key], dict) and isinstance(cfg[key], dict):
                set_defaults(cfg[key], default_cfg[key])

    return cfg


def update_recursive(cfg: dict, new_cfg: dict):
    """Iterate through a recursive config dictionary and update cfg in all fields that are specified in new_cfg.

    == Usage ===
    >>> update_recursive(cfg={0:0, 1:{1:0}, 2:{2:1}},
    ...                  new_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 1}, 2: {2: 2, 1: 1}}
    >>> update_recursive(cfg={0:0, 1:1, 2:2},
    ...                  new_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 1}, 2: {1: 1, 2: 2}}
    """
    for key in new_cfg:
        if (key in cfg) and isinstance(new_cfg[key], dict) and isinstance(cfg[key], dict):
            # Both dictionaries have another layer -> update recursively
            update_recursive(cfg[key], new_cfg[key])
        else:
            cfg[key] = new_cfg[key]

    return cfg


def iter_items_recursive(cfg: dict, loc=tuple()):
    """Recursively iterate through all key-value pairs in ´cfg´ and sub-dictionaries.

    Args:
        cfg (dict): Config dictionary, potentially containing sub-dictionaries.
        loc (tuple): Specifies the sequene of keys that lead to the current sub-dictionary.
    Yields:
        tuple: key-value pairs of the bottom level dictionaries

    == Usage ===
    >>> list(iter_items_recursive({0: 0, 1: {1: 0}, 2: {2: 1, 1: 1}}))
    [(0, 0, ()), (1, 0, (1,)), (2, 1, (2,)), (1, 1, (2,))]
    """
    for key, value in cfg.items():
        if isinstance(value, dict):
            yield from iter_items_recursive(value, loc + (key, ))
        else:
            yield key, value, loc


def get_along_axis(a: NDArray, index: int, axis: int):
    """Get the index-th entry in the axis-th dimension of array a.
    Examples:
        >>> get_along_axis(a=np.arange(6).reshape((2,3)), index=2, axis=1)
        array([2, 5])
    """
    I = [slice(None)] * a.ndim
    I[axis] = index
    return a[tuple(I)]


def inner1d(x, y):
    return np.einsum("...i,...i", x, y)


def pmf_categorical_with_replacement(idxs: list[int], p: NDArray[float]):
    prob = 0
    for idxs_perm in map(list, permutations(idxs)):
        prob += np.prod(p[idxs_perm]) / np.prod(1-np.cumsum(p[idxs_perm][:-1]))
    return prob


def trunc_exp_rv(low, high, scale, size):
    rnd_cdf = np.random.uniform(stats.expon.cdf(x=low, scale=scale),
                                stats.expon.cdf(x=high, scale=scale),
                                size=size)
    return stats.expon.ppf(q=rnd_cdf, scale=scale)


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    # traceback.print_stack(file=log)
    warning_trace = traceback.format_stack()
    warning_trace_str = "".join(["\n\t|" + l for l in "".join(warning_trace).split("\n")])
    message = str(message) + warning_trace_str
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def activate_verbose_warnings():
    warnings.showwarning = warn_with_traceback


def process_memory(pid: int = None, unit="B") -> int:
    """Memory usage of the process with give `pid`,
    or the current process if `pid` is None."""
    mem_in_bytes = psutil.Process(pid).memory_info().rss
    if unit == "B":
        return mem_in_bytes
    elif unit == "KB":
        return mem_in_bytes >> 10
    elif unit == "MB":
        return mem_in_bytes >> 20
    elif unit == "GB":
        return mem_in_bytes >> 30
    elif unit == "TB":
        return mem_in_bytes >> 40
    else:
        raise ValueError(f"Unknown unit `{unit}`")


def heat_binary_probability(p: float, temperature: float) -> float:
    """Take the probability of a binary event to the power of (1/temperature)
    and renormalize over a positive and negative outcome.

    == Usage ===
    >>> heat_binary_probability(0.5, 2)
    0.5
    >>> round(heat_binary_probability(1/3, 0.5), 6)
    0.2
    """
    pow = 1 / temperature
    p_pow = p ** pow
    return p_pow / (p_pow + (1 - p)**pow)


def onehot_to_integer_encoding(onehot: NDArray[bool], none_index: int = -1, axis: int = -1) -> jnp.ndarray:
    """Convert one-hot encoding to integer encoding.

    == Usage ===
    >>> onehot_to_integer_encoding(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool))
    Array([ 0,  1,  2, -1], dtype=int32)
    """
    int_encoding = jnp.where(onehot.any(axis=axis), jnp.argmax(onehot, axis=axis), none_index)
    return int_encoding


def normalize_weights(
    weights: NDArray[float],  # shape: (n_features, 1 + n_confounders)
    has_components: NDArray[bool]  # shape: (n_objects, 1 + n_confounders)
) -> NDArray[float]:  # shape: (n_objects, n_features, 1 + n_confounders)
    """This function assigns each site a weight if it has a likelihood and zero otherwise
    Args:
        weights: the weights to normalize
        has_components: indicators for which objects are affected by cluster and confounding effects
    Return:
        the weight_per site
    """
    # Find the unique patterns in `has_components` and remember the inverse mapping
    pattern, pattern_inv = np.unique(has_components, axis=0, return_inverse=True)

    # Calculate the normalized weights per pattern
    w_per_pattern = pattern[:, None, :] * weights[None, :, :]
    w_per_pattern /= np.sum(w_per_pattern, axis=-1, keepdims=True)

    # Broadcast the normalized weights per pattern to the objects where the patterns appeared using pattern_inv
    return w_per_pattern[pattern_inv]


# # TODO: temporary home - sample_categorical is not a general utility.
# #   Move to sbayes/simulate/ or delete once its callers are confirmed.
EYES = {}

def sample_categorical(p, binary_encoding=False):
    """Sample from a (multidimensional) categorical distribution. The
    probabilities for every category are given by `p`

    Args:
        p (np.array): Array defining the probabilities of every category at
            every site of the output array. The last axis defines the categories
            and should sum up to 1.
            shape: (*output_dims, n_states)
        binary_encoding(bool): Return samples in binary encoding?
    Returns
        np.array: Samples of the categorical distribution.
            shape: output_dims
                or
            shape: (output_dims, n_states)
    """
    *output_dims, n_states = p.shape

    assert np.all(p >= 0)

    cdf = np.cumsum(p, axis=-1)
    assert np.allclose(cdf[..., -1], 1.)
    cdf /= cdf[..., [-1]]
    z = np.random.random(output_dims + [1])

    samples = np.argmax(z < cdf, axis=-1)
    if binary_encoding:
        if n_states not in EYES:
            EYES[n_states] = np.eye(n_states, dtype=bool)
        eye = EYES[n_states]
        return eye[samples]
    else:
        return samples


if __name__ == "__main__":
    import doctest
    doctest.testmod()
