from __future__ import annotations
import copy
import datetime
import functools
import jax
import numpy as np
import pandas as pd
import sys
import textwrap
import time
import traceback
import warnings

from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpy.typing import NDArray
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from typing import Union, TextIO
from unidecode import unidecode


FLOAT_TYPE = np.float32
EPS = np.finfo(FLOAT_TYPE).eps

PathLike = Union[str, Path]
"""Convenience type for cases where `str` or `Path` are acceptable types."""

def encode_cluster(cluster: NDArray[np.bool_]) -> str:
    """Format one cluster as a compact bit-string, one character per object.

    Args:
        cluster: the objects assigned to the cluster. shape: (n_objects,)

    Returns:
        A string of `0`s and `1`s, e.g. `"0110"` for four objects of which the middle
        two are in the cluster.
    """
    return ''.join(cluster.astype(int).astype(str))


def decode_cluster(cluster_str: str) -> NDArray[np.bool_]:
    """Parse a bit-string into the assignment array of one cluster.

    Args:
        cluster_str: a string of `0`s and `1`s, one character per object

    Returns:
        The objects assigned to the cluster. shape: (n_objects,)
    """
    return np.array(list(cluster_str)).astype(int).astype(bool)

def format_cluster_columns(clusters: NDArray[np.bool_]) -> str:
    """Format the clusters of one sample as tab-separated bit-strings.

    Args:
        clusters: the cluster assignments. shape: (n_clusters, n_objects)

    Returns:
        One bit-string per cluster, separated by tabs.
    """
    return '\t'.join(map(encode_cluster, clusters))


def parse_cluster_columns(clusters_encoded: str) -> NDArray[np.bool_]:
    """Parse the tab-separated bit-strings of one sample into an assignment array.

    Args:
        clusters_encoded: one bit-string per cluster, separated by tabs

    Returns:
        The cluster assignments. shape: (n_clusters, n_objects)
    """
    return np.array(list(map(decode_cluster, clusters_encoded.split('\t'))))



def default_experiment_name() -> str:
    """Generate a default name for an experiment from the current time.

    Returns:
        The current time as `YYYY-MM-DD_HH-MM-SS`, safe to use in file and directory names.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



def normalize_str(s: str | float) -> str | float:
    """Transliterate a CSV cell to ASCII and strip surrounding whitespace.

    Missing values (NaN) are returned unchanged.

    Args:
        s: a cell value, either a string or NaN for a missing value

    Returns:
        The transliterated, stripped string, or `s` itself if it is not a string.
    """
    if not isinstance(s, str):
        return s
    return unidecode(s).strip()


def read_data_csv(csv_path: PathLike) -> pd.DataFrame:
    """Read a data CSV with every cell as a string, normalized by `normalize_str`.

    Cells that are empty or contain only whitespace become NaN. Strings such as
    `NA` or `null` are kept as values, not treated as missing.

    Args:
        csv_path: path to the CSV file

    Returns:
        The table with normalized column names and cell values.
    """
    data = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])
    data.columns = [normalize_str(c) for c in data.columns]
    data = data.map(normalize_str)
    return data.mask(data == "")


def cap_counts(counts: NDArray[np.floating], cap_to: float) -> NDArray[np.floating]:
    """Cap the total of each count vector at `cap_to`, keeping its proportions.

    Used to limit the weight of empirical prior counts. Vectors whose total is
    already at most `cap_to`, including all-zero vectors, are returned unchanged.

    Args:
        counts: counts of categorical states. shape: (..., n_states)
        cap_to: the maximum total per count vector. Must be positive.

    Returns:
        The capped counts. shape: same as `counts`
    """
    if cap_to <= 0:
        raise ValueError(f"`cap_to` must be positive, got {cap_to}.")
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=-1, keepdims=True)
    return counts * (cap_to / np.maximum(totals, cap_to))


def normalize(x: ArrayLike, axis: int = -1) -> Array:
    """Normalize `x` so that it sums to 1 along `axis`.

    Slices that sum to zero become NaN; add a small constant (e.g. `EPS`) first
    if that can happen.

    Args:
        x: non-negative values to normalize
        axis: the axis that will sum to 1

    Returns:
        The normalized array. shape: same as `x`
    """
    x = jnp.asarray(x)
    return x / jnp.sum(x, axis=axis, keepdims=True)


def decompose_config_path(config_path: PathLike) -> tuple[Path, Path]:
    """Return the directory of a config file and the file's absolute path.

    Relative paths inside the config are resolved against this directory.

    Args:
        config_path: path to the config file, absolute or relative to the working directory

    Returns:
        The base directory and the absolute config path.
    """
    abs_config_path = Path(config_path).absolute()
    return abs_config_path.parent, abs_config_path


def fix_relative_path(path: PathLike, base_directory: PathLike) -> Path:
    """Resolve `path` against `base_directory` unless it is already absolute.

    Args:
        path: an absolute path, or a path relative to `base_directory`
        base_directory: the directory relative paths refer to, usually the config file's directory

    Returns:
        `path` if it is absolute, otherwise `base_directory / path`.
    """
    return Path(base_directory) / path


def timeit(units: str = "s"):
    """Decorator that prints the runtime of each call to the decorated function.

    Args:
        units: the unit of the printed runtime, one of `h`, `m`, `s`, `ms`, `µs`, `ns`
    """
    scale = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3, "µs": 1e-6, "ns": 1e-9}[units]

    def decorator(func):
        @functools.wraps(func)
        def timed(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            print(f"Runtime {func.__name__}: {(time.perf_counter() - start) / scale:.2f}{units}")
            return result
        return timed

    return decorator


def get_best_permutation(
    clusters: NDArray[np.bool_ | np.floating],
    prev_cluster_sum: NDArray[np.integer | np.floating],
) -> NDArray[np.intp]:
    """Find the ordering of `clusters` that best matches previously seen clusters.

    Cluster labels are arbitrary, so the same cluster can appear under a different
    index in each sample. The permutation maximizes the total agreement (shared
    membership) between each previous cluster and the new cluster assigned to it.

    Args:
        clusters: cluster memberships of the new sample. shape: (n_clusters, n_objects)
        prev_cluster_sum: accumulated memberships of the previous samples.
            shape: (n_clusters, n_objects)

    Returns:
        `perm` such that `clusters[perm]` is aligned with `prev_cluster_sum`.
        shape: (n_clusters,)
    """
    if clusters.shape != prev_cluster_sum.shape:
        raise ValueError(
            f"Shape mismatch: clusters {clusters.shape}, previous sum {prev_cluster_sum.shape}."
        )
    agreement = cluster_agreement(prev_cluster_sum, clusters)
    _, perm = linear_sum_assignment(agreement, maximize=True)
    return perm


def cluster_agreement(clusters_a: NDArray, clusters_b: NDArray) -> NDArray:
    """Count the objects shared by each pair of clusters.

    Args:
        clusters_a: cluster memberships. shape: (n_clusters_a, n_objects)
        clusters_b: cluster memberships. shape: (n_clusters_b, n_objects)

    Returns:
        Entry `[i, j]` is the number of objects in both `clusters_a[i]` and
        `clusters_b[j]`, or the summed product of memberships for continuous
        inputs. shape: (n_clusters_a, n_clusters_b)
    """
    return clusters_a @ clusters_b.T


def log_expit(x: ArrayLike) -> Array:
    """Compute log(expit(x)) = log(1 / (1 + exp(-x))) in a numerically stable way.

    Args:
        x: input values

    Returns:
        The log-sigmoid of `x`, elementwise. shape: same as `x`
    """
    return jax.nn.log_sigmoid(x)


def update_recursive(cfg: dict, new_cfg: dict) -> dict:
    """Deep-merge `new_cfg` into `cfg`, in place.

    Nested dicts are merged key by key. Any other value in `new_cfg` replaces the
    value in `cfg`, including when one side is a dict and the other is not.

    Args:
        cfg: the dict to update, modified in place
        new_cfg: the overrides, not modified

    Returns:
        `cfg`, for convenience.
    """
    for key, value in new_cfg.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            update_recursive(cfg[key], value)
        else:
            cfg[key] = copy.deepcopy(value)
    return cfg


def warn_with_traceback(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    """Replacement for `warnings.showwarning` that also prints the call stack.

    Installed by `activate_verbose_warnings` to find where a warning originates.
    The signature matches `warnings.showwarning`.
    """
    stack = "".join(traceback.format_stack()[:-1])  # drop this function's own frame
    indented = textwrap.indent(stack, "\t| ")
    log = file if file is not None else sys.stderr
    log.write(warnings.formatwarning(f"{message}\n{indented}", category, filename, lineno, line))


def activate_verbose_warnings() -> None:
    """Print the call stack with every warning, for the rest of this process."""
    warnings.showwarning = warn_with_traceback



def onehot_to_integer_encoding(
    onehot: ArrayLike, none_index: int = -1, axis: int = -1
) -> Array:
    """Convert a one-hot encoding to integer indices.

    Args:
        onehot: boolean one-hot array. At most one entry per slice along `axis`
            should be True; if several are, the first one wins.
        none_index: the value for slices with no True entry
        axis: the axis holding the one-hot dimension

    Returns:
        The index of the True entry per slice, or `none_index`.
        shape: `onehot.shape` without `axis`
    """
    onehot = jnp.asarray(onehot)
    return jnp.where(onehot.any(axis=axis), jnp.argmax(onehot, axis=axis), none_index)


def sample_categorical(p: ArrayLike, binary_encoding: bool = False) -> NDArray:
    """Draw one sample per site from categorical distributions.

    Uses NumPy's global random state; seed it with `np.random.seed` for
    reproducible samples.

    Args:
        p: probabilities of each state at each site. The last axis holds the states
            and must sum to 1 (small float drift is corrected).
            shape: (*output_dims, n_states)
        binary_encoding: return one-hot samples instead of state indices

    Returns:
        The sampled state indices, shape: output_dims,
        or their one-hot encoding, shape: (*output_dims, n_states).
    """
    p = np.asarray(p)
    *output_dims, n_states = p.shape

    if not np.all(p >= 0):  # also catches NaN
        raise ValueError("Probabilities must be non-negative and not NaN.")
    cdf = np.cumsum(p, axis=-1)
    if not np.allclose(cdf[..., -1], 1.0, atol=1e-3):
        raise ValueError("Probabilities must sum to 1 along the last axis.")
    cdf /= cdf[..., -1:]

    u = np.random.random((*output_dims, 1))
    samples = np.argmax(u < cdf, axis=-1)

    if binary_encoding:
        return np.eye(n_states, dtype=bool)[samples]
    return samples