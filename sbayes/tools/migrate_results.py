"""Migrate old-format sBayes results to the new consolidated format.

Adds a JSON metadata attribute and a /derived/likelihood group to existing
samples_*.h5 files, making them compatible with Results.from_h5().

This tool reads the legacy format only. Its key names (`areal_` stats columns,
mixed confounder key conventions) describe files written by older sBayes versions
and must not follow later renames.

Usage:
    python -m sbayes.tools.migrate_results /path/to/results/
"""
from __future__ import annotations

import argparse
import json
import re
import tables
import warnings

from pathlib import Path
from sbayes.results import Results
from sbayes.tools._h5 import read_array
from typing import Any


# Suppress PyTables warnings about natural names (e.g. "Categorical[2]")
warnings.simplefilter("ignore", category=tables.NaturalNameWarning)

Partition = dict[str, Any]
"""A partition entry of the metadata: type, feature names, state names and the h5
keys of its cluster and confounder effects."""

GAUSSIAN_KEY = ("gaussian", 2)
POISSON_KEY = ("poisson", 1)

# noinspection PyTypeChecker
# "blosc:zlib" is valid; PyTables' annotation only lists the base library names.
LIKELIHOOD_FILTERS = tables.Filters(
    complevel=9, complib="blosc:zlib", bitshuffle=True, fletcher32=True
)
NA_VALUES_FILTERS = tables.Filters(complevel=9, fletcher32=True)

# --- h5 helpers ----------------------------------------------------------------------

def root_keys(h5_file: tables.File) -> set[str]:
    """Return the names of all nodes directly under the root group.

    Only the top level is listed; nodes inside groups such as `/derived` are not
    included. Legacy samples files store all parameter arrays at the root, so these
    are the parameter keys.

    Args:
        h5_file: an open h5 file

    Returns:
        The node names, without the leading `/`.
    """
    return {node._v_name for node in h5_file.list_nodes("/")}


def has_root_attr(h5_file: tables.File, name: str) -> bool:
    """Check whether the root group has the attribute `name`.

    PyTables has no direct existence check for node attributes, so this tries to read
    the attribute and treats the `AttributeError` raised for a missing one as False.

    Args:
        h5_file: an open h5 file
        name: the attribute name, e.g. `metadata`

    Returns:
        True if the attribute exists, otherwise False.
    """
    try:
        h5_file.get_node_attr("/", name)
    except AttributeError:
        return False
    return True


def node_shape(h5_file: tables.File, key: str) -> tuple[int, ...]:
    """Return the shape of the array stored at a root-level node.

    Legacy samples files store each parameter as an array at the root, e.g.
    `cluster_effect_{partition}` with shape
    (n_samples, n_chains, n_clusters, n_features, n_states).

    Args:
        h5_file: an open h5 file
        key: the node name, without the leading `/`

    Returns:
        The shape of the array.

    Raises:
        ValueError: if the node is not an array.
    """
    node = h5_file.get_node("/", key)
    if not isinstance(node, tables.Array) or node.shape is None:
        raise ValueError(f"Expected an array at /{key}, found {type(node).__name__}.")
    return tuple(int(n) for n in node.shape)


# --- Metadata reconstruction ---------------------------------------------------------

def discover_partitions(h5_keys: list[str]) -> list[Partition]:
    """Identify partitions and their types from the root-level h5 key names.

    Keys ending in `_raw` (NumPyro unconstrained representations) are ignored.

    Args:
        h5_keys: the names of the root-level nodes

    Returns:
        One partition per discovered cluster effect, with `_name`, `type` and
        `cluster_effect_keys` set.
    """
    partitions: list[Partition] = []
    seen_names: set[str] = set()

    cluster_effect_keys = sorted(
        k for k in h5_keys
        if k.startswith("cluster_effect_") and not k.endswith("_raw")
    )

    for key in cluster_effect_keys:
        stem = key[len("cluster_effect_"):]

        if stem.endswith("_variance"):
            # Covered by the matching _mean key
            continue
        elif stem.endswith("_mean"):
            partition_name = stem[: -len("_mean")]
            partitions.append({
                "_name": partition_name,
                "type": "gaussian",
                "cluster_effect_keys": [
                    f"cluster_effect_{partition_name}_mean",
                    f"cluster_effect_{partition_name}_variance",
                ],
            })
        elif stem.endswith("_rate"):
            partition_name = stem[: -len("_rate")]
            partitions.append({
                "_name": partition_name,
                "type": "poisson",
                "cluster_effect_keys": [f"cluster_effect_{partition_name}_rate"],
            })
        else:
            partition_name = stem
            if partition_name in seen_names:
                continue
            partitions.append({
                "_name": partition_name,
                "type": "categorical",
                "cluster_effect_keys": [f"cluster_effect_{partition_name}"],
            })

        seen_names.add(partition_name)

    return partitions


def get_partition_state_names(h5_file: tables.File, partition: Partition) -> list[str]:
    """Return the state names of a partition.

    Legacy files do not store state names, so they are reconstructed:
    categorical states are numbered `s0`, `s1`, ... after the last axis of the
    cluster effect array, whose legacy shape is
    (n_samples, n_chains, n_clusters, n_features, n_states). Gaussian and Poisson
    partitions have fixed names for their parameters.

    Args:
        h5_file: the open legacy samples file
        partition: the partition, with `type` and `cluster_effect_keys` set

    Returns:
        `["s0", "s1", ...]` for categorical, `["mean", "variance"]` for Gaussian,
        `["rate"]` for Poisson partitions.

    Raises:
        ValueError: if the partition type is unknown.
    """
    p_type = partition["type"]
    if p_type == "categorical":
        n_states = node_shape(h5_file, partition["cluster_effect_keys"][0])[4]
        return [f"s{s}" for s in range(n_states)]
    if p_type == "gaussian":
        return ["mean", "variance"]
    if p_type == "poisson":
        return ["rate"]
    raise ValueError(f"Unknown partition type: {p_type}")


def build_confounder_effect_keys(
    partition: Partition, confounder_names: list[str], h5_keys: set[str]
) -> dict[str, list[str]]:
    """Map each confounder to the h5 keys of its effect on a partition.

    Legacy files name categorical confounder effects by confounder name, but
    Gaussian and Poisson ones by confounder index. Both conventions are read here.

    Args:
        partition: the partition, with `_name` and `type` set
        confounder_names: the confounders, in the order they were indexed
        h5_keys: the names of the root-level nodes

    Returns:
        The effect keys per confounder. Confounders without stored effects are omitted.
    """
    result: dict[str, list[str]] = {}
    p_name = partition["_name"]
    p_type = partition["type"]

    for i_c, conf_name in enumerate(confounder_names):
        if p_type == "categorical":
            key = f"conf_effect_{conf_name}_{p_name}"
            if key in h5_keys:
                result[conf_name] = [key]
        elif p_type == "gaussian":
            key_mean = f"conf_effect_{i_c}_{p_name}_mean"
            key_var = f"conf_effect_{i_c}_{p_name}_variance"
            if key_mean in h5_keys:
                result[conf_name] = [key_mean, key_var]
        elif p_type == "poisson":
            key_rate = f"conf_effect_{i_c}_{p_name}_rate"
            if key_rate in h5_keys:
                result[conf_name] = [key_rate]

    return result


def extract_component_names(columns: list[str], feature_names: list[str]) -> list[str]:
    """Extract the mixture component names from the weight columns of a stats file.

    Weight columns are named `w_{component}_{feature}`, e.g. `w_areal_f1` or
    `w_family_f1`. The components are read off the columns of the first feature,
    in the order they appear, since every feature has the same components.

    Args:
        columns: the column names of the stats file
        feature_names: the feature names, in order

    Returns:
        The component names, e.g. `["areal", "family"]`. Empty if there are no features.
    """
    if not feature_names:
        return []
    suffix = f"_{feature_names[0]}"
    components: list[str] = []
    for c in columns:
        if c.startswith("w_") and c.endswith(suffix):
            comp = c[len("w_"): -len(suffix)]
            if comp not in components:
                components.append(comp)
    return components


def infer_feature_partition_key(
    columns: list[str], feature_names: list[str], first_cluster: str
) -> dict[str, tuple[str, int]]:
    """Infer the type and number of states of each feature from a stats file.

    Legacy stats files have one cluster effect column per cluster, feature and
    state, named `areal_{cluster}_{feature}_{state}`. The states are read off the
    columns of the first cluster, since all clusters have the same ones. The result
    is matched against the partitions found in the h5 file to assign features to them.

    Args:
        columns: the column names of the stats file
        feature_names: the feature names, in order
        first_cluster: the name of the first cluster, e.g. `a0`

    Returns:
        `(type, n_states)` per feature: `("gaussian", 2)` for states `mean`/`variance`,
        `("poisson", 1)` for `rate`, and `("categorical", n)` otherwise, where `n` is
        the number of state columns.
    """
    prefix = f"areal_{first_cluster}_"
    result: dict[str, tuple[str, int]] = {}
    for f in feature_names:
        f_prefix = f"{prefix}{f}_"
        states = [c[len(f_prefix):] for c in columns if c.startswith(f_prefix)]
        if set(states) == {"mean", "variance"}:
            result[f] = GAUSSIAN_KEY
        elif set(states) == {"rate"}:
            result[f] = POISSON_KEY
        else:
            result[f] = ("categorical", len(states))
    return result


def assign_features_to_partitions(
    partitions: list[Partition],
    feature_names: list[str],
    feature_keys: dict[str, tuple[str, int]],
    h5_file: tables.File,
) -> None:
    """Assign each feature to the partition with the same type and number of states.

    Legacy files don't record which feature belongs to which partition, so the
    assignment is reconstructed.

    Adds `feature_names` to each partition, in place. Features keep their global order.

    Args:
        partitions: the partitions, with `type` and `cluster_effect_keys` set
        feature_names: all feature names, in order
        feature_keys: `(type, n_states)` per feature, from `infer_feature_partition_key`
        h5_file: the open legacy samples file

    Raises:
        ValueError: if a partition type is unknown.
    """
    for p in partitions:
        p_type = p["type"]
        if p_type == "categorical":
            n_states = node_shape(h5_file, p["cluster_effect_keys"][0])[4]
            match_key = ("categorical", n_states)
        elif p_type == "gaussian":
            match_key = GAUSSIAN_KEY
        elif p_type == "poisson":
            match_key = POISSON_KEY
        else:
            raise ValueError(f"Unknown partition type: {p_type}")

        p["feature_names"] = [f for f in feature_names if feature_keys[f] == match_key]


def build_metadata(samples_h5_path: Path, stats_path: Path) -> dict[str, Any] | None:
    """Reconstruct the metadata of a legacy samples file.

    Args:
        samples_h5_path: the legacy samples file
        stats_path: its companion stats file

    Returns:
        The metadata, or None if the partitions do not account for all features.
    """
    stats_df = Results.read_stats(stats_path)
    columns = stats_df.columns.tolist()

    cluster_names = Results.get_cluster_names(columns)
    feature_names = [c[len("w_areal_"):] for c in columns if c.startswith("w_areal_")]
    component_names = extract_component_names(columns, feature_names)
    groups_by_confounders = Results.get_groups_by_confounder(columns)
    confounder_names = list(groups_by_confounders.keys())
    feature_keys = infer_feature_partition_key(columns, feature_names, cluster_names[0])

    with tables.open_file(str(samples_h5_path), mode="r") as f:
        h5_keys = root_keys(f)
        partitions = discover_partitions(list(h5_keys))
        assign_features_to_partitions(partitions, feature_names, feature_keys, f)

        for p in partitions:
            p["state_names"] = get_partition_state_names(f, p)
            p["confounder_effect_keys"] = build_confounder_effect_keys(
                p, confounder_names, h5_keys
            )
            # Internal key, not part of the output format
            del p["_name"]

    assigned = sum(len(p["feature_names"]) for p in partitions)
    if assigned != len(feature_names):
        warnings.warn(
            f"Feature count mismatch in {samples_h5_path}: stats has "
            f"{len(feature_names)} features but partitions cover {assigned}. Skipping."
        )
        return None

    return {
        "cluster_names": cluster_names,
        "feature_names": feature_names,
        "component_names": component_names,
        "confounders": groups_by_confounders,
        "partitions": partitions,
    }


# --- Migration -----------------------------------------------------------------------

def first_existing(candidates: list[Path]) -> Path | None:
    """Return the first path in `candidates` that exists.

    Used to find companion files whose names changed between sBayes versions:
    the candidates are listed in order of preference.

    Args:
        candidates: the paths to check, in order of preference

    Returns:
        The first existing path, or None if none of them exists.
    """
    return next((p for p in candidates if p.exists()), None)


def copy_likelihood(target: tables.File, likelihood_path: Path) -> None:
    """Copy the likelihood from a legacy likelihood file into `/derived` of `target`.

    Legacy runs wrote the likelihood to a separate `likelihood_*.h5` file; the new
    format stores it in the samples file as `/derived/likelihood`, next to the
    optional NA mask `/derived/na_values`. If the source has no likelihood, a
    warning is issued and nothing is copied.

    Args:
        target: the samples file, open for writing
        likelihood_path: the legacy likelihood file
    """
    with tables.open_file(str(likelihood_path), mode="r") as lh_file:
        if "/likelihood" not in lh_file:
            warnings.warn(f"No likelihood data in {likelihood_path.name}")
            return

        lh_data = read_array(lh_file, "/likelihood")
        if "/derived" not in target:
            target.create_group("/", "derived")

        target.create_carray(
            where="/derived", name="likelihood",
            obj=lh_data, atom=tables.Float64Col(), filters=LIKELIHOOD_FILTERS,
        )
        print(f"  Copied likelihood {lh_data.shape} from {likelihood_path.name}")

        if "/na_values" in lh_file:
            target.create_carray(
                where="/derived", name="na_values",
                obj=read_array(lh_file, "/na_values"), atom=tables.BoolCol(),
                filters=NA_VALUES_FILTERS,
            )
            print(f"  Copied na_values from {likelihood_path.name}")


def migrate_one(samples_h5_path: Path, force: bool = False) -> bool:
    """Migrate a single samples_*.h5 file in place.

    Adds the metadata attribute (unless present and `force` is False) and copies the
    likelihood from the companion likelihood file into `/derived` (if not present).

    Args:
        samples_h5_path: the legacy samples file, located at `.../K{k}/samples_{run}.h5`
        force: rewrite the metadata even if it already exists

    Returns:
        True if the file was changed, False if it was skipped.
    """
    k, run_id = parse_run_location(samples_h5_path)
    directory = samples_h5_path.parent

    # Companion files, with or without K in the file name
    stats_path = first_existing([
        directory / f"stats_K{k}_{run_id}.tsv",
        directory / f"stats_K{k}_{run_id}.txt",
        directory / f"stats_{run_id}.tsv",
        directory / f"stats_{run_id}.txt",
    ])
    if stats_path is None:
        warnings.warn(f"No stats file found for {samples_h5_path}, skipping.")
        return False

    likelihood_path = first_existing([
        directory / f"likelihood_K{k}_{run_id}.h5",
        directory / f"likelihood_{run_id}.h5",
    ])

    with tables.open_file(str(samples_h5_path), mode="r") as f:
        has_metadata = has_root_attr(f, "metadata")
        has_derived = "/derived/likelihood" in f

    need_metadata = force or not has_metadata
    need_derived = not has_derived and likelihood_path is not None

    if not need_metadata and not need_derived:
        print("  Already migrated — skipping.")
        return False

    metadata = build_metadata(samples_h5_path, stats_path) if need_metadata else None
    if need_metadata and metadata is None:
        return False

    with tables.open_file(str(samples_h5_path), mode="a") as f:
        if metadata is not None:
            f.set_node_attr("/", "metadata", json.dumps(metadata))
            print(
                f"  Wrote metadata ({len(metadata['feature_names'])} features, "
                f"{len(metadata['partitions'])} partition(s), "
                f"{len(metadata['confounders'])} confounder(s))"
            )

        if not has_derived:
            if likelihood_path is not None:
                copy_likelihood(f, likelihood_path)
            else:
                print("  No likelihood file found — /derived group not created.")

    return True


def main(results_dir: Path, force: bool = False) -> None:
    """Migrate all legacy samples files in a directory tree.

    Errors in individual files are reported as warnings; the remaining files are
    still migrated.

    Args:
        results_dir: the root directory, containing `K*/` result folders at any depth
        force: rewrite the metadata even where it already exists
    """
    # Legacy MC3 runs also wrote samples files for the heated chains; skip those
    samples_files = sorted(
        p for p in results_dir.rglob("samples_*.h5") if ".chain" not in p.name
    )

    if not samples_files:
        print(f"No samples_*.h5 files found in {results_dir}")
        return

    print(f"Found {len(samples_files)} samples h5 file(s) in {results_dir}\n")

    migrated = 0
    for path in samples_files:
        print(f"[{path.relative_to(results_dir)}]")
        try:
            if migrate_one(path, force=force):
                migrated += 1
        except Exception as e:
            warnings.warn(f"Error migrating {path}: {e}")

    print(f"\nDone. Migrated {migrated}/{len(samples_files)} file(s).")


def cli() -> None:
    """Parse command-line arguments and migrate the results."""
    parser = argparse.ArgumentParser(
        description="Migrate old-format sBayes results to the new consolidated format. "
                    "Adds JSON metadata and /derived/likelihood to samples_*.h5 files."
    )
    parser.add_argument(
        "results_dir", type=Path,
        help="Root directory containing K*/ result folders (or a parent thereof).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-write metadata even if it already exists.",
    )
    args = parser.parse_args()
    main(args.results_dir, force=args.force)


if __name__ == "__main__":
    cli()