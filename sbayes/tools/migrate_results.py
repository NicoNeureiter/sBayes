"""Migrate old-format sBayes results to the new consolidated format.

Adds JSON metadata attribute and /derived/likelihood group to existing
samples_*.h5 files, making them compatible with Results.from_h5().

Usage:
    python -m sbayes.tools.migrate_results /path/to/results/
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import tables

from sbayes.results import Results

# Suppress PyTables warnings about natural names (e.g. "Categorical[2]")
warnings.simplefilter("ignore", category=tables.NaturalNameWarning)


def discover_partitions(h5_keys: list[str]) -> list[dict]:
    """Identify partitions and their types from h5 root-level key names.

    Returns a list of partition dicts with '_name', 'type', and
    'cluster_effect_keys' populated.  Keys ending in '_raw' (NumPyro
    unconstrained representations) are ignored.
    """
    partitions = []
    seen_names = set()

    cluster_effect_keys = sorted(
        k for k in h5_keys
        if k.startswith("cluster_effect_") and not k.endswith("_raw")
    )

    for key in cluster_effect_keys:
        stem = key[len("cluster_effect_"):]

        if stem.endswith("_variance"):
            # Paired with _mean — skip
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


def get_partition_n_features(h5_file: tables.File, partition: dict) -> int:
    """Get the number of features from the first cluster effect array shape.

    Old-format array shapes: (n_samples, n_chains, n_clusters, n_features[, n_states])
    """
    key = partition["cluster_effect_keys"][0]
    return h5_file.root._v_children[key].shape[3]


def get_partition_state_names(h5_file: tables.File, partition: dict) -> list[str]:
    """Determine state names based on partition type and array shape."""
    if partition["type"] == "categorical":
        key = partition["cluster_effect_keys"][0]
        n_states = h5_file.root._v_children[key].shape[4]
        return [f"s{s}" for s in range(n_states)]
    elif partition["type"] == "gaussian":
        return ["mean", "variance"]
    elif partition["type"] == "poisson":
        return ["rate"]


def build_confounder_effect_keys(
    partition: dict, confounder_names: list[str], h5_keys: set[str]
) -> dict[str, list[str]]:
    """Build confounder_effect_keys mapping for a partition."""
    result = {}
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
    """Extract component names from weight columns (w_{comp}_{feature})."""
    if not feature_names:
        return []
    first_f = feature_names[0]
    suffix = f"_{first_f}"
    components = []
    for c in columns:
        if c.startswith("w_") and c.endswith(suffix):
            comp = c[len("w_"): -len(suffix)]
            if comp not in components:
                components.append(comp)
    return components


def infer_feature_partition_key(
    columns: list[str], feature_names: list[str], first_cluster: str
) -> dict[str, tuple[str, int]]:
    """For each feature, infer its partition type and n_states from stats columns.

    Returns {feature_name: (type_str, n_states)}.
    Categorical features: states are s0, s1, ... → ("categorical", n_states)
    Gaussian features: states are mean, variance → ("gaussian", 2)
    Poisson features: states are rate → ("poisson", 1)
    """
    prefix = f"areal_{first_cluster}_"
    result = {}
    for f in feature_names:
        f_prefix = f"{prefix}{f}_"
        states = [c[len(f_prefix):] for c in columns if c.startswith(f_prefix)]
        if set(states) == {"mean", "variance"}:
            result[f] = ("gaussian", 2)
        elif set(states) == {"rate"}:
            result[f] = ("poisson", 1)
        else:
            result[f] = ("categorical", len(states))
    return result


def assign_features_to_partitions(
    partitions: list[dict],
    feature_names: list[str],
    feature_keys: dict[str, tuple[str, int]],
    h5_file: tables.File,
) -> None:
    """Assign feature names to partitions based on type and n_states match.

    Modifies partitions in-place, adding 'feature_names' to each.
    Features within each partition preserve their global order.
    """
    for p in partitions:
        p_type = p["type"]
        if p_type == "categorical":
            key = p["cluster_effect_keys"][0]
            n_states = h5_file.root._v_children[key].shape[4]
            match_key = ("categorical", n_states)
        elif p_type == "gaussian":
            match_key = ("gaussian", 2)
        elif p_type == "poisson":
            match_key = ("poisson", 1)

        p["feature_names"] = [
            f for f in feature_names if feature_keys[f] == match_key
        ]


def migrate_one(samples_h5_path: Path, force: bool = False) -> bool:
    """Migrate a single samples_*.h5 file.

    Returns True if migration was performed, False if skipped.
    """
    # Parse K and run index from path: .../K{n}/samples_{run}.h5
    k_folder = samples_h5_path.parent.name
    k = int(k_folder[1:])
    run_id = int(samples_h5_path.stem.rpartition("_")[-1])

    # Find companion stats file (.tsv or .txt, with or without K in filename)
    stats_candidates = [
        samples_h5_path.parent / f"stats_K{k}_{run_id}.tsv",
        samples_h5_path.parent / f"stats_K{k}_{run_id}.txt",
        samples_h5_path.parent / f"stats_{run_id}.tsv",
        samples_h5_path.parent / f"stats_{run_id}.txt",
    ]
    # print(stats_candidates)
    stats_path = next((p for p in stats_candidates if p.exists()), None)
    if stats_path is None:
        warnings.warn(f"No stats file found for {samples_h5_path}, skipping.")
        return False

    # Find companion likelihood file (with or without K in filename)
    likelihood_candidates = [
        samples_h5_path.parent / f"likelihood_K{k}_{run_id}.h5",
        samples_h5_path.parent / f"likelihood_{run_id}.h5",
    ]
    likelihood_path = next((p for p in likelihood_candidates if p.exists()), None)

    # Check current state
    with tables.open_file(str(samples_h5_path), mode="r") as f:
        has_metadata = hasattr(f.root._v_attrs, "metadata")
        has_derived = (
            hasattr(f.root, "derived")
            and hasattr(f.root.derived, "likelihood")
        )

    need_metadata = not has_metadata or force
    need_derived = not has_derived and likelihood_path is not None

    if not need_metadata and not need_derived:
        print(f"  Already migrated — skipping.")
        return False

    # --- Build metadata from stats + h5 structure ---
    if need_metadata:
        stats_df = Results.read_stats(stats_path)
        columns = stats_df.columns.tolist()

        cluster_names = Results.get_cluster_names(columns)
        feature_names = [
            c[len("w_areal_"):] for c in columns if c.startswith("w_areal_")
        ]
        component_names = extract_component_names(columns, feature_names)
        groups_by_confounders = Results.get_groups_by_confounder(columns)
        confounder_names = list(groups_by_confounders.keys())

        # Discover partitions from h5 keys and assign features by type
        feature_keys = infer_feature_partition_key(columns, feature_names, cluster_names[0])

        with tables.open_file(str(samples_h5_path), mode="r") as f:
            h5_keys = set(f.root._v_children.keys())
            partitions = discover_partitions(list(h5_keys))
            assign_features_to_partitions(partitions, feature_names, feature_keys, f)

            for p in partitions:
                p["state_names"] = get_partition_state_names(f, p)
                p["confounder_effect_keys"] = build_confounder_effect_keys(
                    p, confounder_names, h5_keys
                )
                # Remove internal key (not part of the output format)
                del p["_name"]

        assigned = sum(len(p["feature_names"]) for p in partitions)
        if assigned != len(feature_names):
            warnings.warn(
                f"Feature count mismatch: stats has {len(feature_names)} features "
                f"but partitions cover {assigned}. Skipping."
            )
            return False

        metadata = {
            "cluster_names": cluster_names,
            "feature_names": feature_names,
            "component_names": component_names,
            "confounders": groups_by_confounders,
            "partitions": partitions,
        }

    # --- Write to h5 ---
    with tables.open_file(str(samples_h5_path), mode="a") as f:
        if need_metadata:
            f.root._v_attrs.metadata = json.dumps(metadata)
            print(f"  Wrote metadata ({len(feature_names)} features, "
                  f"{len(partitions)} partition(s), {len(confounder_names)} confounder(s))")

        if need_derived:
            with tables.open_file(str(likelihood_path), mode="r") as lh_file:
                if not hasattr(lh_file.root, "likelihood"):
                    warnings.warn(f"No likelihood data in {likelihood_path.name}")
                else:
                    lh_data = lh_file.root.likelihood[:]

                    if "/derived" not in f:
                        f.create_group(f.root, "derived")

                    lh_filters = tables.Filters(
                        complevel=9, complib="blosc:zlib",
                        bitshuffle=True, fletcher32=True,
                    )
                    f.create_carray(
                        where=f.root.derived, name="likelihood",
                        obj=lh_data, atom=tables.Float64Col(),
                        filters=lh_filters,
                    )
                    print(f"  Copied likelihood {lh_data.shape} from {likelihood_path.name}")

                    if hasattr(lh_file.root, "na_values"):
                        na_data = lh_file.root.na_values[:]
                        f.create_carray(
                            where=f.root.derived, name="na_values",
                            obj=na_data, atom=tables.BoolCol(),
                            filters=tables.Filters(complevel=9, fletcher32=True),
                        )
                        print(f"  Copied na_values from {likelihood_path.name}")

        elif not has_derived and likelihood_path is None:
            print(f"  No likelihood file found — /derived group not created.")

    return True


def main(results_dir: Path, force: bool = False):
    """Migrate all old-format results in a directory tree."""
    samples_files = sorted(results_dir.rglob("samples_*.h5"))

    # Skip hot-chain files from MC3
    samples_files = [p for p in samples_files if ".chain" not in p.name]

    if not samples_files:
        print(f"No samples_*.h5 files found in {results_dir}")
        return

    print(f"Found {len(samples_files)} samples h5 file(s) in {results_dir}\n")

    migrated = 0
    for path in samples_files:
        rel = path.relative_to(results_dir)
        print(f"[{rel}]")
        try:
            if migrate_one(path, force=force):
                migrated += 1
        except Exception as e:
            warnings.warn(f"Error migrating {path}: {e}")

    print(f"\nDone. Migrated {migrated}/{len(samples_files)} file(s).")


def cli():
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
    return main(args.results_dir, force=args.force)


if __name__ == "__main__":
    cli()
