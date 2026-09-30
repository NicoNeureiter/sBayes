"""Estimate an empirical prior for categorical features from sBayes data files.

For every categorical feature, the observed states are counted and turned into
Dirichlet concentration parameters. Counts are taken either over all objects (the
universal prior) or separately for each group of a confounder (e.g. one prior per
language family). One JSON file is written per group, `{group}_prior.json`.

Usage:
    python -m sbayes.tools.estimate_empirical_prior \\
        --data prior_data.csv --featureTypes feature_types.yaml --output priors/ \\
        --confounder family --concentration 1.0 --maxCounts 10

Arguments that are not given on the command line are asked for in dialogs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from ruamel.yaml import YAML

from sbayes.load_data import CategoricalFeatures, Features, read_features_from_csv
from sbayes.tools._dialogs import LazyRoot, ask_directory, ask_open_file
from sbayes.util import PathLike, cap_counts, read_data_csv

UNIVERSAL = "universal"
"""Pseudo-confounder: count over all objects instead of per group."""

METADATA_COLUMNS = ("id", "name", "x", "y")

PriorCounts = dict[str, dict[str, float]]
"""Concentration parameters per feature and state."""


# --- Core ----------------------------------------------------------------------------

def count_states(
    partition: CategoricalFeatures, members: NDArray[np.bool_] | None = None
) -> NDArray[np.int_]:
    """Count how often each state of each feature occurs.

    Missing values are not counted.

    Args:
        partition: the categorical features
        members: the objects to count over. shape: (n_objects,). None counts all objects.

    Returns:
        The counts. shape: (n_features, n_states)
    """
    binary = partition.to_binary()  # shape: (n_objects, n_features, n_states)
    if members is not None:
        binary = binary[members]
    return binary.sum(axis=0)


def concentration_from_counts(
    counts: NDArray, concentration: float, max_counts: float | None
) -> NDArray[np.floating]:
    """Turn state counts into Dirichlet concentration parameters.

    The hyper-prior concentration is added to every state first, then the total per
    feature is capped at `max_counts`, keeping the proportions. The result therefore
    never exceeds `max_counts`, including the hyper-prior contribution.

    Args:
        counts: the state counts. shape: (n_features, n_states)
        concentration: the hyper-prior concentration added to each state
            (1.0 corresponds to a uniform hyper-prior)
        max_counts: the maximum total concentration per feature, or None for no cap

    Returns:
        The concentration parameters. shape: (n_features, n_states)
    """
    alpha = counts + concentration
    if max_counts is not None:
        alpha = cap_counts(alpha, max_counts)
    return alpha


def build_prior_counts(
    features: Features,
    members: NDArray[np.bool_] | None,
    concentration: float,
    max_counts: float | None,
) -> PriorCounts:
    """Compute the concentration parameters of all categorical features for one group.

    Non-categorical features are skipped.

    Args:
        features: all features of the data
        members: the objects of the group. shape: (n_objects,). None for all objects.
        concentration: the hyper-prior concentration added to each state
        max_counts: the maximum total concentration per feature, or None for no cap

    Returns:
        The concentration parameters per feature and state.
    """
    prior: PriorCounts = {}
    for partition in features.partitions:
        if not isinstance(partition, CategoricalFeatures):
            continue

        alpha = concentration_from_counts(
            count_states(partition, members), concentration, max_counts
        )
        for i_f, feature in enumerate(partition.names):
            prior[str(feature)] = {
                str(state): float(alpha[i_f, i_s])
                for i_s, state in enumerate(partition.state_names[i_f])
            }
    return prior


def load_features(
    data_path: PathLike, feature_types_path: PathLike, confounder: str
) -> tuple[Features, dict[str, NDArray[np.bool_] | None]]:
    """Load the data and the objects of each group to count over.

    Every column that is neither metadata nor a feature is loaded as a confounder,
    since the loader rejects unrecognised columns.

    Args:
        data_path: the empirical data (CSV)
        feature_types_path: the feature types (YAML)
        confounder: the confounder whose groups get separate priors, or `universal`

    Returns:
        The features, and the member mask per group (None for the universal prior).

    Raises:
        ValueError: if `confounder` is not a column of the data.
    """
    columns = list(read_data_csv(data_path).columns)
    feature_names = set(YAML(typ="safe").load(Path(feature_types_path)))
    other_columns = [
        c for c in columns if c not in METADATA_COLUMNS and c not in feature_names
    ]

    if confounder != UNIVERSAL and confounder not in columns:
        raise ValueError(f"Confounder column '{confounder}' not found in {data_path}.")

    _, features, confounders = read_features_from_csv(
        data_path=data_path,
        feature_types_path=feature_types_path,
        confounder_names=list(dict.fromkeys([confounder, *other_columns])),
    )

    if confounder == UNIVERSAL:
        return features, {UNIVERSAL: None}

    groups = confounders[confounder]
    return features, {
        str(name): groups.group_assignment[i]
        for i, name in enumerate(groups.group_names)
    }


def prior_file_name(group: str) -> str:
    """Build the output file name for a group, replacing characters unsafe in file names."""
    return re.sub(r"[^\w.-]", "_", group) + "_prior.json"


def write_priors(priors: dict[str, PriorCounts], output_dir: Path) -> None:
    """Write one JSON file per group.

    Raises:
        ValueError: if two group names map to the same file name.
    """
    file_names = {group: prior_file_name(group) for group in priors}
    if len(set(file_names.values())) < len(file_names):
        raise ValueError(f"Group names collide as file names: {file_names}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for group, prior in priors.items():
        path = output_dir / file_names[group]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(prior, f, indent=4, ensure_ascii=False)
        print(f"Wrote the prior for {group} ({len(prior)} features) to {path}")


# --- Entry point ---------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate an empirical prior for categorical features."
    )
    parser.add_argument("--data", type=Path,
                        help="Empirical data as a CSV file. Asked for in a dialog if omitted.")
    parser.add_argument("--featureTypes", type=Path,
                        help="Feature types YAML file. Asked for in a dialog if omitted.")
    parser.add_argument("--output", type=Path,
                        help="Directory for the output files. Asked for in a dialog if omitted.")
    parser.add_argument("--confounder", default=UNIVERSAL,
                        help="Confounder whose groups get separate priors, or "
                             f"'{UNIVERSAL}' for one prior over all objects (default).")
    parser.add_argument("--concentration", type=float, default=1.0,
                        help="Hyper-prior concentration added to each state "
                             "(1.0 corresponds to a uniform hyper-prior).")
    parser.add_argument("--maxCounts", type=float, default=None,
                        help="Maximum total concentration per feature, including the "
                             "hyper-prior (default: no cap).")
    args = parser.parse_args(argv)

    if args.concentration < 0:
        parser.error("--concentration must be non-negative.")
    if args.maxCounts is not None and args.maxCounts <= 0:
        parser.error("--maxCounts must be positive.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_path: Path | None = args.data
    feature_types_path: Path | None = args.featureTypes
    output_dir: Path | None = args.output

    root = LazyRoot()
    try:
        if data_path is None:
            data_path = ask_open_file(
                root, "Select the empirical data as a CSV file.", [("CSV files", "*.csv")]
            )
            if data_path is None:
                sys.exit("No data file selected.")

        if feature_types_path is None:
            feature_types_path = ask_open_file(
                root, "Select the feature types YAML file.",
                [("YAML files", "*.yaml *.yml")], directory=data_path.parent,
            )
            if feature_types_path is None:
                sys.exit("No feature types file selected.")

        if output_dir is None:
            output_dir = ask_directory(
                root, "Select a directory for the output files.", directory=data_path.parent
            )
            if output_dir is None:
                sys.exit("No output directory selected.")
    finally:
        root.destroy()

    features, groups = load_features(data_path, feature_types_path, args.confounder)
    priors = {
        group: build_prior_counts(features, members, args.concentration, args.maxCounts)
        for group, members in groups.items()
    }
    write_priors(priors, output_dir)


if __name__ == "__main__":
    main()