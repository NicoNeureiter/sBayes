"""Guess the feature types of sBayes data files and write them to a feature_types.yaml.

Usage:
    python -m sbayes.tools.guess_feature_types --input features.csv --output feature_types.yaml --excludeColumns family

Arguments that are not given on the command line are asked for in dialogs. Pass
`--excludeColumns` without values to exclude no columns without opening a dialog.
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Callable, Collection, Sequence, TypeVar

from ruamel.yaml import YAML
from sbayes.load_data import FeatureType
from sbayes.tools._dialogs import LazyRoot, ask_save_file
from sbayes.util import PathLike, read_data_csv


T = TypeVar("T")
REQUIRED_COLUMNS = ("id",)
METADATA_COLUMNS = ("id", "name", "x", "y")


# --- Core ----------------------------------------------------------------------------

def read_feature_columns(csv_path: PathLike) -> list[str]:
    """Return the names of all non-metadata columns in a data file.

    Raises:
        ValueError: if a required column is missing.
    """
    data = read_data_csv(csv_path)
    _check_required_columns(data.columns, csv_path)
    return [c for c in data.columns if c not in METADATA_COLUMNS]


def collect_feature_states(
    csv_path: PathLike, exclude_columns: Collection[str] = ()
) -> dict[str, set[str]]:
    """Collect the observed (non-missing) states of every feature in a data file.

    Args:
        csv_path: the data file
        exclude_columns: non-feature columns to skip, e.g. confounders

    Returns:
        The set of observed states per feature.

    Raises:
        ValueError: if a required column is missing or an excluded column does not exist.
    """
    data = read_data_csv(csv_path)
    _check_required_columns(data.columns, csv_path)

    unknown = set(exclude_columns) - set(data.columns)
    if unknown:
        raise ValueError(f"Excluded columns not found in {csv_path}: {sorted(unknown)}")

    skip = set(METADATA_COLUMNS) | set(exclude_columns)
    return {
        str(c): {str(v) for v in data[c].dropna()}
        for c in data.columns
        if c not in skip
    }


def merge_feature_states(
    per_file: dict[PathLike, dict[str, set[str]]],
) -> dict[str, set[str]]:
    """Combine the feature states of several data files.

    Raises:
        ValueError: if the files do not contain the same features.
    """
    merged: dict[str, set[str]] = {}
    for path, states in per_file.items():
        if merged and merged.keys() != states.keys():
            raise ValueError(
                f"Features do not match between the input files:\n"
                f"\tMissing in {path}: {sorted(merged.keys() - states.keys())}\n"
                f"\tOnly in {path}: {sorted(states.keys() - merged.keys())}"
            )
        for feature, values in states.items():
            merged.setdefault(feature, set()).update(values)
    return merged


def guess_feature_type(states: Collection[str]) -> FeatureType:
    """Guess the type of a feature from its observed states.

    - categorical: non-numeric states, or only the values 0 and 1
    - poisson: non-negative integer values
    - logitnormal: numbers between 0 and 1 (inclusive), at least one not an integer
    - gaussian: any other numbers

    Integer values are recognised regardless of formatting, so `3` and `3.0` are
    treated the same.

    Args:
        states: the observed, non-missing states. Must not be empty.
    """
    if not states:
        raise ValueError("Cannot guess the type of a feature without observed states.")

    floats = _parse_all(states, float)
    if floats is None or not all(math.isfinite(v) for v in floats):
        return FeatureType.categorical

    ints = [int(v) for v in floats] if all(v.is_integer() for v in floats) else None
    if ints is not None:
        if set(ints) <= {0, 1}:
            return FeatureType.categorical
        return FeatureType.poisson if min(ints) >= 0 else FeatureType.gaussian

    if all(0 <= v <= 1 for v in floats):
        return FeatureType.logitnormal
    return FeatureType.gaussian


def describe_feature(states: Collection[str]) -> dict:
    """Build the feature_types.yaml entry for one feature.

    Categorical features list their states in alphabetical order; numeric features
    give the observed range.
    """
    feature_type = guess_feature_type(states)

    if feature_type is FeatureType.categorical:
        return {"type": feature_type.value, "states": sorted(states)}

    values = [float(s) for s in states]
    if feature_type is FeatureType.poisson:
        values = [int(v) for v in values]
    return {"type": feature_type.value, "states": {"min": min(values), "max": max(values)}}


def build_feature_types(feature_states: dict[str, set[str]]) -> dict[str, dict]:
    """Build the feature_types.yaml content for all features.

    Raises:
        ValueError: if a feature has no observed values. sBayes cannot load such
            features, so they must be removed from the data.
    """
    empty = [f for f, states in feature_states.items() if not states]
    if empty:
        raise ValueError(
            f"Features without any observed values: {empty}. "
            f"Remove these columns from the data."
        )

    feature_types = {f: describe_feature(states) for f, states in feature_states.items()}

    single_state = [
        f for f, entry in feature_types.items()
        if entry["type"] == FeatureType.categorical.value and len(entry["states"]) < 2
    ]
    if single_state:
        warnings.warn(f"Categorical features with only one observed state: {single_state}")

    return feature_types


def write_feature_types(feature_types: dict[str, dict], output_path: PathLike) -> None:
    """Write the feature types to a YAML file."""
    yml = YAML()
    yml.indent(mapping=2, sequence=4, offset=2)
    yml.default_flow_style = False
    with open(output_path, "w", encoding="utf-8") as f:
        yml.dump(feature_types, f)


def _check_required_columns(columns: Collection[str], csv_path: PathLike) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"Required columns {missing} missing in {csv_path}.")


def _parse_all(states: Collection[str], cast: Callable[[str], T]) -> list[T] | None:
    """Convert all states with `cast`, or return None if any of them fails."""
    try:
        return [cast(s) for s in states]
    except ValueError:
        return None


# --- GUI -----------------------------------------------------------------------------
# tkinter is imported lazily so that the command-line path works without Tk.

def ask_input_files(root: LazyRoot) -> list[Path]:
    """Ask for one or more data files, possibly from several directories."""
    from tkinter import filedialog, messagebox

    paths: list[Path] = []
    directory = "."
    while True:
        selected = filedialog.askopenfilenames(
            parent=root.get(),
            title="Select data files in CSV format.",
            initialdir=directory,
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if selected:
            paths.extend(Path(p) for p in selected)
            directory = str(paths[-1].parent)
        if not messagebox.askyesno(
            "Additional data files", "Would you like to add more data files?",
            parent=root.get(),
        ):
            return paths


def ask_confounders(root: LazyRoot, columns: Sequence[str]) -> list[str] | None:
    """Ask which columns are confounders. Returns None if the dialog is closed."""
    import tkinter as tk

    parent = root.get()
    window = tk.Toplevel(parent)
    window.title("Select confounders")
    height = min(800, window.winfo_screenheight() - 100)
    window.geometry(f"400x{height}")

    result: list[str] | None = None
    variables = {c: tk.BooleanVar(window) for c in columns}

    def submit() -> None:
        nonlocal result
        result = [c for c, var in variables.items() if var.get()]
        window.destroy()

    # Pack the button first so it stays visible when the list is long
    tk.Button(window, text="Submit", command=submit).pack(side="bottom", pady=10)

    container = tk.Frame(window)
    container.pack(fill="both", expand=True, padx=10, pady=10)
    canvas = tk.Canvas(container)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    frame = tk.Frame(canvas)
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    for c, var in variables.items():
        tk.Checkbutton(frame, text=c, variable=var, anchor="w").pack(fill="x")

    window.protocol("WM_DELETE_WINDOW", window.destroy)
    parent.wait_window(window)
    return result


# --- Entry point ---------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guess the feature types of sBayes data files and write a feature_types.yaml."
    )
    parser.add_argument("--input", nargs="+", type=Path,
                        help="The input features CSV file(s). Asked for in a dialog if omitted.")
    parser.add_argument("--output", type=Path,
                        help="The output YAML file. Asked for in a dialog if omitted.")
    parser.add_argument("--excludeColumns", nargs="*",
                        help="Non-feature columns to exclude, e.g. confounders. Asked for in a "
                             "dialog if omitted; pass the flag without values to exclude nothing.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_paths: list[Path] | None = args.input
    output_path: Path | None = args.output
    exclude_columns: list[str] | None = args.excludeColumns

    root = LazyRoot()
    try:
        if input_paths is None:
            input_paths = ask_input_files(root)
            if not input_paths:
                sys.exit("No input files selected.")

        if exclude_columns is None:
            exclude_columns = ask_confounders(root, read_feature_columns(input_paths[0]))
            if exclude_columns is None:
                sys.exit("Cancelled.")

        feature_states = merge_feature_states(
            {p: collect_feature_states(p, exclude_columns) for p in input_paths}
        )
        feature_types = build_feature_types(feature_states)

        if output_path is None:
            output_path = ask_save_file(
                root, "Select an output file in YAML format.",
                [("YAML files", "*.yaml")], default_name="feature_types.yaml",
                directory=input_paths[0].parent,
            )
            if output_path is None:
                sys.exit("No output file selected.")
    finally:
        root.destroy()

    write_feature_types(feature_types, output_path)
    print(f"Wrote the types of {len(feature_types)} features to {output_path}")


if __name__ == "__main__":
    main()