"""Convert Dirichlet prior counts from a CSV table to the JSON format read by sBayes.

The CSV has a `feature` column and one column per state; empty cells mean the state
does not occur for that feature.

Usage:
    python -m sbayes.tools.convert_prior_csv_to_json --csv counts.csv --output counts.json
"""
from __future__ import annotations

import argparse
import json
import pandas as pd

from pathlib import Path
from sbayes.util import PathLike, normalize_str
from typing import Sequence


def _as_count(v: float) -> int | float:
    """Return `v` as an int if it is a whole number."""
    return int(v) if v.is_integer() else v


def read_prior_counts(csv_path: PathLike) -> dict[str, dict[str, int | float]]:
    """Read prior counts per feature and state from a CSV file.

    Raises:
        ValueError: if the `feature` column is missing, a feature appears twice, or
            a count is not numeric.
    """
    counts = pd.read_csv(csv_path)
    counts.columns = [normalize_str(c) for c in counts.columns]
    if "feature" not in counts.columns:
        raise ValueError(f"Required column 'feature' missing in {csv_path}.")

    counts["feature"] = counts["feature"].map(normalize_str)
    duplicates = counts["feature"][counts["feature"].duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate features in {csv_path}: {duplicates}")
    counts = counts.set_index("feature")

    try:
        counts = counts.astype(float)
    except ValueError as e:
        raise ValueError(f"Non-numeric prior counts in {csv_path}: {e}") from e

    result: dict[str, dict[str, int | float]] = {}
    for feature, row in counts.iterrows():
        row = row.dropna()
        result[str(feature)] = {
            str(state): _as_count(float(v))
            for state, v in zip(row.index, row.to_numpy(dtype=float))
        }
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert Dirichlet prior counts from CSV to JSON."
    )
    parser.add_argument("--csv", type=Path, required=True, help="The input CSV file")
    parser.add_argument("--output", type=Path, required=True, help="The output JSON file")
    args = parser.parse_args(argv)

    counts = read_prior_counts(args.csv)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()