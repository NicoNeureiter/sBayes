"""Find pairs of categorical features that are statistically dependent in a data set.

Every pair of features is tested for independence with a chi-squared test on their
contingency table; objects with a missing value in either feature are ignored. The
significant pairs are written to a PDF, one row per pair, showing the expected counts
under independence, the observed counts, and their difference.

Needs matplotlib and seaborn: pip install sbayes[tools]

Usage:
    python -m sbayes.tools.find_correlated_features --input features.csv --output correlations.pdf
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import NamedTuple, Sequence

import pandas as pd
from ruamel.yaml import YAML
from scipy.stats import chi2_contingency

from sbayes.load_data import FeatureType
from sbayes.tools._dialogs import LazyRoot, ask_open_file, ask_save_file
from sbayes.util import PathLike, read_data_csv

METADATA_COLUMNS = ("id", "name", "x", "y")


class CorrelatedPair(NamedTuple):
    """One pair of features whose independence was rejected."""

    p_value: float
    feature_1: str
    feature_2: str
    observed: pd.DataFrame
    expected: pd.DataFrame


# --- Core ----------------------------------------------------------------------------

def select_categorical_features(
    data: pd.DataFrame,
    feature_types_path: PathLike | None = None,
    exclude_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Return the columns of `data` holding categorical features.

    Args:
        data: the data table
        feature_types_path: the feature types YAML. If given, only features declared
            categorical are kept. If None, every non-metadata column is used.
        exclude_columns: further columns to drop, e.g. confounders

    Returns:
        The data restricted to the categorical feature columns.

    Raises:
        ValueError: if an excluded column does not exist, or no features are left.
    """
    unknown = set(exclude_columns) - set(data.columns)
    if unknown:
        raise ValueError(f"Columns to exclude not found in the data: {sorted(unknown)}")

    skip = set(METADATA_COLUMNS) | set(exclude_columns)
    columns = [c for c in data.columns if c not in skip]

    if feature_types_path is not None:
        feature_types = YAML(typ="safe").load(Path(feature_types_path))
        columns = [
            c for c in columns
            if feature_types.get(c, {}).get("type") == FeatureType.categorical.value
        ]

    if not columns:
        raise ValueError("No categorical features found in the data.")
    return data[columns]


def find_correlated_pairs(
    features: pd.DataFrame, p_threshold: float, bonferroni: bool = True
) -> tuple[list[CorrelatedPair], float]:
    """Test all pairs of features for independence and keep the significant ones.

    Pairs that do not overlap in at least two states of each feature are skipped;
    this happens for conditional features that are only defined for some objects.

    The chi-squared approximation is unreliable when the expected counts are very
    small, so pairs of rare states can show up as significant.

    Args:
        features: the categorical features, one column per feature
        p_threshold: the significance level
        bonferroni: divide the threshold by the number of tests, correcting for
            multiple testing

    Returns:
        The significant pairs, sorted by p-value, and the threshold actually applied.
    """
    pairs = list(combinations(features.columns, 2))
    threshold = p_threshold / len(pairs) if bonferroni and pairs else p_threshold

    correlated = []
    for f1, f2 in pairs:
        column_1: pd.Series = features[f1]
        column_2: pd.Series = features[f2]
        observed = pd.crosstab(column_1, column_2)
        if min(observed.shape) <= 1:
            continue

        result = chi2_contingency(observed)
        if result.pvalue < threshold:
            expected = pd.DataFrame(
                result.expected_freq, index=observed.index, columns=observed.columns
            )
            correlated.append(
                CorrelatedPair(float(result.pvalue), str(f1), str(f2), observed, expected)
            )

    correlated.sort(key=lambda pair: pair.p_value)
    return correlated, threshold


def plot_correlations(pairs: Sequence[CorrelatedPair], output_path: PathLike) -> None:
    """Write one row of heatmaps per correlated pair to a PDF.

    Args:
        pairs: the correlated pairs, in the order they should be plotted
        output_path: the output PDF

    Raises:
        ImportError: if matplotlib or seaborn are not installed.
        ValueError: if `pairs` is empty.
    """
    if not pairs:
        raise ValueError("Nothing to plot.")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import Normalize, PowerNorm
    except ImportError as e:
        raise ImportError(
            "find_correlated_features needs matplotlib and seaborn: "
            "pip install sbayes[tools]"
        ) from e

    n = len(pairs)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n), squeeze=False)

    for i, pair in enumerate(pairs):
        deviation = pair.observed - pair.expected
        limit = max(float(deviation.abs().to_numpy().max()), 1.0)

        sns.heatmap(pair.expected, annot=pair.expected, fmt=".1f", cmap="viridis",
                    annot_kws={"fontsize": 12}, ax=axes[i, 0], norm=PowerNorm(0.5))
        sns.heatmap(pair.observed, annot=pair.observed, fmt="d", cmap="viridis",
                    annot_kws={"fontsize": 12}, ax=axes[i, 1], norm=PowerNorm(0.5))
        sns.heatmap(deviation, annot=deviation, fmt=".1f", cmap="RdBu",
                    annot_kws={"fontsize": 12}, ax=axes[i, 2],
                    norm=Normalize(vmin=-limit, vmax=limit))

        axes[i, 0].set_title("expected counts if\nfeatures were independent",
                             fontweight="bold", pad=12)
        axes[i, 1].set_title(f"observed counts\n[{pair.feature_1}] × [{pair.feature_2}]",
                             fontweight="bold", pad=12)
        axes[i, 2].set_title("observed - expected counts\n"
                             f"(independence rejected with p={pair.p_value:.2g})",
                             fontweight="bold", pad=12)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.2 / n, bottom=0.01 + 0.15 / n, hspace=0.8, wspace=0.6)
    fig.savefig(output_path)
    plt.close(fig)


# --- Entry point ---------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find pairs of categorical features that are statistically dependent."
    )
    parser.add_argument("--input", type=Path,
                        help="The data CSV file. Asked for in a dialog if omitted.")
    parser.add_argument("--output", type=Path,
                        help="The output PDF file. Asked for in a dialog if omitted.")
    parser.add_argument("--featureTypes", type=Path,
                        help="Feature types YAML file. If given, only features declared "
                             "categorical are tested.")
    parser.add_argument("--excludeColumns", nargs="*", default=[],
                        help="Further columns to exclude, e.g. confounders.")
    parser.add_argument("-p", "--pThreshold", type=float, default=0.0001,
                        help="The significance level (default: 0.0001).")
    parser.add_argument("--noCorrection", action="store_true",
                        help="Do not apply the Bonferroni correction for the number of "
                             "tested pairs.")
    parser.add_argument("--maxPairs", type=int, default=50,
                        help="Maximum number of pairs to plot, most significant first "
                             "(default: 50).")
    args = parser.parse_args(argv)

    if not 0 < args.pThreshold <= 1:
        parser.error("--pThreshold must be between 0 and 1.")
    if args.maxPairs < 1:
        parser.error("--maxPairs must be at least 1.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_path: Path | None = args.input
    output_path: Path | None = args.output

    root = LazyRoot()
    try:
        if data_path is None:
            data_path = ask_open_file(
                root, "Select a data file in CSV format.", [("CSV files", "*.csv")]
            )
            if data_path is None:
                sys.exit("No data file selected.")

        if output_path is None:
            output_path = ask_save_file(
                root, "Select an output file in PDF format.", [("PDF files", "*.pdf")],
                default_name="correlations.pdf", directory=data_path.parent,
            )
            if output_path is None:
                sys.exit("No output file selected.")
    finally:
        root.destroy()

    features = select_categorical_features(
        read_data_csv(data_path), args.featureTypes, args.excludeColumns
    )
    print(f"Testing {features.shape[1]} features for pairwise dependence...")

    pairs, threshold = find_correlated_pairs(
        features, args.pThreshold, bonferroni=not args.noCorrection
    )
    print(f"Found {len(pairs)} correlated pair(s) at p < {threshold:.2g}.")
    if not pairs:
        return

    for pair in pairs[: args.maxPairs]:
        print(f"  [{pair.feature_1}] × [{pair.feature_2}]: p = {pair.p_value:.2g}")
    if len(pairs) > args.maxPairs:
        print(f"Plotting the {args.maxPairs} most significant pairs.")

    plot_correlations(pairs[: args.maxPairs], output_path)
    print(f"Wrote the plots to {output_path}")


if __name__ == "__main__":
    main()