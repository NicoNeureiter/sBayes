import os
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, Normalize
import pandas as pd
from scipy.stats import stats, chi2_contingency
import seaborn as sns

from sbayes.load_data import Features
from sbayes.util import normalize_str, read_data_csv


def dict_to_df(d):
    # Count maximum number of values (i.e. number of rows in df)
    n_rows = max(len(values) for values in d.values())

    # Make a dictionary of lists, padded to n_rows
    d_padded = {}
    for k, values in d.items():
        d_padded[k] = list(values) + [None]*(n_rows - len(values))

    return pd.DataFrame(d_padded)


def collect_feature_states(data):
    METADATA_COLUMNS = ['id', 'name', 'family', 'x', 'y']
    for column in METADATA_COLUMNS:
        if column not in data.columns:
            raise ValueError(f'Required column \'{column}\' missing in data file.')
    features = data.drop(METADATA_COLUMNS, axis=1).applymap(normalize_str)
    return dict_to_df({f: set(features[f].dropna().unique()) for f in features.columns})


def main(args):
    # CLI
    parser = argparse.ArgumentParser(description='Tool to find features with significant correlation in a data set.')
    parser.add_argument('--input', required=True, type=Path, help='The input CSV file')
    parser.add_argument('--output', required=True, type=Path, help='The output PDF file')
    parser.add_argument('-p', '--pThreshold', type=float, default=0.0001,
                        help='The significance level (p-value threshold) for plotting correlation between two features.')

    args = parser.parse_args(args)
    data_path = args.input
    out_path = args.output
    p_thresh = args.pThreshold

    # GUI
    if data_path is None:
        # Import tkinter only when needed (CLI works without it)
        import tkinter as tk
        from tkinter import filedialog

        # Ask the user for the data file
        tk.Tk().withdraw()
        data_path = filedialog.askopenfilename(title='Select a data file in CSV format.',
                                               filetypes=(('csv files', '*.csv'), ('all files', '*.*')))

    # Read all input files and collect all states for each feature
    data = read_data_csv(data_path)

    METADATA_COLUMNS = ['id', 'name', 'family', 'x', 'y']
    for column in METADATA_COLUMNS:
        if column not in data.columns:
            raise ValueError(f'Required column \'{column}\' missing in data file.')
    features = data.drop(METADATA_COLUMNS, axis=1).applymap(normalize_str)
    print(features.shape)

    corr_features = []
    for f1, f2 in list(combinations(features.columns, 2)):
        crosstab = pd.crosstab(features[f1], features[f2])
        if min(crosstab.shape) <= 1:
            # One of the features has only one state in the languages where the two overlap
            # (usually conditional features)
            continue

        chi_squared = chi2_contingency(crosstab)

        if chi_squared.pvalue < p_thresh:
            print(f'Correlation between [{f1}] and [{f2}].')
            print(f'Chi-squared test statistic = {chi_squared.statistic}')
            print(f'Chi-squared test p-value = {chi_squared.pvalue}')
            print()
            corr_features.append((chi_squared.pvalue, f1, f2, chi_squared, crosstab))


    n = len(corr_features)
    print(f'Plotting {n} correlated features...')
    fig: plt.Figure
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))

    corr_features.sort()
    for i, (p_value, f1, f2, chi_squared, crosstab) in enumerate(corr_features):

        expected_freq = pd.DataFrame.from_records(chi_squared.expected_freq, columns=crosstab.columns)
        expected_freq.index = crosstab.index

        deviation = crosstab - expected_freq

        sns.heatmap(expected_freq, annot=expected_freq, fmt='.1f', annot_kws={'fontsize': 12}, ax=axes[i, 0], cmap='viridis', norm=PowerNorm(0.5))
        sns.heatmap(crosstab, annot=crosstab, fmt='d', annot_kws={'fontsize': 12}, ax=axes[i, 1], cmap='viridis', norm=PowerNorm(0.5))
        sns.heatmap(deviation, annot=deviation, fmt='.1f', annot_kws={'fontsize': 12}, ax=axes[i, 2], cmap='RdBu', norm=Normalize(vmin=-15, vmax=15))

        axes[i, 0].set_title('expected counts if \n features were independent', fontweight='bold', pad=12)
        axes[i, 1].set_title(f'observed counts', fontweight='bold', pad=12)
        axes[i, 2].set_title(f'observed - expected counts \n (independence rejected with p={chi_squared.pvalue:.2g})', fontweight='bold', pad=12)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.2 / n, bottom=0.01 + 0.15 / n, hspace=0.8, wspace=0.6)

    fig.savefig(out_path)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
