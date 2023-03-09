import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog, messagebox

from sbayes.util import normalize_str, read_data_csv

ORDER_STATES = True
'''bool: Whether to order the features states alphabetically'''


def select_open_file(default_dir='.'):
    path = filedialog.askopenfilename(
        title='Select a data file in CSV format.',
        initialdir=default_dir,
        filetypes=(('csv files', '*.csv'), ('all files', '*.*'))
    )
    return path


def select_save_file(default_dir='.', default_name='feature_states.csv'):
    path = filedialog.asksaveasfile(
        title='Select an output file in CSV format.',
        initialdir=default_dir,
        initialfile=default_name,
        filetypes=(('csv files', '*.csv'), ('all files', '*.*'))
    )
    return path


def ask_more_files():
    MsgBox = tk.messagebox.askquestion ('Additional data files','Would you like to add more data files?')
    return MsgBox == 'yes'


def collect_feature_states(features_path):
    features = read_data_csv(features_path)
    METADATA_COLUMNS = ['id', 'name', 'family', 'x', 'y']
    for column in METADATA_COLUMNS:
        if column not in features.columns:
            raise ValueError(f'Required column \'{column}\' missing in file {features_path}.')
    features = features.drop(METADATA_COLUMNS, axis=1)
    features = features.applymap(normalize_str)
    return {f: set(features[f].dropna().unique()) for f in features.columns}


def dict_to_df(d):
    # Count maximum number of values (i.e. number of rows in df)
    n_rows = max(len(values) for values in d.values())

    # Make a dictionary of lists, padded to n_rows
    d_padded = {}
    for k, values in d.items():
        d_padded[k] = list(values) + [None]*(n_rows - len(values))

    return pd.DataFrame(d_padded)


def main(args):
    # CLI
    parser = argparse.ArgumentParser(description="Tool to extract feature states from sBayes data files.")
    parser.add_argument("--input", nargs="*", type=Path, help="The input CSV files")
    parser.add_argument("--output", nargs="?", type=Path, help="The output CSV file")

    args = parser.parse_args(args)
    csv_paths = args.input

    # GUI
    if (csv_paths is None) or (len(csv_paths) == 0):
        tk.Tk().withdraw()

        # Ask the user for input files
        csv_paths = []
        current_directory = '.'
        more_files = True
        while more_files:
            new_path = select_open_file(default_dir=current_directory)
            if new_path == '':
                # Skip when user presses cancel
                pass
            else:
                csv_paths.append(new_path)
                current_directory = os.path.dirname(new_path)

            more_files = ask_more_files()

    else:
        # If input paths are provided through CLI, use the first path as the current directory
        current_directory = os.path.dirname(csv_paths[0])

    # Read all input files and collect all states for each feature
    feature_states = None
    for path in csv_paths:
        new_feature_states = collect_feature_states(path)

        if feature_states is None:
            feature_states = new_feature_states
        else:
            if set(feature_states.keys()) != set(new_feature_states.keys()):
                out = '\nFeatures do not match between the different input files:'
                out += '\n\tPreviously loaded features: \t %s' % sorted(feature_states.keys())
                out += '\n\tFeatures in %s: \t %s' % (path, sorted(new_feature_states.keys()))
                out += '\n\tPreviously loaded, but missing in %s: \t %s' % (path, sorted(set(feature_states.keys()) - set(new_feature_states.keys())))
                out += '\n\tPresent in %s, but missing in previous files : \t %s' % (path, sorted(set(new_feature_states.keys()) - set(feature_states.keys())))
                raise ValueError(out)

            for f in feature_states.keys():
                feature_states[f].update(new_feature_states[f])

    # Remove NAs and order states alphabetically (if ´ORDER_STATES´ is set)
    for f in feature_states:
        # if np.nan in feature_states[f]:
        #     feature_states[f].remove(np.nan)

        if ORDER_STATES:
            feature_states[f] = sorted(feature_states[f])

    # Cast into a pandas dataframe
    feature_states_df = dict_to_df(feature_states)

    # Ask user for the output file and save the feature_states there
    if args.output:
        output_path = args.output
    else:
        output_path = select_save_file(default_dir=current_directory)

    # Store the feature_states in a csv file
    feature_states_df.to_csv(output_path, index=False, lineterminator='\n')


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
