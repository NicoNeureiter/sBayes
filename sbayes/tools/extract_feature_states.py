import os

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog, messagebox

from sbayes.util import normalize_str


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
    features = pd.read_csv(features_path, sep=',', dtype=str)
    features = features.drop(['id', 'name', 'family', 'x', 'y'], axis=1)
    features = features.applymap(normalize_str)
    return {f: set(features[f].unique()) for f in features.columns}


def dict_to_df(d):
    # Count maximum number of values (i.e. number of rows in df)
    n_rows = max(len(values) for values in d.values())

    # Make a dictionary of lists, padded to n_rows
    d_padded = {}
    for k, values in d.items():
        d_padded[k] = list(values) + [None]*(n_rows - len(values))

    return pd.DataFrame(d_padded)


if __name__ == '__main__':
    tk.Tk().withdraw()

    # Ask the user for input files
    csv_paths = []
    current_directory = '.'
    more_files = True
    while more_files:
        csv_paths.append(select_open_file(default_dir=current_directory))
        current_directory = os.path.dirname(csv_paths[-1])
        more_files = ask_more_files()

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
                raise ValueError(out)

            for f in feature_states.keys():
                feature_states[f].update(new_feature_states[f])

    # Remove NAs and order states alphabetically (if ´ORDER_STATES´ is set)
    for f in feature_states:
        if np.nan in feature_states[f]:
            feature_states[f].remove(np.nan)

        if ORDER_STATES:
            feature_states[f] = sorted(feature_states[f])

    # Cast into a pandas dataframe
    feature_states_df = dict_to_df(feature_states)

    # Ask user for the output file and save the feature_states there
    output_path = select_save_file(default_dir=current_directory)
    feature_states_df.to_csv(output_path, index=False)
