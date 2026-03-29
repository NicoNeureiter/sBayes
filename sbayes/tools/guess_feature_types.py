import os
import argparse
import tkinter as tk

import pandas as pd
try:
    import ruamel.yaml as yaml
except ImportError:
    import ruamel_yaml as yaml

from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

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


def select_save_file(default_dir='.', default_name='feature_types.yaml'):
    path = filedialog.asksaveasfile(
        title='Select an output file in YAML format.',
        initialdir=default_dir,
        initialfile=default_name,
        filetypes=(('YAML files', '*.yaml'), ('all files', '*.*'))
    )
    return path


def ask_more_files():
    MsgBox = tk.messagebox.askquestion('Additional data files', 'Would you like to add more data files?')
    return MsgBox == 'yes'


def ask_confounders(col_names):
    selected_confounders = []

    def submit():
        for k, v in confounders_vars.items():
            if v.get() == 1:
                selected_confounders.append(k)
        root.quit()

    root = tk.Tk()
    root.title("Select Confounders")

    screen_height = root.winfo_screenheight()
    win_height = min(800, screen_height - 100)
    root.geometry(f"400x{win_height}")

    # Scrollable frame for checkboxes
    container = tk.Frame(root)
    container.pack(fill="both", expand=True, padx=10, pady=10)

    canvas = tk.Canvas(container, height=win_height - 80)  # explicit height
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")  # pack scrollbar BEFORE canvas
    canvas.pack(side="left", fill="both", expand=True)

    # Checkboxes inside scrollable frame
    confounders_vars = {}
    for c in col_names:
        confounders_vars[c] = tk.IntVar(root)
        tk.Checkbutton(scroll_frame, text=c, variable=confounders_vars[c],
                       onvalue=1, offvalue=0, anchor="w").pack(fill="x")

    # --- Submit button always at the bottom ---
    submit_button = tk.Button(root, text="Submit", command=submit)
    submit_button.pack(pady=10)

    root.mainloop()
    return selected_confounders


def collect_feature_states(features_path):

    features = read_data_csv(features_path)

    metadata_columns = ['id', 'name', 'x', 'y']
    for column in metadata_columns:
        if column not in features.columns:
            raise ValueError(f'Required column \'{column}\' missing in file {features_path}.')
    features = features.drop(metadata_columns, axis=1)

    # Ask users for the names of the confounders
    confounder_columns = ask_confounders(features.columns)
    features = features.drop(confounder_columns, axis=1)
    features = features.map(normalize_str)
    return {f: set(features[f].dropna().unique()) for f in features.columns}


def dict_to_df(d):
    # Count maximum number of values (i.e. number of rows in df)
    n_rows = max(len(values) for values in d.values())

    # Make a dictionary of lists, padded to n_rows
    d_padded = {}
    for k, values in d.items():
        d_padded[k] = list(values) + [None]*(n_rows - len(values))

    return pd.DataFrame(d_padded)


def is_number(s):
    """Helper function to check if a string is a number
    :param s: string to check
    :return(bool) is s a number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_integer(s):
    """Helper function to check if a string is an integer
    :param s: string to check
    :return(bool) is s an integer?
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_binary_integer(s):
    """Helper function to check if a string is a binary integer
    :param s: string to check
    :return(bool) is s a binary integer?
    """
    try:
        if int(s) == 0 or int(s) == 1:
            return True
        else:
            return False
    except ValueError:
        return False


def is_percentage(s):
    """Helper function to check if a string is a percentage
    :param s: string to check
    :return(bool) is s an integer
    """
    try:
        if 0 < float(s) < 1:
            return True
        else:
            return False
    except ValueError:
        return False


def guess_feature_type(f):
    """ Guesses the type of the features in f
    categorical: observations belong to two or more categories
    gaussian: observations are continuous measurements
    poisson: observations are count variables
    logit-normal: observations are percentages
    :param f: feature vector
    :return type guess for each feature vector
    """
    if not all(is_number(o) for o in f):
        type_guess = "categorical"
    else:
        if all(is_integer(o) for o in f):
            if all(is_binary_integer(o) for o in f):
                type_guess = "categorical"
            else:
                type_guess = "poisson"
        else:
            if all(is_percentage(o) for o in f):
                type_guess = "logit-normal"
            else:
                type_guess = "gaussian"
    return type_guess


def main(args):
    # CLI
    parser = argparse.ArgumentParser(description="Tool to extract feature types from sBayes data files.")
    parser.add_argument("--input", nargs="*", type=Path, help="The input features.csv file(s)")
    parser.add_argument("--output", nargs="?", type=Path, help="The output feature_types.yaml file")
    parser.add_argument("--excludeColumns", nargs="*", type=str,
                        help="Column names to exclude (e.g., confounders).")
    args = parser.parse_args(args)
    csv_paths = args.input
    exclude_columns = args.excludeColumns

    # GUI
    gui_required = (csv_paths is None
                    or exclude_columns is None)

    # GUI
    if gui_required:
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
    for f in feature_states.copy():
        # if np.nan in feature_states[f]:
        #     feature_states[f].remove(np.nan)

        if ORDER_STATES:
            feature_states[f] = sorted(feature_states[f])
        # Guess the type of each feature
        type_guess = guess_feature_type(feature_states[f])

        # Return the type and the applicable states / range of states
        if type_guess == "categorical":
            feature_states[f] = dict(type=type_guess, states=feature_states[f])
        elif type_guess == "poisson":
            int_features = [int(s) for s in feature_states[f]]
            feature_states[f] = dict(type=type_guess, states=dict(min=min(int_features), max=max(int_features)))
        else:
            float_features = [float(s) for s in feature_states[f]]
            feature_states[f] = dict(type=type_guess, states=dict(min=min(float_features), max=max(float_features)))

    # Ask user for the output file and save the feature_states there
    if args.output:
        output_path = args.output
    else:
        output_path = select_save_file(default_dir=current_directory)

    yml = yaml.YAML()
    yml.indent(offset=2)
    yml.default_flow_style = False
    yml.dump(feature_states, output_path)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
