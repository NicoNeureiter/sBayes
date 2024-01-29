import numpy as np
from pathlib import Path
import json
import os
import argparse

from sbayes.load_data import read_features_from_csv, Confounder
from sbayes.util import scale_counts


def zip_internal_external(names):
    return zip(names['internal'], names['external'])


def main(args):
    # ===== CLI =====
    parser = argparse.ArgumentParser(description="Tool to extract parameters for an empirical inheritance prior from sBayes data files.")
    # parser.add_argument("--confounder", nargs="?", type=Path,
    #                     help="The name of the confounder for which to extract the data.")
    parser.add_argument("--data", nargs="?", type=Path, help="The input CSV file")
    parser.add_argument("--featureStates", nargs="?", type=Path, help="The feature states CSV file")
    parser.add_argument("--output", nargs="?", type=Path, help="The output directory")
    parser.add_argument("--add", nargs="?", default=1.0, type=float, help="Concentration of the hyper-prior (1.0 is Uniform)")
    parser.add_argument("--scaleCounts", nargs="?", default=None, type=float, help="An upper bound on the concentration of the prior (default is infinity/no upper bound).")

    args = parser.parse_args(args)
    prior_data_file = args.data
    feature_states_file = args.featureStates
    output_directory = args.output
    hyper_prior_concentration = args.add
    max_counts = args.scaleCounts

    # ===== GUI =====
    gui_required = (prior_data_file is None
                    or feature_states_file is None
                    or output_directory is None)
    if gui_required:
        import tkinter as tk
        from tkinter import filedialog

        tk.Tk().withdraw()
        current_directory = '.'

        if prior_data_file is None:
            # Ask the user for data file
            prior_data_file = filedialog.askopenfilename(
                title='Select the data file in CSV format.',
                initialdir=current_directory,
                filetypes=(('csv files', '*.csv'), ('all files', '*.*'))
            )
            current_directory = os.path.dirname(prior_data_file)

        if feature_states_file is None:
            # Ask the user for feature states file
            feature_states_file = filedialog.askopenfilename(
                title='Select the feature_states file in CSV format.',
                initialdir=current_directory,
                filetypes=(('csv files', '*.csv'), ('all files', '*.*'))
            )
            current_directory = os.path.dirname(feature_states_file)

        if output_directory is None:
            # Ask the user for output directory
            output_directory = filedialog.askdirectory(
                title='Select the output directory.',
                initialdir=current_directory,
            )

    prior_data_file = Path(prior_data_file)
    feature_states_file = Path(feature_states_file)
    output_directory = Path(output_directory)

    objects, features, confounders = read_features_from_csv(
        data_path=prior_data_file,
        feature_states_path=feature_states_file,
        confounder_names=['family'],
    )
    families: Confounder = confounders['family']

    for i_fam, family_name in enumerate(families.group_names):
        family_members = families.group_assignment[i_fam]
        features_fam = features.values[family_members, :, :]  # shape: (n_family_members, n_features, n_states)
        counts = np.sum(features_fam, axis=0)                 # shape: (n_features, n_states)

        # Apply the scale_counts if provided
        if max_counts is not None:
            counts = scale_counts(counts, max_counts)

        counts_dict = {}
        for i_f, feature in enumerate(features.names):
            counts_dict[feature] = {}
            for i_s, state in enumerate(features.state_names[i_f]):
                counts_dict[feature][state] = hyper_prior_concentration + counts[i_f, i_s]

        with open(output_directory / f'{family_name}.json', 'w') as prior_file:
            json.dump(counts_dict, prior_file, indent=4)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
