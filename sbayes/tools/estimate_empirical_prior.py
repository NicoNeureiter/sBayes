import numpy as np
from pathlib import Path
import json
import os
import argparse

from numpyro.distributions.util import categorical
from sbayes.load_data import read_features_from_csv, Confounder, Features, CategoricalFeatures
from sbayes.util import scale_counts


def zip_internal_external(names):
    return zip(names['internal'], names['external'])


def main(args):
    # CLI
    parser = argparse.ArgumentParser(description="Tool to estimate an empirical prior for an sBayes analysis.")
    parser.add_argument("--data", type=Path, help="Empirical data as a CSV file.")
    parser.add_argument("--featureTypes", type=Path, help="Feature types YAML file.")
    parser.add_argument("--output", type=Path, help="Directory where output files will be written.")
    parser.add_argument("--confounder", type=str, default="universal", help="Confounder to include when estimating the prior.")
    parser.add_argument("--CategoricalConcentration", type=float, default=1.0, help="Concentration of the hyper-prior (1.0 corresponds to a uniform prior).")
    parser.add_argument( "--CategoricalMaxScale", type=float, default=None, help="Maximum allowed scale (broadness) of the categorical prior.")
    parser.add_argument("--PoissonMaxScale", type=float, default=None, help="Maximum allowed scale (broadness) of the Poisson prior.")
    parser.add_argument("--GaussianMaxScale", type=float,  default=None, help="Maximum allowed scale (broadness) of the Gaussian prior.")

    args = parser.parse_args(args)
    prior_data_file = args.data
    feature_types_file = args.featureTypes
    output_dir = args.output
    confounder = args.confounder
    categorical_hyper_prior_concentration = args.CategoricalConcentration
    categorical_max_scale = args.CategoricalMaxScale
    poisson_max_scale = args.PoissonMaxScale
    gaussian_max_scale = args.GaussianMaxScale

    if confounder is None:
        confounder = "universal"

    # GUI
    gui_required = (prior_data_file is None
                    or feature_types_file is None
                    or output_dir is None)
    if gui_required:
        import tkinter as tk
        from tkinter import filedialog

        tk.Tk().withdraw()
        current_directory = '.'

        if prior_data_file is None:
            # Ask the user for datafile
            prior_data_file = filedialog.askopenfilename(
                title='Select the empirical data as a CSV file.',
                initialdir=current_directory,
                filetypes=(('csv files', '*.csv'), ('all files', '*.*'))

            )
            current_directory = os.path.dirname(prior_data_file)

        if feature_types_file is None:
            # Ask the user for feature types file
            feature_types_file = filedialog.askopenfilename(
                title='Select the features types files in YAML format.',
                initialdir=current_directory,
                filetypes=(('yaml files', '*.yaml'), ('all files', '*.*'))
            )
            current_directory = os.path.dirname(feature_types_file)

        if output_dir is None:
            output_dir = filedialog.askdirectory(
                title='Select a directory to save output files.',
                initialdir=current_directory
            )

    prior_data_file = Path(prior_data_file)
    feature_types_file = Path(feature_types_file)
    output_dir = Path(output_dir)


    objects, features, confounders = read_features_from_csv(
        data_path=prior_data_file,
        feature_types_path=feature_types_file,
        confounder_names=[confounder],
    )
    features: Features = features

    if confounder == "universal":
        groups: Confounder | None = None
        group_names = ["universal"]

    else:
        # Extract the Confounder object and its group names for iteration
        groups: Confounder = confounders[confounder]
        group_names = groups.group_names

    for i_group, group_name in enumerate(group_names):

        counts_dict = {}

        for partition in features.partitions:
            if not isinstance(partition, CategoricalFeatures):
                #todo: estimate prior for non-categorical features
                continue  # Skip non-categorical features

            if group_name == "universal":
                # No grouping: use the full features object as-is
                features_group = partition.to_binary()

            else:
                # Subset features to only the members belonging to this group
                group_members = groups.group_assignment[i_group]
                features_group = partition.to_binary()[group_members, :, :]
                # Resulting shape: (n_group_members, n_features, n_states)

            # Count the occurrences of each state in each feature in the partition
            counts = np.sum(features_group, axis=0)  # shape: (n_features, n_states)

            # Apply the scale_counts if provided
            if categorical_max_scale is not None:
                counts = scale_counts(counts, categorical_max_scale)

            for i_f, feature in enumerate(partition.names):
                counts_dict[feature] = {}
                for i_s, state in enumerate(partition.state_names[i_f]):
                    counts_dict[feature][state] = categorical_hyper_prior_concentration + counts[i_f, i_s]

        with open(Path(output_dir, f"{group_name}_prior.json"), 'w') as prior_file:
            json.dump(counts_dict, prior_file, indent=4)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
