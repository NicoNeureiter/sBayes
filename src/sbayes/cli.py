import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMC
from sbayes.simulation import Simulation

NUMBER_AREAS_GRID = range(1, 8)


def run_experiment(experiment, data, run, initial_sample=None):
    mcmc = MCMC(data=data, experiment=experiment)
    mcmc.log_setup()

    # Sample
    mcmc.sample(initial_sample=initial_sample)

    # Save samples to file
    mcmc.log_statistics()
    mcmc.save_samples(run=run)

    # Use the last sample as the new initial sample
    return mcmc.samples['last_sample']


def main():
    parser = argparse.ArgumentParser(
        description="An MCMC algorithm to identify contact zones")
    parser.add_argument("config", nargs="?", type=Path,
                        help="The JSON configuration file")
    args = parser.parse_args()

    # 0. Ask for config file via files-dialog, if not provided as argument.
    config = args.config
    if config is None:
        tk.Tk().withdraw()
        config = filedialog.askopenfilename(
            title='Select a config file in JSON format.',
            initialdir='..',
            filetypes=(('json files', '*.json'),('all files', '*.*'))
        )

    # Initialize the experiment
    experiment = Experiment(config_file=config, logging=True)

    if experiment.is_simulation():
        # The data is defined by a ´Simulation´ object
        data = Simulation(experiment=experiment)
        data.run_simulation()
        data.log_simulation()

    else:
        # Experiment based on a specified (in config) data-set
        data = Data(experiment=experiment)
        data.load_features()

        # Counts for priors
        data.load_universal_counts()
        data.load_inheritance_counts()

        # Log
        data.log_loading()

    initial_sample = None

    # Rerun experiment to check for consistency
    for run in range(experiment.config['mcmc']['N_RUNS']):
        if isinstance(experiment.config['mcmc']['N_AREAS'], str):
            assert experiment.config['mcmc']['N_AREAS'].lower() == 'tbd'

            # Run the experiment multiple times to determine the number of areas.
            for N in NUMBER_AREAS_GRID:
                # Update config information according to the current setup
                experiment.config['mcmc']['N_AREAS'] = N

                # Run the experiment with the specified number of areas
                initial_sample = run_experiment(experiment, data, run,
                                                initial_sample=initial_sample)

        else:
            # Run the experiment once, with the specified settings
            run_experiment(experiment, data, run)

        initial_sample = None
