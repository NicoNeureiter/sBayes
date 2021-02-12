import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMC
from sbayes.simulation import Simulation


def run_experiment(experiment, data, run):
    mcmc = MCMC(data=data, experiment=experiment)
    mcmc.log_setup()

    # Warm-up
    mcmc.warm_up()

    # Sample from posterior
    mcmc.sample()

    # Save samples to file
    mcmc.log_statistics()
    mcmc.save_samples(run=run)

    # Use the last sample as the new initial sample
    return mcmc.samples['last_sample']


def main(config=None, experiment_name=None):
    if config is None:
        parser = argparse.ArgumentParser(
            description="An MCMC algorithm to identify contact zones")
        parser.add_argument("config", nargs="?", type=Path,
                            help="The JSON configuration file")
        args = parser.parse_args()
        config = args.config

    # 0. Ask for config file via files-dialog, if not provided as argument.
    if config is None:
        tk.Tk().withdraw()
        config = filedialog.askopenfilename(
            title='Select a config file in JSON format.',
            initialdir='..',
            filetypes=(('json files', '*.json'),('all files', '*.*'))
        )

    # Initialize the experiment
    experiment = Experiment(experiment_name=experiment_name,
                            config_file=config, log=True)

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

    # Rerun experiment to check for consistency
    for run in range(experiment.config['mcmc']['N_RUNS']):
        n_areas = experiment.config['model']['N_AREAS']
        if isinstance(n_areas, list):
            # Run the experiment multiple times to determine the number of areas.
            for N in n_areas:
                # Update config information according to the current setup
                experiment.config['model']['N_AREAS'] = N

                # Run the experiment with the specified number of areas
                run_experiment(experiment, data, run)
        else:
            # Run the experiment once, with the specified settings
            assert isinstance(n_areas, int)
            run_experiment(experiment, data, run)


if __name__ == '__main__':
    main()
