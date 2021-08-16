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


def main(config: Path = None,
         experiment_name: str = None,
         custom_settings: dict = None):
    if config is None:
        parser = argparse.ArgumentParser(
            description="An MCMC algorithm to identify contact zones")
        parser.add_argument("config", nargs="?", type=Path,
                            help="The JSON configuration file")
        parser.add_argument("name", nargs="?", type=str,
                            help="The experiment name used for logging and as the name of the results directory.")
        args = parser.parse_args()
        config = args.config
        experiment_name = args.name

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
                            config_file=config,
                            custom_settings=custom_settings,
                            log=True)

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
        data.load_geo_cost_matrix()

        # Log
        data.log_loading()

    # Rerun experiment to check for consistency
    for run in range(experiment.config['mcmc']['n_runs']):
        n_areas = experiment.config['model']['n_areas']
        iterate_or_run(
            x=n_areas,
            config_setter=lambda x: experiment.config['model'].__setitem__('n_areas', x),
            function=lambda x: run_experiment(experiment, data, run)
        )


def iterate_over_parameter(values, config_setter, function, print_message=None):
    """Iterate over each value in ´values´, apply ´config_setter´ (to update the config
    dictionary) and run ´function´."""
    for i, value in enumerate(values):
        if print_message is not None:
            print(print_message.format(value=value, i=i))
        config_setter(value)
        function(value)


def iterate_or_run(x, config_setter, function, print_message=None):
    """If ´x´ is list, iterate over all values in ´x´ and run ´function´ for each value.
    Otherwise directly apply ´function´ to ´x´."""
    if type(x) in [tuple, list, set]:
        iterate_over_parameter(x, config_setter, function, print_message)
    else:
        function(x)


if __name__ == '__main__':
    main()
