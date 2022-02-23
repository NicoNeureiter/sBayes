import multiprocessing
from itertools import product

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from sbayes.experiment_setup import Experiment, update_recursive
from sbayes.load_data import Data
from sbayes.simulation import Simulation
from sbayes.mcmc_setup import MCMC


def run_experiment(
        experiment: Experiment,
        data: Data,
        i_run: int,
        custom_settings: dict = None,
):
    if custom_settings is not None:
        update_recursive(experiment.config, custom_settings)

    mcmc = MCMC(data=data, experiment=experiment)
    mcmc.log_setup()

    # Warm-up
    mcmc.warm_up()

    # Sample from posterior
    mcmc.sample(run=i_run)

    # Save samples to file
    mcmc.log_statistics()
    mcmc.save_samples(run=i_run)

    # Use the last sample as the new initial sample
    return mcmc.samples['last_sample']


def main(
        config: Path,
        experiment_name: str = None,
        custom_settings: dict = None,
        processes: int = 1,
):
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
    i_run_range = list(range(experiment.config['mcmc']['runs']))
    n_areas_range = experiment.config['model']['areas']
    if type(n_areas_range) not in [tuple, list, set]:
        assert isinstance(n_areas_range, int)
        n_areas_range = [n_areas_range]

    def runner(i_run, n_areas):
        run_experiment(
            experiment=experiment,
            data=data,
            i_run=i_run,
            custom_settings={'model': {'areas': n_areas}}
        )

    for i in i_run_range:
        for k in n_areas_range:
            runner(i, k)

    # if processes <= 1:
    #     map(runner, product(i_run_range, n_areas_range))
    # else:
    #     pool = multiprocessing.Pool(processes=processes)
    #     pool.map(runner, product(i_run_range, n_areas_range))


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


def cli():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='An MCMC algorithm to identify contact zones')
    parser.add_argument('config', type=Path,
                        help='The JSON configuration file')
    parser.add_argument('name', nargs='?', type=str,
                        help='The experiment name used for logging and as the name of the results directory.')
    parser.add_argument('threads', nargs='?', type=int, default=1,
                        help='The number of parallel processes.')

    args = parser.parse_args()

    config = args.config

    # Ask for config file via files-dialog, if not provided as argument.
    if config is None:
        tk.Tk().withdraw()
        config = filedialog.askopenfilename(
            title='Select a config file in JSON format.',
            initialdir='..',
            filetypes=(('json files', '*.json'), ('all files', '*.*'))
        )

    main(
        config=config,
        experiment_name=args.name,
        processes=args.threads
    )


if __name__ == '__main__':
    cli()
