import multiprocessing
from itertools import product

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from sbayes.experiment_setup import Experiment
from sbayes.util import PathLike
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMCSetup


def run_experiment(
    config: PathLike,
    experiment_name: str,
    custom_settings: dict = None,
    i_run: int = 0,
):
    # Initialize the experiment
    experiment = Experiment(
        config_file=config,
        experiment_name=experiment_name,
        custom_settings=custom_settings,
        log=True,
    )

    # Experiment based on a specified (in config) data-set
    data = Data.from_experiment(experiment)

    # Set up MCMC
    mcmc = MCMCSetup(data=data, experiment=experiment)
    mcmc.log_setup()

    # Warm-up and run MCMC sampling
    mcmc.warm_up()
    mcmc.sample(run=i_run)

    # Use the last sample as the new initial sample
    return mcmc.samples.last_sample


def runner(args):
    """A wrapper for `run_experiment` to make it callable using the pool.map interface."""
    i_run, n_clusters, config, experiment_name = args
    # run_experiment(config, f"{experiment_name}/K{n_clusters}_{i_run}",
    run_experiment(
        config=config,
        experiment_name=experiment_name,
        custom_settings={"model": {"clusters": n_clusters}, "mcmc": {"runs": 1}},
        i_run=i_run,
    )


def main(
    config: PathLike,
    experiment_name: str = None,
    custom_settings: dict = None,
    processes: int = 1,
):
    # Initialize the experiment
    experiment = Experiment(
        config_file=config,
        experiment_name=experiment_name,
        custom_settings=custom_settings,
        log=False,
    )

    # Extract the range of run repetitions and number of clusters
    i_run_range = list(range(experiment.config.mcmc.runs))
    n_clusters_range = experiment.config.model.clusters
    if type(n_clusters_range) not in [tuple, list, set]:
        assert isinstance(n_clusters_range, int)
        n_clusters_range = [n_clusters_range]

    # Define configurations for each distinct sBayes run that needs to be executed
    run_configurations = product(
        i_run_range, n_clusters_range, [config], [experiment.experiment_name]
    )

    # Run all configurations sequentially or in parallel
    if processes <= 1:
        for cfg in run_configurations:
            runner(cfg)
    else:
        pool = multiprocessing.Pool(processes=processes)
        pool.map(runner, run_configurations)


def cli():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="An MCMC algorithm to identify contact zones"
    )

    # The only required (positional) argument is the path to the config file:
    parser.add_argument(
        "config",
        type=Path,
        help="The JSON configuration file"
    )

    # Optional named arguments:
    parser.add_argument(
        "-n", "--name",
        nargs="?",
        type=str,
        help="The experiment name used for logging and as the name of the results directory.",
    )
    parser.add_argument(
        "-t", "--threads",
        nargs="?",
        type=int,
        default=1,
        help="The number of parallel processes.",
    )

    args = parser.parse_args()
    config = args.config

    # Ask for config file via files-dialog, if not provided as argument.
    if config is None:
        tk.Tk().withdraw()
        config = filedialog.askopenfilename(
            title="Select a config file in JSON format.",
            initialdir="..",
            filetypes=(("json files", "*.json"), ("all files", "*.*")),
        )

    main(config=config, experiment_name=args.name, processes=args.threads)


if __name__ == "__main__":
    cli()
