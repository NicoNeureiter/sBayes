import multiprocessing
import warnings
from copy import deepcopy
from itertools import product
import argparse
from pathlib import Path

from pydantic import PositiveInt

from sbayes.experiment_setup import Experiment
from sbayes.util import PathLike, update_recursive, activate_verbose_warnings
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMCSetup


def run_experiment(
    config: PathLike,
    experiment_name: str,
    custom_settings: dict = None,
    resume: bool = False,
    i_run: int = 0,
):
    # Initialize the experiment
    experiment = Experiment(
        config_file=config,
        experiment_name=experiment_name,
        custom_settings=custom_settings,
        log=True,
        i_run=i_run,
    )

    # Experiment based on a specified (in config) data-set
    data = Data.from_experiment(experiment)
    data.logger = None

    # Set up and run MCMC
    mcmc = MCMCSetup(data=data, experiment=experiment)
    mcmc.log_setup()

    if experiment.config.mcmc.mc3.activate:
        mcmc.sample_mc3(run=i_run, resume=resume)
    else:
        mcmc.sample(run=i_run, resume=resume)


def runner(args):
    """A wrapper for `run_experiment` to make it callable using the pool.map interface."""
    i_run, n_clusters, config, experiment_name, custom_settings, resume = args
    # run_experiment(config, f"{experiment_name}/K{n_clusters}_{i_run}",

    run_settings = deepcopy(custom_settings) if custom_settings else {}
    update_recursive(run_settings, {"model": {"clusters": n_clusters}, "mcmc": {"runs": 1}})

    run_experiment(
        config=config,
        experiment_name=experiment_name,
        custom_settings=run_settings,
        resume=resume,
        i_run=i_run,
    )


def main(
    config: PathLike,
    experiment_name: str = None,
    custom_settings: dict = None,
    processes: int = 1,
    resume: bool = False,
    n_clusters: int | list[int] = None,
    i_run: int = None,
):
    # Initialize the experiment
    experiment = Experiment(
        config_file=config,
        experiment_name=experiment_name,
        custom_settings=custom_settings,
        log=False,
    )

    # Define a range of run IDs. Either a fixed value via CLI or a ranges defined by `runs` in the config file.
    n_runs = experiment.config.mcmc.runs
    if i_run is None:
        i_run_range = list(range(n_runs))
    else:
        if experiment.config.mcmc.runs > 1:
            raise ValueError(f"Fixing the runID (runID={i_run}) in the CLI and setting the number of runs > 1 (runs={n_runs})in the config file is incompatible.")
        i_run_range = [i_run]

    # Use n_clusters from CLI args or from config.
    if n_clusters is None:
        n_clusters = experiment.config.model.clusters
    else:
        warnings.warn(f"The number of clusters was set as a command-line argument, so the config file "
                      f"entry `clusters={experiment.config.model.clusters}` will be ignored.")
    # If n_cluster is a scalar, wrap it in a single element list
    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]

    # Define configurations for each distinct sBayes run that needs to be executed
    run_configurations = list(product(
        i_run_range, n_clusters, [config], [experiment.experiment_name], [custom_settings], [resume]
    ))

    # Run all configurations sequentially or in parallel
    if processes <= 1:
        for cfg in run_configurations:
            runner(cfg)
    else:
        pool = multiprocessing.Pool(processes=processes)
        pool.map(runner, run_configurations)


def cli():
    """Command line interface."""

    # When in debug mode, activate verbose warning (printing stack trace)
    if __debug__:
        activate_verbose_warnings()

    # Initialize CLI argument parser
    parser = argparse.ArgumentParser(
        description="An MCMC algorithm to detect clusters in the presence of confounders."
    )

    # The only required (positional) argument is the path to the config file:
    parser.add_argument(
        "config",
        type=Path,
        help="The YAML (or JSON) configuration file"
    )

    # Optional named CLI arguments:
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
    parser.add_argument(
        "-r", "--resume",
        nargs="?",
        type=bool,
        default=False,
        help="Whether to resume a previous run (requires experiment name, runID and number of clusters to match).",
    )
    parser.add_argument(
        "-K", "--numClusters",
        nargs="*",
        type=PositiveInt,
        help="The number of clusters (overrides value in config file). Multiple values will result in multiple runs.",
    )
    parser.add_argument(
        "-i", "--runID",
        nargs="?",
        type=PositiveInt,
        help="The index of this sBayes run. Used for distinguishing multiple runs in the same experiment directory.",
    )

    args = parser.parse_args()
    config = args.config

    # Ask for config file via files-dialog, if not provided as argument.
    if config is None:
        # Only import tkinter if it is needed
        import tkinter as tk
        from tkinter import filedialog

        # Open file dialog to select the config file
        tk.Tk().withdraw()
        config = filedialog.askopenfilename(
            title="Select a config file in YAML or JSON format.",
            initialdir="..",
            filetypes=(("json files", ".json"), ("yaml files", ".yaml .yml"), ("all files", "*.*")),
        )

    main(config=config, experiment_name=args.name, processes=args.threads,
         resume=args.resume, n_clusters=args.numClusters, i_run=args.runID)


if __name__ == "__main__":
    cli()
