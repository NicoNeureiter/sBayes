from __future__ import annotations

import argparse
import numpyro
import os
import warnings

from copy import deepcopy
from pathlib import Path
from pydantic import PositiveInt

from sbayes.experiment_setup import Experiment
from sbayes.util import PathLike, update_recursive
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMCSetup
from typing import NamedTuple


class RunTask(NamedTuple):
    i_run: int
    n_clusters: int
    config: PathLike
    experiment_name: str
    custom_settings: dict | None
    resume: bool
    use_gpu: bool
    num_cpus: int


def set_numpyro_platform(use_gpu: bool, num_cpus: int) -> None:
    """Set the NumPyro compute platform for the current process.

    Must be called before any JAX or NumPyro operation, since the platform
    cannot be changed once JAX has initialised. On GPU, JAX falls back to CPU
    if no GPU is available. On CPU, the host device count is set so NumPyro can
    run that many chains in parallel.

    Args:
        use_gpu: If True, run inference on a GPU (falling back to CPU if none
            is available). If False, run on CPU.
        num_cpus: Number of CPU devices to make available for parallel chains.
            Only applied when running on CPU.
    """
    if use_gpu:
        os.environ["JAX_PLATFORMS"] = "cuda,cpu"
        numpyro.set_platform("gpu")
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"
        numpyro.set_platform("cpu")
        numpyro.set_host_device_count(num_cpus)


def run_experiment(
        config: PathLike,
        experiment_name: str,
        custom_settings: dict | None = None,
        resume: bool = False,
        i_run: int = 0,
) -> None:
    """Set up and run a single sBayes analysis from a configuration file.

    Loads the experiment configuration and data, sets up the MCMC, and runs
    the sampler for one run. This is the core execution path called once per
    (run, cluster-count) combination.

    Args:
        config: Path to the YAML (or JSON) configuration file.
        experiment_name: Name used for logging and as the results directory name.
        custom_settings: Optional dict of config overrides applied on top of the
            configuration file (e.g. cluster count, number of runs).
        resume: If True, resume a previous run. Requires the experiment name,
            run ID and cluster count to match the previous run.
        i_run: Index of this run, distinguishing runs with the same cluster count
            and experiment name. Default is 0.
    """
    # Initialise the experiment

    with Experiment(config_file=config, experiment_name=experiment_name,
                    custom_settings=custom_settings, log=True, i_run=i_run) as experiment:
        data = Data.from_experiment(experiment)
        mcmc = MCMCSetup(data=data, experiment=experiment)
        mcmc.log_setup()
        mcmc.sample(run=i_run, resume=resume)

def runner(task: RunTask) -> None:
    """Execute a single analysis from a run task, for pool.map.

    Wraps `run_experiment` so it can be called with a single argument, as
    required by `multiprocessing.Pool.map`. Sets the NumPyro compute platform
    in the worker process (which must happen before any JAX/NumPyro operation),
    forces the run to a single cluster count and a single run, then delegates
    to `run_experiment`.

    Args:
        task: The run configuration to execute, as produced by `main`.
    """
    # Platform must be set in the worker process, before any JAX/NumPyro operation
    set_numpyro_platform(task.use_gpu, task.num_cpus)

    # Override the config for this specific run: one cluster count, one run
    run_settings = deepcopy(task.custom_settings) if task.custom_settings else {}
    update_recursive(
        run_settings,
        new_cfg={
            "model": {"clusters": task.n_clusters},
            "mcmc": {"runs": 1},
        },
    )

    run_experiment(
        config=task.config,
        experiment_name=task.experiment_name,
        custom_settings=run_settings,
        resume=task.resume,
        i_run=task.i_run,
    )


def main(
        config: PathLike,
        experiment_name: str | None = None,
        custom_settings: dict | None = None,
        processes: int = 1,
        resume: bool = False,
        n_clusters: int | list[int] | None = None,
        i_run: int | None = None,
        use_gpu: bool = False,
        num_cpus: int = 1,
) -> None:
    """Fan out an sBayes analysis into individual runs and execute them.

    Reads the run count and cluster counts from the configuration file (unless
    overridden by arguments), builds one `RunTask` per (run, cluster-count)
    combination, and executes them either sequentially or in parallel.

    Args:
        config: Path to the YAML (or JSON) configuration file.
        experiment_name: Name for logging and the results directory. Defaults
            to a date/time stamp if not given.
        custom_settings: Optional config overrides applied on top of the file.
        processes: Number of parallel worker processes. 1 (default) runs the
            tasks sequentially.
        resume: If True, resume previous runs.
        n_clusters: Cluster count(s) to run. Overrides the config value if given;
            a single int is wrapped in a list. Defaults to the config value.
        i_run: Index of a single run to execute. If None, all runs in
            range(config.mcmc.runs) are executed.
        use_gpu: If True, run inference on a GPU.
        num_cpus: Number of CPU devices for parallel chains.
    """
    # Read run and cluster settings from the config (logging off — this is planning only)
    experiment = Experiment(
        config_file=config,
        experiment_name=experiment_name,
        custom_settings=custom_settings,
        log=False,
        create_experiment_folder=False,
        copy_config=False
    )

    # Run IDs: a single run via argument, or range(runs) from the config
    n_runs = experiment.config.mcmc.runs
    if i_run is None:
        i_run_range = list(range(n_runs))
    else:
        i_run_range = [i_run]

    # Cluster counts: from arguments (overriding config) or from the config
    if n_clusters is None:
        n_clusters = experiment.config.model.clusters
    else:
        warnings.warn(
            f"The number of clusters was set as a command-line argument, so the "
            f"config file entry `clusters={experiment.config.model.clusters}` will be ignored."
        )
    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]

    # One task per (run, cluster-count) combination
    tasks = [
        RunTask(
            i_run=i,
            n_clusters=k,
            config=config,
            experiment_name=experiment.experiment_name,
            custom_settings=custom_settings,
            resume=resume,
            use_gpu=use_gpu,
            num_cpus=num_cpus,
        )
        for i in i_run_range
        for k in n_clusters
    ]

    # Execute sequentially or in parallel
    if processes <= 1:
        for task in tasks:
            runner(task)
    else:
        import multiprocessing
        pool = multiprocessing.Pool(processes=processes)
        pool.map(runner, tasks)


def cli() -> None:
    """Parse command-line arguments and launch an sBayes analysis."""
    parser = argparse.ArgumentParser(
        description="Bayesian inference of clusters in the presence of confounders."
    )

    # The only required (positional) argument is the path to the config file:
    parser.add_argument(
        "config",
        type=Path,
        help="The YAML (or JSON) configuration file.",
    )

    # Optional named arguments:
    parser.add_argument(
        "-n", "--name",
        nargs="?", type=str,
        help="Experiment name, used for logging and as the results directory name "
             "(default: the current date/time).",
    )
    parser.add_argument(
        "-t", "--threads",
        nargs="?", type=PositiveInt, default=1,
        help="Number of parallel runs. Default 1 (all runs executed sequentially).",
    )
    parser.add_argument(
        "-r", "--resume",
        action="store_true",
        help="Resume a previous run (requires experiment name, run ID and cluster "
             "count to match).",
    )
    parser.add_argument(
        "-g", "--gpu",
        action="store_true",
        help="Run NumPyro inference on a GPU.",
    )
    parser.add_argument(
        "-c", "--numCPUs",
        nargs="?", type=int, default=1,
        help="Number of CPUs to use for parallel NumPyro chains.",
    )
    parser.add_argument(
        "-K", "--numClusters",
        nargs="*", type=PositiveInt,
        help="[DEVELOPER OPTION] Number of clusters (overrides the config value). "
             "Multiple values result in multiple runs.",
    )
    parser.add_argument(
        "-i", "--runID",
        nargs="?", type=PositiveInt,
        help="[DEVELOPER OPTION] Index of this run, to distinguish runs with the "
             "same cluster count and experiment name.",
    )

    args = parser.parse_args()

    # Set the NumPyro platform before any JAX/NumPyro operation
    set_numpyro_platform(args.gpu, args.numCPUs)

    main(
        config=args.config,
        experiment_name=args.name,
        processes=args.threads,
        resume=args.resume,
        n_clusters=args.numClusters,
        i_run=args.runID,
        use_gpu=args.gpu,
        num_cpus=args.numCPUs,
    )


if __name__ == "__main__":
    cli()

