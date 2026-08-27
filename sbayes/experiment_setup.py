from __future__ import annotations
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from sbayes.util import set_experiment_name, PathLike
from sbayes.config.config import SBayesConfig

class Experiment:
    """An sBayes experiment: loads and validates the config, sets up paths and logging.

    An Experiment bundles everything needed to run an analysis from a config file:
    the parsed configuration, the results directory, and the logger. It is
    constructed once per run (for the actual analysis) and may also be constructed
    without side effects for planning (see `create_experiment_folder` / `copy_config`).

    Attributes:
        experiment_name: Name of this experiment run (and of the results folder).
        i_run: Index of this run, distinguishing runs with the same config.
        config: The parsed and validated configuration.
        path_results: Path to the results directory, or None if not created.
        logger: The logger used throughout the run.
    """

    def __init__(
        self,
        config_file: PathLike,
        experiment_name: str | None = None,
        custom_settings: dict | None = None,
        log: bool = True,
        i_run: int = 0,
        create_experiment_folder: bool = True,
        copy_config: bool = True,
    ):
        """Initialise an experiment from a configuration file.

        Args:
            config_file: Path to the YAML (or JSON) configuration file.
            experiment_name: Name for the run and results folder. Defaults to a
                generated date/time-based name if not given.
            custom_settings: Optional config overrides applied on top of the file.
            log: If True, set up a logger with stream and file handlers and write
                the initial experiment log. If False, use a no-op logger.
            i_run: Index of this run. Used in the log filename and stored for later.
            create_experiment_folder: If True, create the results directory. Set
                False for a planning-only experiment that should not touch the disk.
            copy_config: If True (and the results folder exists), copy the config
                file into the results directory for reproducibility.
        """
        # Resolve the experiment name (generate one if not provided)
        self.experiment_name = experiment_name or set_experiment_name()
        self.i_run = i_run

        # Load, merge overrides, and validate the config
        self.config = SBayesConfig.from_config_file(config_file, custom_settings)

        # Create the results directory (unless this is a planning-only experiment)
        self.path_results = None
        if create_experiment_folder:
            self.path_results = self.init_results_directory(self.config, self.experiment_name)

        # Set up logging
        if log:
            self.logger = self.init_logger(f"{self.experiment_name}.{self.i_run}")
            if self.path_results is not None:
                self.add_logger_file(self.path_results)
            self.log_experiment()
        else:
            self.logger = logging.getLogger(f"sbayes.{self.experiment_name}.{self.i_run}")
            self.logger.addHandler(logging.NullHandler())

        # Copy the config into the results directory for reproducibility
        if copy_config and self.path_results is not None:
            shutil.copy(
                src=config_file,
                dst=self.path_results / Path(config_file).name,
            )

    def __enter__(self) -> "Experiment":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def init_results_directory(config: SBayesConfig, experiment_name: str) -> Path:
        """Create and return the results directory for this experiment.

        Args:
            config: The parsed configuration (provides the base results path).
            experiment_name: Name of this experiment, used as the subdirectory name.

        Returns:
            The path to the created results directory.
        """
        path_results = config.results.path / experiment_name
        path_results.mkdir(parents=True, exist_ok=True)
        return path_results

    @staticmethod
    def init_logger(name: str) -> logging.Logger:
        """Return a run-specific logger, adding a stream handler if not present.

        Uses a per-run logger name so that each run logs independently. The
        handler guard makes repeated calls idempotent: the stream handler is
        added only once, even if the same logger is requested again.

        Args:
            name: Run-specific suffix identifying this logger (e.g.
                "<experiment_name>.<i_run>"), namespaced under "sbayes.".

        Returns:
            The logger for this run.
        """
        logger = logging.getLogger(f"sbayes.{name}")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        return logger

    def add_logger_file(self, path_results: Path) -> None:
        """Add a file handler writing this run's log to the results directory.

        The log file is named per run (by cluster count and run index) so that
        parallel or repeated runs do not write to the same file. An existing log
        file at that path is removed first so each run starts with a fresh log.

        Args:
            path_results: The results directory in which to create the log file.
        """
        log_path = path_results / f"experiment_{self.config.model.clusters}_{self.i_run}.log"
        # Remove an existing log file so the run starts fresh; ignore if absent
        try:
            os.remove(log_path)
        except FileNotFoundError:
            pass
        log_file_handler = logging.FileHandler(filename=log_path)
        self.logger.addHandler(log_file_handler)

    def log_experiment(self) -> None:
        """Write the initial experiment information to the log.

        Records the experiment name, results location, and start time. Assumes the
        logger and its handlers are already set up. Called once at construction.
        """
        self.logger.info("Experiment: %s", self.experiment_name)
        self.logger.info("File location for results: %s", self.path_results)
        self.logger.info("Start time and date: %s",
                         datetime.now().strftime("%H:%M:%S %d.%m.%Y"))

    def close(self) -> None:
        """Close and detach all log handlers for this run.

        Closes each handler (flushing and releasing file handles) and removes it
        from the logger, so the per-run logger is left clean. Call when the run
        is finished.
        """
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
