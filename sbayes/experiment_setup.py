from __future__ import annotations
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from sbayes.util import set_experiment_name, PathLike
from sbayes.config.config import SBayesConfig


class Experiment:

    """sBayes experiment class. Takes care of loading and verifying the config file,
    handling paths, setting up logging...

    Attributes:
        experiment_name (str): The name of the experiment run (= name of results folder)
        config (SBayesConfig): The config parsed into a python dictionary.
        path_results (Path): The path to the results-directory.
        logger (logging.Logger): The logger used throughout the run of the experiment.
    """

    def __init__(
        self,
        config_file: PathLike,
        experiment_name: str | None = None,
        custom_settings: dict | None = None,
        log: bool = True,
        i_run: int = 0,
    ):

        # Naming and shaming
        self.experiment_name = experiment_name or set_experiment_name()

        # Remember the index of this experiment run
        self.i_run = i_run

        # Load and parse the config file
        self.config = SBayesConfig.from_config_file(config_file, custom_settings)

        # Set results path
        self.path_results = self.init_results_directory(self.config, self.experiment_name)

        # Print the initial log message
        if log:
            # Initialize the logger
            self.logger = self.init_logger()
            self.log_experiment()

        # Copy the config file to the results-directory
        shutil.copy(
            src=config_file,
            dst=self.path_results / os.path.basename(config_file)
        )

    def init_results_directory(self, config: SBayesConfig, experiment_name: str) -> Path:
        """Create subdirectory for this experiment, add it to the logger and the return path."""
        path_results = config.results.path / experiment_name
        os.makedirs(path_results, exist_ok=True)
        return path_results

    @staticmethod
    def init_logger() -> logging.Logger:
        """Initialize the logger with a stream handler"""
        logger = logging.Logger("sbayesLogger", level=logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
        return logger

    def add_logger_file(self, path_results: Path):
        """Add a file handler to write logging information to a log-file"""
        log_path = path_results / f"experiment_{self.config.model.clusters}_{self.i_run}.log"
        log_file_handler = logging.FileHandler(filename=log_path)
        self.logger.addHandler(log_file_handler)

    def log_experiment(self):
        """Start writing information on the experiment to the logger."""
        self.add_logger_file(self.path_results)
        self.logger.info("Experiment: %s", self.experiment_name)
        self.logger.info("File location for results: %s", self.path_results)
        self.logger.info("Start time and date: %s", datetime.now().strftime("%H:%M:%S %d.%m.%Y"))

    def close(self):
        """Close the log file handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
