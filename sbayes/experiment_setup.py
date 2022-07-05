#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the Experiment"""
import logging
import os
from pathlib import Path
from typing import Optional

from sbayes.util import set_experiment_name
from sbayes.config.config import SBayesConfig


class Experiment:

    """sBayes experiment class. Takes care of loading and verifying the config file,
    handling paths, setting up logging...

    Attributes:
        experiment_name (str): The name of the experiment run (= name of results folder)
        config_file (Path): The path to the config_file.
        config (dict): The config parsed into a python dictionary.
        base_directory (Path): The directory containing the config file.
        path_results (Path): The path to the results folder.
        logger (logging.Logger): The logger used throughout the run of the experiment.

    """

    def __init__(self,
                 experiment_name: str = None,
                 config_file: Optional[Path] = None,
                 custom_settings: Optional[dict] = None,
                 log: bool = True):

        # Naming and shaming
        if experiment_name is None:
            self.experiment_name = set_experiment_name()
        else:
            self.experiment_name = experiment_name

        self.config_file = None
        self.config = {}
        self.base_directory = None
        self.path_results = None

        self.logger = self.init_logger()

        if config_file is not None:
            self.load_config(config_file, custom_settings=custom_settings)

        if log:
            self.log_experiment()

    def load_config(self,
                    config_file: Path,
                    custom_settings: Optional[dict] = None):

        self.config = SBayesConfig.from_config_file(config_file, custom_settings)

        # Set results path
        self.path_results = self.config.results.path / self.experiment_name

        if not os.path.exists(self.path_results):
            os.makedirs(self.path_results)

        self.add_logger_file(self.path_results)

    @staticmethod
    def init_logger():
        logger = logging.Logger('sbayesLogger', level=logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
        return logger

    def add_logger_file(self, path_results):
        log_path = path_results / 'experiment.log'
        self.logger.addHandler(logging.FileHandler(filename=log_path))

    def log_experiment(self):
        self.logger.info("Experiment: %s", self.experiment_name)
        self.logger.info("File location for results: %s", self.path_results)
