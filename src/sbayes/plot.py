""" Class Plot and its child classes Trace, Map

Currently takes pickle files as input (to be changed to csv later)
After the input files will be changed, these functions should be moved to MCMC:

samples2res, match_zones and rank_zones
"""

import json
import os

from src.util import load_from


# General Plot class
# Manages loading of input data, config files and plot graphic parameters
class Plot:
    def __init__(self):
        self.config = {}
        self.config_file = None

        self.path_results = None
        self.path_data = None
        self.path_plots = None

    # Config functions
    def load_config(self, config_file):

        # Get parameters from config_file
        self.config_file = config_file

        # Read config file
        self.read_config()

        # Verify config
        self.verify_config()

        # Assign global variables for more convenient workflow
        self.path_results = self.config['input']['path_results']
        self.path_data = self.config['input']['path_data']
        self.path_plots = self.config['input']['path_results'] + '/plots'

        if not os.path.exists(self.path_plots):
            os.makedirs(self.path_plots)

    def read_config(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

    def verify_config(self):
        pass

    # Functions related to the current scenario (run in a loop over scenarios, i.e. n_zones)
    def set_scenario_path(self, current_scenario):
        current_run_path = f"{self.path_plots}/nz{current_scenario}_{self.config['input']['run']}/"

        if not os.path.exists(current_run_path):
            os.makedirs(current_run_path)

    # This should be later rewritten to take csv as input
    def read_results(self, current_scenario):
        sample_path = f"{self.path_results}/number_zones_n{current_scenario}_{self.config['input']['run']}.pkl"
        samples = load_from(sample_path)
        return samples


# Child class Trace
# Inherits basic functions from Plot, has specific functions for trace plots
class Trace(Plot):
    pass


# Child class Map
# Inherits basic functions from Plot, has specific functions for maps
class Map(Plot):
    pass
