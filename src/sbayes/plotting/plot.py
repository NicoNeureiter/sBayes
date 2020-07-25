""" Class Plot

Defines general functions which are used in the child classes Trace and Map
Manages loading of input data, config files and the general graphic parameters of the plots
This class is not used for plotting itself
"""

import csv
import json
import os

from sbayes.preprocessing import compute_network, read_sites
from sbayes.util import parse_area_columns


class Plot:
    def __init__(self, simulation=False):

        # Flag for simulation
        self.is_simulation = simulation

        # Config variables
        self.config = {}
        self.config_file = None

        # Path variables
        self.path_results = None
        self.path_data = None
        self.path_plots = None

        # Input areas and stats
        self.areas = []
        self.stats = []

        # Input sites, site_names, network
        self.sites = None
        self.site_names = None
        self.network = None

        # Input ground truth areas and stats (for simulation)
        if self.is_simulation:
            self.areas_ground_truth = []

        # Dictionary with all the input results
        self.results = {}

    ####################################
    # Configure the parameters
    ####################################
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

    ####################################
    # Read the data and the results
    ####################################

    # Functions related to the current scenario (run in a loop over scenarios, i.e. n_zones)
    # Set the results path for the current scenario
    def set_scenario_path(self, current_scenario):
        current_run_path = f"{self.path_plots}/n{self.config['input']['run']}_{current_scenario}/"

        if not os.path.exists(current_run_path):
            os.makedirs(current_run_path)

        return current_run_path

    # Read sites, site_names, network
    # Read the data from the file sites.csv
    def read_data(self, data_path):
        self.sites, self.site_names, _ = read_sites(data_path)
        self.network = compute_network(self.sites)

    # Read areas
    # Read the data from the files:
    # ground_truth/areas.txt
    # <experiment_path>/areas_<scenario>.txt
    @staticmethod
    def read_areas(txt_path):
        result = []

        with open(txt_path, 'r') as f_sample:

            # This makes len(result) = number of areas (flipped array)

            # Split the sample
            # len(byte_results) equals the number of samples
            byte_results = (f_sample.read()).split('\n')

            # Get the number of areas
            n_areas = len(byte_results[0].split('\t'))

            # Append empty arrays to result, so that len(result) = n_areas
            for i in range(n_areas):
                result.append([])

            # Process each sample
            for sample in byte_results:

                # Exclude empty lines
                if len(sample) > 0:

                    # Parse each sample
                    # len(parsed_result) equals the number of areas
                    # parse_area_columns.shape equals (n_areas, n_sites)
                    parsed_sample = parse_area_columns(sample)

                    # Add each item in parsed_area_columns to the corresponding array in result
                    for j in range(len(parsed_sample)):

                        # For ground truth
                        if len(parsed_sample) == 1:
                            result[j] = parsed_sample[j]

                        # For all samples
                        else:
                            result[j].append(parsed_sample[j])

        return result

    # Helper function for read_stats
    # Used for reading: weights, alpha, beta, gamma
    @staticmethod
    def read_dictionary(txt_path, lines, current_key, search_key, param_dict):
        if 'ground_truth' in txt_path:
            if current_key.startswith(search_key):
                param_dict[current_key] = lines[current_key]
        else:
            if current_key.startswith(search_key):
                if current_key in param_dict:
                    param_dict[current_key].append(lines[current_key])
                else:
                    param_dict[current_key] = []
        return param_dict

    # Helper function for read_stats
    # Used for reading: true_posterior, true_likelihood, true_prior,
    # true_weights, true_alpha, true_beta, true_gamma,
    # recall, precision
    @staticmethod
    def read_simulation_stats(txt_path, lines):
        recall, precision, true_families = [], [], []
        true_weights, true_alpha, true_beta, true_gamma = {}, {}, {}, {}
        true_posterior, true_likelihood, true_prior = 0, 0, 0

        if 'ground_truth' in txt_path:
            true_posterior = lines['posterior']
            true_likelihood = lines['likelihood']
            true_prior = lines['prior']

            for key in lines:
                true_weights = Plot.read_dictionary(txt_path, lines, key, 'w_', true_weights)
                true_alpha = Plot.read_dictionary(txt_path, lines, key, 'alpha_', true_alpha)
                true_beta = Plot.read_dictionary(txt_path, lines, key, 'beta_', true_beta)
                true_gamma = Plot.read_dictionary(txt_path, lines, key, 'gamma_', true_gamma)

        else:
            recall.append(lines['recall'])
            precision.append(lines['precision'])

        return recall, precision, \
            true_posterior, true_likelihood, true_prior, \
            true_weights, true_alpha, true_beta, true_gamma

    # Helper function for read_stats
    # Bind all statistics together into the dictionary self.results
    def bind_stats(self, txt_path, posterior, likelihood, prior,
                   weights, alpha, beta, gamma,
                   posterior_single_zones, likelihood_single_zones, prior_single_zones,
                   recall, precision,
                   true_posterior, true_likelihood, true_prior,
                   true_weights, true_alpha, true_beta, true_gamma):

        if 'ground_truth' in txt_path:
            self.results['true_posterior'] = true_posterior
            self.results['true_likelihood'] = true_likelihood
            self.results['true_prior'] = true_prior
            self.results['true_weights'] = true_weights
            self.results['true_alpha'] = true_alpha
            self.results['true_beta'] = true_beta
            self.results['true_gamma'] = true_gamma

        else:
            self.results['posterior'] = posterior
            self.results['likelihood'] = likelihood
            self.results['prior'] = prior
            self.results['weights'] = weights
            self.results['alpha'] = alpha
            self.results['beta'] = beta
            self.results['gamma'] = gamma
            self.results['posterior_single_zones'] = posterior_single_zones
            self.results['likelihood_single_zones'] = likelihood_single_zones
            self.results['prior_single_zones'] = prior_single_zones
            self.results['recall'] = recall
            self.results['precision'] = precision

    # Read stats
    # Read the results from the files:
    # ground_truth/stats.txt
    # <experiment_path>/stats_<scenario>.txt
    def read_stats(self, txt_path, simulation_flag):
        posterior, likelihood, prior = [], [], []
        weights, alpha, beta, gamma, posterior_single_zones, likelihood_single_zones, prior_single_zones =\
            {}, {}, {}, {}, {}, {}, {}
        recall, precision, true_posterior, true_likelihood, true_prior, true_weights, \
            true_alpha, true_beta, true_gamma = None, None, None, None, None, None, None, None, None

        with open(txt_path, 'r') as f_stats:
            csv_reader = csv.DictReader(f_stats, delimiter='\t')
            for lines in csv_reader:
                posterior.append(lines['posterior'])
                likelihood.append(lines['likelihood'])
                prior.append(lines['prior'])

                for key in lines:
                    weights = Plot.read_dictionary(txt_path, lines, key, 'w_', weights)
                    alpha = Plot.read_dictionary(txt_path, lines, key, 'alpha_', alpha)
                    beta = Plot.read_dictionary(txt_path, lines, key, 'beta_', beta)
                    gamma = Plot.read_dictionary(txt_path, lines, key, 'gamma_', gamma)
                    posterior_single_zones = Plot.read_dictionary(txt_path, lines, key, 'post_', posterior_single_zones)
                    likelihood_single_zones = Plot.read_dictionary(txt_path, lines, key, 'lh_', likelihood_single_zones)
                    prior_single_zones = Plot.read_dictionary(txt_path, lines, key, 'prior_', prior_single_zones)

                if simulation_flag:
                    recall, precision, true_posterior, true_likelihood, true_prior, \
                        true_weights, true_alpha, true_beta, true_gamma = Plot.read_simulation_stats(txt_path, lines)

        self.bind_stats(txt_path, posterior, likelihood, prior, weights, alpha, beta, gamma, posterior_single_zones,
                        likelihood_single_zones, prior_single_zones, recall, precision, true_posterior,
                        true_likelihood, true_prior, true_weights, true_alpha, true_beta, true_gamma)

    # Read results
    # Call all the previous functions
    # Bind the results together into the results dictionary
    def read_results(self, current_scenario):

        # Read areas
        areas_path = f"{self.path_results}/n{self.config['input']['run']}/" \
                     f"areas_n{self.config['input']['run']}_{current_scenario}.txt"
        self.areas = self.read_areas(areas_path)
        self.results['zones'] = self.areas

        # Read stats
        stats_path = f"{self.path_results}/n{self.config['input']['run']}/" \
                     f"stats_n{self.config['input']['run']}_{current_scenario}.txt"
        self.read_stats(stats_path, self.is_simulation)

        # Read ground truth files
        if self.is_simulation:
            areas_ground_truth_path = f"{self.path_results}/n{self.config['input']['run']}/ground_truth/areas.txt"
            self.areas_ground_truth = self.read_areas(areas_ground_truth_path)
            self.results['true_zones'] = self.areas_ground_truth

            stats_ground_truth_path = f"{self.path_results}/n{self.config['input']['run']}/ground_truth/stats.txt"
            self.read_stats(stats_ground_truth_path, self.is_simulation)
