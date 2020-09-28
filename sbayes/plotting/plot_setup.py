""" Class Plot

Defines general functions which are used in the child classes Trace and Map
Manages loading of input data, config files and the general graphic parameters of the plots
"""

import csv
import json
import os

import matplotlib.pyplot as plt

from sbayes.preprocessing import compute_network, read_sites, assign_family
from sbayes.util import parse_area_columns, read_features_from_csv


class Plot:
    def __init__(self, simulated_data=False):

        # Flag for simulation
        self.is_simulation = simulated_data

        # Config variables
        self.config = {}
        self.config_file = None

        # Path variables
        self.path_results = None
        self.path_data = None
        self.path_plots = None

        self.path_areas = None
        self.path_stats = None

        if self.is_simulation:
            self.path_ground_truth_areas = None
            self.path_ground_truth_stats = None

        self.number_features = 0

        # Input sites, site_names, network, ...
        self.sites = None
        self.site_names = None
        self.network = None
        self.locations = None
        self.dist_mat = None
        self.families = None
        self.family_names = None

        # Dictionary with all the MCMC results
        self.results = {}

        # Needed for the weights and parameters plotting
        plt.style.use('seaborn-paper')
        plt.tight_layout()

    ####################################
    # Configure the parameters
    ####################################
    def load_config(self, config_file):

        # Get parameters from config_file
        self.config_file = config_file

        # Read config file
        self.read_config()

        # Convert lists to tuples
        self.convert_config(self.config)

        # Verify config
        self.verify_config()

        # Assign global variables for more convenient workflow
        self.path_results = self.config['input']['path_plots']
        self.path_data = self.config['input']['path_data']
        self.path_plots = self.config['input']['path_plots'] + '/plots'
        self.path_areas = self.config['input']['path_areas']
        self.path_stats = self.config['input']['path_stats']

        if self.is_simulation:
            self.path_ground_truth_areas = self.config['input']['path_ground_truth_areas']
            self.path_ground_truth_stats = self.config['input']['path_ground_truth_stats']

        if not os.path.exists(self.path_plots):
            os.makedirs(self.path_plots)

    def read_config(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

    @staticmethod
    def convert_config(d):
        for k, v in d.items():
            if isinstance(v, dict):
                Plot.convert_config(v)
            else:
                if k != 'scenarios' and k != 'post_freq_lines' and type(v) == list:
                    d[k] = tuple(v)

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
    def read_data(self):
        print('Reading input data...')
        if self.is_simulation:
            self.sites, self.site_names, _ = read_sites(self.path_data,
                                                        retrieve_family=True, retrieve_subset=True)
            self.families, self.family_names = assign_family(1, self.sites)
        else:
            self.sites, self.site_names, _, _, _, self.families, self.family_names, _ = \
                read_features_from_csv(self.path_data)
        self.network = compute_network(self.sites)
        self.locations, self.dist_mat = self.network['locations'], self.network['dist_mat']

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

                        # todo: fix
                        # # For ground truth
                        # if len(parsed_sample) == 1:
                        #     result[j] = parsed_sample[j]

                        # # For all samples
                        # else:
                        result[j].append(parsed_sample[j])

        return result

    # Helper function for read_stats
    # Used for reading: weights, alpha, beta, gamma
    @staticmethod
    def read_dictionary(txt_path, lines, current_key, search_key, param_dict):
        if 'ground_truth' in txt_path:
            if current_key.startswith(search_key):
                param_dict[current_key] = float(lines[current_key])
        else:
            if current_key.startswith(search_key):
                if current_key in param_dict:
                    param_dict[current_key].append(float(lines[current_key]))
                else:
                    param_dict[current_key] = []
                    param_dict[current_key].append(float(lines[current_key]))
        return param_dict

    # Helper function for read_stats
    # Used for reading: true_posterior, true_likelihood, true_prior,
    # true_weights, true_alpha, true_beta, true_gamma,
    # recall, precision
    @staticmethod
    def read_simulation_stats(txt_path, lines):
        true_weights, true_alpha, true_beta, true_gamma = {}, {}, {}, {}

        true_posterior = float(lines['posterior'])
        true_likelihood = float(lines['likelihood'])
        true_prior = float(lines['prior'])

        for key in lines:
            true_weights = Plot.read_dictionary(txt_path, lines, key, 'w_', true_weights)
            true_alpha = Plot.read_dictionary(txt_path, lines, key, 'alpha_', true_alpha)
            true_beta = Plot.read_dictionary(txt_path, lines, key, 'beta_', true_beta)
            true_gamma = Plot.read_dictionary(txt_path, lines, key, 'gamma_', true_gamma)

        return true_posterior, true_likelihood, true_prior, true_weights, true_alpha, true_beta, true_gamma

    # Helper function for read_stats
    # Bind all statistics together into the dictionary self.results
    def bind_stats(self, txt_path, sample_id, posterior, likelihood, prior,
                   weights, alpha, beta, gamma,
                   posterior_single_areas, likelihood_single_areas, prior_single_areas,
                   recall, precision,
                   true_posterior, true_likelihood, true_prior,
                   true_weights, true_alpha, true_beta, true_gamma, feature_names):

        if 'ground_truth' in txt_path:
            self.results['true_posterior'] = true_posterior
            self.results['true_likelihood'] = true_likelihood
            self.results['true_prior'] = true_prior
            self.results['true_weights'] = true_weights
            self.results['true_alpha'] = true_alpha
            self.results['true_beta'] = true_beta
            self.results['true_gamma'] = true_gamma

        else:
            self.results['sample_id'] = sample_id
            self.results['posterior'] = posterior
            self.results['likelihood'] = likelihood
            self.results['prior'] = prior
            self.results['weights'] = weights
            self.results['alpha'] = alpha
            self.results['beta'] = beta
            self.results['gamma'] = gamma
            self.results['posterior_single_areas'] = posterior_single_areas
            self.results['likelihood_single_areas'] = likelihood_single_areas
            self.results['prior_single_areas'] = prior_single_areas
            self.results['recall'] = recall
            self.results['precision'] = precision
            self.results['feature_names'] = feature_names

    # Read stats
    # Read the results from the files:
    # ground_truth/stats.txt
    # <experiment_path>/stats_<scenario>.txt
    def read_stats(self, txt_path, simulation_flag):
        sample_id, posterior, likelihood, prior, recall, precision = [], [], [], [], [], []
        weights, alpha, beta, gamma, posterior_single_areas, likelihood_single_areas, prior_single_areas =\
            {}, {}, {}, {}, {}, {}, {}
        true_posterior, true_likelihood, true_prior, true_weights, \
            true_alpha, true_beta, true_gamma = None, None, None, None, None, None, None

        with open(txt_path, 'r') as f_stats:
            csv_reader = csv.DictReader(f_stats, delimiter='\t')
            for lines in csv_reader:
                try:
                    sample_id.append(int(lines['Sample']))
                except KeyError:
                    pass
                posterior.append(float(lines['posterior']))
                likelihood.append(float(lines['likelihood']))
                prior.append(float(lines['prior']))

                for key in lines:
                    weights = Plot.read_dictionary(txt_path, lines, key, 'w_', weights)
                    alpha = Plot.read_dictionary(txt_path, lines, key, 'alpha_', alpha)
                    beta = Plot.read_dictionary(txt_path, lines, key, 'beta_', beta)
                    gamma = Plot.read_dictionary(txt_path, lines, key, 'gamma_', gamma)
                    posterior_single_areas = Plot.read_dictionary(txt_path, lines, key, 'post_', posterior_single_areas)
                    likelihood_single_areas = Plot.read_dictionary(txt_path, lines, key, 'lh_', likelihood_single_areas)
                    prior_single_areas = Plot.read_dictionary(txt_path, lines, key, 'prior_', prior_single_areas)

                if simulation_flag:
                    if 'ground_truth' in txt_path:
                        true_posterior, true_likelihood, true_prior, true_weights,\
                            true_alpha, true_beta, true_gamma = Plot.read_simulation_stats(txt_path, lines)
                    else:
                        recall.append(float(lines['recall']))
                        precision.append(float(lines['precision']))

        # Names of distinct features
        feature_names = []
        for key in weights:
            if 'universal' in key:
                feature_names.append(str(key).rsplit('_', 1)[1])

        self.bind_stats(txt_path, sample_id, posterior, likelihood, prior, weights, alpha, beta, gamma,
                        posterior_single_areas, likelihood_single_areas, prior_single_areas, recall, precision,
                        true_posterior, true_likelihood, true_prior, true_weights, true_alpha, true_beta, true_gamma, feature_names)

    # Read results
    # Call all the previous functions
    # Bind the results together into the results dictionary
    def read_results(self, model=None):

        self.results = {}
        if model is None:
            print('Reading results...')
            path_areas = self.path_areas
            path_stats = self.path_stats

        else:
            print('Reading results of model %s...' % model)
            path_areas = [p for p in self.path_areas if 'areas_' + str(model) + '_' in p][0]
            path_stats = [p for p in self.path_stats if 'stats_' + str(model) + '_' in p][0]

        self.results['areas'] = self.read_areas(path_areas)
        self.read_stats(path_stats, self.is_simulation)

        # Read ground truth files
        if self.is_simulation:

            if model is None:
                path_ground_truth_areas = self.path_ground_truth_areas
                path_ground_truth_stats = self.path_ground_truth_stats

            else:
                path_ground_truth_areas = [p for p in self.path_ground_truth_areas if
                                           str(model) + '/ground_truth/areas' in p][0]
                path_ground_truth_stats = [p for p in self.path_ground_truth_stats if
                                           str(model) + '/ground_truth/stats' in p][0]

            self.results['true_zones'] = self.read_areas(path_ground_truth_areas)
            self.read_stats(path_ground_truth_stats, self.is_simulation)

    def get_model_names(self):

        last_part = [p.rsplit('/', 1)[-1] for p in list(self.path_areas)]
        name = [p.rsplit('_')[1] for p in last_part]

        return name
