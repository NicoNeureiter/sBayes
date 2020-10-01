import csv
import json
import math
import os
from copy import deepcopy
from itertools import compress
from statistics import median

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from descartes import PolygonPatch
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial import Delaunay
from shapely import geometry
from shapely.ops import cascaded_union, polygonize

from sbayes.postprocessing import compute_dic
from sbayes.preprocessing import compute_network, read_sites, assign_family
from sbayes.util import add_edge, compute_delaunay
from sbayes.util import fix_default_config
from sbayes.util import gabriel_graph_from_delaunay
from sbayes.util import parse_area_columns, read_features_from_csv
from sbayes.util import round_int


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

        # Copy-pasted from Map
        self.base_directory = None

        # Path to all default configs
        #self.config_default = "config/plotting/config_plot_maps.json"
        self.config_default = "config/plotting"

        # Map parameters
        self.ax = None
        self.fig = None
        self.map_parameters = {}
        self.leg_zones = []
        self.all_labels = []
        self.area_labels = []

        self.leg_line_width = []
        self.line_width_label = []

        # Additional parameters
        self.world = None
        self.rivers = None

    # From plot_setup:

    ####################################
    # Configure the parameters
    ####################################
    def load_config(self, config_file):

        # Get parameters from config_custom (for particular experiment)
        self.config_file = config_file

        # Read config file
        self.read_config()

        # Convert lists to tuples
        self.convert_config(self.config)

        # Verify config
        self.verify_config()

        # Assign global variables for more convenient workflow
        self.path_results = self.config['input']['path_results']
        self.path_data = self.config['input']['path_data']
        self.path_plots = self.config['input']['path_results'] + '/plots'
        self.path_areas = list(self.config['input']['path_areas'])
        self.path_stats = list(self.config['input']['path_stats'])

        if self.is_simulation:
            self.path_ground_truth_areas = self.config['input']['path_ground_truth_areas']
            self.path_ground_truth_stats = self.config['input']['path_ground_truth_stats']

        if not os.path.exists(self.path_plots):
            os.makedirs(self.path_plots)

        self.add_default_configs()

    def read_config(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

    # Load default config parameters
    def add_default_configs(self):

        self.config_default = fix_default_config(self.config_default)

        for root, dirs, files in os.walk(self.config_default):
            for file in files:
                current_config = os.path.join(root, file)

                with open(current_config, 'r') as f:
                    new_config = json.load(f)

                    # If the key already exists
                    for key in new_config:
                        if key in self.config:
                            self.config[key].update(new_config[key])
                        else:
                            self.config[key] = new_config[key]

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
        weights, alpha, beta, gamma, posterior_single_areas, likelihood_single_areas, prior_single_areas = \
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
                    posterior_single_areas = Plot.read_dictionary(txt_path, lines, key, 'post_',
                                                                  posterior_single_areas)
                    likelihood_single_areas = Plot.read_dictionary(txt_path, lines, key, 'lh_',
                                                                   likelihood_single_areas)
                    prior_single_areas = Plot.read_dictionary(txt_path, lines, key, 'prior_', prior_single_areas)

                if simulation_flag:
                    if 'ground_truth' in txt_path:
                        true_posterior, true_likelihood, true_prior, true_weights, \
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
                        true_posterior, true_likelihood, true_prior, true_weights, true_alpha, true_beta,
                        true_gamma, feature_names)

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
        last_part = [p.rsplit('/', 1)[-1] for p in self.path_areas]
        name = [p.rsplit('_')[1] for p in last_part]
        return name

    # From map.py:

    ##############################################################
    # Copy-pasted functions needed for plot_posterior_map
    ##############################################################
    # for Olga: For maps show should always be False, and can be removed
    def style_axes(self, ax, locations, show=True, offset=None, x_extend=None, y_extend=None):
        """ Function to style the axes of a plot
        Args:
            ax (axes.Axes): Axis of the plot.
            locations (np.array): Locations of points.
            show (bool): Whether to show the axes.
            offset (float): Offset of axes.
            x_extend (tuple): x extend of plot.
            y_extend (tuple): y extend of plot.
        Returns:
            (tuple): Extend of plot.
        """

        pp = self.map_parameters
        # getting axes ranges and rounding them
        x_min, x_max = np.min(locations[:, 0]), np.max(locations[:, 0])
        y_min, y_max = np.min(locations[:, 1]), np.max(locations[:, 1])

        # if specific offsets were passed use them, otherwise use same offset for all
        # For Olga: offsets should be provided in the config only, and if missing deduced from the data
        if offset is not None:
            # For Olga: remove
            x_min, x_max = round_int(x_min, 'down', offset), round_int(x_max, 'up', offset)
            y_min, y_max = round_int(y_min, 'down', offset), round_int(y_max, 'up', offset)
        elif self.config['graphic']['x_extend'] is not None and self.config['graphic']['y_extend'] is not None:
            x_min, x_max = self.config['graphic']['x_extend']
            y_min, y_max = self.config['graphic']['y_extend']
        else:
            x_coords, y_coords = locations.T
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
        # setting axes limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        # x axis
        # for Olga: For maps show should always be False, and can be removed
        x_step = (x_max - x_min) // 5
        x_ticks = np.arange(x_min, x_max + x_step, x_step) if show else []
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, fontsize=pp['fontsize'])

        # y axis
        # for : For maps show should always be False, and can be removed
        y_step = (y_max - y_min) // 5
        y_ticks = np.arange(y_min, y_max + y_step, y_step) if show else []
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, fontsize=pp['fontsize'])

        return (x_min, x_max, y_min, y_max)

    def compute_alpha_shapes(self, sites, alpha):

        # Olga: sites, net are already loaded in Plot

        """Compute the alpha shape (concave hull) of a set of sites
        Args:
            sites (np.array): subset of sites around which to create the alpha shapes (e.g. family, zone, ...)
            net (dict): The full network containing all sites.
            alpha (float): alpha value to influence the gooeyness of the convex hull Smaller numbers don't fall inward
            as much as larger numbers. Too large, and you lose everything!"

        Returns:
            (polygon): the alpha shape"""

        all_sites = self.network['locations']
        points = all_sites[sites[0]]
        # print(points.shape)
        tri = Delaunay(points, qhull_options="QJ Pp")

        edges = set()
        edge_nodes = []

        # loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle
        for ia, ib, ic in tri.vertices:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]

            # Lengths of sides of triangle
            a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
            c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

            # Semiperimeter of triangle
            s = (a + b + c) / 2.0

            # Area of triangle by Heron's formula
            area = math.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_r = a * b * c / (4.0 * area)

            # print(f'{circum_r} {1.0/alpha}')
            if circum_r < 1.0 / alpha:
                add_edge(edges, edge_nodes, points, ia, ib)
                add_edge(edges, edge_nodes, points, ib, ic)
                add_edge(edges, edge_nodes, points, ic, ia)

        m = geometry.MultiLineString(edge_nodes)

        triangles = list(polygonize(m))
        polygon = cascaded_union(triangles)

        return polygon

    def add_zone_boundary(self, is_in_zone, alpha, annotation=None, color='#000000'):
        """ Function to add bounding boxes around zones
        Args:
            ax (axes.Axes): Axis of the plot.
            locations (np.array): Locations of points.
            net (unknown): Network of points.
            is_in_zone (np.array): Boolean array indicating if in zone.
            alpha (float): Value for alpha shapes.
            annotation (string): If passed, zone is annotated with this.
            color (string): Color of zone.
        Returns:
            leg_zones: Legend.
        """

        # use form plotting param
        fontsize = 18
        # color_zones = '#000000'

        cp_locations = self.locations[is_in_zone[0], :]

        leg_zone = None
        if cp_locations.shape[0] > 0:  # at least one contact point in zone

            alpha_shape = self.compute_alpha_shapes([is_in_zone], alpha)

            # smooth_shape = alpha_shape.buffer(100, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
            smooth_shape = alpha_shape.buffer(60, resolution=16, cap_style=1, join_style=1, mitre_limit=10.0)
            # smooth_shape = alpha_shape
            patch = PolygonPatch(smooth_shape, ec=color, lw=1, ls='-', alpha=1, fill=False,
                                 zorder=1)
            leg_zone = self.ax.add_patch(patch)
        else:
            print('computation of bbox not possible because no contact points')

        # only adding a label (numeric) if annotation turned on and more than one zone
        if annotation is not None:
            x_coords, y_coords = cp_locations.T
            x, y = np.mean(x_coords), np.mean(y_coords)
            self.ax.text(x, y, annotation, fontsize=fontsize, color=color)

        return leg_zone

    def areas_to_graph(self, area, burn_in, post_freq):

        # exclude burn-in
        end_bi = math.ceil(len(area) * burn_in)
        area = area[end_bi:]

        # compute frequency of each point in zone
        area = np.asarray(area)
        n_samples = area.shape[0]

        zone_freq = np.sum(area, axis=0)/n_samples
        in_graph = zone_freq >= post_freq
        locations = self.locations[in_graph]
        n_graph = len(locations)

        # getting indices of points in area
        area_indices = np.argwhere(in_graph)

        if n_graph > 3:
            # computing the delaunay
            delaunay = compute_delaunay(locations)
            graph_connections = gabriel_graph_from_delaunay(delaunay, locations)

        elif n_graph == 3:
            graph_connections = np.array([[0, 1], [1, 2], [2, 0]]).astype(int)

        elif n_graph == 2:
            graph_connections = np.array([[0, 1]]).astype(int)

        else:
            raise ValueError('No points in contact zone!')

        lines = []
        line_weights = []

        for index in graph_connections:
            # count how often p0 and p1 are together in the posterior of the area
            p = [area_indices[index[0]][0], area_indices[index[1]][0]]
            together_in_area = np.sum(np.all(area[:, p], axis=1)) / n_samples
            lines.append(self.locations[[*p]])
            line_weights.append(together_in_area)

        return in_graph, lines, line_weights

    ##############################################################
    # New functions needed for plot_posterior_map
    ##############################################################

    ##############################################################
    # Main initial functions for plot_posterior_map
    ##############################################################

    # Get relevant map parameters from the json file
    # for Olga: parameters should be defined in the config, rather than here. My bad, I know :)
    def get_map_parameters(self):
        self.map_parameters = self.config['plot_type']['general']
        if self.is_simulation:
            self.map_parameters.update(self.config['plot_type']['plot_posterior_map_simulated'])
        else:
            if self.config['input']['experiment'] == "sa":
                self.map_parameters.update(self.config['plot_type']['plot_posterior_map_sa'])
            if self.config['input']['experiment'] == "balkan":
                self.map_parameters.update(self.config['plot_type']['plot_posterior_map_balkan'])

    # Initialize the map
    def initialize_map(self):
        # Olga: this function should only read what is in the config, see inside
        self.get_map_parameters()
        plt.rcParams["axes.linewidth"] = self.map_parameters['frame_width']
        # for Olga: constrained layout drops a warning. Could you check?
        self.fig, self.ax = plt.subplots(figsize=(self.map_parameters['fig_width'],
                                                  self.map_parameters['fig_height']),
                                         constrained_layout=True)
        if self.config['input']['subset']:
            self.plot_subset()

        self.ax.scatter(*self.locations.T, s=self.config['graphic']['size'], c="darkgrey", alpha=1, linewidth=0)

    ##############################################################
    # Visualization functions for plot_posterior_map
    ##############################################################
    def add_color(self, i, flamingo, simulated_family):
        # The colors for each area could go to the config file
        # zone_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
        # This is actually sort of a joke: if one area has the shape of a flamingo, use a flamingo colour for it
        #  in the map. Anyway, colors should go to the config. If can be removed.

        if flamingo:
            # flamingo_color = '#F48AA7'
            color = self.config['graphic']['flamingo_color'] if len(self.results['areas']) == 1 \
                else self.config['graphic']['zone_colors'][i]
        else:
            color = self.config['graphic']['zone_colors'][i]
        # Same here: when simulating families, one has the shape of a banana. If so, use banana color.
        # Should go to the config. If can be removed.
        if simulated_family:
            # banana_color = '#f49f1c'
            color = self.config['graphic']['banana_color'] if len(self.results['areas']) == 1 \
                else self.config['graphic']['zone_colors'][i]
        return color

    def add_label(self, is_in_zone, current_color):
        # Find all languages in areas
        loc_in_zone = self.locations[is_in_zone, :]
        labels_in_zone = list(compress(self.sites['id'], is_in_zone))
        self.all_labels.append(labels_in_zone)

        for loc in range(len(loc_in_zone)):
            # add a label at a spatial offset of 20000 and 10000. Rather than hard-coding it,
            # this might go into the config.
            x, y = loc_in_zone[loc]
            x += 20000
            y += 10000
            # Same with the font size for annotations. Should probably go to the config.
            anno_opts = dict(xy=(x, y), fontsize=14, color=current_color)
            self.ax.annotate(labels_in_zone[loc] + 1, **anno_opts)

    # Bind together the functions above
    def visualize_areas(self, flamingo, simulated_family, post_freq, burn_in, label_languages, plot_area_stats):

        # If likelihood for single areas are displayed: add legend entries with likelihood information per area
        if plot_area_stats:
            self.add_likelihood_legend()
            self.add_likelihood_info()
        else:
            for i, _ in enumerate(self.results['areas']):
                self.area_labels.append(f'$Z_{i + 1}$')

        # Color areas
        for i, area in enumerate(self.results['areas']):
            current_color = self.add_color(i, flamingo, simulated_family)

            # This function computes a Delaunay graph for all points which are in the posterior with at least p_freq
            in_graph, lines, line_w = self.areas_to_graph(area, burn_in, post_freq=post_freq)

            self.ax.scatter(*self.locations[in_graph].T, s=self.config['graphic']['size'], c=current_color)

            for li in range(len(lines)):
                self.ax.plot(*lines[li].T, color=current_color, lw=line_w[li]*self.config['graphic']['size_line'],
                             alpha=0.6)

            # This adds small lines to the legend (one legend entry per area)
            line_legend = Line2D([0], [0], color=current_color, lw=6, linestyle='-')
            self.leg_zones.append(line_legend)

            # Labels the languages in the areas
            # Should go into a separate function
            if label_languages:
                self.add_label(in_graph, current_color)

            # Again, this is only relevant for simulated data and should go into a separate function
            # Olga: doesn't seem to be a good decision to move this out to a separate function,
            # because it's called inside of a loop and looks quite short
            if self.is_simulation:
                try:
                    # Adds a bounding box for the ground truth areas
                    # showing if the algorithm has correctly identified them
                    self.add_zone_boundary(self.results['true_zones'][i], alpha=0.001, color='#000000')
                # Olga: are there any other potential errors? can we somehow get rid of this try-except statement?
                # (maybe it would be better to add some 'verify' function above with raising some warning;
                # try-except is better to avoid)
                except IndexError:
                    continue

        # add to legend
        legend_zones = self.ax.legend(
            self.leg_zones,
            self.area_labels,
            title_fontsize=18,
            title='Contact areas',
            frameon=True,
            edgecolor='#ffffff',
            framealpha=1,
            fontsize=16,
            ncol=1,
            columnspacing=1,
            loc='upper left',
            bbox_to_anchor=self.map_parameters['area_legend_position']
        )
        legend_zones._legend_box.align = "left"
        self.ax.add_artist(legend_zones)

    # TODO: This function should be rewritten in a nicer way; probably split into two functions,
    #  or find a better way of dividing things into simulated and real
    def color_families(self, family_array, colors, families=None):
        # Initialize empty legend handle
        handles = []

        # Iterate over all family names
        for i, family in enumerate(family_array):

            family_color = colors[i]
            family_fill, family_border = family_color, family_color

            # Find all languages belonging to a family
            is_in_family = families[i] == 1
            family_locations = self.locations[is_in_family, :]

            # For simulated data
            if self.is_simulation:
                print(self.config['graphic']['family_alpha_shape'])
                alpha_shape = self.compute_alpha_shapes([is_in_family], self.config['graphic']['family_alpha_shape'])
                smooth_shape = alpha_shape.buffer(60, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
                patch = PolygonPatch(smooth_shape, fc=family_fill, ec=family_border, lw=1, ls='-', alpha=1, fill=True,
                                     zorder=-i)
                self.ax.add_patch(patch)

                # Add legend handle
                handle = Patch(facecolor=family_color, edgecolor=family_color, label="simulated family")

                handles.append(handle)

            # For real data
            else:

                # Adds a color overlay for each language in a family
                self.ax.scatter(*family_locations.T, s=self.config['graphic']['size'] * 15, c=family_color, alpha=1,
                                linewidth=0, zorder=-i,
                                label=family)

                # For languages with more than three members: instead of one dot per language,
                # combine several languages in an alpha shape (a polygon)
                if self.config['graphic']['family_alpha_shape'] is not None and np.count_nonzero(is_in_family) > 3:
                    alpha_shape = self.compute_alpha_shapes([is_in_family],
                                                            self.config['graphic']['family_alpha_shape'])

                    # making sure that the alpha shape is not empty
                    if not alpha_shape.is_empty:
                        smooth_shape = alpha_shape.buffer(40000, resolution=16, cap_style=1, join_style=1,
                                                          mitre_limit=5.0)
                        patch = PolygonPatch(smooth_shape, fc=family_fill, ec=family_border, lw=1, ls='-', alpha=1,
                                             fill=True, zorder=-i)
                        leg_family = self.ax.add_patch(patch)

                # Add legend handle
                handle = Patch(facecolor=family_color, edgecolor=family_color, label=family)

                handles.append(handle)

        # Olga: can we pass the parameters in a different way?
        # title_fontsize and fontsize are reversed, is that correct?
        if self.is_simulation:
            # Define the legend
            legend_families = self.ax.legend(
                handles=handles,
                title_fontsize=16,
                fontsize=18,
                frameon=True,
                edgecolor='#ffffff',
                framealpha=1,
                ncol=1,
                columnspacing=1,
                handletextpad=2.3,
                loc='upper left',
                bbox_to_anchor=self.map_parameters['family_legend_position']
            )
            self.ax.add_artist(legend_families)

        else:
            # (Hard-coded) parameters should probably go to config
            # Defines the legend for families
            legend_families = self.ax.legend(
                handles=handles,
                title='Language family',
                title_fontsize=18,
                fontsize=16,
                frameon=True,
                edgecolor='#ffffff',
                framealpha=1,
                ncol=1,
                columnspacing=1,
                loc='upper left',
                bbox_to_anchor=self.map_parameters['family_legend_position']
            )
            self.ax.add_artist(legend_families)

    def add_legend_lines(self, post_freq_lines):
        # This adds a legend displaying what the line thickness corresponds to.

        post_freq_lines.sort(reverse=True)

        # Iterates over all values in post_freq_lines and for each adds a legend entry
        for k in range(len(post_freq_lines)):

            # Create line
            line = Line2D([0], [0], color="black", lw=self.config['graphic']['size_line'] * post_freq_lines[k],
                          linestyle='-')
            self.leg_line_width.append(line)

            # Add legend text
            prop_l = int(post_freq_lines[k] * 100)
            self.line_width_label.append(f'{prop_l}%')

        # Adds everything to the legend
        legend_line_width = self.ax.legend(
                self.leg_line_width,
                self.line_width_label,
                title_fontsize=18,
                title='Frequency of edge in posterior',
                frameon=True,
                edgecolor='#ffffff',
                framealpha=1,
                fontsize=16,
                ncol=1,
                columnspacing=1,
                loc='upper left',
                bbox_to_anchor=self.map_parameters['freq_legend_position']
            )

        legend_line_width._legend_box.align = "left"
        self.ax.add_artist(legend_line_width)

    def add_sa_legend(self):
        self.config['graphic']['x_unit'] = (self.config['graphic']['x_extend'][1] -
                                            self.config['graphic']['x_extend'][0]) / 100
        self.config['graphic']['y_unit'] = (self.config['graphic']['y_extend'][1] -
                                            self.config['graphic']['y_extend'][0]) / 100
        self.ax.axhline(self.config['graphic']['y_extend'][0] + self.config['graphic']['y_unit'] * 71,
                        0.02, 0.20, lw=1.5, color="black")

        self.ax.add_patch(
            patches.Rectangle(
                (self.config['graphic']['x_extend'][0], self.config['graphic']['y_extend'][0]),
                self.config['graphic']['x_unit'] * 25, self.config['graphic']['y_unit'] * 100,
                color="white"
            ))

    def add_balkan_legend(self):
        self.config['graphic']['x_unit'] = (self.config['graphic']['x_extend'][1] -
                                            self.config['graphic']['x_extend'][0]) / 100
        self.config['graphic']['y_unit'] = (self.config['graphic']['y_extend'][1] -
                                            self.config['graphic']['y_extend'][0]) / 100

        self.ax.add_patch(
            patches.Rectangle(
                (self.config['graphic']['x_extend'][0], self.config['graphic']['y_extend'][0]),
                self.config['graphic']['x_unit'] * 25, self.config['graphic']['y_unit'] * 100,
                color="white"
            ))
        self.ax.axhline(self.config['graphic']['y_extend'][0] + self.config['graphic']['y_unit'] * 56,
                        0.02, 0.20, lw=1.5, color="black")
        self.ax.axhline(self.config['graphic']['y_extend'][0] + self.config['graphic']['y_unit'] * 72,
                        0.02, 0.20, lw=1.5, color="black")

    def add_simulation_legend(self):
        self.config['graphic']['x_unit'] = (self.config['graphic']['x_extend'][1] -
                                            self.config['graphic']['x_extend'][0]) / 100
        self.config['graphic']['y_unit'] = (self.config['graphic']['y_extend'][1] -
                                            self.config['graphic']['y_extend'][0]) / 100
        self.ax.add_patch(
            patches.Rectangle(
                (self.config['graphic']['x_extend'][0], self.config['graphic']['y_extend'][0]),
                self.config['graphic']['x_unit'] * 55,
                self.config['graphic']['y_unit'] * 30,
                color="white"
            ))
        # The legend looks a bit different, as it has to show both the inferred areas and the ground truth
        self.ax.annotate("INFERRED", (
            self.config['graphic']['x_extend'][0] + self.config['graphic']['x_unit'] * 3,
            self.config['graphic']['y_extend'][0] + self.config['graphic']['y_unit'] * 23),
                         fontsize=20)
        self.ax.annotate("GROUND TRUTH", (
            self.config['graphic']['x_extend'][0] + self.config['graphic']['x_unit'] * 38.5,
            self.config['graphic']['y_extend'][0] + self.config['graphic']['y_unit'] * 23),
                         fontsize=20)
        self.ax.axvline(self.config['graphic']['x_extend'][0] + self.config['graphic']['x_unit'] * 37,
                        0.05, 0.18, lw=2, color="black")

    def add_secondary_legend(self):
        # for Olga: should be defined in the config
        # reduces to single function add_legend, with parameter "simulation"
        # for Sa map
        if self.config['input']['experiment'] == "sa":
            self.add_sa_legend()

        # for Balkan map
        if self.config['input']['experiment'] == "balkan":
            self.add_balkan_legend()

        # for simulated data
        if self.is_simulation:
            self.add_simulation_legend()

    def add_overview_map(self):
        # All hard-coded parameters (width, height, lower_left, ... could go to the config file.
        # Olga: all the input parameters should be removed later (from inset_axes)
        axins = inset_axes(self.ax, width=self.config['overview']['width'],
                           height=self.config['overview']['height'],
                           bbox_to_anchor=self.map_parameters['overview_position'],
                           loc=self.config['overview']['location'],
                           bbox_transform=self.ax.transAxes)
        axins.tick_params(labelleft=False, labelbottom=False, length=0)

        # Map extend of the overview map
        # x_extend_overview and y_extend_overview --> to config
        axins.set_xlim(self.config['graphic']['x_extend_overview'])
        axins.set_ylim(self.config['graphic']['y_extend_overview'])

        # Again, this function needs map data to display in the overview map.
        self.add_background_map(axins)

        # add overview to the map
        axins.scatter(*self.locations.T, s=self.config['graphic']['size'] / 2, c="darkgrey", alpha=1, linewidth=0)

        # adds a bounding box around the overview map
        bbox_width = self.config['graphic']['x_extend'][1] - self.config['graphic']['x_extend'][0]
        bbox_height = self.config['graphic']['y_extend'][1] - self.config['graphic']['y_extend'][0]
        bbox = mpl.patches.Rectangle((self.config['graphic']['x_extend'][0], self.config['graphic']['y_extend'][0]),
                                     bbox_width, bbox_height, ec='k', fill=False,
                                     linestyle='-')
        axins.add_patch(bbox)

    # Helper function
    @staticmethod
    def scientific(x):
        print(x)
        b = int(np.log10(x))
        a = x / 10 ** b
        return '%.2f \cdot 10^{%i}' % (a, b)

    def add_likelihood_legend(self):
        # Legend for area labels
        self.area_labels = ["      log-likelihood per area"]

        lh_per_area = np.array(list(self.results['likelihood_single_areas'].values())).astype(float)
        to_rank = np.mean(lh_per_area, axis=1)
        p = to_rank[np.argsort(-to_rank)]

        for i, lh in enumerate(p):
            self.area_labels.append(f'$Z_{i + 1}: \, \;\;\; {int(lh)}$')

    ##############################################################
    # Additional things for plot_posterior_map
    # (likelihood info box, background map, subset related stuff)
    ##############################################################
    # Add background map
    def add_background_map(self, ax):
        # If yes, the user needs to define a valid spatial coordinate reference system(proj4)
        # and provide background map data
        if self.config['input']['proj4'] is None and self.config['input']['geojson_map'] is None:
            raise Exception('If you want to use a map provide a geojson and a crs')

        # Adds the geojson map provided by user as background map
        self.world = gpd.read_file(self.config['input']['geojson_map'])
        self.world = self.world.to_crs(self.config['input']['proj4'])
        self.world.plot(ax=ax, color='w', edgecolor='black', zorder=-100000)

    # Add rivers
    def add_rivers(self, ax):
        # The user can also provide river data. Looks good on a map :)
        if self.config['input']['geojson_river'] is not None:
            self.rivers = gpd.read_file(self.config['input']['geojson_river'])
            self.rivers = self.rivers.to_crs(self.config['input']['proj4'])
            self.rivers.plot(ax=ax, color=None, edgecolor="skyblue", zorder=-10000)

    # Add likelihood info box
    def add_likelihood_info(self):
        extra = patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        self.leg_zones.append(extra)

    # Load subset data
    # Helper function for add_subset
    def plot_subset(self):
        # Get sites in subset
        is_in_subset = [x == 1 for x in self.sites['subset']]
        sites_all = deepcopy(self.sites)
        # Get the relevant information for all sites in the subset (ids, names, ..)
        for key in self.sites.keys():
            if type(self.sites[key]) == list:
                self.sites[key] = list(np.array(self.sites[key])[is_in_subset])
            else:
                self.sites[key] = self.sites[key][is_in_subset, :]
        self.locations = self.sites['locations']

        # plot all points not in the subset in light grey
        not_in_subset = np.logical_not(is_in_subset)
        other_locations = sites_all['locations'][not_in_subset]
        self.ax.scatter(*other_locations.T, s=self.config['graphic']['size'], c="gainsboro", alpha=1, linewidth=0)

        # Add a visual bounding box to the map to show the location of the subset on the map
        x_coords, y_coords = self.locations.T
        offset = 100
        x_min, x_max = min(x_coords) - offset, max(x_coords) + offset
        y_min, y_max = min(y_coords) - offset, max(y_coords) + offset
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        bbox = mpl.patches.Rectangle((x_min, y_min), bbox_width, bbox_height, ec='grey', fill=False,
                                     lw=1.5, linestyle='-.')
        self.ax.add_patch(bbox)
        # Adds a small label that reads "Subset"
        self.ax.text(x_max, y_max + 200, 'Subset', fontsize=18, color='#000000')

    # Check all the previous additional functions
    def visualize_additional_map_elements(self, lh_single_zones):
        # Does the plot have a background map?
        # Could go to extra function (add_background_map), which is only called if relevant
        if self.config['graphic']['bg_map']:
            self.add_background_map(self.ax)
            self.add_rivers(self.ax)

    def return_correspondence_table(self, fname, ncol=3):
        """ Which language belongs to which number? This table will tell you more
        Args:
            sites(dict): dict of all languages
            fname(str): name of the figure
            sites_in_zone(list): list of sites per zone
            ncol(int); number of columns in the output table
        """
        fig, ax = plt.subplots()

        sites_id = []
        sites_names = []
        s = [j for sub in self.all_labels for j in sub]

        for i in range(len(self.sites['id'])):
            if i in s:
                sites_id.append(self.sites['id'][i])
                sites_names.append(self.sites['names'][i])

        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        n_col = ncol
        n_row = math.ceil(len(sites_names) / n_col)

        l = [[] for _ in range(n_row)]

        for i in range(len(sites_id)):
            col = i % n_row
            nr = str(sites_id[i] + 1)
            l[col].append(nr)
            l[col].append(sites_names[i])

        # Fill up empty cells
        for i in range(len(l)):
            if len(l[i]) != n_col * 2:
                fill_up_nr = n_col * 2 - len(l[i])
                for f in range(fill_up_nr):
                    l[i].append("")

        widths = [0.05, 0.3] * int(((len(l[0])) / 2))

        table = ax.table(cellText=l, loc='center', cellLoc="left", colWidths=widths)
        table.set_fontsize(40)

        for key, cell in table.get_celld().items():
            cell.set_linewidth(0)
        fig.tight_layout()
        fig.savefig(f"{self.path_plots + '/correspondence'}.{self.map_parameters['save_format']}",
                    bbox_inches='tight', dpi=400, format=self.map_parameters['save_format'])

    ##############################################################
    # This is the plot_posterior_map function from plotting_old
    ##############################################################
    # for Olga: all parameters should be passed from the new map config file
    def posterior_map(self,
                      post_freq_legend, burn_in=0.2,
                      post_freq=0.8,
                      plot_area_stats=False, flamingo=False, simulated_family=False,
                      label_languages=False, add_overview=False,
                      plot_families=False,
                      return_correspondence=False,  # for Olga: This creates a separate table of all languages which are in an area, should probably go into a subplolt?
                      fname='mst_posterior'):

        """ This function creates a scatter plot of all sites in the posterior distribution. The color of a site reflects
            its frequency in the posterior

            Args:
                mcmc_res (dict): the MCMC samples neatly collected in a dict;
                Olga: not needed as an input parameter, because it's already in the init parameters of the parent class Plot
                sites (dict): a dictionary containing the location tuples (x,y) and the id of each site
                Olga: not needed as an input parameter, because it's already in the init parameters of the parent class Plot

                post_freq_legend (list): threshold values for lines
                                        e.g. [0.7, 0.5, 0.3] will display three different line categories:
                                        - a thick line for edges which are in more than 70% of the posterior
                                        - a medium-thick line for edges between 50% and 70% in the posterior
                                        - and a thin line for edges between 30% and 50% in the posterior
                burn_in (float): Fraction of samples, which are discarded as burn-in
                x_extend (tuple): (min, max)-extend of the map in x-direction (longitude) --> Olga: move to config: DONE
                y_extend (tuple): (min, max)-extend of the map in y-direction (latitude) --> and move to config: DONE
                simulated_data(bool): are the plots for real-world or simulated data?
                Olga: not needed as an input parameter, because it's already in the init parameters of the parent class Plot
                experiment(str): either "sa" or "balkan", will load different plotting parameters. Olga: Should go to plotting
                                 config file instead: DONE
                bg_map (bool: Plot a background map? --> Olga: to config: DONE
                geojson_map(str): File path to geoJSON background map --> Olga: to config: DONE
                proj4(str): Coordinate reference system of the language data. --> Olga: Should go to config: DONE
                or could be passed as meta data to the sites
                geo_json_river(str): File path to river data. --> Olga: should go to config: DONE
                subset(boolean): Is there a subset in the data, which should be displayed differently?
                                 Only relevant for one experiment. --> Olga Should go to config: DONE
                plot_area_stats(bool): Add box containing information about the likelihood of single areas to the plot?
                flamingo(bool): Sort of a joke. Does one area have the shape of a flamingo. If yes use flamingo colors for plotting.
                simulated_family(bool): Only for simulated data. Are families also simulated?
                size(float): Size of the dots (languages) in the plot --> Olga: move to config: DONE
                size_line(float): Line thickness. Gives in combination with post_freq_lines the line thickness of the edges in an area
                                  Olga -> should go to config: DONE
                label_languages(bool): Label the languages in areas?
                add_overview(bool): Add an overview map?
                x_extend_overview(tuple): min, max)-extend of the overview map in x-direction (longitude) --> Olga: config: DONE
                y_extend_overview(tuple): min, max)-extend of the overview map in y-direction (latitude) --> Olga: config: DONE
                plot_families(np.array): a boolean assignment of sites to families
                    shape(n_families, n_sites)
                family_alpha_shape(float): controls how far languages of the same family have to be apart to be grouped
                                           into a single alpha shape (for display only)  --> Olga: config: DONE
                fname (str): a path of the output file.
                return_correspondence(bool): return the labels of all languages which are shown in the map
                                            --> Olga: I think this can be solved differently, with a separate function: TODO

            """
        print('Plotting map...')

        # Is the function used for simulated data or real-world data? Both require different plotting parameters.
        # if for real world-data: South America or Balkans?
        # for Olga: this should be defined in the config
        #self.get_map_parameters()

        # Initialize the plot
        # Needed in every plot

        ##############################################################
        # Main initial function
        ##############################################################
        self.initialize_map()

        # Get the areas from the samples
        # Needed in every plot
        #areas = self.results['areas']

        # This computes the Delaunay triangulation of the sites
        # Needed in every plot

        # Olga: net is already loaded in Plot; net = self.network
        # net = compute_network(sites)

        # Plots all languages on the map
        # Needed in every plot
        # Could go together with all the other stuff that's always done to a separate function
        # self.cmap, _ = self.get_cmap(0.5, name='YlOrRd', lower_ts=1.2)
        # self.ax.scatter(*self.locations.T, s=self.config['graphic']['size'], c=[self.cmap(0)], alpha=1, linewidth=0)

        ##############################################################
        # Additional check
        ##############################################################
        self.visualize_additional_map_elements(plot_area_stats)

        ##############################################################
        # Visualization
        ##############################################################
        # This iterates over all areas in the posterior and plots each with a different color (minimum spanning tree)
        self.visualize_areas(flamingo, simulated_family, post_freq, burn_in, label_languages, plot_area_stats)

        ##############################################################
        # Legend
        ##############################################################
        # Add a small legend displaying what the line thickness corresponds to.
        self.add_legend_lines(post_freq_legend)

        # Depending on the background (sa, balkan, simulated), we want to place additional legend entries
        # at different positions in the map in order not to block map content and best use the available space.
        # This should rather go to the config file.
        # Unfortunately, positions have to be provided in map units, which makes things a bit opaque.
        # Once in the config, the functions below can go.
        # this could be called: add_family_legend
        self.add_secondary_legend()

        # This adds an overview map to the main map
        # Could go into a separate function
        if add_overview:
            # Olga: this needs testing (not sure if 'axins' should be global variable or not)
            self.add_overview_map()

        ##############################################################
        # Families
        ##############################################################
        # If families and family names are provided, this adds an overlay color for all language families in the map
        # including a legend entry.
        # Should go to a separate function
        if plot_families:
            self.color_families(self.family_names['external'],
                                self.config['graphic']['family_colors'],
                                families=self.families)
        # Again this adds alpha shapes for families for simulated data.
        # I think the reason why this was coded separately, is that many of the parameters
        # change compared to real world-data
        # Should probably be handled in the config file instead
        # and should be merged with adding families as seen above
        # if simulated_family:
        #     self.color_families(self.families,
        #                         self.config['graphic']['true_family_colors'])


        ##############################################################
        # The following rest of the code is not rewritten yet
        ##############################################################
        # for simulated data: add a legend entry in the shape of a little polygon for ground truth
        if self.is_simulation:
            class TrueZone(object):
                pass

            class TrueZoneHandler(object):
                def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                    x0, y0 = handlebox.xdescent + 10, handlebox.ydescent + 10

                    patch = patches.Polygon([[x0, y0],
                                             [x0 + 40, y0 + 20],
                                             [x0 + 60, y0 - 10],
                                             [x0 + 50, y0 - 20],
                                             [x0 + 30, y0 - 20]],
                                            ec='black', lw=1, ls='-', alpha=1, fill=False,
                                            joinstyle="round", capstyle="butt")
                    handlebox.add_artist(patch)
                    return patch

            # if families are simulated too add a little colored polygon for ground truth families
            if simulated_family:
                self.map_parameters['poly_legend_position'] = \
                    (self.map_parameters['poly_legend_position'][0],
                     self.map_parameters['poly_legend_position'][1] + 0.1)

            # define legend
            legend_true_zones = self.ax.legend([TrueZone()], ['simulated area\n(bounding polygon)'],
                                          handler_map={TrueZone: TrueZoneHandler()},
                                          bbox_to_anchor=self.map_parameters['poly_legend_position'],
                                          title_fontsize=16,
                                          loc='upper left',
                                          frameon=True,
                                          edgecolor='#ffffff',
                                          handletextpad=4,
                                          fontsize=18,
                                          ncol=1,
                                          columnspacing=1)

            self.ax.add_artist(legend_true_zones)

        # styling the axes, might be hardcoded

        self.style_axes(self.ax, self.locations, show=False, x_extend=self.config['graphic']['x_extend'], y_extend=self.config['graphic']['y_extend'])

        # Save the plot

        self.fig.savefig(f"{self.path_plots + fname}.{self.map_parameters['save_format']}",
                         bbox_inches='tight', dpi=400, format=self.map_parameters['save_format'])
        # Should the labels displayed in the map be returned? These are later added as a separate legend (
        # outside this hell of a function)
        plt.close(self.fig)
        if return_correspondence and label_languages:
            self.return_correspondence_table(fname=fname)

    # From general_plot.py
    ####################################
    # Probability simplex, grid plot
    ####################################
    @staticmethod
    def get_corner_points(n, offset=0.5 * np.pi):
        """Generate corner points of a equal sided ´n-eck´."""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + offset
        return np.array([np.cos(angles), np.sin(angles)]).T

    @staticmethod
    def fill_outside(polygon, color, ax=None):
        """Fill the area outside the given polygon with ´color´.
        Args:
            polygon (np.array): The polygon corners in a numpy array.
                shape: (n_corners, 2)
            ax (plt.Axis): The pyplot axis.
            color (str or tuple): The fill color.
        """
        if ax is None:
            ax = plt.gca()

        n_corners = polygon.shape[0]
        i_left = np.argmin(polygon[:, 0])
        i_right = np.argmax(polygon[:, 0])

        # Find corners of bottom face
        i = i_left
        bot_x = [polygon[i, 0]]
        bot_y = [polygon[i, 1]]
        while i % n_corners != i_right:
            i += 1
            bot_x.append(polygon[i, 0])
            bot_y.append(polygon[i, 1])

        # Find corners of top face
        i = i_left
        top_x = [polygon[i, 0]]
        top_y = [polygon[i, 1]]
        while i % n_corners != i_right:
            i -= 1
            top_x.append(polygon[i, 0])
            top_y.append(polygon[i, 1])

        ymin, ymax = ax.get_ylim()
        plt.fill_between(bot_x, ymin, bot_y, color=color)
        plt.fill_between(top_x, ymax, top_y, color=color)

    # Transform weights into needed format
    def transform_weights(self, feature, b_in, gt=False):

        if not gt:
            universal_array = []
            contact_array = []
            inheritance_array = []
            sample_dict = self.results['weights']
            for key in sample_dict:
                split_key = key.split("_")
                if 'w' == split_key[0]:
                    if 'universal' == split_key[1] and str(feature) == split_key[2]:
                        universal_array = sample_dict[key][b_in:]
                    elif 'contact' == split_key[1] and str(feature) == split_key[2]:
                        contact_array = sample_dict[key][b_in:]
                    elif 'inheritance' == split_key[1] and str(feature) == split_key[2]:
                        inheritance_array = sample_dict[key][b_in:]

            sample = np.column_stack([universal_array, contact_array, inheritance_array]).astype(np.float)
            return sample

        else:

            true_universal = []
            true_contact = []
            true_inheritance = []

            true_dict = self.results['true_weights']

            for key in true_dict:
                split_key = key.split("_")
                if 'w' == split_key[0]:
                    if 'universal' == split_key[1] and str(feature) == split_key[2]:
                        true_universal = true_dict[key][b_in:]
                    elif 'contact' == split_key[1] and str(feature) == split_key[2]:
                        true_contact = true_dict[key][b_in:]
                    elif 'inheritance' == split_key[1] and str(feature) == split_key[2]:
                        true_inheritance = true_dict[key][b_in:]
            ground_truth = np.array([true_universal, true_contact, true_inheritance]).astype(np.float)
            return ground_truth

    def transform_probability_vectors(self, feature, parameter, b_in, gt=False):

        if not gt:
            if "alpha" in parameter:
                sample_dict = self.results['alpha']
            elif "beta" in parameter:
                sample_dict = self.results['beta']
            elif "gamma" in parameter:
                sample_dict = self.results['gamma']
            else:
                raise ValueError("parameter must be alpha, beta or gamma")
            p_dict = {}
            states = []

            for key in sample_dict:

                if str(feature + '_') in key and parameter in key:
                    state = str(key).rsplit('_', 1)[1]
                    p_dict[state] = sample_dict[key][b_in:]
                    states.append(state)

            sample = np.column_stack([p_dict[s] for s in p_dict]).astype(np.float)
            return sample, states
        else:
            if "alpha" in parameter:
                true_dict = self.results['true_alpha']
            elif "beta" in parameter:
                true_dict = self.results['true_beta']
            elif "gamma" in parameter:
                true_dict = self.results['true_gamma']
            else:
                raise ValueError("parameter must alpha, beta or gamma")

            true_prob = []

            for key in true_dict:
                if str(feature + '_') in key and parameter in key:
                    true_prob.append(true_dict[key])

            return np.array(true_prob).astype(np.float)

    # Sort weights by median contact
    def get_parameters(self, b_in, parameter="weights"):

        par = {}
        true_par = {}
        states = {}

        # get samples
        for i in self.results['feature_names']:
            if parameter == "weights":
                p = self.transform_weights(feature=i, b_in=b_in)
                par[i] = p

            elif "alpha" in parameter or "beta" in parameter or "gamma" in parameter:
                p, state = self.transform_probability_vectors(feature=i, parameter=parameter, b_in=b_in)

                par[i] = p
                states[i] = state

        # get ground truth
        if self.is_simulation:
            for i in self.results['feature_names']:
                if parameter == "weights":
                    true_p = self.transform_weights(feature=i, b_in=b_in, gt=True)
                    true_par[i] = true_p

                elif "alpha" in parameter or "beta" in parameter or "gamma" in parameter:
                    true_p = self.transform_probability_vectors(feature=i, parameter=parameter, b_in=b_in)
                    true_par[i] = true_p
        else:
            true_par = None
        return par, true_par, states

    def sort_by_weights(self, w):
        sort_by = {}
        for i in self.results['feature_names']:
            sort_by[i] = median(w[i][:, 1])
        ordering = sorted(sort_by, key=sort_by.get, reverse=True)
        return ordering

    # Probability simplex (for one feature)
    @staticmethod
    def plot_weights(samples, feature, true_weights=False, labels=None, ax=None, mean_weights=False):
        """Plot a set of weight vectors in a 2D representation of the probability simplex.

        Args:
            samples (np.array): Sampled weight vectors to plot.
            feature (str): Name of the feature for which weights are being plotted
            true_weights (np.array): true weight vectors (only for simulated data)
            labels (list[str]): Labels for each weight dimension.
            ax (plt.Axis): The pyplot axis.
            mean_weights (bool): Plot the mean of the weights?
        """

        if ax is None:
            ax = plt.gca()
        n_samples, n_weights = samples.shape

        # Compute corners
        corners = Plot.get_corner_points(n_weights)
        # Bounding box
        xmin, ymin = np.min(corners, axis=0)
        xmax, ymax = np.max(corners, axis=0)

        # Project the samples
        samples_projected = samples.dot(corners)

        # color map
        cmap = sns.cubehelix_palette(light=1, start=.5, rot=-.75, as_cmap=True)

        # Density and scatter plot
        plt.title(str(feature), loc='left', fontdict={'fontweight': 'bold', 'fontsize': 16})
        x = samples_projected.T[0]
        y = samples_projected.T[1]
        sns.kdeplot(x, y, shade=True, shade_lowest=True, cut=30, n_levels=100,
                    clip=([xmin, xmax], [ymin, ymax]), cmap=cmap)
        plt.scatter(x, y, color='k', lw=0, s=1, alpha=0.2)

        # Draw simplex and crop outside
        plt.fill(*corners.T, edgecolor='k', fill=False)
        Plot.fill_outside(corners, color='w', ax=ax)

        if true_weights:
            true_projected = true_weights.dot(corners)
            plt.scatter(*true_projected.T, color="#ed1696", lw=0, s=100, marker="*")

        if mean_weights:
            mean_projected = np.mean(samples, axis=0).dot(corners)
            plt.scatter(*mean_projected.T, color="#ed1696", lw=0, s=100, marker="o")

        if labels is not None:
            for xy, label in zip(corners, labels):
                xy *= 1.08  # Stretch, s.t. labels don't overlap with corners
                plt.text(*xy, label, ha='center', va='center', fontdict={'fontsize': 16})

        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout(0)
        plt.plot()

    @staticmethod
    def plot_probability_vectors(samples, feature, true_p=None, labels=None, ax=None, title=False):
        """Plot a set of weight vectors in a 2D representation of the probability simplex.

        Args:
            samples (np.array): Sampled weight vectors to plot.
            feature (str): Name of the feature for which weights are being plotted
            true_p (np.array): true probability vectors (only for simulated data)
            labels (list[str]): Labels for each weight dimension.
            title (bool): plot title
            ax (plt.Axis): The pyplot axis.
        """

        if ax is None:
            ax = plt.gca()
        n_samples, n_p = samples.shape
        # color map
        cmap = sns.cubehelix_palette(light=1, start=.5, rot=-.75, as_cmap=True)
        if n_p == 2:
            # plt.title(str(feature), loc='center', fontdict={'fontweight': 'bold', 'fontsize': 20})
            x = samples.T[0]
            y = np.zeros(n_samples)
            sns.distplot(x, rug=True, hist=False, kde_kws={"shade": True, "lw": 0, "clip": (0, 1)}, color="g",
                         rug_kws={"color": "k", "alpha": 0.01, "height": 0.03})
            # sns.kdeplot(x, shade=True, color="r", clip=(0, 1))
            # plt.scatter(x, y, color='k', lw=0, s=1, alpha=0.2)
            # plt.axhline(y=0, color='k', linestyle='-', lw=0.5, xmin=0, xmax=1)

            ax.axes.get_yaxis().set_visible(False)
            # ax.annotate('', xy=(0, -0.5), xytext=(1, -0.1),
            #            arrowprops=dict(arrowstyle="-", color='b'))

            if true_p is not None:
                plt.scatter(true_p[0], 0, color="#ed1696", lw=0, s=100, marker="*")

            if labels is not None:
                for x, label in enumerate(labels):
                    plt.text(x, -0.5, label, ha='center', va='top', fontdict={'fontsize': 16})
            if title:
                plt.title(str(feature), loc='left', fontdict={'fontweight': 'bold', 'fontsize': 16})

            plt.plot([0, 1], [0, 0], c="k", lw=0.5)
            plt.xlim(0, 1)
            plt.axis('off')
            # plt.tight_layout(0)

        elif n_p > 2:
            # Compute corners
            corners = Plot.get_corner_points(n_p)
            # Bounding box
            xmin, ymin = np.min(corners, axis=0)
            xmax, ymax = np.max(corners, axis=0)

            # Project the samples
            samples_projected = samples.dot(corners)

            # Density and scatter plot
            if title:
                plt.title(str(feature), loc='center', fontdict={'fontweight': 'bold', 'fontsize': 20})
            x = samples_projected.T[0]
            y = samples_projected.T[1]
            sns.kdeplot(x, y, shade=True, shade_lowest=True, cut=30, n_levels=100,
                        clip=([xmin, xmax], [ymin, ymax]), cmap=cmap)
            plt.scatter(x, y, color='k', lw=0, s=1, alpha=0.05)

            # Draw simplex and crop outside

            plt.fill(*corners.T, edgecolor='k', fill=False)
            Plot.fill_outside(corners, color='w', ax=ax)

            if true_p is not None:
                true_projected = true_p.dot(corners)
                plt.scatter(*true_projected.T, color="#ed1696", lw=0, s=100, marker="*")

            if labels is not None:
                for xy, label in zip(corners, labels):
                    xy *= 1.08  # Stretch, s.t. labels don't overlap with corners
                    plt.text(*xy, label, ha='center', va='center', fontdict={'fontsize': 16})

            plt.axis('equal')
            plt.axis('off')
            plt.tight_layout(0)

        plt.plot()

    # Find number of features
    # def find_num_features(self):
    #     for key in self.results['weights']:
    #         num = re.search('[0-9]+', key)
    #         if num:
    #             if int(num.group(0)) > self.number_features:
    #                 self.number_features = int(num.group(0))

    # Make a grid with all features (sorted by median contact)
    # By now we assume number of features to be 35; later this should be rewritten for any number of features
    # using find_num_features
    def plot_weights_grid(self, fname, labels=None, burn_in=0.4):

        print('Plotting weights...')
        burn_in = int(len(self.results['posterior']) * burn_in)

        weights, true_weights, _ = self.get_parameters(parameter="weights", b_in=burn_in)
        ordering = self.sort_by_weights(weights)

        n_plots = 10
        n_col = 5
        n_row = math.ceil(n_plots / n_col)

        fig, axs = plt.subplots(n_row, n_col, figsize=(15, 5))
        position = 1

        features = ordering[:n_plots]

        for f in features:
            plt.subplot(n_row, n_col, position)
            if self.is_simulation:
                self.plot_weights(weights[f], feature=f, true_weights=true_weights[f], labels=labels)
            else:
                self.plot_weights(weights[f], feature=f, labels=labels, mean_weights=True)
            print(position, "of", n_plots, "plots finished")
            position += 1

        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        fig.savefig(self.path_plots + fname + ".pdf", dpi=400, format="pdf")

    # This is not changed yet
    def plot_probability_grid(self, fname, p_name="gamma_a1", burn_in=0.4, title=False):
        """Creates a ridge plot for parameters with two states

       Args:
           p_name (str): name of parameter vector (either alpha, beta_* or gamma_*)
           burn_in (float): fraction of the samples which should be discarded as burn_in
       """
        print('Plotting probabilities...')
        burn_in = int(len(self.results['posterior']) * burn_in)

        weights, true_weights, _ = self.get_parameters(parameter="weights", b_in=burn_in)

        ordering = self.sort_by_weights(weights)

        p, true_p, states = self.get_parameters(parameter=p_name, b_in=burn_in)

        n_plots = 10
        n_col = 5
        n_row = math.ceil(n_plots / n_col)

        fig, axs = plt.subplots(n_row, n_col, figsize=(15, 5))

        position = 1

        features = ordering[:n_plots]

        for f in features:
            plt.subplot(n_row, n_col, position)

            if self.is_simulation:
                self.plot_probability_vectors(p[f], feature=f, true_p=true_p[f], labels=states[f], title=title)
            else:
                self.plot_probability_vectors(p[f], feature=f, labels=states[f], title=title)
            print(position, "of", n_plots, "plots finished")
            position += 1

        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        fig.savefig(self.path_plots + fname + ".pdf", dpi=400, format="pdf")

    def plot_dic(self, models, burn_in, true_model=None):
        """This function plots the dics. What did you think?
        Args:

            models(dict): A dict of different models for which the DIC is evaluated
            burn_in (float): Fraction of samples discarded as burn-in, when computing the DIC
            true_model (str): id of true model
        """
        print('Plotting DIC...')

        fig, ax = plt.subplots(figsize=(20, 10))
        x = list(models.keys())
        y = []

        # Compute the DIC for each model
        for m in x:
            lh = models[m]['likelihood']
            dic = compute_dic(lh, burn_in)
            y.append(dic)

        # Limits
        ax.plot(x, y, lw=1, color='#000000', label='DIC')
        y_min, y_max = min(y), max(y)
        y_range = y_max - y_min

        x_min = 0
        x_max = len(x) - 1

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min - y_range * 0.1, y_max + y_range * 0.1])

        # Labels and ticks
        ax.set_xlabel('Number of areas', fontsize=10, fontweight='bold')
        ax.set_ylabel('DIC', fontsize=10, fontweight='bold')

        labels = list(range(1, len(x) + 1))
        ax.set_xticklabels(labels, fontsize=8)

        y_ticks = np.linspace(y_min, y_max, 6)
        ax.set_yticks(y_ticks)
        yticklabels = [f'{y_tick:.0f}' for y_tick in y_ticks]
        ax.set_yticklabels(yticklabels, fontsize=10)
        pos_true_model = [idx for idx, val in enumerate(x) if val == true_model][0]

        if true_model is not None:
            color_burn_in = 'grey'
            ax.axvline(x=pos_true_model, lw=1, color=color_burn_in, linestyle='--')
            ypos_label = y_min + y_range * 0.15
            plt.text(pos_true_model - 0.05, ypos_label, 'Simulated areas', rotation=90, size=10,
                     color=color_burn_in)

        fig.savefig(self.path_plots + '/dic.pdf', dpi=400, format="pdf", bbox_inches='tight')

    def plot_trace(self, burn_in=0.2, parameter='likelihood', fname="trace", ylim=None, ground_truth=False,
                   show_every_k_sample=1):
        """
        Function to plot the trace of a parameter
        Args:
            burn_in (float): First n% of samples are burn-in
            parameter (str): Parameter for which to plot the trace
            fname (str): a path followed by a the name of the file
            ylim (tuple): limits on the y-axis
            ground_truth(bool): show ground truth
            show_every_k_sample (int): show every 1, 1+k,1+2k sample and skip over the rest
        """
        # For Olga: change all three parameters to config file entries
        plt.rcParams["axes.linewidth"] = 1
        fig, ax = plt.subplots(figsize=(10, 8))

        if parameter == 'recall_and_precision':
            y = self.results['recall'][::show_every_k_sample]
            y2 = self.results['precision'][::show_every_k_sample]
            x = self.results['sample_id'][::show_every_k_sample]
            ax.plot(x, y, lw=0.5, color='#e41a1c', label="recall")
            ax.plot(x, y2, lw=0.5, color='dodgerblue', label="precision")

        else:
            try:
                y = self.results[parameter][::show_every_k_sample]

            except KeyError:
                raise ValueError("Cannot compute trace. " + parameter + " is not a valid parameter.")

            x = self.results['sample_id'][::show_every_k_sample]
            ax.plot(x, y, lw=0.5, color='#e41a1c', label=parameter)

        if ylim is None:
            y_min, y_max = min(y), max(y)

        else:
            y_min, y_max = ylim

        y_range = y_max - y_min
        x_min, x_max = 0, x[-1]

        if ground_truth:
            ground_truth_parameter = 'true_' + parameter
            y_gt = self.results[ground_truth_parameter]
            ax.axhline(y=y_gt, xmin=x[0], xmax=x[-1], lw=1, color='#fdbf6f',
                       linestyle='-',
                       label='ground truth')
            y_min, y_max = [min(y_min, y_gt), max(y_max, y_gt)]

        # Show burn-in in plot
        end_bi = math.ceil(x[-1] * burn_in)
        end_bi_label = math.ceil(x[-1] * (burn_in - 0.03))

        color_burn_in = 'grey'
        ax.axvline(x=end_bi, lw=1, color=color_burn_in, linestyle='--')
        ypos_label = y_min + y_range * 0.15
        plt.text(end_bi_label, ypos_label, 'Burn-in', rotation=90, size=10, color=color_burn_in)

        # Ticks and labels
        n_ticks = 6 if int(burn_in * 100) % 20 == 0 else 12
        x_ticks = np.linspace(x_min, x_max, n_ticks)
        x_ticks = [round(t, -5) for t in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{x_tick:.0f}' for x_tick in x_ticks], fontsize=6)

        f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
        g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
        plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(g))
        ax.set_xlabel('Iteration', fontsize=8, fontweight='bold')

        y_ticks = np.linspace(y_min, y_max, 5)
        ax.set_yticks(y_ticks)
        y_ticklabels = [f'{y_tick:.1f}' for y_tick in y_ticks]
        ax.set_yticklabels(y_ticklabels, fontsize=6)

        # Limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min - y_range * 0.1, y_max + y_range * 0.1])

        # Legend
        ax.legend(loc=4, prop={'size': 8}, frameon=False)

        # Save
        fig.savefig(self.path_plots + fname + '.pdf', dpi=400, format="pdf", bbox_inches='tight')
        plt.close(fig)

    def plot_trace_lh_prior(self, burn_in=0.2, fname="trace", show_every_k_sample=1, lh_lim=None, prior_lim=None):
        fig, ax1 = plt.subplots(figsize=(10, 8))

        lh = self.results['likelihood'][::show_every_k_sample]
        prior = self.results['prior'][::show_every_k_sample]
        x = self.results['sample_id'][::show_every_k_sample]

        # Plot lh on axis 1
        ax1.plot(x, lh, lw=0.5, color='#e41a1c', label='likelihood')

        # Plot prior on axis 2
        ax2 = ax1.twinx()
        ax2.plot(x, prior, lw=0.5, color='dodgerblue', label='prior')

        x_min, x_max = 0, x[-1]
        if lh_lim is None:
            lh_min, lh_max = min(lh), max(lh)
        else:
            lh_min, lh_max = lh_lim
        lh_range = lh_max - lh_min

        if prior_lim is None:
            prior_min, prior_max = min(lh), max(lh)
        else:
            prior_min, prior_max = prior_lim
        prior_range = prior_max - prior_min

        # Labels and ticks
        n_ticks = 6 if int(burn_in * 100) % 20 == 0 else 12
        x_ticks = np.linspace(x_min, x_max, n_ticks)
        x_ticks = [round(t, -5) for t in x_ticks]
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([f'{x_tick:.0f}' for x_tick in x_ticks], fontsize=6)

        f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
        g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
        plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(g))
        ax1.set_xlabel('Iteration', fontsize=8, fontweight='bold')

        lh_ticks = np.linspace(lh_min, lh_max, 6)
        ax1.set_yticks(lh_ticks)
        lh_ticklabels = [f'{lh_tick:.0f}' for lh_tick in lh_ticks]
        ax1.set_yticklabels(lh_ticklabels, fontsize=6, color='#e41a1c')
        ax1.set_ylabel('log-likelihood', fontsize=8, fontweight='bold', color='#e41a1c')

        prior_ticks = np.linspace(prior_min, prior_max, 6)
        ax2.set_yticks(prior_ticks)
        prior_ticklabels = [f'{prior_tick:.0f}' for prior_tick in prior_ticks]
        ax2.set_yticklabels(prior_ticklabels, fontsize=8, color='dodgerblue')
        ax2.set_ylabel('prior', fontsize=8, fontweight='bold', color='dodgerblue')

        # Show burn-in in plot
        end_bi = math.ceil(x[-1] * burn_in)
        end_bi_label = math.ceil(x[-1] * (burn_in - 0.03))

        color_burn_in = 'grey'
        ax1.axvline(x=end_bi, lw=1, color=color_burn_in, linestyle='--')
        ypos_label = prior_min + prior_range * 0.15
        plt.text(end_bi_label, ypos_label, 'Burn-in', rotation=90, size=10, color=color_burn_in)

        # Limits
        ax1.set_ylim([lh_min - lh_range * 0.1, lh_max + lh_range * 0.1])
        ax2.set_ylim([prior_min - prior_range * 0.1, prior_max + prior_range * 0.1])

        # Save
        fig.savefig(self.path_plots + fname + '.pdf', dpi=400, format="pdf", bbox_inches='tight')
        plt.close(fig)

    def plot_recall_precision_over_several_models(self, models):

        fig, ax = plt.subplots(figsize=(10, 8))

        recall = []
        precision = []

        # Retrieve recall and precision for every model and plot
        for m in list(models.keys()):
            recall.extend(models[m]['recall'])
            precision.extend(models[m]['precision'])

        x = list(range(len(recall)))
        ax.plot(x, recall, lw=0.5, color='#e41a1c', label='recall')
        ax.plot(x, precision, lw=0.5, color='dodgerblue', label='precision')

        # Limits
        ax.set_ylim(bottom=0)
        x_min, x_max = 0, len(recall)
        y_min, y_max, y_step = 0, 1, 0.2
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max + 0.1])

        # Labels
        n_models = len(list(models.keys()))
        n_ticks = n_models + 1
        x_ticks = np.linspace(x_min, x_max, n_ticks)

        x_ticks_offset = x_ticks[1] / 2
        x_ticks = [x_tick - x_ticks_offset for x_tick in x_ticks if x_tick > 0]
        ax.set_xticks(x_ticks)
        x_ticklabels = [f'{x_ticklabel:.0f} areas' for x_ticklabel in np.linspace(1, n_models, n_models)]
        x_ticklabels[0] = '1 area'
        ax.set_xticklabels(x_ticklabels, fontsize=6)

        minor_locator = AutoMinorLocator(2)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(which='minor', axis='x', color='#000000', linestyle='-')
        ax.set_axisbelow(True)

        y_ticks = np.arange(y_min, y_max + y_step, y_step)
        ax.set_yticks(y_ticks)
        y_ticklabels = [f'{y_tick:.1f}' for y_tick in y_ticks]
        y_ticklabels[0] = '0'
        ax.set_yticklabels(y_ticklabels, fontsize=6)

        ax.legend(loc=4, prop={'size': 6}, frameon=True, framealpha=1, facecolor='#ffffff',
                  edgecolor='#ffffff')

        fig.savefig(self.path_plots + '/recall_precision_over_models.pdf', dpi=400, format="pdf",
                    bbox_inches='tight')
        plt.close(fig)