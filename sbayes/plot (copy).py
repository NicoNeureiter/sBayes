import json
import math
import os
from copy import deepcopy
from itertools import compress
from statistics import median
import argparse
from pathlib import Path

import pandas as pd
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
from sbayes.util import sort_by_similarity


class Plot:
    def __init__(self, simulated_data=False):

        # Flag for simulation
        self.is_simulation = simulated_data

        # Config variables
        self.config = {}
        self.config_file = None

        # Path variables
        self.path_output = None
        self.path_features = None
        self.path_feature_states = None
        self.path_sites = None
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
        self.config_default = "config/plotting"

        self.geo_config = {}
        self.content_config = {}

        self.legend_config = {}
        self.graphic_config = {}
        self.ground_truth_config = {}
        self.output_config = {}

        # Figure parameters
        self.ax = None
        self.fig = None
        self.x_min = self.x_max = self.y_min = self.y_max = None
        self.bbox = None

        self.leg_areas = []
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
        self.path_output = self.config['output_folder']
        if self.is_simulation:
            self.path_sites = self.config['data']['sites']
        else:
            self.path_features = self.config['data']['features']
            self.path_feature_states = self.config['data']['feature_states']

        self.path_plots = self.path_output + '/plots'
        self.path_areas = list(self.config['results']['areas'])
        self.path_stats = list(self.config['results']['stats'])

        if self.is_simulation:
            self.path_ground_truth_areas = self.config['results']['ground_truth_areas']
            self.path_ground_truth_stats = self.config['results']['ground_truth_stats']

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
    # def set_scenario_path(self, current_scenario):
    #     current_run_path = f"{self.path_plots}/n{self.config['input']['run']}_{current_scenario}/"
    #
    #     if not os.path.exists(current_run_path):
    #         os.makedirs(current_run_path)
    #
    #     return current_run_path

    # Read sites, site_names, network
    def read_data(self):

        print('Reading input data...')
        if self.is_simulation:

            self.sites, self.site_names, _ = read_sites(self.path_sites,
                                                        retrieve_family=True, retrieve_subset=True)
            self.families, self.family_names = assign_family(1, self.sites)
        else:
            self.sites, self.site_names, _, _, _, _, self.families, self.family_names, _ = \
                read_features_from_csv(self.path_features, self.path_feature_states)
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
    def read_dictionary(dataframe, search_key, ground_truth=False):
        param_dict = {}
        for column_name in dataframe.columns:
            if column_name.startswith(search_key):
                param_dict[column_name] = dataframe[column_name].to_numpy(dtype=float)
                if ground_truth:
                    assert param_dict[column_name].shape == (1,)
                    param_dict[column_name] = param_dict[column_name][0]

        return param_dict

    # Helper function for read_stats
    # Used for reading: true_posterior, true_likelihood, true_prior,
    # true_weights, true_alpha, true_beta, true_gamma,
    # recall, precision
    @staticmethod
    def read_simulation_stats(txt_path, dataframe):
        if 'ground_truth' in txt_path:
            true_posterior = dataframe['posterior'][0]
            true_likelihood = dataframe['likelihood'][0]
            true_prior = dataframe['prior'][0]

            true_weights = Plot.read_dictionary(dataframe, 'w_', ground_truth=True)
            true_alpha = Plot.read_dictionary(dataframe, 'alpha_', ground_truth=True)
            true_beta = Plot.read_dictionary(dataframe, 'beta_', ground_truth=True)
            true_gamma = Plot.read_dictionary(dataframe, 'gamma_', ground_truth=True)

            recall, precision = None, None

        else:
            recall = dataframe['recall'].to_numpy(dtype=int)
            precision = dataframe['precision'].to_numpy(dtype=int)

            true_weights, true_alpha, true_beta, true_gamma = None, None, None, None
            true_posterior, true_likelihood, true_prior = None, None, None

        return recall, precision, \
            true_posterior, true_likelihood, true_prior, \
            true_weights, true_alpha, true_beta, true_gamma

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

        sample_id, recall, precision = None, None, None
        true_posterior, true_likelihood, true_prior, true_weights = None, None, None, None
        true_alpha, true_beta, true_gamma = None, None, None

        # Load the stats using pandas
        df_stats = pd.read_csv(txt_path, delimiter='\t')

        # Read scalar parameters from the dataframe
        if 'Sample' in df_stats:
            sample_id = df_stats['Sample'].to_numpy(dtype=int)
        posterior = df_stats['posterior'].to_numpy(dtype=float)
        likelihood = df_stats['likelihood'].to_numpy(dtype=float)
        prior = df_stats['prior'].to_numpy(dtype=float)

        # Read multivalued parameters from the dataframe into dicts
        weights = Plot.read_dictionary(df_stats, 'w_')
        alpha = Plot.read_dictionary(df_stats, 'alpha_')
        beta = Plot.read_dictionary(df_stats, 'beta_')
        gamma = Plot.read_dictionary(df_stats, 'gamma_')
        posterior_single_areas = Plot.read_dictionary(df_stats, 'post_')
        likelihood_single_areas = Plot.read_dictionary(df_stats, 'lh_')
        prior_single_areas = Plot.read_dictionary(df_stats, 'prior_')

        # For simulation runs: read statistics based on ground-truth
        if simulation_flag:
            recall, precision, true_posterior, true_likelihood, true_prior, \
                true_weights, true_alpha, true_beta, true_gamma = Plot.read_simulation_stats(txt_path, df_stats)

        # Names of distinct features
        feature_names = []
        for key in weights:
            if 'universal' in key:
                feature_names.append(str(key).rsplit('_', 1)[1])

        # Bind all statistics together into the dictionary self.results
        self.bind_stats(txt_path, sample_id, posterior, likelihood, prior, weights, alpha, beta, gamma,
                        posterior_single_areas, likelihood_single_areas, prior_single_areas, recall, precision,
                        true_posterior, true_likelihood, true_prior, true_weights, true_alpha, true_beta, true_gamma,
                        feature_names)

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
    # Load default config parameters
    def add_config_default(self):

        self.config_default = fix_default_config(self.config_default)

        with open(self.config_default, 'r') as f:
            new_config = json.load(f)

            # If the key already exists
            for key in new_config:
                if key in self.config:
                    self.config[key].update(new_config[key])
                else:
                    self.config[key] = new_config[key]

    def style_axes(self):
        """ Function to style the axes of a plot

        Returns:
            (tuple): Extend of plot.
        """

        try:

            self.x_min, self.x_max = self.geo_config['x_extend']
            self.y_min, self.y_max = self.geo_config['y_extend']

        except KeyError:
            x_min, x_max = np.min(self.locations[:, 0]), np.max(self.locations[:, 0])
            y_min, y_max = np.min(self.locations[:, 1]), np.max(self.locations[:, 1])

            x_offset = (x_max - x_min) * 0.01
            y_offset = (y_max - y_min) * 0.01

            self.x_min = x_min + x_offset
            self.x_max = x_max + x_offset
            self.y_min = y_min - y_offset
            self.y_max = y_max + y_offset

        # setting axis limits
        self.ax.set_xlim([self.x_min, self.x_max])
        self.ax.set_ylim([self.y_min, self.y_max])

        # Removing axis labels
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Bounding Box
        self.bbox = geometry.Polygon([(self.x_min, self.y_min),
                                      (self.x_max, self.y_min),
                                      (self.x_max, self.y_max),
                                      (self.x_min, self.y_max),
                                      (self.x_min, self.y_min)])

    def compute_alpha_shapes(self, sites, alpha_shape):

        """Compute the alpha shape (concave hull) of a set of sites
        Args:
            sites (np.array): subset of sites around which to create the alpha shapes (e.g. family, area, ...)
            alpha_shape (float): parameter controlling the convexity of the alpha shape
        Returns:
            (polygon): the alpha shape"""

        all_sites = self.locations
        points = all_sites[sites[0]]

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

            "alpha value to influence the shape of the convex hull Smaller numbers don't fall inward "
            "as much as larger numbers. Too large, and you lose everything!"

            if circum_r < 1.0 / alpha_shape:
                add_edge(edges, edge_nodes, points, ia, ib)
                add_edge(edges, edge_nodes, points, ib, ic)
                add_edge(edges, edge_nodes, points, ic, ia)

        m = geometry.MultiLineString(edge_nodes)

        triangles = list(polygonize(m))
        polygon = cascaded_union(triangles)

        return polygon

    def add_area_boundary(self, is_in_area, annotation=None, color='#000000'):
        """ Function to add bounding boxes around areas
        Args:
            is_in_area (np.array): Boolean array indicating if a point is in area.
            annotation (string): If passed, area is annotated with this.
            color (string): Color of area.
        Returns:
            leg_area: Legend.
        """

        # use form plotting param
        font_size = 18
        cp_locations = self.locations[is_in_area[0], :]

        leg_area = None
        if cp_locations.shape[0] > 0:  # at least one contact point in area

            alpha_shape = self.compute_alpha_shapes(sites=[is_in_area],
                                                    alpha_shape=self.ground_truth_config['area_alpha_shape'])

            # smooth_shape = alpha_shape.buffer(100, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
            smooth_shape = alpha_shape.buffer(60, resolution=16, cap_style=1, join_style=1, mitre_limit=10.0)
            # smooth_shape = alpha_shape
            patch = PolygonPatch(smooth_shape, ec=color, lw=1, ls='-', alpha=1, fill=False,
                                 zorder=1)
            leg_area = self.ax.add_patch(patch)
        else:
            print('computation of bbox not possible because no contact points')

        # only adding a label (numeric) if annotation turned on and more than one area
        if annotation is not None:
            x_coords, y_coords = cp_locations.T
            x, y = np.mean(x_coords), np.mean(y_coords)
            self.ax.text(x, y, annotation, fontsize=font_size, color=color)

        return leg_area

    def areas_to_graph(self, area):

        # exclude burn-in
        end_bi = math.ceil(len(area) * self.content_config['burn_in'])
        area = area[end_bi:]

        # compute frequency of each point in area
        area = np.asarray(area)
        n_samples = area.shape[0]

        area_freq = np.sum(area, axis=0) / n_samples
        in_graph = area_freq >= self.content_config['min_posterior_frequency']
        locations = self.locations[in_graph]
        n_graph = len(locations)

        # getting indices of points in area
        area_indices = np.argwhere(in_graph)

        if n_graph > 3:
            # computing the delaunay
            delaunay = compute_delaunay(locations)
            graph_connections = gabriel_graph_from_delaunay(delaunay, locations)

        elif n_graph == 3:
            graph_connections = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)

        elif n_graph == 2:
            graph_connections = np.array([[0, 1]], dtype=int)

        else:
            raise ValueError('No points in contact area!')

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
    def get_config_parameters(self):
        self.content_config = self.config['map']['content']
        self.graphic_config = self.config['map']['graphic']
        self.legend_config = self.config['map']['legend']
        self.output_config = self.config['map']['output']
        self.geo_config = self.config['map']['geo']

        if self.is_simulation:
            self.ground_truth_config = self.config['map']['ground_truth']

    # Initialize the map
    def initialize_map(self):
        self.get_config_parameters()

        # for Olga: constrained layout drops a warning. Could you check?
        self.fig, self.ax = plt.subplots(figsize=(self.output_config['fig_width'],
                                                  self.output_config['fig_height']),
                                         constrained_layout=True)
        if self.content_config['subset']:
            self.plot_subset()

        self.ax.scatter(*self.locations.T, s=self.graphic_config['point_size'],
                        c="darkgrey", alpha=1, linewidth=0)

    ##############################################################
    # Visualization functions for plot_posterior_map
    ##############################################################
    def add_color(self, i):
        color = self.graphic_config['area_colors'][i]
        return color

    def add_label(self, is_in_area, current_color):
        # Find all languages in areas
        loc_in_area = self.locations[is_in_area, :]
        labels_in_area = list(compress(self.sites['id'], is_in_area))
        self.all_labels.append(labels_in_area)

        for loc in range(len(loc_in_area)):
            # add a label at a spatial offset of 20000 and 10000. Rather than hard-coding it,
            # this might go into the config.
            x, y = loc_in_area[loc]
            x += self.content_config['label_offset'][0]
            y += self.content_config['label_offset'][1]
            # Same with the font size for annotations. Should probably go to the config.
            anno_opts = dict(xy=(x, y), fontsize=14, color=current_color)
            self.ax.annotate(labels_in_area[loc] + 1, **anno_opts)

    # Bind together the functions above
    def visualize_areas(self):
        self.all_labels = []
        self.area_labels = []

        # If likelihood for single areas are displayed: add legend entries with likelihood information per area
        if self.legend_config['areas']['add_area_stats']:
            self.add_likelihood_legend()
            self.add_likelihood_info()
        else:
            for i, _ in enumerate(self.results['areas']):
                self.area_labels.append(f'$Z_{i + 1}$')

        # Color areas
        for i, area in enumerate(self.results['areas']):
            current_color = self.add_color(i)

            # This function computes a Gabriel graph for all points which are in the posterior with at least p_freq
            in_graph, lines, line_w = self.areas_to_graph(area)

            self.ax.scatter(*self.locations[in_graph].T, s=self.graphic_config['point_size'], c=current_color)

            for li in range(len(lines)):
                self.ax.plot(*lines[li].T, color=current_color, lw=line_w[li] * self.graphic_config['line_width'],
                             alpha=0.6)

            # This adds small lines to the legend (one legend entry per area)
            line_legend = Line2D([0], [0], color=current_color, lw=6, linestyle='-')
            self.leg_areas.append(line_legend)

            # Labels the languages in the areas
            # Should go into a separate function
            if self.content_config['label_languages']:
                self.add_label(in_graph, current_color)

            # Again, this is only relevant for simulated data and should go into a separate function
            # Olga: doesn't seem to be a good decision to move this out to a separate function,
            # because it's called inside of a loop and looks quite short
        if self.is_simulation:
            for i in self.results['true_zones']:
                # Adds a bounding box for the ground truth areas
                # showing if the algorithm has correctly identified them
                self.add_area_boundary(i, color='#000000')

        # add to legend
        legend_areas = self.ax.legend(
            self.leg_areas,
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
            bbox_to_anchor=self.legend_config['areas']['position']
        )
        legend_areas._legend_box.align = "left"
        self.ax.add_artist(legend_areas)

    # TODO: This function should be rewritten in a nicer way; probably split into two functions,
    #  or find a better way of dividing things into simulated and real
    def color_families(self):
        # Initialize empty legend handle
        handles = []
        family_array = self.family_names['external']
        families = self.families

        # Iterate over all family names
        for i, family in enumerate(family_array):
            family_color = self.graphic_config['family_colors'][i]
            family_fill, family_border = family_color, family_color

            # Find all languages belonging to a family
            is_in_family = families[i] == 1
            family_locations = self.locations[is_in_family, :]

            # For simulated data
            if self.is_simulation:
                alpha_shape = self.compute_alpha_shapes(sites=[is_in_family],
                                                        alpha_shape=self.graphic_config['family_alpha_shape'])
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
                self.ax.scatter(*family_locations.T, s=self.graphic_config['point_size'] * 15,
                                c=family_color, alpha=1, linewidth=0, zorder=-i, label=family)

                # For languages with more than three members combine several languages in an alpha shape (a polygon)
                if np.count_nonzero(is_in_family) > 3:
                    alpha_shape = self.compute_alpha_shapes(sites=[is_in_family],
                                                            alpha_shape=self.graphic_config['family_alpha_shape'])

                    # making sure that the alpha shape is not empty
                    if not alpha_shape.is_empty:
                        smooth_shape = alpha_shape.buffer(40000, resolution=16,
                                                          cap_style=1, join_style=1,
                                                          mitre_limit=5.0)
                        patch = PolygonPatch(smooth_shape, fc=family_fill, ec=family_border,
                                             lw=1, ls='-', alpha=1, fill=True, zorder=-i)
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
                bbox_to_anchor=self.legend_config['families']['position']
            )
            self.ax.add_artist(legend_families)

        else:
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
                bbox_to_anchor=self.legend_config['families']['position']
            )
            self.ax.add_artist(legend_families)

    def add_legend_lines(self):

        # This adds a legend displaying what line width corresponds to
        line_width = list(self.legend_config['posterior_frequency']['default_values'])
        line_width.sort(reverse=True)

        self.line_width_label = []
        self.leg_line_width = []
        self.leg_areas = []

        # Iterates over all values in post_freq_lines and for each adds a legend entry
        for k in line_width:
            # Create line
            line = Line2D([0], [0], color="black", linestyle='-',
                          lw=self.graphic_config['line_width'] * k)
            self.leg_line_width.append(line)

            # Add legend text
            prop_l = int(k * 100)
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
            bbox_to_anchor=self.legend_config['posterior_frequency']['position']
        )

        legend_line_width._legend_box.align = "left"
        self.ax.add_artist(legend_line_width)

    def add_secondary_legend(self):

        x_unit = (self.x_max - self.x_min) / 100
        y_unit = (self.y_max - self.y_min) / 100

        if not self.is_simulation:

            self.ax.axhline(self.y_min + y_unit * 71,
                            0.02, 0.20, lw=1.5, color="black")

            self.ax.add_patch(
                patches.Rectangle(
                    (self.x_min, self.y_min),
                    x_unit * 25, y_unit * 100,
                    color="white"
                ))
        else:
            self.ax.add_patch(
                patches.Rectangle(
                    (self.x_min, self.y_min),
                    x_unit * 55,
                    y_unit * 30,
                    color="white"
                ))

            # The legend looks a bit different, as it has to show both the inferred areas and the ground truth
            self.ax.annotate("INFERRED", (
                self.x_min + x_unit * 3,
                self.y_min + y_unit * 23), fontsize=20)

            self.ax.annotate("GROUND TRUTH", (
                self.x_min + x_unit * 38.5,
                self.y_min + y_unit * 23), fontsize=20)

            self.ax.axvline(self.x_min + x_unit * 37, 0.05, 0.18, lw=2, color="black")

    def add_overview_map(self):
        if not self.is_simulation:
            axins = inset_axes(self.ax, width=self.legend_config['overview']['width'],
                               height=self.legend_config['overview']['height'],
                               bbox_to_anchor=self.legend_config['overview']['position'],
                               loc='lower left',
                               bbox_transform=self.ax.transAxes)
            axins.tick_params(labelleft=False, labelbottom=False, length=0)

            # Map extend of the overview map
            if self.legend_config['overview']['x_extend'] is not None:
                axins.set_xlim(self.legend_config['overview']['x_extend'])

            else:
                x_overview = [self.x_min - self.x_min * 0.1, self.x_max + self.x_max * 0.1]
                axins.set_xlim(x_overview)

            if self.legend_config['overview']['y_extend'] is not None:

                y_overview = [self.y_min - self.y_min * 0.1, self.y_max + self.y_max * 0.1]
                axins.set_xlim(y_overview)

            # Again, this function needs map data to display in the overview map.
            self.add_background_map(axins)

            # add overview to the map
            axins.scatter(*self.locations.T, s=self.graphic_config['point_size'] / 2, c="darkgrey",
                          alpha=1, linewidth=0)

            # adds a bounding box around the overview map
            bbox_width = self.x_max - self.x_min
            bbox_height = self.y_max - self.y_min
            bbox = mpl.patches.Rectangle((self.x_min, self.x_max),
                                         bbox_width, bbox_height, ec='k', fill=False, linestyle='-')
            axins.add_patch(bbox)

    # Helper function
    @staticmethod
    def scientific(x):

        b = int(np.log10(x))
        a = x / 10 ** b
        return '%.2f \cdot 10^{%i}' % (a, b)

    def add_likelihood_legend(self):

        # Legend for area labels
        self.area_labels = ["      log-likelihood per area"]

        lh_per_area = np.array(list(self.results['likelihood_single_areas'].values()), dtype=float)
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
        if self.geo_config['proj4'] is None or self.geo_config['base_map']['geojson_map'] is None:
            raise Exception('If you want to use a map, provide a geojson and a crs!')

        # Adds the geojson map provided by user as background map
        self.world = gpd.read_file(self.geo_config['base_map']['geojson_map'])
        self.world = self.world.to_crs(self.geo_config['proj4'])
        # self.world = gpd.clip(self.world, self.bbox)
        self.world.plot(ax=ax, color='w', edgecolor='black', zorder=-100000)

    # Add rivers
    def add_rivers(self, ax):
        # The user can also provide river data. Looks good on a map :)
        if self.geo_config['base_map']['geojson_river'] is not None:
            self.rivers = gpd.read_file(self.geo_config['base_map']['geojson_river'])
            self.rivers = self.rivers.to_crs(self.geo_config['proj4'])
            self.rivers.plot(ax=ax, color=None, edgecolor="skyblue", zorder=-10000)

    # Add likelihood info box
    def add_likelihood_info(self):
        extra = patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        self.leg_areas.append(extra)

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
        self.ax.scatter(*other_locations.T, s=self.graphic_config['point_size'], c="gainsboro", alpha=1, linewidth=0)

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
        self.sites = sites_all
        
    def visualize_base_map(self):
        if self.is_simulation:
            pass
        else:
            if self.geo_config['base_map']['add']:
                self.add_background_map(ax=self.ax)
                self.add_rivers(ax=self.ax)

    def modify_legend(self):
        class TrueArea(object):
            pass

        class TrueAreaHandler(object):
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

        legend_true_area = self.ax.legend([TrueArea()], ['simulated area\n(bounding polygon)'],
                                          handler_map={TrueArea: TrueAreaHandler()},
                                          bbox_to_anchor=self.ground_truth_config['true_area_polygon_position'],
                                          title_fontsize=16,
                                          loc='upper left',
                                          frameon=True,
                                          edgecolor='#ffffff',
                                          handletextpad=4,
                                          fontsize=18,
                                          ncol=1,
                                          columnspacing=1)

        self.ax.add_artist(legend_true_area)

    def return_correspondence_table(self, file_name, file_format="pdf", ncol=2):
        """ Which language belongs to which number? This table will tell you more
        Args:
            file_name (str): name of the plot
            file_format (str): format of the output file
            ncol(int): number of columns in the output table
        """
        fig, ax = plt.subplots()

        sites_id = []
        sites_names = []
        sites_color = []

        for i in range(len(self.sites['id'])):
            for s in range(len(self.all_labels)):
                if self.sites['id'][i] in self.all_labels[s]:

                    sites_id.append(self.sites['id'][i])
                    sites_names.append(self.sites['names'][i])
                    sites_color.append(self.graphic_config['area_colors'][s])

        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        n_col = ncol
        n_row = math.ceil(len(sites_names) / n_col)

        table_fill = [[] for _ in range(n_row)]
        color_fill = [[] for _ in range(n_row)]

        for i in range(len(sites_id)):
            col = i % n_row
            nr = str(sites_id[i] + 1)

            # Append label and Name
            table_fill[col].append(nr)
            color_fill[col].append(sites_color[i])

            table_fill[col].append(sites_names[i])
            color_fill[col].append(sites_color[i])

        # Fill up empty cells
        for i in range(len(table_fill)):
            if len(table_fill[i]) != n_col * 2:
                fill_up_nr = n_col * 2 - len(table_fill[i])
                for f in range(fill_up_nr):
                    table_fill[i].append("")
                    color_fill[i].append("#000000")

        widths = [0.05, 0.3] * int(((len(table_fill[0])) / 2))

        table = ax.table(cellText=table_fill, loc='center', cellLoc="left", colWidths=widths)
        table.set_fontsize(40)

        # iterate through cells of a table and set color to the one used in the map
        # table_props = table.properties()
        # table_cells = table_props['child_artists']
        # # for c in range(len(table_cells)):
        # #     try:
        # #         table_cells[c].get_text().set_color(color_for_cells[c])
        # #     except IndexError:
        # #         pass
        for i in range(n_row):
            for j in range(2 * n_col):
                table[(i, j)].get_text().set_color(color_fill[i][j])

        for key, cell in table.get_celld().items():
            cell.set_linewidth(0)
        fig.tight_layout()
        fig.savefig(f"{self.path_plots + '/correspondence_' + file_name}.{file_format}",
                    bbox_inches='tight', dpi=400, format=file_format)

    ##############################################################
    # This is the plot_posterior_map function from plotting_old
    ##############################################################
    # for Olga: all parameters should be passed from the new map config file
    def posterior_map(self, file_name='mst_posterior', file_format="pdf",
                      return_correspondence=False):

        """ This function creates a scatter plot of all sites in the posterior distribution.

            Args:
                file_name (str): a path of the output file.
                file_format (str): file format of output figure
                return_correspondence (bool): return the labels of all languages in an area in a separate table?
            """
        print('Plotting map...')

        # Initialize the plot
        self.initialize_map()

        # Styling the axes
        self.style_axes()

        # Add a base map
        self.visualize_base_map()

        # Iterates over all areas in the posterior and plots each with a different color
        self.visualize_areas()

        # Add a small legend displaying what the line width corresponds to.
        self.add_legend_lines()

        # Places additional legend entries on the map
        self.add_secondary_legend()

        # This adds an overview map to the map
        if self.legend_config['overview']['add_overview'] == 'true':
            self.add_overview_map()

        # Visualizes language families
        if self.content_config['plot_families']:
            self.color_families()

        # Modify the legend for simulated data
        if self.is_simulation:
            self.modify_legend()

        # Save the plot
        self.fig.savefig(f"{self.path_plots + '/' + file_name}.{file_format}",
                         bbox_inches='tight', dpi=400, format=file_format)

        if return_correspondence and self.content_config['label_languages']:
            self.return_correspondence_table(file_name=file_name)
        plt.clf()
        plt.close(self.fig)

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
                        true_universal = true_dict[key]
                    elif 'contact' == split_key[1] and str(feature) == split_key[2]:
                        true_contact = true_dict[key]
                    elif 'inheritance' == split_key[1] and str(feature) == split_key[2]:
                        true_inheritance = true_dict[key]
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

            if len(p_dict) == 0:
                raise ValueError(f'No samples found for parameter {parameter}')

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

            return np.array(true_prob, dtype=float)

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
                    true_p = self.transform_probability_vectors(feature=i, parameter=parameter, b_in=b_in, gt=True)
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
    def plot_weights(samples, feature, title, true_weights=None,
                     labels=None, ax=None, mean_weights=False, plot_samples=False):
        """Plot a set of weight vectors in a 2D representation of the probability simplex.

        Args:
            samples (np.array): Sampled weight vectors to plot.
            feature (str): Name of the feature for which weights are being plotted
            true_weights (np.array): the ground truth weights
            labels (list[str]): Labels for each weight dimension.
            ax (plt.Axis): The pyplot axis.
            mean_weights (bool): Plot the mean of the weights?
            plot_samples (bool): Add a scatter plot overlay of the actual samples?
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
        if title:
            plt.text(-0.7, 0.6, str(feature), fontdict={'fontweight': 'bold', 'fontsize': 12})
        x = samples_projected.T[0]
        y = samples_projected.T[1]
        sns.kdeplot(x, y, shade=True, shade_lowest=True, cut=30, n_levels=100,
                    clip=([xmin, xmax], [ymin, ymax]), cmap=cmap)

        if plot_samples:
            plt.scatter(x, y, color='k', lw=0, s=1, alpha=0.2)

        # Draw simplex and crop outside
        plt.fill(*corners.T, edgecolor='k', fill=False)
        Plot.fill_outside(corners, color='w', ax=ax)

        if true_weights is not None:
            true_weights_projected = true_weights.dot(corners)
            plt.scatter(*true_weights_projected.T, color="#ed1696", lw=0, s=50, marker="*")

        if mean_weights:
            mean_projected = np.mean(samples, axis=0).dot(corners)
            plt.scatter(*mean_projected.T, color="#ed1696", lw=0, s=50, marker="o")

        if labels is not None:
            for xy, label in zip(corners, labels):
                xy *= 1.08  # Stretch, s.t. labels don't overlap with corners
                plt.text(*xy, label, ha='center', va='center', fontdict={'fontsize': 12})

        plt.xlim(xmin - 0.1, xmax + 0.1)
        plt.ylim(ymin - 0.1, ymax + 0.1)
        plt.axis('off')
        plt.tight_layout(0)
        plt.plot()

    @staticmethod
    def plot_probability_vectors(samples, feature, true_p=None, labels=None, ax=None, title=False, plot_samples=False):
        """Plot a set of weight vectors in a 2D representation of the probability simplex.

        Args:
            samples (np.array): Sampled weight vectors to plot.
            feature (str): Name of the feature for which weights are being plotted
            true_p (np.array): true probability vectors (only for simulated data)
            labels (list[str]): Labels for each weight dimension.
            title (bool): plot title
            ax (plt.Axis): The pyplot axis.
            plot_samples (bool): Add a scatter plot overlay of the actual samples?
        """
        if ax is None:
            ax = plt.gca()
        n_samples, n_p = samples.shape
        # color map
        cmap = sns.cubehelix_palette(light=1, start=.5, rot=-.75, as_cmap=True)
        if n_p == 2:
            # plt.title(str(feature), loc='center', fontdict={'fontweight': 'bold', 'fontsize': 20})
            x = samples.T[1]
            sns.distplot(x, rug=True, hist=False, kde_kws={"shade": True, "lw": 0, "clip": (0, 1)}, color="g",
                         rug_kws={"color": "k", "alpha": 0.01, "height": 0.03})

            ax.axes.get_yaxis().set_visible(False)

            if true_p is not None:
                plt.scatter(true_p[1], 0, color="#ed1696", lw=0, s=100, marker="*")

            if labels is not None:
                for x, label in enumerate(labels):
                    if x == 0:
                        x = -0.05
                    if x == 1:
                        x = 1.05
                    plt.text(x, 0.1, label, ha='center', va='top', fontdict={'fontsize': 10})
            if title:
                plt.text(0.3, 4, str(feature), fontsize=12, fontweight='bold')

            plt.plot([0, 1], [0, 0], c="k", lw=0.5)

            ax.axes.set_ylim([-0.2, 5])
            ax.axes.set_xlim([0, 1])

            plt.axis('off')
            plt.tight_layout(0)

        elif n_p > 2:
            # Compute corners
            corners = Plot.get_corner_points(n_p)
            # Bounding box
            xmin, ymin = np.min(corners, axis=0)
            xmax, ymax = np.max(corners, axis=0)

            # Sort features s.t. correlated ones are next to each other (minimizes distortion through projection)
            if n_p > 3:
                feat_similarity = np.dot(samples.T, samples)
                assert feat_similarity.shape == (n_p, n_p)
                feat_order = sort_by_similarity(feat_similarity)
                samples = samples[:, feat_order]
                if labels is not None:
                    labels = np.asarray(labels)[feat_order]

            # Project the samples
            samples_projected = samples.dot(corners)

            # Density and scatter plot
            if title:
                plt.text(-0.8, 0.8, str(feature), fontsize=12, fontweight='bold')

            x = samples_projected.T[0]
            y = samples_projected.T[1]
            sns.kdeplot(x, y, shade=True, shade_lowest=True, cut=30, n_levels=100,
                        clip=([xmin, xmax], [ymin, ymax]), cmap=cmap)

            if plot_samples:
                plt.scatter(x, y, color='k', lw=0, s=1, alpha=0.05)

            # Draw simplex and crop outside
            plt.fill(*corners.T, edgecolor='k', fill=False)
            Plot.fill_outside(corners, color='w', ax=ax)

            if true_p is not None:
                true_projected = true_p.dot(corners)
                plt.scatter(*true_projected.T, color="#ed1696", lw=0, s=100, marker="*")

            if labels is not None:
                for xy, label in zip(corners, labels):
                    xy *= 1.1  # Stretch, s.t. labels don't overlap with corners
                    plt.text(*xy, label, ha='center', va='center', wrap=True, fontdict={'fontsize': 10})

            plt.xlim(xmin - 0.1, xmax + 0.1)
            plt.ylim(ymin - 0.1, ymax + 0.1)
            plt.axis('off')

            plt.tight_layout(0)

        plt.plot()

    def plot_weights_grid(self, file_name, file_format="pdf"):

        print('Plotting weights...')
        burn_in = int(len(self.results['posterior']) * self.config['weights_plot']['burn_in'])

        weights, true_weights, _ = self.get_parameters(parameter="weights", b_in=burn_in)

        ordering = self.sort_by_weights(weights)

        n_plots = self.config['weights_plot']['k_best']
        n_col = self.config['weights_plot']['n_columns']
        n_row = math.ceil(n_plots / n_col)
        width = self.config['weights_plot']['output']['fig_width_subplot']
        height = self.config['weights_plot']['output']['fig_height_subplot']
        fig, axs = plt.subplots(n_row, n_col, figsize=(width * n_col, height * n_row))
        position = 1

        features = ordering[:n_plots]
        labels = self.config['weights_plot']['labels']
        n_empty = n_row * n_col - n_plots

        for e in range(1, n_empty + 1):
            axs[-1, -e].axis('off')

        for f in features:
            plt.subplot(n_row, n_col, position)

            if self.is_simulation:
                self.plot_weights(weights[f], feature=f, title=self.config['weights_plot']['title'],
                                  true_weights=true_weights[f], labels=labels)
            else:
                self.plot_weights(weights[f], feature=f, title=self.config['weights_plot']['title'],
                                  labels=labels, mean_weights=True)
            print(position, "of", n_plots, "plots finished")
            position += 1
        plt.subplots_adjust(wspace=self.config['weights_plot']['output']['spacing_horizontal'],
                            hspace=self.config['weights_plot']['output']['spacing_vertical'])

        fig.savefig(self.path_plots + '/' + file_name + '.' + file_format, dpi=400, format=file_format)

    # This is not changed yet
    def plot_probability_grid(self, file_name, file_format="pdf"):
        """Creates a ridge plot for parameters with two states

       Args:
           file_name (str): name of the output file
           file_format (str): output file format
       """
        print('Plotting probabilities...')
        burn_in = int(len(self.results['posterior']) * self.config['probabilities_plot']['burn_in'])

        n_plots = self.config['probabilities_plot']['k']
        n_col = self.config['probabilities_plot']['n_columns']
        n_row = math.ceil(n_plots / n_col)
        width = self.config['probabilities_plot']['output']['fig_width_subplot']
        height = self.config['probabilities_plot']['output']['fig_height_subplot']

        # weights, true_weights, _ = self.get_parameters(parameter="weights", b_in=burn_in)
        # ordering = self.sort_by_weights(weights)

        p, true_p, states = self.get_parameters(parameter=self.config['probabilities_plot']['parameter'], b_in=burn_in)
        fig, axs = plt.subplots(n_row, n_col, figsize=(width * n_col, height * n_row), )

        features = self.results['feature_names']
        n_empty = n_row * n_col - n_plots

        for e in range(1, n_empty + 1):
            axs[-1, -e].axis('off')

        position = 1

        for f in features[0:n_plots]:
            plt.subplot(n_row, n_col, position)

            if self.is_simulation:
                self.plot_probability_vectors(p[f], feature=f, true_p=true_p[f], labels=states[f],
                                              title=self.config['probabilities_plot']['title'])
            else:
                self.plot_probability_vectors(p[f], feature=f, labels=states[f],
                                              title=self.config['probabilities_plot']['title'])
            print(position, "of", n_plots, "plots finished")
            position += 1

        plt.subplots_adjust(wspace=self.config['probabilities_plot']['output']['spacing_horizontal'],
                            hspace=self.config['probabilities_plot']['output']['spacing_vertical'])
        fig.savefig(self.path_plots + '/' + file_name + '.' + file_format, dpi=400,
                    format=file_format)

    def plot_dic(self, models, file_name, file_format="pdf"):
        """This function plots the dics. What did you think?
        Args:
            file_name (str): name of the output file
            file_format (str): output file format
            models(dict): A dict of different models for which the DIC is evaluated
        """
        print('Plotting DIC...')
        width = self.config['dic_plot']['output']['fig_width']
        height = self.config['dic_plot']['output']['fig_height']

        fig, ax = plt.subplots(figsize=(width, height))
        x = list(models.keys())
        y = []

        # Compute the DIC for each model
        for m in x:
            lh = models[m]['likelihood']
            dic = compute_dic(lh, self.config['dic_plot']['burn_in'])
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
        try:
            if self.config['dic_plot']['true_n'] is not None:
                pos_true_model = [idx for idx, val in enumerate(x) if val == self.config['dic_plot']['true_n']][0]
                color_burn_in = 'grey'
                ax.axvline(x=pos_true_model, lw=1, color=color_burn_in, linestyle='--')
                ypos_label = y_min + y_range * 0.15
                plt.text(pos_true_model - 0.05, ypos_label, 'Simulated areas', rotation=90, size=10,
                         color=color_burn_in)
        except KeyError:
            pass
        fig.savefig(self.path_plots + '/' + file_name + '.' + file_format, dpi=400, format=file_format)

    def plot_trace(self, file_name="trace", show_every_k_sample=1, file_format="pdf"):
        """
        Function to plot the trace of a parameter
        Args:
            file_name (str): a path followed by a the name of the file
            file_format (str): format of the output file
            show_every_k_sample (int): show every 1, 1+k,1+2k sample and skip over the rest
        """
        # For Olga: change all three parameters to config file entries
        plt.rcParams["axes.linewidth"] = 1
        fig, ax = plt.subplots(figsize=(self.config['plot_trace']['output']['fig_width'],
                                        self.config['plot_trace']['output']['fig_height']))
        parameter = self.config['plot_trace']['parameter']
        if parameter == 'recall_and_precision':
            y = self.results['recall'][::show_every_k_sample]
            y2 = self.results['precision'][::show_every_k_sample]
            x = self.results['sample_id'][::show_every_k_sample]
            ax.plot(x, y, lw=0.5, color=self.config['plot_trace']['color'][0], label="recall")
            ax.plot(x, y2, lw=0.5, color=self.config['plot_trace']['color'][1], label="precision")
            y_min = 0
            y_max = 1

        else:
            try:
                y = self.results[parameter][::show_every_k_sample]

            except KeyError:
                raise ValueError("Cannot compute trace. " + self.config['plot_trace']['parameter']
                                 + " is not a valid parameter.")

            x = self.results['sample_id'][::show_every_k_sample]
            ax.plot(x, y, lw=0.5, color=self.config['plot_trace']['color'][0], label=parameter)
            y_min, y_max = min(y), max(y)

        y_range = y_max - y_min
        x_min, x_max = 0, x[-1]

        if self.config['plot_trace']['ground_truth']['add']:
            ground_truth_parameter = 'true_' + parameter
            y_gt = self.results[ground_truth_parameter]
            ax.axhline(y=y_gt, xmin=x[0], xmax=x[-1], lw=1, color='#fdbf6f',
                       linestyle='-',
                       label='ground truth')
            y_min, y_max = [min(y_min, y_gt), max(y_max, y_gt)]

        # Show burn-in in plot
        end_bi = math.ceil(x[-1] * self.config['plot_trace']['burn_in'])
        end_bi_label = math.ceil(x[-1] * (self.config['plot_trace']['burn_in'] - 0.03))

        color_burn_in = 'grey'
        ax.axvline(x=end_bi, lw=1, color=color_burn_in, linestyle='--')
        ypos_label = y_min + y_range * 0.15
        plt.text(end_bi_label, ypos_label, 'Burn-in', rotation=90, size=10, color=color_burn_in)

        # Ticks and labels
        n_ticks = 6 if int(self.config['plot_trace']['burn_in'] * 100) % 20 == 0 else 12
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
        ax.set_ylim([y_min - y_range * 0.01, y_max + y_range * 0.01])

        # Legend
        ax.legend(loc=4, prop={'size': 8}, frameon=False)

        # Save
        fig.savefig(self.path_plots + '/' + file_name + '.' + file_format,
                    dpi=400, format=file_format, bbox_inches='tight')
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

    def plot_recall_precision_over_all_models(self, models, file_name, file_format="pdf"):
        width = self.config['recall_precision_over_all_models_plot']['output']['fig_width']
        height = self.config['recall_precision_over_all_models_plot']['output']['fig_height']
        fig, ax = plt.subplots(figsize=(width, height))

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

        fig.savefig(self.path_plots + '/' + file_name + '.' + file_format,
                    dpi=400, format=file_format, bbox_inches='tight')
        plt.close(fig)


def cli(args=None):
    if args is None:
        parser = argparse.ArgumentParser(
            description="plot results of a sBayes analysis")
        parser.add_argument("config", nargs="?", type=Path,
                            help="The JSON configuration file")
        args = parser.parse_args()

    config = args.config

    results_per_model = {}
    plot = Plot(simulated_data=False)
    plot.load_config(config_file=config)
    plot.read_data()

    # Get model names
    names = plot.get_model_names()

    for m in names:

        # Read results for each model
        plot.read_results(model=m)

        print(plot.results['feature_names'])
        exit()

        print('Plotting model', m)

        # How often does a point have to be in the posterior to be visualized in the map?
        min_posterior_frequency = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        mpf_counter = 1
        print('Plotting results for ' + str(len(min_posterior_frequency)) + ' different mpf values')

        for mpf in min_posterior_frequency:

            print('Current mpf: ' + str(mpf) + ' (' + str(mpf_counter) + ' out of ' +
                  str(len(min_posterior_frequency)) + ')')

            # Assign new mpf values
            plot.config['map']['content']['min_posterior_frequency'] = mpf

            # Plot maps
            try:
                plot.posterior_map(file_name='posterior_map_' + m + '_' + str(mpf), return_correspondence=True)
            except ValueError:
                pass

            mpf_counter += 1

        # Plot weights
        plot.plot_weights_grid(file_name='weights_grid_' + m)

        # Plot probability grids
        parameter = ["gamma_a1", "gamma_a2", "gamma_a3", "gamma_a4", "gamma_a5"]
        for p in parameter:
            try:
                plot.config['probabilities_plot']['parameter'] = p
                plot.plot_probability_grid(file_name='prob_grid_' + m + '_' + p)
            except ValueError:
                pass

        # Collect all models for DIC plot
        results_per_model[m] = plot.results

    # Plot DIC over all models
    plot.plot_dic(results_per_model, file_name='dic')



if __name__ == '__main__':

    cli()
