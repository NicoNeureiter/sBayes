import json
import math
import os
from itertools import compress
from statistics import median
import argparse
from pathlib import Path

try:
    import importlib.resources as pkg_resources     # PYTHON >= 3.7

except ImportError:
    import importlib_resources as pkg_resources     # PYTHON < 3.7

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
from sbayes.preprocessing import read_sites, assign_family
from sbayes.util import add_edge, compute_delaunay
from sbayes.util import fix_default_config
from sbayes.util import gabriel_graph_from_delaunay
from sbayes.util import parse_area_columns, read_features_from_csv
from sbayes.cli import iterate_or_run
from sbayes.experiment_setup import REQUIRED, set_defaults
from sbayes import config


DEFAULT_CONFIG = json.loads(pkg_resources.read_text(config, 'default_config_plot.json'))


class Plot:
    def __init__(self):

        # Config variables
        self.config = {}
        self.config_file = None

        # Path variables
        self.path_features = None
        self.path_feature_states = None
        self.path_plots = None

        self.path_areas = None
        self.path_stats = None

        # Input sites, site_names, locations, ...
        self.sites = None
        self.site_names = None
        self.locations = None

        self.families = None
        self.family_names = None

        # Dictionary with all the MCMC results
        self.results = {}

        # Needed for the weights and parameters plotting
        plt.style.use('seaborn-paper')
        # plt.tight_layout()

        # Path to all default configs
        self.config_default = "config/plotting"

    ####################################
    # Configure the parameters
    ####################################

    def load_config(self, config_file):
        # Get parameters from config_custom (for particular experiment)
        self.base_directory, self.config_file = self.decompose_config_path(config_file)

        # Read the user specified config file
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

        # Load defaults
        set_defaults(self.config, DEFAULT_CONFIG)

        # Convert lists to tuples
        self.convert_config(self.config)

        # Verify config
        self.verify_config()

        if not self.config['map']['geo']['map_projection']:
            self.config['map']['geo']['map_projection'] = self.config['data']['projection']

        if self.config['map']['content']['type'] == 'density_map':
            self.config['map']['legend']['correspondence']['color_labels'] = False

        # NN TODO: This is fixing relative paths from experiment_setup.py. Needs some
        #          adaptation here and in config files.
        # # Set results path
        # self.path_results = '{path}/{experiment}/'.format(
        #     path=self.config['results']['RESULTS_PATH'],
        #     experiment=self.experiment_name
        # )
        #
        # # Compile relative paths, to be relative to config file
        # self.path_results = self.fix_relative_path(self.path_results)
        #
        # if not os.path.exists(self.path_results):
        #     os.makedirs(self.path_results)
        #
        # self.add_logger_file(self.path_results)


        # Assign global variables for more convenient workflow
        self.path_plots = self.config['results']['path_out']
        self.path_features = self.config['data']['features']
        self.path_feature_states = self.config['data']['feature_states']

        self.path_areas = list(self.config['results']['path_in']['areas'])
        self.path_stats = list(self.config['results']['path_in']['stats'])

        if not os.path.exists(self.path_plots):
            os.makedirs(self.path_plots)

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

    @staticmethod
    def decompose_config_path(config_path):
        abs_config_path = Path(config_path).absolute()
        base_directory = abs_config_path.parent
        return base_directory, abs_config_path

    def fix_relative_path(self, path):
        """Make sure that the provided path is either absolute or relative to
        the config file directory.

        Args:
            path (Path or str): The original path (absolute or relative).

        Returns:
            Path: The fixed path.
         """
        path = Path(path)
        if path.is_absolute():
            return path
        else:
            return self.base_directory / path

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
        self.sites, self.site_names, _, _, _, _, self.families, self.family_names, _ =\
            read_features_from_csv(self.path_features, self.path_feature_states)
        self.locations = self.sites['locations']

    # Read areas
    # Read the data from the files:
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
                        result[j].append(parsed_sample[j])

        return result

    # Helper function for read_stats
    # Used for reading: weights, alpha, beta, gamma
    @staticmethod
    def read_dictionary(dataframe, search_key):
        param_dict = {}
        for column_name in dataframe.columns:
            if column_name.startswith(search_key):
                param_dict[column_name] = dataframe[column_name].to_numpy(dtype=float)

        return param_dict

    # Helper function for read_stats
    # Bind all statistics together into the dictionary self.results
    def bind_stats(self, sample_id, posterior, likelihood, prior,
                   weights, alpha, beta, gamma,
                   posterior_single_areas, likelihood_single_areas, prior_single_areas, feature_names):
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
        self.results['feature_names'] = feature_names

    # Read stats
    # Read the results from the files:
    # <experiment_path>/stats_<scenario>.txt
    def read_stats(self, txt_path):

        sample_id = None
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

        # Names of distinct features
        feature_names = []
        for key in weights:
            if 'universal' in key:
                feature_names.append(str(key).rsplit('_', 1)[1])

        # Bind all statistics together into the dictionary self.results
        self.bind_stats(sample_id, posterior, likelihood, prior, weights, alpha, beta, gamma,
                        posterior_single_areas, likelihood_single_areas, prior_single_areas, feature_names)

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
        self.read_stats(path_stats)

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

    @staticmethod
    def get_extent(cfg_geo, locations):
        if not cfg_geo['extent']['x']:
            x_min, x_max = np.min(locations[:, 0]), np.max(locations[:, 0])
            x_offset = (x_max - x_min) * 0.3
            x_min = x_min - x_offset
            x_max = x_max + x_offset
        else:
            x_min, x_max = cfg_geo['extent']['x']

        if not cfg_geo['extent']['y']:
            y_min, y_max = np.min(locations[:, 1]), np.max(locations[:, 1])
            y_offset = (y_max - y_min) * 0.1
            y_min = y_min - y_offset
            y_max = y_max + y_offset

        else:
            y_min, y_max = cfg_geo['extent']['y']

        extent = {'x_min': x_min,
                  'x_max': x_max,
                  'y_min': y_min,
                  'y_max': y_max}

        return extent

    @staticmethod
    def compute_bbox(extent):
        bbox = geometry.Polygon([(extent['x_min'], extent['y_min']),
                                 (extent['x_max'], extent['y_min']),
                                 (extent['x_max'], extent['y_max']),
                                 (extent['x_min'], extent['y_max']),
                                 (extent['x_min'], extent['y_min'])])
        return bbox

    @staticmethod
    def style_axes(extent, ax):

        # setting axis limits
        ax.set_xlim([extent['x_min'], extent['x_max']])
        ax.set_ylim([extent['y_min'], extent['y_max']])

        # Removing axis labels
        ax.set_xticks([])
        ax.set_yticks([])

    @staticmethod
    def compute_alpha_shapes(points, alpha_shape):

        """Compute the alpha shape (concave hull) of a set of sites
        Args:
            points (np.array): subset of locations around which to create the alpha shapes (e.g. family, area, ...)
            alpha_shape (float): parameter controlling the convexity of the alpha shape
        Returns:
            (polygon): the alpha shape"""

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

    @staticmethod
    def areas_to_graph(area, locations_map_crs, cfg_content):

        # exclude burn-in
        end_bi = math.ceil(len(area) * cfg_content['burn_in'])
        area = area[end_bi:]

        # compute frequency of each point in area
        area = np.asarray(area)
        n_samples = area.shape[0]

        # Plot a density map or consensus map?
        if cfg_content['type'] == 'density_map':
            in_graph = np.ones(area.shape[1], dtype=bool)

        else:
            area_freq = np.sum(area, axis=0) / n_samples
            in_graph = area_freq >= cfg_content['min_posterior_frequency']

        locations = locations_map_crs[in_graph]
        n_graph = len(locations)

        # getting indices of points in area
        area_indices = np.argwhere(in_graph)

        # For density map: plot all edges
        if cfg_content['type'] == 'density_map':

            a = np.arange(area.shape[1])
            b = np.array(np.meshgrid(a, a))
            c = b.T.reshape(-1, 2)
            graph_connections = c[c[:, 0] < c[:, 1]]

        # For consensus map plot Gabriel graph
        else:
            if n_graph > 3:
                # computing the delaunay
                delaunay = compute_delaunay(locations)
                graph_connections = gabriel_graph_from_delaunay(delaunay, locations)

            elif n_graph == 3:
                graph_connections = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)

            elif n_graph == 2:
                graph_connections = np.array([[0, 1]], dtype=int)

            else:
                return in_graph, [], []

        lines = []
        line_weights = []

        for index in graph_connections:
            # count how often p0 and p1 are together in the posterior of the area
            p = [area_indices[index[0]][0], area_indices[index[1]][0]]
            together_in_area = np.sum(np.all(area[:, p], axis=1)) / n_samples
            lines.append(locations_map_crs[[*p]])
            line_weights.append(together_in_area)

        return in_graph, lines, line_weights

    # Reproject from data CRS to map CRD
    def reproject_to_map_crs(self, map_proj):

        data_proj = self.config['data']['projection']

        if map_proj != data_proj:

            print(f'Reprojecting locations from {data_proj} to {map_proj}.')
            loc = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*self.locations.T), crs=data_proj)
            loc_re = loc.to_crs(map_proj)

            x_re = loc_re.geometry.x
            y_re = loc_re.geometry.y
            locations_reprojected = np.array(list(zip(x_re,y_re)))
            return locations_reprojected
        else:
            return self.locations

    # Initialize the map
    @staticmethod
    def initialize_map(locations, cfg_graphic, ax):

        ax.scatter(*locations.T, s=cfg_graphic['languages']['size'],
                   c=cfg_graphic['languages']['color'], alpha=1, linewidth=0)


    # Add labels to languages in an area
    def add_label(self, is_in_area, current_color, locations_map_crs, extent, ax):

        # Find all languages in areas
        loc_in_area = locations_map_crs[is_in_area, :]
        labels_in_area = list(compress(self.sites['id'], is_in_area))

        for loc in range(len(loc_in_area)):

            range_x = extent['x_max'] - extent['x_min']
            range_y = extent['y_max'] - extent['y_min']
            x, y = loc_in_area[loc]
            x += range_x/200
            y += range_y/200
            anno_opts = dict(xy=(x, y), fontsize=10, color=current_color)
            ax.annotate(labels_in_area[loc] + 1, **anno_opts)

        return labels_in_area


    def visualize_areas(self, locations_map_crs, extent,
                        cfg_content, cfg_graphic, cfg_legend, ax):
        all_labels = []
        # If log-likelihood is displayed: add legend entries with likelihood information per area
        if cfg_legend['areas']['add'] and cfg_legend['areas']['log-likelihood']:
            area_labels, legend_areas = self.add_log_likelihood_legend()

        else:
            area_labels = []
            legend_areas = []
            for i, _ in enumerate(self.results['areas']):
                area_labels.append(f'$Z_{i + 1}$')

        # if self.content_config['type'] == 'density_map':
        #     self.ax.scatter(*self.locations.T,
        #                     s=self.graphic_config['point_size'], c="black")

        # Color areas

        for i, area in enumerate(self.results['areas']):

            # This function computes a Gabriel graph for all points which are in the posterior with at least p_freq
            in_graph, lines, line_w = self.areas_to_graph(area, locations_map_crs, cfg_content)

            current_color = cfg_graphic['areas']['color'][i]

            if cfg_content['type'] == 'density_map':

                for li in range(len(lines)):
                    ax.plot(*lines[li].T, color=current_color, lw=line_w[li] * cfg_graphic['areas']['width'],
                            alpha=line_w[li]*cfg_graphic['areas']['alpha'])

            else:
                ax.scatter(*locations_map_crs[in_graph].T, s=cfg_graphic['areas']['size'], c=current_color)
                for li in range(len(lines)):
                    ax.plot(*lines[li].T, color=current_color,
                            lw=line_w[li] * cfg_graphic['areas']['width'], alpha=cfg_graphic['areas']['alpha'])

            # This adds small lines to the legend (one legend entry per area)
            line_legend = Line2D([0], [0], color=current_color, lw=6, linestyle='-')
            legend_areas.append(line_legend)

            # Label the languages in the areas
            if cfg_graphic['languages']['label']:

                # Use neutral color for density map labels
                if cfg_content['type'] == 'density_map':
                    current_color = "black"

                all_labels.append(self.add_label(in_graph, current_color, locations_map_crs, extent, ax))

        if cfg_legend['areas']['add']:
            # add to legend
            legend_areas = ax.legend(legend_areas, area_labels, title_fontsize=18, title='Contact areas',
                                     frameon=True, edgecolor='#ffffff', framealpha=1, fontsize=16, ncol=1,
                                     columnspacing=1, loc='upper left',
                                     bbox_to_anchor=cfg_legend['areas']['position'])


            legend_areas._legend_box.align = "left"
            ax.add_artist(legend_areas)

        return all_labels


    def color_families(self, locations_maps_crs, cfg_graphic, cfg_legend, ax):
        family_array = self.family_names['external']
        families = self.families

        # Initialize empty legend handle
        handles = []

        # Iterate over all family names
        for i, family in enumerate(family_array):
            family_color = cfg_graphic['families']['color'][i]
            family_fill, family_border = family_color, family_color

            # Find all languages belonging to a family
            is_in_family = families[i] == 1
            family_locations = locations_maps_crs[is_in_family, :]

            # Adds a color overlay for each language in a family
            ax.scatter(*family_locations.T, s=cfg_graphic['families']['size'],
                       c=family_color, alpha=1, linewidth=0, zorder=-i, label=family)

            # For languages with more than three members combine several languages in an alpha shape (a polygon)
            if np.count_nonzero(is_in_family) > 3:
                alpha_shape = self.compute_alpha_shapes(points=family_locations,
                                                        alpha_shape=cfg_graphic['families']['shape'])

                # making sure that the alpha shape is not empty
                if not alpha_shape.is_empty:
                    smooth_shape = alpha_shape.buffer(cfg_graphic['families']['buffer'], resolution=16,
                                                      cap_style=1, join_style=1,
                                                      mitre_limit=5.0)
                    patch = PolygonPatch(smooth_shape, fc=family_fill, ec=family_border,
                                         lw=1, ls='-', alpha=1, fill=True, zorder=-i)
                    ax.add_patch(patch)

            # Add legend handle
            handle = Patch(facecolor=family_color, edgecolor=family_color, label=family)
            handles.append(handle)

        if cfg_legend['families']['add']:

            legend_families = ax.legend(handles=handles, title='Language family', title_fontsize=18,
                                        fontsize=16, frameon=True, edgecolor='#ffffff', framealpha=1,
                                        ncol=1, columnspacing=1, loc='upper left',
                                        bbox_to_anchor=cfg_legend['families']['position'])
            ax.add_artist(legend_families)

    @staticmethod
    def add_legend_lines(cfg_graphic, cfg_legend, ax):

        # This adds a legend displaying what line width corresponds to
        line_width = list(cfg_legend['lines']['reference_frequency'])
        line_width.sort(reverse=True)

        line_width_label = []
        leg_line_width = []

        # Iterates over all values in post_freq_lines and for each adds a legend entry
        for k in line_width:
            # Create line
            line = Line2D([0], [0], color="black", linestyle='-',
                          lw=cfg_graphic['areas']['width'] * k)

            leg_line_width.append(line)

            # Add legend text
            prop_l = int(k * 100)
            line_width_label.append(f'{prop_l}%')

        # Adds everything to the legend
        legend_line_width = ax.legend(leg_line_width, line_width_label, title_fontsize=18,
                                      title='Frequency of edge in posterior', frameon=True, edgecolor='#ffffff',
                                      framealpha=1, fontsize=16, ncol=1, columnspacing=1, loc='upper left',
                                      bbox_to_anchor=cfg_legend['lines']['position'])

        legend_line_width._legend_box.align = "left"
        ax.add_artist(legend_line_width)

    def add_overview_map(self, locations_map_crs, extent, cfg_geo, cfg_graphic, cfg_legend, ax):
        axins = inset_axes(ax, width=cfg_legend['overview']['width'],
                           height=cfg_legend['overview']['height'],
                           bbox_to_anchor=cfg_legend['overview']['position'],
                           loc='lower left',
                           bbox_transform=ax.transAxes)
        axins.tick_params(labelleft=False, labelbottom=False, length=0)

        # Map extend of the overview map
        if not cfg_legend['overview']['extent']['x']:
            x_spacing = (extent['x_max'] - extent['x_min']) * 0.9
            x_min = extent['x_min'] - x_spacing
            x_max = extent['x_max'] + x_spacing
        else:
            x_min = cfg_legend['overview']['extent']['x'][0]
            x_max = cfg_legend['overview']['extent']['x'][1]

        if not cfg_legend['overview']['extent']['y']:
            y_spacing = (extent['y_max'] - extent['y_min']) * 0.9
            y_min = extent['y_min'] - y_spacing
            y_max = extent['y_max'] + y_spacing

        else:
            y_min = cfg_legend['overview']['extent']['y'][0]
            y_max = cfg_legend['overview']['extent']['y'][1]


        axins.set_xlim([x_min, x_max])
        axins.set_ylim([y_min, y_max])

        overview_extent = {'x_min': x_min, 'x_max': x_max,
                           'y_min': y_min, 'y_max': y_max}

        # Again, this function needs map data to display in the overview map.
        bbox = self.compute_bbox(overview_extent)
        self.add_background_map(bbox, cfg_geo, cfg_graphic, axins)

        # add overview to the map
        axins.scatter(*locations_map_crs.T, s=cfg_graphic['languages']['size'] / 2,
                      c=cfg_graphic['languages']['color'], alpha=1, linewidth=0)

        # adds a bounding box around the overview map
        bbox_width = extent['x_max'] - extent['x_min']
        bbox_height = extent['y_max'] - extent['y_min']

        bbox = mpl.patches.Rectangle((extent['x_min'], extent['y_min']),
                                     bbox_width, bbox_height, ec='k', fill=False, linestyle='-')
        axins.add_patch(bbox)

    # Helper function
    @staticmethod
    def scientific(x):
        b = int(np.log10(x))
        a = x / 10 ** b
        return '%.2f \cdot 10^{%i}' % (a, b)

    def add_log_likelihood_legend(self):

        # Legend for area labels
        area_labels = ["      log-likelihood per area"]

        lh_per_area = np.array(list(self.results['likelihood_single_areas'].values()), dtype=float)
        to_rank = np.mean(lh_per_area, axis=1)
        p = to_rank[np.argsort(-to_rank)]

        for i, lh in enumerate(p):
            area_labels.append(f'$Z_{i + 1}: \, \;\;\; {int(lh)}$')


        extra = patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        # Line2D([0], [0], color=None, lw=6, linestyle='-')

        return area_labels, [extra]

    @staticmethod
    def add_background_map(bbox, cfg_geo, cfg_graphic, ax):
        # Adds the geojson polygon geometries provided by the user as a background map
        world = gpd.read_file(cfg_geo['base_map']['geojson_polygon'])
        world = world.to_crs(cfg_geo['map_projection'])
        world = gpd.clip(world, bbox)

        cfg_polygon = cfg_graphic['base_map']['polygon']
        world.plot(ax=ax, facecolor=cfg_polygon['color'],
                   edgecolor=cfg_polygon['outline_color'],
                   lw=cfg_polygon['outline_width'],
                   zorder=-100000)

    @staticmethod
    def add_rivers(bbox, cfg_geo, cfg_graphic, ax):
        # The user can provide geojson line geometries, for example those for rivers. Looks good on a map :)
        rivers = gpd.read_file(cfg_geo['base_map']['geojson_line'])
        rivers = rivers.to_crs(cfg_geo['map_projection'])
        rivers = gpd.clip(rivers, bbox)

        cfg_line = cfg_graphic['base_map']['line']
        rivers.plot(ax=ax, color=None, edgecolor=cfg_line['color'],
                    lw=cfg_line['width'], zorder=-10000)

    def visualize_base_map(self, extent, cfg_geo, cfg_graphic, ax):

        if cfg_geo['base_map']['add']:
            bbox = self.compute_bbox(extent)
            if not cfg_geo['base_map']['geojson_polygon']:
                print(f'Cannot add base map. Please provide a geojson_polygon')
            else:
                self.add_background_map(bbox, cfg_geo, cfg_graphic, ax)
            if not cfg_geo['base_map']['geojson_line']:
                pass
            else:
                self.add_rivers(bbox, cfg_geo, cfg_graphic, ax)

    def add_correspondence_table(self, all_labels, cfg_graphic, cfg_legend, ax_c):
        """ Which language belongs to which number? This table will tell you more"""

        sites_id = []
        sites_names = []
        sites_color = []

        for i in range(len(self.sites['id'])):
            for s in range(len(all_labels)):
                if self.sites['id'][i] in all_labels[s]:

                    sites_id.append(self.sites['id'][i])
                    sites_names.append(self.sites['names'][i])
                    sites_color.append(cfg_graphic['areas']['color'][s])

        n_col = cfg_legend['correspondence']['n_columns']
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

        widths = [0.06, 0.2] * int(((len(table_fill[0])) / 2))
        y_min = -(cfg_legend['correspondence']['table_height'] + 0.01)
        table = ax_c.table(cellText=table_fill, cellLoc="left", colWidths=widths,
                           bbox=(0.01, y_min, 0.98, cfg_legend['correspondence']['table_height']))

        table.auto_set_font_size(False)
        table.set_fontsize(cfg_legend['correspondence']['font_size'])
        table.scale(1, 2)

        if cfg_legend['correspondence']['color_labels']:

            for i in range(n_row):
                for j in range(2 * n_col):
                    table[(i, j)].get_text().set_color(color_fill[i][j])

        for key, cell in table.get_celld().items():
            cell.set_linewidth(0)


    ##############################################################
    # This is the plot_posterior_map function from plotting_old
    ##############################################################

    def posterior_map(self, file_name='mst_posterior'):

        """ This function creates a scatter plot of all sites in the posterior distribution.

            Args:
                file_name (str): a path of the output file.
            """
        print('Plotting map...')
        cfg_content = self.config['map']['content']
        cfg_geo = self.config['map']['geo']
        cfg_graphic = self.config['map']['graphic']
        cfg_legend = self.config['map']['legend']
        cfg_output = self.config['map']['output']


        for f in plt.get_fignums():
            plt.close(f)

        # Initialize the plot

        fig, ax = plt.subplots(figsize=(cfg_output['width'],
                                        cfg_output['height']), constrained_layout=True)
        #     ax_c = []
        #
        # else:
        #     fig, (ax, ax_c) = plt.subplots(nrows=2, figsize=(cfg_output['width'],
        #                                                      cfg_output['height']*2),
        #                                    gridspec_kw={'height_ratios': [4, 1]},
        #                                    constrained_layout=True)

        locations_map_crs = self.reproject_to_map_crs(cfg_geo['map_projection'])

        # Get extent
        extent = self.get_extent(cfg_geo, locations_map_crs)

        # Initialize the map
        self.initialize_map(locations_map_crs, cfg_graphic, ax)

        # Style the axes
        self.style_axes(extent, ax)

        # Add a base map
        self.visualize_base_map(extent, cfg_geo, cfg_graphic, ax)

        # Iterates over all areas in the posterior and plots each with a different color
        all_labels = self.visualize_areas(locations_map_crs, extent, cfg_content, cfg_graphic, cfg_legend, ax)

        # Visualizes language families
        if cfg_content['plot_families']:
            self.color_families(locations_map_crs, cfg_graphic, cfg_legend, ax)

        # Add main legend
        if cfg_legend['lines']['add']:
            self.add_legend_lines(cfg_graphic, cfg_legend, ax)

        # # Places additional legend entries on the map
        #
        # if self.legend_config['overview']['add_to_legend'] or\
        #     self.legend_config['areas']['add_to_legend'] or\
        #         self.legend_config['families']['add_to_legend']:
        #
        #     self.add_secondary_legend()

        # This adds an overview map to the map
        if cfg_legend['overview']['add']:
            self.add_overview_map(locations_map_crs, extent, cfg_geo, cfg_graphic, cfg_legend, ax)

        if cfg_legend['correspondence']['add'] and cfg_graphic['languages']['label']:
            if cfg_content['type']== "density_map":
                all_labels = [self.sites['id']]

            self.add_correspondence_table(all_labels, cfg_graphic, cfg_legend, ax)

        # Save the plot

        file_format = cfg_output['format']

        fig.savefig(f"{self.path_plots + '/' + file_name}.{file_format}",bbox_inches='tight',
                    dpi=cfg_output['resolution'], format=file_format)
        plt.close(fig)

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
    def transform_weights(self, feature, b_in):

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

    def transform_probability_vectors(self, feature, parameter, b_in):

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
            if key.startswith(parameter + '_' + feature + '_'):
                state = str(key).rsplit('_', 1)[1]
                p_dict[state] = sample_dict[key][b_in:]
                states.append(state)

        sample = np.column_stack([p_dict[s] for s in p_dict]).astype(np.float)
        return sample, states


    # Get preferences or weights from relevant features
    def get_parameters(self, b_in, features, parameter="weights"):

        par = {}
        states = {}
        # if features is empty, get parameters for all features
        if not features:
            feature_names = self.results['feature_names']
        else:
            feature_names = list(self.results['feature_names'][i-1] for i in features)

        # get samples
        for i in feature_names:

            if parameter == "weights":
                p = self.transform_weights(feature=i, b_in=b_in)
                par[i] = p

            elif "alpha" in parameter or "beta" in parameter or "gamma" in parameter:
                p, state = self.transform_probability_vectors(feature=i, parameter=parameter, b_in=b_in)

                par[i] = p
                states[i] = state

        return par, states

    def sort_by_weights(self, w):
        sort_by = {}
        for i in self.results['feature_names']:
            sort_by[i] = median(w[i][:, 1])
        ordering = sorted(sort_by, key=sort_by.get, reverse=True)
        return ordering

    # Probability simplex (for one feature)
    @staticmethod
    def plot_weight(samples, feature, title,
                    labels=None, ax=None, mean_weights=False, plot_samples=False):
        """Plot a set of weight vectors in a 2D representation of the probability simplex.

        Args:
            samples (np.array): Sampled weight vectors to plot.
            feature (str): Name of the feature for which weights are being plotted
            title (str): Plot title
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
        sns.kdeplot(data=x, data2=y, shade=True,  cut=30, n_levels=100,
                    clip=([xmin, xmax], [ymin, ymax]), cmap=cmap)

        if plot_samples:
            plt.scatter(x, y, color='k', lw=0, s=1, alpha=0.2)

        # Draw simplex and crop outside
        plt.fill(*corners.T, edgecolor='k', fill=False)
        Plot.fill_outside(corners, color='w', ax=ax)

        if mean_weights:
            mean_projected = np.mean(samples, axis=0).dot(corners)
            plt.scatter(*mean_projected.T, color="#ed1696", lw=0, s=50, marker="o")

        if labels is not None:
            for xy, label in zip(corners, labels):
                xy *= 1.08  # Stretch, s.t. labels don't overlap with corners
                plt.text(*xy, label, ha='center', va='center', fontdict={'fontsize': 6})

        plt.xlim(xmin - 0.1, xmax + 0.1)
        plt.ylim(ymin - 0.1, ymax + 0.1)
        plt.axis('off')
        plt.plot()

    @staticmethod
    def plot_preference(samples, feature, labels=None, ax=None, title=False, plot_samples=False):
        """Plot a set of weight vectors in a 2D representation of the probability simplex.

        Args:
            samples (np.array): Sampled weight vectors to plot.
            feature (str): Name of the feature for which weights are being plotted
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
            x = samples.T[1]
            sns.distplot(x, rug=True, hist=False, kde_kws={"shade": True, "lw": 0, "clip": (0, 1)}, color="g",
            # sns.displot(x=x, rug=True, kde_kws={"shade": True, "lw": 0, "clip": (0, 1)}, color="g",
                         rug_kws={"color": "k", "alpha": 0.01, "height": 0.03})

            ax.axes.get_yaxis().set_visible(False)

            if labels is not None:
                for x, label in enumerate(labels):
                    if x == 0:
                        x = -0.05
                    if x == 1:
                        x = 1.05
                    plt.text(x, 0.1, label, ha='center', va='top', fontdict={'fontsize': 6})
            if title:
                plt.text(0.3, 4, str(feature), fontsize=6, fontweight='bold')

            plt.plot([0, 1], [0, 0], c="k", lw=0.5)

            ax.axes.set_ylim([-0.2, 5])
            ax.axes.set_xlim([0, 1])

            plt.axis('off')

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
                plt.text(-0.8, 0.8, str(feature), fontsize=6, fontweight='bold')

            x = samples_projected.T[0]
            y = samples_projected.T[1]
            sns.kdeplot(x=x, y=y, shade=True, thresh=0, cut=30, n_levels=100,
                        clip=([xmin, xmax], [ymin, ymax]), cmap=cmap)

            if plot_samples:
                plt.scatter(x, y, color='k', lw=0, s=1, alpha=0.05)

            # Draw simplex and crop outside
            plt.fill(*corners.T, edgecolor='k', fill=False)
            Plot.fill_outside(corners, color='w', ax=ax)

            if labels is not None:
                for xy, label in zip(corners, labels):
                    xy *= 1.1  # Stretch, s.t. labels don't overlap with corners
                    plt.text(*xy, label, ha='center', va='center', fontdict={'fontsize': 6})

            plt.xlim(xmin - 0.1, xmax + 0.1)
            plt.ylim(ymin - 0.1, ymax + 0.1)
            plt.axis('off')

        plt.plot()

    def plot_weights(self, file_name):

        print('Plotting weights...')
        cfg_weights = self.config['weight_plot']
        burn_in = int(len(self.results['posterior']) * cfg_weights['content']['burn_in'])

        weights, _ = self.get_parameters(parameter="weights", b_in=burn_in,
                                         features=cfg_weights['content']['features'])

        # Todo: reactivate?
        # ordering = self.sort_by_weights(weights)
        # features = ordering[:n_plots]

        features = weights.keys()
        n_plots = len(features)
        n_col = cfg_weights['graphic']['n_columns']
        n_row = math.ceil(n_plots / n_col)

        width = cfg_weights['output']['width_subplot']
        height = cfg_weights['output']['height_subplot']
        fig, axs = plt.subplots(n_row, n_col, figsize=(width * n_col, height * n_row))
        position = 1

        labels = cfg_weights['graphic']['labels']
        n_empty = n_row * n_col - n_plots

        for e in range(1, n_empty + 1):
            if len(axs.shape) == 1:
                axs[-e].axis('off')
            else:
                axs[-1, -e].axis('off')

        for f in features:
            plt.subplot(n_row, n_col, position)

            self.plot_weight(weights[f], feature=f, title=cfg_weights['graphic']['title'],
                             labels=labels, mean_weights=True)

            print(position, "of", n_plots, "plots finished")
            position += 1

        # todo: spacing in config?
        width_spacing = 0.1
        height_spacing = 0.1
        plt.subplots_adjust(wspace=width_spacing,
                            hspace=height_spacing)
        file_format = cfg_weights['output']['format']
        resolution = cfg_weights['output']['resolution']
        fig.savefig(self.path_plots + '/' + file_name + '.' + file_format,
                    dpi=resolution, format=file_format)

    # This is not changed yet
    def plot_preferences(self, file_name):
        """Creates preference plots for universal, areas and families

       Args:
           file_name (str): name of the output file
       """
        print('Plotting preferences...')
        cfg_preference = self.config['preference_plot']
        burn_in = int(len(self.results['posterior']) * cfg_preference['content']['burn_in'])

        width = cfg_preference['output']['width_subplot']
        height = cfg_preference['output']['height_subplot']

        # todo: spacing in config?
        width_spacing = 0.2
        height_spacing = 0.2

        file_format = cfg_preference['output']['format']
        resolution = cfg_preference['output']['resolution']

        gk = self.results['gamma'].keys()
        ugk = list(set([k.split('_')[1] for k in gk]))
        available_gamma_keys = ['gamma_' + u for u in ugk]

        if not cfg_preference['content']['preference']:
            available_preferences = available_gamma_keys
        else:
            available_preferences = []
            for p in cfg_preference['content']['preference']:
                if p == 'universal':
                    available_preferences.append("alpha")
                elif "area" in p:
                    gamma = "gamma_a" + p.split('_')[1]
                    if gamma not in available_gamma_keys:
                        continue
                    else:
                        available_preferences.append(gamma)
                else:
                    beta = "beta_" + p
                    available_preferences.append(beta)

        for p in available_preferences:

            preferences, states = \
                self.get_parameters(parameter=p, b_in=burn_in,
                                    features=cfg_preference['content']['features'])
            features = preferences.keys()

            n_plots = len(features)
            n_col = cfg_preference['graphic']['n_columns']
            n_row = math.ceil(n_plots / n_col)

            fig, axs = plt.subplots(n_row, n_col, figsize=(width * n_col, height * n_row), )
            n_empty = n_row * n_col - n_plots

            for e in range(1, n_empty + 1):
                if len(axs.shape) == 1:
                    axs[-e].axis('off')
                else:
                    axs[-1, -e].axis('off')

            position = 1

            for f in features:
                plt.subplot(n_row, n_col, position)
                if cfg_preference['graphic']['labels']:
                    labels = states[f]
                else:
                    labels = None

                self.plot_preference(preferences[f], feature=f, labels=labels,
                                     title=cfg_preference['graphic']['title'])

                print(p, ": ", position, "of", n_plots, "plots finished")
                position += 1

            plt.subplots_adjust(wspace=width_spacing, hspace=height_spacing)
            fig.savefig(self.path_plots + '/' + file_name + '_' + p + '.' + file_format,
                        dpi=resolution, format=file_format)


    def plot_dic(self, models, file_name):
        """This function plots the dics. What did you think?
        Args:
            file_name (str): name of the output file
            models(dict): A dict of different models for which the DIC is evaluated
        """
        print('Plotting DIC...')
        cfg_dic = self.config['dic_plot']

        width = cfg_dic['output']['width']
        height = cfg_dic['output']['height']

        file_format = cfg_dic['output']['format']
        resolution = cfg_dic['output']['resolution']

        fig, ax = plt.subplots(figsize=(width, height))
        available_models = list(models.keys())
        if not cfg_dic['content']['model']:
            x = available_models
        else:
            x = [available_models[i-1] for i in cfg_dic['content']['model']]
        y = []

        if len(x) < 0:
            print('Need at least 2 models for DIC plot.')
            return

        # Compute the DIC for each model
        for m in x:
            lh = models[m]['likelihood']
            dic = compute_dic(lh, self.config['dic_plot']['content']['burn_in'])
            y.append(dic)

        if cfg_dic['graphic']['line_plot']:
            lw = 1
        else:
            lw = 0

        ax.plot(x, y, lw=lw, color='#000000',
                label='DIC', marker='o')

        # Limits
        y_min, y_max = min(y), max(y)
        y_range = y_max - y_min

        x_min = -0.2
        x_max = len(x) - 0.8

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min - y_range * 0.1, y_max + y_range * 0.1])

        # Labels and ticks
        ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        ax.set_ylabel('DIC', fontsize=10, fontweight='bold')

        if not cfg_dic['graphic']['labels']:
            labels = x
        else:
            labels = cfg_dic['graphic']['labels']

        ax.set_xticklabels(labels, fontsize=8)
        y_ticks = np.linspace(y_min, y_max, 6)
        ax.set_yticks(y_ticks)
        yticklabels = [f'{y_tick:.0f}' for y_tick in y_ticks]
        ax.set_yticklabels(yticklabels, fontsize=10)

        fig.savefig(self.path_plots + '/' + file_name + '.' + file_format, dpi=resolution, format=file_format)


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
        g = lambda k, pos: "${}$".format(f._formatSciNotation('%1.10e' % k))
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
        g = lambda k, pos: "${}$".format(f._formatSciNotation('%1.10e' % k))
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


    def plot_pies(self, file_name, file_format="pdf"):
        print('Plotting pie charts ...')

        areas = np.array(self.results['areas'])
        end_bi = math.ceil(areas.shape[1] * self.content_config['burn_in'])

        samples = areas[:, end_bi:, ]

        n_plots = samples.shape[2]
        n_samples = samples.shape[1]

        samples_per_area = np.sum(samples, axis=1)

        # Grid
        n_col = 3
        n_row = math.ceil(n_plots / n_col)

        width = self.config['pie_plot']['output']['fig_width_subplot']
        height = self.config['pie_plot']['output']['fig_height_subplot']

        fig, axs = plt.subplots(n_row, n_col, figsize=(width * n_col, height * n_row))

        for l in range(n_col*n_row):

            ax_col = int(np.floor(l/n_row))
            ax_row = l - n_row * ax_col

            if l < n_plots:
                per_lang = samples_per_area[:, l]
                no_area = n_samples - per_lang.sum()

                x = per_lang.tolist()
                col = list(self.graphic_config['area_colors'])[:len(x)]

                # Append samples that are not in an area
                x.append(no_area)
                col.append("lightgrey")

                axs[ax_row, ax_col].pie(x, colors=col, radius=15)
                label = str(self.sites['id'][l] + 1) + ' ' + str(self.sites['names'][l])
                axs[ax_row, ax_col].text(-160, 0, label, size=15, va='center', ha="left")
                axs[ax_row, ax_col].set_xlim([-160, 5])
                axs[ax_row, ax_col].set_ylim([-10, 10])

            axs[ax_row, ax_col].axis('off')

        # Style remaining empty cells in the plot
        n_empty = n_row * n_col - n_plots
        for e in range(1, n_empty):
            axs[-1, -e].axis('off')

        plt.subplots_adjust(wspace=self.config['pie_plot']['output']['spacing_horizontal'],
                            hspace=self.config['pie_plot']['output']['spacing_vertical'])

        fig.savefig(self.path_plots + '/' + file_name + '.' + file_format,
                    dpi=400, format=file_format, bbox_inches='tight')


def main(config=None, only_plot=None):
    # TODO adapt paths according to experiment_name (if provided)
    # TODO add argument defining which plots to generate (all [default], map, weights, ...)
    ALL_PLOT_TYPES = ['map', 'weights_plot', 'probabilities_plot', 'pie_plot']

    # If no plot type is specified, plot everything in the config file
    plot_types = ALL_PLOT_TYPES

    if config is None:
        parser = argparse.ArgumentParser(description="Plot the results of a sBayes run.")
        parser.add_argument("config", type=Path, help="The JSON configuration file")
        parser.add_argument("type", nargs='?', type=str,
                            help="The type of plot to generate")
        args = parser.parse_args()

        config = args.config
        if args.type is not None:
            if args.type not in ALL_PLOT_TYPES:
                raise ValueError('Unknown plot type: ' + args.type)
            plot_types = [args.type]

    results_per_model = {}
    plot = Plot(simulated_data=False)
    plot.load_config(config_file=config)
    plot.read_data()

    # Get model names
    names = plot.get_model_names()

    def should_be_plotted(plot_type):
        """A plot type should only be generated if it
            1) is specified in the config file and
            2) is in the reuqested list of plot types."""
        return (plot_type in plot.config) and (plot_type in plot_types)

    for m in names:
        # Read results for model ´m´
        plot.read_results(model=m)
        print('Plotting model', m)

        # Plot the reconstructed areas on a map
        # TODO (NN) For now we always plot the map, since the other plotting functions
        #  depend on the preprocessing done in plot_map. I suggest we resolve this when
        #  separating area summarization from plotting.
        # if should_be_plotted('map'):
        plot_map(plot, m)

        # Plot the reconstructed mixture weights in simplex plots
        if should_be_plotted('weights_plot'):
            plot.plot_weights_grid(file_name='weights_grid_' + m)

        # Plot the reconstructed probability vectors in simplex plots
        if should_be_plotted('probabilities_plot'):
            iterate_or_run(
                x=plot.config['probabilities_plot']['parameter'],
                config_setter=lambda x: plot.config['probabilities_plot'].__setitem__('parameter', x),
                function=lambda x: plot.plot_probability_grid(file_name=f'prob_grid_{m}_{x}')
            )

        # Plot the reconstructed areas in pie-charts
        # (one per language, showing how likely the language is to be in each area)
        if should_be_plotted('pie_plot'):
            plot.plot_pies(file_name= 'plot_pies_' + m)

        results_per_model[m] = plot.results

    # Plot DIC over all models
    if should_be_plotted('dic_plot'):
        plot.plot_dic(results_per_model, file_name='dic')


def plot_map(plot, m):
    map_type = plot.config['map']['content']['type']

    if map_type == plot.config['map']['content']['type'] == 'density_map':
        plot.posterior_map(file_name='posterior_map_' + m)

    elif map_type == plot.config['map']['content']['type'] == 'density_map':
        iterate_or_run(
            x=plot.config['map']['content']['min_posterior_frequency'],
            config_setter=lambda x: plot.config['map']['content'].__setitem__('min_posterior_frequency', x),
            function=lambda x: plot.posterior_map(file_name=f'posterior_map_{m}_{x}'),
            print_message='Current mpf: {value} ({i} out of {len(mpf_values)})'
        )
    else:
        raise ValueError(f'Unknown map type: {map_type}  (in the config file "map" -> "content" -> "type")')


if __name__ == '__main__':
    main()

