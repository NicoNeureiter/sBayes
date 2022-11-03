from __future__ import annotations

import json
import logging
import math
import os
from argparse import Namespace
from enum import Enum
from functools import lru_cache
from itertools import compress
from pathlib import Path
import typing as typ

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
from numpy.typing import NDArray
import seaborn as sns
import colorsys

from descartes import PolygonPatch
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import CRS
from scipy.spatial import Delaunay
from shapely import geometry
from shapely.ops import cascaded_union, polygonize

from sbayes.postprocessing import compute_dic
from sbayes.results import Results
from sbayes.util import add_edge, compute_delaunay, set_defaults
from sbayes.util import fix_relative_path
from sbayes.util import gabriel_graph_from_delaunay
from sbayes.util import parse_cluster_columns
from sbayes.util import read_data_csv
from sbayes.util import PathLike
from sbayes.load_data import Objects
from sbayes import config as config_package
from sbayes import maps as maps_package

DEFAULT_CONFIG = json.loads(pkg_resources.read_text(config_package, 'default_config_plot.json'))


class Plot:

    # Attributes
    config: dict[str, ...]
    config_file: Path
    base_directory: Path
    all_cluster_paths: list[Path]
    all_stats_paths: list[Path]

    # Constant class attributes
    pref_color_map = sns.cubehelix_palette(light=1, start=.5, rot=-.75, as_cmap=True)
    # pref_color_map = sns.color_palette("rocket_r", as_cmap=True)

    def __init__(self):

        # Config variables
        self.config = {}
        self.config_file = None
        self.base_directory = None

        # Path variables
        self.path_features = None
        self.path_feature_states = None
        self.path_plots = None
        self.cluster_path = None
        self.stats_path = None

        # Input sites, site_names, locations, ...
        self.objects = None
        self.site_names = None
        self.locations = None

        self.families = None
        self.family_names = None

        # Dictionary with all the MCMC results
        self.results: typ.Dict[str, Results] = {}

        # Needed for the weights and parameters plotting
        plt.style.use('seaborn-paper')
        # plt.tight_layout()

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

        # Fix relative paths and assign global variables for more convenient workflow
        self.path_plots = fix_relative_path(self.config['results']['path_out'], self.base_directory)

        self.path_features = fix_relative_path(self.config['data']['features'], self.base_directory)
        self.path_feature_states = fix_relative_path(self.config['data']['feature_states'], self.base_directory)

        input_paths = self.config['results']['path_in']
        self.all_cluster_paths = [fix_relative_path(i, self.base_directory) for i in input_paths['clusters']]
        self.all_stats_paths = [fix_relative_path(i, self.base_directory) for i in input_paths['stats']]

        if not os.path.exists(self.path_plots):
            os.makedirs(self.path_plots)

    @staticmethod
    def convert_config(d):
        for k, v in d.items():
            if isinstance(v, dict):
                Plot.convert_config(v)
            else:
                if type(v) == list:
                    d[k] = tuple(v)

    def verify_config(self):
        pass

    @staticmethod
    def decompose_config_path(config_path) -> tuple[Path, Path]:
        abs_config_path = Path(config_path).absolute()
        base_directory = abs_config_path.parent
        return base_directory, abs_config_path


    ####################################
    # Read the data and the results
    ####################################

    # Functions related to the current scenario (run in a loop over scenarios, i.e. n_clusters)
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
        data = read_data_csv(self.path_features)
        self.objects = Objects.from_dataframe(data)
        self.locations = self.objects.locations

    # Read clusters
    # Read the data from the files:
    # <experiment_path>/clusters_<scenario>.txt
    @staticmethod
    def read_clusters(txt_path):
        result = []

        with open(txt_path, 'r') as f_sample:

            # This makes len(result) = number of clusters (flipped array)

            # Split the sample
            # len(byte_results) equals the number of samples
            byte_results = (f_sample.read()).split('\n')

            # Get the number of clusters
            n_clusters = len(byte_results[0].split('\t'))

            # Append empty arrays to result, so that len(result) = n_clusters
            for i in range(n_clusters):
                result.append([])

            # Process each sample
            for sample in byte_results:

                # Exclude empty lines
                if len(sample) > 0:

                    # Parse each sample
                    # len(parsed_result) equals the number of clusters
                    # parse_cluster_columns.shape equals (n_clusters, n_sites)
                    parsed_sample = parse_cluster_columns(sample)

                    # Add each item in parsed_cluster_columns to the corresponding array in result
                    for j in range(len(parsed_sample)):
                        result[j].append(parsed_sample[j])

        return result

    @staticmethod
    def read_dictionary(dataframe: pd.DataFrame, search_key: str) -> typ.Dict[str, NDArray]:
        """Helper function for read_stats. Used for reading weights and preferences. """
        param_dict = {}
        for column_name in dataframe.columns:
            if column_name.startswith(search_key):
                param_dict[column_name] = dataframe[column_name].to_numpy(dtype=float)

        return param_dict

    def iterate_over_models(self) -> str:
        for clusters_path, stats_path in zip(sorted(self.all_cluster_paths),
                                             sorted(self.all_stats_paths)):
            prefix, _, model_name = str(clusters_path.stem).partition('_')
            results = Results.from_csv_files(
                clusters_path=clusters_path,
                parameters_path=stats_path,
                burn_in=self.config['map']['content']['burn_in'],
                # TODO move burn_in to results section of config
            )

            self.results[model_name] = results

            yield model_name, results

    def get_model_names(self):
        last_part = [str(p).replace('\\', '/').rsplit('/', 1)[-1] for p in self.all_cluster_paths]
        name = [str(p).rsplit('_')[1] for p in last_part]
        return name

    # From map.py:
    ##############################################################
    # Copy-pasted functions needed for plot_posterior_map
    ##############################################################
    # Load default config parameters

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
        bbox = geometry.box(extent['x_min'], extent['y_min'],
                            extent['x_max'], extent['y_max'])
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
            points (np.array): subset of locations around which to create the alpha shapes (e.g. family, cluster, ...)
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
    def clusters_to_graph(cluster, locations_map_crs, cfg_content):

        # exclude burn-in
        end_bi = math.ceil(len(cluster) * cfg_content['burn_in'])
        cluster = cluster[end_bi:]

        # compute frequency of each point in cluster
        cluster = np.asarray(cluster)
        n_samples = cluster.shape[0]

        # Plot a density map or consensus map?
        if cfg_content['type'] == 'density_map':
            in_graph = np.ones(cluster.shape[1], dtype=bool)

        else:
            cluster_freq = np.sum(cluster, axis=0) / n_samples
            in_graph = cluster_freq >= cfg_content['min_posterior_frequency']

        locations = locations_map_crs[in_graph]
        n_graph = len(locations)

        # getting indices of points in cluster
        cluster_indices = np.argwhere(in_graph)

        # For density map: plot all edges
        if cfg_content['type'] == 'density_map':

            a = np.arange(cluster.shape[1])
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
            # count how often p0 and p1 are together in the posterior of the cluster
            p = [cluster_indices[index[0]][0], cluster_indices[index[1]][0]]
            together_in_cluster = np.sum(np.all(cluster[:, p], axis=1)) / n_samples
            lines.append(locations_map_crs[[*p]])
            line_weights.append(together_in_cluster)

        return in_graph, lines, line_weights

    def reproject_to_map_crs(self, map_proj: str) -> NDArray[float]:
        """Reproject from data CRS to map CRD"""
        data_proj = self.config['data']['projection']

        if map_proj != data_proj:
            print(f'Reprojecting locations from {data_proj} to {map_proj}.')
            loc = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*self.locations.T), crs=data_proj)
            loc_re = loc.to_crs(map_proj).geometry
            return np.array([loc_re.x, loc_re.y]).T
        else:
            return self.locations

    @staticmethod
    def initialize_map(locations, cfg_graphic, ax):

        ax.scatter(*locations.T, s=cfg_graphic['languages']['size'],
                   color=cfg_graphic['languages']['color'], linewidth=0)

    @staticmethod
    def get_cluster_colors(n_clusters: int, custom_colors=None):
        cm = plt.get_cmap('gist_rainbow')
        if custom_colors is None:
            return list(cm(np.linspace(0, 1, n_clusters, endpoint=False)))
        else:
            provided = np.array([colors.to_rgba(c) for c in custom_colors])
            additional = cm(np.linspace(0, 1, n_clusters - len(custom_colors), endpoint=False))
            return list(np.concatenate((provided, additional), axis=0))

    def add_labels(self, cfg_content, locations_map_crs, cluster_labels, cluster_colors, extent, ax):
        """Add labels to languages"""

        all_loc = list(range(locations_map_crs.shape[0]))

        # Offset
        offset_x = (extent['x_max'] - extent['x_min'])/200
        offset_y = (extent['y_max'] - extent['y_min'])/200

        # Label languages
        for i in all_loc:
            label_color = "black"
            in_cluster = False
            for j in range(len(cluster_labels)):
                if i in cluster_labels[j]:
                    # Recolor labels (only for consensus_map)
                    if cfg_content['type'] == 'consensus_map':
                        label_color = cluster_colors[j]
                    in_cluster = True

            if cfg_content['labels'] == 'in_cluster' and not in_cluster:
                pass
            else:
                self.annotate_label(xy=locations_map_crs[i], label=i + 1, color=label_color,
                                    offset_x=offset_x, offset_y=offset_y, ax=ax)

    @staticmethod
    def annotate_label(xy, label, color, offset_x, offset_y, ax):
        x = xy[0]+offset_x
        y = xy[1]+offset_y
        anno_opts = dict(xy=(x, y), fontsize=10, color=color)
        ax.annotate(label, **anno_opts)

    def visualize_clusters(self, results: Results, locations_map_crs, cfg_content, cfg_graphic, cfg_legend, ax):
        cluster_labels = []
        # If log-likelihood is displayed: add legend entries with likelihood information per cluster
        if cfg_legend['clusters']['add'] and cfg_legend['clusters']['log-likelihood']:
            cluster_labels_legend, legend_clusters = self.add_log_likelihood_legend(
                results.likelihood_single_clusters
            )

        else:
            cluster_labels_legend = []
            legend_clusters = []
            for i, _ in enumerate(results.clusters):
                cluster_labels_legend.append(f'$Z_{i + 1}$')

        # Color clusters
        if len(cfg_graphic['clusters']['color']) == 0:
            print(f'No colors for clusters provided in map>graphic>clusters>color '
                  f'in the config plot file ({self.config_file}). I am using default colors instead.')

            cluster_colors = self.get_cluster_colors(n_clusters=len(results.clusters))

        elif len(cfg_graphic['clusters']['color']) < len(results.clusters):

            print(f"Too few colors for clusters ({len(cfg_graphic['clusters']['color'])} provided, "
                  f"{len(results.clusters)} needed) in map>graphic>clusters>color in the config plot "
                  f"file ({self.config_file}). I am adding default colors.")
            cluster_colors = self.get_cluster_colors(n_clusters=len(results.clusters),
                                               custom_colors=cfg_graphic['clusters']['color'])

        else:
            cluster_colors = list(cfg_graphic['clusters']['color'])

        for i, cluster in enumerate(results.clusters):

            # This function computes a Gabriel graph for all points which are in the posterior with at least p_freq
            in_cluster, lines, line_w = self.clusters_to_graph(cluster, locations_map_crs, cfg_content)

            current_color = cluster_colors[i]

            if cfg_content['type'] == 'density_map':

                for li in range(len(lines)):
                    ax.plot(*lines[li].T, color=current_color, lw=line_w[li] * cfg_graphic['clusters']['width'],
                            alpha=line_w[li]*cfg_graphic['clusters']['alpha'])

            else:
                ax.scatter(*locations_map_crs[in_cluster].T, s=cfg_graphic['clusters']['size'], color=current_color)
                for li in range(len(lines)):
                    ax.plot(*lines[li].T, color=current_color,
                            lw=line_w[li] * cfg_graphic['clusters']['width'], alpha=cfg_graphic['clusters']['alpha'])

            # This adds small lines to the legend (one legend entry per cluster)
            line_legend = Line2D([0], [0], color=current_color, lw=6, linestyle='-')
            legend_clusters.append(line_legend)

            # Label the languages in the clusters
            if cfg_graphic['languages']['label']:

                # Use neutral color for density map labels
                if cfg_content['type'] == 'density_map':
                    current_color = "black"

                cluster_labels.append(list(compress(self.objects.indices, in_cluster)))

        if cfg_legend['clusters']['add']:
            # add to legend
            legend_clusters = ax.legend(legend_clusters, cluster_labels_legend, title_fontsize=18, title='Contact clusters',
                                     frameon=True, edgecolor='#ffffff', framealpha=1, fontsize=16, ncol=1,
                                     columnspacing=1, loc='upper left',
                                     bbox_to_anchor=cfg_legend['clusters']['position'])

            legend_clusters._legend_box.align = "left"
            ax.add_artist(legend_clusters)

        return cluster_labels, cluster_colors

    @staticmethod
    def lighten_color(color, amount=0.2):
        # FUnction to lighten up colors
        c = colorsys.rgb_to_hls(*color)
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

    # TODO: generalize to confounders, make a special case or remove
    # def color_families(self, locations_maps_crs, cfg_graphic, cfg_legend, ax):
    #     family_array = self.family_names['external']
    #     families = self.families
    #     cm = plt.get_cmap('gist_rainbow')
    #
    #     if len(cfg_graphic['families']['color']) == 0:
    #         print(f'No colors for families provided in map>graphic>families>color '
    #               f'in the config plot file ({self.config_file}). I am using default colors instead.')
    #         family_colors = cm(np.linspace(0.0, 0.8, len(self.families)))
    #
    #         # lighten colors up a bit
    #         family_colors = [self.lighten_color(c[:3]) for c in family_colors]
    #
    #     elif len(cfg_graphic['families']['color']) < len(self.families):
    #
    #         print(f"Too few colors for families ({len(cfg_graphic['families']['color'])} provided, "
    #               f"{len(self.families)} needed) in map>graphic>clusters>color in the config plot "
    #               f"file ({self.config_file}). I am adding default colors.")
    #         provided = [colors.to_rgba(c) for c in cfg_graphic['families']['color']]
    #         additional = cm(np.linspace(0, 0.8, len(self.families) - len(cfg_graphic['families']['color'])))
    #         family_colors = provided + [self.lighten_color(c[:3]) for c in additional]
    #
    #     else:
    #         family_colors = cfg_graphic['families']['color']
    #
    #     # Initialize empty legend handle
    #     handles = []
    #
    #     # Iterate over all family names
    #     for i, family in enumerate(family_array):
    #
    #         family_color = family_colors[i]
    #
    #         # Find all languages belonging to a family
    #         is_in_family = families[i] == 1
    #         family_locations = locations_maps_crs[is_in_family, :]
    #
    #         # Adds a color overlay for each language in a family
    #         ax.scatter(*family_locations.T, s=cfg_graphic['families']['size'],
    #                    color=family_color, linewidth=0, zorder=-i, label=family)
    #
    #         # For languages with more than three members combine several languages in an alpha shape (a polygon)
    #         if np.count_nonzero(is_in_family) > 3:
    #             try:
    #                 alpha_shape = self.compute_alpha_shapes(points=family_locations,
    #                                                         alpha_shape=cfg_graphic['families']['shape'])
    #
    #                 # making sure that the alpha shape is not empty
    #                 if not alpha_shape.is_empty:
    #                     smooth_shape = alpha_shape.buffer(cfg_graphic['families']['buffer'], resolution=16,
    #                                                       cap_style=1, join_style=1,
    #                                                       mitre_limit=5.0)
    #                     patch = PolygonPatch(smooth_shape, fc=family_color, ec=family_color,
    #                                          lw=1, ls='-', fill=True, zorder=-i)
    #                     ax.add_patch(patch)
    #             # When languages in the same family have identical locations, alpha shapes cannot be computed
    #             except ZeroDivisionError:
    #                 pass
    #
    #         # Add legend handle
    #         handle = Patch(facecolor=family_color, edgecolor=family_color, label=family)
    #         handles.append(handle)
    #
    #     if cfg_legend['families']['add']:
    #
    #         legend_families = ax.legend(handles=handles, title='Language family', title_fontsize=18,
    #                                     fontsize=16, frameon=True, edgecolor='#ffffff', framealpha=1,
    #                                     ncol=1, columnspacing=1, loc='upper left',
    #                                     bbox_to_anchor=cfg_legend['families']['position'])
    #         ax.add_artist(legend_families)

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
                          lw=cfg_graphic['clusters']['width'] * k)

            leg_line_width.append(line)

            # Add legend text
            line_width_label.append(f'{k:.0%}')

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
                      color=cfg_graphic['languages']['color'], linewidth=0)

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

    def add_log_likelihood_legend(self, likelihood_single_clusters: dict):

        # Legend for cluster labels
        cluster_labels = ["      log-likelihood per cluster"]

        lh_per_cluster = np.array(list(likelihood_single_clusters.values()), dtype=float)
        to_rank = np.mean(lh_per_cluster, axis=1)
        p = to_rank[np.argsort(-to_rank)]

        for i, lh in enumerate(p):
            cluster_labels.append(f'$Z_{i + 1}: \, \;\;\; {int(lh)}$')

        extra = patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        # Line2D([0], [0], color=None, lw=6, linestyle='-')

        return cluster_labels, [extra]

    def add_background_map(self, bbox, cfg_geo, cfg_graphic, ax):
        # Adds the geojson polygon geometries provided by the user as a background map

        if cfg_geo['base_map'].get('geojson_polygon') == '<DEFAULT>':
            with pkg_resources.path(maps_package, 'land.geojson') as world_path:
                world = gpd.read_file(world_path)
        else:
            world_path = fix_relative_path(cfg_geo['base_map']['geojson_polygon'], self.base_directory)
            world = gpd.read_file(world_path)

        map_crs = CRS(cfg_geo['map_projection'])

        try:
            lon_0 = map_crs.coordinate_operation.params[0].value

        except AttributeError or KeyError:
            lon_0 = 0.0
            print(f"I could not find the false origin (lon=0) of the projection {cfg_geo['map_projection']}. "
                  f"It is assumed that the false origin is 0.0. "
                  f"If plotting distorts the base map polygons, use a different projection, "
                  f"preferably one with a well-defined false origin.")

        if lon_0 > 0:
            anti_meridian = lon_0 - 180
        else:
            anti_meridian = lon_0 + 180

        offset = 10
        clip_box_1 = geometry.box(anti_meridian + 1, -90, 180 + offset, 90)
        clip_box_2 = geometry.box(anti_meridian - 1, -90, -180 + offset, 90)
        clip_box = gpd.GeoSeries([clip_box_1, clip_box_2], crs=world.crs)

        world = gpd.clip(world, clip_box)
        world = world.to_crs(cfg_geo['map_projection'])

        world = gpd.clip(world, bbox)

        cfg_polygon = cfg_graphic['base_map']['polygon']
        world.plot(ax=ax, facecolor=cfg_polygon['color'],
                   edgecolor=cfg_polygon['outline_color'],
                   lw=cfg_polygon['outline_width'],
                   zorder=-100000)

    def add_rivers(self, cfg_geo, cfg_graphic, ax):
        # The user can provide geojson line geometries, for example those for rivers. Looks good on a map :)
        if cfg_geo['base_map'].get('geojson_line', '') == '<DEFAULT>':
            with pkg_resources.path(maps_package, 'rivers_lake.geojson') as rivers_path:
                rivers = gpd.read_file(rivers_path)
        else:
            rivers_path = fix_relative_path(cfg_geo['base_map']['geojson_line'], self.base_directory)
            rivers = gpd.read_file(rivers_path)

        rivers = rivers.to_crs(cfg_geo['map_projection'])

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
                self.add_rivers(cfg_geo, cfg_graphic, ax)

    def add_correspondence_table(
            self,
            cluster_labels: typ.List[typ.List[str]],
            cluster_colors: typ.List,
            cfg_legend: dict,
            ax_c: plt.axis
    ):
        """ Which language belongs to which number? This table will tell you more"""
        plt.gca()
        sites_id = []
        sites_names = []
        sites_color = []

        for obj_id, obj_name  in zip(self.objects.indices, self.objects.names):
            label_added = False

            for s in range(len(cluster_labels)):
                if obj_id in cluster_labels[s]:
                    sites_id.append(obj_id)
                    sites_names.append(obj_name)
                    sites_color.append(cluster_colors[s])
                    label_added = True

            if not label_added:
                if cfg_legend['correspondence']['show_all']:
                    sites_id.append(obj_id)
                    sites_names.append(obj_name)
                    sites_color.append("black")

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

        widths = [0.02, 0.2] * int(((len(table_fill[0])) / 2))
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

    def posterior_map(self, results: Results, file_name='mst_posterior'):

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

        locations_map_crs = self.reproject_to_map_crs(cfg_geo['map_projection'])

        # Get extent
        extent = self.get_extent(cfg_geo, locations_map_crs)

        # Initialize the map
        self.initialize_map(locations_map_crs, cfg_graphic, ax)

        # Style the axes
        self.style_axes(extent, ax)

        # Add a base map
        self.visualize_base_map(extent, cfg_geo, cfg_graphic, ax)

        # Iterates over all clusters in the posterior and plots each with a different color
        cluster_labels, cluster_colors = self.visualize_clusters(
            results=results,
            locations_map_crs=locations_map_crs,
            cfg_content=cfg_content,
            cfg_graphic=cfg_graphic,
            cfg_legend=cfg_legend,
            ax=ax
        )

        if cfg_content['labels'] == 'all' or cfg_content['labels'] == 'in_cluster':
            self.add_labels(cfg_content, locations_map_crs, cluster_labels, cluster_colors, extent, ax)

        # Visualizes language families
        if cfg_content['plot_families']:
            logging.warning('plotting families is not currently supported.')
            # self.color_families(locations_map_crs, cfg_graphic, cfg_legend, ax)

        # Add main legend
        if cfg_legend['lines']['add']:
            self.add_legend_lines(cfg_graphic, cfg_legend, ax)

        # # Places additional legend entries on the map
        #
        # if self.legend_config['overview']['add_to_legend'] or\
        #     self.legend_config['clusters']['add_to_legend'] or\
        #         self.legend_config['families']['add_to_legend']:
        #
        #     self.add_secondary_legend()

        # This adds an overview map
        if cfg_legend['overview']['add']:

            self.add_overview_map(locations_map_crs, extent, cfg_geo, cfg_graphic, cfg_legend, ax)

        if cfg_legend['correspondence']['add'] and cfg_graphic['languages']['label']:
            if cfg_content['type'] == "density_map":
                cluster_labels = [self.objects.indices]

            if any(len(labels) > 0 for labels in cluster_labels):
                self.add_correspondence_table(cluster_labels, cluster_colors, cfg_legend, ax)

        # Save the plot
        file_format = cfg_output['format']
        fig.savefig(self.path_plots / f"{file_name}.{file_format}", bbox_inches='tight',
                    dpi=cfg_output['resolution'], format=file_format)
        plt.close(fig)

    # From general_plot.py
    ####################################
    # Probability simplex, grid plot
    ####################################
    @staticmethod
    @lru_cache(maxsize=128)
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
        if n_corners <= 2:
            raise ValueError('Can only plot polygons with >2 corners')

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
        ax.fill_between(bot_x, ymin, bot_y, color=color)
        ax.fill_between(top_x, ymax, top_y, color=color)

    # Transform weights into needed format
    # def transform_weights(self, feature, b_in):
    #
    #     universal_array = []
    #     contact_array = []
    #     inheritance_array = []
    #     sample_dict = self.results['weights']
    #     for key in sample_dict:
    #         split_key = key.split("_")
    #         if 'w' == split_key[0]:
    #             if 'universal' == split_key[1] and str(feature) == split_key[2]:
    #                 universal_array = sample_dict[key][b_in:]
    #             elif 'contact' == split_key[1] and str(feature) == split_key[2]:
    #                 contact_array = sample_dict[key][b_in:]
    #             elif 'inheritance' == split_key[1] and str(feature) == split_key[2]:
    #                 inheritance_array = sample_dict[key][b_in:]
    #
    #     sample = np.column_stack([universal_array, contact_array, inheritance_array]).astype(np.float)
    #     # Shape: n_samples * 3
    #     return sample
    #
    # def transform_probability_vectors(self, feature, parameter, b_in):
    #
    #     if "alpha" in parameter:
    #         sample_dict = self.results['alpha']
    #     elif "beta" in parameter:
    #         sample_dict = self.results['beta']
    #     elif "gamma" in parameter:
    #         sample_dict = self.results['gamma']
    #     else:
    #         raise ValueError("parameter must be alpha, beta or gamma")
    #
    #     p_dict = {}
    #     states = []
    #
    #     for key in sample_dict:
    #         if key.startswith(parameter + '_' + feature + '_'):
    #             state = str(key).rsplit('_', 1)[1]
    #             p_dict[state] = sample_dict[key][b_in:]
    #             states.append(state)
    #
    #     sample = np.column_stack([p_dict[s] for s in p_dict]).astype(np.float)
    #     return sample, states
    #
    # # Get preferences or weights from relevant features
    # def get_parameters(self, b_in, features, parameter="weights"):
    #
    #     par = {}
    #     states = {}
    #     # if features is empty, get parameters for all features
    #     if not features:
    #         feature_names = self.current_results.feature_names
    #     else:
    #         feature_names = list(self.current_results.feature_names[i-1] for i in features)
    #
    #     # get samples
    #     for f in feature_names:
    #
    #         if parameter == "weights":
    #             # p = self.transform_weights(feature=feat_name, b_in=b_in)
    #             p = self.current_results.weights[f]
    #             par[f] = p
    #
    #         elif "alpha" in parameter or "beta" in parameter or "gamma" in parameter:
    #             p, state = self.transform_probability_vectors(feature=f, parameter=parameter, b_in=b_in)
    #
    #             par[f] = p
    #             states[f] = state
    #
    #     return par, states
    #
    # def sort_by_weights(self, w):
    #     sort_by = {}
    #     for f in self.current_results.feature_names:
    #         sort_by[f] = median(w[f][:, 1])
    #     ordering = sorted(sort_by, key=sort_by.get, reverse=True)
    #     return ordering


    @staticmethod
    def plot_weight(
        samples: NDArray[float],
        feature: str,
        cfg_legend: dict,
        ax: plt.Axes | None = None,
        mean_weights: bool = False,
        plot_samples: bool = False,
        lw: float | None = None,
    ):
        """Plot a set of weight vectors in a 2D representation of the probability simplex.

        Args:
            samples: Sampled weight vectors to plot.
            feature: Name of the feature for which weights are being plotted
            ax: The pyplot axis.
            cfg_legend: legend info from the config plot file
            mean_weights: Plot the mean of the weights?
            plot_samples: Add a scatter plot overlay of the actual samples?
            lw: Line width of the triangular border delineating the probability simplex.
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

        # Density and scatter plot
        title = cfg_legend['title']
        if title['add']:
            # ax.set_title(str(feature), pad=15,
            #              fontdict={'fontweight': 'bold', 'fontsize': title['font_size']})
            ax.text(
                title['position'][0], title['position'][1], str(feature),
                fontdict={'fontweight': 'bold', 'fontsize': title['font_size']},
                transform=ax.transAxes
            )

        x = samples_projected.T[0]
        y = samples_projected.T[1]

        sns.kdeplot(x=x, y=y, shade=True,  cut=30, n_levels=20,
                    clip=([xmin, xmax], [ymin, ymax]), cmap=Plot.pref_color_map, ax=ax)

        if plot_samples:
            ax.scatter(x, y, color='k', lw=0, s=1, alpha=0.2)

        # Draw simplex and crop outside
        ax.fill(*corners.T, edgecolor='k', fill=False, lw=lw)
        Plot.fill_outside(corners, color='w', ax=ax)

        if mean_weights:
            mean_projected = np.mean(samples, axis=0).dot(corners)
            ax.scatter(*mean_projected.T, color="#ed1696", lw=0, s=50, marker="o")

        labels = cfg_legend['labels']

        if labels['add']:
            for xy, label in zip(corners, labels['names']):
                xy = xy*1.15 - 0.05  # Stretch, s.t. labels don't overlap with corners
                ax.text(*xy, label, ha='center', va='center', fontdict={'fontsize': labels['font_size']})

        ax.set_xlim([xmin - 0.1, xmax + 0.1])
        ax.set_ylim([ymin - 0.1, ymax + 0.1])
        ax.axis('off')

    @staticmethod
    def plot_preference(samples, feature, cfg_legend, label_names, ax=None, plot_samples=False, color=None):
        """Plot a set of weight vectors in a 2D representation of the probability simplex.

        Args:
            samples (np.array): Sampled weight vectors to plot.
            feature (str): Name of the feature for which weights are being plotted
            cfg_legend(dict): legend info from the config plot file
            label_names(list): labels for each corner
            ax (plt.Axis): The pyplot axis.
            plot_samples (bool): Add a scatter plot overlay of the actual samples?
        """
        if ax is None:
            ax = plt.gca()
        if color is None:
            color = 'g'

        n_samples, n_p = samples.shape
        # color map
        title = cfg_legend['title']
        labels = cfg_legend['labels']

        if n_p == 2:
            x = samples.T[1]
            sns.kdeplot(x, color=color, ax=ax, fill=True, lw=1, clip=(0, 1))
            # if cfg_legend['rug']:     # TODO Make the rug plot option?
            #     sns.rugplot(x, color="k", alpha=0.02, height=-0.03, ax=ax, clip_on=False)

            ax.axes.get_yaxis().set_visible(False)

            if labels['add']:
                for x, label in enumerate(label_names):
                    if x == 0:
                        x = 0.05
                    if x == 1:
                        x = 0.95
                    ax.text(x, -0.05, label, ha='center', va='top',
                             fontdict={'fontsize': labels['font_size']}, transform=ax.transAxes)
            if title['add']:
                ax.text(title['position'][0], title['position'][1],
                         str(feature), fontsize=title['font_size'], fontweight='bold', transform=ax.transAxes)

            ax.plot([0, 1], [0, 0], lw=1, color=color, clip_on=False)

            ax.set_ylim([0, None])
            ax.set_xlim([-0.01, 1.01])

            ax.axis('off')

        elif n_p > 2:
            # Compute corners
            corners = Plot.get_corner_points(n_p)
            # Bounding box

            xmin, ymin = np.min(corners, axis=0)
            xmax, ymax = np.max(corners, axis=0)

            # Project the samples
            samples_projected = samples.dot(corners)

            # Density and scatter plot
            if title['add']:
                ax.text(title['position'][0], title['position'][1],
                         str(feature), fontsize=title['font_size'], fontweight='bold', transform=ax.transAxes)

            x = samples_projected.T[0]
            y = samples_projected.T[1]

            sns.kdeplot(x=x, y=y, shade=True, thresh=0, cut=30, n_levels=100, ax=ax,
                        clip=([xmin, xmax], [ymin, ymax]), cmap=Plot.pref_color_map)

            if plot_samples:
                ax.scatter(x, y, color='k', lw=0, s=1, alpha=0.05)

            # Draw simplex and crop outside
            ax.fill(*corners.T, edgecolor='k', fill=False)
            Plot.fill_outside(corners, color='w', ax=ax)

            if labels['add']:
                for xy, label in zip(corners, label_names):
                    xy *= 1.2  # Stretch, s.t. labels don't overlap with corners
                    # Split long labels
                    if (" " in label or "-" in label) and len(label) > 10:
                        white_or_dash = [i for i, ltr in enumerate(label) if (ltr == " " or ltr == "-")]
                        mid_point = len(label)/2
                        break_label = min(white_or_dash, key=lambda x: abs(x - mid_point))
                        label = label[:break_label] + "\n" + label[break_label:]

                    ax.text(*xy, label, ha='center', va='center',
                             fontdict={'fontsize': labels['font_size']})

            ax.set_xlim([xmin - 0.1, xmax + 0.1])
            ax.set_ylim([ymin - 0.1, ymax + 0.1])
            ax.axis('off')

    def plot_weights(self, results: Results, file_name: PathLike):
        print('Plotting weights...')

        cfg_weights = self.config['weight_plot']
        feature_subset = cfg_weights['content']['features']
        weights = results.weights
        if feature_subset:
            weights = {f: weights[f] for f in feature_subset}

        features = weights.keys()
        n_plots = len(features)
        n_col = cfg_weights['output']['n_columns']
        n_row = math.ceil(n_plots / n_col)

        width = cfg_weights['output']['width_subplot']
        height = cfg_weights['output']['height_subplot']
        fig, axs = plt.subplots(n_row, n_col, figsize=(width * n_col, height * n_row))

        position = 1
        n_empty = n_row * n_col - n_plots

        for e in range(1, n_empty + 1):
            axs.flatten()[-e].axis('off')

        for f in features:
            plt.subplot(n_row, n_col, position)
            self.plot_weight(weights[f], feature=f, cfg_legend=cfg_weights['legend'], mean_weights=True)
            print(position, "of", n_plots, "plots finished")
            position += 1

        # todo: spacing in config?
        width_spacing = 0.1
        height_spacing = 0.1
        plt.subplots_adjust(wspace=width_spacing,
                            hspace=height_spacing)
        file_format = cfg_weights['output']['format']
        resolution = cfg_weights['output']['resolution']

        fig.savefig(self.path_plots / f'{file_name}.{file_format}', bbox_inches='tight',
                    dpi=resolution, format=file_format)
        plt.close(fig)

    def plot_weights_and_prefs(
        self,
        results: Results,
        feature_name: str,
        # file_name: PathLike,
    ):
        n_components = 1 + results.n_confounders
        max_groups = max(len(groups) for groups in results.groups_by_confounders.values())
        fig, axes = plt.subplots(nrows=n_components, ncols=2 + max_groups,
                                 figsize=(4 + max_groups, 2 + n_components),
                                 gridspec_kw={'width_ratios': [1.8, .8] + [1]*max_groups})

        plt.tight_layout()
        axes[0, 0].text(
            0, 0, f"feature {feature_name}",
            fontsize=10, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round, pad=1.0, rounding_size=0.5', facecolor='#eeeeee', lw=0.8, edgecolor='k')
        )
        axes[0, 0].set_xlim([-1, 1])
        axes[0, 0].set_ylim([-2, 3])

        for ax in axes.flatten():
            ax.axis('off')

        self.plot_weight(
            samples=results.weights[feature_name],
            # feature='weights',
            feature='',
            cfg_legend=self.config['feature_plot']['legend'],
            mean_weights=True,
            ax=axes[n_components // 2, 0],
            lw=.8,
        )

        preferences = {
            **results.confounding_effects,
            'cluster': results.areal_effect,
        }

        for i, (component, prefs_by_group) in enumerate(preferences.items()):
            axes[i, 1].text(0, 0,
                            component.replace('cluster', 'contact').replace('family', 'inheritance'),
                            fontsize=8)
            axes[i, 1].set_ylim([-2, 3])
            for j, (group, pref_by_feat) in enumerate(prefs_by_group.items()):
                axes[i, j + 2].get_shared_y_axes().join(axes[i, 2], axes[i, j + 2])

                if component == "cluster":
                    group = "Area " + group[1:]
                self.plot_preference(
                    pref_by_feat[feature_name],
                    feature='' if group == '<ALL>' else group,
                    label_names=results.get_states_for_feature_name(feature_name),
                    cfg_legend=self.config['feature_plot']['legend'],
                    ax=axes[i, j + 2],
                    color='#005570',
                )



        plt.show()

    # This is not changed yet
    def plot_preferences(self, results: Results, file_name: str):
        """Creates preference plots for universal, clusters and families

       Args:
           results: the results from a sbayes run.
           file_name: name of the output file
       """
        print('Plotting preferences...')
        cfg_preference = self.config['preference_plot']
        # burn_in = int(len(self.results['posterior']) * cfg_preference['content']['burn_in'])

        width = cfg_preference['output']['width_subplot']
        height = cfg_preference['output']['height_subplot']

        # todo: spacing in config?
        width_spacing = 0.2
        height_spacing = 0.2

        n_plots = results.n_features
        n_col = cfg_preference['output']['n_columns']
        n_row = math.ceil(n_plots / n_col)

        file_format = cfg_preference['output']['format']
        resolution = cfg_preference['output']['resolution']

        # Combine all preferences into one dictionary of structure
        #   {component: {feature: state_probabilities}}
        # where `component` is a cluster or a confounder group.
        preferences = {**results.areal_effect}
        for conf_name, conf_effect in results.confounding_effects.items():
            for group, preference in conf_effect.items():
                preferences[f'{conf_name}_{group}'] = preference

        # Only show the specified list of preferences, if present in the config
        which_prefs = cfg_preference['content']['preference']
        if which_prefs:
            preferences = {k: v for k, v in preferences.items() if k in which_prefs}

        # Plot each preference in a separate plot
        for component, pref_by_feat in preferences.items():
            fig, axs = plt.subplots(n_row, n_col, figsize=(width * n_col, height * n_row), )
            n_empty = n_row * n_col - n_plots

            for e in range(1, n_empty + 1):
                axs.flatten()[-e].axis('off')

            position = 1

            for f, pref in pref_by_feat.items():
                plt.subplot(n_row, n_col, position)

                states = results.get_states_for_feature_name(f)
                self.plot_preference(pref, feature=f, label_names=states,
                                     cfg_legend=cfg_preference['legend'])

                print(component, ": ", position, "of", n_plots, "plots finished")
                position += 1

            plt.subplots_adjust(wspace=width_spacing, hspace=height_spacing)
            fig.savefig(self.path_plots / f'{file_name}_{component}.{file_format}',
                        bbox_inches='tight', dpi=resolution, format=file_format)
            plt.close(fig)

    def plot_dic(self, models: dict, file_name: str):
        """This function plots the dics. What did you think?
        Args:
            models: A dict of different models for which the DIC is evaluated
            file_name: name of the output file
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

        fig.savefig(self.path_plots / f'{file_name}.{file_format}', bbox_inches='tight',
                    dpi=resolution, format=file_format)

    def plot_trace(self, results: Results, file_name="trace", show_every_k_sample=1, file_format="pdf"):
        """
        Function to plot the trace of a parameter
        Args:
            results: the results from a sbayes run.
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
            y = results['recall'][::show_every_k_sample]
            y2 = results['precision'][::show_every_k_sample]
            x = results.sample_id[::show_every_k_sample]
            ax.plot(x, y, lw=0.5, color=self.config['plot_trace']['color'][0], label="recall")
            ax.plot(x, y2, lw=0.5, color=self.config['plot_trace']['color'][1], label="precision")
            y_min = 0
            y_max = 1

        else:
            try:
                y = results[parameter][::show_every_k_sample]

            except KeyError:
                raise ValueError("Cannot compute trace. " + self.config['plot_trace']['parameter']
                                 + " is not a valid parameter.")

            x = results.sample_id[::show_every_k_sample]
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

    def plot_trace_lh_prior(self, results: Results, burn_in=0.2, fname="trace",
                            show_every_k_sample=1, lh_lim=None, prior_lim=None):
        fig, ax1 = plt.subplots(figsize=(10, 8))

        lh = results.likelihood[::show_every_k_sample]
        prior = results.prior[::show_every_k_sample]
        x = results.sample_id[::show_every_k_sample]

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
        x_ticklabels = [f'{x_ticklabel:.0f} clusters' for x_ticklabel in np.linspace(1, n_models, n_models)]
        x_ticklabels[0] = '1 cluster'
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

    def plot_pies(self, results: Results, file_name: PathLike):

        print('Plotting pie charts ...')
        cfg_pie = self.config['pie_plot']

        clusters = np.array(results.clusters)
        n_clusters = clusters.shape[0]
        end_bi = math.ceil(clusters.shape[1] * cfg_pie['content']['burn_in'])

        samples = clusters[:, end_bi:, ]

        n_plots = samples.shape[2]
        n_samples = samples.shape[1]

        samples_per_cluster = np.sum(samples, axis=1)

        # Grid
        n_col = cfg_pie['output']['n_columns']
        n_row = math.ceil(n_plots / n_col)

        width = cfg_pie['output']['width']
        height = cfg_pie['output']['height']

        fig, axs = plt.subplots(n_row, n_col, figsize=(width * n_col, height * n_row))

        if len(self.config['map']['graphic']['clusters']['color']) == 0:
            print(f'I tried to color the pie charts the same as the map, but no colors were provided in '
                  f'map>graphic>clusters>color in the config plot file ({self.config_file}). '
                  f'I am using default colors instead.')

            all_colors = self.get_cluster_colors(n_clusters=n_clusters)

        elif len(self.config['map']['graphic']['clusters']['color']) < n_clusters:

            print(f'I tried to color the pie charts the same as the map, but not enough colors were provided in '
                  f'map>graphic>clusters>color in the config plot file ({self.config_file}). '
                  f'I am adding default colors.')
            all_colors = self.get_cluster_colors(n_clusters, custom_colors=self.config['map']['graphic']['clusters']['color'])
        else:
            print(f'I am using the colors in map>graphic>clusters>color '
                  f'in the config plot file ({self.config_file}) to color the pie charts.')
            all_colors = list(self.config['map']['graphic']['clusters']['color'])

        for l in range(n_col*n_row):

            ax_col = int(np.floor(l/n_row))
            ax_row = l - n_row * ax_col

            if l < n_plots:
                per_lang = samples_per_cluster[:, l]
                no_cluster = n_samples - per_lang.sum()

                x = per_lang.tolist()
                col = all_colors[:len(x)]

                # Append samples that are not in an cluster
                x.append(no_cluster)
                col.append("lightgrey")

                axs[ax_row, ax_col].pie(x, colors=col, radius=15)

                label = str(self.objects['names'][l])

                # break long labels
                if (" " in label or "-" in label) and len(label) > 10:
                    white_or_dash = [i for i, ltr in enumerate(label) if (ltr == " " or ltr == "-")]
                    mid_point = len(label) / 2
                    break_label = min(white_or_dash, key=lambda x: abs(x - mid_point))
                    if " " in label:
                        label = label[:break_label] + '\n' + label[break_label+1:]
                    elif "-" in label:
                        label = label[:break_label] + '-\n' + label[break_label+1:]

                axs[ax_row, ax_col].text(0.20, 0.5, str(self.objects.indices[l] + 1), size=15, va='center', ha="right",
                                         transform=axs[ax_row, ax_col].transAxes)

                axs[ax_row, ax_col].text(0.25, 0.5, label, size=15, va='center', ha="left",
                                         transform=axs[ax_row, ax_col].transAxes)
                axs[ax_row, ax_col].set_xlim([0, 160])
                axs[ax_row, ax_col].set_ylim([-10, 10])

            axs[ax_row, ax_col].axis('off')

        # Style remaining empty cells in the plot
        n_empty = n_row * n_col - n_plots
        for e in range(1, n_empty):
            axs[-1, -e].axis('off')

        plt.subplots_adjust(wspace=cfg_pie['output']['spacing_horizontal'],
                            hspace=cfg_pie['output']['spacing_vertical'])

        file_format = cfg_pie['output']['format']
        fig.savefig(self.path_plots / f'{file_name}.{file_format}', bbox_inches='tight',
                    dpi=cfg_pie['output']['resolution'], format=file_format)
        plt.close(fig)


class PlotType(Enum):
    map = 'map'
    weights_plot = 'weights_plot'
    preference_plot = 'preference_plot'
    pie_plot = 'pie_plot'
    feature_plot = 'feature_plot'
    dic_plot = 'dic_plot'

    @classmethod
    def values(cls) -> list[str]:
        return [str(e.value) for e in cls]


def main(config, plot_types: list[PlotType] = None, args: Namespace = None):
    # TODO adapt paths according to experiment_name (if provided)
    # If no plot type is specified, plot everything in the config file

    if plot_types is None:
        plot_types = list(PlotType)

    plot = Plot()
    plot.load_config(config_file=config)
    plot.read_data()

    def should_be_plotted(plot_type: PlotType):
        """A plot type should only be generated if it
            1) is specified in the config file and
            2) is in the requested list of plot types."""
        return (plot_type.value in plot.config) and (plot_type in plot_types)

    for m, results in plot.iterate_over_models():
        print('Plotting model', m)

        # Plot the reconstructed clusters on a map
        if should_be_plotted(PlotType.map):
            plot_map(plot, results, m)

        # Plot the reconstructed mixture weights in simplex plots
        if should_be_plotted(PlotType.weights_plot):
            plot.plot_weights(results, file_name='weights_grid_' + m)

        # Plot the reconstructed probability vectors in simplex plots
        if should_be_plotted(PlotType.preference_plot):
            plot.plot_preferences(results, file_name=f'prob_grid_{m}')

        # Plot the reconstructed clusters in pie-charts
        # (one per language, showing how likely the language is to be in each cluster)
        if should_be_plotted(PlotType.pie_plot):
            plot.plot_pies(results, file_name= 'plot_pies_' + m)

        # if should_be_plotted(PlotType.feature_plot):
        if should_be_plotted(PlotType.feature_plot):
            if args.feature_name is None:
                logging.warning("Skipping 'feature_plot', since not feature_name was provided.")
                # TODO: If feature_name is None, iterate over all features and save to file.
            else:
                plot.plot_weights_and_prefs(results, args.feature_name)

    # Plot DIC over all models
    if should_be_plotted(PlotType.dic_plot):
        plot.plot_dic(plot.results, file_name='dic')


def plot_map(plot: Plot, results: Results, m: str):
    config_map = plot.config['map']
    map_type = config_map['content']['type']

    if map_type == config_map['content']['type'] == 'density_map':
        plot.posterior_map(results, file_name='posterior_map_' + m)

    elif map_type == config_map['content']['type'] == 'consensus_map':
        min_posterior_frequency = config_map['content']['min_posterior_frequency']
        if min_posterior_frequency in [tuple, list, set]:
            for mpf in min_posterior_frequency:
                config_map['content'].__setitem__('min_posterior_frequency', mpf)
                plot.posterior_map(results, file_name=f'posterior_map_{m}_{mpf}')
        else:
            plot.posterior_map(results, file_name=f'posterior_map_{m}_{min_posterior_frequency}')
    else:
        raise ValueError(f'Unknown map type: {map_type}  (in the config file "map" -> "content" -> "type")')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot the results of a sBayes run.')
    parser.add_argument('config', type=Path, help='The JSON configuration file')
    parser.add_argument('type', nargs='?', type=str, help='The type of plot to generate')
    parser.add_argument('feature_name', nargs='?', type=str, help='The feature to show in a `feature_plot`')
    args = parser.parse_args()

    plot_types = None
    if args.type is not None:
        if args.type not in PlotType.values():
            raise ValueError(f"Unknown plot type: '{args.type}'. Choose from {PlotType.values()}.")
        plot_types = [PlotType(args.type)]

    main(args.config, plot_types=plot_types, args=args)
