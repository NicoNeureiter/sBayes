""" Class Map

Inherits basic functions from Plot
Defines specific functions for map plots
"""

import math
import warnings
from copy import deepcopy
from itertools import compress

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from descartes import PolygonPatch
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay
from scipy.special import logsumexp
from shapely import geometry
from shapely.ops import cascaded_union, polygonize

from sbayes.plotting.plot_setup import Plot
from sbayes.util import add_edge, compute_delaunay
from sbayes.util import round_int
from sbayes.util import gabriel_graph_from_delaunay

warnings.simplefilter(action='ignore', category=FutureWarning)


class Map(Plot):

    def __init__(self, simulated_data=False):

        # Load init function from the parent class Plot
        super().__init__(simulated_data=simulated_data)

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
    def load_subset(self):
        # Get sites in subset
        is_in_subset = [x == 1 for x in self.sites['subset']]
        sites_all = deepcopy(self.sites)
        # Get the relevant information for all sites in the subset (ids, names, ..)
        for key in self.sites.keys():
            if type(self.sites[key]) == list:
                self.sites[key] = list(np.array(self.sites[key])[is_in_subset])
            else:
                self.sites[key] = self.sites[key][is_in_subset, :]
        return is_in_subset, sites_all

    # Add subset
    def add_subset(self):

        is_in_subset, sites_all = self.load_subset()

        # Again only for this one experiment with a subset
        # could go to subset function. Low priority.
        # plot all points not in the subset in light grey
        not_in_subset = np.logical_not(is_in_subset)
        other_locations = sites_all['locations'][not_in_subset]
        self.ax.scatter(*other_locations.T, s=self.config['graphic']['size'], c="darkgrey", alpha=1, linewidth=0)

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
        # This is only valid for one single experiment
        # where we perform the analysis on a biased subset.
        # The following lines of code selects only those sites which are in the subset
        # Low priority: could go to a separate function
        if self.config['input']['subset']:
            self.add_subset()

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

        self.fig.savefig(f"{self.path_plots + fname}.{self.map_parameters['save_format']}", bbox_inches='tight', dpi=400, format=self.map_parameters['save_format'])

        # Should the labels displayed in the map be returned? These are later added as a separate legend (
        # outside this hell of a function)
        if return_correspondence and label_languages:
            return self.all_labels
