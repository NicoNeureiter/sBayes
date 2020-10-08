""" Class Map

Inherits basic functions from Plot
Defines specific functions for map plots
"""
import json
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

from scipy.spatial import Delaunay
from shapely import geometry
from shapely.ops import cascaded_union, polygonize

from sbayes.plotting.plot_setup import Plot
from sbayes.util import (add_edge, fix_default_config,
                         gabriel_graph_from_delaunay,
                         compute_delaunay)

warnings.simplefilter(action='ignore', category=FutureWarning)


class Map(Plot):

    def __init__(self, simulated_data=False):

        # Load init function from the parent class Plot
        super().__init__(simulated_data=simulated_data)

        self.base_directory = None

        # Load default config
        self.config_default = "config/plotting/config_plot_maps.json"

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

            alpha_shape = self.compute_alpha_shapes(sites = [is_in_area],
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

        area_freq = np.sum(area, axis=0)/n_samples
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
            graph_connections = np.array([[0, 1], [1, 2], [2, 0]]).astype(int)

        elif n_graph == 2:
            graph_connections = np.array([[0, 1]]).astype(int)

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
            x += 20000
            y += 10000
            # Same with the font size for annotations. Should probably go to the config.
            anno_opts = dict(xy=(x, y), fontsize=14, color=current_color)
            self.ax.annotate(labels_in_area[loc] + 1, **anno_opts)

    # Bind together the functions above
    def visualize_areas(self):

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
                self.ax.plot(*lines[li].T, color=current_color, lw=line_w[li]*self.graphic_config['line_width'],
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
            # x_extend_overview and y_extend_overview --> to config
            axins.set_xlim(self.legend_config['overview']['x_extend'])
            axins.set_ylim(self.legend_config['overview']['y_extend'])

            # Again, this function needs map data to display in the overview map.
            self.add_background_map(axins)

            # add overview to the map
            axins.scatter(*self.locations.T, s=self.graphic_config['point_size'] / 2, c="darkgrey",
                          alpha=1, linewidth=0)

            # adds a bounding box around the overview map
            bbox_width = self.geo_config['x_extend'][1] - self.geo_config['x_extend'][0]
            bbox_height = self.geo_config['y_extend'][1] - self.geo_config['y_extend'][0]
            bbox = mpl.patches.Rectangle((self.geo_config['x_extend'][0], self.geo_config['y_extend'][0]),
                                         bbox_width, bbox_height, ec='k', fill=False, linestyle='-')
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
        if self.geo_config['proj4'] is None or self.geo_config['base_map']['geojson_map'] is None:
            raise Exception('If you want to use a map, provide a geojson and a crs!')

        # Adds the geojson map provided by user as background map
        self.world = gpd.read_file(self.geo_config['base_map']['geojson_map'])
        self.world = self.world.to_crs(self.geo_config['proj4'])
        #self.world = gpd.clip(self.world, self.bbox)
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

    def return_correspondence_table(self, file_name, file_format="pdf", ncol=3):
        """ Which language belongs to which number? This table will tell you more
        Args:
            file_name (str): name of the plot
            file_format (str): format of the output file
            ncol(int): number of columns in the output table
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
        if self.legend_config['overview']['add_overview']:
            self.add_overview_map()

        # Visualizes language families
        if self.content_config['plot_families']:
            self.color_families()

        # Modify the legend for simulated data
        if self.is_simulation:
            self.modify_legend()

        # Save the plot
        self.fig.savefig(f"{self.path_plots + '/'+ file_name}.{file_format}",
                         bbox_inches='tight', dpi=400, format=file_format)

        if return_correspondence and self.content_config['label_languages']:
            self.return_correspondence_table(file_name=file_name)

        plt.close(self.fig)
