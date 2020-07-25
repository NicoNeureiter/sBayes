""" Class Map

Inherits basic functions from Plot
Defines specific functions for map plots
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from copy import deepcopy
from descartes import PolygonPatch
import geopandas as gpd
from itertools import compress
import math
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
from matplotlib import patches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay
from scipy.special import logsumexp
from shapely import geometry
from shapely.ops import cascaded_union, polygonize


from sbayes.preprocessing import compute_network
from sbayes.plotting.plot import Plot
from sbayes.util import zones_autosimilarity, add_edge, compute_delaunay, colorline # compute_mst_posterior
from sbayes.util import bounding_box, round_int, linear_rescale, round_single_int, round_multiple_ints



class Map(Plot):

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

        pp = self.get_plotting_params()
        # getting axes ranges and rounding them
        x_min, x_max = np.min(locations[:, 0]), np.max(locations[:, 0])
        y_min, y_max = np.min(locations[:, 1]), np.max(locations[:, 1])

        # if specific offsets were passes use them, otherwise use same offset for all
        if offset is not None:
            x_min, x_max = round_int(x_min, 'down', offset), round_int(x_max, 'up', offset)
            y_min, y_max = round_int(y_min, 'down', offset), round_int(y_max, 'up', offset)
        elif x_extend is not None and y_extend is not None:
            x_min, x_max = x_extend
            y_min, y_max = y_extend
        else:
            x_coords, y_coords = locations.T
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

        # setting axes limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        # x axis
        x_step = (x_max - x_min) // 5
        x_ticks = np.arange(x_min, x_max + x_step, x_step) if show else []
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, fontsize=pp['fontsize'])

        # y axis
        y_step = (y_max - y_min) // 5
        y_ticks = np.arange(y_min, y_max + y_step, y_step) if show else []
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, fontsize=pp['fontsize'])

        return (x_min, x_max, y_min, y_max)

    def compute_alpha_shapes(self, sites, net, alpha):

        """Compute the alpha shape (concave hull) of a set of sites
        Args:
            sites (np.array): subset of sites around which to create the alpha shapes (e.g. family, zone, ...)
            net (dict): The full network containing all sites.
            alpha (float): alpha value to influence the gooeyness of the convex hull Smaller numbers don't fall inward
            as much as larger numbers. Too large, and you lose everything!"

        Returns:
            (polygon): the alpha shape"""

        all_sites = net['locations']
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


    def add_zone_boundary(self, ax, locations, net, is_in_zone, alpha, annotation=None, color='#000000'):
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
        color_zones = '#000000'

        cp_locations = locations[is_in_zone[0], :]

        leg_zone = None
        if cp_locations.shape[0] > 0:  # at least one contact point in zone

            alpha_shape = self.compute_alpha_shapes([is_in_zone], net, alpha)

            # smooth_shape = alpha_shape.buffer(100, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
            smooth_shape = alpha_shape.buffer(60, resolution=16, cap_style=1, join_style=1, mitre_limit=10.0)
            # smooth_shape = alpha_shape
            patch = PolygonPatch(smooth_shape, ec=color, lw=1, ls='-', alpha=1, fill=False,
                                 zorder=1)
            leg_zone = ax.add_patch(patch)
        else:
            print('computation of bbox not possible because no contact points')

        # only adding a label (numeric) if annotation turned on and more than one zone
        if annotation is not None:
            x_coords, y_coords = cp_locations.T
            x, y = np.mean(x_coords), np.mean(y_coords)
            ax.text(x, y, annotation, fontsize=fontsize, color=color)

        return leg_zone


    def add_minimum_spanning_tree_new(self, ax, zone, locations, dist_mat, burn_in, post_freq_line, size=25,
                                      size_line=3, color='#ff0000'):

        # exclude burn-in
        end_bi = math.ceil(len(zone) * burn_in)
        zone = zone[end_bi:]

        # get number of samples (excluding burn-in) and number of points
        n_samples = len(zone)
        n_points = len(locations)

        # container to count edge weight for mst
        mst_posterior_freq = np.zeros((n_points, n_points), np.float64)

        # container for points in contact zone
        points_posterior_freq = np.zeros((n_points,), np.float64)

        # compute mst for each sample and update mst posterior
        for sample in zone:

            points_posterior_freq = points_posterior_freq + sample

            # getting indices of contact points
            cp_indices = np.argwhere(sample)

            # subsetting to contact points
            cp_locations = locations[sample, :]
            cp_dist_mat = dist_mat[sample, :][:, sample]
            n_contact_points = len(cp_locations)

            if n_contact_points > 3:
                # computing the minimum spanning tree of contact points
                cp_delaunay = compute_delaunay(cp_locations)
                cp_mst = minimum_spanning_tree(cp_delaunay.multiply(cp_dist_mat))

                # converting minimum spanning tree to boolean array denoting whether contact points are connected
                cp_mst = cp_mst.toarray()
                cp_connections = cp_mst > 0

                for index, connected in np.ndenumerate(cp_connections):
                    if connected:
                        # getting indices of contact points
                        i1, i2 = cp_indices[index[0]], cp_indices[index[1]]
                        mst_posterior_freq[i1, i2] = mst_posterior_freq[i1, i2] + 1
                        mst_posterior_freq[i2, i1] = mst_posterior_freq[i2, i1] + 1

            # compute delaunay only works for n points > 3 but mst can be computed for 3 and 2 points
            elif n_contact_points == 3:
                connected = []
                dist_matrix = np.linalg.norm(cp_locations - cp_locations[:, None], axis=-1)
                if dist_matrix[0, 1] < dist_matrix[0, 2]:
                    connected.append([0, 1])
                    if dist_matrix[2, 0] < dist_matrix[2, 1]:
                        connected.append([2, 0])
                    else:
                        connected.append([2, 1])
                else:
                    connected.append([0, 2])
                    if dist_matrix[1, 0] < dist_matrix[1, 2]:
                        connected.append([1, 0])
                    else:
                        connected.append([1, 2])
                # replace indices from subset with indice from all points
                # print(cp_indices)
                # connected = [[cp_indices[i][0], cp_indices[j][0]] for i, j in connected]
                for i, j in connected:
                    i, j = cp_indices[i][0], cp_indices[j][0]
                    mst_posterior_freq[i, j] = mst_posterior_freq[i, j] + 1
                    mst_posterior_freq[j, i] = mst_posterior_freq[j, i] + 1

                # print(connected)

            # for 2 contact points the two points are simply connected with a line
            elif n_contact_points == 2:
                print(cp_indices)
            # for 1 point don't do anything
            elif n_contact_points == 1:
                pass
            # if there aren't any contact points in the zone print a warning
            else:
                print('Warning: No points in contact zone!')
            # end of populating mst posterior

        # converting absolute counts to frequencies
        mst_posterior_freq = mst_posterior_freq / n_samples
        points_posterior_freq = points_posterior_freq / n_samples

        post_freq_line.sort(reverse=True)
        for index, freq in np.ndenumerate(mst_posterior_freq):
            if freq >= min(post_freq_line):

                # Assign each line in the posterior MST to one class of line thickness
                for k in post_freq_line:
                    if freq >= k:
                        lw = size_line * k

                        # compute line width
                        # min_lw = size / 100
                        # freq_norm = (freq - ts_posterior_freq) / (1 - ts_posterior_freq)
                        # lw = min_lw + freq_norm * size_factor

                        # getting locations of the two contact points and plotting line
                        locs = locations[[*index]]
                        ax.plot(*locs.T, color=color, lw=lw)
                        pass

        # getting boolean array of points in minimum spanning tree
        is_in_mst = points_posterior_freq > min(post_freq_line)

        # plotting points of minimum spanning tree
        cp_locations = locations[is_in_mst, :]
        ax.scatter(*cp_locations.T, s=size, c=color)

        return is_in_mst

    def get_cmap(self, ts_lf, name='YlOrRd', lower_ts=0.2):
        """ Function to generate a colormap
        Args:
            ts_lf (float): Anything below this frequency threshold is grey.
            name (string): Name of the colormap.
            lower_ts (float): Threshold to manipulate colormap.
        Returns:
            (LinearSegmentedColormap): Colormap
            (Normalize): Object for normalization of frequencies
        """
        grey_tone = 128  # from 0 (dark) to 1 (white)
        lf_color = (grey_tone / 256, grey_tone / 256, grey_tone / 256)  # grey 128
        colors = [lf_color, (256 / 256, 256 / 256, 0 / 256), (256 / 256, 0 / 256, 0 / 256)]  # use only for custom cmaps
        primary_cmap = plt.cm.get_cmap(name)
        primary_colors = [primary_cmap(c) for c in np.linspace(lower_ts, 1, 4)]
        primary_colors = primary_colors[::-1] if name == 'autumn' else primary_colors
        colors = [lf_color] + primary_colors
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=1000)
        norm = mpl.colors.Normalize(vmin=ts_lf, vmax=1.2)

        return cmap, norm

    # helper functions for posterior frequency plotting functions (vanilla, family and map)
    def get_plotting_params(self, plot_type="general"):
        """ Here we store various plotting parameters
        Args:
            plot_type (str): Parameters for which type og plot? One of: "plot_posterior",
                             "plot_trace_recall_precision", "plot_trace_lh", ...
        Returns:
            (dict): a dictionary comprising various plotting parameters
        """
        plot_parameters = {
            'fontsize': 16,  # overall fontsize value
            'line_thickness': 2,  # thickness of lines in plots
            'frame_width': 1.5,  # width of frame of plots
            'save_format': 'pdf',  # output format for plots
        }
        if plot_type == "general":
            return plot_parameters

        elif plot_type == "plot_posterior_map_sa":
            plot_parameters['fig_width'] = 20
            plot_parameters['fig_height'] = 10
            plot_parameters['area_legend_position'] = (0.02, 0.71)
            plot_parameters['freq_legend_position'] = (0.3, 0.2)
            plot_parameters['family_legend_position'] = (0.02, 0.98)
            plot_parameters['overview_position'] = (0.02, 0.01, 1, 1)
            return plot_parameters

        elif plot_type == "plot_empty_map":
            plot_parameters['fig_width'] = 20
            plot_parameters['fig_height'] = 10
            plot_parameters['overview_position'] = (0.02, 0.01, 1, 1)
            return plot_parameters

        elif plot_type == "plot_posterior_map_balkan":
            plot_parameters['fig_width'] = 20
            plot_parameters['fig_height'] = 10
            plot_parameters['area_legend_position'] = (0.02, 0.71)
            plot_parameters['freq_legend_position'] = (0.02, 0.55)
            plot_parameters['family_legend_position'] = (0.02, 0.965)
            plot_parameters['overview_position'] = (0.02, -0.01, 1, 1)
            return plot_parameters

        elif plot_type == "plot_trace_recall_precision":
            plot_parameters['fig_width'] = 10
            plot_parameters['fig_height'] = 8
            plot_parameters['color_burn_in'] = "grey"
            return plot_parameters

        elif plot_type == "plot_trace_lh":
            plot_parameters['fig_width'] = 10
            plot_parameters['fig_height'] = 8
            plot_parameters['color_burn_in'] = "grey"
            return plot_parameters

        elif plot_type == "plot_dics_simulated":
            plot_parameters['fig_width'] = 10
            plot_parameters['fig_height'] = 8
            plot_parameters['color_burn_in'] = "grey"
            return plot_parameters

        elif plot_type == "plot_dics":
            plot_parameters['fig_width'] = 9
            plot_parameters['fig_height'] = 6
            plot_parameters['color_burn_in'] = "grey"
            return plot_parameters

        elif plot_type == "plot_traces":
            plot_parameters['fig_width'] = 10
            plot_parameters['fig_height'] = 8
            return plot_parameters

        elif plot_type == "plot_posterior_map_simulated":
            plot_parameters['fig_width'] = 20
            plot_parameters['fig_height'] = 10
            plot_parameters['area_legend_position'] = (0.02, 0.2)
            plot_parameters['freq_legend_position'] = (0.15, 0.2)
            plot_parameters['poly_legend_position'] = (0.37, 0.15)
            plot_parameters['family_legend_position'] = (0.39, 0.10)
            return plot_parameters

        elif plot_type == "plot_trace_lh_with_prior":
            plot_parameters['fig_width'] = 8
            plot_parameters['fig_height'] = 7
            plot_parameters['color_burn_in'] = "grey"
            return plot_parameters


    ##############################################################
    # This is the plot_posterior_map function from plotting_old
    ##############################################################
    def number_zones(self,
                     mcmc_res, sites, post_freq_lines, burn_in=0.2, simulated_data=False,
                     lh_single_zones=False, flamingo=False, simulated_family=False,
                     label_languages=False, add_overview=False,
                     families=None, family_names=None,
                     return_correspondence=False,
                     fname='mst_posterior'):

        """ This function creates a scatter plot of all sites in the posterior distribution. The color of a site reflects
            its frequency in the posterior

            Args:
                mcmc_res (dict): the MCMC samples neatly collected in a dict
                sites (dict): a dictionary containing the location tuples (x,y) and the id of each site
                post_freq_lines (list): threshold values for lines
                                        e.g. [0.7, 0.5, 0.3] will display three different line categories:
                                        - a thick line for edges which are in more than 70% of the posterior
                                        - a medium-thick line for edges between 50% and 70% in the posterior
                                        - and a thin line for edges between 30% and 50% in the posterior
                burn_in (float): Fraction of samples, which are discarded as burn-in
                x_extend (tuple): (min, max)-extend of the map in x-direction (longitude) --> Olga: move to config: DONE
                y_extend (tuple): (min, max)-extend of the map in y-direction (latitude) --> and move to config: DONE
                simulated_data(bool): are the plots for real-world or simulated data?
                experiment(str): either "sa" or "balkan", will load different plotting parameters. Olga: Should go to plotting
                                 config file instead: DONE
                bg_map (bool: Plot a background map? --> Olga: to config: DONE
                geojson_map(str): File path to geoJSON background map --> Olga: to config: DONE
                proj4(str): Coordinate reference system of the language data. --> Olga: Should go to config: DONE
                or could be passed as meta data to the sites: TODO
                geo_json_river(str): File path to river data. --> Olga: should go to config: DONE
                subset(boolean): Is there a subset in the data, which should be displayed differently?
                                 Only relevant for one experiment. --> Olga Should go to config: DONE
                lh_single_zones(bool): Add box containing information about the likelihood of single areas to the plot?
                flamingo(bool): Sort of a joke. Does one area have the shape of a flamingo. If yes use flamingo colors for plotting.
                simulated_family(bool): Only for simulated data. Are families also simulated?
                size(float): Size of the dots (languages) in the plot --> Olga: move to config: DONE
                size_line(float): Line thickness. Gives in combination with post_freq_lines the line thickness of the edges in an area
                                  Olga -> should go to config: DONE
                label_languages(bool): Label the languages in areas?
                add_overview(bool): Add an overview map?
                x_extend_overview(tuple): min, max)-extend of the overview map in x-direction (longitude) --> Olga: config: DONE
                y_extend_overview(tuple): min, max)-extend of the overview map in y-direction (latitude) --> Olga: config: DONE
                families(np.array): a boolean assignment of sites to families
                    shape(n_families, n_sites)
                family_names(dict): a dict comprising both the external (Arawak, Chinese, French, ...) and internal (0,1,2...)
                                    family names
                family_alpha_shape(float): controls how far languages of the same family have to be apart to be grouped
                                           into a single alpha shape (for display only)  --> Olga: config: DONE
                fname (str): a path of the output file.
                return_correspondence(bool): return the labels of all languages which are shown in the map
                                            --> Olga: I think this can be solved differently, with a separate function: TODO
                show_axes(bool): show x- and y- axis? --> I think we can hardcode this to false: DONE
                TODO: change the function style_axes
                frame_offset(float): offset of x and y- axis --> If show_axes is False this is not needed anymore: DONE

            """


        experiment = self.config['input']['experiment']
        proj4 = self.config['input']['proj4']
        geojson_map = self.config['input']['geojson_map']
        geo_json_river = self.config['input']['geo_json_river']

        subset = self.config['general']['subset']

        x_extend = tuple(self.config['graphic']['x_extend'])
        y_extend = tuple(self.config['graphic']['y_extend'])

        try:
            x_extend_overview = tuple(self.config['graphic']['x_extend_overview'])
        except TypeError:
            x_extend_overview = self.config['graphic']['x_extend_overview']

        try:
            y_extend_overview = tuple(self.config['graphic']['y_extend_overview'])
        except TypeError:
            y_extend_overview = self.config['graphic']['y_extend_overview']

        bg_map = self.config['graphic']['bg_map']
        size = self.config['graphic']['size']
        size_line = self.config['graphic']['size_line']
        family_alpha_shape = self.config['graphic']['family_alpha_shape']


        # Is the function used for simulated data or real-world data? Both require different plotting parameters.
        # for Olga: should go to config file
        if simulated_data:
            pp = self.get_plotting_params(plot_type="plot_posterior_map_simulated")

        # if for real world-data: South America or Balkans?
        # for Olga: plotting parameters in pp should be defined in the config file. Can go.
        else:
            if experiment == "sa":
                pp = self.get_plotting_params(plot_type="plot_posterior_map_sa")
            if experiment == "balkan":
                pp = self.get_plotting_params(plot_type="plot_posterior_map_balkan")

        # Initialize the plot
        # Needed in every plot
        plt.rcParams["axes.linewidth"] = pp['frame_width']
        fig, ax = plt.subplots(figsize=(pp['fig_width'], pp['fig_height']), constrained_layout=True)

        # Does the plot have a background map?
        # Could go to extra function (add_background_map), which is only called if relevant
        if bg_map:

            # If yes, the user needs to define a valid spatial coordinate reference system(proj4)
            # and provide background map data
            if proj4 is None and geojson_map is None:
                raise Exception('If you want to use a map provide a geojson and a crs')

            # Adds the geojson map provided by user as background map
            world = gpd.read_file(geojson_map)
            world = world.to_crs(proj4)
            world.plot(ax=ax, color='w', edgecolor='black', zorder=-100000)

            # The user can also provide river data. Looks good on a map :)
            if geo_json_river is not None:
                rivers = gpd.read_file(geo_json_river)
                rivers = rivers.to_crs(proj4)
                rivers.plot(ax=ax, color=None, edgecolor="skyblue", zorder=-10000)

        # Get the areas from the samples
        # Needed in every plot
        zones = mcmc_res['zones']

        # This is only valid for one single experiment
        # where we perform the analysis on a biased subset.
        # The following lines of code selects only those sites which are in the subset
        # Low priority: could go to a separate function
        if subset:

            # Get sites in subset
            is_in_subset = [x == 1 for x in sites['subset']]
            sites_all = deepcopy(sites)

            # Get the relevant information for all sites in the subset (ids, names, ..)
            for key in sites.keys():

                if type(sites[key]) == list:
                    sites[key] = list(np.array(sites[key])[is_in_subset])

                else:
                    sites[key] = sites[key][is_in_subset, :]

        # This computes the Delaunay triangulation of the sites
        # Needed in every plot
        net = compute_network(sites)
        locations, dist_mat = net['locations'], net['dist_mat']

        # Plots all languages on the map
        # Needed in every plot
        # Could go together with all the other stuff that's always done to a separate function
        cmap, _ = self.get_cmap(0.5, name='YlOrRd', lower_ts=1.2)
        ax.scatter(*locations.T, s=size, c=[cmap(0)], alpha=1, linewidth=0)

        # Again only for this one experiment with a subset
        # could go to subset function. Low priority.
        if subset:
            # plot all points not in the subset in light grey
            not_in_subset = np.logical_not(is_in_subset)
            other_locations = sites_all['locations'][not_in_subset]
            ax.scatter(*other_locations.T, s=size, c=[cmap(0)], alpha=1, linewidth=0)

            # Add a visual bounding box to the map to show the location of the subset on the map
            x_coords, y_coords = locations.T
            offset = 100
            x_min, x_max = min(x_coords) - offset, max(x_coords) + offset
            y_min, y_max = min(y_coords) - offset, max(y_coords) + offset
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            bbox = mpl.patches.Rectangle((x_min, y_min), bbox_width, bbox_height, ec='grey', fill=False,
                                         lw=1.5, linestyle='-.')
            ax.add_patch(bbox)
            # Adds a small label that reads "Subset"
            ax.text(x_max, y_max + 200, 'Subset', fontsize=18, color='#000000')

        leg_zones = []

        # Adds an info box showing the likelihood of each area
        # Could go to a separate function
        if lh_single_zones:
            extra = patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            leg_zones.append(extra)

        # This iterates over all areas in the posterior and plots each with a different color (minimum spanning tree)
        all_labels = []
        for i, zone in enumerate(zones):
            # The colors for each area could go to the config file
            zone_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
            # This is actually sort of a joke: if one area has the shape of a flamingo, use a flamingo colour for it
            #  in the map. Anyway, colors should go to the config. If can be removed.
            if flamingo:
                flamingo_color = '#F48AA7'
                c = flamingo_color if len(zones) == 1 else zone_colors[i]
            else:
                c = zone_colors[i]
            # Same here: when simulating families, one has the shape of a banana. If so, use banana color.
            # Should go to the config. If can be removed.
            if simulated_family:
                banana_color = '#f49f1c'
                c = banana_color if len(zones) == 1 else zone_colors[i]

            # This function visualizes each area (edges of the minimum spanning tree that appear
            # in the posterior according to thresholds set in post_freq_lines
            # I think it makes sense to redo this function. I'll do the same annotating before.
            is_in_zone = self.add_minimum_spanning_tree_new(ax, zone, locations, dist_mat, burn_in, post_freq_lines,
                                                       size=size, size_line=size_line, color=c)
            # This adds small lines to the legend (one legend entry per area)
            line = Line2D([0], [0], color=c, lw=6, linestyle='-')
            leg_zones.append(line)

            # Again, this is only relevant for simulated data and should go into a separate function
            if simulated_data:
                try:
                    # Adds a bounding box for the ground truth areas showing if the algorithm has correctly identified them

                    self.add_zone_boundary(ax, locations, net, mcmc_res['true_zones'][i], alpha=0.001, color='#000000')
                except:
                    continue

            # Labels the languages in the areas
            # Should go into a separate function
            if label_languages:
                # Find all languages in areas
                loc_in_zone = locations[is_in_zone, :]
                labels_in_zone = list(compress(sites['id'], is_in_zone))
                all_labels.append(labels_in_zone)

                for l in range(len(loc_in_zone)):
                    # add a label at a spatial offset of 20000 and 10000. Rather than hard-coding it,
                    # this might go into the config.
                    x, y = loc_in_zone[l]
                    x += 20000
                    y += 10000
                    # Same with the font size for annotations. Should probably go to the config.
                    anno_opts = dict(xy=(x, y), fontsize=14, color=c)
                    ax.annotate(labels_in_zone[l] + 1, **anno_opts)

        # This is actually the beginning of a series of functions that add a small legend
        # displaying what the line thickness corresponds to.
        # Could go to a separate function
        leg_line_width = []
        line_width_label = []
        post_freq_lines.sort(reverse=True)

        # Iterates over all threshold values in post_freq_lines and for each adds one
        # legend entry in a neutral color (black)
        for k in range(len(post_freq_lines)):
            line = Line2D([0], [0], color="black", lw=size_line * post_freq_lines[k], linestyle='-')
            leg_line_width.append(line)

            # Adds legend text.
            prop_l = int(post_freq_lines[k] * 100)

            if k == 0:
                line_width_label.append(f'$\geq${prop_l}%')

            else:
                prop_s = int(post_freq_lines[k - 1] * 100)
                line_width_label.append(f'$\geq${prop_l}% and $<${prop_s}%')

        # This adds an overview map to the main map
        # Could go into a separate function
        if add_overview:
            # All hard-coded parameters (width, height, lower_left, ... could go to the config file.
            axins = inset_axes(ax, width=3.8, height=4, bbox_to_anchor=pp['overview_position'],
                               loc='lower left', bbox_transform=ax.transAxes)
            axins.tick_params(labelleft=False, labelbottom=False, length=0)

            # Map extend of the overview map
            # x_extend_overview and y_extend_overview --> to config
            axins.set_xlim(x_extend_overview)
            axins.set_ylim(y_extend_overview)

            # Again, this function needs map data to display in the overview map.
            if proj4 is not None and geojson_map is not None:
                world = gpd.read_file(geojson_map)
                world = world.to_crs(proj4)
                world.plot(ax=axins, color='w', edgecolor='black', zorder=-100000)

            # add overview to the map
            axins.scatter(*locations.T, s=size / 2, c=[cmap(0)], alpha=1, linewidth=0)

            # adds a bounding box around the overview map
            bbox_width = x_extend[1] - x_extend[0]
            bbox_height = y_extend[1] - y_extend[0]
            bbox = mpl.patches.Rectangle((x_extend[0], y_extend[0]), bbox_width, bbox_height, ec='k', fill=False,
                                         linestyle='-')
            axins.add_patch(bbox)

        # Depending on the background (sa, balkan, simulated), we want to place additional legend entries
        # at different positions in the map in order not to block map content and best use the available space.
        # This should rather go to the config file.
        # Unfortunately, positions have to be provided in map units, which makes things a bit opaque.
        # Once in the config, the functions below can go.
        # for Sa map
        if experiment == "sa":
            x_unit = (x_extend[1] - x_extend[0]) / 100
            y_unit = (y_extend[1] - y_extend[0]) / 100
            ax.axhline(y_extend[0] + y_unit * 71, 0.02, 0.20, lw=1.5, color="black")

            ax.add_patch(
                patches.Rectangle(
                    (x_extend[0], y_extend[0]),
                    x_unit * 25, y_unit * 100,
                    color="white"
                ))
        # for Balkan map
        if experiment == "balkan":
            x_unit = (x_extend[1] - x_extend[0]) / 100
            y_unit = (y_extend[1] - y_extend[0]) / 100

            ax.add_patch(
                patches.Rectangle(
                    (x_extend[0], y_extend[0]),
                    x_unit * 25, y_unit * 100,
                    color="white"
                ))
            ax.axhline(y_extend[0] + y_unit * 56, 0.02, 0.20, lw=1.5, color="black")
            ax.axhline(y_extend[0] + y_unit * 72, 0.02, 0.20, lw=1.5, color="black")

        # for simulated data
        if simulated_data:
            x_unit = (x_extend[1] - x_extend[0]) / 100
            y_unit = (y_extend[1] - y_extend[0]) / 100
            ax.add_patch(
                patches.Rectangle(
                    (x_extend[0], y_extend[0]),
                    x_unit * 55,
                    y_unit * 30,
                    color="white"
                ))
            # The legend looks a bit different, as it has to show both the inferred areas and the ground truth
            ax.annotate("INFERRED", (x_extend[0] + x_unit * 3, y_extend[0] + y_unit * 23), fontsize=20)
            ax.annotate("GROUND TRUTH", (x_extend[0] + x_unit * 38.5, y_extend[0] + y_unit * 23), fontsize=20)
            ax.axvline(x_extend[0] + x_unit * 37, 0.05, 0.18, lw=2, color="black")

        # If families and family names are provided, this adds an overlay color for all language families in the map
        # including a legend entry.
        # Should go to a separate function

        if families is not None and family_names is not None:
            # Family colors, should probably go to config
            # family_colors = ['#377eb8', '#4daf4a', '#984ea3', '#a65628', '#f781bf', '#999999', '#ffff33', '#e41a1c', '#ff7f00']
            family_colors = ['#b3e2cd', '#f1e2cc', '#cbd5e8', '#f4cae4', '#e6f5c9', '#d3d3d3']

            # Initialize empty legend handle
            handles = []

            # Iterate over all family names
            for i, family in enumerate(family_names['external']):
                # print(i, family)

                # Find all languages belonging to a family
                is_in_family = families[i] == 1
                family_locations = locations[is_in_family, :]
                family_color = family_colors[i]

                # Adds a color overlay for each language in a family
                ax.scatter(*family_locations.T, s=size * 15, c=family_color, alpha=1, linewidth=0, zorder=-i,
                           label=family)

                family_fill, family_border = family_color, family_color

                # For languages with more than three members: instead of one dot per language,
                # combine several languages in an alpha shape (a polygon)
                if family_alpha_shape is not None and np.count_nonzero(is_in_family) > 3:
                    alpha_shape = self.compute_alpha_shapes([is_in_family], net, family_alpha_shape)

                    # making sure that the alpha shape is not empty
                    if not alpha_shape.is_empty:
                        smooth_shape = alpha_shape.buffer(40000, resolution=16, cap_style=1, join_style=1,
                                                          mitre_limit=5.0)
                        patch = PolygonPatch(smooth_shape, fc=family_fill, ec=family_border, lw=1, ls='-', alpha=1,
                                             fill=True, zorder=-i)
                        leg_family = ax.add_patch(patch)

                # Add legend handle
                handle = Patch(facecolor=family_color, edgecolor=family_color, label=family)
                handles.append(handle)

                # (Hard-coded) parameters should probably go to config
                # Defines the legend for families
                legend_families = ax.legend(
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
                    bbox_to_anchor=pp['family_legend_position']
                )
                ax.add_artist(legend_families)

        # Again this adds alpha shapes for families for simulated data.
        # I think the reason why this was coded separately, is that many of the parameters
        # change compared to real world-data
        # Should probably be handled in the config file instead
        # and should be merged with adding families as seen above
        if simulated_family:
            families = mcmc_res['true_families']
            family_colors = ['#add8e6', '#f1e2cc', '#cbd5e8', '#f4cae4', '#e6f5c9']
            # handles for legend
            handles = []
            for i, is_in_family in enumerate(families):
                # plot points belonging to family

                family_color = family_colors[i]

                family_alpha_shape = 0.001
                family_fill = family_color
                family_border = family_color
                alpha_shape = self.compute_alpha_shapes([is_in_family], net, family_alpha_shape)
                # print(is_in_family, net, alpha_shape)
                smooth_shape = alpha_shape.buffer(60, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
                # smooth_shape = alpha_shape
                patch = PolygonPatch(smooth_shape, fc=family_fill, ec=family_border, lw=1, ls='-', alpha=1, fill=True,
                                     zorder=-i)
                ax.add_patch(patch)

                # adding legend handle
                handle = Patch(facecolor=family_color, edgecolor=family_color, label="simulated family")
                handles.append(handle)

            # Define the legend
            legend_families = ax.legend(
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
                bbox_to_anchor=pp['family_legend_position']
            )
            ax.add_artist(legend_families)

        # If likelihood for single areas are displayed: add legend entries with likelihood information per area
        if lh_single_zones:
            def scientific(x):
                b = int(np.log10(x))
                a = x / 10 ** b
                return '%.2f \cdot 10^{%i}' % (a, b)

            # Legend for area labels
            zone_labels = ["         probability of area"]

            # This assumes that the likelihoods for single areas (lh_a1, lh_a2, lh_a3, ...)
            # have been collected in mcmc_res under the key shown below
            post_per_area = np.asarray(mcmc_res['sample_posterior_single_areas'])
            to_rank = np.mean(post_per_area, axis=0)

            # probability per area in log-space
            p_total = logsumexp(to_rank)
            p = to_rank[np.argsort(-to_rank)] - p_total

            for i, exp in enumerate(p):
                lh_value = np.exp(exp)
                # print(lh_value)
                lh_value = scientific(lh_value)
                zone_labels.append(f'$Z_{i + 1}: \, \;\;\; {lh_value}$')

        else:
            zone_labels = []
            for i, _ in enumerate(zones):
                zone_labels.append(f'$Z_{i + 1}$')

        # Define legend
        legend_zones = ax.legend(
            leg_zones,
            zone_labels,
            title_fontsize=18,
            title='Contact areas',
            frameon=True,
            edgecolor='#ffffff',
            framealpha=1,
            fontsize=16,
            ncol=1,
            columnspacing=1,
            loc='upper left',
            bbox_to_anchor=pp['area_legend_position']
        )
        legend_zones._legend_box.align = "left"
        ax.add_artist(legend_zones)

        # Defines the legend entry for Line width
        legend_line_width = ax.legend(
            leg_line_width,
            line_width_label,
            title_fontsize=18,
            title='Frequency of edge in posterior',
            frameon=True,
            edgecolor='#ffffff',
            framealpha=1,
            fontsize=16,
            ncol=1,
            columnspacing=1,
            loc='upper left',
            bbox_to_anchor=pp['freq_legend_position']
        )

        legend_line_width._legend_box.align = "left"
        ax.add_artist(legend_line_width)

        # for simulated data: add a legend entry in the shape of a little polygon for ground truth
        if simulated_data:
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
                pp['poly_legend_position'] = (pp['poly_legend_position'][0], pp['poly_legend_position'][1] + 0.05)

            # define legend
            legend_true_zones = ax.legend([TrueZone()], ['simulated area\n(bounding polygon)'],
                                          handler_map={TrueZone: TrueZoneHandler()},
                                          bbox_to_anchor=pp['poly_legend_position'],
                                          title_fontsize=16,
                                          loc='upper left',
                                          frameon=True,
                                          edgecolor='#ffffff',
                                          handletextpad=4,
                                          fontsize=18,
                                          ncol=1,
                                          columnspacing=1)

            ax.add_artist(legend_true_zones)

        # styling the axes, might be hardcoded

        self.style_axes(ax, locations, show=False, x_extend=x_extend, y_extend=y_extend)

        # Save the plot
        fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
        plt.close(fig)

        # Should the labels displayed in the map be returned? These are later added as a separate legend (
        # outside this hell of a function)
        if return_correspondence and label_languages:
            return all_labels


