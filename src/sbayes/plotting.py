#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sbayes.util import zones_autosimilarity, add_edge, compute_delaunay, colorline, compute_mst_posterior
from sbayes.util import bounding_box, round_int, linear_rescale, round_single_int, round_multiple_ints
from sbayes.preprocessing import compute_network
from scipy.stats import gamma, linregress
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.ops import cascaded_union, polygonize
from shapely import geometry
from descartes import PolygonPatch
from copy import deepcopy
import geopandas as gpd
from itertools import compress
import os
from scipy.special import logsumexp
os.environ["PROJ_LIB"] = "C:/Users/ranacher/Anaconda3/Library/share"
plt.style.use('seaborn-paper')
plt.tight_layout()

def get_colors():
    """This function creates a dict with colors
    Returns:
        (dict): a dictionary comprising the plot colors
    """

    plot_colors = {'histogram': {'fitted_line': (1.000, 0.549, 0.000),      # darkorange
                                 'background': (1, 1, 1)},                  # white
                   'zone': {'background_nodes': (0.502, 0.502, 0.502),      # grey
                            'in_zone': (1.000, 0.549, 0.000),               # darkorange
                            'triangulation': (1.000, 0.549, 0.000)},
                   'trace': {'lh': (0.502, 0.502, 0.502),                   # grey
                             'maximum': (1.000, 0.549, 0.000),              # darkorange
                             'precision': (0.216, 0.494, 0.722),            # blue
                             'recall': (1.000, 0.549, 0.000)},              # darkorange
                   'boxplot': {'median': (1.000, 0.549, 0.000),             # darkorange
                               'whiskers': (0.502, 0.502, 0.502),           # grey
                               'box': (1.000, 0.549, 0.000, 0.2)},          # darkorange (transparent)
                   'zones': {'background_nodes': (0.502, 0.502, 0.502),
                             'in_zones': [(0.894, 0.102, 0.11),             # red
                                          (0.216, 0.494, 0.722),            # blue
                                          (0.302, 0.686, 0.29),             # green
                                          (0.596, 0.306, 0.639),            # violett
                                          (1.000, 0.549, 0.000),            # darkorange)
                                          (1, 1, 0.2),                      # yellow
                                          (0.651, 0.337, 0.157),            # brown
                                          (0.969, 0.506, 0.749),            # pinkish
                                          (0, 0, 0)],                       # black ]
                             'triangulation': (0.4, 0., 0.)}}
    return plot_colors


# helper functions for posterior frequency plotting functions (vanilla, family and map)
def get_plotting_params(plot_type="general"):
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


def get_cmap(ts_lf, name='YlOrRd', lower_ts=0.2):
    """ Function to generate a colormap
    Args:
        ts_lf (float): Anything below this frequency threshold is grey.
        name (string): Name of the colormap.
        lower_ts (float): Threshold to manipulate colormap.
    Returns:
        (LinearSegmentedColormap): Colormap
        (Normalize): Object for normalization of frequencies
    """
    grey_tone = 128 # from 0 (dark) to 1 (white)
    lf_color = (grey_tone / 256, grey_tone / 256, grey_tone / 256)  # grey 128
    colors = [lf_color, (256/256, 256/256, 0/256), (256/256, 0/256, 0/256)] # use only for custom cmaps
    primary_cmap = plt.cm.get_cmap(name)
    primary_colors = [primary_cmap(c) for c in np.linspace(lower_ts, 1, 4)]
    primary_colors = primary_colors[::-1] if name == 'autumn' else primary_colors
    colors = [lf_color] + primary_colors
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=1000)
    norm = mpl.colors.Normalize(vmin=ts_lf, vmax=1.2)

    return cmap, norm

def add_posterior_frequency_legend(fig, axes, ts_lf, cmap, norm, ts_posterior_freq, show_ts=True):
    """ Add a posterior frequency legend to the plot
    Args:
        fig (Figure): Figure to add the legend to.
        axes (list): List of axes.
        ts_lf (ts_lf): Lower frequency threshold.
        cmap (LinearSegmentedColormap): Colormap.
        norm (Normalize): Object for normalization of frequency.
        ts_posterior_freq (float): Posterior frequency threshold.
        show_ts (boolean): Include the posterior frequency threshold in legend.
    """
    pp = get_plotting_params()

    # unpacking axes
    ax_lf, ax_hf, ax_title = axes

    # setting up low frequency color bar

    # defining ticks
    cbar_ticks, cbar_ticklabels = [0.5], [f'<{ts_lf * 100:.0f}']

    # defining colorbar
    lf_color = cmap(0)
    lf_cmap_legend = mpl.colors.ListedColormap([lf_color])
    cbar = mpl.colorbar.ColorbarBase(ax_lf, cmap=lf_cmap_legend, norm=mpl.colors.BoundaryNorm([0,1], lf_cmap_legend.N), ticks=cbar_ticks, orientation='horizontal')
    cbar.ax.tick_params(size=0)

    # offsetting the label of low frequency colorbar
    offset = mpl.transforms.ScaledTranslation(0, -0.045, fig.dpi_scale_trans)
    for label in cbar.ax.xaxis.get_majorticklabels(): label.set_transform(label.get_transform() + offset)

    # adding tick labels
    cbar.ax.tick_params(labelsize=pp['fontsize'])
    cbar.ax.set_xticklabels(cbar_ticklabels)

    # setting up high frequency color bar
    n_ticks = int((100 - ts_lf * 100) / 10 + 1)
    n_ticks = n_ticks if ts_posterior_freq * 100 % 10 == 0 else n_ticks * 2
    cbar_ticks = np.linspace(ts_lf, 1, n_ticks)
    cbar_ticklabels = np.linspace(ts_lf, 1, n_ticks)
    cbar_ticklabels = [f'{round(t * 100, 0):.0f}' for t in cbar_ticklabels]
    cbar = mpl.colorbar.ColorbarBase(ax_hf, cmap=cmap, norm=norm, boundaries=np.linspace(ts_lf,1,1000),
                                     orientation='horizontal', ticks=cbar_ticks)
    cbar.ax.tick_params(labelsize=pp['fontsize'])
    cbar.ax.set_xticklabels(cbar_ticklabels)

    # adding a line in the colorbar showing the posterior frequency threshold
    if show_ts:
        cbar_step = int(100 - ts_lf * 100) // (n_ticks - 1)
        index_ts = int(ts_posterior_freq * 100 - ts_lf * 100) // cbar_step

        cbar_ticklabels[index_ts] = f'{cbar_ticklabels[index_ts]} (ts)'
        cbar.ax.set_xticklabels(cbar_ticklabels)
        cbar.ax.plot([linear_rescale(ts_posterior_freq, ts_lf, 1, 0, 1)] * 2, [0, 1], 'k', lw=1)

    # finally adding a title
    ax_title.text(0.5, 0, s='Frequency of point in posterior (%)', fontsize=pp['fontsize'], horizontalalignment='center')

def add_posterior_frequency_points(ax, zones, locations, ts, cmap, norm, nz=-1, burn_in=0.2, size=25):
    """ Add posterior frequency points to axis
    Args:
        ax (axes.Axes): Axis of the plot.
        zones (np.array): Zone data from mcmc results.
        locations (np.array): Locations of points.
        ts (float): Posterior frequency threshold.
        cmap (LinearSegmentedColormap): Colormap.
        norm (Normalize): Object for normalization of frequency.
        nz (int): Which zone to plot? If -1 all zones are plotted.
        burn_in (float): Percentage of samples to be excluded.
        size (int): Size of points.
    """
    # getting number of zones
    n_zones = len(zones)

    # plot all zones
    if nz == -1:
        # get samples from all zones
        n_samples = len(zones[0])
        zones_reformatted = [sum(k) for k in zip(*zones)]

        # exclude burn-in
        end_bi = math.ceil(len(zones_reformatted) * burn_in)
        posterior_freq = (np.sum(zones_reformatted[end_bi:], axis=0, dtype=np.int32) / (n_samples - end_bi))
        # print('All Points', np.sort(posterior_freq)[::-1][:5])


    # plot only one zone (passed as argument)
    else:
        # get samples of the zone
        zone = zones[nz-1]
        n_samples = len(zone)

        # exclude burn-in
        end_bi = math.ceil(n_samples * burn_in)

        # compute frequency of each point in that zone
        posterior_freq = (np.sum(zone[end_bi:], axis=0, dtype=np.int32) / (n_samples - end_bi))
        # print(np.sort(posterior_freq)[::-1][:10])


    # plotting all low posterior frequency (lf) points (posterior frequency < ts_cmap)
    lf_color = cmap(0)
    is_lf_point = posterior_freq < ts
    lf_locations = locations[is_lf_point,:]


    ax.scatter(*lf_locations.T, s=size, c=[lf_color], alpha=1, linewidth=0, edgecolor='black')


    # plotting all high posterior frequency (hf) points
    is_hf_point = np.logical_not(is_lf_point)
    hf_locations = locations[is_hf_point]
    hf_posterior_freq = posterior_freq[is_hf_point]

    # sorting points based on their posterior frequency
    order = np.argsort(hf_posterior_freq)
    hf_posterior_freq = hf_posterior_freq[order]
    hf_locations = hf_locations[order]

    ax.scatter(*hf_locations.T, s=size, c=hf_posterior_freq, cmap=cmap, norm=norm, alpha=1, linewidth=0,
               edgecolor='black')

def add_zone_bbox(ax, zones, locations, net, nz, n_zones, burn_in, ts_posterior_freq, offset, annotate=True, fontsize=18):
    """ Function to add bounding boxes around zones
    Args:
        ax (axes.Axes): Axis of the plot.
        zones (np.array): Zone data from mcmc results.
        locations (np.array): Locations of points.
        net (unknown): Network of points.
        nz (int): Which zone to plot? If -1 all zones are plotted.
        n_zones (int): Number of contact zones.
        burn_in (float): Percentage of samples to be excluded.
        ts_posterior_freq (float): Posterior frequency threshold.
        offset (float): Offset of bounding boxes around zones.
        annotate (bool): Whether to add annotations to zones.
    Returns:
        leg_zones: Legend.
    """

    # create list with all zone indices
    indices_zones = [nz-1] if nz != -1 else range(n_zones)
    color_zones = '#000000'

    for zone_index in indices_zones:

        # print(f'Zone {zone_index + 1} / {n_zones}. Index {zone_index}')

        # get samples of the zone
        zone = zones[zone_index]
        n_samples = len(zone)

        # exclude burn-in
        end_bi = math.ceil(n_samples * burn_in)

        # compute frequency of each point in that zone
        posterior_freq = (np.sum(zone[end_bi:], axis=0, dtype=np.int32) / (n_samples - end_bi))

        is_contact_point = posterior_freq > ts_posterior_freq
        cp_locations = locations[is_contact_point]
        # print(f'Max posterior freq {np.max(posterior_freq)}')

        leg_zone = None
        if cp_locations.shape[0] > 0: # at least one contact point in zone

            zone_bbox = bounding_box(cp_locations)
            x_min, x_max, y_min, y_max = zone_bbox['x_min'], zone_bbox['x_max'], zone_bbox['y_min'], zone_bbox['y_max']
            x_min, x_max = round_int(x_min, 'down', offset), round_int(x_max, 'up', offset)
            y_min, y_max = round_int(y_min, 'down', offset), round_int(y_max, 'up', offset)

            bbox_ll = (x_min, y_min)
            bbox_height = y_max - y_min
            bbox_width = x_max - x_min
            bbox = mpl.patches.Rectangle(bbox_ll, bbox_width, bbox_height, fill=False, edgecolor=color_zones, lw=2)

            # leg_zone = ax.add_patch(bbox)


            alpha_shape = compute_alpha_shapes([is_contact_point], net, 0.002)

            # smooth_shape = alpha_shape.buffer(100, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
            smooth_shape = alpha_shape.buffer(150, resolution=16, cap_style=1, join_style=1, mitre_limit=10.0)
            # smooth_shape = alpha_shape
            patch = PolygonPatch(smooth_shape, ec=color_zones, lw=1, ls='-', alpha=1, fill=False,
                                 zorder=-1)
            leg_zone = ax.add_patch(patch)



            # only adding a label (numeric) if annotation turned on and more than one zone
            if annotate and n_zones > 1:
                zone_name = f'{zone_index + 1}'
                zone_name_yoffset = bbox_height + 100
                zone_name_xoffset = bbox_width - 180
                ax.text(bbox_ll[0] + zone_name_xoffset, bbox_ll[1] + zone_name_yoffset, zone_name, fontsize=fontsize, color=color_zones)
        else:
            print('computation of bbox not possible because no contact points')

    return leg_zone


def add_zone_boundary(ax, locations, net, is_in_zone, alpha, annotation=None, color='#000000'):
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

    cp_locations = locations[is_in_zone, :]

    leg_zone = None
    if cp_locations.shape[0] > 0: # at least one contact point in zone


        alpha_shape = compute_alpha_shapes([is_in_zone], net, alpha)

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


def style_axes(ax, locations, show=True, offset=None, x_extend=None, y_extend=None):
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

    pp = get_plotting_params()
    # getting axes ranges and rounding them
    x_min, x_max = np.min(locations[:,0]), np.max(locations[:,0])
    y_min, y_max = np.min(locations[:,1]), np.max(locations[:,1])

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
    x_ticks = np.arange(x_min, x_max+x_step, x_step) if show else []
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=pp['fontsize'])

    # y axis
    y_step = (y_max - y_min) // 5
    y_ticks = np.arange(y_min, y_max+y_step, y_step) if show else []
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, fontsize=pp['fontsize'])

    return (x_min, x_max, y_min, y_max)

def get_axes(fig):
    """ Function to generate a group of axes used for the special legend.
    Args:
        ax (axes.Axes): Axis of the plot.
    Returns:
        (axes.Axes): Main plot axis.
        (list): List of individual axes for colorbar
    """

    nrows, ncols = 100, 10
    height_ratio = 4

    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)

    # main plot ax
    ax = fig.add_subplot(gs[:-height_ratio, :])

    # cbar axes
    hspace = 2
    cbar_offset = 2
    cbar_title_ax = fig.add_subplot(gs[-height_ratio:-height_ratio + hspace, :])
    cbar_title_ax.set_axis_off()
    hide_ax = fig.add_subplot(gs[-height_ratio + hspace:, 0:cbar_offset])
    hide_ax.set_axis_off()
    cbar1_ax = fig.add_subplot(gs[-height_ratio + hspace:, cbar_offset])
    cbar2_ax = fig.add_subplot(gs[-height_ratio + hspace:, cbar_offset + 1:ncols - cbar_offset])
    hide_ax = fig.add_subplot(gs[-height_ratio + hspace:, ncols - cbar_offset:])
    hide_ax.set_axis_off()
    cbar_axes = (cbar1_ax, cbar2_ax, cbar_title_ax)

    return ax, cbar_axes


def add_minimum_spanning_tree(ax, zone, locations, dist_mat, burn_in, ts_posterior_freq, cmap, norm, lw=2, size=25):

    pp = get_plotting_params()
    n_fragments = 100
    n_samples = len(zone)

    # exclude burn in and then compute posterior frequency of each point in the zone
    end_bi = math.ceil(n_samples * burn_in)
    posterior_freq = (np.sum(zone[end_bi:], axis=0, dtype=np.int32) / (n_samples - end_bi))
    print('MST', np.sort(posterior_freq)[::-1][:5])

    # subsetting locations, posterior frequencies, and distance matrix to contact points (cp)
    is_contact_point = posterior_freq > ts_posterior_freq
    cp_locations = locations[is_contact_point, :]
    cp_posterior_freq = posterior_freq[is_contact_point]
    cp_dist_mat = dist_mat[is_contact_point]
    cp_dist_mat = cp_dist_mat[:, is_contact_point]

    n_contact_points = len(cp_locations)

    # plot minimum spanning tree dependent on the number of contact points in zone
    # plot normal minimum spanning for more than 3 points, else handle special cases
    if n_contact_points > 3:
        # computing the minimum spanning tree of contact points
        cp_delaunay = compute_delaunay(cp_locations)
        cp_mst = minimum_spanning_tree(cp_delaunay.multiply(cp_dist_mat))

        # converting minimum spanning tree to boolean array denoting whether contact points are connected
        cp_mst = cp_mst.toarray()
        cp_connections = cp_mst > 0

        # plotting every edge (connections of points) of the network
        for index, connected in np.ndenumerate(cp_connections):
            if connected:
                i1, i2 = index
                # locations of the two contact points and their respective posterior frequencies
                cp1_loc, cp2_loc = cp_locations[i1], cp_locations[i2]
                cp1_freq, cp2_freq = cp_posterior_freq[i1], cp_posterior_freq[i2]

                # computing color gradient between the two contact points
                x = np.linspace(cp1_loc[0], cp2_loc[0], n_fragments)
                y = np.linspace(cp1_loc[1], cp2_loc[1], n_fragments)
                freq_gradient = np.linspace(cp1_freq, cp2_freq, n_fragments)

                # plotting color gradient line
                colorline(ax, x, y, z=freq_gradient, cmap=cmap, norm=norm, linewidth=lw)
    # compute delaunay only works for n points > 3 but mst can be computed for 3 and 2 points
    elif n_contact_points == 3:
        connected = []
        dist_matrix = np.linalg.norm(cp_locations - cp_locations[:, None], axis=-1)
        if dist_matrix[0,1] < dist_matrix[0,2]:
            connected.append([0,1])
            if dist_matrix[2,0] < dist_matrix[2,1]:
                connected.append([2,0])
            else:
                connected.append([2,1])
        else:
            connected.append([0,2])
            if dist_matrix[1,0] < dist_matrix[1,2]:
                connected.append([1,0])
            else:
                connected.append([1,2])
        for index in connected:
            i1, i2 = index
            # locations of the two contact points and their respective posterior frequencies
            cp1_loc, cp2_loc = cp_locations[i1], cp_locations[i2]
            cp1_freq, cp2_freq = cp_posterior_freq[i1], cp_posterior_freq[i2]

            # computing color gradient between the two contact points
            x = np.linspace(cp1_loc[0], cp2_loc[0], n_fragments)
            y = np.linspace(cp1_loc[1], cp2_loc[1], n_fragments)
            freq_gradient = np.linspace(cp1_freq, cp2_freq, n_fragments)

            colorline(ax, x, y, z=freq_gradient, cmap=cmap, norm=norm, linewidth=lw)

    # for 2 contact points the two points are simply connected with a line
    elif n_contact_points == 2:
        # computing color gradient between the two contact points
        x = np.linspace(*cp_locations, n_fragments)
        y = np.linspace(*cp_locations, n_fragments)
        freq_gradient = np.linspace(cp_posterior_freq, n_fragments)

        colorline(ax, x, y, z=freq_gradient, cmap=cmap, norm=norm, linewidth=lw)
    # for 1 point don't do anything
    elif n_contact_points == 1: pass
    # if there aren't any contact points in the zone print a warning
    else:
        print('Warning: No points in contact zone!')

    order = np.argsort(cp_posterior_freq)
    ax.scatter(*cp_locations[order].T, s=size * 2, c=cp_posterior_freq[order], cmap=cmap, norm=norm, alpha=1,
               linewidth=0,
               edgecolor='black')

    extend_locations = cp_locations if len(cp_locations) > 3 else locations

    return extend_locations


def add_minimum_spanning_tree_new(ax, zone, locations, dist_mat, burn_in, post_freq_line, size=25,
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
        cp_dist_mat = dist_mat[sample, :][:,sample]
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
            if dist_matrix[0,1] < dist_matrix[0,2]:
                connected.append([0,1])
                if dist_matrix[2,0] < dist_matrix[2,1]:
                    connected.append([2,0])
                else:
                    connected.append([2,1])
            else:
                connected.append([0,2])
                if dist_matrix[1,0] < dist_matrix[1,2]:
                    connected.append([1,0])
                else:
                    connected.append([1,2])
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
        elif n_contact_points == 1: pass
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


def plot_posterior_frequency(mcmc_res, net, nz=-1, burn_in=0.2, show_zone_bbox=False, zone_bbox_offset=200,
                              ts_posterior_freq=0.7, ts_low_frequency=0.5, frame_offset=200, show_axes=True,
                              size=25, fname='posterior_frequency'):
    """ This function creates a scatter plot of all sites in the posterior distribution. The color of a site reflects
    its frequency in the posterior

    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        net (dict): The full network containing all sites.
        nz (int): For multiple zones: which zone should be plotted? If -1, plot all.
        burn_in (float): Percentage of first samples which is discarded as burn-in
        show_zone_bbox (boolean): Adds box(es) with annotation to zone(s)
        ts_posterior_freq (float): If zones are annotated this threshold
        size (int): size of points
        cmap (matplotlib.cm): colormap for posterior frequency of points
        fname (str): a path followed by a the name of the file
    """


    # gemeral plotting parameters
    pp = get_plotting_params()

    plt.rcParams["axes.linewidth"] = pp['frame_width']
    fig = plt.figure(figsize=(pp['fig_width'], pp['fig_height']), constrained_layout=True)

    # getting figure axes
    ax, cbar_axes = get_axes(fig)

    # getting mcmc data and locations of points
    zones = mcmc_res['zones']
    n_zones = len(zones)
    locations, dist_mat = net['locations'], net['dist_mat']

    # adding scatter plot and corresponding colorbar legend
    cmap, norm = get_cmap(ts_low_frequency, name='YlOrRd', lower_ts=0.2)
    add_posterior_frequency_points(ax, zones, locations, ts_low_frequency, cmap, norm, nz=nz, burn_in=burn_in, size=size)
    add_posterior_frequency_legend(fig, cbar_axes, ts_low_frequency, cmap, norm, ts_posterior_freq, fontsize=pp['fontsize'])

    for zone in zones:
        add_minimum_spanning_tree(ax, zone, locations, dist_mat, burn_in, ts_posterior_freq, cmap, norm, lw=1, size=size/2)

    leg_zone = False
    if show_zone_bbox:
        leg_zone = add_zone_bbox(ax, zones, locations, net, nz, n_zones, burn_in, ts_posterior_freq, zone_bbox_offset)

    # styling the axes
    style_axes(ax, locations, frame_offset, show=show_axes, fontsize=pp['fontsize'])

    # adding a legend to the plot
    #if leg_zone:
    #    ax.legend((leg_zone,), ('Contact zone',), frameon=False, fontsize=pp['fontsize'], loc='upper right', ncol=1, columnspacing=1)

    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)


def plot_minimum_spanning_tree(mcmc_res, net, z=1, burn_in=0.2, ts_posterior_freq=0.7, ts_low_frequency=0.5,
                                show_axes=True, frame_offset=200, annotate=False, size=50, fname='minimum_spanning_tree.png'):
    """ This function plots the minimum spanning tree of the sites that are above the posterior frequency threshold.

    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        network (dict): The full network containing all sites.
        z (int): which zone should be plotted?
        burn_in (float): Percentage of first samples which is discarded as burn-in
        ts_posterior_freq (float): threshold for sites to be included in the mst
        offset (int): offset sets the amount of contextual information to be shown around the mst
        fname (str): a path followed by a the name of the file
    """

    # gemeral parameters parameters
    pp = get_plotting_params()

    plt.rcParams["axes.linewidth"] = pp['frame_width']
    fig = plt.figure(figsize=(pp['fig_width'], pp['fig_height']), constrained_layout=True)

    nrows, ncols = 100, 10
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)
    height_ratio = 4
    ax = fig.add_subplot(gs[:-height_ratio, :])

    hspace = 2
    cbar_offset = 2
    cbar_title_ax = fig.add_subplot(gs[-height_ratio:-height_ratio+hspace, :])
    cbar_title_ax.set_axis_off()
    hide_ax = fig.add_subplot(gs[-height_ratio + hspace:, 0:cbar_offset])
    hide_ax.set_axis_off()
    cbar1_ax = fig.add_subplot(gs[-height_ratio+hspace:, cbar_offset])
    cbar2_ax = fig.add_subplot(gs[-height_ratio+hspace:, cbar_offset+1:ncols-cbar_offset])
    hide_ax = fig.add_subplot(gs[-height_ratio + hspace:, ncols-cbar_offset:])
    hide_ax.set_axis_off()
    cbar_axes = (cbar1_ax, cbar2_ax, cbar_title_ax)

    # getting mcmc data, locations of points and distance matrix
    zones = mcmc_res['zones']
    n_zones = len(zones)
    zone_index = z - 1
    zone = zones[zone_index]
    n_samples = len(zone)
    locations, dist_mat = net['locations'], net['dist_mat']


    # adding scatter plot and corresponding colorbar legend
    cmap, norm = get_cmap(ts_low_frequency, name='YlOrRd', lower_ts=0.2)
    add_posterior_frequency_points(ax, zones, locations, ts_low_frequency, cmap, norm, nz=z, burn_in=burn_in, size=size)
    add_posterior_frequency_legend(fig, cbar_axes, ts_low_frequency, cmap, norm, ts_posterior_freq)

    # plotting minimum spanning tree of contact points
    extend_locations = add_minimum_spanning_tree(ax, zone, locations, dist_mat, burn_in, ts_posterior_freq, cmap, norm, size=size)

    if annotate and n_zones > 1:
        zone_color = '#000000'
        ax.tick_params(color=zone_color, labelcolor=zone_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(zone_color)
            spine.set_linewidth(pp['frame_width'])


        anno_opts = dict(xy=(0.95, 0.92), xycoords='axes fraction', fontsize=pp['fontsize'], color=zone_color, va='center', ha='center')
        ax.annotate(f'{z}', **anno_opts)


    add_zone_bbox(ax, zones, locations, net, z, n_zones, burn_in, ts_posterior_freq, 200, annotate=False)


    # styling axes
    style_axes(ax, extend_locations, offset=frame_offset, show=show_axes)

    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)


def plot_posterior_zoom(mcmc_res, sites, burn_in=0.2, x_extend=None, size=25, size_line=3,
                        y_extend=None, post_freq_lines =0.6, frame_offset=None, plot_family=False, flamingo=False,
                        show_axes=True, subset=False,  fname='mst_posterior'):
    """ This function creates a scatter plot of all sites in the posterior distribution. The color of a site reflects
    its frequency in the posterior

    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        sites(dict): all sites (languages) and their locations
        burn_in (float): Percentage of first samples which is discarded as burn-in
        show_zone_boundaries (boolean): Adds polygon(s) with annotation to zone(s)
        x_extend (tuple): Extend the plot along x-axis
        y_extend (tuple): Extend of the plot along y-axis
        flamingo(bool: Is there a flamingo?
        plot_family (bool): Plot family shapes?
        ts_posterior_freq (float): If zones are annotated this threshold
        size (int): size of points
        fname (str): a path followed by a the name of the file
    """

    class TrueZone(object):
        pass

    class TrueZoneHandler(object):
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent+10, handlebox.ydescent+10

            patch = patches.Polygon([[x0, y0],
                                     [x0 + 40, y0+20],
                                     [x0+60, y0-10],
                                     [x0+50, y0-20],
                                     [x0+30, y0-20]],
                                    ec='black', lw=1, ls='-', alpha=1, fill=False,
                                    joinstyle="round", capstyle="butt")
            handlebox.add_artist(patch)
            return patch

    # general plotting parameters
    pp = get_plotting_params(plot_type="plot_posterior_map_simulated")

    plt.rcParams["axes.linewidth"] = pp['frame_width']
    fig, ax = plt.subplots(figsize=(pp['fig_width'], pp['fig_height']), constrained_layout=True)

    # getting zones data from mcmc results
    zones = mcmc_res['zones']

    # # computing network and getting locations and distance matrix
    # if subset:
    #     # subsetting the sites
    #     is_in_subset = [x == 1 for x in sites['subset']]
    #     sites_all = deepcopy(sites)
    #
    #     for key in sites.keys():
    #
    #         if type(sites[key]) == list:
    #             sites[key] = list(np.array(sites[key])[is_in_subset])
    #
    #         else:
    #             sites[key] = sites[key][is_in_subset, :]
    #
    net = compute_network(sites)
    #
    locations, dist_mat = net['locations'], net['dist_mat']
    #
    # plotting all points
    cmap, _ = get_cmap(0.5, name='YlOrRd', lower_ts=1.2)
    ax.scatter(*locations.T, s=size, c=[cmap(0)], alpha=1, linewidth=0)
    #
    # if subset:
    #     # plot all points not in the subset
    #     not_in_subset = np.logical_not(is_in_subset)
    #     other_locations = sites_all['locations'][not_in_subset]
    #     ax.scatter(*other_locations.T, s=size, c=[cmap(0)], alpha=1, linewidth=0)
    #
    #     # add boundary of subset to plot
    #     x_coords, y_coords = locations.T
    #     offset = 100
    #     x_min, x_max = min(x_coords) - offset, max(x_coords) + offset
    #     y_min, y_max = min(y_coords) - offset, max(y_coords) + offset
    #     bbox_width = x_max - x_min
    #     bbox_height = y_max - y_min
    #     bbox = mpl.patches.Rectangle((x_min, y_min), bbox_width, bbox_height, ec='grey', fill=False,
    #                                  lw=1.5, linestyle='-.')
    #     ax.add_patch(bbox)
    #     ax.text(x_max, y_max+200, 'Subset', fontsize=18, color='#000000')
    #
    # # plotting minimum spanningtree for each zone
    leg_zones = []
    zone_labels = []

    for i, zone in enumerate(zones):
        zone_colors = ['#7570b3', '#1b9e77', '#d95f02', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
        if flamingo:
            flamingo_color = '#F48AA7'
            c = flamingo_color if len(zones) == 1 else zone_colors[i]
        else:
            c = zone_colors[i]

        try:
            is_in_zone = add_minimum_spanning_tree_new(ax, zone, locations, dist_mat, burn_in, post_freq_lines,
                                                       size=size, color=c)
        except:
            continue
        line = Line2D([], [], color=c, lw=2, linestyle='-', marker="o", markeredgecolor=c, markerfacecolor=c,
                      markeredgewidth=1.2, markersize=6)
        leg_zones.append(line)
        zone_labels.append(f'$Z_{i + 1}$')


        try:
                # ann = f'$simulated \, Z_{i + 1}$' if len(zones) > 1 else f'$simulated \, z$'
            add_zone_boundary(ax, locations, net, mcmc_res['true_zones'][i], alpha=0.001, color='#000000')
        except:
            continue
        # Customize plotting layout

    # if plot_family:
    #     families = mcmc_res['true_families']
    #     family_colors = ['#b3e2cd', '#f1e2cc', '#cbd5e8', '#f4cae4', '#e6f5c9']
    #     # handles for legend
    #     handles = []
    #     for i, is_in_family in enumerate(families):
    #             # plot points belonging to family
    #
    #             family_color = family_colors[i]
    #
    #             family_alpha_shape = 0.001
    #             family_fill = family_color
    #             family_border = family_color
    #             alpha_shape = compute_alpha_shapes([is_in_family], net, family_alpha_shape)
    #             # print(is_in_family, net, alpha_shape)
    #             smooth_shape =alpha_shape.buffer(60, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
    #             # smooth_shape = alpha_shape
    #             patch = PolygonPatch(smooth_shape, fc=family_fill, ec=family_border, lw=1, ls='-', alpha=1, fill=True,
    #                                  zorder=-i)
    #             ax.add_patch(patch)
    #
    #             # adding legend handle
    #             handle = Patch(facecolor=family_color, edgecolor=family_color, label="Simulated family")
    #             handles.append(handle)
    #
    # # styling the axes
    # style_axes(ax, locations, show=show_axes, offset=frame_offset, x_extend=x_extend, y_extend=y_extend)
    #
    # # adding a legend to the plot
    # legend_zones = ax.legend(
    #     leg_zones,
    #     zone_labels,
    #     numpoints=2,
    #     title_fontsize=18,
    #     title='Contact areas ($Z$) in posterior distribution',
    #     frameon=True,
    #     edgecolor='#ffffff',
    #     framealpha=1,
    #     fontsize=16,
    #     ncol=2,
    #     columnspacing=1,
    #     loc='upper right',
    #     bbox_to_anchor=(0.01, 0.02)
    # )
    # ax.add_artist(legend_zones)
    # # # North - East arrows
    # # arr_e_t = (0.95, 1.8)
    # # arr_e_h = (1.45, 1.8)
    # # arr_n_t = (1, 1.85)
    # # arr_n_h = (1, 2.35)
    # # arrow_east = patches.FancyArrowPatch(arr_e_t, arr_e_h, mutation_scale=8,
    # #                                      transform=fig.dpi_scale_trans, lw=0.5, ec="black", fc="black")
    # # arrow_north = patches.FancyArrowPatch(arr_n_t, arr_n_h, mutation_scale=8,
    # #                                      transform=fig.dpi_scale_trans, lw=0.5, ec="black", fc="black")
    # # ax.add_patch(arrow_east)
    # # ax.add_patch(arrow_north)
    # # ax.text(arr_e_h[0]+0.05, arr_e_t[1] - 0.05, "East", fontsize=12, transform=fig.dpi_scale_trans)
    # # ax.text(arr_n_h[0]-0.2, arr_n_h[1] + 0.05, "North", fontsize=12, transform=fig.dpi_scale_trans)
    #
    # if show_zone_boundaries:
    #     legend_true_zones = ax.legend([TrueZone()], ['Bounding polygon of true $Z$'],
    #                                   handler_map={TrueZone: TrueZoneHandler()},
    #                                   bbox_to_anchor=(0.35, 0.045),
    #                                   title_fontsize=16,
    #                                   loc='upper_right',
    #                                   frameon=True,
    #                                   edgecolor='#ffffff',
    #                                   handletextpad=4,
    #                                   fontsize=18,
    #                                   ncol=1,
    #                                   columnspacing=1)
    #     ax.add_artist(legend_true_zones)
    #
    # if plot_family:
    #     legend_families = ax.legend(
    #         handles=handles,
    #         title_fontsize=16,
    #         fontsize = 18,
    #         frameon = True,
    #         edgecolor = '#ffffff',
    #         framealpha = 1,
    #         ncol = 1,
    #         columnspacing = 1,
    #         handletextpad=4,
    #         loc='upper right',
    #         bbox_to_anchor=(0.35, 0)
    #     )
    #     ax.add_artist(legend_families)


    #fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    #plt.close(fig)

    # if zone boundaries is on we also create a zoomed in plot for each contact zone

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    for i, zone in enumerate(zones):

            # first we compute the extend of the contact zone to compute the figure width height ration
            locations_zone = locations[is_in_zone, :]
            x_coords, y_coords = locations_zone.T
            offset = 200
            x_extend = (min(x_coords) - offset, max(x_coords) + offset)
            y_extend = (min(y_coords) - offset, max(y_coords) + offset)
            print(x_extend)
            print(y_extend)

            # compute width and height of zoomed in figure
            width_ratio = 1. / (x_max - x_min) * (x_extend[1] - x_extend[0])
            height_ratio = 1. / (y_max - y_min) * (y_extend[1] - y_extend[0])
            print(width_ratio, height_ratio)

            fig_size_factor = 6
            fig_width = width_ratio * pp['fig_width'] * fig_size_factor
            fig_height = height_ratio * pp['fig_height'] * fig_size_factor
            print(fig_width, fig_height)

            fig, ax_zone = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
            size_factor = 1.5

            ax_zone.scatter(*locations.T, s=size*size_factor, c=[cmap(0)], alpha=1, linewidth=0)

            zone_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
            flamingo_color = '#F48AA7'
            c = flamingo_color if len(zones) == 1 else zone_colors[i]
            is_in_zone = add_minimum_spanning_tree_new(ax_zone, zone, locations, dist_mat, burn_in, post_freq_lines,
                                                       size=size*1.5, size_line=size_line, color=c)
            add_zone_boundary(ax_zone, locations, net, is_in_zone, alpha=0.001, color='#000000')


            style_axes(ax_zone, locations, show=show_axes, x_extend=x_extend, y_extend=y_extend)

            ax_zone.set_xlim(x_extend)
            ax_zone.set_ylim(y_extend)

            fig.savefig(f"{fname}_z{i + 1}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
            plt.close(fig)


def plot_posterior_map(mcmc_res, sites, post_freq_lines, burn_in=0.2, x_extend=None, y_extend=None, simulated_data=False,
                       experiment=None,  bg_map=False, proj4=None, geojson_map=None, geo_json_river=None, subset=False,
                       lh_single_zones=False, flamingo=False, simulated_family=False, size=25, size_line=3,
                       label_languages=False, add_overview=False, x_extend_overview=None, y_extend_overview=None,
                       frame_offset=None, show_axes=True,
                       families=None, family_names=None,
                       family_alpha_shape=None, return_correspondence=False,
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
        x_extend (tuple): (min, max)-extend of the map in x-direction (longitude) --> Olga: move to config
        y_extend (tuple): (min, max)-extend of the map in y-direction (latitude) --> and move to config
        simulated_data(bool): are the plots for real-world or simulated data?
        experiment(str): either "sa" or "balkan", will load different plotting parameters. Olga: Should go to plotting
                         config file instead.
        bg_map (bool: Plot a background map? --> Olga: to config
        geojson_map(str): File path to geoJSON background map --> Olga: to config
        proj4(str): Coordinate reference system of the language data. --> Olga: Should go to config. or could be passed as meta data to the sites.
        geo_json_river(str): File path to river data. --> Olga: should go to config
        subset(boolean): Is there a subset in the data, which should be displayed differently?
                         Only relevant for one experiment. --> Olga Should go to config
        lh_single_zones(bool): Add box containing information about the likelihood of single areas to the plot?
        flamingo(bool): Sort of a joke. Does one area have the shape of a flamingo. If yes use flamingo colors for plotting.
        simulated_family(bool): Only for simulated data. Are families also simulated?
        size(float): Size of the dots (languages) in the plot --> Olga: move to config.
        size_line(float): Line thickness. Gives in combination with post_freq_lines the line thickness of the edges in an area
                          Olga -> should go to config.
        label_languages(bool): Label the languages in areas?
        add_overview(bool): Add an overview map?
        x_extend_overview(tuple): min, max)-extend of the overview map in x-direction (longitude) --> Olga: config
        y_extend_overview(tuple): min, max)-extend of the overview map in y-direction (latitude) --> Olga: config
        families(np.array): a boolean assignment of sites to families
            shape(n_families, n_sites)
        family_names(dict): a dict comprising both the external (Arawak, Chinese, French, ...) and internal (0,1,2...)
                            family names
        family_alpha_shape(float): controls how far languages of the same family have to be apart to be grouped
                                   into a single alpha shape (for display only)  --> Olga: config
        fname (str): a path of the output file.
        return_correspondence(bool): return the labels of all languages which are shown in the map
                                    --> Olga: I think this can be solved differently, with a seprate function.
        show_axes(bool): show x- and y- axis? --> I think we can hardcode this to false.
        frame_offset(float): offset of x and y- axis --> If show_axes is False this is not needed anymore

    """

    # Is the function used for simulated data or real-world data? Both require different plotting parameters.
    # for Olga: should go to config file
    if simulated_data:
        pp = get_plotting_params(plot_type="plot_posterior_map_simulated")

    # if for real world-data: South America or Balkans?
    # for Olga: plotting parameters in pp should be defined in the config file. Can go.
    else:
        if experiment == "sa":
            pp = get_plotting_params(plot_type="plot_posterior_map_sa")
        if experiment == "balkan":
            pp = get_plotting_params(plot_type="plot_posterior_map_balkan")

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
    cmap, _ = get_cmap(0.5, name='YlOrRd', lower_ts=1.2)
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
        ax.text(x_max, y_max+200, 'Subset', fontsize=18, color='#000000')

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
        is_in_zone = add_minimum_spanning_tree_new(ax, zone, locations, dist_mat, burn_in, post_freq_lines,
                                                   size=size, size_line=size_line, color=c)
        # This adds small lines to the legend (one legend entry per area)
        line = Line2D([0], [0], color=c, lw=6, linestyle='-')
        leg_zones.append(line)

        # Again, this is only relevant for simulated data and should go into a separate function
        if simulated_data:
            try:
                # Adds a bounding box for the ground truth areas showing if the algorithm has correctly identified them

                add_zone_boundary(ax, locations, net, mcmc_res['true_zones'][i], alpha=0.001, color='#000000')
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
                ax.annotate(labels_in_zone[l]+1, **anno_opts)

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
            prop_s = int(post_freq_lines[k-1] * 100)
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
        axins.scatter(*locations.T, s=size/2, c=[cmap(0)], alpha=1, linewidth=0)

        # adds a bounding box around the overview map
        bbox_width = x_extend[1] - x_extend[0]
        bbox_height = y_extend[1] - y_extend[0]
        bbox = mpl.patches.Rectangle((x_extend[0], y_extend[0]), bbox_width, bbox_height, ec='k', fill=False, linestyle='-')
        axins.add_patch(bbox)

    # Depending on the background (sa, balkan, simulated), we want to place additional legend entries
    # at different positions in the map in order not to block map content and best use the available space.
    # This should rather go to the config file.
    # Unfortunately, positions have to be provided in map units, which makes things a bit opaque.
    # Once in the config, the functions below can go.
    # for Sa map
    if experiment =="sa":
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
                x_unit * 25, y_unit*100,
                color="white"
            ))
        ax.axhline(y_extend[0] + y_unit * 56, 0.02, 0.20, lw=1.5, color="black")
        ax.axhline(y_extend[0] + y_unit * 72, 0.02, 0.20, lw=1.5, color="black")

    # for simulated data
    if simulated_data:
        x_unit = (x_extend[1] - x_extend[0])/100
        y_unit = (y_extend[1] - y_extend[0])/100
        ax.add_patch(
            patches.Rectangle(
                (x_extend[0], y_extend[0]),
                x_unit*55,
                y_unit*30,
                color="white"
            ))
        # The legend looks a bit different, as it has to show both the inferred areas and the ground truth
        ax.annotate("INFERRED", (x_extend[0] + x_unit*3, y_extend[0] + y_unit*23), fontsize=20)
        ax.annotate("GROUND TRUTH", (x_extend[0] + x_unit*38.5, y_extend[0] + y_unit*23), fontsize=20)
        ax.axvline(x_extend[0] + x_unit*37, 0.05, 0.18, lw=2, color="black")

    # If families and family names are provided, this adds an overlay color for all language families in the map
    # including a legend entry.
    # Should go to a separate function

    if families is not None and family_names is not None:
        # Family colors, should probably go to config
        #family_colors = ['#377eb8', '#4daf4a', '#984ea3', '#a65628', '#f781bf', '#999999', '#ffff33', '#e41a1c', '#ff7f00']
        family_colors = ['#b3e2cd', '#f1e2cc', '#cbd5e8', '#f4cae4', '#e6f5c9', '#d3d3d3']

        # Initialize empty legend handle
        handles = []

        # Iterate over all family names
        for i, family in enumerate(family_names['external']):
            # print(i, family)

            # Find all languages belonging to a family
            is_in_family = families[i] == 1
            family_locations = locations[is_in_family,:]
            family_color = family_colors[i]

            # Adds a color overlay for each language in a family
            ax.scatter(*family_locations.T, s=size*15, c=family_color, alpha=1, linewidth=0, zorder=-i, label=family)

            family_fill, family_border = family_color, family_color

            # For languages with more than three members: instead of one dot per language,
            # combine several languages in an alpha shape (a polygon)
            if family_alpha_shape is not None and np.count_nonzero(is_in_family) > 3:
                alpha_shape = compute_alpha_shapes([is_in_family], net, family_alpha_shape)

                # making sure that the alpha shape is not empty
                if not alpha_shape.is_empty:
                    smooth_shape = alpha_shape.buffer(40000, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
                    patch = PolygonPatch(smooth_shape, fc=family_fill, ec=family_border, lw=1, ls='-', alpha=1, fill=True, zorder=-i)
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
                alpha_shape = compute_alpha_shapes([is_in_family], net, family_alpha_shape)
                # print(is_in_family, net, alpha_shape)
                smooth_shape =alpha_shape.buffer(60, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
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
            a = x / 10**b
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
        for i,_ in enumerate(zones):
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

    style_axes(ax, locations, show=show_axes, offset=frame_offset, x_extend=x_extend, y_extend=y_extend)

    # Save the plot
    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)

    # Should the labels displayed in the map be returned? These are later added as a separate legend (
    # outside this hell of a function)
    if return_correspondence and label_languages:
        return all_labels


def plot_empty_map(sites, x_extend=None, y_extend=None,
                   show_axes=True, experiment=None,
                   bg_map=False, proj4=None, geojson_map=None, geo_json_river=None,
                   add_overview=False, frame_offset=None, x_extend_overview=None, y_extend_overview=None,
                   size=25, fname='mst_posterior'):
    """ This function plots an empty map of the study area
    Args:

    """

    pp = get_plotting_params(plot_type="plot_empty_map")
    plt.rcParams["axes.linewidth"] = pp['frame_width']
    fig, ax = plt.subplots(figsize=(pp['fig_width'], pp['fig_height']), constrained_layout=True)

    if bg_map:
        if proj4 is None and geojson_map is None:
            raise Exception('If you want to use a map provide a geojson and a crs')

        world = gpd.read_file(geojson_map)
        world = world.to_crs(proj4)
        world.plot(ax=ax, color='w', edgecolor='black', zorder=-100000)

        if geo_json_river is not None:

            rivers = gpd.read_file(geo_json_river)
            rivers = rivers.to_crs(proj4)
            rivers.plot(ax=ax, color=None, edgecolor="skyblue", zorder=-10000)

    net = compute_network(sites)
    locations, dist_mat = net['locations'], net['dist_mat']

    # plotting all points
    cmap, _ = get_cmap(0.5, name='YlOrRd', lower_ts=1.2)
    ax.scatter(*locations.T, s=size, c=[cmap(0)], alpha=1, linewidth=0)

    for l in range(len(locations)):
        x, y = locations[l]
        if experiment == "sa":
            if sites['id'][l] + 1 in [3, 8]:
                x += -50000
            elif sites['id'][l] + 1 in [21, 36, 38, 22, 25, 45, 49, 57, 68, 83, 84, 88, 89, 100]:
                x += -60000
            elif sites['id'][l] + 1 in [37]:
                x += -30000
            elif sites['id'][l] + 1 in [64]:
                x += -80000
            else:
                x += 20000
            if sites['id'][l] + 1 in [12, 74, 82, 61, 65, 55, 64, 85]:
                y += -10000
            elif sites['id'][l] + 1 in [21, 36, 38, 22, 25, 45, 49, 57, 37, 68, 83, 84, 88, 89, 100]:
                y += 20000
            else:
                y += 10000

        if experiment == "balkan":
            if sites['id'][l] + 1 in [11, 17]:
                x += -40000
            else:
                x += 10000
            if sites['id'][l] + 1 in [14, 3]:
                y += -5000
            elif sites['id'][l] + 1 in [11, 6]:
                y += -15000
            else:
                y += 5000
        anno_opts = dict(xy=(x, y), fontsize=12, color="black")

        ax.annotate(sites['id'][l]+1, **anno_opts)

    if add_overview:

        axins = inset_axes(ax, width=3.8, height=4, bbox_to_anchor=pp['overview_position'],
                           loc='lower left', bbox_transform=ax.transAxes)
        axins.tick_params(labelleft=False, labelbottom=False, length=0)

        axins.set_xlim(x_extend_overview)
        axins.set_ylim(y_extend_overview)

        if proj4 is not None and geojson_map is not None:

            world = gpd.read_file(geojson_map)
            world = world.to_crs(proj4)
            world.plot(ax=axins, color='w', edgecolor='black', zorder=-100000)

        axins.scatter(*locations.T, s=size/2, c=[cmap(0)], alpha=1, linewidth=0)

        # add bounding box of plot
        bbox_width = x_extend[1] - x_extend[0]
        bbox_height = y_extend[1] - y_extend[0]
        bbox = mpl.patches.Rectangle((x_extend[0], y_extend[0]), bbox_width, bbox_height, ec='k', fill=False, linestyle='-')
        axins.add_patch(bbox)

    # styling the axes
    style_axes(ax, locations, show=show_axes, offset=frame_offset, x_extend=x_extend, y_extend=y_extend)

    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    print("ready")
    plt.close(fig)





def plot_posterior_frequency_map_new(mcmc_res, net, labels=False, families=False, family_names=False, nz=-1, burn_in=0.2,
                            bg_map=False, proj4=None, geojson_map=None, family_alpha_shape=0.00001,
                            geo_json_river=None, extend_params=None,
                            size=20, fname='posterior_frequency', ts_low_frequency=0.5,
                            ts_posterior_freq=0.8, show_zone_bbox=False, zone_bbox_offset=1,
                                     show_axes=True):
    """ This function creates a scatter plot of all sites in the posterior distribution. The color of a site reflects
    its frequency in the posterior

    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        net (dict): The full network containing all sites.
        nz (int): For multiple zones: which zone should be plotted? If -1, plot all.
        burn_in (float): Percentage of first samples which is discarded as burn-in
        plot_family (str): Visualize all sites belonging to a family (either "alha_shapes", "color" or None)
        family_alpha_shape (float): Alpha value passed to the function compute_alpha_shapes
        family_color (str): Color of family in plot
        bg_map (bool): Use a background map for for the visualization?
        proj4 (str): projection information when using a background map
        geojson_map (str): file location of the background map
        geo_json_river (str): file location of river data (for making the background map a bit nicer)
        offset_factor (float): map extent is tailored to the location of the sites. This defines the offset.
        plot_edges (bool): Plot the edges of the mst triangulation for the zone?
        labels (bool): Plot the labels of the families?
        labels_offset (float, float): Offset of the labels in both x and y
        size (int): size of the points
        fname (str): a path followed by a the name of the file
    """



    # gemeral plotting parameters
    pp = get_plotting_params()

    plt.rcParams["axes.linewidth"] = pp['frame_width']
    fig = plt.figure(figsize=(pp['fig_width'], pp['fig_height']), constrained_layout=True)

    # getting figure axes
    ax, cbar_axes = get_axes(fig)

    # getting mcmc data and locations of points
    zones = mcmc_res['zones']
    n_zones = len(zones)
    locations, dist_mat = net['locations'], net['dist_mat']
    # print(locations)

    if bg_map:
        if proj4 is None and geojson_map is None:
            raise Exception('If you want to use a map provide a geojson and a crs')

        world = gpd.read_file(geojson_map)
        world = world.to_crs(proj4)
        # world.plot(ax=ax, color=(.95,.95,.95), edgecolor='grey')
        world.plot(ax=ax, color='w', edgecolor='black', zorder=-100000)

        if geo_json_river is not None:

            rivers = gpd.read_file(geo_json_river)
            rivers = rivers.to_crs(proj4)
            rivers.plot(ax=ax, color=None, edgecolor="skyblue", zorder=-10000)


    # adding scatter plot and corresponding colorbar legend
    cmap, norm = get_cmap(ts_low_frequency, name='YlOrRd', lower_ts=0.2)
    add_posterior_frequency_points(ax, zones, locations, ts_low_frequency, cmap, norm, nz=nz, burn_in=burn_in, size=size)
    add_posterior_frequency_legend(fig, cbar_axes, ts_low_frequency, cmap, norm, ts_posterior_freq, fontsize=pp['fontsize'])

    if nz == -1:
        for i, zone in enumerate(zones):
            print(f'Zone {i + 1}', end=' ')
            add_minimum_spanning_tree(ax, zone, locations, dist_mat, burn_in, ts_posterior_freq, cmap, norm, lw=1, size=size/2)
            continue
    else:
        add_minimum_spanning_tree(ax, zones[nz-1], locations, dist_mat, burn_in, ts_posterior_freq, cmap, norm, lw=1, size=size / 2)
        pass

    leg_zone = False
    if show_zone_bbox:
        leg_zone = add_zone_bbox(ax, zones, locations, net, nz, n_zones, burn_in, ts_posterior_freq, zone_bbox_offset)

    # styling the axes
    limits = style_axes(
        ax,
        locations,
        None,
        show = False,
        fontsize = pp['fontsize'],
        x_offsets = extend_params['lng_offsets'],
        y_offsets = extend_params['lat_offsets']
    )

    # add overview
    add_overview = True
    if add_overview:

        axins = inset_axes(ax, width=3.8, height=4, loc=3)
        axins.tick_params(labelleft=False, labelbottom=False, length=0)

        axins.set_xlim(extend_params['lng_extend_overview'])
        axins.set_ylim(extend_params['lat_extend_overview'])

        if proj4 is None and geojson_map is None:
            raise Exception('If you want to use a map provide a geojson and a crs')

        world = gpd.read_file(geojson_map)
        world = world.to_crs(proj4)
        # world.plot(ax=ax, color=(.95,.95,.95), edgecolor='grey')
        world.plot(ax=axins, color='w', edgecolor='black', zorder=-100000)

        # rivers = gpd.read_file(geo_json_river)
        # rivers = rivers.to_crs(proj4)
        # rivers.plot(ax=axins, color=None, edgecolor="skyblue", zorder=-10000)

        add_posterior_frequency_points(axins, zones, locations, ts_low_frequency, cmap, norm, nz=nz, burn_in=burn_in, size=size/2)


        # add bounding box of plot
        x_min, x_max, y_min, y_max = limits
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        bbox = mpl.patches.Rectangle((x_min, y_min), bbox_width, bbox_height, ec='k', fill=False, linestyle='-')
        # bbox = mpl.patches.Rectangle((x_min, y_min), 1000, 1000)
        axins.add_patch(bbox)

        # axins.set_title('Overview study area', fontsize=pp['fontsize'])

    # annotating languages
    if ('names' in net.keys()) and labels:
        names = net['names']
        for i, name in enumerate(names):
            if name in labels:
                x, y = locations[i,:]
                x += 20000; y += 10000
                # print(i, name)
                anno_opts = dict(xy=(x, y), fontsize=14, color='k')
                ax.annotate(name, **anno_opts)
                # ax.scatter(*xy, s=50, color='green')

    if list(families) and list(family_names):
        family_colors = ['#377eb8', '#4daf4a', '#984ea3', '#a65628', '#f781bf', '#999999', '#ffff33', '#e41a1c', '#ff7f00']
        family_colors = ['#b3e2cd', '#f1e2cc', '#cbd5e8', '#f4cae4', '#e6f5c9']
        family_leg = []

        # handles for legend
        handles = []

        for i, family in enumerate(family_names['external']):
            # print(i, family)

            # plot points belonging to family
            is_in_family = families[i] == 1
            family_locations = locations[is_in_family,:]
            family_color = family_colors[i]

            # debugging stuff
            # print(f'Number of points in family {family_locations.shape}')
            ax.scatter(*family_locations.T, s=size*5, c=family_color, alpha=1, linewidth=0, zorder=0, label=family)

            family_fill, family_border = family_color, family_color

            # language family has to have at least 3 members
            if np.count_nonzero(is_in_family) > 3:
                alpha_shape = compute_alpha_shapes([is_in_family], net, family_alpha_shape)

                # making sure that the polygon is not empty
                if not alpha_shape.is_empty:
                    # print(len(is_in_family))
                    # print(is_in_family, net, alpha_shape)
                    # print(alpha_shape)
                    # print(type(alpha_shape))
                    # print(alpha_shape.is_empty)
                    smooth_shape = alpha_shape.buffer(30000, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
                    # smooth_shape = alpha_shape
                    patch = PolygonPatch(smooth_shape, fc=family_fill, ec=family_border, lw=1, ls='-', alpha=1, fill=True, zorder=-1)
                    leg_family = ax.add_patch(patch)


            # adding legend handle
            handle = Patch(facecolor=family_color, edgecolor=family_color, label=family)
            handles.append(handle)

    # ax.legend(title='Language family', title_fontsize=pp['fontsize'], frameon=True, edgecolor='#ffffff', framealpha=1, fontsize=pp['fontsize'], loc='upper left', ncol=1, columnspacing=1)
    ax.legend(
        handles = handles,
        title = 'Language family',
        title_fontsize = 22,
        fontsize = 22,
        frameon = True,
        edgecolor = '#ffffff',
        framealpha = 1,
        loc = 'upper left',
        ncol = 1,
        columnspacing = 1
    )

    # adding a legend to the plot
    if leg_zone:
        ax.legend((leg_zone,), ('Contact zone',), frameon=False, fontsize=pp['fontsize'], loc='upper right', ncol=1, columnspacing=1)

    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)



def compute_alpha_shapes(sites, net, alpha):

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


def plot_trace_recall_precision(mcmc_res, steps_per_sample, burn_in=0.2, recall=True, precision=True, fname='trace_recall_precision'):
    """
    Function to plot the trace of the MCMC chains both in terms of likelihood and recall
    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        steps_per_sample (int): How many steps did the MCMC take between each two samples?
        burn_in (float): First n% of samples are burn-in
        recall (boolean): plot recall?
        precision (boolean): plot precision?
        fname (str): a path followed by a the name of the file
    """

    pp = get_plotting_params(plot_type="plot_trace_recall_precision")

    plt.rcParams["axes.linewidth"] = pp['frame_width']

    fig, ax = plt.subplots(figsize=(pp['fig_width'], pp['fig_height']))

    # Recall
    if recall:
        y = mcmc_res['recall']
        x = range(len(y))
        x = [i * steps_per_sample for i in x]
        # col['trace']['recall']
        ax.plot(x, y, lw=pp['line_thickness'], color='#e41a1c', label='Recall')

    # Precision
    if precision:
        y = mcmc_res['precision']
        x = range(len(y))
        x = [i * steps_per_sample for i in x]
        # col['trace']['precision']
        ax.plot(x, y, lw=pp['line_thickness'], color='#377eb8', label='Precision')

    ax.set_ylim(bottom=0)

    # Find index of first sample after burn-in
    end_bi = math.ceil(len(x) *steps_per_sample * burn_in)
    end_bi_label = math.ceil(len(x) *steps_per_sample * (burn_in - 0.03))

    color_burn_in = 'grey'
    ax.axvline(x=end_bi, lw=pp['line_thickness'], color=pp['color_burn_in'], linestyle='--')
    plt.text(end_bi_label, 0.5, 'Burn-in', rotation=90, size=pp['fontsize'], color=pp['color_burn_in'])

    xmin, xmax = 0, x[-1] + steps_per_sample
    ax.set_xlim([xmin, xmax])
    n_ticks = 6 if int(burn_in * 100) % 20 == 0 else 12
    x_ticks = np.linspace(xmin, xmax, n_ticks)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{x_tick:.0f}' for x_tick in x_ticks], fontsize=pp['fontsize'])

    y_min, y_max, y_step = 0, 1, 0.2
    ax.set_ylim([y_min, y_max + (y_step / 2)])
    y_ticks = np.arange(y_min, y_max + y_step, y_step)
    ax.set_yticks(y_ticks)
    y_ticklabels = [f'{y_tick:.1f}' for y_tick in y_ticks]
    y_ticklabels[0] = '0'
    ax.set_yticklabels(y_ticklabels, fontsize=pp['fontsize'])

    f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
    plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(g))

    ax.set_xlabel('Iteration', fontsize=pp['fontsize'], fontweight='bold')
    ax.legend(loc=4, prop={'size': pp['fontsize']}, frameon=False)

    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)



def plot_traces(recall, precision, fname='trace_recalls_precisions'):
    """
    Function to plot the trace of the MCMC chains both in terms of likelihood and recall
    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        burn_in (float): First n% of samples are burn-in
        recall (boolean): plot recall?
        precision (boolean): plot precision?
        fname (str): a path followed by a the name of the file
    """

    pp = get_plotting_params(plot_type="plot_traces")
    print(pp)
    plt.rcParams["axes.linewidth"] = pp['frame_width']
    fig, ax = plt.subplots(figsize=(pp['fig_width'], pp['fig_height']))

    n_zones = 7

    # Recall
    x = list(range(len(recall)))
    y = recall
    ax.plot(x, y, lw=pp['line_thickness'], color='#e41a1c', label='Recall')

    # Precision
    x = list(range(len(precision)))
    y = precision
    ax.plot(x, y, lw=pp['line_thickness'], color='#377eb8', label='Precision')

    ax.set_ylim(bottom=0)

    x_min, x_max = 0, len(recall)

    ax.set_xlim([x_min, x_max])
    n_ticks = n_zones + 1
    x_ticks = np.linspace(x_min, x_max, n_ticks)
    x_ticks_offset = 500
    x_ticks = [x_tick - x_ticks_offset for x_tick in x_ticks if x_tick > 0]
    ax.set_xticks(x_ticks)
    x_ticklabels = [f'{x_ticklabel:.0f} areas' for x_ticklabel in np.linspace(1, 7, n_zones)]
    x_ticklabels[0] = '1 area'
    # x_ticklabels = [f'{x_ticklabel:.0f}' for x_ticklabel in np.linspace(1, 7, n_zones)]
    # ax.set_xlabel('Number of zones', fontsize=pp['fontsize'], fontweight='bold')
    ax.set_xticklabels(x_ticklabels, fontsize=pp['fontsize'])

    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(which='minor', axis='x', color='#000000', linestyle='-')
    ax.set_axisbelow(True)


    y_min, y_max, y_step = 0, 1, 0.2
    ax.set_ylim([y_min, y_max + (y_step / 2)])
    y_ticks = np.arange(y_min, y_max + y_step, y_step)
    ax.set_yticks(y_ticks)
    y_ticklabels = [f'{y_tick:.1f}' for y_tick in y_ticks]
    y_ticklabels[0] = '0'
    ax.set_yticklabels(y_ticklabels, fontsize=pp['fontsize'])

    ax.legend(loc=4, prop={'size': pp['fontsize']}, frameon=True, framealpha=1, facecolor='#ffffff', edgecolor='#ffffff')

    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)


def plot_dics(dics, simulated_data=False, threshold=False, fname='DICs'):
    """This function plots dics. What did you think?
    Args:
        dics(dict): A dict of DICs from different models

    """
    if simulated_data:
        pp = get_plotting_params(plot_type="plot_dics_simulated")
    else:
        pp = get_plotting_params(plot_type="plot_dics")

    plt.rcParams["axes.linewidth"] = pp['frame_width']

    fig, ax = plt.subplots(figsize=(pp['fig_width'], pp['fig_height']))

    x = list(dics.keys())
    y = list(dics.values())


    ax.plot(x, y, lw=pp['line_thickness'], color='#000000', label='DIC')


    y_min, y_max = min(y), max(y)

    # round y min and y max to 1000 up and down, respectively
    n_digits = len(str(int(y_min))) - 1
    convertor = 10 ** (n_digits - 2)
    y_min_old, y_max_old = y_min, y_max
    y_min = int(np.floor(y_min / convertor) * convertor)
    y_max = int(np.ceil(y_max / convertor) * convertor)

    ax.set_ylim([y_min, y_max])
    y_ticks = np.linspace(y_min, y_max, 6)
    ax.set_yticks(y_ticks)
    yticklabels = [f'{y_tick:.0f}' for y_tick in y_ticks]
    ax.set_yticklabels(yticklabels, fontsize=pp['fontsize'])

    if threshold:
        ax.axvline(x=threshold, lw=pp['line_thickness'], color=pp['color_burn_in'], linestyle='-')
        ypos_label = y_min + (y_max - y_min) / 2
        # ax.text(threshold, ypos_label, 'threshold', rotation=90, size=pp['fontsize'], color=pp['color_burn_in'])

    ax.set_ylabel('DIC', fontsize=pp['fontsize'], fontweight='bold')

    x_min, x_max = min(x), max(x)
    ax.set_xlim([x_min, x_max])
    x_ticks = np.linspace(x_min, x_max, len(x))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([int(x_tick) for x_tick in x_ticks], fontsize=pp['fontsize'])

    ax.set_xlabel('Number of areas', fontsize=pp['fontsize'], fontweight='bold')


    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)


def plot_trace_lh(mcmc_res, steps_per_sample, burn_in=0.2, true_lh=True,  fname='trace_likelihood.png'):
    """
    Function to plot the trace of the MCMC chains both in terms of likelihood and recall
    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        steps_per_sample (int): How many steps did the MCMC take between each two samples?
        burn_in: (float): First n% of samples are burn-in
        true_lh (boolean): Visualize the true likelihood
        fname (str): a path followed by a the name of the file
    """

    pp = get_plotting_params(plot_type="plot_trace_lh")

    plt.rcParams["axes.linewidth"] = pp['frame_width']


    fig, ax = plt.subplots(figsize=(pp['fig_width'], pp['fig_height']))
    n_zones = len(mcmc_res['zones'])
    y = mcmc_res['lh']
    x = list(range(len(y)))
    x = [i * steps_per_sample for i in x]

    if true_lh:
        ax.axhline(y=mcmc_res['true_lh'], xmin=x[0], xmax=x[-1], lw=2, color='#fdbf6f', linestyle='-', label='True')
    ax.plot(x, y, lw=pp['line_thickness'], color='#6a3d9a', linestyle='-', label='Posterior')

    y_min, y_max = min(y), max(y)

    # round y min and y max to 100 up and down, respectively
    n_digits = len(str(int(y_min))) - 1
    convertor = 10 ** (n_digits - 3)
    y_min_old, y_max_old = y_min, y_max
    y_min = int(np.floor(y_min / convertor) * convertor)
    y_max = int(np.ceil(y_max / convertor) * convertor)

    # add burn-in line and label
    end_bi = math.ceil(len(x)*steps_per_sample * burn_in)
    end_bi_label = math.ceil(len(x) *steps_per_sample * (burn_in - 0.03))

    ax.axvline(x=end_bi, lw=pp['line_thickness'], color=pp['color_burn_in'], linestyle='--')
    ypos_label = y_min + (y_max - y_min) / 2
    ax.text(end_bi_label, ypos_label, 'Burn-in', rotation=90, size=pp['fontsize'], color=pp['color_burn_in'])

    ax.set_ylim([y_min, y_max])
    y_ticks = np.linspace(y_min, y_max, 6)
    ax.set_yticks(y_ticks)
    yticklabels = [f'{y_tick:.0f}' for y_tick in y_ticks]
    ax.set_yticklabels(yticklabels, fontsize=pp['fontsize'])

    xmin, xmax, xstep = 0, x[-1] + steps_per_sample,  steps_per_sample * 200
    ax.set_xlim([xmin, xmax])
    xticks = np.arange(xmin, xmax+xstep, xstep)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=pp['fontsize'])

    f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(g))
    plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(g))

    ax.set_xlabel('Iteration', fontsize=pp['fontsize'], fontweight='bold')

    if n_zones == 1:
        yaxis_label = "log-likelihood"
    else:
        yaxis_label = "log-likelihood"
    ax.set_ylabel(yaxis_label, fontsize=pp['fontsize'], fontweight='bold')

    ax.legend(loc=4, prop={'size': pp['fontsize']}, frameon=False)


    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)


def plot_trace_lh_with_prior(mcmc_res,  steps_per_sample, burn_in=0.2, lh_range=None, prior_range=None, labels=None, fname='trace_likelihood'):
    """
    Function to plot the trace of the MCMC chains both in terms of likelihood and recall
    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        steps_per_sample (int): How many steps did the MCMC take between each two samples?
        burn_in: (float): First n% of samples are burn-in
        true_lh (boolean): Visualize the true likelihood
        fname (str): a path followed by a the name of the file
    """

    pp = get_plotting_params(plot_type="plot_trace_lh_with_prior")

    plt.rcParams["axes.linewidth"] = pp['frame_width']

    n_zones = len(mcmc_res['zones'])

    fig, ax1 = plt.subplots(figsize=(pp['fig_width'], pp['fig_height']))

    x = list(range(len(mcmc_res['lh'])))
    x = [i * steps_per_sample for i in x]

    # create shared x axis
    xmin, xmax, xstep = 0, x[-1] + steps_per_sample, steps_per_sample * 200

    ax1.set_xlim([xmin, xmax])
    xticks = np.arange(xmin, xmax+xstep, xstep)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks, fontsize=pp['fontsize'])

    ax1.set_xlabel('Iteration', fontsize=pp['fontsize'], fontweight='bold')

    # create first y axis showing likelihood
    color_lh = 'tab:red'
    lh = mcmc_res['lh']
    ax1.plot(x, lh, lw=pp['line_thickness'], color=color_lh, linestyle='-', label=labels[0])

    if lh_range is None:
        y_min, y_max = min(lh), max(lh)

        # round y min and y max to 100 up and down, respectively
        n_digits = len(str(int(y_min))) - 1
        convertor = 10 ** (n_digits - 3)
        y_min_old, y_max_old = y_min, y_max
        y_min = int(np.floor(y_min / convertor) * convertor)
        y_max = int(np.ceil(y_max / convertor) * convertor)
    else:
        y_min, y_max = lh_range


    ax1.set_ylim([y_min, y_max])
    y_ticks = np.linspace(y_min, y_max, 6)
    ax1.set_yticks(y_ticks)
    yticklabels = [f'{y_tick:.0f}' for y_tick in y_ticks]
    ax1.set_yticklabels(yticklabels, fontsize=pp['fontsize'], color=color_lh)

    yaxis_label = 'log-likelihood'
    ax1.set_ylabel(yaxis_label, fontsize=pp['fontsize'], fontweight='bold', color=color_lh)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color_prior = 'tab:blue'
    prior = mcmc_res['prior']
    ax2.plot(x, prior, lw=pp['line_thickness'], color=color_prior, linestyle='-', label=labels[1])
    yaxis_label = 'prior'
    ax2.set_ylabel(yaxis_label, fontsize=pp['fontsize'], fontweight='bold', color=color_prior)

    if prior_range is None:
        prior_range = (min(prior), max(prior))
    y_min, y_max = prior_range
    ax2.set_ylim([y_min, y_max])
    y_ticks = np.linspace(y_min, y_max, 6)
    ax2.set_yticks(y_ticks)
    yticklabels = [f'{y_tick:.0f}' for y_tick in y_ticks]
    ax2.set_yticklabels(yticklabels, fontsize=pp['fontsize'], color=color_prior)

    # add burn-in line and label
    end_bi = math.ceil(len(x) * steps_per_sample * burn_in)
    end_bi_label = math.ceil(len(x) * steps_per_sample * (burn_in - 0.03))

    ax2.axvline(x=end_bi, lw=pp['line_thickness'], color=pp['color_burn_in'], linestyle='--')
    ypos_label = y_min + (y_max - y_min) / 4
    ax2.text(end_bi_label, ypos_label, 'Burn-in', rotation=90, size=pp['fontsize'], color=pp['color_burn_in'])

    f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
    plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(g))

    # ax1.legend(loc=4, prop={'size': pp['fontsize']}, frameon=False)
    # ax1.legend([lh_handle, prior_handle])

    # ask matplotlib for the plotted objects and their labels
    #lh_handle, _  = ax1.get_legend_handles_labels()
    #lh_label = "log-likelihood"
    #prior_handle, _ = ax2.get_legend_handles_labels()
    #prior_label = "prior"
    #ax2.legend(lh_handle + prior_handle, lh_label + prior_label, loc=4, prop={'size': pp['fontsize']}, frameon=False)

    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)


def plot_histogram_weights(mcmc_res, feature):
    """
        Plots the trace for weight samples
        Args:
            mcmc_res (dict): the output from the MCMC neatly collected in a dict
            feature (int): plot weight for which feature?
        """
    fig, ax = plt.subplots()
    col = get_colors()
    n_weights = len(mcmc_res['weights'])

    # Global weight
    weights = []
    for w in mcmc_res['weights']:
        weights.append(w[feature][1])

    y = weights
    x = range(len(y))
    ax.hist(y, bins=None, range=None)
    #ax.plot(x, y, lw=0.75, color=col['trace']['recall'], label='Weights')

    plt.show()


def plot_zone_size_over_time(mcmc_res, r=0, burn_in=0.2, true_zone=True, fname='zone_size_over_time'):
    """
         Function to plot the zone size in the posterior
         Args:
             mcmc_res (dict): the output from the MCMC neatly collected in a dict
             r (int): which run?
             burn_in: (float): First n% of samples are burn-in
             fname (str): a path followed by a the name of the file
    """

    # colors = get_colors()['zones']['in_zones']
    pp = get_plotting_params()

    plt.rcParams["axes.linewidth"] = pp['frame_width']
    zone_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']

    fig, ax = plt.subplots(figsize=(pp['fig_width'], pp['fig_height']))

    x_mid = [] # label position
    y_max = 0 # y range
    n_zones = len(mcmc_res['zones'])

    for c in range(n_zones):
        size = []

        label = 'True' if n_zones == 1 else f'True (Zone {c})'
        colors = ('#6a3d9a', '#fdbf6f') if n_zones == 1 else (zone_colors[c],) * 2
        linestyle = ('-', '-' ) if n_zones == 1 else ('-', '-.')

        for z in mcmc_res['zones'][c]:
            size.append(np.sum(z))

        x = range(len(size))
        if true_zone:
            true_size = np.sum(mcmc_res['true_zones'][c])
            ax.axhline(y=true_size, xmin=x[0], xmax=x[-1], lw=pp['line_thickness'], color=colors[1],
                       linestyle=linestyle[1], label=label)

        ax.plot(x, size, lw=pp['line_thickness'], color=colors[0], linestyle=linestyle[0], label="Posterior")

        max_size, min_size = max(size), min(size)
        y_max = max_size if max_size > y_max else y_max
        x_mid.append(max_size - min_size)



    # Find index of first sample after burn-in
    end_bi = math.ceil(len(x) * burn_in)
    end_bi_label = math.ceil(len(x) * (burn_in - 0.03))

    y_min = 0
    # round y max to next 10
    n_digits = len(str(y_max))
    convertor = 10 ** (n_digits - 1) if n_digits <= 2 else 10 ** (n_digits - 2)
    y_max_old = y_max
    y_max = int(np.ceil(y_max / convertor) * convertor)


    ax.axvline(x=end_bi, lw=pp['line_thickness'], color=pp['color_burn_in'], linestyle='--')
    ypos_label = y_min + (y_max - y_min) / 2
    ax.text(end_bi_label, ypos_label, 'Burn-in', rotation=90, size=pp['fontsize'], color=pp['color_burn_in'])



    ax.set_ylim(bottom=0)


    xmin, xmax, xstep = 0, 1000, 200
    ax.set_xlim([xmin, xmax])
    xticks = np.arange(xmin, xmax+xstep, xstep)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=pp['fontsize'])




    y_ticks = np.linspace(y_min, y_max, 6)
    ax.set_yticks(y_ticks)
    yticklabels = [f'{y_tick:.0f}' for y_tick in y_ticks]
    yticklabels[0] = '0'
    ax.set_yticklabels(yticklabels, fontsize=pp['fontsize'])

    ax.set_xlabel('Iteration', fontsize=pp['fontsize'], fontweight='bold')
    ax.set_ylabel('Zone size', fontsize=pp['fontsize'], fontweight='bold')

    ax.legend(loc=4, prop={'size': pp['fontsize']}, frameon=False)

    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)


def plot_correlation_weights(mcmc_res,  burn_in=0.2, which_weight="global"):
    """This function plots the correlation between the mean of the estimated weights and the true weights
    Args:
        mcmc_res(dict):  the output from the MCMC, neatly collected in a dict
        burn_in (float): ratio of initial samples that are discarded as burn-in
        which_weight(str): compute correlation for which weight? ("global", "contact", "inheritance")
    """
    if which_weight == "global":
        weight_idx = 0

    elif which_weight == "contact":
        weight_idx = 1

    elif which_weight == "inheritance":
        weight_idx = 2

    else:
        raise ValueError('weight must be "global", "contact" or "inheritance" ')

    fig, ax = plt.subplots()

    # Find index of first sample after burn-in

    end_bi = math.ceil(len(mcmc_res['weights']) * burn_in)
    #end = math.ceil(len(mcmc_res['weights']) * 0.005)

    weights = np.asarray(mcmc_res['weights'][end_bi:])
    w_est = weights[:, :, weight_idx]
    w_mean_est = w_est.mean(axis=0)

    w_true = mcmc_res['true_weights'][:, weight_idx]
    slope, intercept, r_value, p_value, std_err = linregress(w_true, w_mean_est)
    line = slope * w_true + intercept

    # todo: compute range of
    #
    # p_range_per_feature = []
    # for f in range(len(mcmc_res['true_p_global'])):
    #     p_range_per_feature.append(mcmc_res['true_p_global'][f])

    ax.plot(w_true, w_mean_est, 'o')
    ax.plot(w_true, line)

    ax.set_aspect('equal')
    ax.scatter(w_true, w_mean_est)
    ax.set_xlabel('True weights')
    ax.set_ylabel('Mean estimated weights')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.show()


# todo: finish
def plot_correlation_p(mcmc_res,  which_p, burn_in=0.2, which_nr=0, which_cat=0):
    """This function plots the correlation between the mean of the estimated weights and the true weights
    Args:
        mcmc_res(dict):  the output from the MCMC, neatly collected in a dict
        burn_in (float): ratio of initial samples that are discarded as burn-in
        which_p(str): compute correlation for "p_zones" or "p_families"?
        which_nr(int): in case of many zones or families, which one should be visualized? (default=0)
        which_cat(int): which category?
    """
    if which_p == "p_zones" or which_p == "p_families":
        p_est = mcmc_res[which_p][which_nr]
        p_true = mcmc_res["true_" + str(which_p)][which_nr]

    else:
        raise ValueError('which_p must be "p_zones", or "p_families" ')

    p_est_out = []
    for p in p_est:
        f_out = []
        for f in p:
            f_out.append(f[which_cat])
        p_est_out.append(f_out)
    p_est = p_est_out

    p_true_out = []

    for f in p_true:
        p_true_out.append(f[which_cat])
    p_true = np.asarray(p_true_out)

    fig, ax = plt.subplots()

    # Find index of first sample after burn-in

    end_bi = math.ceil(len(p_est) * burn_in)
    p_est = np.asarray(p_est[end_bi:])

    p_mean_est = p_est.mean(axis=0)

    slope, intercept, r_value, p_value, std_err = linregress(p_true, p_mean_est)
    line = slope * p_true + intercept

    ax.plot(p_true, p_mean_est, 'o')
    ax.plot(p_true, line)

    ax.set_aspect('equal')
    ax.scatter(p_true, p_mean_est)
    ax.set_xlabel('True p')
    ax.set_ylabel('Mean estimated p')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.show()


def plot_correspondence_table(sites, fname, sites_in_zone=None, ncol=3):
    """ Which language belongs to which number? This table will tell you
    Args:
        sites(dict): dict of all languages
        fname(str): name of the figure
        sites_in_zone(list): list of sites per zone
        ncol(int); number of columns in the output table
    """
    pp = get_plotting_params()
    fig, ax = plt.subplots()

    if sites_in_zone is not None:
        sites_id = []
        sites_names = []
        s = [j for sub in sites_in_zone for j in sub]

        for i in range(len(sites['id'])):
            if i in s:
                sites_id.append(sites['id'][i])
                sites_names.append(sites['names'][i])

    else:
        sites_id = sites['id']
        sites_names = sites['names']

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    n_col = ncol
    n_row = math.ceil(len(sites_names)/n_col)

    l = [[] for _ in range(n_row)]

    for i in range(len(sites_id)):
        col = i%n_row
        nr = str(sites_id[i] + 1)
        l[col].append(nr)
        l[col].append(sites_names[i])

    # Fill up empty cells
    for i in range(len(l)):
        if len(l[i]) != n_col*2:
            fill_up_nr = n_col*2 - len(l[i])
            for f in range(fill_up_nr):
                l[i].append("")

    widths = [0.05, 0.3] * int(((len(l[0]))/2))

    table = ax.table(cellText=l, loc='center', cellLoc="left", colWidths=widths)
    table.set_fontsize(40)

    for key, cell in table.get_celld().items():
        cell.set_linewidth(0)
    fig.tight_layout()

    if sites_in_zone is not None:
        nz = len(sites_in_zone)
        fig.savefig(f"{fname}_nz{nz}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])

    else:
        fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
