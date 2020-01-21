#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns
plt.style.use('seaborn-paper')
plt.tight_layout()

import numpy as np
import math

from src.util import zones_autosimilarity, add_edge, compute_delaunay, colorline, compute_mst_posterior
from src.util import bounding_box, round_int, linear_rescale, round_single_int, round_multiple_ints
from src.preprocessing import compute_network
from scipy.stats import gamma, linregress
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.ops import cascaded_union, polygonize
from shapely import geometry
from descartes import PolygonPatch
from copy import deepcopy
import geopandas as gpd
from decimal import Decimal
import os
os.environ["PROJ_LIB"] = "C:/Users/ranacher/Anaconda3/Library/share"


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


def plot_triangulation_edges(samples, net, triangulation, ax=None):
    """ This function adds a triangulation to the points in the posterior

    Args:
        samples (np.array): the samples from the MCMC
        net (dict): The full network containing all sites.
        triangulation (str): type of triangulation, either "mst" or "delaunay"
        ax (axis): matlibplot axis
    """

    if ax is None:
        ax = plt.gca()

    all_sites = net['locations']
    # todo: change to np.any

    zone_locations = all_sites[samples[0]]
    dist_mat = net['dist_mat']

    delaunay = compute_delaunay(zone_locations)
    mst = delaunay.multiply(dist_mat)

    if triangulation == "delaunay":
        tri = delaunay

    elif triangulation == "mst":
        tri = mst

    print(tri.shape, "tri")
    # Get data
    col = get_colors()
    adj_mat = net['adj_mat']
    locations = net['locations']

    # Add edge evidence for the sampled zone
    n_samples, n_v = samples.shape
    weights = np.ones(n_samples)
    edge_counts = (weights[:, None] * samples).T.dot(samples)
    edge_counts *= adj_mat.toarray()
    edge_freq = (edge_counts / n_samples).clip(0, 1)
    edge_freq[edge_freq < 0.1] = 0.

    # Plot background
    size = 4
    bg=plt.scatter(*locations.T, s=size, color=col['zones']['background_nodes'])
    print('Nonzero:', np.count_nonzero(edge_freq))

    # Plot posterior
    edges = np.argwhere(edge_freq)
    line_col = col['zones']['triangulation']
    line_col = [(line_col + (c,)) for c in edge_freq[edges[:, 0], edges[:, 1]]]

    lines = LineCollection(locations[edges], colors=line_col)
    ax.add_collection(lines)

    # # Remove axes
    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])

    # # Add legend
    # ax.legend([bg, lines], ['All sites', 'Edges in posterior distribution'], frameon=False, fontsize=10)

    return ax


def plot_posterior(samples, net, ax=None):
    """ This function plots the posterior distribution of contact zones

    Args:
        samples (np.array): the samples from the MCMC
        net (dict): The full network containing all sites.
    """
    if ax is None:
        ax = plt.gca()

    # Get data
    col = get_colors()
    adj_mat = net['adj_mat']
    locations = net['locations']

    # Add edge evidence for the sampled zone
    n_samples, n_v = samples.shape
    weights = np.ones(n_samples)
    edge_counts = (weights[:, None] * samples).T.dot(samples)
    edge_counts *= adj_mat.toarray()
    edge_freq = (edge_counts / n_samples).clip(0, 1)
    edge_freq[edge_freq < 0.1] = 0.

    # Plot background
    size = 4
    bg=plt.scatter(*locations.T, s=size, color=col['zones']['background_nodes'])
    print('Nonzero:', np.count_nonzero(edge_freq))

    # Plot posterior
    edges = np.argwhere(edge_freq)
    line_col = col['zones']['triangulation']
    line_col = [(line_col + (c,)) for c in edge_freq[edges[:, 0], edges[:, 1]]]

    lines = LineCollection(locations[edges], colors=line_col)
    ax.add_collection(lines)

    # # Remove axes
    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])

    # # Add legend
    # ax.legend([bg, lines], ['All sites', 'Edges in posterior distribution'], frameon=False, fontsize=10)

    return ax


def plot_zone(zone, net):
    """ This function plots a contact zone proposed by the the MCMC

    Args:
        zone (np.array): The current zone (boolean array).
        net (dict): The full network containing all sites.
    """

    # Initialize plot
    fig, ax = plt.subplots()
    col = get_colors()
    all_sites = net['locations']
    size = 4
    bg = ax.scatter(*all_sites.T, s=size, color=col['zone']['background_nodes'])
    zo = ax.scatter(*all_sites[zone].T, s=size*3, color=col['zone']['triangulation'])

    # Remove axes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend
    ax.legend([bg, zo], ['All sites', 'Sites in proposed contact zone'], frameon=False, fontsize=10)
    plt.show()


def plot_zones(zones, net, ax=None):
    """ This function plots the parallel contact zones proposed by the MCMC

    Args:
        zones (np.array): The current zone (boolean array).
        net (dict): The full network containing all sites.
    """

    # Initialize plot
    if ax is None:
        return_ax = False
        _, ax = plt.subplots()
    else:
        return_ax = True

    col = get_colors()
    all_sites = net['locations']
    size = 4
    bg = ax.scatter(*all_sites.T, s=size, color=col['zones']['background_nodes'])
    zo = []

    if isinstance(zones, dict):
        zones = list(zones.values())

    for z, zone in enumerate(zones):

        zo.append(ax.scatter(*all_sites[zones[z]].T, s=size * 6, color=col['zones']['in_zones'][int(z)]))

    # Remove axes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend
    ax.legend([bg, zo[0]], ['All sites', 'Sites in proposed contact zones'], frameon=False, fontsize=10)

    if return_ax:
        return ax
    else:
        plt.show()


# def plot_posterior_density(zones, net):
#     # TODO: Labels, Color palette
#
#     """ This function creates a kernel density plot of all sites in the posterior distribution
#
#     Args:
#         zones (np.array): the posterior of all zones
#         net (dict): The full network containing all sites.
#     """
#
#     all_sites = net['locations']
#     points_in_post = []
#
#     for z in zones:
#         try:
#             points_in_post += all_sites[z[0]]
#
#         except ValueError:
#             points_in_post = all_sites[z[0]]
#
#     points_in_post = np.array(points_in_post)
#     ax = sns.kdeplot(points_in_post, cmap="Reds", shade=True, shade_lowest=False)
#
#     # Remove axes
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
#
#     plt.show()



# helper functions for posterior frequency plotting functions (vanilla, family and map)
def get_plotting_params():
    """ Here we store various plotting parameters

    Returns:
        (dict): a dictionary comprising various plotting parameters
    """
    return {
        'fig_width': 20,
        'fig_height': 10,
        'fontsize': 22, # overall fontsize value
        'line_thickness': 2, # thickness of lines in plots
        'frame_width': 1.5, # width of frame of plots
        'color_burn_in': 'grey', # color of burn in line and text
        'save_format': 'svg', # output format for plots
        'n_fragments': 100, # number of fragments for minimum spanning tree edges
    }

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
        smooth_shape = alpha_shape.buffer(160, resolution=16, cap_style=1, join_style=1, mitre_limit=10.0)
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

def add_minimum_spanning_tree_new(ax, zone, locations, dist_mat, burn_in, ts_posterior_freq, size=25, color='#ff0000'):


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

    # factor for sizes
    size_factor = 4

    for index, freq in np.ndenumerate(mst_posterior_freq):
        if freq >= ts_posterior_freq:

            # compute line width
            min_lw = size / 100
            freq_norm = (freq - ts_posterior_freq) / (1 - ts_posterior_freq)
            lw = min_lw + freq_norm * size_factor

            # getting locations of the two contact points and plotting line
            locs = locations[[*index]]
            ax.plot(*locs.T, color=color, lw=lw)

    # getting boolean array of points in minimum spanning tree
    is_in_mst = points_posterior_freq > ts_posterior_freq

    # plotting points of minimum spanning tree
    cp_locations = locations[is_in_mst, :]
    ax.scatter(*cp_locations.T, s=size * size_factor, c=color)

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
    if leg_zone:
        ax.legend((leg_zone,), ('Contact zone',), frameon=False, fontsize=pp['fontsize'], loc='upper right', ncol=1, columnspacing=1)

    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)



def plot_posterior_frequency_family(mcmc_res, net, nz=-1, burn_in=0.2, show_zone_bbox=False, zone_bbox_offset=200,
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
        zone_bbox_offset (int): Offsets.
        ts_posterior_freq (float): If zones are annotated this threshold
        ts_low_frequency (float): Colorbar offset for points with low frequency.
        frame_offset (int): Offsets.
        show_axes (bool): Whether to show the axes or not.
        size (int): size of points
        fname (str): a path followed by a the name of the file
    """

    pp = get_plotting_params()

    # gemeral visualization parameters
    fontsize = 22
    frame_width = 1.5
    plt.rcParams["axes.linewidth"] = frame_width
    width, height = 15, 10

    # initializing figure
    fig = plt.figure(figsize=(width,height), constrained_layout=True)


    ax, cbar_axes = get_axes(fig)


    # getting mcmc data and locations of points
    zones = mcmc_res['zones']
    n_zones = len(zones)
    locations, dist_mat = net['locations'], net['dist_mat']


    # adding scatter plot and corresponding colorbar legend
    cmap, norm = get_cmap(ts_low_frequency, name='YlOrRd', lower_ts=0.2)
    add_posterior_frequency_points(ax, zones, locations, ts_low_frequency, cmap, norm, nz=nz, burn_in=burn_in, size=size)
    add_posterior_frequency_legend(fig, cbar_axes, ts_low_frequency, cmap, norm, ts_posterior_freq, fontsize=fontsize)


    # adding family information (alpha shapes), if available
    if 'true_families' in mcmc_res.keys():
        # print(f'{fname} has true familiy information')
        families = mcmc_res['true_families']
        family_colors = ['#377eb8', '#4daf4a', '#984ea3', '#a65628', '#f781bf', '#999999', '#ffff33', '#e41a1c', '#ff7f00']

        for ifamily, is_in_family in enumerate(families):

            # plot points belonging to family
            family_locations = locations[is_in_family,:]
            family_color = family_colors[ifamily]

            # debugging stuff
            # print(f'Number of points in family {family_locations.shape}')
            # ax.scatter(*family_locations.T, s=size/10, c='k', alpha=1, linewidth=0, edgecolor='black')

            # 0.004
            family_alpha_shape = 0.001
            family_fill = family_color
            family_border = family_color
            alpha_shape = compute_alpha_shapes([is_in_family], net, family_alpha_shape)
            # print(is_in_family, net, alpha_shape)
            # smooth_shape = alpha_shape.buffer(100, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
            smooth_shape = alpha_shape.buffer(100, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
            # smooth_shape = alpha_shape
            patch = PolygonPatch(smooth_shape, fc=family_fill, ec=family_border, lw=1, ls='-', alpha=1, fill=True, zorder=-1)
            leg_family = ax.add_patch(patch)

    leg_zone = False
    if show_zone_bbox:
        leg_zone = add_zone_bbox(ax, zones, locations, net, nz, n_zones, burn_in, ts_posterior_freq, zone_bbox_offset)

    for zone in zones:
        add_minimum_spanning_tree(ax, zone, locations, dist_mat, burn_in, ts_posterior_freq, cmap, norm, size=size/2)


    # styling the axes
    style_axes(ax, locations, frame_offset, show=show_axes, fontsize=fontsize)

    # adding a legend to the plot
    if not leg_zone:
        ax.legend((leg_family,), ('Simulated family',), frameon=False, fontsize=fontsize, loc='upper right', ncol=1, columnspacing=1)
    else:
        ax.legend((leg_zone, leg_family), ('Contact zone', 'Simulated family',), frameon=False, fontsize=fontsize, loc='upper right', ncol=1,
                  columnspacing=1)

    # saving the figure
    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)
    # end of plot posterior frequency family function


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
    add_posterior_frequency_legend(fig, cbar_axes, ts_low_frequency, cmap, norm, ts_posterior_freq, fontsize=pp['fontsize'])

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
    style_axes(ax, extend_locations, frame_offset, show=show_axes, fontsize=pp['fontsize'])

    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)



def plot_posterior_frequency_map(mcmc_res, net, nz=-1, burn_in=0.2, plot_family=None, family_alpha_shape=None,
                             family_color=None, bg_map=False, proj4=None, geojson_map=None,
                             geo_json_river=None, offset_factor=0.03, plot_edges=False,
                             labels=False, labels_offset=None, size=20, cmap=plt.cm.get_cmap('jet'),
                             annotate_zones=False, fname='posterior_frequency'):
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

    fontsize = 24
    frame_width = 1.5


    fig, ax = plt.subplots(figsize=(15, 10))

    zones = mcmc_res['zones']
    n_zones = len(zones)

    # getting positions and name of points
    positions = net['locations'].T
    names = net['names']

    # Find index of first sample after burn-in
    if bg_map:
        if proj4 is None and geojson_map is None:
            raise Exception('If you want to use a map provide a geojson and a crs')

        world = gpd.read_file(geojson_map)
        world = world.to_crs(proj4)
        # world.plot(ax=ax, color=(.95,.95,.95), edgecolor='grey')
        world.plot(ax=ax, color='w', edgecolor='black')

        if geo_json_river is not None:
            rivers = gpd.read_file(geo_json_river)
            rivers = rivers.to_crs(proj4)
            rivers.plot(ax=ax, color=None, edgecolor="skyblue")

    # plot all zones
    if nz == -1:
        # get samples from all zones
        n_samples = len(zones[0])
        zones = [sum(k) for k in zip(*zones)]

        # exclude burn-in
        end_bi = math.ceil(len(zones) * burn_in)
        density = (np.sum(zones[end_bi:], axis=0, dtype=np.int32) / (n_samples - end_bi))

    # plot only one zone (passed as argument)
    else:
        # get samples of the zone
        zone = zones[nz - 1]
        n_samples = len(zone)

        # exclude burn-in
        end_bi = math.ceil(n_samples * burn_in)

        # compute frequency of each point in that zone
        density = (np.sum(zone[end_bi:], axis=0, dtype=np.int32) / (n_samples - end_bi))

    # sorting points according to their frequency in the posterior
    # -> high frequency points are plotted over low frequency points
    X, Y = positions
    positions_sorted = [[c for _, c in sorted(zip(density, coords), key=lambda pair: pair[0])] for coords in positions]

    density_sorted = sorted(density)
    density_sorted = list(map(lambda x: x * 100, density_sorted))

    # adding scatter plot to axes
    ax.scatter(*positions_sorted, c=density_sorted, s=size, cmap=cmap, vmin=0, alpha=0.6, linewidths=0)

    # Add labels for those sites which occur most often in the posterior
    if labels:
        if labels_offset is None:
            labels_offset = (10., 10.)
        for i, name in enumerate(names):
            if density[i] > 0.1:
                plt.annotate(name, all_sites[i] + [labels_offset[0], labels_offset[1]], zorder=11, fontsize=9)

    # Customize plotting layout
    if plot_family == "alpha_shapes":
        alpha_shape = compute_alpha_shapes(mcmc_res['true_families'], net, family_alpha_shape)
        smooth_shape = alpha_shape.buffer(100, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
        patch = PolygonPatch(smooth_shape, fc=family_color, ec=family_color, alpha=0.5,
                             fill=True, zorder=-1)

        ax.add_patch(patch)

    # elif plot_family == "color":
    #     fam_sites = np.sum(mcmc_res['true_families'], axis=0, dtype=np.int32)
    #     fam_sites = np.ma.masked_where(fam_sites == 0, fam_sites)
    #     ax.scatter(*all_sites.T, c=fam_sites, s=size, cmap="inferno", zorder=-1)

    if plot_edges:
        plot_triangulation_edges(samples=np.array(zones[end_bi:]), net=net, triangulation="mst", ax=ax)
        # plot_posterior(np.array(zone[end_bi:]), net, ax=ax)

    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])

    if annotate_zones:
        # create list with all zone indices
        indices_zones = [nz - 1] if nz != -1 else range(n_zones)
        print(f'Zone indices: {indices_zones}')

        for zone_index in indices_zones:
            print()

    # getting axes ranges and rounding them
    x_min, x_max = np.min(positions[0, :]), np.max(positions[0, :])
    y_min, y_max = np.min(positions[1, :]), np.max(positions[1, :])
    x_min, x_max = round_int(x_min, 'down'), round_int(x_max, 'up')
    y_min, y_max = round_int(y_min, 'down'), round_int(y_max, 'up')

    # print(f'x range {x_min}--{x_max}')
    # print(f'y range {y_min}--{y_max}')

    # x axis
    ax.set_xlim([x_min, x_max])
    x_ticks = list(np.linspace(x_min, x_max, 6))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{x_tick:.0f}' for x_tick in x_ticks], fontsize=fontsize)

    # y axis
    ax.set_ylim([y_min, y_max])
    y_ticks = list(np.linspace(y_min, y_max, 6))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y_tick:.0f}' for y_tick in y_ticks], fontsize=fontsize)

    [ax.spines[side].set_linewidth(frame_width) for side in ['left', 'bottom', 'right', 'top']]
    plt.rcParams["axes.linewidth"] = frame_width

    # colorbar
    cbar_ticks = list(np.linspace(0, 1, 6))
    cbar_ticklabels = [f'{round(t * 100, 0):.0f}' for t in cbar_ticks]

    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="3%", pad=1, pack_start=True)
    fig.add_axes(cax)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1),
                                     orientation='horizontal', ticks=cbar_ticks)
    cbar.ax.set_xlabel('Frequency of point in posterior (%)', fontsize=fontsize, labelpad=-75)
    cbar.ax.set_xticklabels(cbar_ticklabels)
    cbar.ax.tick_params(labelsize=fontsize)

    # ax.set_title(title, fontsize=fontsize)

    fig.savefig(fname, bbox_inches='tight', dpi=400)
    # plt.show()


def plot_mst_posterior(mcmc_res, sites, subset=False, burn_in=0.2, show_zone_boundaries=False, x_extend=None,
                       y_extend=None, ts_posterior_freq=0.6, frame_offset=None,
                       show_axes=True, size=25, fname='mst_posterior'):
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
    fig, ax = plt.subplots(figsize=(pp['fig_width'], pp['fig_height']), constrained_layout=True)

    # getting zones data from mcmc results
    zones = mcmc_res['zones']

    # computing network and getting locations and distance matrix
    if subset:
        # subsetting the sites
        is_in_subset = sites['subset']
        sites_all = deepcopy(sites)

        for key in sites.keys():
            if type(sites[key]) == list:
                sites[key] = list(np.array(sites[key])[is_in_subset])
            else:
                sites[key] = sites[key][is_in_subset,:]

    net = compute_network(sites)
    locations, dist_mat = net['locations'], net['dist_mat']

    # plotting all points
    cmap, _ = get_cmap(0.5, name='YlOrRd', lower_ts=1.2)
    ax.scatter(*locations.T, s=size, c=[cmap(0)], alpha=1, linewidth=0)
    if subset:
        # plot all points not in the subset
        not_in_subset = np.logical_not(is_in_subset)
        other_locations = sites_all['locations'][not_in_subset]
        ax.scatter(*other_locations.T, s=size, c=[cmap(0)], alpha=1, linewidth=0)

        # add boundary of subset to plot
        x_coords, y_coords = locations.T
        offset = 100
        x_min, x_max = min(x_coords) - offset, max(x_coords) + offset
        y_min, y_max = min(y_coords) - offset, max(y_coords) + offset
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        bbox = mpl.patches.Rectangle((x_min, y_min), bbox_width, bbox_height, ec='k', fill=False, linestyle='-')
        ax.add_patch(bbox)
        ax.text(x_max, y_max, 'Subset', fontsize=18, color='#000000')

    # plotting minimum spanningtree for each zone
    leg_zone = None
    for i, zone in enumerate(zones):

        zone_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
        flamingo_color = '#F48AA7'
        c = flamingo_color if len(zones) == 1 else zone_colors[i]
        is_in_zone = add_minimum_spanning_tree_new(ax, zone, locations, dist_mat, burn_in, ts_posterior_freq, size=size, color=c)

        if show_zone_boundaries:
            ann = f'$simulated \, Z_{i + 1}$' if len(zones) > 1 else f'$simulated \, z$'
            leg_zone = add_zone_boundary(ax, locations, net, is_in_zone, alpha=0.0002, annotation=ann, color='#000000')

    # styling the axes
    style_axes(ax, locations, show=show_axes, offset=frame_offset, x_extend=x_extend, y_extend=y_extend)

    # adding a legend to the plot
    if leg_zone is not None:
        ax.legend((leg_zone,), ('Contact zone',), frameon=False, fontsize=pp['fontsize'], loc='upper right', ncol=1, columnspacing=1)

    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)

    # if zone boundaries is on we also create a zoomed in plot for each contact zone
    if show_zone_boundaries:
        x_min, x_max = ax.get_xlim()
        print(x_min, x_max)
        y_min, y_max = ax.get_ylim()
        print(y_min, y_max)

        for i, zone in enumerate(zones):

            # first we compute the extend of the contact zone to compute the figure width height ration
            locations_zone = locations[is_in_zone, :]
            x_coords, y_coords = locations_zone.T
            offset = 0
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
            size_factor = 5

            ax_zone.scatter(*locations.T, s=size*size_factor, c=[cmap(0)], alpha=1, linewidth=0)

            zone_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
            flamingo_color = '#F48AA7'
            c = flamingo_color if len(zones) == 1 else zone_colors[i]
            is_in_zone = add_minimum_spanning_tree_new(ax_zone, zone, locations, dist_mat, burn_in, ts_posterior_freq,
                                                       size=size*size_factor, color=c)
            add_zone_boundary(ax_zone, locations, net, is_in_zone, alpha=0.0002, color='#000000')


            style_axes(ax_zone, locations, show=show_axes, x_extend=x_extend, y_extend=y_extend)

            ax_zone.set_xlim(x_extend)
            ax_zone.set_ylim(y_extend)

            fig.savefig(f"{fname}_z{i + 1}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
            plt.close(fig)

def plot_mst_posterior_map(mcmc_res, sites, burn_in=0.2, x_extend=None,
                       y_extend=None, ts_posterior_freq=0.6, lh=None, frame_offset=None,
                       show_axes=True, size=25, labels=False, families=False, family_names=False,
                        bg_map=False, proj4=None, geojson_map=None, family_alpha_shape=None,
                        geo_json_river=None,
                        x_extend_overview=None, y_extend_overview=None, fname='mst_posterior'):
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


    # getting zones data from mcmc results
    zones = mcmc_res['zones']

    net = compute_network(sites)
    locations, dist_mat = net['locations'], net['dist_mat']

    # plotting all points
    cmap, _ = get_cmap(0.5, name='YlOrRd', lower_ts=1.2)
    ax.scatter(*locations.T, s=size, c=[cmap(0)], alpha=1, linewidth=0)

    # plotting minimum spanningtree for each zone
    leg_zones = []
    for i, zone in enumerate(zones):

        zone_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
        flamingo_color = '#F48AA7'
        c = flamingo_color if len(zones) == 1 else zone_colors[i]
        is_in_zone = add_minimum_spanning_tree_new(ax, zone, locations, dist_mat, burn_in, ts_posterior_freq, size=size, color=c)
        line = Line2D([0], [0], color=c, lw=6, linestyle='-')
        leg_zones.append(line)

    # add overview
    add_overview = True
    if add_overview:

        axins = inset_axes(ax, width=3.8, height=4, loc=3)
        axins.tick_params(labelleft=False, labelbottom=False, length=0)

        axins.set_xlim(x_extend_overview)
        axins.set_ylim(y_extend_overview)

        if proj4 is None and geojson_map is None:
            raise Exception('If you want to use a map provide a geojson and a crs')

        world = gpd.read_file(geojson_map)
        world = world.to_crs(proj4)
        world.plot(ax=axins, color='w', edgecolor='black', zorder=-100000)

        axins.scatter(*locations.T, s=size/2, c=[cmap(0)], alpha=1, linewidth=0)
        # add_posterior_frequency_points(axins, zones, locations, ts_low_frequency, cmap, norm, nz=nz, burn_in=burn_in, size=size/2)


        # add bounding box of plot
        bbox_width = x_extend[1] - x_extend[0]
        bbox_height = y_extend[1] - y_extend[0]
        bbox = mpl.patches.Rectangle((x_extend[0], y_extend[0]), bbox_width, bbox_height, ec='k', fill=False, linestyle='-')
        axins.add_patch(bbox)

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
            ax.scatter(*family_locations.T, s=size*15, c=family_color, alpha=1, linewidth=0, zorder=-i, label=family)

            family_fill, family_border = family_color, family_color

            # language family has to have at least 3 members
            if family_alpha_shape is not None and np.count_nonzero(is_in_family) > 3:
                alpha_shape = compute_alpha_shapes([is_in_family], net, family_alpha_shape)

                # making sure that the polygon is not empty
                if not alpha_shape.is_empty:
                    smooth_shape = alpha_shape.buffer(40000, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
                    patch = PolygonPatch(smooth_shape, fc=family_fill, ec=family_border, lw=1, ls='-', alpha=1, fill=True, zorder=-i)
                    leg_family = ax.add_patch(patch)


            # adding legend handle
            handle = Patch(facecolor=family_color, edgecolor=family_color, label=family)
            handles.append(handle)

    # ax.legend(title='Language family', title_fontsize=pp['fontsize'], frameon=True, edgecolor='#ffffff', framealpha=1, fontsize=pp['fontsize'], loc='upper left', ncol=1, columnspacing=1)
    legend_families = ax.legend(
        handles = handles,
        title = 'Language family',
        title_fontsize = 18,
        fontsize = 18,
        frameon = True,
        edgecolor = '#ffffff',
        framealpha = 1,
        ncol = 1,
        columnspacing = 1,
        loc = 'upper left',
        bbox_to_anchor = (0, 0.965)
    )
    ax.add_artist(legend_families)

    # adding a legend to the plot
    zone_labels = []
    for i, exp in enumerate(lh):
        lh_value = np.exp(exp)
        # print(lh_value)
        lh_value = f"{Decimal(lh_value):.2E}"
        zone_labels.append(f'$Z_{i + 1} \, {lh_value}$')

    legend_zones = ax.legend(
        leg_zones,
        zone_labels,
        title_fontsize=18,
        title='Contact zone',
        frameon=True,
        edgecolor='#ffffff',
        framealpha=1,
        fontsize=18,
        ncol=1,
        columnspacing=1,
        loc='upper left',
        bbox_to_anchor=(0, 0.7)
    )
    ax.add_artist(legend_zones)

    # styling the axes
    style_axes(ax, locations, show=show_axes, offset=frame_offset, x_extend=x_extend, y_extend=y_extend)

    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
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



def plot_posterior_frequency_map2(mcmc_res, net, nz=-1, burn_in=0.2, plot_family=None, family_alpha_shape=None,
                             family_color=None, bg_map=False, proj4=None, geojson_map=None,
                             geo_json_river=None, offset_factor=0.03, plot_edges=False,
                             labels=False, labels_offset=None, size=20, cmap=plt.cm.get_cmap('jet'),
                             annotate_zones=False, fname='posterior_frequency_map'):
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

    # defning main plot
    nrows, ncols = 100, 10
    height_ratio = 4
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)
    ax = fig.add_subplot(gs[:-height_ratio, :])

    # defining cbar axes
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

    # getting mcmc data and locations of points
    zones = mcmc_res['zones']
    n_zones = len(zones)
    locations = net['locations']

    # adding scatter plot and corresponding colorbar legend
    cmap, norm = get_cmap(ts_low_frequency, name='YlOrRd', lower_ts=0.2)
    add_posterior_frequency_points(ax, zones, locations, ts_low_frequency, cmap, norm, nz=nz, burn_in=burn_in,
                                   size=size)
    add_posterior_frequency_legend(fig, cbar_axes, ts_low_frequency, cmap, norm, ts_posterior_freq, title_pos=(0.5, 0),
                                   fontsize=pp['fontsize'])







    # Find index of first sample after burn-in
    if bg_map:
        if proj4 is None and geojson_map is None:
            raise Exception('If you want to use a map provide a geojson and a crs')

        world = gpd.read_file(geojson_map)
        world = world.to_crs(proj4)
        # world.plot(ax=ax, color=(.95,.95,.95), edgecolor='grey')
        world.plot(ax=ax, color='w', edgecolor='black')

        if geo_json_river is not None:
            rivers = gpd.read_file(geo_json_river)
            rivers = rivers.to_crs(proj4)
            rivers.plot(ax=ax, color=None, edgecolor="skyblue")

    # plot all zones
    if nz == -1:
        # get samples from all zones
        n_samples = len(zones[0])
        zones = [sum(k) for k in zip(*zones)]

        # exclude burn-in
        end_bi = math.ceil(len(zones) * burn_in)
        density = (np.sum(zones[end_bi:], axis=0, dtype=np.int32) / (n_samples - end_bi))

    # plot only one zone (passed as argument)
    else:
        # get samples of the zone
        zone = zones[nz - 1]
        n_samples = len(zone)

        # exclude burn-in
        end_bi = math.ceil(n_samples * burn_in)

        # compute frequency of each point in that zone
        density = (np.sum(zone[end_bi:], axis=0, dtype=np.int32) / (n_samples - end_bi))

    # sorting points according to their frequency in the posterior
    # -> high frequency points are plotted over low frequency points
    X, Y = positions
    positions_sorted = [[c for _, c in sorted(zip(density, coords), key=lambda pair: pair[0])] for coords in positions]

    density_sorted = sorted(density)
    density_sorted = list(map(lambda x: x * 100, density_sorted))

    # adding scatter plot to axes
    ax.scatter(*positions_sorted, c=density_sorted, s=size, cmap=cmap, vmin=0, alpha=0.6, linewidths=0)

    # Add labels for those sites which occur most often in the posterior
    if labels:
        if labels_offset is None:
            labels_offset = (10., 10.)
        for i, name in enumerate(names):
            if density[i] > 0.1:
                plt.annotate(name, all_sites[i] + [labels_offset[0], labels_offset[1]], zorder=11, fontsize=9)

    # Customize plotting layout
    if plot_family == "alpha_shapes":
        alpha_shape = compute_alpha_shapes(mcmc_res['true_families'], net, family_alpha_shape)
        smooth_shape = alpha_shape.buffer(100, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
        patch = PolygonPatch(smooth_shape, fc=family_color, ec=family_color, alpha=0.5,
                             fill=True, zorder=-1)

        ax.add_patch(patch)

    # elif plot_family == "color":
    #     fam_sites = np.sum(mcmc_res['true_families'], axis=0, dtype=np.int32)
    #     fam_sites = np.ma.masked_where(fam_sites == 0, fam_sites)
    #     ax.scatter(*all_sites.T, c=fam_sites, s=size, cmap="inferno", zorder=-1)

    if plot_edges:
        plot_triangulation_edges(samples=np.array(zones[end_bi:]), net=net, triangulation="mst", ax=ax)
        # plot_posterior(np.array(zone[end_bi:]), net, ax=ax)

    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])

    if show_zone_bbox:
        add_zone_bbox(ax, zones, locations, nz, n_zones, burn_in, ts_posterior_freq, zone_bbox_offset)


    # styling the axes
    style_axes(ax, locations, frame_offset, show=show_axes, fontsize=pp['fontsize'])

    fig.savefig(fname, bbox_inches='tight', dpi=400)




def f_score_mle(posterior):
    # Todo: Remove 0:20000
    # Todo: Add contact zone to posterior
    """ This function computes the precison, recall and f score of the maximum likelihood estimate in the posterior

    Args:
        posterior (tuple): the full posterior of a model

    Returns:
        (float): the precision of the model
        (float): the recall of the model
        (float): the f score of the model

    """
    contact_zones_idxs = get_contact_zones(6)
    n_zones = len(contact_zones_idxs)
    posterior[1]['true_zone'] = np.zeros((n_zones, network['n']), bool)
    for k, cz_idxs in enumerate(contact_zones_idxs.values()):
        posterior[1]['true_zone'][k, cz_idxs] = True

    m = max(posterior[1]['step_likelihoods'][0:20000])
    mle_pos = [i for i, j in enumerate(posterior[1]['step_likelihoods']) if j == m][0]

    best_zone = posterior[0][mle_pos]
    true_zone = posterior[1]['true_zone']

    true_positives = np.sum(np.logical_and(best_zone, true_zone), axis=1)[0]
    false_positives = np.sum(np.logical_and(best_zone, np.logical_not(true_zone)), axis=1)[0]
    false_negatives = np.sum(np.logical_and(np.logical_not(best_zone), true_zone), axis=1)[0]

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score


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


def plot_trace_recall_precision(mcmc_res, burn_in=0.2, recall=True, precision=True, fname='trace_recall_precision'):
    """
    Function to plot the trace of the MCMC chains both in terms of likelihood and recall
    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        burn_in (float): First n% of samples are burn-in
        recall (boolean): plot recall?
        precision (boolean): plot precision?
        fname (str): a path followed by a the name of the file
    """

    pp = get_plotting_params()

    plt.rcParams["axes.linewidth"] = pp['frame_width']


    fig, ax = plt.subplots(figsize=(pp['fig_width'], pp['fig_height']))

    # Recall
    if recall:
        y = mcmc_res['recall']
        x = range(len(y))
        # col['trace']['recall']
        ax.plot(x, y, lw=pp['line_thickness'], color='#e41a1c', label='Recall')

    # Precision
    if precision:
        y = mcmc_res['precision']
        x = range(len(y))
        # col['trace']['precision']
        ax.plot(x, y, lw=pp['line_thickness'], color='#377eb8', label='Precision')

    ax.set_ylim(bottom=0)


    # Find index of first sample after burn-in
    end_bi = math.ceil(len(y) * burn_in)
    end_bi_label = math.ceil(len(y) * (burn_in - 0.04))

    color_burn_in = 'grey'
    ax.axvline(x=end_bi, lw=pp['line_thickness'], color=pp['color_burn_in'], linestyle='--')
    plt.text(end_bi_label, 0.5, 'Burn-in', rotation=90, size=pp['fontsize'], color=pp['color_burn_in'])

    x_min, x_max = 0, 1000
    ax.set_xlim([x_min, x_max])
    n_ticks = 6 if int(burn_in * 100) % 20 == 0 else 12
    x_ticks = np.linspace(x_min, x_max, n_ticks)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{x_tick:.0f}' for x_tick in x_ticks], fontsize=pp['fontsize'])

    y_min, y_max, y_step = 0, 1, 0.2
    ax.set_ylim([y_min, y_max + (y_step / 2)])
    y_ticks = np.arange(y_min, y_max + y_step, y_step)
    ax.set_yticks(y_ticks)
    y_ticklabels = [f'{y_tick:.1f}' for y_tick in y_ticks]
    y_ticklabels[0] = '0'
    ax.set_yticklabels(y_ticklabels, fontsize=pp['fontsize'])


    ax.set_xlabel('Sample', fontsize=pp['fontsize'], fontweight='bold')

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

    pp = get_plotting_params()

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
    x_ticklabels = [f'{x_ticklabel:.0f} Zones' for x_ticklabel in np.linspace(1, 7, n_zones)]
    x_ticklabels[0] = '1 Zone'
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




def plot_dics(dics, threshold=False, fname='DICs'):
    """This function plots dics. What did you think?
    Args:
        dics(dict): A dict of DICs from different models

    """
    pp = get_plotting_params()

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

    ax.set_xlabel('Number of Zones', fontsize=pp['fontsize'], fontweight='bold')


    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)


def plot_trace_lh(mcmc_res, burn_in=0.2, true_lh=True, fname='trace_likelihood.png'):
    """
    Function to plot the trace of the MCMC chains both in terms of likelihood and recall
    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        burn_in: (float): First n% of samples are burn-in
        true_lh (boolean): Visualize the true likelihood
        fname (str): a path followed by a the name of the file
    """

    pp = get_plotting_params()

    plt.rcParams["axes.linewidth"] = pp['frame_width']


    fig, ax = plt.subplots(figsize=(pp['fig_width'], pp['fig_height']))

    n_zones = len(mcmc_res['zones'])

    y = mcmc_res['lh']
    x = range(len(y))
    color = 'red' # col['trace']['lh']



    if true_lh:
        ax.axhline(y=mcmc_res['true_lh'], xmin=x[0], xmax=x[-1], lw=2, color='#fdbf6f', linestyle='-', label='True')
    ax.plot(x, y, lw=pp['line_thickness'], color='#6a3d9a', linestyle='-', label='Predicted')


    y_min, y_max = min(y), max(y)





    # round y min and y max to 100 up and down, respectively
    n_digits = len(str(int(y_min))) - 1
    convertor = 10 ** (n_digits - 3)
    y_min_old, y_max_old = y_min, y_max
    y_min = int(np.floor(y_min / convertor) * convertor)
    y_max = int(np.ceil(y_max / convertor) * convertor)


    # add burn-in line and label
    end_bi = math.ceil(len(x) * burn_in)
    end_bi_label = math.ceil(len(x) * (burn_in - 0.03))

    ax.axvline(x=end_bi, lw=pp['line_thickness'], color=pp['color_burn_in'], linestyle='--')
    ypos_label = y_min + (y_max - y_min) / 2
    ax.text(end_bi_label, ypos_label, 'Burn-in', rotation=90, size=pp['fontsize'], color=pp['color_burn_in'])



    ax.set_ylim([y_min, y_max])
    y_ticks = np.linspace(y_min, y_max, 6)
    ax.set_yticks(y_ticks)
    yticklabels = [f'{y_tick:.0f}' for y_tick in y_ticks]
    ax.set_yticklabels(yticklabels, fontsize=pp['fontsize'])




    xmin, xmax, xstep = 0, 1000, 200
    ax.set_xlim([xmin, xmax])
    xticks = np.arange(xmin, xmax+xstep, xstep)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=pp['fontsize'])


    ax.set_xlabel('Sample', fontsize=pp['fontsize'], fontweight='bold')

    if n_zones == 1:
        yaxis_label = "Log-likelihood of simulated area"
    else:
        yaxis_label = "Log-likelihood of simulated areas"
    ax.set_ylabel(yaxis_label, fontsize=pp['fontsize'], fontweight='bold')

    ax.legend(loc=4, prop={'size': pp['fontsize']}, frameon=False)


    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)


def plot_trace_lh_with_prior(mcmc_res,  burn_in=0.2, lh_range=None, prior_range=None, labels=None, fname='trace_likelihood'):
    """
    Function to plot the trace of the MCMC chains both in terms of likelihood and recall
    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        burn_in: (float): First n% of samples are burn-in
        true_lh (boolean): Visualize the true likelihood
        fname (str): a path followed by a the name of the file
    """

    pp = get_plotting_params()

    plt.rcParams["axes.linewidth"] = pp['frame_width']

    n_zones = len(mcmc_res['zones'])


    fig, ax1 = plt.subplots(figsize=(pp['fig_width'], pp['fig_height']))

    # create shared x axis
    xmin, xmax, xstep = 0, 1000, 200
    x = range(xmax - xmin)

    ax1.set_xlim([xmin, xmax])
    xticks = np.arange(xmin, xmax+xstep, xstep)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks, fontsize=pp['fontsize'])

    ax1.set_xlabel('Sample', fontsize=pp['fontsize'], fontweight='bold')

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

    yaxis_label = 'Log-likelihood of simulated area'
    if n_zones > 1:
        yaxis_label += 's'
    yaxis_label = 'Log-likelihood'
    ax1.set_ylabel(yaxis_label, fontsize=pp['fontsize'], fontweight='bold', color=color_lh)




    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color_prior = 'tab:blue'
    prior = mcmc_res['prior']
    ax2.plot(x, prior, lw=pp['line_thickness'], color=color_prior, linestyle='-', label=labels[1])
    yaxis_label = 'Prior'
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
    end_bi = math.ceil(len(x) * burn_in)
    end_bi_label = math.ceil(len(x) * (burn_in - 0.03))

    ax2.axvline(x=end_bi, lw=pp['line_thickness'], color=pp['color_burn_in'], linestyle='--')
    ypos_label = y_min + (y_max - y_min) / 2
    ax2.text(end_bi_label, ypos_label, 'Burn-in', rotation=90, size=pp['fontsize'], color=pp['color_burn_in'])

    # ax1.legend(loc=4, prop={'size': pp['fontsize']}, frameon=False)
    # ax1.legend([lh_handle, prior_handle])

    # ask matplotlib for the plotted objects and their labels
    lh_handle, lh_label = ax1.get_legend_handles_labels()
    prior_handle, prior_label = ax2.get_legend_handles_labels()
    ax2.legend(lh_handle + prior_handle, lh_label + prior_label, loc=4, prop={'size': pp['fontsize']}, frameon=False)

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


def plot_auto_sim(mcmc_res):
    """
    Function to plot the autosimilarity of consecutive samples in MCMC chains
    Args:
           mcmc_res (dict): the output from the MCMC neatly collected in a dict
    """

    fig, ax = plt.subplots()
    col = get_colors()

    for z in mcmc_res['zones'][0:1]:
        y = []
        for t in range(1, 500):

            y.append(zones_autosimilarity(z, t))
            x = range(len(y))
            ax.plot(x, y, lw=1, color=col['trace']['lh'])

    ax.set_xlabel('lag')
    ax.set_ylabel('Autosimilarity')
    ax.set_title('Autosimilarity plot')
    plt.show()


def plot_proximity_graph(net, zone, graph, triang_type):
    """ This function generates a plot of the entire network, the current zone and its proximity graph
    Args:
        net (dict): The full network containing all sites.
        zone (np.array): The current zone (boolean array).
        graph (dict): Either a delaunay triangulation or a minimum spanning tree of the zone.
        triang_type (str): Type of the triangulation, either "delaunay" or "mst"
    """
    # Initialize plot
    fig, ax = plt.subplots()
    col = get_colors()

    all_sites = net['locations']
    v = zone.nonzero()[0]
    zone_sites = net['locations'][v]

    # Plot background and zones
    size = 4
    bg = ax.scatter(*all_sites.T, s=size, color=col['zones']['background_nodes'])
    zo = ax.scatter(*zone_sites.T, s=size*3, color=col['zones']['triangulation'])

    # Collect all edges in the triangulation in a line collection
    lines = []
    for e in graph.es:
        lines.append([tuple(zone_sites[e.tuple[0]]), tuple(zone_sites[e.tuple[1]])])

    lc = LineCollection(lines, colors=col['zones']['triangulation'], linewidths=0.5)
    ax.add_collection(lc)

    # Customize plotting layout
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    bbox = bounding_box(all_sites)
    offset_x = (bbox['x_max'] - bbox['x_min'])*0.03
    offset_y = (bbox['y_max'] - bbox['y_min']) * 0.03
    plt.xlim(bbox['x_min']-offset_x, bbox['x_max']+offset_x)
    plt.ylim(bbox['y_min']-offset_y, bbox['y_max']+offset_y)

    if triang_type == "delaunay":
        triang_legend = "Delaunay Triangulation"
    elif triang_type == "mst":
        triang_legend = "Minimum Spanning Tree"

    # Add legend
    ax.legend([bg, zo, lc], ['All sites', 'Sites in contact zone', triang_legend], frameon=False, fontsize=15)
    plt.show()


def plot_histogram_empirical_geo_prior(e_g_prior, g_prior_type):
    """

    Args:
        e_g_prior (dict): the empirical geo-prior
        g_prior_type (): the type of geo-prior, either "complete", "delaunay" or "mst"
    """
    # Load the color palette for plotting

    fig, ax = plt.subplots()
    col = get_colors()

    d = e_g_prior[g_prior_type]['empirical']
    ax.hist(d, 80, normed=1, facecolor='grey', edgecolor='white', alpha=0.75)

    # add a 'best fit' line
    x = np.linspace(0, max(d), 500)
    shape, loc, scale = e_g_prior[g_prior_type]['fitted_gamma']
    y = gamma.pdf(x, shape, loc, scale)
    ax.plot(x, y, color=col['histogram']['fitted_line'], linewidth=2)
    ax.set_facecolor(col['histogram']['background'])

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.xlabel('Length [km]', labelpad=20)
    plt.ylabel('Probability', labelpad=40)
    if g_prior_type == "complete":
        title_plot = "Complete Graph"
    if g_prior_type == "delaunay":
        title_plot = " Delaunay Graph"
    if g_prior_type == "mst":
        title_plot = "Minimum Spanning Tree"
    plt.title('Average length of edges in the %s' % title_plot)
    plt.grid(False)
    plt.show()


def plot_geo_prior_vs_feature_lh(mcmc_res, r=0, burn_in=0.2):
    """
        Function to plot the Likelihood and the prior for each chain in the MCMC
        Args:
            mcmc_res (dict): the output from the MCMC neatly collected in a dict
            r (int): which run?
            burn_in: (float): First n% of samples are burn-in
        """
    colors = get_colors()['zones']['in_zones']
    fig, ax = plt.subplots()

    # Where to put the label
    x_mid = []
    n_zones = len(mcmc_res['lh'][r])

    for c in range(n_zones):

        if n_zones == 1:
            label_lh = 'Likelihood'
            label_prior = 'Prior'

        else:
            label_lh = 'Likelihood (zone' + str(c) + ')'
            label_prior = 'Prior (zone' + str(c) + ')'

        y = mcmc_res['lh'][r][c]
        x = range(len(y))
        ax.plot(x, y, lw=0.75, color=colors[c], label=label_lh)

        y = mcmc_res['prior'][r][c]
        x = range(len(y))
        ax.plot(x, y, lw=0.75, color=colors[c], linestyle='--', label=label_prior)
        x_mid.append(max(y) - min(y))  # Where to put the label?

    # Find index of first sample after burn-in
    end_bi = math.ceil(len(y) * burn_in)
    end_bi_label = math.ceil(len(y) * (burn_in - 0.03))

    ax.axvline(x=end_bi, lw=1, color="grey", linestyle='--')
    ax.text(end_bi_label, max(x_mid), 'Burn-in', rotation=90, size=8)

    ax.set_xlabel('Sample')
    ax.legend(loc=4)
    plt.show()


def plot_zone_size_vs_ll(mcmc_res, lh_type, mode, individual=False):

    colors = get_colors()['zones']['in_zones']
    fig, ax = plt.subplots()

    if individual:

        for c in range(len(mcmc_res['zones'])):
            lh = []
            for _ in (range(201)):
                lh.append([])

            pt_in_zone = []
            for z in mcmc_res['zones'][c]:
                pt_in_zone.append(np.sum(z))

            for idx in pt_in_zone:
                lh[idx].append(mcmc_res[lh_type][c][idx])
            lh = lh[5:]
            sumstat_lh = []

            for l in lh:
                if mode == 'mean':
                    sumstat_lh.append(np.mean(l))
                elif mode == 'std':
                    sumstat_lh.append(np.std(l))
                elif mode == 'count':
                    sumstat_lh.append(len(l))

            y = sumstat_lh
            x = range(5, len(sumstat_lh) + 5)
            ax.plot(x, y, lw=0.75, color=colors[c])

    else:
        for c in range(len(mcmc_res['zones'])):
            lh = []
            for _ in (range(201)):
                lh.append([])

        for c in range(len(mcmc_res['zones'])):
            pt_in_zone = []
            for z in mcmc_res['zones'][c]:
                pt_in_zone.append(np.sum(z))

            for idx in pt_in_zone:
                lh[idx].append(mcmc_res[lh_type][c][idx])
        lh = lh[5:]
        sumstat_lh = []
        for l in lh:
            if mode == 'mean':
                sumstat_lh.append(np.mean(l))
            elif mode == 'std':
                sumstat_lh.append(np.std(l))
            elif mode == 'count':
                sumstat_lh.append(len(l))

        y = sumstat_lh
        x = range(5, len(sumstat_lh) + 5)
        ax.plot(x, y, lw=0.75, color=colors[0])

    ax.set_xlabel('Zone size')
    ax.set_xlabel('Mean geo prior')
    plt.title('Number of samples per zone size')
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

        ax.plot(x, size, lw=pp['line_thickness'], color=colors[0], linestyle=linestyle[0], label="Predicted")

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

    ax.set_xlabel('Sample', fontsize=pp['fontsize'], fontweight='bold')
    ax.set_ylabel('Zone size', fontsize=pp['fontsize'], fontweight='bold')

    ax.legend(loc=4, prop={'size': pp['fontsize']}, frameon=False)

    fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
    plt.close(fig)

    
    


def plot_gamma_parameters(ecdf):

    figure, ax = plt.subplots()
    loc = []
    scale = []
    for e in ecdf:

        loc.append(ecdf[e]['mst']['fitted_gamma'][0])
        scale.append(ecdf[e]['mst']['fitted_gamma'][2])

    max_loc = max(loc)
    max_scale = max(scale)

    norm_loc = []
    for l in loc:
        norm_loc.append(l/max_loc)

    norm_scale = []
    for s in scale:
        norm_scale.append(s/max_scale)

    x = range(len(norm_loc))
    ax.plot(x, norm_loc, lw=0.75, color="red")
    ax.plot(x, norm_scale, lw=0.75, color="blue")

    plt.show()


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


def plot_parallel_posterior(post):
    """ This function first sorts the posterior of parallel zones in mcmc_res, such that the first list comprises
    the posterior of the largest zone, the second the posterior of the second largest zone, ... , and then
    creates a boxplot of the resulting sorted posteriors of each zone
    Args:
        post (np.ndarray): the posterior of all parallel zones
    """
    # Get color palette
    colors = get_colors()

    # Sort the array
    z_array = np.vstack(post)
    z_array[::-1].sort(axis=0)


    # make boxplot
    fig, ax = plt.subplots()
    ax.set_title('(log)-Posterior of parallel zones')
    ax.set_xlabel('Parallel zone')
    ax.set_ylabel('(log)-Posterior')

    ax.boxplot(z_array.tolist(), showcaps=False, showmeans=False, patch_artist=True,
               widths=0.2,  medianprops=dict(color=colors['boxplot']['median']),
               whiskerprops=dict(color=colors['boxplot']['whiskers']),
               boxprops=dict(facecolor=colors['boxplot']['box'], linewidth=0.1,
                             color=colors['boxplot']['box']))

    plt.show()


if __name__ == '__main__':
    from src.util import load_from
    from src.config import NETWORK_PATH, FEATURES_PATH, LOOKUP_TABLE_PATH,ECDF_GEO_PATH
    from src.preprocessing import get_contact_zones

    TEST_SAMPLING_DIRECTORY = 'data/results/test/zones/2018-10-02_10-03-13/'

    # Zone, ease and number of runs
    zone = 6
    ease = 1
    n_runs = 1

    mcmc_res = {'lh': [[] for _ in range(n_runs)],
                'prior': [[] for _ in range(n_runs)],
                'recall': [[] for _ in range(n_runs)],
                'precision': [[] for _ in range(n_runs)],
                'zones': [[] for _ in range(n_runs)],
                'posterior': [[] for _ in range(n_runs)],
                'lh_norm': [[] for _ in range(n_runs)],
                'posterior_norm': [[] for _ in range(n_runs)],
                'true_zones':[[] for _ in range(n_runs)]}

    for r in range(n_runs):

        # Load the MCMC results
        sample_path = TEST_SAMPLING_DIRECTORY + 'zone_z' + str(zone) + '_e' + \
                      str(ease) + '_' + str(r) + '.pkl'

        samples = load_from(sample_path)
        # Todo:  Handle parallel zones
        # Todo: Run with burn-in
        for t in range(len(samples['sample_zones'])):

            # Zones, likelihoods and priors
            zones = np.asarray(samples['sample_zones'][t])

            mcmc_res['zones'][r].append(zones)
            mcmc_res['lh'][r].append(samples['sample_likelihoods'][t])
            mcmc_res['prior'][r].append(samples['sample_priors'][t])

            # Normalized likelihood and posterior

            posterior = [x + y for x, y in zip(samples['sample_likelihoods'][t], samples['sample_priors'][t])]
            true_posterior = samples['true_zones_lls'][t] + samples['true_zones_priors'][t]
            mcmc_res['posterior'][r].append(posterior)
            lh_norm = np.asarray(samples['sample_likelihoods'][t]) / samples['true_zones_lls'][t]
            posterior_norm = np.asarray(posterior) / true_posterior

            # Recall and precision
            true_z = samples['true_zones'][t]
            n_true = np.sum(true_z)

            # zones = zones[:, 0, :]
            intersections = np.minimum(zones, true_z)
            total_recall = np.sum(intersections, axis=1)/n_true
            precision = np.sum(intersections, axis=1)/np.sum(zones, axis=1)

            # Store to dict
            mcmc_res['lh_norm'][r].append(lh_norm)
            mcmc_res['posterior_norm'][r].append(posterior_norm)
            mcmc_res['recall'][r].append(total_recall)
            mcmc_res['precision'][r].append(precision)
            mcmc_res['true_zones'][r].append(true_z)

    network = load_from(NETWORK_PATH)

    #print(mcmc_res['posterior'])
    #for u in mcmc_res['posterior']:
    #     plot_parallel_posterior(u)
    # print(mcmc_res['zones'][0])
    #print(mcmc_res['zones'][0][0][0])
    #print(len(mcmc_res['zones'][0][0]))
    #

    plot_posterior_frequency(mcmc_res['zones'], network, nz=0, r=0, burn_in=0.2)

    # plot_geo_prior_vs_feature_lh(mcmc_res, r=0, burn_in=0.2 )

    # plot_zone_size_over_time(mcmc_res, r=0, burn_in=0.2)
    #

    #plot_zone_size_vs_ll(mcmc_res, 'geo_prior', mode='mean', individual=True)

    #ecdf = load_from(ECDF_GEO_PATH)
    #plot_gamma_parameters(ecdf)
    #print(np.sum(mcmc_res['zones'][1][-1][0]))
    #for u in range(len(mcmc_res['zones'])):
    #    plot_zone(mcmc_res['zones'][u][-1][0], network)



    #plot_auto_sim(mcmc_res)
    # POSTERIOR_PATH = 'data/results/test/2018-05-13/sampling_e1_a1_mgenerative_2.pkl'
    # posterior = load_from(POSTERIOR_PATH)
    # plt.close()
    #
    # # Load data
    # network = load_from(NETWORK_PATH)
    # features = load_from(FEATURES_PATH)
    # ll_lookup = load_from(LOOKUP_TABLE_PATH)
    # locations = network['locations']
    # adj_mat = network['adj_mat']

    #
    # # Plot posterior frequency
    # plot_posterior_frequency(posterior[0], network)
    #
    # # Plot posterior density
    # plot_posterior_density(posterior[0][1:30], network)
    #
    # # Compute F-statistics
    # #f = f_score_mle(posterior)
    #
    # # Plot alpha shapes
    # plot_alpha_shapes(posterior[0][1], network, alpha=0.0000000008)

    # Plot the empirical distribution of the geo-prior




    #d = [4000, 20000, 30000, 100000]

    # for s in d:
    #     print(gamma.mean(*x_3))
    #     geo_prior = np.log(1-gamma.cdf(s, *x_1))
    #     print(geo_prior)


    # Plot locations as scatter
    # plt.scatter(*locations.T, s=1, lw=0)
    #
    # # Plot edges weighted by edge ll
    # edges = np.argwhere(adj_mat)
    # n = len(locations)
    # edges_ll = []
    # for v1, v2 in edges:
    #     feature_counts = features[v1] + features[v2]
    #     ll = 0
    #     for f_idx, f_count in enumerate(feature_counts):
    #         ll += ll_lookup[f_idx][2][f_count]
    #     edges_ll.append(ll)
    # edges_ll = np.array(edges_ll)
    # edges_ll /= np.max(edges_ll)
    #
    # rgba_colors = [(0.4, 0.05, 0.2, ll**4) for ll in edges_ll]
    # lines = LineCollection(locations[edges], colors=rgba_colors, linewidth=1.)
    #
    # plt.axes().add_collection(lines)
    #
    # plt.axes().set_xticks([])
    # plt.axes().set_yticks([])
    # plt.tight_layout(True)
    #plt.show()
