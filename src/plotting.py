#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')
import numpy as np
from scipy.stats import gamma
from matplotlib.collections import LineCollection
from src.util import bounding_box


def get_colors():
    """This function creates a dict with colors
    Returns:
        (dict): a dictionary comprising the plot colors
    """

    plot_colors = {'histogram': {'fitted_line': (1.000, 0.549, 0.000),      # darkorange
                                 'background': (1, 1, 1)},                  # white
                   'zone': {'background_nodes': (0.502, 0.502, 0.502),     # grey
                            'in_zone': (1.000, 0.549, 0.000),               # darkorange
                            'triangulation': (1.000, 0.549, 0.000)},
                   'zones': {'background_nodes': (0.502, 0.502, 0.502),
                             'in_zones': [(0.894, 0.102, 0.11),             # red
                                          (0.216, 0.494, 0.722),            # blue
                                          (0.302, 0.686, 0.29),             # green
                                          (0.596, 0.306, 0.639),            # violett
                                          (1.000, 0.549, 0.000),            # darkorange)
                                          (1, 1, 0.2),                      # yellow
                                          (0.651, 0.337, 0.157),            # brown
                                          (0.969, 0.506, 0.749),            # pinkish
                                          (0, 0, 0)]}}                      # black ]
    return plot_colors


def plot_posterior(samples, net):
    """ This function plots the posterior distribution of contact zones

    Args:
        samples (np.array): the samples from the MCMC
        net (dict): The full network containing all sites.
    """

    # Get data
    col = get_colors()
    adj_mat = net['adj_mat']
    locations = net['locations']

    # Initialize plot
    fig, ax = plt.subplots()

    # Add edge evidence for the sampled zone
    n_samples, n_v = samples.shape
    weights = np.ones(n_samples)
    edge_counts = (weights[:, None] * samples).T.dot(samples)
    edge_counts *= adj_mat.toarray()
    edge_freq = (edge_counts / n_samples).clip(0, 1)

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

    # Remove axes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend
    ax.legend([bg, lines], ['All sites', 'Edges in posterior distribution'], frameon=False, fontsize=10)
    plt.show()


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
    bg = ax.scatter(*all_sites.T, s=size, color=col['zones']['background_nodes'])
    zo = ax.scatter(*all_sites[zone].T, s=size*3, color=col['zones']['triangulation'])

    # Remove axes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend
    ax.legend([bg, zo], ['All sites', 'Sites in proposed contact zone'], frameon=False, fontsize=10)
    plt.show()


def plot_zones(zones, net):
    """ This function plots the contact zones proposed by the MCMC

    Args:
        zones (np.array): The current zone (boolean array).
        net (dict): The full network containing all sites.
    """

    # Initialize plot
    fig, ax = plt.subplots()
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


def plot_histogram_empirical_geo_likelihood(e_gl, zone_size, gl_type):
    """

    Args:
        e_gl (dict): the empirical geo-likelihood
        zone_size (): the size of the zone for which the histogram is generated
        gl_type (): the type of geo-likelihood, either "complete", "delaunay" or "mst"
    """
    # Load the color palette for plotting

    fig, ax = plt.subplots()
    col = get_colors()

    d = e_gl[zone_size][gl_type]['empirical']
    ax.hist(d, 80, normed=1, facecolor='grey', edgecolor='white', alpha=0.75)

    # add a 'best fit' line
    x = np.linspace(0, max(d), 500)
    shape, loc, scale = e_gl[zone_size][gl_type]['fitted_gamma']
    y = gamma.pdf(x, shape, loc, scale)
    ax.plot(x, y, color=col['histogram']['fitted_line'], linewidth=2)
    ax.set_facecolor(col['histogram']['background'])

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.xlabel('Length [km]', labelpad=20)
    plt.ylabel('Probability', labelpad=40)
    if gl_type == "complete":
        title_plot = "Complete Graph"
    if gl_type == "delaunay":
        title_plot = " Delaunay Graph"
    if gl_type == "mst":
        title_plot = "Minimum Spanning Tree"
    plt.title('Length of the %s for zones of size %i' % (title_plot, zone_size))
    plt.grid(False)
    plt.show()


if __name__ == '__main__':
    from src.util import load_from
    from src.config import NETWORK_PATH, FEATURES_PATH, LOOKUP_TABLE_PATH,ECDF_GEO_PATH

    k = os.getcwd()
    print(k)

    # Load data
    network = load_from(NETWORK_PATH)
    features = load_from(FEATURES_PATH)
    ll_lookup = load_from(LOOKUP_TABLE_PATH)
    locations = network['locations']
    adj_mat = network['adj_mat']
    ecdf = load_from(ECDF_GEO_PATH)

    # Plot the empirical distribution of the geo-likelihood
    plot_histogram_empirical_geo_likelihood(ecdf, 70, gl_type="mst")

    ecdf_geo = load_from(ECDF_GEO_PATH)
    x_1 = ecdf_geo[20]["mst"]['fitted_gamma']
    x_2 = ecdf_geo[30]["mst"]['fitted_gamma']
    x_3 = ecdf_geo[49]["mst"]['fitted_gamma']


    d = [4000, 20000, 30000, 100000]

    for s in d:
        print(gamma.mean(*x_3))
        geo_lh = np.log(1-gamma.cdf(s, *x_1))
        print(geo_lh)

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
    # plt.show()

from src.model import lookup_log_likelihood

#a = binom_test(9, 10, 0.5, "two-sided")
#b = binom_test(8, 10, 0.5, "greater")
#c = binom_test(9, 10, 0.5, "less")
#d = np.log(b)
#e = -np.log(a)
# print('two-sided:', a, "\n",
#       'greater:', b, "\n",
#       'less:', c, "\n",
#       'complement of two-sided:', 1-a, "\n",
#       'complement of greater:', 1-b, "\n",
#       'complement of less:', 1-c)
# #
# #
#
#
