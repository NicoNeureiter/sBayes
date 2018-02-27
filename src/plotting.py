#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import numpy as np
from scipy.stats import gamma
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection

# Peter
# Plot optimieren
def plot_zone(zone, network):
    locations = network['locations']

    plt.scatter(*locations.T, s=5, alpha=0.2)
    plt.scatter(*locations[zone].T, s=10)
    plt.show()

# PLOT_COLORS = {
#
#     'histogram': 1,
#     'zones': {'background_points': }
#
# }


def get_colors():
    """This function creates a dict with colors
    Returns:
        (dict): a dictionary comprising the plot colors
    """

    plot_colors = { 'histogram': {'fitted_line': 'blue',
                                  'background': "white"},
                    'zones': {'background_points': 'grey',
                              'zones': 'darkorange'}}
    return plot_colors


def plot_posterior(samples, adj_mat, locations, weights=None):
    # Add edge evidence for the sampled zone
    n_samples, n_v = samples.shape
    if weights is None:
        weights = np.ones(n_samples)

    edge_counts = (weights[:, None] * samples).T.dot(samples)

    edge_counts *= adj_mat.toarray()
    edge_freq = (edge_counts / n_samples).clip(0, 1)
    print(edge_freq)

    plt.scatter(*locations.T, s=1, lw=0)
    print('Nonzero:', np.count_nonzero(edge_freq))

    edges = np.argwhere(edge_freq)
    rgba_colors = [(0.4, 0.05, 0.2, c) for c in edge_freq[edges[:, 0], edges[:, 1]]]
    lines = LineCollection(locations[edges], colors=rgba_colors)
    plt.axes().add_collection(lines)

    plt.show()


def plot_proximity_graph(net, zone, graph):
    """ This function generates a plot of the entire network, the current zone and its proximity graph
    Args:
        net (dict): The full network containing all sites.
        zone (np.ndarray): The current zone (boolean array).
        graph (dict): Either a delaunay triangulation or a minimum spanning tree of the zone.
    """

    all_sites = net['locations']
    v = zone.nonzero()[0]
    zone_sites = net['locations'][v]

    lines = []

    for e in graph.es:
        lines.append([tuple(zone_sites[e.tuple[0]]), tuple(zone_sites[e.tuple[1]])])

    lc = LineCollection(lines, colors="blue")
    plt.axes().add_collection(lc)
    plt.plot(*all_sites.T, 'ro', ms=2, color="")
    plt.plot(*zone_sites.T, 'ro', color="orange")
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
    ax.hist(d, 30, normed=1, facecolor='grey', edgecolor='white', alpha=0.75)

    # add a 'best fit' line
    x = np.linspace(0, max(d), 500)
    shape, loc, scale = e_gl[zone_size][gl_type]['fitted_gamma']
    y = gamma.pdf(x, shape, loc, scale)
    ax.plot(x, y, color=col['histogram']['fitted_line'])
    ax.set_facecolor(col['histogram']['background'])

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.xlabel('Length [km]', labelpad=40)
    plt.ylabel('Probability')
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
    from src.model import compute_likelihood
    from src.util import load_from
    from src.config import NETWORK_PATH, FEATURES_PATH, LOOKUP_TABLE_PATH,ECDF_GEO_PATH

    # Load data
    network = load_from(NETWORK_PATH)
    features = load_from(FEATURES_PATH)
    ll_lookup = load_from(LOOKUP_TABLE_PATH)
    locations = network['locations']
    adj_mat = network['adj_mat']
    ecdf = load_from(ECDF_GEO_PATH)

    # Plot the empirical distribution of the geo-likelihood
    plot_histogram_empirical_geo_likelihood(ecdf, 10, gl_type="complete")


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

