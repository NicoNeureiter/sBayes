#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma
from matplotlib.collections import LineCollection

# Peter
# Plot optimieren
def plot_zone(zone, network):
    locations = network['locations']

    plt.scatter(*locations.T, s=5, alpha=0.2)
    plt.scatter(*locations[zone].T, s=10)
    plt.show()


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

# Peter
# fix this plot
# include gabriel usw.
def plot_ecdf_geo_likelihood(ecdf, zone_size):
    x = ecdf[zone_size]['empirical']
    n, bins, patches = plt.hist(x, 30, normed=1, facecolor='grey', edgecolor='white', alpha=0.75)

    # add a 'best fit' line
    #y = gamma.fit(x)

    #plt.plot(x, y, color='r')

    plt.xlabel('Distance')
    plt.ylabel('Probability')
    plt.title('Distribution of distances along the spanning tree for zones of size %i' % zone_size, fontsize=11)
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
    plot_ecdf_geo_likelihood(ecdf, 30)


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