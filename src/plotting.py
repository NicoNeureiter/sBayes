#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


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

    # edge_counts = np.zeros((n_v, n_v))
    # for i, zone in enumerate(samples):
    #     edge_counts += zone[:, None].dot(zone[None, :]) * weights[i]
    edge_counts = (weights[:, None] * samples).T.dot(samples)

    edge_counts *= adj_mat.toarray()
    edge_freq = (edge_counts / n_samples)
    print(edge_freq)

    plt.scatter(*locations.T, s=10, alpha=0.2, lw=0)
    print('Nonzero:', np.count_nonzero(edge_freq))

    edges = np.argwhere(edge_freq)
    rgba_colors = [(0.4, 0.05, 0.2, c**.8) for c in edge_freq[edges[:, 0], edges[:, 1]]]
    lines = LineCollection(locations[edges], colors=rgba_colors)
    plt.axes().add_collection(lines)

    plt.show()