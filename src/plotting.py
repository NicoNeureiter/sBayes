#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
plt.tight_layout()

import numpy as np
import math

from src.util import zones_autosimilarity
from src.preprocessing import estimate_ecdf_n, generate_ecdf_geo_prior
from src.model import compute_geo_prior_particularity
from scipy.stats import gamma
from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection
from shapely.ops import cascaded_union, polygonize
from shapely import geometry
from descartes import PolygonPatch


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


def plot_posterior_density(zones, net):
    # TODO: Labels, Color palette

    """ This function creates a kernel density plot of all sites in the posterior distribution

    Args:
        zones (np.array): the posterior of all zones
        net (dict): The full network containing all sites.
    """

    all_sites = net['locations']
    points_in_post = []

    for z in zones:
        try:
            points_in_post += all_sites[z[0]]

        except ValueError:
            points_in_post = all_sites[z[0]]

    points_in_post = np.array(points_in_post)
    ax = sns.kdeplot(points_in_post, cmap="Reds", shade=True, shade_lowest=False)

    # Remove axes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


def plot_posterior_frequency(zones, net):
    """ This function creates a scatter plot of all sites in the posterior distribution. The color of a site reflects
    its frequency in the posterior

    Args:
        zones (np.array): the posterior of all zones
        net (dict): The full network containing all sites.
    """

    fig, ax = plt.subplots()
    all_sites = net['locations']
    size = 4
    n = len(zones)

    density = (np.sum(zones, axis=0, dtype=np.int32)/n)[0]
    sc = ax.scatter(*all_sites.T, c=density, s=size + size*2*density, cmap='plasma')

    # Customize plotting layout
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    bbox = bounding_box(all_sites)
    offset_x = (bbox['x_max'] - bbox['x_min']) * 0.03
    offset_y = (bbox['y_max'] - bbox['y_min']) * 0.03
    plt.xlim(bbox['x_min'] - offset_x, bbox['x_max'] + offset_x)
    plt.ylim(bbox['y_min'] - offset_y, bbox['y_max'] + offset_y)
    cbar = plt.colorbar(sc, shrink=0.3, orientation="horizontal")
    cbar.ax.get_xaxis().labelpad = -45
    cbar.ax.set_xlabel('Frequency of point in posterior')
    plt.show()


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


def plot_alpha_shapes(zone, net, alpha):
    # ToDo: Make shapes look nice

    """Compute the alpha shape (concave hull) of a zone.

    Args:
        zone (np.array): a contact zone from the posterior
        net (dict): The full network containing all sites.
        alpha (float): alpha value to influence the gooeyness of the border. Smaller numbers don't fall inward
        as much as larger numbers. Too large, and you lose everything! "

    Returns:
        ()"""

    all_sites = net['locations']
    points = all_sites[zone[0]]

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    tri = Delaunay(points, qhull_options="QJ Pp")
    edges = set()
    edge_points = []

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

        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, points, ia, ib)
            add_edge(edges, edge_points, points, ib, ic)
            add_edge(edges, edge_points, points, ic, ia)

    m = geometry.MultiLineString(edge_points)

    triangles = list(polygonize(m))
    polygon = cascaded_union(triangles)
    fig, ax = plt.subplots()
    margin = .3

    bb = bounding_box(all_sites)
    ax.set_xlim([bb['x_min'] - margin, bb['x_max'] + margin])
    ax.set_ylim([bb['y_min'] - margin, bb['y_max'] + margin])

    patch = PolygonPatch(polygon, fc='#999999',
                         ec='#000000', fill=True,
                         zorder=-1)
    ax.add_patch(patch)
    ax.scatter(*points.T)
    plt.show()


def plot_trace_mcmc(mcmc_res, sample_interval):
    """
    Function to plot the trace of the MCMC chains both in terms of likelihood and recall
    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        sample_interval (int): the interval at which samples should be taken from the chains

    """

    fig, ax = plt.subplots()
    col = get_colors()

    for l in mcmc_res['lh']:
        y = l[::sample_interval]
        x = range(len(y))
        ax.plot(x, y, lw=0.75, color=col['trace']['lh'])

    for r in mcmc_res['recall']:
        y = r[::sample_interval]
        x = range(len(y))
        ax.plot(x, y, lw=0.75, color=col['trace']['recall'])

    for p in mcmc_res['precision']:
        y = p[::sample_interval]
        x = range(len(y))
        ax.plot(x, y, lw=0.75, color=col['trace']['precision'])

    ax.axhline(y=1, xmin=x[0], xmax=x[-1], lw=2, color="red")
    ax.set_xlabel('Sample')
    ax.set_ylabel('log LH')
    ax.set_title('Trace plot')
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


#concave_hull, edge_points = plot_alpha_shapes(points, alpha=1.87)
#_ = plot_polygon(concave_hull)
#_ = pl.plot(x, y, 'o', color='#f16824')

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


def plot_geo_prior_vs_feature_lh(mcmc_res, sample_interval):
    print(mcmc_res['feat_lh'])
    fig, ax = plt.subplots()

    for c in mcmc_res['feat_lh'][::sample_interval]:
        y = c[::sample_interval]
        x = range(len(y))
        ax.plot(x, y, lw=0.75, color="red")

    for c in mcmc_res['geo_prior'][::sample_interval]:
        y = c[::sample_interval]
        #yg = []
        #for i in y:
        #    g = i
        #    yg.append(g)
        x = range(len(y))
        ax.plot(x, y, lw=0.75, color="blue")

    ax.set_xlabel('Sample')
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


def plot_zone_size_over_time(mcmc_res):

    colors = get_colors()['zones']['in_zones']
    figure, ax = plt.subplots()

    for c in range(len(mcmc_res['zones'])):

        pt_in_zone = []
        for z in mcmc_res['zones'][c]:
            pt_in_zone.append(np.sum(z))

        x = range(len(pt_in_zone))
        ax.plot(x, pt_in_zone, lw=0.75, color=colors[c])

    plt.show()

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

if __name__ == '__main__':
    from src.util import load_from
    from src.config import NETWORK_PATH, FEATURES_PATH, LOOKUP_TABLE_PATH,ECDF_GEO_PATH
    from src.util import bounding_box
    from src.preprocessing import get_contact_zones

    TEST_SAMPLING_DIRECTORY = 'data/results/test/sampling/2018-07-09_16-49-27/'

    # Annealing, easy zones
    annealing = 1
    ease = 1

    mcmc_res = {'lh': [],
                'recall': [],
                'precision': [],
                'zones': [],
                'feat_lh': [],
                'geo_prior': []}

    for r in range(5):

        # Load the MCMC results
        sample_path = TEST_SAMPLING_DIRECTORY + 'sampling_e' + str(ease) + '_a' + \
                      str(annealing) + '_mparticularity_' + str(r) +'.pkl'

        samples = load_from(sample_path)
        mcmc_res['zones'].append(samples[0])

        mcmc_res['feat_lh'].append(samples[1]['feat_ll'])
        mcmc_res['geo_prior'].append(samples[1]['geo_prior'])


        # Compute the normalized likelihood
        #lh_norm = samples[1]['step_likelihoods'][::2]/samples[1]['true_zones_ll'][0]

        # Compute recall and precision
        true_z = samples[1]['true_zones'][0]
        n_true = np.sum(true_z)

        zones = np.asarray(samples[0])

        zones = zones[:, 0, :]

        intersections = np.minimum(zones, true_z)
        total_recall = np.sum(intersections, axis=1)/n_true
        precision = np.sum(intersections, axis=1)/np.sum(zones, axis=1)

        # Store to dict
        #mcmc_res['lh'].append(lh_norm)
        mcmc_res['recall'].append(total_recall)
        mcmc_res['precision'].append(precision)

    network = load_from(NETWORK_PATH)
    #estimate_ecdf_n(10, 2, network)
    #ecdf = generate_ecdf_geo_prior(network, 5, 90, 500)
    #plot_histogram_empirical_geo_prior(ecdf, gl_type="mst")
    #plot_histogram_empirical_geo_prior(ecdf, gl_type="delaunay")
    #plot_histogram_empirical_geo_prior(ecdf, gl_type="complete")
    #all_zones = []

    for u in range(len(mcmc_res['zones'])):
            plot_posterior_frequency(mcmc_res['zones'][u][70000:], network)
    plot_geo_prior_vs_feature_lh(mcmc_res, 100)
    plot_trace_mcmc(mcmc_res, 100)
    plot_zone_size_over_time(mcmc_res)
    #plot_zone_size_vs_ll(mcmc_res, 'geo_prior', mode='mean', individual=True)

    #ecdf = load_from(ECDF_GEO_PATH)
    #plot_gamma_parameters(ecdf)
    #print(np.sum(mcmc_res['zones'][1][-1][0]))
    for u in range(len(mcmc_res['zones'])):
        plot_zone(mcmc_res['zones'][u][-1][0], network)



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
