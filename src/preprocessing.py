#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import sys
import copy
import logging
from multiprocessing import Pool
from functools import partial

from contextlib import contextmanager

import numpy as np
from scipy import stats
from scipy import sparse
from scipy.sparse.csgraph import minimum_spanning_tree
import psycopg2
import igraph
import math

from src.util import compute_distance, grow_zone, triangulation, \
    compute_delaunay, dump_results
from src.config import *
from src.plotting import plot_proximity_graph



@contextmanager
def psql_connection(commit=False):
    """This function opens a connection to PostgreSQL, performs a DB operation and finally closes the connection

    Args:
        commit (boolean): specifies if the operation is executed
    """

    # Connection settings for PostgreSQL
    conn = psycopg2.connect(dbname='limits-db', port=5432, user='contact_zones',
                            password='letsfindthemcontactzones', host='limits.geo.uzh.ch')
    cur = conn.cursor()
    try:
        yield conn
    except psycopg2.DatabaseError as err:
        error, = err.args
        sys.stderr.write(error.message)
        cur.execute("ROLLBACK")
        raise err
    else:
        if commit:
            cur.execute("COMMIT")
        else:
            cur.execute("ROLLBACK")
    finally:
        conn.close()


@dump_results(NETWORK_PATH, RELOAD_DATA)
def get_network():
    """ This function retrieves the edge list and the coordinates of the simulated languages
    from the DB and then converts these into a spatial network.

    Returns:
        dict: a dictionary containing the network, its vertices and edges, an adjacency list and matrix
        and a distance matrix
    """

    # Get vertices from PostgreSQL
    with psql_connection(commit=True) as connection:
        query_v = "SELECT id, mx AS x, my AS y " \
                  "FROM {table};".format(table=DB_ZONE_TABLE)
        cursor = connection.cursor()
        cursor.execute(query_v)
        rows_v = cursor.fetchall()

    n_v = len(rows_v)
    vertices = list(range(n_v))
    locations = np.zeros((n_v, 2))
    gid_to_idx = {}
    idx_to_gid = {}
    for i, v in enumerate(rows_v):
        gid, x, y = v

        gid_to_idx[gid] = i
        idx_to_gid[i] = gid

        locations[i, 0] = x
        locations[i, 1] = y

    # Get edges from PostgreSQL
    with psql_connection(commit=True) as connection:
        query_e = "SELECT v1, v2 " \
                  "FROM {table};".format(table=DB_EDGE_TABLE)

        cursor = connection.cursor()
        cursor.execute(query_e)
        rows_e = cursor.fetchall()

    n_e = len(rows_e)
    edges = np.zeros((n_e, 2)).astype(int)
    for i, e in enumerate(rows_e):
        edges[i, 0] = gid_to_idx[e[0]]
        edges[i, 1] = gid_to_idx[e[1]]

    # Adjacency list and matrix
    adj_list = [[] for _ in range(n_v)]
    adj_mat = sparse.lil_matrix((n_v, n_v))

    for v1, v2 in edges:
        adj_list[v1].append(v2)
        adj_list[v2].append(v1)
        adj_mat[v1, v2] = 1
        adj_mat[v2, v1] = 1

    adj_list = np.array(adj_list)
    adj_mat = adj_mat.tocsr()

    # Graph
    g = igraph.Graph()
    g.add_vertices(vertices)

    for e in edges:
        dist = compute_distance(edges[e[0]], edges[e[1]])
        g.add_edge(e[0], e[1], weight=dist)

    # Distance matrix
    diff = locations[:, None] - locations
    dist_mat = np.linalg.norm(diff, axis=-1)

    net = {'vertices': vertices,
           'edges': edges,
           'locations': locations,
           'adj_list': adj_list,
           'adj_mat': adj_mat,
           'n': n_v,
           'm': n_e,
           'graph': g,
           'dist_mat': dist_mat}
    return net


@dump_results(CONTACT_ZONES_PATH, RELOAD_DATA)
def get_contact_zones(zone_id):
    """This function retrieves contact zones from the DB
    Args:
        zone_id(int or tuple of ints) : the id(s) of the zone(s) in the DB
    Returns:
        dict: the contact zones
        """
    # For single zones
    if isinstance(zone_id, int):
        with psql_connection(commit=True) as connection:
            query_cz = "SELECT cz, array_agg(id) " \
                       "FROM cz_sim.contact_zones_raw " \
                       "WHERE cz = {list_id}" \
                       "GROUP BY cz".format(list_id=zone_id)
            cursor = connection.cursor()
            cursor.execute(query_cz)
            rows_cz = cursor.fetchall()
            contact_zones = {}
            for cz in rows_cz:
                contact_zones[cz[0]] = cz[1]
        return contact_zones

    # For parallel zones
    elif isinstance(zone_id, tuple):
        if all(isinstance(x, int) for x in zone_id):
            with psql_connection(commit=True) as connection:
                query_cz = "SELECT cz, array_agg(id) " \
                           "FROM cz_sim.contact_zones_raw " \
                           "WHERE cz IN {list_id}" \
                           "GROUP BY cz".format(list_id=zone_id)
                cursor = connection.cursor()
                cursor.execute(query_cz)
                rows_cz = cursor.fetchall()
                contact_zones = {}
                for cz in rows_cz:
                    contact_zones[cz[0]] = cz[1]
            return contact_zones
        else:
            raise ValueError('zone_id must be int or a list of ints')
    else:
        raise ValueError('zone_id must be int or a list of ints')


@dump_results(FEATURES_BG_PATH, RELOAD_DATA)
def simulate_background_distribution(m_feat, n_sites, p_min, p_max):
    """This function draws <n_sites> samples from a Binomial distribution for <m_feat> binary features, where
    1 implies the presence of the feature, and 0 the absence.

    Args:
        m_feat (int): number of features
        n_sites (int): number of sites for which feature are simulated
        p_min (float): the minimum probability that a feature is 1
        p_max (float): the maximum probability that a feature is 1

    Returns:
        dict: the simulated features
        """
    successes = np.random.uniform(p_min, p_max, m_feat)
    it = np.nditer(successes, flags=['f_index'])
    features = {}

    for s in it:
        f = np.random.binomial(n=1, p=s, size=n_sites)
        f_idx = 'f' + str(it.index + 1)
        features[f_idx] = f

    return features


@dump_results(FEATURES_PATH, RELOAD_DATA)
def simulate_contact(r_feat, features, p, contact_zones):
    """This function simulates language contact. For each contact zone the function randomly chooses <n_feat> features,
    for which the similarity is increased.

    Args:
        r_feat (float: the ratio of all features for which the function simulates contact
        features (dict): features for which contact is simulated
        p (float): probability of success, defines the degree of similarity in the contact zone
        contact_zones (dict): a region of sites for which contact is simulated

    Returns:
        np.ndarray: the adjusted features
        """
    features_adjusted = copy.deepcopy(features)
    n_feat = math.ceil(r_feat * TOTAL_N_FEATURES)

    # Iterate over all contact zones
    for cz in contact_zones:

        # Choose <n_feat> features for which the similarity is increased
        f_to_adjust = np.random.choice(list(features.keys()), n_feat, replace=False)

        for f in f_to_adjust:
            # increase similarity either towards presence (1) or absence (0)
            # p = np.random.choice([p, 1-p])
            f_adjusted = np.random.binomial(n=1, p=p, size=len(contact_zones[cz]))
            for a, _ in enumerate(f_adjusted):
                idx = contact_zones[cz][a]
                features_adjusted[f][idx] = f_adjusted[a]

    features_adjusted_mat = np.ndarray.transpose(np.array([features_adjusted[i] for i in features_adjusted.keys()]))

    return features_adjusted_mat


@dump_results(FEATURE_PROB_PATH, RELOAD_DATA)
def compute_feature_prob(feat):
    """This function computes the base-line probabilities for a feature to be present in the data.

    Args:
        feat (np.ndarray): a matrix of all features

    Returns:
        array: the probability of each feature to be present """

    n = len(feat)
    present = np.count_nonzero(feat, axis=0)
    p_present = present/n

    return p_present


def estimate_ecdf_n(n, nr_samples, net, plot=False):
    logging.info('Estimating empirical CDF for n = %i' % n)

    dist_mat = net['dist_mat']

    complete_sum = []
    delaunay_sum = []
    mst_sum = []

    for _ in range(nr_samples):

        # a)
        zone, _ = grow_zone(n, net)

        # b
        # Complete graph
        complete_sum.append(dist_mat[zone][:, zone].sum())

        # Delaunay graph
        triang = triangulation(net, zone)
        delaunay_sum.append(triang.sum())

        # Minimum spanning tree
        mst = minimum_spanning_tree(triang)
        mst_sum.append(mst.sum())

        if plot and n == 15:
            # Plot graphs
            plot_proximity_graph(net, zone, triang, "delaunay")

            # Minimum spanning tree
            plot_proximity_graph(net, zone, mst, "mst")

    #  c) generate an ecdf for each size n
    #  the ecdf comprises an empirical distribution and a fitted gamma distribution for each type of graph

    ecdf = {'complete': {'fitted_gamma': stats.gamma.fit(complete_sum, floc=0)},
            'delaunay': {'fitted_gamma': stats.gamma.fit(delaunay_sum, floc=0)},
            'mst': {'fitted_gamma': stats.gamma.fit(mst_sum, floc=0)}}

    del complete_sum
    del delaunay_sum
    del mst_sum

    return ecdf


@dump_results(ECDF_GEO_PATH, RELOAD_DATA)
def generate_ecdf_geo_likelihood(net, min_n, max_n, nr_samples, plot=False):
    """ This function generates an empirical cumulative density function (ecdf), which is then used to compute the
    geo-likelihood of a contact zone. The function
    a) grows <nr samples> contact zones of size n, where n is between <min_n> and <max_n>,
    b) for each contact zone: generates a complete graph, delaunay graph and a minimum spanning tree
    and computes the summed length of each graph's edges
    c) for each size n: generates an ecdf of all summed lengths
    d) fits a gamma function to the ecdf

    Args:
        net (dict): network containing the graph, location,...
        min_n (int): the minimum number of languages in a zone
        max_n (int): the maximum number of languages in a zone
        nr_samples (int): the number of samples in the ecdf
        plot(boolean): Plot the triangulation?

    Returns:
        dict: a dictionary comprising the empirical and fitted ecdf for all types of graphs and all sizes n
        """

    n_values = range(min_n, max_n+1)

    estimate_ecdf_n_ = partial(estimate_ecdf_n,
                              nr_samples=nr_samples, net=net, plot=False)

    def pool_init():
        import gc
        gc.collect()

    with Pool(4) as pool:
        ecdf = pool.map(estimate_ecdf_n_, n_values, initializer=pool_init,
                        maxtasksperchild=1)

    # ecdf = []
    # for n in n_values:
    #     ecdf.append(estimate_ecdf_n(n, nr_samples, net))
    #     print(sys.getsizeof(ecdf) / 1000.)

    return {n: e for n, e in zip(n_values, ecdf)}


@dump_results(RANDOM_WALK_COV_PATH, RELOAD_DATA)
def estimate_random_walk_covariance(net):
    dist_mat = net['dist_mat']
    locations = net['locations']

    delaunay = compute_delaunay(locations)
    mst = minimum_spanning_tree(delaunay.multiply(dist_mat))
    # mst += mst.T  # Could be used as data augmentation? (enforces 0 mean)

    # Compute difference vectors along mst
    i1, i2 = mst.nonzero()
    diffs = locations[i1] - locations[i2]

    # Center at (0, 0)
    diffs -= np.mean(diffs, axis=0)[None, :]

    return np.cov(diffs.T)


@dump_results(LOOKUP_TABLE_PATH, RELOAD_DATA)
def precompute_feature_likelihood(min_size, max_size, feat_prob):
    """This function generates a lookup table of likelihoods
    Args:
        min_size (int): the minimum number of languages in a zone.
        max_size (int): the maximum number of languages in a zone.
        feat_prob (np.array): the probability of a feature to be present.
    Returns:
        dict: the lookup table of likelihoods for a specific feature,
            sample size and observed presence.
    """

    # The binomial test computes the p-value of having k or more (!) successes out of n trials,
    # given a specific probability of success
    # For a two-sided binomial test, simply remove "greater".
    # The function returns -log (p_value), which is more sensitive to exceptional observations.

    def ll(p_zone, s, p_global):
        p = stats.binom_test(p_zone, s, p_global, 'greater')
        try:
            return - math.log(p)
        except Exception as e:
            print(p_zone, s, p_global, p)
            raise e

    lookup_dict = {}
    for i_feat, p_global in enumerate(feat_prob):
        lookup_dict[i_feat] = {}
        for s in range(min_size, max_size + 1):
            lookup_dict[i_feat][s] = {}
            for p_zone in range(s + 1):
                lookup_dict[i_feat][s][p_zone] = ll(p_zone, s, p_global)

    return lookup_dict


# TODO Nico: pickle files laden oder nicht Problem