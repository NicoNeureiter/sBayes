#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import sys
import copy

from contextlib import contextmanager
from scipy.stats import gamma

import numpy as np
import scipy.sparse as sps
import scipy.spatial as spatial
import psycopg2
import igraph

from src.util import compute_distance, grow_zone, triangulation
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
    adj_mat = sps.lil_matrix((n_v, n_v))

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


def get_contact_zones():
    """This function retrieves all contact zones from the DB

    Returns:
        dict: the contact zones
        """
    with psql_connection(commit=True) as connection:
        query_cz = "SELECT cz, array_agg(id) " \
                   "FROM cz_sim.contact_zones_raw " \
                   "WHERE cz != 0 " \
                   "GROUP BY cz"

        cursor = connection.cursor()
        cursor.execute(query_cz)
        rows_cz = cursor.fetchall()
        contact_zones = {}
        for cz in rows_cz:
            contact_zones[cz[0]] = cz[1]
    return contact_zones


def simulate_background_distribution(m_feat, n_sites):
    """This function draws <n_sites> samples from a Binomial distribution for <m_feat> binary features, where
    1 implies the presence of the feature, and 0 the absence. For ech feature the probability of success is drawn from
    a uniform distribution between 0.05 and 0.95. Thus, some feature are balanced and some are skewed towards 0 or 1.

    Args:
        m_feat (int): number of features
        n_sites (int): number of sites for which feature are simulated

    Returns:
        dict: the simulated features
        """
    successes = np.random.uniform(0.05, 0.95, m_feat)
    it = np.nditer(successes, flags=['f_index'])
    features = {}

    for s in it:
        f = np.random.binomial(n=1, p=s, size=n_sites)
        f_idx = 'f' + str(it.index + 1)
        features[f_idx] = f

    return features


def simulate_contact(n_feat, features, p, contact_zones):
    """This function simulates language contact. For each contact zone the function randomly chooses <n_feat> features,
    for which the similarity is increased.

    Args:
        n_feat (int): the number of features for which the function simulates contact
        features (dict): features for which contact is simulated
        p (float): probability of success, defines the degree of similarity in the contact zone
        contact_zones (dict): a region of sites for which contact is simulated

    Returns:
        np.ndarray: the adjusted features
        """
    features_adjusted = copy.deepcopy(features)

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

    dist_mat = net['dist_mat']
    ecdf = {}
    for n in range(min_n, max_n+1):
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
            delaunay_sum.append(sum(triang.es['weight']))

            # Minimum spanning tree
            mst = triang.spanning_tree(weights=triang.es["weight"])
            mst_sum.append(sum(mst.es['weight']))

            if plot and n == 15:
                # Plot graphs
                plot_proximity_graph(net, zone, triang, "delaunay")

                # Minimum spanning tree
                plot_proximity_graph(net, zone, mst, "mst")

        #  c) generate an ecdf for each size n
        #  the ecdf comprises an empirical distribution and a fitted gamma distribution for each type of graph

        ecdf[n] = {'complete': {'empirical': np.sort(complete_sum),
                                'fitted_gamma': gamma.fit(complete_sum, floc=0)},
                   'delaunay': {'empirical': np.sort(delaunay_sum),
                                'fitted_gamma': gamma.fit(delaunay_sum, floc=0)},
                   'mst': {'empirical': np.sort(mst_sum),
                           'fitted_gamma': gamma.fit(mst_sum, floc=0)}}
    return ecdf

# TODO Nico: pickle files laden oder nicht Problem







