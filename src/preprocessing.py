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
import psycopg2
import igraph

from src.util import compute_distance, grow_zone

# Peter
# -> config

DB_ZONE_TABLE = 'cz_sim.contact_zones_raw'
DB_EDGE_TABLE = 'cz_sim.delaunay_edge_list'


@contextmanager
# Peter
# -> change to Google stylk, change settings docstring style to google
def psql_connection(commit=False):
    """
        This function opens a connection to PostgreSQL,
        performs a DB operation and finally closes the connection
        :In
        - commit: a Boolean variable that specifies if the operation is actually executed
        :Out
        -
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

# Peter
# -> change to Google stylk, change settings docstring style to google
def get_network():
    """
        This function first retrieves the edge list and the coordinates of the simulated languages
        from the DB, then converts these into a spatial network.
        :In
        -
        :Out
        - net: a dictionary containing the network (igraph), its vertices and edges, and its
        adjacency matrix
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

# Peter
# -> change to Google stylk, change settings docstring style to google
def get_contact_zones():
    """
        This function retrieves all contact zones from the DB
        :In
        -
        :Out
        - contact_zones: a dict with all contact zones
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

# Peter
# -> change to Google stylk, change settings docstring style to google
def simulate_background_distribution(m_feat, n_sites):
    """
        This function draws m samples from a Binomial distribution for n binary features (0 = absence, 1 = presence)
        The probability of success (1) of each feature is drawn from a uniform distribution between 0.05 and 0.95
        Thus, a feature can be considerably skewed towards 0 or 1, but there are no extreme distributions
        in which one state is hardly possible at all

        :In
        - m_feat: number of features
        - n_sites: number of sites for which a feature is simulated
        :Out
        - features: a dictionary comprising the simulated features
        """
    successes = np.random.uniform(0.05, 0.95, m_feat)
    it = np.nditer(successes, flags=['f_index'])
    features = {}

    for s in it:
        f = np.random.binomial(n=1, p=s, size=n_sites)
        f_idx = 'f' + str(it.index + 1)
        features[f_idx] = f

    return features

# Peter
# -> change to Google stylk, change settings docstring style to google
def simulate_contact(n_feat, features, p, contact_zones):
    """
        This function simulates contact in a contact zone. For each zone the function randomly
        chooses n features, for which the similarity is increased
        :In
        - n_feat: number of features for which the function simulates contact
        - features: dict of features for which contact is simulated
        - p: Probability of success, the strength of the similarity in the contact zone
        - contact_zones: region of neighbouring features for which contact is simulated
        :Out
        - features_adjusted: a dict containing the adjusted features
        """
    features_adjusted = copy.deepcopy(features)
    # Iterate over all contact zones
    for cz in contact_zones:

        # Choose n features for which the similarity is increased
        f_to_adjust = np.random.choice(list(features.keys()), n_feat, replace=False)

        for f in f_to_adjust:
            # increase similarity either towards presence (1) or absence (0)
            p = np.random.choice([p, 1-p])
            f_adjusted = np.random.binomial(n=1, p=p, size=len(contact_zones[cz]))
            for a, _ in enumerate(f_adjusted):
                idx = contact_zones[cz][a]
                features_adjusted[f][idx] = f_adjusted[a]
    # return as numpy array to improve run time of mcmc

    features_adjusted_mat = np.ndarray.transpose(np.array([features_adjusted[i] for i in features_adjusted.keys()]))

    return features_adjusted_mat

# Peter
# -> change to Google stylk, change settings docstring style to google
def compute_feature_prob(feat):
    """
        This function computes the base-line probabilities for a feature to be present
        :In
        - feat: a matrix of features
        :Out
        - p_present: a vector containing the probabilities of features to be present"""
    n = len(feat)
    present = np.count_nonzero(feat, axis=0)
    p_present = present/n

    return p_present

# Peter
# -> change to Google stylk, change settings docstring style to google
# Full, delauney, gabriel, mst
def generate_ecdf_geo_likelihood(net, min_n, max_n, samples_per_n):

    """
        This function generates an empirical cumulative density function (ecdf), which represents
        the geo likelihood of a contact zone.
        The function
        a) grows x contact zones of size n, where n is between min_n and max_n,
        b) for each contact zone: generate the minimum spanning tree and extracts its Euclidean distance
        c) for each size n: generates an ecdf for all cumulative distances of all minimum spanning trees
        : In
        - net: a network graph
        - min_n: the minimum number of languages in a zone
        - max_n: the maximum number of languages in a zone
        - sample_per_n: the number of samples in the ecdf
        : Out
        - ecdf: the empirical cumulative density function for all distances of all minimum spanning trees of size n"""
    adj_mat = net['adj_mat']
    dist_mat = net['dist_mat']
    ecdf = {}
    for n in range(min_n, max_n+1):
        dist_sum = []

        for _ in range(samples_per_n):

            # a) grow contact zones
            zone = grow_zone(n, adj_mat)

            # v = zone.nonzero()[0]
            #
            # # b) generate a minimum spanning tree and extract its Euclidean distance
            # sub_g = net['graph'].subgraph(v)
            # span_tree = sub_g.spanning_tree(weights=sub_g.es["weight"])
            # dist_span_tree.append(sum(span_tree.es['weight']))

            dist_sum.append(dist_mat[zone][:, zone].sum())

        #  c) generate an ecdf for each size n
        #  the ecdf comprises an empirical distribution and a fitted gamma distribution

        ecdf[n] = {'empirical': np.sort(dist_sum),
                   'fitted_gamma': gamma.fit(dist_sum)}

    return ecdf

# TODO Nico: pickle files laden oder nicht Problem







