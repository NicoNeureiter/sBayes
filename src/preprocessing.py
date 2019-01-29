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
from scipy import stats, sparse
from scipy.sparse.csgraph import minimum_spanning_tree
import psycopg2
import igraph
import math

from src.util import compute_distance, grow_zone, triangulation, \
    compute_delaunay, dump_results
from src.config import *




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
                  "FROM {table} " \
                  "ORDER BY id;".format(table=DB_ZONE_TABLE)
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

    # Edges
    delaunay = compute_delaunay(locations)
    v1, v2 = delaunay.toarray().nonzero()
    edges = np.column_stack((v1, v2))

    # Adjacency Matrix
    adj_mat = delaunay.tocsr()

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
           'adj_mat': adj_mat,
           'n': n_v,
           'm': edges.shape[0],
           'graph': g,
           'dist_mat': dist_mat}

    return net


def get_network_subset(areal_subset):
    """ This function is similar to get_network(), with the main difference that it retrieves a subset of the network
        from the DB, rather than the full network. Moreover, the returned subset is not pickled.

        Args:
            areal_subset (int): id of the subset. If None the complete network is retrieved.
        Returns:
            dict: a dictionary containing the network, its vertices and edges, an adjacency list and matrix
            and a distance matrix
        """

    # Get only those vertices which belong to a specific subset of the network
    with psql_connection(commit=True) as connection:
        query_v = "SELECT id, mx AS x, my AS y " \
                  "FROM {table} " \
                  "WHERE cz = {id} " \
                  "ORDER BY id;".format(table=DB_ZONE_TABLE, id=areal_subset)
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

    # Edges
    delaunay = compute_delaunay(locations)
    v1, v2 = delaunay.toarray().nonzero()
    edges = np.column_stack((v1, v2))

    # Adjacency Matrix
    adj_mat = delaunay.tocsr()

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
           'adj_mat': adj_mat,
           'n': n_v,
           'm': edges.shape[0],
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
                       "FROM {table} " \
                       "WHERE cz = {list_id} " \
                       "GROUP BY cz".format(table=DB_ZONE_TABLE, list_id=zone_id)
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
                           "FROM {table} " \
                           "WHERE cz IN {list_id} " \
                           "GROUP BY cz".format(table=DB_ZONE_TABLE, list_id=zone_id)
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


def define_contact_features(n_feat, r_contact_feat, contact_zones):
    """ This function returns all features and those features for which contact is simulated
    Args:
        n_feat (int): number of features
        r_contact_feat (float): percentage of features for which contact is simulated
        contact_zones (dict): a region of sites for which contact is simulated

    Returns:
        list, dict: all features, the features for which contact is simulated (per contact zone)
    """

    n_contact_feat = int(math.ceil(r_contact_feat * n_feat))
    features = []
    for f in range(n_feat):
        features.append('f' + str(f + 1))

    contact_features = {}
    for cz in contact_zones:

        contact_features[cz] = np.random.choice(features, n_contact_feat, replace=False)

    return features, contact_features


@dump_results(FEATURES_BG_PATH, RELOAD_DATA)
def simulate_background_distribution(features, contact_features, n_sites, p_min, p_max, p_max_contact):
    """This function draws <n_sites> samples from a random Binomial distribution to simulate a background distribution
    for <all_features>, where 1 implies the presence of the feature, and 0 the absence.
    For those features that are in <contact_features> the probability of success is limited to [0, p_max_contact]

    Args:
        features (list): names of all features
        contact_features (dict): names of all contact features
        n_sites (int): number of sites for which feature are simulated
        p_min (float): the minimum probability
        p_max (float): the maximum probability
        p_max_contact: (float): the maximum probability of a feature for which contact is simulated

    Returns:
        dict: the background distribution
        """
    contact = set({x for v in contact_features.values() for x in v})

    features_bg = {}
    for f in features:

        if f in contact:
            success = np.random.uniform(p_min, p_max_contact-0.2, 1)

        else:
            success = np.random.uniform(p_min, p_max, 1)

        bg = np.random.binomial(n=1, p=success, size=n_sites)
        features_bg[f] = bg

    return features_bg


@dump_results(FEATURES_PATH, RELOAD_DATA)
def simulate_contact(features, contact_features, p, contact_zones):
    """This function simulates language contact. For each contact zone the function randomly chooses <n_feat> features,
    for which the similarity is increased.

    Args:
        features (dict): all features
        contact_features(list): features for which contact is simulated
        p (float): probability of success, defines the degree of similarity in the contact zone
        contact_zones (dict): a region of sites for which contact is simulated

    Returns:
        np.ndarray: the adjusted features
        """
    features_adjusted = copy.deepcopy(features)

    # Iterate over all contact zones
    for cz in contact_zones:

        # Choose <n_feat> features for which the similarity is increased

        for f in contact_features[cz]:

            # increase similarity

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


def estimate_ecdf_n(n, nr_samples, net):

    logging.info('Estimating empirical CDF for n = %i' % n)

    dist_mat = net['dist_mat']

    complete_stat = []
    delaunay_stat = []
    mst_stat = []

    for _ in range(nr_samples):

        zone, _ = grow_zone(n, net)

        # Mean distance / Mean squared distance

        # Complete graph
        complete = sparse.triu(dist_mat[zone][:, zone])
        n_complete = complete.nnz
        # complete_stat.append(complete.sum()/n_complete)   # Mean
        # complete_stat.append(np.sum(complete.toarray() **2.) / n_complete)  # Mean squared

        # Delaunay graph
        triang = triangulation(net, zone)
        delaunay = sparse.triu(triang)
        n_delaunay = delaunay.nnz
        # delaunay_stat.append(delaunay.sum()/n_delaunay)   # Mean
        # delaunay_stat.append(np.sum(delaunay.toarray() **2.) / n_delaunay)  # Mean squared

        # Minimum spanning tree (MST)
        mst = minimum_spanning_tree(triang)
        n_mst = mst.nnz
        # mst_stat.append(mst.sum()/n_mst)     # Mean
        # mst_stat.append(np.sum(mst.toarray() **2) / n_mst)  # Mean squared

        # Max distance
        # # Complete graph
        complete_max = dist_mat[zone][:, zone].max(axis=None)
        complete_stat.append(complete_max)

        # Delaunay graph
        #triang = triangulation(net, zone)
        delaunay_max = triang.max(axis=None)
        delaunay_stat.append(delaunay_max)

        # MST
        #mst = minimum_spanning_tree(triang)
        mst_max = mst.max(axis=None)
        mst_stat.append(mst_max)

    distances = {'complete': complete_stat, 'delaunay': delaunay_stat, 'mst': mst_stat}

    return distances


@dump_results(ECDF_GEO_PATH, RELOAD_DATA)
def generate_ecdf_geo_prior(net, min_n, max_n, nr_samples):
    """ This function generates an empirical cumulative density function (ecdf), which is then used to compute the
    geo-prior of a contact zone. The function
    a) grows <nr samples> contact zones of size n, where n is between <min_n> and <max_n>,
    b) for each contact zone: generates a complete graph, delaunay graph and a minimum spanning tree
    and computes the summed length of each graph's edges
    c) for each size n: generates an ecdf of all summed lengths
    d) fits a gamma function to the ecdf

    Args:
        net (dict): network containing the graph, location,...
        min_n (int): the minimum number of languages in a zone
        max_n (int): the maximum number of languages in a zone
        nr_samples (int): the number of samples in the ecdf per zone size

    Returns:
        dict: a dictionary comprising the empirical and fitted ecdf for all types of graphs
        """

    n_values = range(min_n, max_n+1)

    estimate_ecdf_n_ = partial(estimate_ecdf_n, nr_samples=nr_samples, net=net)

    with Pool(7, maxtasksperchild=1) as pool:
        distances = pool.map(estimate_ecdf_n_, n_values)

    complete = []
    delaunay = []
    mst = []

    for d in distances:

        complete.extend(d['complete'])
        delaunay.extend(d['delaunay'])
        mst.extend(d['mst'])

    # Fit a gamma distribution distribution to each type of graph
    ecdf = {'complete': {'fitted_gamma': stats.gamma.fit(complete, floc=0), 'empirical': complete},
            'delaunay': {'fitted_gamma': stats.gamma.fit(delaunay, floc=0), 'empirical': delaunay},
            'mst': {'fitted_gamma': stats.gamma.fit(mst, floc=0), 'empirical': mst}}

    return ecdf

@dump_results(RANDOM_WALK_COV_PATH, True)  #RELOAD_DATA)
def estimate_random_walk_covariance(net):
    dist_mat = net['dist_mat']
    locations = net['locations']

    delaunay = compute_delaunay(locations)
    mst = delaunay.multiply(dist_mat)
    # mst = minimum_spanning_tree(delaunay.multiply(dist_mat))
    # mst += mst.T  # Could be used as data augmentation? (enforces 0 mean)

    # Compute difference vectors along mst
    i1, i2 = mst.nonzero()
    diffs = locations[i1] - locations[i2]

    # Center at (0, 0)
    diffs -= np.mean(diffs, axis=0)[None, :]

    return np.cov(diffs.T)


@dump_results(LOOKUP_TABLE_PATH, RELOAD_DATA)
def precompute_feature_likelihood(min_size, max_size, feat_prob, log_surprise=True):
    """This function generates a lookup table of likelihoods
    Args:
        min_size (int): the minimum number of languages in a zone.
        max_size (int): the maximum number of languages in a zone.
        feat_prob (np.array): the probability of a feature to be present.
        log_surprise: define surprise with logarithm (see below)
    Returns:
        dict: the lookup table of likelihoods for a specific feature,
            sample size and observed presence.
    """

    # The binomial test computes the p-value of having k or more (!) successes out of n trials,
    # given a specific probability of success
    # For a two-sided binomial test, simply remove "greater".
    # The p-value is then used to compute surprise either as
    # a) -log(p-value)
    # b) 1/p-value,
    # the latter being more sensitive to exceptional observations.
    # and then returns the log likelihood.

    def ll(p_zone, s, p_global, log_surprise):
        p = stats.binom_test(p_zone, s, p_global, 'greater')

        if log_surprise:
            try:
                lh = -math.log(p)
            except Exception as e:
                print(p_zone, s, p_global, p)
                raise e
        else:
            try:
                lh = 1/p
            except ZeroDivisionError:
                lh = math.inf

        # Log-Likelihood
        try:
            return math.log(lh)
        except ValueError:
            return -math.inf

    lookup_dict = {}
    for i_feat, p_global in enumerate(feat_prob):
        lookup_dict[i_feat] = {}
        for s in range(min_size, max_size + 1):
            lookup_dict[i_feat][s] = {}
            for p_zone in range(s + 1):
                lookup_dict[i_feat][s][p_zone] = ll(p_zone, s, p_global, log_surprise)

    return lookup_dict


# TODO Nico: pickle files laden oder nicht Problem
