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

from src.util import compute_distance, triangulation, \
    compute_delaunay, dump_results
from src.config import *



EPS = np.finfo(float).eps

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
def get_network(table=None):
    """ This function retrieves the edge list and the coordinates of the simulated languages
    from the DB and then converts these into a spatial network.

    Returns:
        dict: a dictionary containing the network, its vertices and edges, an adjacency list and matrix
        and a distance matrix
    """
    if table is None:
        table = DB_ZONE_TABLE
    # Get vertices from PostgreSQL
    with psql_connection(commit=True) as connection:
        query_v = "SELECT gid, mx AS x, my AS y " \
                  "FROM {table} " \
                  "ORDER BY gid;".format(table=table)

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
           'dist_mat': dist_mat,
           'gid_to_idx': gid_to_idx,
           'idx_to_gid': idx_to_gid}

    return net


def get_network_subset(areal_subset, table=None):
    """ This function retrieves a subset of the network
        from the DB. Moreover it returns a matching between the full network and the subset.

        Args:
            areal_subset (int): id of the subset.
            full_network (np.ndarray): the full network
        Returns:
            dict: a dictionary containing the network, its vertices and edges, an adjacency list and matrix
            and a distance matrix
        """
    if table is None:
        table = DB_ZONE_TABLE
    # Get only those vertices which belong to a specific subset of the network
    with psql_connection(commit=True) as connection:
        query_v = "SELECT gid, mx AS x, my AS y " \
                  "FROM {table} " \
                  "WHERE cz = {id} " \
                  "ORDER BY gid;".format(table=table, id=areal_subset)
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
           'dist_mat': dist_mat,
           'gid_to_idx': gid_to_idx,
           'idx_to_gid': idx_to_gid}

    return net


@dump_results(CONTACT_ZONES_PATH, RELOAD_DATA)
def get_contact_zones(zone_id, table=None):
    """This function retrieves contact zones from the DB
    Args:
        zone_id(int or tuple of ints) : the id(s) of the zone(s) in the DB
        table(string): the name of the table in the DB
    Returns:
        dict: the contact zones
        """
    if table is None:
        table = DB_ZONE_TABLE
    # For single zones
    if isinstance(zone_id, int):
        with psql_connection(commit=True) as connection:
            query_cz = "SELECT cz, array_agg(gid) " \
                       "FROM {table} " \
                       "WHERE cz = {list_id} " \
                       "GROUP BY cz".format(table=table, list_id=zone_id)
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
                query_cz = "SELECT cz, array_agg(gid) " \
                           "FROM {table} " \
                           "WHERE cz IN {list_id} " \
                           "GROUP BY cz".format(table=table, list_id=zone_id)
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


def assign_na(features, n_na):
    """ Randomly assign NAs to features. Makes the simulated data more realistic.
    Args:
        features(np.ndarray): binary feature array, with shape = (sites, features, categories)
        n_na: number of NAs added
    returns: features(np.ndarray): binary feature array, with shape = (sites, features, categories)
    """

    features = features.astype(float)
    # Choose a random site and feature and set to None
    for _ in range(n_na):

        na_site = np.random.choice(a=features.shape[0], size=1)
        na_feature = np.random.choice(a=features.shape[1], size=1)
        features[na_site, na_feature, :] = None


# Deprecated
def define_contact_features(n_feat, r_contact_feat, contact_zones):
    """ This function returns a list of features and those features for which contact is simulated
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


# Deprecated
@dump_results(FEATURES_BG_PATH, RELOAD_DATA)
def simulate_background_distribution(n_features, n_sites,
                                     return_as="np.array", cat_axis=True):
    """This function randomly draws <n_sites> samples from a Categorical distribution
    for <all_features>. Features can have up to four categories, most features are binary, though.

    Args:
        n_features (int): number of simulated features
        n_sites (int): number of sites for which feature are simulated
        return_as: (string): the data type of the returned background and probabilities, either "dict" or "np.array"
        cat_axis (boolean): return categories as separate axis? only evaluated when <return_as> is "np.array"

    Returns:
        (dict, dict) or (np.ndarray, dict): the background distribution and the probabilities to simulate them
        """
    # Define features

    features = []
    for f in range(n_features):
        features.append('f' + str(f + 1))

    features_bg = {}
    prob_bg = {}

    for f in features:

        # Define the number of categories per feature (i.e. how many categories can the feature take)
        nr_cats = np.random.choice(a=[2, 3, 4], size=1, p=[0.7, 0.2, 0.1])[0]
        cats = list(range(0, nr_cats))

        # Randomly define a probability for each category from a Dirichlet (3, ..., 3) distribution
        # The mass is in the center of the simplex, extreme values are less likely
        p_cats = np.random.dirichlet(np.repeat(3, len(cats)), 1)[0]

        # Assign each site to one of the categories according to p_cats
        bg = np.random.choice(a=cats, p=p_cats, size=n_sites)

        features_bg[f] = bg
        prob_bg[f] = p_cats

    if return_as == "dict":
        return features_bg, prob_bg

    elif return_as == "np.array":

        # Sort by key (-f) ...
        keys = [f[1:] for f in features_bg.keys()]
        keys_sorted = ['f' + str(s) for s in sorted(list(map(int, keys)))]

        # ... and store to np.ndarray
        features_bg_array = np.ndarray.transpose(np.array([features_bg[i] for i in keys_sorted]))

        # Leave the dimensions as is
        if not cat_axis:
            return features_bg_array, prob_bg

        # Add categories as dimension
        else:
            cats = np.unique(features_bg_array)
            features_bg_cat = []

            for cat in cats:
                cat_axis = np.expand_dims(np.where(features_bg_array == cat, 1, 0), axis=2)
                features_bg_cat.append(cat_axis)
            features_bg_cat_array = np.concatenate(features_bg_cat, axis=2)

            return features_bg_cat_array, prob_bg
    else:
        raise ValueError('return_as must be either "dict" or np.array:')


def sample_categorical(p):
    """Sample from a (multidimensional) categorical distribution. The
    probabilities for every category are given by `p`

    Args:
        p (np.array): Array defining the probabilities of every category at
            every site of the output array. The last axis defines the categories
            and should sum up to 1.
            shape: (*output_dims, n_categories)
    Returns
        np.array: Samples of the categorical distribution.
            shape: output_dims
    """
    *output_dims, n_categories = p.shape

    cdf = np.cumsum(p, axis=-1)
    z = np.expand_dims(np.random.random(output_dims), axis=-1)

    return np.argmax(z < cdf, axis=-1)


def sample_from_likelihood(zones, families, p_zones, p_global, p_families, weights, cat_axis=True):
    """Compute the likelihood of all sites in `zone`. The likelihood is
    defined as a mixture of the global distribution `p_global` and the maximum
    likelihood distribution of the zone itself.

    Args:
        zones (np.array[bool]): Binary array indicating the assignment of sites to zones.
            shape: (n_sites, n_zones)
        families (np.array): Binary array indicating the assignment of a site to
            a language family.
            shape: (n_sites, n_families)
        p_zones (np.array[float]): The categorical probabilities of every
            category in every features in every zones.
            shape: (n_zones, n_features, n_categories)
        p_global (np.array[float]): The global categorical probabilities of
            every category in every feature.
            shape: (n_features, n_categories)
        p_families (np.array): The probabilities of every category in every
            language family.
            shape: (n_families, n_features, n_categories)
        weights (np.array[float]): The mixture coefficient controlling how much
            each feature is explained by the zone, global distribution and
            language families distribution.
            shape: (n_features, 3)
        cat_axis (boolean): return categories as separate axis? only evaluated when <return_as> is "np.array"

    Returns:
        np.array: The sampled categories for all sites and features (and categories if cat_axis=True)
        shape: either (n_sites, n_features) or (n_sites, n_features, n_categories)
    """
    n_sites, n_zones = zones.shape
    n_features, n_categories = p_global.shape

    # Are the weights fine?
    if families is None:

        assert np.allclose(a=np.sum(weights[:, [0, 1]], axis=-1), b=1., rtol=EPS)

    else:
        assert np.allclose(a=np.sum(weights, axis=-1), b=1., rtol=EPS)

    features = np.zeros((n_sites, n_features), dtype=int)
    for i_feat in range(n_features):
            # Compute the feature likelihood matrix (for all sites and all categories)
            lh_zone = zones.dot(p_zones[:, i_feat, :])
            lh_global = p_global[np.newaxis, i_feat, :]

            lh_feature = weights[i_feat, 0] * lh_zone \
                       + weights[i_feat, 1] * lh_global \

            # Simulate family
            if families is not None:

                lh_family = families.dot(p_families[:, i_feat, :])
                lh_feature += weights[i_feat, 2] * lh_family

            # Sample from the categorical distribution defined by lh_feature
            features[:, i_feat] = sample_categorical(lh_feature)

    # # Todo Remove once we are sure both lh return the same results
    # # Export old_features for testing
    # features_dict = {}
    # for f in range(features.shape[-1]):
    #     f_name = 'f' + str(f+1)
    #     features_dict[f_name] = features[:, f]

    # Add categories as dimension
    if cat_axis:
        cats = np.unique(features)
        features_cat = np.zeros((n_sites, n_features, len(cats)), dtype=int)

        for cat in cats:
            features_cat[:, :, cat] = np.where(features == cat, 1, 0)
        return features_cat, features
    else:
        return features


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
def compute_p_global(feat):
    """This function computes the global probabilities for a feature f to belongs to category cat;

    Args:
        feat (np.ndarray): a matrix of all features

    Returns:
        array: the empirical probability that feature f belongs to category cat"""
    n_sites, n_features, n_categories = feat.shape
    p = np.zeros((n_features, n_categories), dtype=float)

    for f in range(n_features):
        p[f] = np.sum(feat[:, f, :], axis=0) / n_sites
    return p


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
def precompute_feature_likelihood_old(min_size, max_size, feat_prob, log_surprise=True):
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

    def ll(s_zone, n_zone, p_global, log_surprise):
        p_value = stats.binom_test(s_zone, n_zone, p_global, 'greater')

        if log_surprise:
            try:
                lh = -math.log(p_value)
            except Exception as e:
                print(s_zone, n_zone, p_global, p_value)
                raise e
        else:
            try:
                lh = 1/p_value
            except ZeroDivisionError:
                lh = math.inf

        # Log-Likelihood
        try:
            return math.log(lh)
        except ValueError:
            return -math.inf

    lookup_dict = {}

    for features, categories in feat_prob.items():
        lookup_dict[features] = {}
        for cat, p_global in categories.items():
            # -1 denotes missing values
            if cat != -1:
                lookup_dict[features][cat] = {}
                for n_zone in range(min_size, max_size + 1):
                    lookup_dict[features][cat][n_zone] = {}
                    for s_zone in range(n_zone + 1):
                        lookup_dict[features][cat][n_zone][s_zone] = \
                            ll(s_zone, n_zone, p_global, log_surprise)

    return lookup_dict


@dump_results(FEATURES_PATH, RELOAD_DATA)
def get_features(table=None, feature_names=None):
    """ This function retrieves features from the geodatabase
    Args:
        table(string): the name of the table
    Returns:
        dict: an np.ndarray of all features
    """

    feature_columns = ','.join(feature_names)

    if table is None:
        table = DB_ZONE_TABLE
    # Get vertices from PostgreSQL
    with psql_connection(commit=True) as connection:
        query_v = "SELECT {feature_columns} " \
                  "FROM {table} " \
                  "ORDER BY gid;".format(feature_columns=feature_columns, table=table)

        cursor = connection.cursor()
        cursor.execute(query_v)
        rows_v = cursor.fetchall()

        features = []
        for f in rows_v:
            f_db = list(f)

            # Replace None with -1
            features.append([-1 if x is None else x for x in f_db])

        return np.array(features)

