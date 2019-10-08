#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import pickle
import igraph
import time as _time
import numpy as np
from src.util import compute_delaunay, compute_distance
import sys
from contextlib import contextmanager
import psycopg2
from src.deprecated.config import *


# util.py

def hash_array(a):
    """Hash function for numpy arrays.
    (not 100% efficient, due to copying in .tobytes()).
    """
    return a.data.tobytes()


def hash_bool_array(a):
    """Hash function for boolean numpy arrays. More space efficient due to .packbits().
    (not 100% efficient, due to copying in .tobytes()).
    """
    return np.packbits(a).data.tobytes()


def cache_kwarg(kwarg_key, hash_function=hash):
    """Decorator to cache functions based on a specified keyword argument.
    Caution: Changes in other arguments are ignored!

    Args:
        kwarg_key (str): The key of the keyword argument to be cached.
        hash_function (callable): Custom hash function (for types with no default hash method).

    Returns:
        callable: Cached function.
    """
    def decorator(fn):
        cache = {}

        def fn_cached(*args, **kwargs):
            kwarg_value = kwargs[kwarg_key]
            hashed = hash_function(kwarg_value)

            if hashed in cache:
                return cache[hashed]
            else:
                result = fn(*args, **kwargs)
                cache[hashed] = result
                return result

        return fn_cached

    return decorator


def cache_global_lh(fn):
    """This is a cache decorator tailored to the function compute_global_likelihood

        Args:
            fn(callable): The function to cache, i.e compute_global_likelihood

        Returns:
            (callable): The cached global likelihood
        """
    global result
    result = None

    def cached_fn(*args, **kwargs):
        global result
        if result is None:
            result = fn(*args, **kwargs)
        return result

    return cached_fn


def cache_decorator(fn):
    """Decorator to cache functions.

    Args:
        fn(callable): The function to cache

    Returns:
        (callable): The cached function
    """

    global result
    result = None

    def cached_fn(*args, **kwargs):
        global result
        if result is None:
            result = fn(*args, **kwargs)
        else:
            if kwargs.pop('recompute', False):
                result = fn(*args, **kwargs)

        return result

    return cached_fn


def cache_arg(arg_id, hash_function=hash):
    """Decorator to cache functions based on a specified argument.
    Caution: Changes in other arguments are ignored!

    Args:
        arg_id (int): The index of the argument to be cached (index in args tuple, i.e. in the
            order of the arguments, as specified in the function header).
        hash_function: Custom hash function (for types with no default hash method).

    Returns:
        callable: Cached function.
    """
    def decorator(fn):
        cache = {}

        def fn_cached(*args, **kwargs):
            arg_value = args[arg_id]
            hashed = hash_function(arg_value)

            if hashed in cache:
                return cache[hashed]
            else:
                result = fn(*args, **kwargs)
                cache[hashed] = result
                return result

        return fn_cached

    return decorator


def timeit(fn):
    """Timing decorator. Measures and prints the time that a function took from call
    to return.
    """
    def fn_timed(*args, **kwargs):
        start = _time.time()

        result = fn(*args, **kwargs)

        time_passed = _time.time() - start
        print('%r  %2.2f ms' % (fn.__name__, time_passed * 1000))

        return result

    return fn_timed


def triangulation(net, zone):
    """ This function computes a delaunay triangulation for a set of input locations in a zone
    Args:
        net (dict): The full network containing all sites.
        zone (np.array): The current zone (boolean array).

    Returns:
        (np.array): the Delaunay triangulation as a weighted adjacency matrix.
    """

    dist_mat = net['dist_mat']
    locations = net['locations']
    v = zone.nonzero()[0]

    # Perform the triangulation
    delaunay_adj_mat = compute_delaunay(locations[v])

    # Multiply with weights (distances)
    g = delaunay_adj_mat.multiply(dist_mat[v][:, v])

    return g


def categories_from_features(features):
    """
        This function returns the number of categories per feature f
        Args:
            features(np.ndarray): features

        Returns:
        (dict) : categories per feature, number of categories per feature
        """
    features_t = np.transpose(features, (1, 2, 0))

    features_cat = []
    for f in features_t:

        features_cat.append(np.count_nonzero(np.sum(f, axis=1)))

    return features_cat


# preprocessing.py


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


def dump_results(path, reevaluate=False):

    def dump_decorator(fn):

        def fn_dumpified(*args, **kwargs):

            reeval = kwargs.pop('reevaluate', reevaluate)
            if 'path' in kwargs:
                file_path = kwargs.pop('path')
            else:
                file_path = path
            if (not reeval) and os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    res = pickle.load(f)
            else:
                res = fn(*args, **kwargs)
                with open(file_path, 'wb') as f:
                    pickle.dump(res, f)
            return res
        return fn_dumpified
    return dump_decorator


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



@contextmanager
def psql_connection(commit=False):
    """This function opens a connection to PostgreSQL, performs a DB operation and then closes the connection

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

def get_network_db(table=None):
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

def get_contact_zones_db(zone_id, table=None):
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

    # For multiple zones
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

# deprecated
def get_network_subset(areal_subset, table=None):
    """ This function retrieves the edge list and the coordinates of the simulated languages
    from the DB and then convertsretrieves the edge list and the coordinates of the simulated languages
    from the DB and then converts thes

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

# Import/export snippets

# Export the simulated features together with other information on the sites
write_languages_to_csv(features=features_sim, sites=sites_sim, families=families_sim,
                       file="data\simulated_features_per_site.csv")

# Export the simulated p_global and p_family
# write_p_to_csv(prob=p_global, file="data\simulated_p_global.csv")

p_family = []
for fam in range(families_sim.shape[0]):
    file = "data\simulated_p_family_" + str(fam) + ".csv"
    p_family_fam = compute_p_from_data(features_sim[families_sim[fam], :, :], return_counts=True)
    write_p_to_csv(prob=p_family_fam.astype(int), file=file)
    p_family.append(p_family_fam)
p_family = np.asarray(p_family)

# Import the features again together with other information on the sites
sites_in, site_names_in, features_in, feature_names_in, category_names_per_feature_in, \
families_in, family_names_in = read_languages_from_csv(file="data\simulated_features_per_site.csv")

# Import p_global and p_family
category_names_in, feature_names_in, p_global_in = read_p_from_csv(file="data\simulated_p_global.csv")

for fam in family_names_in:
    file = "data\simulated_p_" + str(fam) + ".csv"
    category_names_in[fam], feature_names_in[fam], p_family_in_fam = read_p_from_csv(file=file)
    p_family_in.append(p_family_in_fam)
p_family_in = np.asarray(p_family_in)

# Compare for consistency
print(np.array_equal(features_sim, features_in), "features are the same")
print(np.array_equal(families_sim, families_in), "families are the same")
print(np.array_equal(sites_sim['locations'], sites_in['locations']), "sites are the same")
print(np.array_equal(p_global, p_global_in), "p_global is the same")
print(np.array_equal(p_family, p_family_in), "p_global is the same")

# todo: remove after testing is complete
for fam in range(families_sim.shape[0]):
    file = "data/freq_family_" + str(fam) + ".csv"
    fam_features = features_sim[families_sim[fam], :, :]

    n_fam_sites, n_fam_features, n_fam_categories = fam_features.shape
    freq = np.zeros((n_fam_features, n_fam_categories), dtype=float)

    for f in range(n_fam_features):
        counts = np.sum(fam_features[:, f, :], axis=0)
        freq[f] = counts

    write_freq_to_csv(freq=freq.astype(int), categories=categories_sim, file=file)

if PRIOR['p_families'] == "dirichlet":
    fam_freq, categories = get_family_frequencies(PRIOR['p_families_parameters']['files'])

    if categories_sim != categories:
        print("Categories in family frequency data and features do not match. Check for consistency!")

    dirichlet = freq_to_dirichlet(fam_freq, categories)
    PRIOR['p_families_parameters']['dirichlet'] = dirichlet
    PRIOR['p_families_parameters']['categories'] = categories


# model.py

# deprecated
def compute_likelihood_particularity(zone, features, ll_lookup):

    """ This function computes the feature likelihood of a zone. The function performs a one-sided binomial test yielding
    a p-value for the observed presence of a feature in the zone given the presence of the feature in the data.
    The likelihood of a zone is the negative logarithm of this p-value. Zones with more present features than expected
    by the presence of a feature in the data have a small p-value and are thus indicative of language contact.

    Args:
        zone(np.array): The current zone (boolean array).
        features(np.array): The feature matrix.
        ll_lookup(dict): A lookup table of likelihoods for all features for a specific zone size.

    Returns:
        float: The feature-likelihood of the zone.
    """

    idx = zone.nonzero()[0]
    zone_size = len(idx)

    # Count the number of languages per category in the zone
    log_lh = []
    for f_idx, f in enumerate(np.transpose(features[idx])):

        bin_test_per_cat = []
        for cat in ll_lookup[f_idx].keys():

            # -1 denotes missing values
            #Todo Remove 0, only for testing
            if cat != -1 and cat != 0:
                cat_count = (f == cat).sum(0)
                # Perform Binomial test
                bin_test_per_cat.append(ll_lookup[f_idx][cat][zone_size][cat_count])

        # Keep the most surprising category per feature
        log_lh.append(max(bin_test_per_cat))

    return sum(log_lh)


# deprecated
def compute_feature_likelihood_old(zone, features, ll_lookup):

    """ This function computes the feature likelihood of a zone. The function performs a one-sided binomial test yielding
    a p-value for the observed presence of a feature in the zone given the presence of the feature in the data.
    The likelihood of a zone is the negative logarithm of this p-value. Zones with more present features than expected
    by the presence of a feature in the data have a small p-value and are thus indicative of language contact.
    Args:
        zone(np.array): The current zone (boolean array).
        features(np.array): The feature matrix.
        ll_lookup(dict): A lookup table of likelihoods for all features for a specific zone size.
    Returns:
        float: The feature-likelihood of the zone.
    """

    idx = zone.nonzero()[0]
    zone_size = len(idx)

    # Count the presence and absence
    present = features[idx].sum(axis=0)
    log_lh = 0

    for f_idx, f_freq in enumerate(present):
        log_lh += ll_lookup[f_idx][zone_size][f_freq]

    return log_lh

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


#deprecated
def generate_ecdf_geo_prior(net, min_n, max_n, nr_samples):
    """ This function generates an empirical cumulative density function (ecdf), which is then used to compute the
    geo-prior of a contact zone. The function
    a) grows <nr samples> contact zones of size n, where n is between <min_n> and <max_n>,
    b) for each contact zone: generates a complete graph, delaunay graph and a minimum spanning tree
    and computes the summed length of each graph's edges
    c) for each size n: generates an ecdf of all summed lengths
    d) fits a gamma function to the ecdf

    Args:
        net (dict): network containing the graph, locations,...
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

# deprecated
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


# deprecated
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


