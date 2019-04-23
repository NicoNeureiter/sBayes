#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import pickle
import random
import time as _time

import igraph
import numpy as np
import scipy.spatial as spatial
from scipy.sparse import csr_matrix

from math import sqrt


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

dct = {}
dct.pop('from_cache', False)


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


def compute_distance(a, b):
    """ This function computes the Euclidean distance between two points a and b

    Args:
        a (array): The x and y coordinates of a point in a metric CRS.
        b (array): The x and y coordinates of a point in a metric CRS.

    Returns:
        float: Distance between a and b
        """

    a = np.asarray(a)
    b = np.asarray(b)
    ab = b-a
    dist = sqrt(ab[0]**2 + ab[1]**2)

    return dist


def dump(data, path):
    """Dump the given data to the given path (using pickle)."""
    with open(path, 'wb') as dump_file:
        pickle.dump(data, dump_file)


def load_from(path):
    """Load and return data from the given path (using pickle)."""
    with open(path, 'rb') as dump_file:
        return pickle.load(dump_file)


def bounding_box(points):
    """ This function retrieves the bounding box for a set of 2-dimensional input points
    Args:
        points (numpy.array): Point tuples (x,y) for which the bounding box is computed

    Returns:
        (dict): the bounding box of the points
    """
    x = [x[0] for x in points]
    y = [x[1] for x in points]
    box = {'x_max': max(x),
           'y_max': max(y),
           'x_min': min(x),
           'y_min': min(y)}

    return box


@cache_arg(0, hash_bool_array)
def get_neighbours(zone, already_in_zone, adj_mat):
    """This function computes the neighbourhood of a zone, excluding vertices already
    belonging to this zone or any other zone.

    Args:
        zone (np.array): The current contact zone (boolean array)
        already_in_zone (np.array): All nodes in the network already assigned to a zone (boolean array)
        adj_mat (np.array): The adjacency matrix (boolean)

    Returns:
        np.array: The neighborhood of the zone (boolean array)
    """

    # Get all neighbors of the current zone, excluding all vertices that are already in a zone

    neighbours = np.logical_and(adj_mat.dot(zone), ~already_in_zone)
    return neighbours


def compute_delaunay(locations):
    n, _ = locations.shape
    delaunay = spatial.Delaunay(locations, qhull_options="QJ Pp")

    indptr, indices = delaunay.vertex_neighbor_vertices
    data = np.ones_like(indices)
    return csr_matrix((data, indices, indptr), shape=(n, n))


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


def zones_autosimilarity(zones, t):
    """
    This function computes the similarity of consecutive zones in a chain
    Args:
        zones (list): zones
        t (integer): lag between consecutive zones in the chain

    Returns:
    (float) : mean similarity between zones in the chain with lag t
    """
    z = np.asarray(zones)
    z = z[:, 0, :]
    unions = np.maximum(z[t:], z[:-t])
    intersections = np.minimum(z[t:], z[:-t])
    sim_norm = np.sum(intersections, axis=1) / np.sum(unions, axis=1)

    return np.mean(sim_norm)


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

