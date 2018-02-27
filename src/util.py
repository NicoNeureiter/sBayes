#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import pickle
import random
import igraph
import numpy as np
import time as _time
import scipy.spatial as spatial

from math import sqrt


def reachable_vertices(x, adj_mat):
    return adj_mat.dot(x).clip(0, 1)


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
    with open(path, 'wb') as dump_file:
        pickle.dump(data, dump_file)


def load_from(path):
    with open(path, 'rb') as dump_file:
        return pickle.load(dump_file)


def get_neighbours(zone, adj_mat):
    """Compute the neighbourhood of a zone (excluding vertices from the zone itself).

    Args:
        zone (np.ndarray): the current zone (boolean array)
        adj_mat (np.ndarray): Adjacency Matrix (boolean)

    Returns:
        np.ndarray: The neighborhood of the zone (boolean array)
    """
    return np.logical_and(adj_mat.dot(zone), ~zone)

# Peter
# -> sampling
def grow_zone(size, adj_mat):
    """Grow a zone of size <size> by starting from a random point and iteratively
    adding random neighbours.

    Args:
        size (int): Desired size of the zone.
        adj_mat (np.ndarray): Adjacency Matrix (boolean)

    Returns:
        np.ndarray: The randomly generated zone of size <size>.
    """
    n = adj_mat.shape[0]
    zone = np.zeros(n).astype(bool)

    # Choose starting point
    i = random.randrange(n)
    zone[i] = 1

    for _ in range(size-1):
        neighbours = get_neighbours(zone, adj_mat)

        # Add a neighbour to the zone.
        i_new = random.choice(neighbours.nonzero()[0])
        zone[i_new] = 1

    return zone


def triangulation(net, zone):
    """ This function computes a delaunay triangulation for a set of input locations in a zone
    Args:
        net (dict): The full network containing all sites.
        zone (np.ndarray): The current zone (boolean array).

    Returns:
        (graph): the delaunay triangulation as a weighted graph
    """

    dist_mat = net['dist_mat']
    v = zone.nonzero()[0]

    # Perform the triangulation
    locations = net['locations'][v]
    delaunay = spatial.Delaunay(locations, qhull_options="QJ Pp")

    # Initialize the graph
    g = igraph.Graph()
    g.add_vertices(len(v))

    for t in range(delaunay.nsimplex):
        edge = sorted([delaunay.simplices[t, 0], delaunay.simplices[t, 1]])
        g.add_edge(edge[0], edge[1], weight=dist_mat[v[edge[0]], v[edge[1]]])

        edge = sorted([delaunay.simplices[t, 0], delaunay.simplices[t, 2]])
        g.add_edge(edge[0], edge[1], weight=dist_mat[v[edge[0]], v[edge[1]]])

        edge = sorted([delaunay.simplices[t, 1], delaunay.simplices[t, 2]])
        g.add_edge(edge[0], edge[1], weight=dist_mat[v[edge[0]], v[edge[1]]])

    return g


# edges.append(coords)
#
# edge = sorted([delaunay.simplices[t, 0], delaunay.simplices[t, 2]])
# coords = {'A': tuple(locations[edge[0]]),
#           'B': tuple(locations[edge[1]]),
#           'length': dist_mat[v[edge[0]], v[edge[1]]]}
# edges.append(coords)
#
# edge = sorted([delaunay.simplices[t, 1], delaunay.simplices[t, 2]])
# coords = {'A': tuple(locations[edge[0]]),
#           'B': tuple(locations[edge[1]]),
#           'length': dist_mat[v[edge[0]], v[edge[1]]]}
# edges.append(coords)
#
# locations = net['locations'][v]
# sub_g = net['graph'].subgraph(v, implementation="create_from_scratch")
# mst = sub_g.spanning_tree(weights=sub_g.es["weight"])
#
# for e in sub_g.es:
#     coords = {'A': tuple(locations[e.tuple[0]]),
#               'B': tuple(locations[e.tuple[1]]),
#               'length': e['weight']}
#
#     edges.append(coords)