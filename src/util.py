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


def grow_zone(size, net, already_in_zone=None):
    """ This function grows a zone of size <size> excluding any of the nodes in <already_in_zone>.
    Args:
        size (int): The number of nodes in the zone.
        net (dict): A dictionary comprising all sites of the network.
        already_in_zone (np.array): All nodes in the network already assigned to a zone (boolean array)

    Returns:
        (np.array, np.array): The new zone (boolean), all nodes in the network already assigned to a zone (boolean)

    """
    n = net['adj_mat'].shape[0]

    if already_in_zone is None:
        already_in_zone = np.zeros(n, bool)

    # Initialize the zone
    zone = np.zeros(n).astype(bool)

    # Get all vertices that already belong to a zone (occupied_n) and all free vertices (free_n)
    occupied_n = np.nonzero(already_in_zone)[0]
    free_n = set(range(n)) - set(occupied_n)
    i = random.sample(free_n, 1)[0]
    zone[i] = already_in_zone[i] = 1

    for _ in range(size-1):

        neighbours = get_neighbours(zone, already_in_zone, net['adj_mat'])
        # Add a neighbour to the zone
        i_new = random.choice(neighbours.nonzero()[0])
        zone[i_new] = already_in_zone[i_new] = 1

    return zone, already_in_zone



def triangulation(net, zone):
    """ This function computes a delaunay triangulation for a set of input locations in a zone
    Args:
        net (dict): The full network containing all sites.
        zone (np.array): The current zone (boolean array).

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