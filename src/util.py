#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import pickle
import random
from math import sqrt, atan2

import numpy as np
import time as _time

import scipy as sp


def beta_binom_logpmf(alpha, beta, n, k):
    part_1 = sp.special.comb(n, k)
    part_2 = sp.special.betaln(k + alpha, n - k + beta)
    part_3 = sp.special.betaln(alpha, beta)

    return np.mean(np.log(part_1) + part_2 - part_3)


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
    """
        This function computes the Euclidean distance between two points a and b
        :In
          - A: x and y coordinates of the point a, in a metric CRS
          - B: x and y coordinates of the point b, in a metric CRS.
        :Out
          - dist: the Euclidean distance from a to b
        """
    a = np.asarray(a)
    b = np.asarray(b)
    ab = b-a
    dist = sqrt(ab[0]**2 + ab[1]**2)

    return dist


def compute_direction(a, b):
    """
        This function computes the direction between two points a and b in clockwise direction
        north = 0 , east = pi/2, south = pi, west = 3pi/2
        :In
          - A: x and y coordinates of the point a, in a metric CRS
          - B: x and y coordinates of the point b, in a metric CRS.
        :Out
          - dir_rad: the direction from A to B in radians
        """
    a = np.asarray(a)
    b = np.asarray(b)
    ba = b - a
    return (np.pi/2 - atan2(ba[1], ba[0])) % (2*np.pi)


def dump(data, path):
    with open(path, 'wb') as dump_file:
        pickle.dump(data, dump_file)


def load_from(path):
    with open(path, 'rb') as dump_file:
        return pickle.load(dump_file)


def get_neighbours(zone, adj_mat):
    """Compute the neighbourhood of a zone (excluding vertices from the zone itself)."""
    return np.logical_and(adj_mat.dot(zone), ~zone)


def grow_zone(size, adj_mat):
    """Grow a zone of size <size> by starting from a random point and iteratively
    adding random neighbours.

    Args:
        size (int): Desired size of the zone.
        adj_mat (np.ndarray): Adjacency Matrix

    Returns:
        (np.ndarray): The randomly generated zone of size <size>.
    """
    n = adj_mat.shape[0]
    zone = np.zeros(n).astype(bool)

    # Choose starting point
    i = random.randrange(size)
    zone[i] = 1

    for _ in range(size-1):
        neighbours = get_neighbours(zone, adj_mat)

        # Add a neighbour to the zone.
        i_new = random.choice(neighbours.nonzero()[0])
        zone[i_new] = 1

    return zone