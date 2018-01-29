#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import sys
from contextlib import contextmanager

import numpy as np
import pandas
import psycopg2
from igraph import Graph


@contextmanager
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
                  "FROM cz_sim.contact_zone_7_test;"
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
                  "FROM cz_sim.delaunay_edge_list;"

        cursor = connection.cursor()
        cursor.execute(query_e)
        rows_e = cursor.fetchall()

    n_e = len(rows_e)
    edges = np.zeros((n_e, 2)).astype(int)
    for i, e in enumerate(rows_e):
        edges[i, 0] = gid_to_idx[e[0]]
        edges[i, 1] = gid_to_idx[e[1]]

    import scipy.sparse as sps
    adj_list = [[] for _ in range(n_v)]
    adj_mat = sps.lil_matrix((n_v, n_v))

    for v1, v2 in edges:
        adj_list[v1].append(v2)
        adj_list[v2].append(v1)
        adj_mat[v1, v2] = 1
        adj_mat[v2, v1] = 1

    adj_list = np.array(adj_list)
    adj_mat = adj_mat.tocsr()

    net = {'vertices': vertices,
           'edges': edges,
           'locations': locations,
           'adj_list': adj_list,
           'adj_mat': adj_mat,
           'n': n_v,
           'm': n_e
           # 'network': lang_network,
           # 'bbox': bbox
           }
    return net


def get_features():
    """
        This function retrieves the features of all simulated languages from the DB
        :In
        -
        :Out
        - f: a binary matrix of all features
        """
    with psql_connection(commit=True) as connection:

        # Retrieve the features for all languages
        f = pandas.read_sql_query("SELECT f1, f2, f3, f4, f5, f6,"
                                  "f7, f8, f9, f10, f11, f12, f13,"
                                  "f14, f15, f16, f17, f18, f19, f20,"
                                  "f21, f22, f23, f24, f25, f26, f27,"
                                  "f28, f29, f30 "
                                  "FROM cz_sim.contact_zone_7_test ORDER BY id ASC;", connection)
    f = f.as_matrix()

    # Change A to 0 and P to 1
    f[f == 'A'] = 0
    f[f == 'P'] = 1

    return f


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