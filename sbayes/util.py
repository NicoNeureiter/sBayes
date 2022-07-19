#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import datetime
import time
import csv
import os
from pathlib import Path
from functools import lru_cache
from math import sqrt, floor, ceil
from itertools import combinations, permutations
from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

import pandas as pd
import scipy
import scipy.spatial as spatial
from scipy.special import betaln, expit
import scipy.stats as stats
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


EPS = np.finfo(float).eps


FAST_DIRICHLET = True
dirichlet_logpdf = stats.dirichlet._logpdf if FAST_DIRICHLET else stats.dirichlet.logpdf
dirichlet_pdf = stats.dirichlet.pdf


PathLike = Union[str, Path]
"""Convenience type for cases where `str` or `Path` are acceptable types."""


class FamilyError(Exception):
    pass


def encode_cluster(cluster: NDArray[bool]) -> str:
    """Format the given cluster as a compact bit-string."""
    cluster_s = cluster.astype(int).astype(str)
    return ''.join(cluster_s)


def decode_cluster(cluster_str: str) -> NDArray[bool]:
    """Read a bit-string and parse it into an area array."""
    return np.array(list(cluster_str)).astype(int).astype(bool)


def format_cluster_columns(clusters: NDArray[bool]) -> str:
    """Format the given array of clusters as tab separated strings."""
    clusters_encoded = map(encode_cluster, clusters)
    return '\t'.join(clusters_encoded)


def parse_cluster_columns(clusters_encoded: str) -> NDArray[bool]:
    """Read tab-separated area encodings into a two-dimensional area array."""
    clusters_decoded = map(decode_cluster, clusters_encoded.split('\t'))
    return np.array(list(clusters_decoded))


def compute_distance(a, b):
    """ This function computes the Euclidean distance between two points a and b

    Args:
        a (list): The x and y coordinates of a point in a metric CRS.
        b (list): The x and y coordinates of a point in a metric CRS.

    Returns:
        float: Distance between a and b
    """

    a = np.asarray(a)
    b = np.asarray(b)
    ab = b-a
    dist = sqrt(ab[0]**2 + ab[1]**2)

    return dist


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


def get_neighbours(cluster, already_in_cluster, adjacency_matrix):
    """This function returns the neighbourhood of a cluster as given in the adjacency_matrix, excluding sites already
    belonging to this or any other cluster.

    Args:
        cluster (np.array): The current cluster (boolean array)
        already_in_cluster (np.array): All sites already assigned to a cluster (boolean array)
        adjacency_matrix (np.array): The adjacency matrix of the sites (boolean)

    Returns:
        np.array: The neighborhood of the cluster (boolean array)
    """

    # Get all neighbors of the current zone, excluding all vertices that are already in a zone

    neighbours = np.logical_and(adjacency_matrix.dot(cluster), ~already_in_cluster)
    return neighbours


def compute_delaunay(locations):
    """Computes the Delaunay triangulation between a set of point locations

    Args:
        locations (np.array): a set of locations
            shape (n_sites, n_spatial_dims = 2)
    Returns:
        (np.array) sparse matrix of Delaunay triangulation
            shape (n_edges, n_edges)
    """
    n = len(locations)

    if n < 4:
        # scipy's Delaunay triangulation fails for <3. Return a fully connected graph:
        return csr_matrix(1-np.eye(n, dtype=int))

    delaunay = spatial.Delaunay(locations, qhull_options="QJ Pp")

    indptr, indices = delaunay.vertex_neighbor_vertices
    data = np.ones_like(indices)

    return csr_matrix((data, indices, indptr), shape=(n, n))


def gabriel_graph_from_delaunay(delaunay, locations):
    delaunay = delaunay.toarray()
    # converting delaunay graph to boolean array denoting whether points are connected
    delaunay = delaunay > 0

    # Delaunay indices and locations
    delaunay_connections = []
    delaunay_locations = []

    for index, connected in np.ndenumerate(delaunay):
        if connected:
            # getting indices of points in area
            i1, i2 = index[0], index[1]
            if [i2, i1] not in delaunay_connections:
                delaunay_connections.append([i1, i2])
                delaunay_locations.append(locations[[*[i1, i2]]])
    delaunay_connections = np.sort(np.asarray(delaunay_connections), axis=1)
    delaunay_locations = np.asarray(delaunay_locations)

    # Find the midpoint on all Delaunay edges
    m = (delaunay_locations[:, 0, :] + delaunay_locations[:, 1, :]) / 2

    # Find the radius sphere between each pair of nodes
    r = np.sqrt(np.sum((delaunay_locations[:, 0, :] - delaunay_locations[:, 1, :]) ** 2, axis=1)) / 2

    # Use the kd-tree function in Scipy's spatial module
    tree = spatial.cKDTree(locations)
    # Find the nearest point for each midpoint
    n = tree.query(x=m, k=1)[0]
    # If nearest point to m is at a distance r, then the edge is a Gabriel edge
    g = n >= r * 0.999  # The factor is to avoid precision errors in the distances

    return delaunay_connections[g]


def n_smallest_distances(a, n, return_idx: bool):
    """ This function finds the n smallest distances in a distance matrix

    >>> n_smallest_distances([
    ... [0, 2, 3, 4],
    ... [2, 0, 5, 6],
    ... [3, 5, 0, 7],
    ... [4, 6, 7, 0]], 3, return_idx=False)
    array([2, 3, 4])

    >>> n_smallest_distances([
    ... [0, 2, 3, 4],
    ... [2, 0, 5, 6],
    ... [3, 5, 0, 7],
    ... [4, 6, 7, 0]], 3, return_idx=True)
    (array([1, 2, 3]), array([0, 0, 0]))

    Args:
        a (np.array): The distane matrix
        n (int): The number of distances to return
        return_idx (bool): return the indices of the points (True) or rather the distances (False)

    Returns:
        (np.array): the n_smallest distances
    or
        (np.array, np.array): the indices between which the distances are smallest
    """
    a_tril = np.tril(a)
    a_nn = a_tril[np.nonzero(a_tril)]
    smallest_n = np.sort(a_nn)[: n]
    a_idx = np.isin(a_tril, smallest_n)

    if return_idx:
        return np.where(a_idx)
    else:
        return smallest_n


def set_experiment_name():
    """Get the current time and use it to name the current experiment
    Returns:
         (str): the name of the current experiment
    """
    now = datetime.datetime.now().__str__().rsplit('.')[0]
    now = now[:-3]
    now = now.replace(':', '-')
    now = now.replace(' ', '_')

    return now


def clusters_autosimilarity(cluster, t):
    """
    This function computes the similarity of consecutive cluster in a chain
    Args:
        cluster (list): cluster
        t (integer): lag between consecutive cluster in the chain

    Returns:
        (float) : mean similarity between cluster in the chain with lag t
    """
    z = np.asarray(cluster)
    z = z[:, 0, :]
    unions = np.maximum(z[t:], z[:-t])
    intersections = np.minimum(z[t:], z[:-t])
    sim_norm = np.sum(intersections, axis=1) / np.sum(unions, axis=1)

    return np.mean(sim_norm)


def range_like(a):
    """Return a list of incrementing integers (range) with same length as `a`."""
    return list(range(len(a)))


# Encoding
def encode_states(features_raw, feature_states):
    # Define shapes
    n_states, n_features = feature_states.shape
    features_bin_shape = features_raw.shape + (n_states,)
    n_sites, _ = features_raw.shape
    assert n_features == _

    # Initialize arrays and counts
    features_bin = np.zeros(features_bin_shape, dtype=int)
    applicable_states = np.zeros((n_features, n_states), dtype=bool)
    state_names = []
    na_number = 0

    # Binary vectors used for encoding
    one_hot = np.eye(n_states)

    for f_idx in range(n_features):
        f_name = feature_states.columns[f_idx]
        f_states = feature_states[f_name]

        # Define applicable states for feature f
        applicable_states[f_idx] = ~f_states.isna()

        # Define external and internal state names
        s_ext = f_states.dropna().to_list()
        s_int = range_like(s_ext)
        state_names.append(s_ext)

        # Map external to internal states for feature f
        ext_to_int = dict(zip(s_ext, s_int))
        f_raw = features_raw[f_name]
        f_enc = f_raw.map(ext_to_int)
        if not (set(f_raw.dropna()).issubset(set(s_ext))):
            print(set(f_raw.dropna()) - set(s_ext))
            print(s_ext)
        assert set(f_raw.dropna()).issubset(set(s_ext))  # All states should map to an encoding

        # Binarize features
        f_applicable = ~f_enc.isna().to_numpy()
        f_enc_applicable = f_enc[f_applicable].astype(int)

        features_bin[f_applicable, f_idx] = one_hot[f_enc_applicable]

        # Count NA
        na_number += np.count_nonzero(f_enc.isna())

    features = {
        'values': features_bin.astype(bool),
        'states': applicable_states,
        'state_names': state_names
    }

    return features, na_number


def normalize_str(s: str) -> str:
    if pd.isna(s):
        return s
    return str.strip(s)


def read_data_csv(csv_path: PathLike) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype=str).applymap(normalize_str)


def read_costs_from_csv(file: str, logger=None):
    """This is a helper function to read the cost matrix from a csv file
        Args:
            file: file location of the csv file
            logger: Logger objects for printing info message.

        Returns:
            pd.DataFrame: cost matrix
        """

    data = pd.read_csv(file, dtype=str, index_col=0)
    if logger:
        logger.info(f"Geographical cost matrix read from {file}.")
    return data


def write_languages_to_csv(features, sites, families, file):
    """This is a helper function to export features as a csv file
    Args:
        features (np.array): features
            shape: (n_sites, n_features, n_categories)
        sites (dict): sites with unique id
        families (np.array): families
            shape: (n_families, n_sites)
        file(str): output csv file
    """
    families = families.transpose(1, 0)

    with open(file, mode='w', encoding='utf-8') as csv_file:
        f_names = list(range(features.shape[1]))
        csv_names = ['f' + str(f) for f in f_names]
        csv_names = ["name", "x", "y", "family"] + csv_names
        writer = csv.writer(csv_file)
        writer.writerow(csv_names)

        for i in sites['id']:
            # name
            name = "site_" + str(i)
            # location
            x, y = sites['locations'][i]
            # features
            f = np.where(features[i] == 1)[1].tolist()
            # family
            fam = np.where(families[i] == 1)[0].tolist()
            if not fam:
                fam = ""
            else:
                fam = "family_" + str(fam[0])
            writer.writerow([name] + [x] + [y] + [fam] + f)


def write_feature_occurrence_to_csv(occurrence, categories, file):
    """This is a helper function to export the occurrence of features in families or globally to a csv
    Args:
        occurrence: the occurrence of each feature, either as a relative frequency or counts
        categories: the possible categories per feature
        file(str): output csv file
    """

    with open(file, mode='w', encoding='utf-8') as csv_file:
        features = list(range(occurrence.shape[0]))
        feature_names = ['f' + str(f) for f in features]
        cats = list(range(occurrence.shape[1]))
        cat_names = ['cat' + str(c) for c in cats]
        csv_names = ["feature"] + cat_names
        writer = csv.writer(csv_file)
        writer.writerow(csv_names)

        for f in range(len(feature_names)):
            # feature name
            f_name = feature_names[f]
            # frequencies
            p = occurrence[f, :].tolist()
            idx = categories[f]
            for i in range(len(p)):
                if i not in idx:
                    p[i] = ""
            writer.writerow([f_name] + p)


def read_feature_occurrence_from_csv(file, feature_states_file):
    """This is a helper function to import the occurrence of features in families (or globally) from a csv
        Args:
            file(str): path to the csv file containing feature-state counts
            feature_states_file (str): path to csv file containing features and states

        Returns:
            np.array :
            The occurrence of each feature, either as relative frequencies or counts, together with feature
            and category names
    """

    # Load data and feature states
    counts_raw = pd.read_csv(file, index_col='feature')
    feature_states = pd.read_csv(feature_states_file, dtype=str)
    n_states, n_features = feature_states.shape

    # Check that features match
    assert set(counts_raw.index) == set(feature_states.columns)

    # Replace NAs by 0.
    counts_raw[counts_raw.isna()] = 0.

    # Collect feature and state names
    feature_names = {'external': feature_states.columns.to_list(),
                     'internal': list(range(n_features))}
    state_names = {'external': [[] for _ in range(n_features)],
                   'internal': [[] for _ in range(n_features)]}

    # Align state columns with feature_states file
    counts = np.zeros((n_features, n_states))
    for f_idx in range(n_features):
        f_name = feature_states.columns[f_idx]         # Feature order is given by ´feature_states_file´
        for s_idx in range(n_states):
            s_name = feature_states[f_name][s_idx]     # States order is given by ´feature_states_file´
            if pd.isnull(s_name):                      # ...same for applicable states per feature
                continue

            counts[f_idx, s_idx] = counts_raw.loc[f_name, s_name]

            state_names['external'][f_idx].append(s_name)
            state_names['internal'][f_idx].append(s_idx)

    # # Sanity check
    # Are the data count data?
    if not all(float(y).is_integer() for y in np.nditer(counts)):
        out = f"The data in {file} must be count data."
        raise ValueError(out)

    return counts.astype(int), feature_names, state_names


def inheritance_counts_to_dirichlet(counts, states, outdated_features=None, dirichlet=None):
    """This is a helper function transform the family counts to alpha values that
    are then used to define a dirichlet distribution

    Args:
        counts(np.array): the family counts
            shape: (n_families, n_features, n_states)
        states(list): states per feature in each of the families
    Returns:
        list: the dirichlet distributions, neatly stored in a dict
    """
    n_fam, n_feat, n_cat = counts.shape
    if dirichlet is None:
        dirichlet = [None] * n_fam

    for fam in range(n_fam):
        dirichlet[fam] = counts_to_dirichlet(counts[fam], states,
                                             outdated_features=outdated_features,
                                             dirichlet=dirichlet[fam])
    return dirichlet


def scale_counts(counts, scale_to, prior_inheritance=False):
    """Scales the counts for parametrizing the prior on universal probabilities (or inheritance in a family)

        Args:
            counts (np.array): the counts of categorical data.
                shape: (n_features, n_states) or (n_families, n_features, n_states)
            scale_to (float): the counts are scaled to this value
            prior_inheritance (bool): are these inheritance counts?
        Returns:
            np.array: the rescaled counts
                shape: same as counts.shape
    """
    counts_sum = np.sum(counts, axis=-1)
    counts_sum = np.where(counts_sum == 0, EPS, counts_sum)
    scale_factor = scale_to / counts_sum

    scale_factor = np.where(scale_factor < 1, scale_factor, 1)
    return counts * scale_factor[..., None]


def counts_to_dirichlet(
        counts: Sequence[Sequence[int]],
        states: Sequence[int],
        prior='uniform',
        outdated_features=None,
        dirichlet=None
):
    """This is a helper function to transform counts of categorical data
    to parameters of a dirichlet distribution.

    Args:
        counts (np.array): the counts of categorical data.
            shape: (n_features, n_states)
        states (np.array): applicable states/categories per feature
            shape: (n_features)
        prior (str): Use one of the following uninformative priors:
            'uniform': A uniform prior probability over the probability simplex Dir(1,...,1)
            'jeffrey': The Jeffrey's prior Dir(0.5,...,0.5)
            'naught': A natural exponential family prior Dir(0,...,0).
        outdated_features (IndexSet): Indices of the features where the counts changed
                                  (i.e. they need to be updated).
    Returns:
        list: a dirichlet distribution derived from pseudocounts
    """
    n_features = len(counts)

    prior_map = {'uniform': 1, 'jeffrey': 0.5, 'naught': 0}

    if outdated_features is None or outdated_features.all:
        outdated_features = range(n_features)
        dirichlet = [None] * n_features
    else:
        assert dirichlet is not None

    for feat in outdated_features:
        cat = states[feat]
        # Add 1 to alpha values (1,1,...1 is a uniform prior)
        pseudocounts = counts[feat, cat] + prior_map[prior]
        dirichlet[feat] = pseudocounts

    return dirichlet


def touch(fname):
    """Create an empty file at path `fname`."""
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()


def mkpath(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isdir(path):
        touch(path)


def add_edge(edges, edge_nodes, coords, i, j):
    """Add an edge between the i-th and j-th points, if not in edges already.
    Args:
        edges (set): set of edges
        edge_nodes(list): coordinates of all nodes in all edges
        coords(float, float): point coordinates of sites
        i (int): i-th point
        j (int): j-th point
        """
    if (i, j) in edges or (j, i) in edges:
        # already added
        return

    edges.add((i, j))
    edge_nodes.append(coords[[i, j]])


def collect_gt_for_writing(samples, data, config):

    gt = dict()
    gt_col_names = ['posterior', 'likelihood', 'prior']
    gt['posterior'] = samples['true_prior'] + samples['true_ll']
    gt['likelihood'] = samples['true_ll']
    gt['prior'] = samples['true_prior']

    # weights
    for f in range(len(data.feature_names['external'])):
        # universal pressure
        w_universal_name = 'w_universal_' + str(data.feature_names['external'][f])
        if w_universal_name not in gt_col_names:
            gt_col_names += [w_universal_name]
        gt[w_universal_name] = samples['true_weights'][f][0]

        # contact
        w_contact_name = 'w_contact_' + str(data.feature_names['external'][f])
        if w_contact_name not in gt_col_names:
            gt_col_names += [w_contact_name]
        gt[w_contact_name] = samples['true_weights'][f][1]

        # inheritance
        if config['model']['inheritance']:
            w_inheritance_name = 'w_inheritance_' + str(data.feature_names['external'][f])
            if w_inheritance_name not in gt_col_names:
                gt_col_names += [w_inheritance_name]
            gt[w_inheritance_name] = samples['true_weights'][f][2]

    # alpha
    for f in range(len(data.feature_names['external'])):
        for st in range(len(data.state_names['external'][f])):
            feature_name = 'alpha_' + str(data.feature_names['external'][f])\
                           + '_' + str(data.state_names['external'][f][st])
            if feature_name not in gt_col_names:
                gt_col_names += [feature_name]
            gt[feature_name] = samples['true_p_global'][0][f][st]

    # gamma
    for a in range(len(data.clusters)):
        for f in range(len(data.feature_names['external'])):
            for st in range(len(data.state_names['external'][f])):
                feature_name = 'gamma_' + 'a' + str(a + 1) \
                               + '_' + str(data.feature_names['external'][f]) + '_' \
                               + str(data.state_names['external'][f][st])
                if feature_name not in gt_col_names:
                    gt_col_names += [feature_name]
                gt[feature_name] = samples['true_p_zones'][a][f][st]

    # beta
    if config['simulation']['inheritance']:
        for fam in range(len(data.family_names['external'])):
            for f in range(len(data.feature_names['external'])):
                for st in range(len(data.state_names['external'][f])):
                    feature_name = 'beta_' + str(data.family_names['external'][fam]) \
                                   + '_' + str(data.feature_names['external'][f]) \
                                   + '_' + str(data.state_names['external'][f][st])
                    if feature_name not in gt_col_names:
                        gt_col_names += [feature_name]
                    gt[feature_name] = samples['true_p_families'][fam][f][st]
    # Single areas
    if 'true_lh_single_cluster' in samples.keys():
        for a in range(len(data.clusters)):
            lh_name = 'lh_a' + str(a + 1)
            prior_name = 'prior_a' + str(a + 1)
            posterior_name = 'post_a' + str(a + 1)

            gt_col_names += [lh_name]
            gt[lh_name] = samples['true_lh_single_cluster'][a]

            gt_col_names += [prior_name]
            gt[prior_name] = samples['true_prior_single_cluster'][a]
            gt_col_names += [posterior_name]
            gt[posterior_name] = samples['true_posterior_single_cluster'][a]

    return gt, gt_col_names


def collect_gt_clusters_for_writing(samples):
    return format_cluster_columns(samples['true_clusters'])


def collect_clusters_for_writing(s, samples):
    cluster_row = format_cluster_columns(samples['sample_clusters'][s])
    return cluster_row


def collect_row_for_writing(s, samples, data, config, steps_per_sample):
    row = dict()
    column_names = ['Sample', 'posterior', 'likelihood', 'prior']
    row['Sample'] = int(s * steps_per_sample)
    row['posterior'] = samples['sample_prior'][s] + samples['sample_likelihood'][s]
    row['likelihood'] = samples['sample_likelihood'][s]
    row['prior'] = samples['sample_prior'][s]

    # Cluster size
    for i, cluster in enumerate(samples['sample_clusters'][s]):
        col_name = f'size_a{i}'
        column_names.append(col_name)
        row[col_name] = np.count_nonzero(cluster)

    # weights
    for f in range(len(data.features['names'])):

        # Areal effect
        w_cluster_effect = f"w_cluster_effect_{str(data.features['names'][f])}"
        column_names += [w_cluster_effect]
        # index of cluster_effect = 0
        # todo: use source_index instead of remembering the order
        row[w_cluster_effect] = samples['sample_weights'][s][f][0]

        # Confounding effects
        for i, k in enumerate(data.confounders):
            w_confounder = f"w_{k}_{str(data.features['names'][f])}"
            column_names += [w_confounder]
            # todo: use source_index instead of remembering the order
            # index of confounding effect starts with 1
            row[w_confounder] = samples['sample_weights'][s][f][i+1]

    # Cluster effect
    for a in range(config['model']['clusters']):
        for f in range(len(data.features['names'])):
            for st in data.features['state_names'][f]:
                feature_name = f"cluster_a{str(a + 1)}_{str(data.features['names'][f])}_{str(st)}"
                idx = data.features['state_names'][f].index(st)
                column_names += [feature_name]
                row[feature_name] = samples['sample_cluster_effect'][s][a][f][idx]

    # Confounding effects
    for k, v in data.confounders.items():
        for group in range(len(data.confounders[k])):
            for f in range(len(data.features['names'])):
                for st in data.features['state_names'][f]:
                    feature_name = f"{k}_{v['names'][group]}_{str(data.features['names'][f])}_{str(st)}"
                    idx = data.features['state_names'][f].index(st)
                    column_names += [feature_name]
                    row[feature_name] = samples['sample_confounding_effects'][k][s][group][f][idx]

    # todo: reactivate
    # Recall and precision
    # if data.is_simulated:
    #     sample_z = np.any(samples['sample_zones'][s], axis=0)
    #     true_z = np.any(samples['true_zones'], axis=0)
    #     n_true = np.sum(true_z)
    #     intersections = np.minimum(sample_z, true_z)
    #
    #     total_recall = np.sum(intersections, axis=0) / n_true
    #     precision = np.sum(intersections, axis=0) / np.sum(sample_z, axis=0)
    #
    #     column_names += ['recall']
    #     row['recall'] = total_recall
    #
    #     column_names += ['precision']
    #     row['precision'] = precision

    # Single areas
    if 'sample_lh_single_cluster' in samples.keys():
        for a in range(config['model']['clusters']):
            lh_name = 'lh_a' + str(a + 1)
            prior_name = 'prior_a' + str(a + 1)
            posterior_name = 'post_a' + str(a + 1)

            column_names += [lh_name]
            row[lh_name] = samples['sample_lh_single_cluster'][s][a]

            column_names += [prior_name]
            row[prior_name] = samples['sample_prior_single_cluster'][s][a]

            column_names += [posterior_name]
            row[posterior_name] = samples['sample_posterior_single_cluster'][s][a]

    return row, column_names


def samples2file(samples, data, config, paths):
    """
    Writes the MCMC to two text files, one for MCMC parameters and one for areas.

    Args:
        samples (dict): samples
        data (Data): object of class data (features, priors, ...)
        config(dict): config information
        paths(dict): file path for stats and clusters
    """
    print("Writing results to file ...")
    # Write ground truth to file (for simulated data only)
    # todo: reactivate
    # if data.is_simulated:
    #
    #     try:
    #         with open(paths['gt'], 'w', newline='') as file:
    #             gt, gt_col_names = collect_gt_for_writing(samples=samples, data=data, config=config)
    #             writer = csv.DictWriter(file, fieldnames=gt_col_names, delimiter='\t')
    #             writer.writeheader()
    #             writer.writerow(gt)
    #
    #     except IOError:
    #         print("I/O error")
    #
    #     try:
    #         with open(paths['gt_areas'], 'w', newline='') as file:
    #             gt_areas = collect_gt_areas_for_writing(samples=samples)
    #             file.write(gt_areas)
    #             file.close()
    #
    #     except IOError:
    #         print("I/O error")

    # Results
    steps_per_sample = float(config['mcmc']['steps'] / config['mcmc']['samples'])

    # Statistics
    try:
        writer = None
        with open(paths['parameters'], 'w', newline='') as file:

            for s in range(len(samples['sample_clusters'])):
                row, column_names = collect_row_for_writing(s=s, samples=samples, data=data, config=config,
                                                            steps_per_sample=steps_per_sample)
                if s == 0:
                    writer = csv.DictWriter(file, fieldnames=column_names, delimiter='\t')
                    writer.writeheader()
                if writer:
                    writer.writerow(row)
        file.close()
    except IOError:
        print("I/O error")

    # Areas
    try:
        with open(paths['clusters'], 'w', newline='') as file:
            for s in range(len(samples['sample_clusters'])):
                clusters = collect_clusters_for_writing(s, samples)
                file.write(clusters + '\n')
            file.close()

    except IOError:
        print("I/O error")


def linear_rescale(value, old_min, old_max, new_min, new_max):
    """
    Function to linear rescale a number to a new range

    Args:
         value (float): number to rescale
         old_min (float): old minimum of value range
         old_max (float): old maximum of value range
         new_min (float): new minimum of value range
         new_max (float): new maximum of vlaue range
    """

    return (new_max - new_min) / (old_max - old_min) * (value - old_max) + old_max


def round_single_int(n, mode='up', position=2, offset=1):
    """
    Function to round an integer for the calculation of axes limits.
    For example (position=2, offset=0):
    up: 113 -> 120, 3456 -> 3500
    down: 113 -> 110, 3456 -> 3400

    Args:
         n (int): integer number to round
         mode (str): round 'up' or 'down'
         position (int):
         offset (int): adding offset to rounded number

    == Usage ===
    >>> round_single_int(113, 'up', 2, 0)
    120
    >>> round_single_int(3456, 'up', 2, 0)
    3500
    >>> round_single_int(113, 'down', 2, 0)
    110
    >>> round_single_int(3456, 'down', 2, 0)
    3400
    >>> round_single_int(3456, 'down', 3, 0)
    3450
    >>> round_single_int(3456, 'down', 2, 1)
    3300
    """

    # convert to int if necessary and get number of digits
    n = int(n) if isinstance(n, float) else n
    n_digits = len(str(n)) if n > 0 else len(str(n)) - 1

    # check for validity of input parameters
    if mode != 'up' and mode != 'down':
        raise Exception(f'Unknown mode: "{mode}". Use either "up" or "down".')
    if position > n_digits:
        raise Exception(f'Position {position} is not valid for a number with only {n_digits} digits.')

    # special rules for 1 and 2 digit numbers
    if n_digits == 1:
        n_rounded = n - offset if mode == 'down' else n + offset

    elif n_digits == 2:
        if position == 1:
            base = n // 10 * 10
            n_rounded = base - offset * 10 if mode == 'down' else base + ((offset + 1) * 10)
        else:
            assert (position == 2)
            n_rounded = n - offset if mode == 'down' else n + offset

    else:
        if not position == n_digits:
            factor = 10 ** (n_digits - position)
            base = (n // factor) * factor
            n_rounded = base - offset * factor if mode == 'down' else base + ((offset + 1) * factor)
        else:
            n_rounded = n - offset if mode == 'down' else n + offset

    return n_rounded


def round_multiple_ints(ups, downs, position=2, offset=1):

    ups = [int(n) for n in ups]
    downs = [int(n) for n in downs]

    # find value with fewest digits
    fewest_digits = len(str(np.min(ups + downs)))

    ups_rounded = []
    for n in ups:
        length = len(str(n))
        n_rounded = round_single_int(n, 'up', position + length - fewest_digits, offset)
        ups_rounded.append(n_rounded)

    downs_rounded = []
    for n in downs:
        length = len(str(n))
        n_rounded = round_single_int(n, 'down', position + length - fewest_digits, offset)
        downs_rounded.append(n_rounded)

    return ups_rounded, downs_rounded


def round_int(n, mode='up', offset=0):
    """
    Function to round an integer for the calculation of axes limits.
    For example:
    up: 113 -> 120, 3456 -> 3500
    down: 113 -> 110, 3456 -> 3450

    Args:
         n (int): integer number to round
         mode (str): round 'up' or 'down'
         offset (int): adding offset to rounded number
    """

    n = int(n) if isinstance(n, float) else n
    convertor = 10 ** (len(str(offset)) - 1)

    if n > offset:  # number is larger than offset (must be positive)
        if mode == 'up':
            n_rounded = ceil(n / convertor) * convertor
            n_rounded += offset
        if mode == 'down':
            n_rounded = floor(n / convertor) * convertor
            n_rounded -= offset
        else:
            raise Exception('unkown mode')
    else:  # number is smaller than offset (can be negative)
        if n >= 0:
            n_rounded = offset + convertor if mode == 'up' else -offset
        else:
            # for negative numbers we use round_int with inversed mode and the positive number
            inverse_mode = 'up' if mode == 'down' else 'down'
            n_rounded = round_int(abs(n), inverse_mode, offset)
            n_rounded = - n_rounded

    return n_rounded


def colorline(ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3):
    """
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    from: https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    def make_segments(x, y):
        """
        Create list of line segments from x and y coordinates, in the correct format for LineCollection:
        an array of the form   numlines x (points per line) x 2 (x and y) array
        """

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        return segments

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=1, zorder=1)

    # ax = plt.gca()
    ax.add_collection(lc)

    return lc


def normalize(x, axis=-1):
    """Normalize ´x´ s.t. the last axis sums up to 1.

    Args:
        x (np.array): Array to be normalized.
        axis (int): The axis to be normalized (will sum up to 1).

    Returns:
         np.array: x with normalized s.t. the last axis sums to 1.

    == Usage ===
    >>> normalize(np.ones((2, 4)))
    array([[0.25, 0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25, 0.25]])
    >>> normalize(np.ones((2, 4)), axis=0)
    array([[0.5, 0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5, 0.5]])
    """
    return x / np.sum(x, axis=axis, keepdims=True)


def mle_weights(samples):
    """Compute the maximum likelihood estimate for categorical samples.

    Args:
        samples (np.array):
    Returns:
        np.array: the MLE for the probability vector in the categorical distribution.
    """
    counts = np.sum(samples, axis=0)
    return normalize(counts)


def assign_na(features, n_na):
    """ Randomly assign NAs to features. Makes the simulated data more realistic. A feature is NA if for one
    site a feature is 0 in all categories
    Args:
        features(np.ndarray): binary feature array
            shape: (sites, features, categories)
        n_na: number of NAs added
    returns: features(np.ndarray): binary feature array, with shape = (sites, features, categories)
    """

    features = features.astype(float)
    # Choose a random site and feature and set to None
    for _ in range(n_na):

        na_site = np.random.choice(a=features.shape[0], size=1)
        na_feature = np.random.choice(a=features.shape[1], size=1)
        features[na_site, na_feature, :] = 0

    return features


def assess_correlation_probabilities(p_universal, p_contact, p_inheritance, corr_th, include_universal=False):
    """Asses the correlation of probabilities in simulated data

        Args:
            p_universal (np.array): universal state probabilities
                shape (n_features, n_states)
            p_contact (np.array): state probabilities in areas
                shape: (n_areas, n_features, n_states)
            p_inheritance (np.array): state probabilities in families
                shape: (n_families, n_features, n_states)
            corr_th (float): correlation threshold
            include_universal (bool): Should p_universal also be checked for independence?

        """
    if include_universal:
        if p_inheritance is not None:
            samples = np.vstack((p_universal[np.newaxis, :, :], p_contact, p_inheritance))
        else:
            samples = np.vstack((p_universal[np.newaxis, :, :], p_contact))
    else:
        if p_inheritance is not None:
            samples = np.vstack((p_contact, p_inheritance))
        else:
            samples = p_contact

    n_samples = samples.shape[0]
    n_features = samples.shape[1]

    comb = list(combinations(list(range(n_samples)), 2))
    correlated_probability_vectors = 0

    for f in range(n_features):
        for c in comb:
            p0 = samples[c[0], f]
            p1 = samples[c[1], f]
            p_same_state = np.dot(p0, p1)
            if p_same_state > corr_th:
                correlated_probability_vectors += 1
    return correlated_probability_vectors


def get_max_size_list(start, end, n_total, k_groups):
    """Returns a list of maximum cluster sizes between a start and end size
    Entries of the list are later used to vary max_size in different chains during warmup

    Args:
        start(int): start size
        end(int): end size
        n_total(int): entries in the final list, i.e.number of chains
        k_groups(int): number of groups with the same max_size

    Returns:
        (list) list of different max_sizes
    """
    n_per_group = ceil(n_total/k_groups)
    max_sizes = np.linspace(start=start, stop=end, num=k_groups, endpoint=False, dtype=int)
    max_size_list = list(np.repeat(max_sizes, n_per_group))

    return max_size_list[0:n_total]


def log_binom(n, k):
    """Compute the logarithm of (n choose k), i.e. the binomial coefficient of `n` and `k`.

    Args:
        n (int or np.array): Population size.
        k (int or np.array): Sample size.
    Returns:
        double: log(n choose k)

    == Usage ===
    >>> log_binom(10, np.arange(3))
    array([0.        , 2.30258509, 3.80666249])
    >>> log_binom(np.arange(1, 4), 1)
    array([0.        , 0.69314718, 1.09861229])
    """
    return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def log_multinom(n, ks):
    """Compute the logarithm of (n choose k1,k2,...), i.e. the multinomial coefficient of
    `n` and the integers in the list `ks`. The sum of the sample sizes (the numbers in
     `ks`) may not exceed the population size (`n`).

    Args:
        n (int): Population size.
        ks (list[int] or np.array): Sample sizes

    Returns:
        double: log(n choose k1,k2,...)

    """
    ks = np.asarray(ks)
    # assert np.all(ks >= 0)
    # assert np.sum(ks) <= n

    # Simple special case
    if np.sum(ks) == 0:
        return 0.

    # Filter out 0-samples
    ks = ks[ks > 0]

    log_i = np.log(1 + np.arange(n))
    log_i_cumsum = np.cumsum(log_i)

    # Count all permutations of the total population
    m = np.sum(log_i)

    # Subtract all permutation within the samples (with sample sizes specified in `ks`).
    m -= np.sum(log_i_cumsum[ks-1])

    # If there are is a remainder in the population, that was not assigned to any of the
    # samples, subtract all permutations of the remainder population.
    rest = n - np.sum(ks)
    # assert rest >= 0
    if rest > 0:
        m -= log_i_cumsum[rest-1]

    # assert m >= 0, m
    return m


def decompose_config_path(config_path: PathLike) -> (Path, Path):
    """Extract the base directory of `config_path` and return the path itself as an
    absolute path."""
    abs_config_path = Path(config_path).absolute()
    base_directory = abs_config_path.parent
    return base_directory, abs_config_path


def fix_relative_path(path: PathLike, base_directory: PathLike) -> Path:
    """Make sure that the provided path is either absolute or relative to the config file directory.

    Args:
        path: The original path (absolute or relative).
        base_directory: The base directory

    Returns:
        The fixed path.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    else:
        return base_directory / path


def timeit(units='s'):
    SECONDS_PER_UNIT = {
        'h': 3600.,
        'm': 60.,
        's': 1.,
        'ms': 1E-3,
        'µs': 1E-6,
        'ns': 1E-9
    }
    unit_scaler = SECONDS_PER_UNIT[units]

    def timeit_decorator(func):

        def timed_func(*args, **kwargs):


            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            passed = (end - start) / unit_scaler

            print(f'Runtime {func.__name__}: {passed:.2f}{units}')

            return result

        return timed_func

    return timeit_decorator

@lru_cache(maxsize=128)
def get_permutations(n: int) -> list[tuple[int]]:
    return list(permutations(range(n)))


def get_best_permutation(
        areas: NDArray[bool],  # shape = (n_areas, n_sites)
        prev_area_sum: NDArray[int],  # shape = (n_areas, n_sites)
) -> tuple[int]:
    """Return a permutation of areas that would align the areas in the new sample with previous ones."""
    def clustering_agreement(p):
        """In how many sites does permutation `p` previous samples?"""
        return np.sum(prev_area_sum * areas[p, :])

    all_permutations = get_permutations(areas.shape[0])
    return max(all_permutations, key=clustering_agreement)


if scipy.__version__ >= '1.8.0':
    log_expit = scipy.special.log_expit
else:
    def log_expit(*args, **kwargs):
        return np.log(expit(*args, **kwargs))


def set_defaults(cfg: dict, default_cfg: dict):
    """Iterate through a recursive config dictionary and set all fields that are not
    present in cfg to the default values from default_cfg.

    == Usage ===
    >>> set_defaults(cfg={0:0, 1:{1:0}, 2:{2:1}},
    ...              default_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 0}, 2: {2: 1, 1: 1}}
    >>> set_defaults(cfg={0:0, 1:1, 2:2},
    ...              default_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: 1, 2: 2}
    """
    for key in default_cfg:
        if key not in cfg:
            # Field ´key´ is not defined in cfg -> use default
            cfg[key] = default_cfg[key]

        else:
            # Field ´key´ is defined in cfg
            # -> update recursively if the field is a dictionary
            if isinstance(default_cfg[key], dict) and isinstance(cfg[key], dict):
                set_defaults(cfg[key], default_cfg[key])

    return cfg


def update_recursive(cfg: dict, new_cfg: dict):
    """Iterate through a recursive config dictionary and update cfg in all fields that are specified in new_cfg.

    == Usage ===
    >>> update_recursive(cfg={0:0, 1:{1:0}, 2:{2:1}},
    ...                  new_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 1}, 2: {2: 2, 1: 1}}
    >>> update_recursive(cfg={0:0, 1:1, 2:2},
    ...                  new_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 1}, 2: {1: 1, 2: 2}}
    """
    for key in new_cfg:
        if (key in cfg) and isinstance(new_cfg[key], dict) and isinstance(cfg[key], dict):
            # Both dictionaries have another layer -> update recursively
            update_recursive(cfg[key], new_cfg[key])
        else:
            cfg[key] = new_cfg[key]

    return cfg


def iter_items_recursive(cfg: dict, loc=tuple()):
    """Recursively iterate through all key-value pairs in ´cfg´ and sub-dictionaries.

    Args:
        cfg (dict): Config dictionary, potentially containing sub-dictionaries.
        loc (tuple): Specifies the sequene of keys that lead to the current sub-dictionary.
    Yields:
        tuple: key-value pairs of the bottom level dictionaries

    == Usage ===
    >>> list(iter_items_recursive({0: 0, 1: {1: 0}, 2: {2: 1, 1: 1}}))
    [(0, 0, ()), (1, 0, (1,)), (2, 1, (2,)), (1, 1, (2,))]
    """
    for key, value in cfg.items():
        if isinstance(value, dict):
            yield from iter_items_recursive(value, loc + (key, ))
        else:
            yield key, value, loc


def categorical_log_probability(x: NDArray[bool], p: NDArray[float]) -> NDArray[float]:
    """Compute the log-probability of observations `x` under categorical distribution `p`.

    Args:
        x: observations in one-hot encoding. Each column (on last axis) contains exactly one 1.
            shape: (*distr_shape, n_categories)
        p: probability of each state in each dimension of the distribution. Last axis is
                normalised to one.
            shape: (*distr_shape, n_categories)

    Returns:
        The log-probability for each observation elementwise.
            shape: distr_shape

    """
    return np.log(np.sum(x*p, axis=-1))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
