#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import datetime
import time
import csv
import os
from math import sqrt, floor, ceil

import typing as t

import numpy as np
import pandas as pd
import scipy.spatial as spatial
from scipy.special import betaln
import scipy.stats as stats
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.stats import chi2_contingency

from itertools import combinations
from fastcluster import linkage

EPS = np.finfo(float).eps

FAST_DIRICHLET = True
if FAST_DIRICHLET:
    def dirichlet_pdf(x, alpha): return np.exp(stats.dirichlet._logpdf(x, alpha))
    dirichlet_logpdf = stats.dirichlet._logpdf

    # ## Since the PDF is evaluated on the same samples again we could use
    # ## an LRU cache for further speed-up (not properly tested yet):
    #
    # from lru import LRU
    # from scipy.special import gammaln, xlogy
    #
    # cache = LRU(10000)
    #
    # def dirichlet_logpdf(x, alpha):
    #     key = alpha.tobytes()
    #
    #     if hash in cache:
    #         lnB = cache[key]
    #
    #     else:
    #         lnB = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    #         cache[key] = lnB
    #
    #     return -lnB  np.sum((xlogy(alpha - 1, x.T)).T, 0)
    #
    # # dirichlet_logpdf = stats.dirichlet._logpdf
    #
    # def dirichlet_pdf(x, alpha):
    #     return np.exp(dirichlet_logpdf(x, alpha))

else:
    dirichlet_pdf = stats.dirichlet.pdf
    dirichlet_logpdf = stats.dirichlet.logpdf


class FamilyError(Exception):
    pass


def encode_area(area):
    """Format the given area as a compact bit-string."""
    area_s = area.astype(int).astype(str)
    return ''.join(area_s)


def decode_area(area_str):
    """Read a bit-string and parse it into an area array."""
    return np.array(list(area_str)).astype(int).astype(bool)


def format_area_columns(areas):
    """Format the given array of areas as tab separated strings."""
    areas_encoded = map(encode_area, areas)
    return '\t'.join(areas_encoded)


def parse_area_columns(areas_encoded):
    """Read tab-separated area encodings into a two-dimensional area array."""
    areas_decoded = map(decode_area, areas_encoded.split('\t'))
    return np.array(list(areas_decoded))


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
    """Computes the Delaunay triangulation between a set of point locations

    Args:
        locations (np.array): a set of locations
            shape (n_sites, n_spatial_dims = 2)
    Returns:
        (np.array) sparse matrix of Delaunay triangulation
            shape (n_edges, n_edges)
    """

    n = len(locations)
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
    """Gets the current time and uses it to name the current experiment
    Returns:
         (str): the name of the current experiment
    """
    now = datetime.datetime.now().__str__().rsplit('.')[0]
    now = now[:-3]
    now = now.replace(':', '-')
    now = now.replace(' ', '_')

    return now


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
    state_names = {'external': [], 'internal': []}
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
        state_names['external'].append(s_ext)
        state_names['internal'].append(s_int)

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

    return features_bin.astype(bool), state_names, applicable_states, na_number


def normalize_str(s):
    if pd.isna(s):
        return s
    return str.strip(s)


def read_features_from_csv(file, feature_states_file):
    """This is a helper function to import data (sites, features, family membership,...) from a csv file
    Args:
        file (str): file location of the csv file
        feature_states_file (str): file location of the meta data for the features
    Returns:
        (dict, dict, np.array, dict, dict, np.array, np.array, dict, str) :
        The language date including sites, site names, all features, feature names and state names per feature,
        as well as family membership and family names and log information
    """
    data = pd.read_csv(file, dtype=str)
    data = data.applymap(normalize_str)

    try:
        x = data.pop('x')
        y = data.pop('y')
        l_id = data.pop('id')
        name = data.pop('name')
        family = data.pop('family')
    except KeyError:
        raise KeyError('The csv must contain columns "x", "y", "id","name", "family"')

    # Load the valid features-states
    feature_states = pd.read_csv(feature_states_file, dtype=str)
    feature_states = feature_states.applymap(normalize_str)
    feature_names_ext = feature_states.columns.to_numpy()

    # Make sure the same features are specified in the data file and in the feature_states file
    assert set(feature_states.columns) == set(data.columns)

    # sites
    n_sites, n_features = data.shape
    locations = np.zeros((n_sites, 2))
    site_id = []

    for i in range(n_sites):
        # Define location tuples
        locations[i, 0] = float(x[i])
        locations[i, 1] = float(y[i])

        # The order in the list maps name to id and id to name
        # name could be any unique identifier, id is an integer from 0 to len(name)
        site_id.append(i)

    sites = {'locations': locations,
             'id': site_id,
             'cz': None,
             'names': name}
    site_names = {'external': l_id,
                  'internal': list(range(n_sites))}

    # features
    features, state_names, applicable_states, na_number = encode_states(data, feature_states)
    feature_names = {'external': feature_names_ext,
                     'internal': list(range(n_features))}

    # family
    family_names_ordered = np.unique(family.dropna()).tolist()
    n_families = len(family_names_ordered)

    families = np.zeros((n_families, n_sites), dtype=int)
    for fam in range(n_families):
        families[fam, np.where(family == family_names_ordered[fam])] = 1

    family_names = {'external': family_names_ordered,
                    'internal': list(range(n_families))}

    log = f"{n_sites} sites with {n_features} features read from {file}. {na_number} NA value(s) found."

    return sites, site_names, features, feature_names, state_names, applicable_states, families, family_names, log


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


def write_feature_occurrence_to_csv(occurr, categories, file):
    """This is a helper function to export the occurrence of features in families or globally to a csv
    Args:
        occurr: the occurrence of each feature, either as a relative frequency or counts
        categories: the possible categories per feature
        file(str): output csv file
    """

    with open(file, mode='w', encoding='utf-8') as csv_file:
        features = list(range(occurr.shape[0]))
        feature_names = ['f' + str(f) for f in features]
        cats = list(range(occurr.shape[1]))
        cat_names = ['cat' + str(c) for c in cats]
        csv_names = ["feature"] + cat_names
        writer = csv.writer(csv_file)
        writer.writerow(csv_names)

        for f in range(len(feature_names)):
            # feature name
            f_name = feature_names[f]
            # frequencies
            p = occurr[f, :].tolist()
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


def counts_to_dirichlet(counts: t.Sequence[t.Sequence[int]], states: t.Sequence[int], prior='uniform', outdated_features=None, dirichlet=None):
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
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()


def mkpath(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isdir(path):
        touch(path)


def add_edge(edges, edge_nodes, coords, i, j):
    """
    Add an edge between the i-th and j-th points, if not in edges already
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
        if config['model']['INHERITANCE']:
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
    for a in range(len(data.areas)):
        for f in range(len(data.feature_names['external'])):
            for st in range(len(data.state_names['external'][f])):
                feature_name = 'gamma_' + 'a' + str(a + 1) \
                               + '_' + str(data.feature_names['external'][f]) + '_' \
                               + str(data.state_names['external'][f][st])
                if feature_name not in gt_col_names:
                    gt_col_names += [feature_name]
                gt[feature_name] = samples['true_p_zones'][a][f][st]

    # beta
    if config['simulation']['INHERITANCE']:
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
    if 'true_lh_single_zones' in samples.keys():
        for a in range(len(data.areas)):
            lh_name = 'lh_a' + str(a + 1)
            prior_name = 'prior_a' + str(a + 1)
            posterior_name = 'post_a' + str(a + 1)

            gt_col_names += [lh_name]
            gt[lh_name] = samples['true_lh_single_zones'][a]

            gt_col_names += [prior_name]
            gt[prior_name] = samples['true_prior_single_zones'][a]
            gt_col_names += [posterior_name]
            gt[posterior_name] = samples['true_posterior_single_zones'][a]

    return gt, gt_col_names


def collect_gt_areas_for_writing(samples):
    return format_area_columns(samples['true_zones'])


def collect_areas_for_writing(s, samples):
    area_row = format_area_columns(samples['sample_zones'][s])
    return area_row


def collect_row_for_writing(s, samples, data, config, steps_per_sample):
    row = dict()
    column_names = ['Sample', 'posterior', 'likelihood', 'prior']
    row['Sample'] = int(s * steps_per_sample)
    row['posterior'] = samples['sample_prior'][s] + samples['sample_likelihood'][s]
    row['likelihood'] = samples['sample_likelihood'][s]
    row['prior'] = samples['sample_prior'][s]

    # Area sizes
    for i, area in enumerate(samples['sample_zones'][s]):
        col_name = f'size_a{i}'
        column_names.append(col_name)
        row[col_name] = np.count_nonzero(area)

    # weights
    for f in range(len(data.feature_names['external'])):
        # universal pressure
        w_universal_name = 'w_universal_' + str(data.feature_names['external'][f])
        column_names += [w_universal_name]

        row[w_universal_name] = samples['sample_weights'][s][f][0]

        # contact
        w_contact_name = 'w_contact_' + str(data.feature_names['external'][f])
        column_names += [w_contact_name]
        row[w_contact_name] = samples['sample_weights'][s][f][1]

        # inheritance
        if config['model']['INHERITANCE']:
            w_inheritance_name = 'w_inheritance_' + str(data.feature_names['external'][f])
            column_names += [w_inheritance_name]
            row[w_inheritance_name] = samples['sample_weights'][s][f][2]

    # alpha
    for f in range(len(data.feature_names['external'])):
        for st in data.state_names['external'][f]:
            feature_name = 'alpha_' + str(data.feature_names['external'][f]) \
                           + '_' + str(st)
            idx = data.state_names['external'][f].index(st)
            column_names += [feature_name]
            row[feature_name] = samples['sample_p_global'][s][0][f][idx]

    # gamma
    for a in range(config['model']['N_AREAS']):
        for f in range(len(data.feature_names['external'])):
            for st in data.state_names['external'][f]:
                feature_name = 'gamma_' + 'a' + str(a + 1) \
                               + '_' + str(data.feature_names['external'][f]) + '_' \
                               + str(st)
                idx = data.state_names['external'][f].index(st)

                column_names += [feature_name]
                row[feature_name] = samples['sample_p_zones'][s][a][f][idx]

    # beta
    if config['model']['INHERITANCE']:
        for fam in range(len(data.family_names['external'])):
            for f in range(len(data.feature_names['external'])):
                for st in data.state_names['external'][f]:
                    feature_name = 'beta_' + str(data.family_names['external'][fam]) \
                                   + '_' + str(data.feature_names['external'][f]) \
                                   + '_' + str(st)
                    idx = data.state_names['external'][f].index(st)
                    column_names += [feature_name]

                    row[feature_name] = samples['sample_p_families'][s][fam][f][idx]

    # Recall and precision
    if data.is_simulated:
        sample_z = np.any(samples['sample_zones'][s], axis=0)
        true_z = np.any(samples['true_zones'], axis=0)
        n_true = np.sum(true_z)
        intersections = np.minimum(sample_z, true_z)

        total_recall = np.sum(intersections, axis=0) / n_true
        precision = np.sum(intersections, axis=0) / np.sum(sample_z, axis=0)

        column_names += ['recall']
        row['recall'] = total_recall

        column_names += ['precision']
        row['precision'] = precision

    # Single areas
    if 'sample_lh_single_zones' in samples.keys():
        for a in range(config['model']['N_AREAS']):
            lh_name = 'lh_a' + str(a + 1)
            prior_name = 'prior_a' + str(a + 1)
            posterior_name = 'post_a' + str(a + 1)

            column_names += [lh_name]
            row[lh_name] = samples['sample_lh_single_zones'][s][a]

            column_names += [prior_name]
            row[prior_name] = samples['sample_prior_single_zones'][s][a]

            column_names += [posterior_name]
            row[posterior_name] = samples['sample_posterior_single_zones'][s][a]

    return row, column_names


def samples2file(samples, data, config, paths):
    """
    Writes the MCMC to two text files, one for MCMC parameters and one for areas.

    Args:
        samples (dict): samples
        data (Data): object of class data (features, priors, ...)
        config(dict): config information
        paths(dict): file path for stats, areas and ground truth
    """
    print("Writing results to file ...")
    # Write ground truth to file (for simulated data only)
    if data.is_simulated:

        try:
            with open(paths['gt'], 'w', newline='') as file:
                gt, gt_col_names = collect_gt_for_writing(samples=samples, data=data, config=config)
                writer = csv.DictWriter(file, fieldnames=gt_col_names, delimiter='\t')
                writer.writeheader()
                writer.writerow(gt)

        except IOError:
            print("I/O error")

        try:
            with open(paths['gt_areas'], 'w', newline='') as file:
                gt_areas = collect_gt_areas_for_writing(samples=samples)
                file.write(gt_areas)
                file.close()

        except IOError:
            print("I/O error")

    # Results
    steps_per_sample = float(config['mcmc']['N_STEPS'] / config['mcmc']['N_SAMPLES'])
    # Statistics
    try:
        writer = None
        with open(paths['parameters'], 'w', newline='') as file:

            for s in range(len(samples['sample_zones'])):
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
        with open(paths['areas'], 'w', newline='') as file:
            for s in range(len(samples['sample_zones'])):
                areas = collect_areas_for_writing(s, samples)
                file.write(areas + '\n')
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

    # print(f'rounding multiples {ups} {downs} {fewest_digits}')

    ups_rounded = []
    for n in ups:
        length = len(str(n))
        # print(f'n {n} length {length}')
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

    if n > offset: # number is larger than offset (must be positive)
        if mode == 'up':
            n_rounded = ceil(n / convertor) * convertor
            n_rounded += offset
        if mode == 'down':
            n_rounded = floor(n / convertor) * convertor
            n_rounded -= offset
        else:
            raise Exception('unkown mode')
    else: # number is smaller than offset (can be negative)
        if n >= 0:
            n_rounded = offset + convertor if mode == 'up' else -offset
        else:
            # for negative numbers we use round_int with inversed mode and the positive number
            inverse_mode = 'up' if mode == 'down' else 'down'
            n_rounded = round_int(abs(n), inverse_mode, offset)
            n_rounded = - n_rounded

    return n_rounded


def colorline(ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
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
    """Returns a list of maximum area sizes between a start and end size 
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
    """Compute the logarithm of (n choose k), i.e. the binomial coefficient of n and k.

    Args:
        n (int or np.array): Populations size..
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


# Fix path for default config files (in the folder sbayes/sbayes/config)
def fix_default_config(default_config_path):
    default_config_path = default_config_path.strip()
    if os.path.isabs(default_config_path):
        abs_config_path = default_config_path
        return abs_config_path
    else:

        # Get the beginning of the path, before "experiments"
        abs_config_path = ''
        path_elements = os.path.abspath(default_config_path).split('/')

        # For Windows
        if len(path_elements) == 1:
            path_elements = os.path.abspath(default_config_path).split("\\")

        for element in path_elements:
            if element == 'sbayes':
                break
            else:
                abs_config_path += element + '/'

        # Add the part that will be always there
        abs_config_path += 'sbayes/sbayes/' + \
                           '/'.join([os.path.dirname(default_config_path),
                                     os.path.basename(default_config_path)])

        return abs_config_path.replace("\\", "/")


# These two functions are copy pasted from experiment_setup.py, but they are not used yet
# todo: use them in the plotting classes
def decompose_config_path(config_path):
    config_path = config_path.strip()
    if os.path.isabs(config_path):
        abs_config_path = config_path
    else:
        abs_config_path = os.path.abspath(config_path)

    base_directory = os.path.dirname(abs_config_path)

    return base_directory, abs_config_path.replace("\\", "/")


def fix_relative_path(base_directory, path):
    """Make sure that the provided path is either absolute or relative to
    the config file directory.

    Args:
        path (str): The original path (absolute or relative).

    Returns:
        str: The fixed path.
     """
    path = path.strip()
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(base_directory, path).replace("\\", "/")


def seriation(Z, N, cur_index):
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)


def sort_by_similarity(similarity_matrix):
    n, _ = similarity_matrix.shape
    assert n == _, 'Similarity matrix needs to be a square matrix.'

    max_sim = np.max(similarity_matrix)
    dendrogram = linkage(max_sim-similarity_matrix, method='ward', preserve_input=True)
    order = seriation(dendrogram, n, 2 * n - 2)

    return order


def plot_similarity_matrix(similarities, names, show_similarity_overlay=False):
    n = len(similarities)

    order = sort_by_similarity(similarities)
    similarities = similarities[order, :][:, order]
    names_ordered = names[order]

    fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
    plt.imshow(similarities, origin='lower')
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names_ordered, rotation=-90, fontsize=9)
    ax.set_yticklabels(names_ordered, fontsize=9)

    if show_similarity_overlay:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, '%.2f' % similarities[i, j],
                        ha='center', va='center', color='w', fontsize=5)

    plt.show()


def timeit(func):

    def timed_func(*args, **kwargs):

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print('Runtime %s: %.4fs' % (func.__name__, (end - start)))

        return result

    return timed_func


if __name__ == "__main__":
    import doctest
    doctest.testmod()
