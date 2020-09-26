#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import datetime
import csv
import os
from math import sqrt, floor, ceil

import numpy as np
import scipy.spatial as spatial
from scipy.special import betaln
import scipy.stats as stats
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from itertools import combinations


EPS = np.finfo(float).eps

FAST_DIRICHLET = False
if FAST_DIRICHLET:
    def dirichlet_pdf(x, alpha): return np.exp(stats.dirichlet._logpdf(x, alpha))
    dirichlet_logpdf = stats.dirichlet._logpdf
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

    n, _ = locations.shape
    delaunay = spatial.Delaunay(locations, qhull_options="QJ Pp")

    indptr, indices = delaunay.vertex_neighbor_vertices
    data = np.ones_like(indices)

    return csr_matrix((data, indices, indptr), shape=(n, n))


def n_smallest_distances(a, n, return_idx):
    """ This function finds the n smallest distances in a distance matrix

    Args:
        a (np.array): The distane matrix
        n (int): The number of distances to return
        return_idx (bool): return the indices of the points (True) or rather the distances (False)

    Returns:
        (np.array): the n_smallest distances
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


# Encoding
def encode_states(features_in):
    state_names = {'external': [],
                   'internal': []}
    features_enc = []
    for f in np.swapaxes(features_in, 0, 1):
        states_ext = list(np.unique(f))
        states_int = []
        if "" in states_ext:
            states_ext.remove("")

        s_idx = 0
        for s in list(states_ext):
            f = np.where(f == s, s_idx, f)
            states_int.append(s_idx)
            s_idx += 1
        state_names['external'].append(states_ext)
        state_names['internal'].append(states_int)
        features_enc.append(f)

    features_enc = np.column_stack([f for f in features_enc])
    enc_states = list(np.unique(features_enc))

    features_bin = []
    na_number = 0
    for cat in enc_states:
        if cat == "":
            na_number = np.count_nonzero(np.where(features_enc == cat, 1, 0))
        else:
            cat_axis = np.expand_dims(np.where(features_enc == cat, 1, 0), axis=2)
            features_bin.append(cat_axis)

    # Find all applicable states
    applicable_states = np.zeros((len(state_names['internal']), len(features_bin)))

    for f in range(len(state_names['internal'])):
        applicable_states[f, state_names['internal'][f]] = 1

    return np.concatenate(features_bin, axis=2), state_names, applicable_states.astype(bool), na_number


def read_features_from_csv(file):
    """This is a helper function to import data (sites, features, family membership,...) from a csv file
    Args:
        file (str): file location of the csv file
    Returns:
        (dict, dict, np.array, dict, dict, np.array, dict, str) :
        The language date including sites, site names, all features, feature names and state names per feature,
        as well as family membership and family names and log information
    """
    columns = []
    feature_names_ordered = []
    with open(file, 'rU', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if columns:
                for i, value in enumerate(row):
                    columns[i].append(value)
            else:
                # first row
                feature_names_ordered = [value for value in row]
                columns = [[value] for value in row]

    csv_as_dict = {c[0]: c[1:] for c in columns}
    try:
        x = csv_as_dict.pop('x')
        y = csv_as_dict.pop('y')
        l_id = csv_as_dict.pop('id')
        name = csv_as_dict.pop('name')
        family = csv_as_dict.pop('family')
        family = np.array(family)

        feature_names_ordered.remove('x')
        feature_names_ordered.remove('y')
        feature_names_ordered.remove('id')
        feature_names_ordered.remove('name')
        feature_names_ordered.remove('family')
    except KeyError:
        raise KeyError('The csv  must contain columns "x", "y", "id","name", "family"')

    # sites
    locations = np.zeros((len(l_id), 2))
    site_id = []

    for i in range(len(l_id)):
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
                  'internal': list(range(0, len(l_id)))}

    # features
    features_s = np.ndarray.transpose(np.array([csv_as_dict[i] for i in feature_names_ordered]))
    features, state_names, applicable_states, na_number = encode_states(features_s)

    feature_names = {'external': feature_names_ordered,
                     'internal': list(range(0, len(feature_names_ordered)))}

    # family
    family_names_ordered = np.unique(family).tolist()
    family_names_ordered = list(filter(None, family_names_ordered))

    families = np.zeros((len(family_names_ordered), len(l_id)), dtype=int)

    for fam in range(len(family_names_ordered)):
        families[fam, np.where(family == family_names_ordered[fam])] = 1

    family_names = {'external': family_names_ordered,
                    'internal': list(range(0, len(family_names_ordered)))}
    log = \
        str(len(site_names['internal'])) + " sites with " + \
        str(len(feature_names['internal'])) + " features read from " + \
        file + ". " + str(na_number) + " NA value(s) found."

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


def read_feature_occurrence_from_csv(file):
    """This is a helper function to import the occurrence of features in families (or globally) from a csv
        Args:
            file(str): file location of the csv file
        Returns:
            np.array :
            The occurrence of each feature, either as relative frequencies or counts, together with feature
            and category names
    """

    columns = []
    names = []

    with open(file, 'rU', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if columns:
                for i, value in enumerate(row):
                    columns[i].append(value)
            else:
                # first row
                names = [value for value in row]
                columns = [[value] for value in row]
    csv_as_dict = {c[0]: c[1:] for c in columns}

    try:
        feature_names_ordered = csv_as_dict.pop('feature')
        names.remove('feature')

    except KeyError:
        raise KeyError('The csv must contain column "feature"')

    occurr = np.zeros((len(feature_names_ordered), len(names)))

    feature_names = {'external': feature_names_ordered,
                     'internal': list(range(0, len(feature_names_ordered)))}

    names = np.unique(names)

    state_names_ordered = [[] for _ in range(len(feature_names_ordered))]
    for c in range(len(names)):
        row = csv_as_dict[names[c]]

        # Handle missing data
        for f in range(len(row)):
            try:
                row[f] = float(row[f])
                state_names_ordered[f].append(names[c])

            except ValueError:
                if row[f] == '':
                    row[f] = 0
                else:
                    raise ValueError("Frequencies must be numeric!")

        occurr[:, c] = row

    state_names = {'external': state_names_ordered,
                   'internal': [list(range(0, len(c))) for c in state_names_ordered]}

    return occurr, list(names), state_names, feature_names,


def tighten_counts(counts, state_names, count_names):

    # Tighten count matrix
    max_n_states = max([len(s) for s in state_names['external']])

    counts_tight = []
    for f in range(len(counts)):
        c = np.zeros(max_n_states)
        for s in range(len(state_names['external'][f])):
            external = state_names['external'][f][s]
            internal = state_names['internal'][f][s]
            idx = count_names.index(external)
            c[internal] = counts[f][idx]
        counts_tight.append(c)

    return np.row_stack([c for c in counts_tight]).astype(int)


def inheritance_counts_to_dirichlet(counts, categories, outdated_features=None, dirichlet=None):
    """This is a helper function transform the family counts to alpha values that
    are then used to define a dirichlet distribution

    Args:
        counts(np.array): the family counts
            shape(n_families, n_features, n_categories)
        categories(list): categories per feature in each of the families
    Returns:
        list: the dirichlet distributions, neatly stored in a dict
    """
    n_fam, n_feat, n_cat = counts.shape
    if dirichlet is None:
        dirichlet = [None] * n_fam

    for fam in range(n_fam):
        dirichlet[fam] = counts_to_dirichlet(counts[fam], categories,
                                             outdated_features=outdated_features,
                                             dirichlet=dirichlet[fam])
    return dirichlet


def counts_to_dirichlet(counts, categories, prior='uniform', outdated_features=None, dirichlet=None):
    """This is a helper function to transform counts of categorical data
    to parameters of a dirichlet distribution.

    Args:
        counts(np.array): the counts of categorical data.
                    shape(n_features, n_states)
        categories(np.array): applicable states/categories per feature
        prior (str): Use one of the following uninformative priors:
            'uniform': A uniform prior probability over the probability simplex Dir(1,...,1)
            'jeffrey': The Jeffrey's prior Dir(0.5,...,0.5)
            'naught': A natural exponential family prior Dir(0,...,0).
        outdated_features (IndexSet): Indices of the features where the counts changed
                                  (i.e. they need to be updated).
    Returns:
        list: a dirichlet distribution derived from pseudocounts
    """
    n_features, n_categories = counts.shape

    prior_map = {'uniform': 1, 'jeffrey': 0.5, 'naught': 0}

    if outdated_features is None or outdated_features.all:
        outdated_features = range(n_features)
        dirichlet = [None] * n_features
    else:
        assert dirichlet is not None

    for feat in outdated_features:
        cat = categories[feat]
        # Add 1 to alpha values (1,1,...1 is a uniform prior)
        pseudocounts = counts[feat, cat] + prior_map[prior]
        # dirichlet[feat] = stats.dirichlet(pseudocounts)
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
        sample_z = samples['sample_zones'][s][0]
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
    For example:
    up: 113 -> 120, 3456 -> 3500
    down: 113 -> 110, 3456 -> 3450

    Args:
         n (int): integer number to round
         mode (str): round 'up' or 'down'
         offset (int): adding offset to rounded number
    """

    # print(f'rounding {mode} {n} (position {position} offset {offset})')

    # convert to int if necessary and get number of digits
    n = int(n) if isinstance(n, float) else n
    n_digits = len(str(n)) if n > 0 else len(str(n)) - 1

    # check for validity of input parameters
    if mode != 'up' and mode != 'down':
        raise Exception('unkown mode')
        # raise Exception(f'Unknown mode: "{mode}". Use either "up" or "down".')
    if position > n_digits:
        raise Exception('unkown mode')
        # raise Exception(f'Position {position} is not valid for a number with only {n_digits} digits.')

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
            # print(f'factor {factor} base {base}')
            n_rounded = base - offset * factor if mode == 'down' else base + ((offset + 1) * factor)
        else:
            n_rounded = n - offset if mode == 'down' else n + offset

    # print(f'rounded to {n_rounded}')
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

    # print(f'rounding {n} {mode} by {offset}')
    # print(convertor)

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

    Returns:
         np.array: x with normalized s.t. the last axis sums to 1.
    """
    return x / np.sum(x, axis=axis, keepdims=True)


def mle_weights(samples):
    """

    Args:
        samples (np.array):
    Returns:
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


def assess_correlation_probabilities(p_universal, p_contact, p_inheritance, corr_th):
    """Asses the correlation of probabilities

        Args:
            p_universal (np.array): universal state probabilities
                shape (n_features, n_states)
            p_contact (np.array): state probabilities in areas
                shape: (n_areas, n_features, n_states)
            p_inheritance (np.array): state probabilities in families
                shape: (n_families, n_features, n_states)
            corr_th (float): correlation threshold

        """
    if p_inheritance is not None:
        samples = np.vstack((p_universal[np.newaxis, :, :], p_contact, p_inheritance))
    else:
        samples = np.vstack((p_universal[np.newaxis, :, :], p_contact))
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
