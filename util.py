#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import pickle
import numpy as np
import scipy.spatial as spatial
import scipy.stats as stats
from scipy.sparse import csr_matrix
from math import sqrt
import datetime
import csv
import os
import random
from scipy.misc import logsumexp

EPS = np.finfo(float).eps


class FamilyError(Exception):
    pass


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


def transform_weights_from_log(log_weights):
    """Transforms the weights from log space and normalizes them such that they sum to 1

    Args:
        log_weights (np.array): The non-normalized weights in log space
            shape(n_features, 3)
    Returns:
        (np.array): transformed and normalized weights
    """

    # Normalize in original space and transform
    log_weights -= logsumexp(log_weights, keepdims=True)
    weights_norm = np.exp(log_weights)

    return weights_norm


def transform_p_from_log(log_p):
    """Transforms the probabilities from log space and normalizes them such that they sum to 1
    Args:
        log_p (np.array): The non-normalized probabilities in log space
                shape(n_zones, n_features, n_categories)
    Returns:
        (np.array): transformed and normalized weights
    """
    # Transform to original space
    p = np.exp(log_p)

    # Normalize
    p_norm = p / p.sum(axis=2, keepdims=True)

    return p_norm


def transform_weights_to_log(weights):
    """Transforms the weights to log-space
    Args:
        weights (np.array): The weights
            shape(n_features, 3)
    Returns:
        (np.array): transformed weights in log-space
    """

    # Transform to log space
    log_weights = np.log(weights)

    return log_weights


def transform_p_to_log(p):
    """Transforms the probabilities to log-space

    Args:
        p (np.array): The non-normalized probabilities in log space
            shape(n_zones, n_features, n_categories)
    Returns:
        (np.array): transformed probabilities in log-space
     """

    # Transform to log space
    with np.errstate(divide='ignore'):
        log_p = np.log(p)
    return log_p


def read_languages_from_csv(file):
    """This is a helper function to import language data (sites, features, family membership,...) from a csv file
        Args:
            file(str): file location of the csv file
        Returns:
            (dict, dict, np.array, dict, dict, np.array, dict) :
            The language date including sites, site names, all features, feature names and category names per feature,
            as well as family membership and family names
    """
    columns = []
    feature_names_ordered = []
    with open(file, 'rU') as f:
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
        name = csv_as_dict.pop('name')
        family = csv_as_dict.pop('family')
        family = np.array(family)

        feature_names_ordered.remove('x')
        feature_names_ordered.remove('y')
        feature_names_ordered.remove('name')
        feature_names_ordered.remove('family')
    except KeyError:
        raise KeyError('The csv  must contain columns "x", "y", "name", "family')

    # sites
    locations = np.zeros((len(name), 2))
    id = []

    for i in range(len(name)):
        # Define location tuples
        locations[i, 0] = float(x[i])
        locations[i, 1] = float(y[i])

        # The order in the list maps name to id and id to name
        # name could be any unique identifier, id is an integer from 0 to len(name)
        id.append(i)

    sites = {'locations': locations,
             'id': id,
             'cz': None,
             'names': name}
    site_names = {'external': name,
                  'internal': list(range(0, len(name)))}

    # features

    features_with_cat = np.ndarray.transpose(np.array([csv_as_dict[i] for i in feature_names_ordered]))

    cat_names = np.unique(features_with_cat)

    features_cat = []
    for cat in cat_names:
        if cat == "":
            na_number = np.count_nonzero(np.where(features_with_cat == cat, 1, 0))
            print(na_number, "NA value(s) found in the data.")
        else:
            cat_axis = np.expand_dims(np.where(features_with_cat == cat, 1, 0), axis=2)
            features_cat.append(cat_axis)
    features = np.concatenate(features_cat, axis=2)
    feature_names = {'external': feature_names_ordered,
                     'internal': list(range(0, len(feature_names_ordered)))}

    # categories per feature
    category_names_ordered = []

    for f in features_with_cat.transpose():
        cat_per_feature = []
        for cat in cat_names:
            if cat in f and cat != "":
                cat_per_feature.append(cat)
        category_names_ordered.append(cat_per_feature)

    category_names = {'external': category_names_ordered,
                      'internal': [list(range(0, len(c))) for c in category_names_ordered]}

    # family
    family_names_ordered = np.unique(family).tolist()
    family_names_ordered = list(filter(None, family_names_ordered))

    families = np.zeros((len(family_names_ordered), len(name)), dtype=int)

    for fam in range(len(family_names_ordered)):
        families[fam, np.where(family == family_names_ordered[fam])] = 1

    family_names = {'external': family_names_ordered,
                    'internal': list(range(0, len(family_names_ordered)))}

    return sites, site_names, features, feature_names, category_names, families, family_names


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

    with open(file, mode='w', newline='') as csv_file:
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

    with open(file, mode='w', newline='') as csv_file:
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
    cat_names = []
    with open(file, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            if columns:
                for i, value in enumerate(row):
                    columns[i].append(value)
            else:
                # first row
                cat_names = [value for value in row]
                columns = [[value] for value in row]
    csv_as_dict = {c[0]: c[1:] for c in columns}

    try:
        feature_names_ordered = csv_as_dict.pop('feature')
        cat_names.remove('feature')

    except KeyError:
        raise KeyError('The csv must contain column "feature"')
    occurr = np.zeros((len(feature_names_ordered), len(cat_names)))

    feature_names = {'external': feature_names_ordered,
                     'internal': list(range(0, len(feature_names_ordered)))}

    cat_names = np.unique(cat_names)

    cat_names_ordered = [[] for _ in range(len(feature_names_ordered))]
    for c in range(len(cat_names)):
        row = csv_as_dict[cat_names[c]]

        # Handle missing data
        for f in range(len(row)):
            try:
                row[f] = float(row[f])
                cat_names_ordered[f].append(cat_names[c])

            except ValueError:
                if row[f] == '':
                    row[f] = 0
                else:
                    raise ValueError("Frequencies must be numeric!")
        occurr[:, c] = row

    category_names = {'external': cat_names_ordered,
                      'internal': [list(range(0, len(c))) for c in cat_names_ordered]}

    return occurr, category_names, feature_names


def counts_to_dirichlet(counts, categories):
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
    dirichlet = [[] for _ in range(n_fam)]

    for fam in range(n_fam):
        for feat in range(n_feat):
            cat = categories[feat]

            alpha = counts[fam, feat, cat]
            # Add 1 to alpha values (1,1,...1 is a uniform prior)
            alpha = alpha + 1

            dirichlet[fam].append(stats.dirichlet(alpha))

    return dirichlet


def balance_p_array(p_array, balance_by):
    """This is a helper function to balance an array of probabilities, such that no probability is zero

        Args:
            p_array(np.array): the array of probabilities
            balance_by(float): how much of the non-zero probabilities should be distributed to the zero ones (0-1)
        Returns:
            np.array: the balanced p_array
        """
    # Find all zeros and non-zeros

    zero_idx = np.where(p_array == 0)[0]
    nonzero_idx = np.where(p_array != 0)[0]

    # Subtract balance_by from all nonzero probabilities (proportional to their magnitude)
    non_zero_proportion = p_array[nonzero_idx] / sum(p_array[nonzero_idx])
    p_array_nonzero_balanced = p_array[nonzero_idx] - non_zero_proportion * balance_by

    # Balance and update
    p_array_zero_balanced = balance_by / len(zero_idx)

    p_array[zero_idx] = p_array_zero_balanced
    p_array[nonzero_idx] = p_array_nonzero_balanced

    return p_array


def touch(fname):
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()


def mkpath(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isdir(path):
        touch(path)


def form_family_of_size_k(k, net, already_occupied=None, grow_families=True):
    """ This function forms a family of size k, either by
    randomly selecting sites in a network or growing regions in space (similar to grow_zone_of_size_k)
    Args:
        k (int): The size of the family i.e. the number of sites in the family.
        net: dict of network
        already_occupied (np.array): All sites already assigned to a family or to a zone (boolean)

    Returns:
        np.array: The newly formed family (boolean).
        np.array: all nodes in the network already assigned to a family or a zone (boolean).
    """
    n_sites = len(net['vertices'])
    if already_occupied is None:
        already_occupied = np.zeros(n_sites, bool)

    # Initialize the family
    family = np.zeros(n_sites, bool)

    # Find all sites that already belong to a family or a zone (sites_occupied) and those that don't (sites_free)
    sites_occupied = np.nonzero(already_occupied)[0]
    sites_free = set(range(n_sites)) - set(sites_occupied)

    # Grow families
    if grow_families:
        # Take a random free site and use it as seed for the new family
        try:
            i = random.sample(sites_free, 1)[0]
            family[i] = already_occupied[i] = 1
        except ValueError:
            print('No more free sites to grow family.')

        # Grow the zone if possible
        for _ in range(k - 1):

            # todo: self.adj_mat
            neighbours = get_neighbours(family, already_occupied, net['adj_mat'])
            if not np.any(neighbours):
                print('No more free sites to grow family.')

            # Add a neighbour to the zone
            site_new = random.choice(neighbours.nonzero()[0])
            family[site_new] = already_occupied[site_new] = 1

    # Assign random points to families
    else:
        i = random.sample(sites_free, k)
        family[i] = already_occupied[i] = 1

    return family, already_occupied


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