#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import copy
import logging
from multiprocessing import Pool
from functools import partial

import numpy as np
from scipy import stats, sparse
from scipy.sparse.csgraph import minimum_spanning_tree
import igraph
import math
from src.model import normalize_weights
from src.util import (compute_distance, simulate_family_of_size_k, compute_delaunay,
                      read_feature_occurrence_from_csv, global_counts_to_dirichlet, family_counts_to_dirichlet,
                      FamilyError)
import csv


EPS = np.finfo(float).eps


def get_sites(file, retrieve_family=False, retrieve_subset=False):
    """ This function retrieves the simulated sites from a csv, with the following columns:
            name: a unique identifier for each site
            x: the x-coordinate
            y: the y-coordinate
            cz: if a site belongs to a zone this is the id of the simulated zone, 0 otherwise.
            (family: if site belongs to a family this is the id of the simulated zone, 0 otherwise.)
    Args:
        file(str): file location of the csv
        retrieve_family(boolean): retrieve family assignments from the csv
        retrieve_subset(boolean): retrieve assignment to subsets from the csv
    Returns:
        dict, list: a dictionary containing the location tuple (x,y), the id and information about contact zones
            of each point the mapping between name and id is by position
    """

    columns = []
    with open(file, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            if columns:
                for i, value in enumerate(row):
                    columns[i].append(value)
            else:
                # first row
                columns = [[value] for value in row]
    csv_as_dict = {c[0]: c[1:] for c in columns}

    try:
        x = csv_as_dict.pop('x')
        y = csv_as_dict.pop('y')
        name = csv_as_dict.pop('name')
        cz = csv_as_dict.pop('cz')
    except KeyError:
        raise KeyError('The csv  must contain columns "x", "y", "name", "cz')

    if retrieve_family:
        try:
            family = csv_as_dict.pop('family')
        except KeyError:
            KeyError('The csv does not contain family information, i.e. a "family" column')

    if retrieve_subset:

        try:
            subset = csv_as_dict.pop('subset')
        except KeyError:
            KeyError('The csv does not contain subset information, i.e. a "subset" column')

    locations = np.zeros((len(name), 2))
    seq_id = []

    for i in range(len(name)):

        # Define location tuples
        locations[i, 0] = float(x[i])
        locations[i, 1] = float(y[i])

        # The order in the list maps name -> id and id -> name
        # name could be any unique identifier, sequential id are integers from 0 to len(name)
        seq_id.append(i)

    cz = [int(i) for i in cz]
    sites = {'locations': locations,
             'id': seq_id,
             'cz': cz,
             'names': name}

    if retrieve_family:
        family = [int(i) for i in family]
        sites['family'] = family

    if retrieve_subset:
        subset = [int(i) for i in subset]
        sites['subset'] = subset

    site_names = {'external': name,
                  'internal': list(range(0, len(name)))}

    return sites, site_names


def compute_network(sites, subset=None):
    """This function converts a set of sites (language locations plus attributes) into a network (graph).
    If a subset is defined, only those sites in the subset go into the network.

    Args:
        sites(dict): a dict of sites with keys "locations", "id"
        subset(list): boolean assignment of sites to subset
    Returns:
        dict: a network
        """

    if subset is None:

        # Define vertices and edges
        vertices = sites['id']

        # Delaunay triangulation
        delaunay = compute_delaunay(sites['locations'])
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
        diff = sites['locations'][:, None] - sites['locations']
        dist_mat = np.linalg.norm(diff, axis=-1)

        net = {'vertices': vertices,
               'edges': edges,
               'locations': sites['locations'],
               'names': sites['names'],
               'adj_mat': adj_mat,
               'n': len(vertices),
               'm': edges.shape[0],
               'graph': g,
               'dist_mat': dist_mat,
               }

    else:
        sub_idx = np.nonzero(subset)[0]
        vertices = list(range(len(sub_idx)))

        # Delaunay triangulation
        locations = sites['locations'][sub_idx, :]
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

        names = [sites['names'][i] for i in sub_idx]

        net = {'vertices': vertices,
               'edges': edges,
               'locations': locations,
               'names': names,
               'adj_mat': adj_mat,
               'n': len(vertices),
               'm': edges.shape[0],
               'graph': g,
               'dist_mat': dist_mat,
               }
    return net


def features_from_subset(features, subset):
    """This function returns the subset of a feature array
        Args:
            features(np.array): features for each site
                shape(n_sites, n_features, n_categories)
            subset(list): boolean assignment of sites to subset

        Returns:
            np.array: The feature subset
                shape(n_sub_sites, n_features, n_categories)
    """
    sub_idx = np.nonzero(subset)[0]
    return features[sub_idx, :, :]


# deprecated
def get_contact_zones_old(zone_id, sites):
    """This function retrieves those locations from the dict sites that are marked as contact zones (zone_id)
    Args:
        sites: a dict of sites with keys "cz", and "id"
        zone_id(int or tuple of ints): the id(s) of the contact zone(s)
    Returns:
        dict: the contact zones
        """
    contact_zones = {}
    cz = np.asarray(sites['cz'])
    # For single zones
    if isinstance(zone_id, int):
        contact_zones[zone_id] = np.where(cz == zone_id)[0].tolist()

    # For multiple zones
    elif isinstance(zone_id, tuple) and all(isinstance(x, int) for x in zone_id):
            for z in zone_id:
                contact_zones[z] = np.where(cz == z)[0].tolist()

    else:
        raise ValueError('zone_id must be int or a tuple of int')

    return contact_zones

# deprecated
def simulate_zones_old(zone_id, sites_sim):
    """ This function finds out which sites belong to contact zones assigns zone membership accordingly.

		Args:
            zone_id(int): The IDs of the simulated contact zones
            sites_sim (dict): dict with simulates sites
        Returns:
            (np.array): the simulated zones, boolean assignment of site to zone.
                shape(n_zones, n_sites)
    """

    # Retrieve zones
    sites_in_zone = get_contact_zones_old(sites=sites_sim, zone_id=zone_id)
    n_zones = len(sites_in_zone)
    n_sites = len(sites_sim['id'])

    # Assign zone membership
    zones = np.zeros((n_zones, n_sites), bool)
    for k, z_id in enumerate(sites_in_zone.values()):
        zones[k, z_id] = 1

    return zones

def get_contact_zones(zone_id, sites):
    """This function retrieves those locations from the dict sites that are marked as contact zones (zone_id)
    Args:
        sites: a dict of sites with keys "cz", and "id"
        zone_id(int or tuple of ints): the id(s) of the contact zone(s)
    Returns:
        dict: the contact zones
        """
    contact_zones = {}
    cz = np.asarray(sites['cz'])
    # For single zones
    if isinstance(zone_id, int):
        contact_zones[zone_id] = np.where(cz == zone_id)[0].tolist()

    # For multiple zones
    elif isinstance(zone_id, tuple) and all(isinstance(x, int) for x in zone_id):
            for z in zone_id:
                contact_zones[z] = np.where(cz == z)[0].tolist()

    else:
        raise ValueError('zone_id must be int or a tuple of int')

    return contact_zones


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


def simulate_features(zones,  p_global, p_contact, weights, inheritance, p_inheritance=None, families=None):
    """Simulate features for of all sites from the likelihood.

    Args:
        zones (np.array): Binary array indicating the assignment of sites to zones.
            shape: (n_zones, n_sites)
        families (np.array): Binary array indicating the assignment of a site to a language family.
            shape: (n_families, n_sites)
        p_global (np.array[float]): The global categorical probabilities of every category in every feature.
            shape: (n_features, n_categories)
        p_contact (np.array[float]): The categorical probabilities of every category in every features in every zones.
            shape: (n_zones, n_features, n_categories)
        p_inheritance (np.array): The probabilities of every category in every language family.
            shape: (n_families, n_features, n_categories)
        weights (np.array): The mixture coefficient controlling how much each feature is explained by contact,
            global distribution and inheritance.
            shape: (n_features, 3)
        inheritance(bool): Is inheritance (family membership) considered when simulating features?

    Returns:
        np.array: The sampled categories for all sites and features and categories
        shape:  n_sites, n_features, n_categories
    """
    n_zones, n_sites = zones.shape
    n_features, n_categories = p_global.shape

    # Are the weights fine?
    assert np.allclose(a=np.sum(weights, axis=-1), b=1., rtol=EPS)

    # Compute the global assignment and the assignment of sites to zones and families
    global_assignment = np.ones(n_sites)
    zone_assignment = np.any(zones, axis=0)

    if not inheritance:
        assignment = np.array([global_assignment, zone_assignment]).T

    else:
        family_assignment = np.any(families, axis=0)
        assignment = np.array([global_assignment, zone_assignment, family_assignment]).T

    # Normalize the weights for each site depending on whether zones or families are relevant for that site
    # Order of columns in weights: global, contact, (inheritance if available)
    weights = np.repeat(weights[np.newaxis, :, :], n_sites, axis=0)
    normed_weights = normalize_weights(weights, assignment)
    normed_weights = np.transpose(normed_weights, (1, 0, 2))

    features = np.zeros((n_sites, n_features), dtype=int)

    for i_feat in range(n_features):

        # Compute the feature likelihood matrix (for all sites and all categories)
        lh_global = p_global[np.newaxis, i_feat, :].T
        lh_zone = zones.T.dot(p_contact[:, i_feat, :]).T

        lh_feature = normed_weights[i_feat, :, 0] * lh_global + normed_weights[i_feat, :, 1] * lh_zone

        # Families
        if inheritance:

            lh_family = families.T.dot(p_inheritance[:, i_feat, :]).T
            lh_feature += normed_weights[i_feat, :, 2] * lh_family

        # Sample from the categorical distribution defined by lh_feature
        features[:, i_feat] = sample_categorical(lh_feature.T)

    # Categories per feature
    cats_per_feature =[]
    for f in features.transpose():

        cats_per_feature.append(np.unique(f).tolist())

    # Return per category
    cats = np.unique(features)
    features_cat = np.zeros((n_sites, n_features, len(cats)), dtype=int)
    for cat in cats:
            features_cat[:, :, cat] = np.where(features == cat, 1, 0)

    return features_cat, cats_per_feature


def sample_categorical(p):
    """Sample from a (multidimensional) categorical distribution. The
    probabilities for every category are given by `p`

    Args:
        p (np.array): Array defining the probabilities of every category at
            every site of the output array. The last axis defines the categories
            and should sum up to 1.
            shape: (*output_dims, n_categories)
    Returns
        np.array: Samples of the categorical distribution.
            shape: output_dims
    """
    *output_dims, n_categories = p.shape

    cdf = np.cumsum(p, axis=-1)
    z = np.expand_dims(np.random.random(output_dims), axis=-1)

    return np.argmax(z < cdf, axis=-1)


def compute_global_freq_from_data(feat):
    """This function computes the global frequency for a feature f to belongs to category cat;

    Args:
        feat (np.ndarray): a matrix of features
    Returns:
        array: the empirical probability that feature f belongs to category cat"""
    n_sites, n_features, n_categories = feat.shape
    p = np.zeros((n_features, n_categories), dtype=float)

    for f in range(n_features):
        counts = np.sum(feat[:, f, :], axis=0)
        p[f] = counts/n_sites
    return p


def estimate_geo_prior_parameters(network: dict, geo_prior):
    """
    This function estimates parameters for the geo prior
    If geo_prior = "gaussian", the estimated parameter is the covariance matrix of a two-dimensional Gaussian,
    if geo_prior = "distance", the estimated parameter is the scale parameter of an Exponential distribution

    Args:
        network (dict): network containing the graph, location,...
        geo_prior (char): Type of geo prior, "uniform", "distance" or "gaussian"
    Returns:
        (dict): The estimated parameters
    """
    parameters = {"uniform": None,
                  "gaussian": None,
                  "distance": None}

    if geo_prior == "uniform":
        return parameters

    else:
        dist_mat = network['dist_mat']
        locations = network['locations']

        delaunay = compute_delaunay(locations)
        mst = delaunay.multiply(dist_mat)

        # Compute difference vectors along mst
        if geo_prior == "gaussian":
            i1, i2 = mst.nonzero()
            diffs = locations[i1] - locations[i2]

            # Center at (0, 0)
            diffs -= np.mean(diffs, axis=0)[None, :]
            parameters["gaussian"] = np.cov(diffs.T)

        elif geo_prior == "distance":
            distances = mst.tocsr()[mst.nonzero()]
            parameters["distance"] = np.mean(distances)

        else:
            raise ValueError('geo_prior mut be "uniform", "distance" or "gaussian"')
    return parameters


def simulate_zones(zone_id, sites_sim):
    """ This function finds out which sites belong to contact zones assigns zone membership accordingly.

        Args:
            zone_id(int, tuple): The IDs of the simulated contact zones
            sites_sim (dict): dict with simulates sites

        Returns:
            (np.array): the simulated zones, boolean assignment of site to zone.
                shape(n_zones, n_sites)
    """

    # Retrieve zones
    sites_in_zone = {}
    cz = np.asarray(sites_sim['cz'])

    # For single zones
    if isinstance(zone_id, int):
        sites_in_zone[zone_id] = np.where(cz == zone_id)[0].tolist()

    # For multiple zones
    elif isinstance(zone_id, tuple) and all(isinstance(x, int) for x in zone_id):
            for z in zone_id:
                sites_in_zone[z] = np.where(cz == z)[0].tolist()
    else:
        raise ValueError('zone_id must be int or a tuple of int')

    n_zones = len(sites_in_zone)
    n_sites = len(sites_sim['id'])

    # Assign zone membership
    zones = np.zeros((n_zones, n_sites), bool)
    for k, z_id in enumerate(sites_in_zone.values()):
        zones[k, z_id] = 1

    return zones


def simulate_families(fam_id, sites_sim):
    """ This function finds out which sites belong to a family and assigns family membership accordingly.

        Args:
            fam_id(int): The IDs of the simulated families
            sites_sim (dict): dict with simulates sites

        Returns:
            (np.array): the simulated families, boolean assignment of site to zone.
                shape(n_families, n_sites)
    """
    # Retrieve families
    sites_in_families = {}
    family = np.asarray(sites_sim['family'])

    # For single family
    if isinstance(fam_id, int):
        sites_in_families[fam_id] = np.where(family == fam_id)[0].tolist()

    # For multiple families
    elif isinstance(fam_id, tuple) and all(isinstance(x, int) for x in fam_id):
        for f in fam_id:
            sites_in_families[f] = np.where(family == f)[0].tolist()
    else:
        raise ValueError('zone_id must be int or a tuple of int')

    n_families = len(sites_in_families)
    n_sites = len(sites_sim['id'])

    # Assign zone membership
    families = np.zeros((n_families, n_sites), bool)
    for k, z_id in enumerate(sites_in_families.values()):
        families[k, z_id] = 1

    return families


def simulate_weights(f_global, f_contact,  inheritance, n_features, f_inheritance=None):
    """ Simulates weights for all features, that is the influence of global bias, inheritance and contact on the feature.
    Args:
        f_global (float): controls the number of features for which the influence of global bias is high,
            passed as alpha when drawing samples from a dirichlet distribution
        f_contact(float): controls the number of features for which the influence of contact is high,
            passed as alpha when drawing samples from a dirichlet distribution
        f_inheritance: controls the number of features for which the influence of inheritance is high,
            passed as alpha when drawing samples from a dirichlet distribution, only relevant if inheritance = True
        inheritance: Is inheritance evaluated/simulated?
        n_features: Simulate weights for how many features?
    Returns:
        (np.array):
        """
    # Define alpha values which control the influence of contact (and inheritance if available) when simulating features
    if inheritance:
        alpha_sim = [f_global, f_contact, f_inheritance]
    else:
        alpha_sim = [f_global, f_contact]

    # columns in weights: global, contact, (inheritance if available)
    weights = np.random.dirichlet(alpha_sim, n_features)
    return weights


def simulate_assignment_probabilities(n_features, p_number_categories, inheritance, zones, intensity_contact,
                                      intensity_global, intensity_inheritance=None, families=None):
    """ Simulates the categories and then the assignment probabilities to categories (both in zones/families and globally)

       Args:
           n_features(int): number of features to simulate
           p_number_categories(dict): probability of simulating a feature with k categories
           inheritance(bool): Simulate probability ofr inheritance?
           zones (np.array): assignment of sites to zones (Boolean)
                shape(n_zones, n_sites)
           families (np.array): assignment of sites to families
                shape(n_families, n_sites)
           intensity_global(float): controls the intensity of the simulated global effect
           intensity_contact(float): controls the intensity of the simulated contact effect in the zones
           intensity_inheritance(float): controls the intensity of the simulated inheritance in the families
       Returns:
           (np.array, np.array, np.array): The global/zone/family weights per feature
       """
    cat = []
    p_cat = []
    for k, v in p_number_categories.items():
        cat.append(int(k))
        p_cat.append(v)

    # Simulate categories
    n_categories = np.random.choice(a=cat, size=n_features, p=p_cat)

    n_features = len(n_categories)
    max_categories = max(n_categories)
    n_zones = len(zones)

    # Initialize empty assignment probabilities
    p_global = np.zeros((n_features, max_categories), dtype=float)
    p_zones = np.zeros((n_zones, n_features, max_categories), dtype=float)

    # Simulate assignment to categories
    for f in range(n_features):
        cat_f = n_categories[f]

        # Global assignment
        alpha_p_global = [intensity_global] * cat_f
        p_global[f, range(cat_f)] = np.random.dirichlet(alpha_p_global, size=1)

        # Assignment in zones
        alpha_p_zones = [intensity_contact] * cat_f
        for z in range(n_zones):

            p_zones[z, f, range(cat_f)] = np.random.dirichlet(alpha_p_zones, size=1)

    # Simulate Inheritance?
    if not inheritance:
        return p_global, p_zones, None

    else:
        n_families = len(families)
        p_families = np.zeros((n_families, n_features, max_categories), dtype=float)

        for f in range(n_features):
            cat_f = n_categories[f]

            # Assignment in families
            alpha_p_families = [intensity_inheritance] * cat_f

            for fam in range(n_families):
                p_families[fam, f, range(cat_f)] = np.random.dirichlet(alpha_p_families, size=1)

        return p_global, p_zones, p_families


def get_p_global_prior(feature_names, category_names, file_location, log=True):
    """ This is a helper function to import global counts of each category of each feature,
        which then define dirichlet distributions that are used as a prior for p_global.
        Args:
            feature_names(dict): the features names (internal and external)
            category_names(dict): the category names (internal and external)
            file_location (str): The file location of the prior for p_global
            log(bool): write import information to the log-file?
        Returns:
            dict, dict: The dirichlet distribution for each category in each feature,  and the categories
    """
    file = file_location + "/prior_p_global.csv"
    cats = list(np.unique([x for l in category_names['external'] for x in l]))
    categories_ordered = []
    for f in category_names['external']:
        categories_ordered.append([cats.index(i) for i in f])

    # Read the global counts from csv
    counts, category_names_file, feature_names_file = read_feature_occurrence_from_csv(file=file)

    # Sanity check
    # Are the data count data?
    if not all(float(y).is_integer() for y in np.nditer(counts)):
        out = "The data in " + str(file) + " must be count data."
        raise ValueError(out)

    # Same number of features?
    if len(feature_names['external']) != len(feature_names_file['external']) or \
            len(feature_names['internal']) != len(feature_names_file['internal']):
        out = "Different number of features in " + str(file) + " as in features."
        raise ValueError(out)

    # Same feature names?
    for f in range(0, len(feature_names['external'])):
        if feature_names['external'][f] != feature_names_file['external'][f]:
            out = "The external feature " + str(f + 1) + " in " + str(file) \
                    + " differs from the one used in features."
            raise ValueError(out)
        if feature_names['internal'][f] != feature_names_file['internal'][f]:
            out = "The internal feature name " + str(f + 1) + " in " + str(file) \
                    + " differs from the one used in features."
            raise ValueError(out)

    # Same number of categories?
    if len(category_names['external']) != len(category_names_file['external']) or \
            len(category_names['internal']) != len(category_names_file['internal']):
        out = "Different number of categories in " + str(file) + " as in categories."
        raise ValueError(out)

    # Same category names?
    for f in range(0, len(category_names['external'])):
        if category_names['external'][f] != category_names_file['external'][f]:
            out = "The external category names for feature " + str(f + 1) + " in " + str(file) \
                    + " differ from those used in features."
            print(category_names['external'][f], category_names_file['external'][f])
            raise ValueError(out)

        if feature_names['internal'][f] != feature_names_file['internal'][f]:
            out = "The internal category names for " + str(f + 1) + " in " + str(file) \
                    + " differ from those used in features."
            raise ValueError(out)
    if log:
        logging.info('Import prior information for p_global from % s.', file)

    dirichlet = global_counts_to_dirichlet(counts, categories_ordered)
    return dirichlet, categories_ordered


def p_global_prior_from_complement(features, categories, subset):
    """ This is a helper function to compute a prior from p_global from the complement of a subset of simulated data.
        Args:
            features(np.array): the features
                shape(n_sites, n_features, n_categories)
            categories(list): available categeories per feature
            subset(list): boolean assignment of sites to subset

        Returns:
            dict, list: The dirichlet distribution for each category in each feature, and the categories
    """
    complement = [not i for i in subset]
    complement_idx = np.nonzero(complement)[0]
    features_complement = features[complement_idx, :, :]
    counts = np.sum(features_complement, axis=0)
    dirichlet = global_counts_to_dirichlet(counts, categories)

    return dirichlet, categories


def get_p_families_prior(family_names, feature_names, category_names, file_location, log=True):
    """ This is a helper function to import the counts of each feature in the families,
    which define dirichlet distributions that are used as prior for p_family.

    Args:
        family_names(dict): the names of the families (internal and external)
        feature_names(dict): the features names (internal and external)
        category_names(dict): the category names (internal and external
        file_location(str): the file location of the p_family_prior files
        log(bool): write import information to the log-file?
    Returns:
        dict, list: The dirichlet distribution per family for each category in each feature, and the categories
        """

    n_families = len(family_names['external'])
    n_features = len(feature_names['external'])
    n_categories = len(set(x for l in category_names['external'] for x in l))

    cats = list(np.unique([x for l in category_names['external'] for x in l]))
    categories_ordered = []
    for f in category_names['external']:
        categories_ordered.append([cats.index(i) for i in f])

    counts = np.empty([n_families, n_features, n_categories])

    for n in range(len(family_names['external'])):
        file = file_location + "/prior_p_families_" + str.lower(family_names['external'][n]) + ".csv"
        try:
            # Read the family counts from csv
            counts_fam, category_names_file, feature_names_file = read_feature_occurrence_from_csv(file=file)

            # Sanity check
            if not all(float(y).is_integer() for y in np.nditer(counts_fam)):
                out = "The data in " + str(file) + " must be count data."
                raise ValueError(out)

            if len(feature_names['external']) != len(feature_names_file['external']) or \
                    len(feature_names['internal']) != len(feature_names_file['internal']):
                out = "Different number of features in " + str(file) + " as in features."
                raise ValueError(out)

            for f in range(0, len(feature_names['external'])):
                if feature_names['external'][f] != feature_names_file['external'][f]:
                    out = "The external feature " + str(f+1) + " in " + str(file) \
                          + " differs from the one used in features."
                    raise ValueError(out)
                if feature_names['internal'][f] != feature_names_file['internal'][f]:
                    out = "The internal feature name " + str(f+1) + " in " + str(file) \
                          + " differs from the one used in features."
                    raise ValueError(out)

            if len(category_names['external']) != len(category_names_file['external']) or \
                    len(category_names['internal']) != len(category_names_file['internal']):
                out = "Different number of features in " + str(file) + " as in features."
                raise ValueError(out)

            for f in range(0, len(category_names['external'])):
                if category_names['external'][f] != category_names_file['external'][f]:
                    out = "The external category names for feature " + str(f+1) + " in " + str(file) \
                          + " differ from those used in features."
                    print(category_names['external'][f], category_names_file['external'][f])
                    raise ValueError(out)

                if feature_names['internal'][f] != feature_names_file['internal'][f]:
                    out = "The internal category names for " + str(f+1) + " in " + str(file) \
                          + " differ from those used in features."
                    raise ValueError(out)
            counts[n, :, :] = counts_fam

            if log:
                logging.info('Import prior information for %s from % s.', family_names['external'][n], file)

        except FileNotFoundError:

            n_categories = len(set(x for l in category_names['external'] for x in l))
            n_features = len(feature_names['external'])
            counts_fam = np.asarray([[0.] * n_categories for _ in range(n_features)])
            counts[n, :, :] = counts_fam

            print('No prior information for ' + str(family_names['external'][n]) + '. Uniform prior used instead.')

    dirichlet = family_counts_to_dirichlet(counts, categories_ordered)
    return dirichlet, categories_ordered
