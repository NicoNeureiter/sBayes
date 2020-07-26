#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import csv
import numpy as np

from sbayes.model import normalize_weights
from sbayes.util import (compute_distance, compute_delaunay,
                      read_feature_occurrence_from_csv)

EPS = np.finfo(float).eps


def read_sites(file, retrieve_family=False, retrieve_subset=False):
    """ This function reads the simulated sites from a csv, with the following columns:
        name: a unique identifier for each site
        x: the x-coordinate
        y: the y-coordinate
        cz: if a site belongs to a area this is the id of the simulated zone, 0 otherwise.
        (family: if site belongs to a family this is the id of the simulated family, 0 otherwise.)
    Args:
        file(str): file location of the csv
        retrieve_family(boolean): retrieve family assignments from the csv
        retrieve_subset(boolean): retrieve assignment to subsets from the csv
    Returns:
        dict, list: a dictionary containing the location tuples (x,y), the id and information about contact zones
            of each point the mapping between name and id is by position
    """

    columns = []
    with open(file, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            if columns:
                for i, value in enumerate(row):
                    if len(value) < 1:
                        columns[i].append(0)
                    else:
                        columns[i].append(value)
            else:
                # first row
                columns = [[value] for value in row]
    csv_as_dict = {c[0]: c[1:] for c in columns}

    try:
        x = csv_as_dict.pop('x')
        y = csv_as_dict.pop('y')
        name = csv_as_dict.pop('name')
        area = csv_as_dict.pop('area')
    except KeyError:
        raise KeyError('The csv  must contain columns "x", "y", "name", "area')

    locations = np.zeros((len(name), 2))
    seq_id = []

    for i in range(len(name)):

        # Define location tuples
        locations[i, 0] = float(x[i])
        locations[i, 1] = float(y[i])

        # The order in the list maps name -> id and id -> name
        # name could be any unique identifier, sequential id are integers from 0 to len(name)
        seq_id.append(i)

    sites = {'locations': locations,
             'id': seq_id,
             'area': [int(z) for z in area],
             'names': name}

    if retrieve_family:
        try:
            families = csv_as_dict.pop('family')
            sites['family'] = [int(f) for f in families]

        except KeyError:
            KeyError('The csv does not contain family information, i.e. a "family" column')

    if retrieve_subset:
        try:
            subset = csv_as_dict.pop('subset')
            sites['subset'] = [int(s) for s in subset]

        except KeyError:
            KeyError('The csv does not contain subset information, i.e. a "subset" column')

    site_names = {'external': name,
                  'internal': list(range(0, len(name)))}

    log = str(len(name)) + " locations read from " + str(file)
    return sites, site_names, log


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
               'dist_mat': dist_mat,
               }
    return net


def subset_features(features, subset):
    """This function returns the subset of a feature array
        Args:
            features(np.array): features for each site
                shape(n_sites, n_features, n_categories)
            subset(list): boolean assignment of sites to subset

        Returns:
            np.array: The subset
                shape(n_sub_sites, n_features, n_categories)
    """
    sub = np.array(subset, dtype=bool)
    print(sub, "sub")
    return features[sub, :, :]


def simulate_features(areas,  p_universal, p_contact, weights, inheritance, p_inheritance=None, families=None):
    """Simulate features for of all sites from the likelihood.

    Args:
        areas (np.array): Binary array indicating the assignment of sites to areas.
            shape: (n_areas, n_sites)
        families (np.array): Binary array indicating the assignment of a site to a language family.
            shape: (n_families, n_sites)
        p_universal (np.array[float]): The universal probabilities of every state.
            shape: (n_features, n_categories)
        p_contact (np.array[float]): The  contact probabilities of every state in every area.
            shape: (n_areas, n_features, n_categories)
        p_inheritance (np.array): The probabilities of every state in every language family.
            shape: (n_families, n_features, n_categories)
        weights (np.array): The mixture coefficient controlling how much each feature is explained
            by universal pressure, contact, and inheritance.
            shape: (n_features, 3)
        inheritance(bool): Is inheritance (family membership) considered when simulating features?

    Returns:
        np.array: The sampled categories for all sites and features and states
        shape:  n_sites, n_features, n_categories
    """
    n_areas, n_sites = areas.shape
    n_features, n_categories = p_universal.shape

    # Are the weights fine?
    assert np.allclose(a=np.sum(weights, axis=-1), b=1., rtol=EPS)

    # Compute the universal assignment and the assignment of sites to areas and families
    universal_assignment = np.ones(n_sites)
    area_assignment = np.any(areas, axis=0)

    if not inheritance:
        assignment = np.array([universal_assignment, area_assignment]).T

    else:
        family_assignment = np.any(families, axis=0)
        assignment = np.array([universal_assignment, area_assignment, family_assignment]).T

    # Normalize the weights for each site depending on whether areas or families are relevant for that site
    # Order of columns in weights: universal, contact, (inheritance if available)
    weights = np.repeat(weights[np.newaxis, :, :], n_sites, axis=0)
    normed_weights = normalize_weights(weights, assignment)
    normed_weights = np.transpose(normed_weights, (1, 0, 2))

    features = np.zeros((n_sites, n_features), dtype=int)

    for i_feat in range(n_features):

        # Compute the feature likelihood matrix (for all sites and all categories)
        lh_universal = p_universal[np.newaxis, i_feat, :].T
        lh_area = areas.T.dot(p_contact[:, i_feat, :]).T

        lh_feature = normed_weights[i_feat, :, 0] * lh_universal + normed_weights[i_feat, :, 1] * lh_area

        # Families
        if inheritance:

            lh_family = families.T.dot(p_inheritance[:, i_feat, :]).T
            lh_feature += normed_weights[i_feat, :, 2] * lh_family

        # Sample from the categorical distribution defined by lh_feature
        features[:, i_feat] = sample_categorical(lh_feature.T)

    # Categories per feature
    cats_per_feature = []
    for f in features.transpose():

        cats_per_feature.append(np.unique(f).tolist())

    # Return per category
    cats = np.unique(features)
    features_cat = np.zeros((n_sites, n_features, len(cats)), dtype=int)
    for cat in cats:
            features_cat[:, :, cat] = np.where(features == cat, 1, 0)

    feature_names = {'external': ['f' + str(f+1) for f in range(features_cat.shape[1])],
                     'internal': [f for f in range(features_cat.shape[1])]}

    state_names = {'external': cats_per_feature,
                   'internal': cats_per_feature}

    return features_cat, cats_per_feature, feature_names, state_names


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


def simulate_areas(area_id, sites_sim):
    """ This function finds out which sites belong to contact areas and assigns areal membership accordingly.

        Args:
            area_id(int, tuple): The IDs of the simulated contact areas
            sites_sim (dict): dict with simulates sites

        Returns:
            (np.array): the simulated areas, boolean assignment of site to area.
                shape(n_areas, n_sites)
    """

    # Retrieve areas
    sites_in_area = {}
    area = np.asarray(sites_sim['area'])

    # For single area
    if isinstance(area_id, int):
        sites_in_area[area_id] = np.where(area == area_id)[0].tolist()

    # For multiple areas
    elif isinstance(area_id, tuple) and all(isinstance(x, int) for x in area_id):
            for z in area_id:
                sites_in_area[z] = np.where(area == z)[0].tolist()
    else:
        raise ValueError('area_id must be int or a tuple of int')

    n_areas = len(sites_in_area)
    n_sites = len(sites_sim['id'])

    # Assign areal membership
    areas = np.zeros((n_areas, n_sites), bool)
    for k, z_id in enumerate(sites_in_area.values()):
        areas[k, z_id] = 1

    return areas


def simulate_families(fam_id, sites_sim):
    """ This function finds out which sites belong to a family and assigns family membership accordingly.

        Args:
            fam_id(int): The IDs of the simulated families
            sites_sim (dict): dict with simulates sites

        Returns:
            (np.array): the simulated families, boolean assignment of site to family.
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
        raise ValueError('area_id must be int or a tuple of int')

    n_families = len(sites_in_families)
    n_sites = len(sites_sim['id'])

    # Assign family membership
    families = np.zeros((n_families, n_sites), bool)
    for k, z_id in enumerate(sites_in_families.values()):
        families[k, z_id] = 1

    family_names = {'external': ['fam' + str(s + 1) for s in range(families.shape[0])],
                    'internal': [s for s in range(families.shape[0])]}
    return families, family_names


def simulate_weights(i_universal, i_contact,  inheritance, n_features, i_inheritance=None):
    """ Simulates weights for all features, that is the influence of global bias, inheritance and contact on the feature.
    Args:
        i_universal (float): controls the number of features for which the influence of universal pressure is high,
            passed as alpha when drawing samples from a dirichlet distribution
        i_contact(float): controls the number of features for which the influence of contact is high,
            passed as alpha when drawing samples from a dirichlet distribution
        i_inheritance: controls the number of features for which the influence of inheritance is high,
            passed as alpha when drawing samples from a dirichlet distribution, only relevant if inheritance = True
        inheritance: Is inheritance evaluated/simulated?
        n_features: Simulate weights for how many features?
    Returns:
        (np.array):
        """
    # Define alpha values which control the influence of contact (and inheritance if available) when simulating features
    if inheritance:
        alpha_sim = [i_universal, i_contact, i_inheritance]
    else:
        alpha_sim = [i_universal, i_contact]

    # columns in weights: global, contact, (inheritance if available)
    weights = np.random.dirichlet(alpha_sim, n_features)
    return weights


def simulate_assignment_probabilities(n_features, p_number_categories, inheritance, areas, e_universal,
                                      e_contact, e_inheritance=None, families=None):
    """ Simulates the categories and then the assignment probabilities to categories in areas, families and universally

       Args:
           n_features(int): number of features to simulate
           p_number_categories(dict): probability of simulating a feature with k categories
           inheritance(bool): Simulate probability ofr inheritance?
           areas (np.array): assignment of sites to areas (Boolean)
                shape(n_areas, n_sites)
           families(np.array): assignment of sites to families
                shape(n_families, n_sites)
           e_universal (float): controls the entropy of the simulated universal pressure
           e_contact(float): controls the entropy of the simulated contact effect in the areas
           e_inheritance(float): controls the entropy of the simulated inheritance in the families
       Returns:
           (np.array, np.array, np.array): The assignment probabilities (universal, areal, inheritance) per feature
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
    n_areas = len(areas)

    # Initialize empty assignment probabilities
    p_universal = np.zeros((n_features, max_categories), dtype=float)
    p_contact = np.zeros((n_areas, n_features, max_categories), dtype=float)

    # Simulate assignment to categories
    for f in range(n_features):
        cat_f = n_categories[f]

        # Universal assignment
        alpha_p_universal = np.full(shape=cat_f, fill_value=e_universal)
        p_universal[f, range(cat_f)] = np.random.dirichlet(alpha_p_universal, size=1)

        # Assignment in areas
        alpha_p_contact = np.full(shape=cat_f, fill_value=e_contact)
        for z in range(n_areas):
            p_contact[z, f, range(cat_f)] = np.random.dirichlet(alpha_p_contact, size=1)

    # Simulate Inheritance?
    if not inheritance:
        return p_universal, p_contact, None

    else:
        n_families = len(families)
        p_inheritance = np.zeros((n_families, n_features, max_categories), dtype=float)

        for f in range(n_features):
            cat_f = n_categories[f]

            # Assignment in families
            alpha_p_inheritance = [e_inheritance] * cat_f

            for fam in range(n_families):
                p_inheritance[fam, f, range(cat_f)] = np.random.dirichlet(alpha_p_inheritance, size=1)

        return p_universal, p_contact, p_inheritance


def read_universal_counts(feature_names, category_names, file):
    """ This is a helper function to import global counts of each category of each feature,
        which then define dirichlet distributions that are used as a prior for p_global.
        Args:
            feature_names(dict): the features names (internal and external)
            category_names(dict): the category names (internal and external)
            file (str): The file location of the pseudocounts for the prior

        Returns:
            dict, dict: The dirichlet distribution for each category in each feature,  and the categories
    """

    cats = list(np.unique([x for l in category_names['external'] for x in l]))
    categories_ordered = []
    for f in category_names['external']:
        categories_ordered.append([cats.index(i) for i in f])
    try:
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
                out = "The feature " + str(f + 1) + " in " + str(file) \
                        + " differs from the one used in features."
                raise ValueError(out)
            if feature_names['internal'][f] != feature_names_file['internal'][f]:
                out = "The feature name " + str(f + 1) + " in " + str(file) \
                        + " differs from the one used in features."
                raise ValueError(out)

        # Same number of categories?
        if len(category_names['external']) != len(category_names_file['external']) or \
                len(category_names['internal']) != len(category_names_file['internal']):
            out = "Different number of states in " + str(file) + " as in categories."
            raise ValueError(out)

        # Same category names?
        for f in range(0, len(category_names['external'])):
            if category_names['external'][f] != category_names_file['external'][f]:
                out = "The state names for feature " + str(f + 1) + " in " + str(file) \
                        + " differ from those used in features."
                raise ValueError(out)

            if feature_names['internal'][f] != feature_names_file['internal'][f]:
                out = "The state names for " + str(f + 1) + " in " + str(file) \
                        + " differ from those used in features."
                raise ValueError(out)

        log = 'Read universal counts from' + str(file)

    except (KeyError, FileNotFoundError):

        n_categories = len(set(x for l in category_names['external'] for x in l))
        n_features = len(feature_names['external'])
        counts = np.asarray([[0.] * n_categories for _ in range(n_features)])
        counts[:, :] = counts

        log = "No prior information for universal preference. Uniform prior used instead.\n"

    return counts, categories_ordered, log


def counts_from_complement(features, subset):
    """ This is a helper function to compute pseudocounts from the complement of a subset in simulated data.
        Args:
            features(np.array): the features
                shape(n_sites, n_features, n_states)
            subset(list): boolean assignment of sites to subset

        Returns:
            np.array: The pseudocounts
    """
    # Compute pseudocounts from the complement of the subset
    complement = [not i for i in subset]
    complement_idx = np.nonzero(complement)[0]
    features_complement = features[complement_idx, :, :]
    counts = np.sum(features_complement, axis=0)
    return counts


def read_inheritance_counts(family_names, feature_names, category_names, files):
    """ This is a helper function to import the counts of each feature in the families,
    which define dirichlet distributions that are used as prior for p_family.

    Args:
        family_names(dict): the names of the families (internal and external)
        feature_names(dict): the features names (internal and external)
        category_names(dict): the category names (internal and external
        files(dict): path to the file locations
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
    log = str()

    for n in range(len(family_names['external'])):
        try:
            file = files[family_names['external'][n]]

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
                    raise ValueError(out)

                if feature_names['internal'][f] != feature_names_file['internal'][f]:
                    out = "The internal category names for " + str(f+1) + " in " + str(file) \
                          + " differ from those used in features."
                    raise ValueError(out)
            counts[n, :, :] = counts_fam

            log += "Read counts for " + str(family_names['external'][n]) + " from " + str(file) + "\n"

        except (KeyError, FileNotFoundError):

            n_categories = len(set(x for l in category_names['external'] for x in l))
            n_features = len(feature_names['external'])
            counts_fam = np.asarray([[0.] * n_categories for _ in range(n_features)])
            counts[n, :, :] = counts_fam

            log += "No prior information for " + str(family_names['external'][n]) + ". Uniform prior used instead.\n"
    return counts, categories_ordered, log
