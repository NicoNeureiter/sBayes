#!/usr/bin/env python3
# -*- coding: utf-8 -*-
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import csv
import sys

import numpy as np
import pyproj

from sbayes.model import normalize_weights
from sbayes.util import (compute_delaunay,
                         read_feature_occurrence_from_csv,
                         read_features_from_csv,
                         read_costs_from_csv)

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
    filemode = 'r' if sys.version_info >= (3, 4) else 'rU'
    with open(file, filemode) as f:
        reader = csv.reader(f)
        for row in reader:
            if columns:
                for i, value in enumerate(row):
                    if len(value) < 1:
                        columns[i].append(str(0))
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

class compute_network:
    def __init__(
            self,
            sites,
            subset=None,
            crs=None):
        """Convert a set of sites into a network.

        This function converts a set of language locations, with their attributes,
        into a network (graph). If a subset is defined, only those sites in the
        subset go into the network.

        Args:
            sites(dict): a dict of sites with keys "locations", "id"
            subset(list): boolean assignment of sites to subset
        Returns:
            dict: a network

        """
        if crs is not None:
            try:
                from cartopy import crs as geodesic
            except ImportError as e:
                print("Using a coordinate reference system (crs) requires the ´cartopy´ library:")
                print("pip install cartopy")
                raise e

        if subset is None:
            # Define vertices and edges
            vertices = sites['id']

            locations = sites['locations']

            # Distance matrix
            self.names = sites['names']
        else:
            sub_idx = np.nonzero(subset)[0]
            vertices = list(range(len(sub_idx)))

            # Delaunay triangulation
            locations = sites['locations'][sub_idx, :]

            # Distance matrix
            self.names = [sites['names'][i] for i in sub_idx]

        # Delaunay triangulation

        delaunay = compute_delaunay(locations)
        v1, v2 = delaunay.toarray().nonzero()
        edges = np.column_stack((v1, v2))

        # Adjacency Matrix
        adj_mat = delaunay.tocsr()

        if crs is None:
            loc = np.asarray(sites['locations'])
            diff = loc[:, None] - loc
            dist_mat = np.linalg.norm(diff, axis=-1)
        else:
            transformer = pyproj.transformer.Transformer.from_crs(
                crs_from=crs, crs_to=pyproj.crs.CRS("epsg:4326"))
            w_locations = np.vstack(
                transformer.transform(locations[:, 0], locations[:, 1])
            ).T
            geod = geodesic.Geodesic()
            dist_mat = np.hstack([geod.inverse(location, w_locations)[:, :2] for location in w_locations])

        self.vertices = vertices
        self.edges = edges
        self.locations = locations
        self.adj_mat = adj_mat
        self.n = len(vertices)
        self.m = edges.shape[0]
        self.dist_mat = dist_mat

    def __getitem__(self, key: Literal['vertices', 'edges', 'locations', 'names', 'adj_mat', 'n', 'm', 'dist_mat']):
        if key == "vertices":
            return self.vertices
        elif key == "edges":
            return self.edges
        elif key == "locations":
            return self.locations
        elif key == "names":
            return self.names
        elif key == "adj_mat":
            return self.adj_mat
        elif key == "n":
            return self.n
        elif key == "m":
            return self.m
        elif key == "dist_mat":
            return self.dist_mat
        else:
            raise AttributeError(f"Network object has no attribute {key}")

    def __setitem__(self, key: Literal['vertices', 'edges', 'locations', 'names', 'adj_mat', 'n', 'm', 'dist_mat'], value):
        if key == "vertices":
            self.vertices = value
        elif key == "edges":
            self.edges = value
        elif key == "locations":
            self.locations = value
        elif key == "names":
            self.locations = value
        elif key == "adj_mat":
            self.adj̼_mat = value
        elif key == "n":
            self.n = value
        elif key == "m":
            self.m = value
        elif key == "dist_mat":
            self.dist_mat = value
        else:
            raise AttributeError(f"Network object has no attribute {key}")



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
    return features[sub, :, :]


def simulate_features(areas,  p_universal, p_contact, weights, inheritance,
                      p_inheritance=None, families=None,
                      missing_family_as_universal=False):
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
        inheritance (bool): Is inheritance (family membership) considered when simulating features?
        missing_family_as_universal (bool): Add family weights to the universal distribution instead
            of re-normalizing when family is absent.

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
    normed_weights = normalize_weights(weights, assignment,
                                       missing_family_as_universal=missing_family_as_universal)
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

    # Restructure features
    states = np.unique(features)
    features_states = np.zeros((n_sites, n_features, len(states)), dtype=int)
    for st in states:
        features_states[:, :, st] = np.where(features == st, 1, 0)

    # State names
    state_names = [np.nonzero(p_u_f)[0].tolist() for p_u_f in p_universal]
    applicable_states = p_universal > 0.0

    feature_names = {'external': ['f' + str(f+1) for f in range(features_states.shape[1])],
                     'internal': [f for f in range(features_states.shape[1])]}

    state_names = {'external': state_names,
                   'internal': state_names}

    return features_states, applicable_states, feature_names, state_names


EYES = {}
def sample_categorical(p, binary_encoding=False):
    """Sample from a (multidimensional) categorical distribution. The
    probabilities for every category are given by `p`

    Args:
        p (np.array): Array defining the probabilities of every category at
            every site of the output array. The last axis defines the categories
            and should sum up to 1.
            shape: (*output_dims, n_states)
    Returns
        np.array: Samples of the categorical distribution.
            shape: output_dims
                or
            shape: (output_dims, n_states)
    """
    *output_dims, n_states = p.shape

    cdf = np.cumsum(p, axis=-1)
    z = np.random.random(output_dims + [1])

    samples = np.argmax(z < cdf, axis=-1)
    if binary_encoding:
        if n_states not in EYES:
            EYES[n_states] = np.eye(n_states, dtype=bool)
        eye = EYES[n_states]
        return eye[samples]
    else:
        return samples


def assign_area(area_id, sites_sim):
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


def assign_family(fam_id, sites_sim):
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
    families = np.zeros((n_families, n_sites), dtype=bool)
    for k, z_id in enumerate(sites_in_families.values()):
        families[k, z_id] = True

    family_names = {'external': ['fam' + str(s + 1) for s in range(families.shape[0])],
                    'internal': [s for s in range(families.shape[0])]}
    return families, family_names


def simulate_weights(i_universal, i_contact,  inheritance, n_features, i_inheritance=None):
    """ Simulates weights for all features, that is the influence of global preference, inheritance and contact.
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


def read_universal_counts(feature_names, state_names, file, file_type, feature_states_file):
    """ This is a helper function to import global counts of each category of each feature,
        which then define dirichlet distributions that are used as a prior for p_global.
        Args:
            feature_names(dict): the features names (internal and external)
            state_names(dict): the category names (internal and external)
            file (str): The file location of the pseudocounts for the prior
            file_type (str): Two options:
                                ´counts_file´: The counts are given in aggregate form (per features states)
                                ´features_file´: Counts need to be extracted from a features file.
            feature_states_file (str): The csv file containing the possible state values per feature.

        Returns:
            dict, str: The counts for each state in each feature and a log string.
    """

    # Read the global counts from csv
    if file_type == 'counts_file':
        counts, feature_names_file, state_names_file = read_feature_occurrence_from_csv(file, feature_states_file)
    else:
        _, _, features, feature_names_file, state_names_file, *_ = read_features_from_csv(file, feature_states_file)
        counts = np.sum(features, axis=0)

    # # #  Sanity checks  # # #

    # ´counts´ matrix has the right shape
    n_features = len(feature_names_file['external'])
    n_states = max(len(f_states) for f_states in state_names_file['external'])
    assert counts.shape == (n_features, n_states)

    # Are the data count data?
    if not all(float(y).is_integer() for y in np.nditer(counts)):
        out = f"The data in {file} must be count data."
        raise ValueError(out)

    # Same number of features?
    assert len(feature_names['external']) == len(feature_names_file['external'])
    assert len(feature_names['internal']) == len(feature_names_file['internal'])

    # Same feature names?
    for f in range(len(feature_names['external'])):
        assert feature_names['external'][f] == feature_names_file['external'][f]
        assert feature_names['internal'][f] == feature_names_file['internal'][f]

    # Same number of categories?
    assert len(state_names['external']) == len(state_names_file['external'])
    assert len(state_names['internal']) == len(state_names_file['internal'])

    # Same category names?
    for f in range(len(state_names['external'])):
        assert state_names['external'][f] == state_names_file['external'][f]
        assert feature_names['internal'][f] == feature_names_file['internal'][f]

    # Return with log message
    log = f"Read universal counts from {file}"
    return counts, log


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


def read_inheritance_counts(family_names, feature_names, state_names, files, file_type, feature_states_file):
    """ This is a helper function to import the counts of each feature in the families,
    which define dirichlet distributions that are used as prior for p_family.

    Args:
        family_names(dict): the names of the families (internal and external)
        feature_names(dict): the features names (internal and external)
        state_names(dict): the category names (internal and external
        files(dict): path to the file locations
        file_type (str): Two options:
                            ´counts_file´: The counts are given in aggregate form (per features states)
                            ´features_file´: Counts need to be extracted from a features file.

        feature_states_file (str): The csv file containing the possible state values per feature.
    Returns:
        dict, list: The dirichlet distribution per family for each category in each feature, and the categories
        """
    n_families = len(family_names['external'])
    n_features = len(feature_names['external'])
    n_states = max([len(s) for s in state_names['external']])
    counts_all = np.zeros([n_families, n_features, n_states])
    log = str()

    for fam_idx in range(n_families):
        fam_name = family_names['external'][fam_idx]

        if fam_name not in files:
            log += f"No prior information for {fam_name}. Uniform prior used instead.\n"
            continue

        # Load counts for family ´fam_name´
        file = files[fam_name]

        if file_type == 'counts_file':
            counts, feature_names_file, state_names_file = read_feature_occurrence_from_csv(file, feature_states_file)
        else:
            _, _, features, feature_names_file, state_names_file, *_ = read_features_from_csv(file, feature_states_file)
            counts = np.sum(features, axis=0)

        counts_all[fam_idx, :, :] = counts
        log += f"Read counts for {fam_name} from {file}\n"

        # # #  Sanity checks  # # #

        if not all(float(y).is_integer() for y in np.nditer(counts)):
            out = f"The data in {file} must be count data."
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

        if len(state_names['external']) != len(state_names_file['external']) or \
                len(state_names['internal']) != len(state_names_file['internal']):
            out = "Different number of features in " + str(file) + " as in features."
            raise ValueError(out)

        for f in range(0, len(state_names['external'])):
            if state_names['external'][f] != state_names_file['external'][f]:
                out = "The external category names for feature " + str(f+1) + " in " + str(file) \
                      + " differ from those used in features."
                raise ValueError(out)

            if feature_names['internal'][f] != feature_names_file['internal'][f]:
                out = "The internal category names for " + str(f+1) + " in " + str(file) \
                      + " differ from those used in features."
                raise ValueError(out)

    return counts_all.astype(int), log


def read_geo_cost_matrix(site_names, file):
    """ This is a helper function to import the geographical cost matrix.

    Args:
        site_names (dict): the names of the sites or languages (external and internal)
        file: path to the file location

    Returns:

    """
    costs, log = read_costs_from_csv(file)
    assert set(costs.columns) == set(site_names['external'])

    # Sort the data by site names
    sorted_costs = costs.loc[site_names['external'], site_names['external']]

    cost_matrix = np.asarray(sorted_costs).astype(float)

    # Check if matrix is symmetric, if not make symmetric
    if not np.allclose(cost_matrix, cost_matrix.T):
        cost_matrix = (cost_matrix + cost_matrix.T)/2
        log += f".The cost matrix is not symmetric. It was made symmetric by averaging the original" \
               f" costs along the upper and lower triangle."
    return cost_matrix, log



