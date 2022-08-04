#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import csv
import sys
import random
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import pyproj

from sbayes.util import compute_delaunay, read_costs_from_csv, PathLike


def load_canvas(config, logger=None):
    """ This function reads sites from a csv, with the following columns:
        x: the x-coordinate
        y: the y-coordinate
        all other expected columns are defined in the config file
        logger: Logger object for info messages.
    Args:
        config(dict): config file for the simulation
    Returns:
        dict, list: a dictionary containing the location tuples (x,y), the areas and the confounders
    """
    columns = []
    filemode = 'r' if sys.version_info >= (3, 4) else 'rU'
    with open(config['canvas'], filemode) as f:
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
        identifier = csv_as_dict.pop('id')
        cluster = csv_as_dict.pop('cluster')
    except KeyError:
        raise KeyError(f"The canvas csv (\'{config['canvas']}\') must contain columns `x`, `y`, `id` and `cluster`")

    confounders = dict()
    for confounder_name in config['confounding_effects']:
        try:
            confounders[confounder_name] = csv_as_dict.pop(confounder_name)
        except KeyError:
            raise KeyError(f"The canvas csv (\'{config['canvas']}\') must contain the column \'{confounder_name}\'.")

    locations = np.zeros((len(identifier), 2))
    seq_id = []

    for i in range(len(identifier)):

        # Define location tuples
        locations[i, 0] = float(x[i])
        locations[i, 1] = float(y[i])

        # The order in the list maps name -> id and id -> name
        # name could be any unique identifier, sequential id are integers from 0 to len(name)
        seq_id.append(i)

    sites = {'locations': locations,
             'id': identifier,
             'cluster': [int(z) for z in cluster],
             'confounders': confounders}

    site_names = {'external': identifier,
                  'internal': list(range(0, len(identifier)))}
    if logger:
        logger.info(str(len(identifier)) + " locations read from " + str(config['canvas']))

    return sites, site_names


class ComputeNetwork:

    def __init__(
            self,
            sites,
            crs=None):
        """Convert a set of sites into a network.

        This function converts a set of language locations, with their attributes,
        into a network (graph). If a subset is defined, only those sites in the
        subset go into the network.

        Args:
            sites(typ.Union[dict, 'Objects']): a dict of sites with keys "locations", "id"
        Returns:
            dict: a network

        """
        if crs is not None:
            try:
                import cartopy
                if cartopy.__version__ >= '0.18.0':
                    from cartopy.geodesic import Geodesic
                else:
                    from cartopy.crs import Geodesic

            except ImportError as e:
                print("Using a coordinate reference system (crs) requires the ´cartopy´ library:")
                print("pip install cartopy")
                raise e

        # Define vertices
        vertices = sites['id']
        locations = sites['locations']
        self.names = sites['id']

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
            geod = Geodesic()
            dist_mat = np.array([geod.inverse(location, w_locations)[:, 0] for location in w_locations])

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
            self.adj_mat = value
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


EYES = {}


def sample_categorical(p, binary_encoding=False):
    """Sample from a (multidimensional) categorical distribution. The
    probabilities for every category are given by `p`

    Args:
        p (np.array): Array defining the probabilities of every category at
            every site of the output array. The last axis defines the categories
            and should sum up to 1.
            shape: (*output_dims, n_states)
        binary_encoding(bool): Return samples in binary encoding?
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


def assign_to_cluster(sites_sim):
    """ This function finds out which sites belong to a cluster and assigns cluster membership accordingly.
        Args:
            sites_sim (dict): simulates sites
        Returns:
            (np.array): the simulated clusters, boolean assignment of site to a cluster.
                shape(n_clusters, n_sites)
    """

    # Retrieve areas
    cluster = np.asarray(sites_sim['cluster'])
    cluster_ids = np.unique(cluster[cluster != 0])

    sites_in_cluster = dict()
    for z in cluster_ids:
        sites_in_cluster[z] = np.where(cluster == z)[0].tolist()

    n_cluster = len(sites_in_cluster)
    n_sites = len(sites_sim['id'])

    # Assign cluster membership
    cluster_membership = np.zeros((n_cluster, n_sites), bool)
    for k, z_id in enumerate(sites_in_cluster.values()):
        cluster_membership[k, z_id] = 1

    return cluster_membership


def assign_to_confounders(sites_sim):
    """ This function assigns sites to confounders
        Args:
            sites_sim (dict): dict with simulates sites
        Returns:
            (np.array): the simulated confounders, boolean assignment of site to confounder.
                shape(n_families, n_sites)
    """
    confounders = dict()

    # Loop through all confounders
    for k, v in sites_sim['confounders'].items():

        confounder = np.asarray(v)
        confounder_groups = np.unique(confounder[confounder is not None])

        sites_with_confounder = dict()
        for s in confounder_groups:
            sites_with_confounder[s] = np.where(confounder == s)[0].tolist()

        n_states = len(confounder_groups)
        n_sites = len(sites_sim['id'])

        # Assign membership to each of the states of the confounder
        confounder_membership = np.zeros((n_states, n_sites), bool)
        for q, s_id in enumerate(sites_with_confounder.values()):
            confounder_membership[q, s_id] = 1

        group_names = {'external': list(confounder_groups),
                       'internal': [s for s in range(len(confounder_groups))]}

        confounders[k] = {"membership": confounder_membership,
                          "names": group_names}
    return confounders


def simulate_weights(config):
    """ Simulates weights of the areal and the confounding effect on all features
    Args:
        config (dict): config file for the simulation
    Returns:
        (np.array): simulated weights for each effect
        """
    # Define alpha values which control the influence of contact (and inheritance if available) when simulating features

    alpha = [config['cluster_effect']['intensity']]
    for k, v in config['confounding_effects'].items():
        alpha.append(v['intensity'])

    weights = np.random.dirichlet(alpha, config['n_features'])
    return weights


def simulate_assignment_probabilities(config, clusters, confounders):
    """ Simulates states per feature and the assignment
     to states in the clusters and confounders
       Args:
          config(dict): The config file for the simulation
       Returns:
           (dict): The assignment probabilities (areal and confounding effect) per feature
       """
    states = []
    n_states_per_feature = []
    n_features = config['n_features']

    for k, v in config['n_states'].items():
        states.append(int(k))
        n_states_per_feature.extend([int(k)] * int(config['n_features'] * v))

    if len(n_states_per_feature) < config['n_features']:
        missing = config['n_features'] - len(n_states_per_feature)
        n_states_per_feature.extend(np.random.choice(n_states_per_feature, missing))

    random.shuffle(n_states_per_feature)

    # Simulate states
    max_states = max(n_states_per_feature)
    n_clusters = clusters.shape[0]

    # Areal effect
    # Initialize empty arrays
    cluster_effect = np.zeros((n_clusters, n_features, max_states), dtype=float)

    for feat in range(n_features):
        states_f = n_states_per_feature[feat]
        alpha_cluster_effect = np.full(shape=states_f, fill_value=config['cluster_effect']['concentration'])

        # Assignment probabilities per cluster
        for z in range(n_clusters):
            cluster_effect[z, feat, range(states_f)] = np.random.dirichlet(alpha_cluster_effect, size=1)

    p = {'cluster_effect': cluster_effect}

    # Confounding effect
    for k, v in confounders.items():
        n_groups = v['membership'].shape[0]
        p_confounder = np.zeros((n_groups, n_features, max_states), dtype=float)

        for feat in range(n_features):
            states_f = n_states_per_feature[feat]
            alpha_p_confounder = np.full(shape=states_f,
                                         fill_value=config['confounding_effects'][k]['concentration'])
            # Assignment probability per group
            for g in range(n_groups):
                p_confounder[g, feat, range(states_f)] = np.random.dirichlet(alpha_p_confounder, size=1)
        p[k] = p_confounder

    return p


def read_geo_cost_matrix(object_names: Sequence[str], file: PathLike, logger=None) -> NDArray[float]:
    """ This is a helper function to import the geographical cost matrix.

    Args:
        object_names: the names of the objects or languages (external and internal)
        file: path to the file location

    Returns:
        The symmetric cost matrix between objects.
    """
    costs = read_costs_from_csv(file, logger=logger)
    assert set(costs.columns) == set(object_names)

    # Sort the data by object names
    sorted_costs = costs.loc[object_names, object_names]

    cost_matrix = np.asarray(sorted_costs).astype(float)

    # Check if matrix is symmetric, if not make symmetric
    if not np.allclose(cost_matrix, cost_matrix.T):
        cost_matrix = (cost_matrix + cost_matrix.T)/2
        if logger:
            logger.info("The cost matrix is not symmetric. It was made symmetric by "
                        "averaging the original costs in the upper and lower triangle.")
    return cost_matrix



