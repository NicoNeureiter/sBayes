# In this document we test an algorithm to identify linguistic contact zones in simulated data.
# We import the data from a PostgreSQL database.
import random
import math

import numpy as np
from igraph import Graph

from src.preprocessing import (get_network,
                               compute_feature_prob,
                               get_contact_zones,
                               simulate_background_distribution,
                               simulate_contact,
                               generate_ecdf_geo_likelihood, precompute_feature_likelihood)
from src.model import (compute_feature_likelihood,
                       compute_geo_likelihood_particularity)
from src.plotting import plot_zones, plot_posterior
from src.util import timeit, dump, load_from, get_neighbours, grow_zone
from src.config import *


def sample_initial_zones(net, nr_zones):
    """This function generates <nr_zones> initial zones by sampling random vertices and adding their
    neighbourhood.

    Args:
        net (dict): A dictionary comprising all sites of the network.
        nr_zones (int): Number of zones.

    Returns:
        dict: A dictionary comprising all generated zones, each zone is a boolean np.array.
    """
    n = net['adj_mat'].shape[0]
    already_in_zone = np.zeros(n, bool)
    init_zones = {}

    for i in range(nr_zones):
        g = grow_zone(MIN_SIZE, net, already_in_zone)
        init_zones[i] = g[0]
        already_in_zone += g[1]

    return init_zones


def grow_step(zone, net, already_in_zone):
    """This function increases the size of a contact zone.
    Args:
        zone (np.array: The current contact zone (boolean array)
        net (dict): A dictionary comprising all sites of the network.
        already_in_zone (np.array): All nodes in the network already assigned to a zone (boolean array)

    Returns:
        np.array, np.array, float, float: The new zone and all nodes already assigned to a zone (boolean arrays),
        the transition probability and the back probability

    """
    adj_mat = net['adj_mat']
    new_zone = zone.copy()

    zone_idx = zone.nonzero()[0]
    size = len(zone_idx)

    neighbours = get_neighbours(zone, already_in_zone, adj_mat)

    # Add a neighbour to the zone
    try:
        i_new = random.choice(neighbours.nonzero()[0])
    except IndexError as e:
        print(zone_idx)
        print(neighbours.nonzero())
        raise e

    new_zone[i_new] = already_in_zone[i_new] = 1

    new_size = size + 1

    # Transition probability when growing
    q = 1 / np.count_nonzero(neighbours)
    if size > 1:
        q /= 2

    # Back-probability (-> shrinking)
    q_back = 1 / new_size

    if new_size < MAX_SIZE:
        q_back /= 2

    return new_zone, q, q_back


def shrink_step(zone, net, already_in_zone):
    """This function decreases the size of a contact zone.
    Args:
        zone (np.array: The current contact zone (boolean array)
        net (dict): A dictionary comprising all sites of the network
        already_in_zone (np.array): All nodes in the network already assigned to a zone (boolean array)

    Returns:
        np.array, np.array, float, float: The new zone and all nodes already assigned to a zone (boolean arrays),
        the transition probability and the back probability

    """

    adj_mat = net['adj_mat']
    new_zone = zone.copy()

    zone_idx = zone.nonzero()[0]
    size = len(zone_idx)

    removal_candidates = zone_idx

    # Remove a vertex from the zone.
    i_out = random.choice(removal_candidates)
    new_zone[i_out] = already_in_zone[i_out] = 0
    new_size = size - 1

    # Transition probability when shrinking.
    q = 1 / len(removal_candidates)
    if size < MAX_SIZE:
        q /= 2

    # Back-probability (-> growing)
    back_neighbours = get_neighbours(new_zone, already_in_zone, adj_mat)
    q_back = 1 / np.count_nonzero(back_neighbours)

    if new_size > 1:
        q_back /= 2

    return new_zone, q, q_back


def swap_step(zone, net, already_in_zone):
    """Propose an MCMC transition by removing a vertex from the zone and adding another from the
    neighbourhood.

    Args:
        zone (np.array): The current contact zone (boolean).
        net (dict): A dictionary comprising all sites of the network.
        already_in_zone (np.array): All nodes in the network already assigned to a zone (boolean array).

    Returns:
        np.array: The proposed new contact zone.
        float: The proposal probability.
        float: The probability to transition back (new_zone -> prev_zone)
    """

    adj_mat = net['adj_mat']
    new_zone = zone.copy()

    # Compute the neighbourhood
    neighbours = get_neighbours(zone, already_in_zone, adj_mat)
    neighbours_idx = neighbours.nonzero()[0]

    # Add a neighbour to the zone

    try:
        i_new = random.choice(neighbours_idx)
    except IndexError as e:
        print(zone.nonzero())
        print(neighbours.nonzero())
        raise e

    new_zone[i_new] = already_in_zone[i_new] = 1

    zone_idx = new_zone.nonzero()[0]
    removal_candidates = zone_idx

    # Remove a vertex from the zone.
    i_out = random.choice(removal_candidates)
    new_zone[i_out] = already_in_zone[i_out] = 0

    back_neighbours = get_neighbours(zone, already_in_zone, adj_mat)
    q = 1. / np.count_nonzero(neighbours)
    q_back = 1. / np.count_nonzero(back_neighbours)

    return new_zone, q, q_back


def propose_contact_zone(zone, net, already_in_zone):
    """This function proposes a new candidate zone in the network. The new zone differs
    from the previous one by exactly one vertex. An exception are global update steps, which
    are performed with probability p_global and should avoid getting stuck in local
    modes.

    Args:
        zone (np.array): the input zone, which will be modified to generate the new one.
        net (dict): A dictionary comprising all sites of the network.
        already_in_zone (np.array): All nodes in the network already assigned to a zone (boolean array).

    Returns:
        np.array, float, float: The proposed new contact zone, the proposal probability and the back-probability.

    """

    # Decide whether to swap
    if random.random() < P_SWAP:
        return swap_step(zone, net, already_in_zone)

    # Decide whether to grow or to shrink the zone:
    grow = (random.random() < 0.5)

    # Ensure we don't exceed the size limits
    prev_size = np.count_nonzero(zone)
    if prev_size <= MIN_SIZE:
        grow = True
    if prev_size >= MAX_SIZE:
        grow = False

    if grow:
        return grow_step(zone, net, already_in_zone)
    else:
        return shrink_step(zone, net, already_in_zone)


@timeit
def run_metropolis_hastings(net, n_samples, n_steps, feat, lh_lookup, plot_samples=False):
    """ Generate samples according to the likelihood defined in lh_lookup and model.py.
    A sample is defined by a set of vertices (represented by a binary vector).

    args:
        net (dict): The full network containing all sites.
        n_samples (int): The number of samples to be generated.
        n_steps (int): The number of MCMC-steps for every sample.
        feat (np.array): The (binary) feature matrix.
        lh_lookup (dict): Lookup table for more efficient computation of the likelihood.

    kwargs:
        plot_samples (bool): Plot every sampled zone.

    return:
        np.array: generated samples
    """

    # This dictionary stores statistics and results of the MCMC.
    mcmc_stats = {'lh': {'per_zone': {},
                         'aggregated': []},
                  'samples': {'per_zone': {},
                              'aggregated': []},
                  'acceptance_ratio': 0.}

    # Each of the zones gets their own sub-dicts
    for n in range(NUMBER_PARALLEL_ZONES):
        mcmc_stats['lh']['per_zone'][n] = []
        mcmc_stats['samples']['per_zone'][n] = []

    accepted = 0

    # Define the impact of the geo_weigth
    geo_weight = GEO_LIKELIHOOD_WEIGHT * feat.shape[1]

    # Generate random initial zones
    zones = sample_initial_zones(net, NUMBER_PARALLEL_ZONES)
    lh = {}

    for z in zones:
        lh[z] = compute_feature_likelihood(zones[z], feat, lh_lookup)
        lh[z] += geo_weight * compute_geo_likelihood_particularity(zones[z], net, ecdf_geo,
                                                                   subgraph_type="delaunay")

    for i_sample in range(n_samples):

        if RESTART_CHAIN:
            # Restart the chain from a random location after every sample
            zones = sample_initial_zones(net, NUMBER_PARALLEL_ZONES)

            for z in zones:
                lh[z] = compute_feature_likelihood(zones[z], feat, lh_lookup)
                lh[z] += geo_weight * compute_geo_likelihood_particularity(zones[z], net,
                                                                           ecdf_geo,
                                                                           subgraph_type="delaunay")

        for k in range(n_steps):
            for z in zones:

                # Propose a candidate zone
                in_zone = sum(zones.values()).astype(bool)
                candidate_zone, q, q_back = propose_contact_zone(zones[z], net, in_zone)

                # Compute the likelihood of the candidate zone
                lh_cand = compute_feature_likelihood(candidate_zone, feat, lh_lookup)
                lh_geo = geo_weight * compute_geo_likelihood_particularity(
                    candidate_zone, net, ecdf_geo, subgraph_type="delaunay")

                lh_cand += lh_geo

                # This is the core of the MCMC: We compare the candidate to the current zone
                # Usually, we go for the better of the two zones,
                # but sometimes we decide for the candidate, even if it's worse

                a = (lh_cand - lh[z]) + math.log(q_back / q)
                if math.log(random.random()) < a:
                    zones[z] = candidate_zone
                    lh[z] = lh_cand
                    accepted += 1

        for z in zones:

            mcmc_stats['lh']['per_zone'][z].append(lh[z])
            mcmc_stats['samples']['per_zone'][z].append(zones[z])

            mcmc_stats['lh']['aggregated'] += lh[z]

        mcmc_stats['lh']['aggregated'] = sum(lh.values())
        mcmc_stats['samples']['aggregated'] = sum(zones.values()).astype(bool)

        print('Log-Likelihood:        %.2f' % mcmc_stats['lh']['aggregated'])
        print('Size:                  %i' % np.count_nonzero(mcmc_stats['samples']['aggregated']))
        print('Acceptance ratio:  %.2f' % (accepted / ((i_sample + 1) * n_steps)))
        print()

        if plot_samples:
            plot_zones(zones, net)

    mcmc_stats['acceptance_ratio'] = accepted / (n_steps * n_samples)
    return mcmc_stats


if __name__ == "__main__":
    if RELOAD_DATA:

        # Get all necessary data
        # Retrieve the network from the DB
        network = get_network()
        dump(network, NETWORK_PATH)

        # Retrieve the contact zones
        all_zones = [1,2,3,4,5,6,7,8,9,10]
        contact_zones = get_contact_zones(all_zones)
        dump(contact_zones, CONTACT_ZONES_PATH)

        # Simulate distribution of features
        features_bg = simulate_background_distribution(TOTAL_N_FEATURES, len(network['vertices']))
        dump(features_bg, FEATURES_BG_PATH)

        # Simulate contact zones
        features = simulate_contact(N_CONTACT_FEATURES, features_bg, P_CONTACT, contact_zones)
        dump(features, FEATURES_PATH)

        # Compute the probability of a feature to be present or absent
        feature_prob = compute_feature_prob(features)
        dump(feature_prob, FEATURE_PROB_PATH)

        # Compute a lookup table for likelihood
        # this speeds up the processing time of the algorithm
        lh_lookup = precompute_feature_likelihood(1, MAX_SIZE, feature_prob)
        dump(lh_lookup, LOOKUP_TABLE_PATH)

        # Generate an empirical distribution for estimating the geo-likelihood
        ecdf_geo = generate_ecdf_geo_likelihood(network, min_n=MIN_SIZE, max_n=MAX_SIZE,
                                                nr_samples=SAMPLES_PER_ZONE_SIZE, plot=False)
        dump(ecdf_geo, ECDF_GEO_PATH)

    else:
        # Load preprocessed data from dump files
        network = load_from(NETWORK_PATH)
        contact_zones = load_from(CONTACT_ZONES_PATH)
        features_bg = load_from(FEATURES_BG_PATH)
        features = load_from(FEATURES_PATH)
        features_prob = load_from(FEATURE_PROB_PATH)
        lh_lookup = load_from(LOOKUP_TABLE_PATH)
        ecdf_geo = load_from(ECDF_GEO_PATH)

    vertices = network['vertices']
    locations = network['locations']
    adj_mat = network['adj_mat']

    mcmc_stats = run_metropolis_hastings(network, N_SAMPLES, N_STEPS, features,
                                         lh_lookup, plot_samples=False)
    #
    # from src.sampling.zone_sampling import ZoneMCMC
    # zone_sampler = ZoneMCMC(network, features, N_STEPS, MIN_SIZE, MAX_SIZE,
    #                         P_TRANSITION_MODE, GEO_LIKELIHOOD_WEIGHT, lh_lookup,
    #                         n_zones=NUMBER_PARALLEL_ZONES, ecdf_geo=ecdf_geo,
    #                         restart_chain=RESTART_CHAIN,
    #                         simulated_annealing=SIMULATED_ANNEALING, plot_samples=False)
    #
    # samples = zone_sampler.generate_samples(N_SAMPLES)
    # mcmc_stats = zone_sampler.statistics

    # print('Acceptance Ratio:     %.2f' % mcmc_stats['acceptance_ratio'])
    # print('Log-Likelihood:       %.2f' % mcmc_stats['sample_likelihoods'][-1])
    # print('Size:                 %r' % np.count_nonzero(samples[-1], axis=-1))
