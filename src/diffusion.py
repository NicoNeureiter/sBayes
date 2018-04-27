import random
import numpy as np
import math
import matplotlib.pyplot as plt

from itertools import chain
from src.config import *
from src.preprocessing import get_network
from src.plotting import get_colors
from src.model import compute_feature_likelihood, compute_geo_likelihood_particularity


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


def grow_zone(size, net, already_in_zone):
    """ This function grows a zone of size <size> excluding any of the nodes in <already_in_zone>.
    Args:
        size (int): The number of nodes in the zone.
        net (dict): A dictionary comprising all sites of the network.
        already_in_zone (np.array): All nodes in the network already assigned to a zone (boolean array)

    Returns:
        (np.array, np.array): The new zone (boolean), all nodes in the network already assigned to a zone (boolean)

    """
    n = net['adj_mat'].shape[0]

    # Initialize the zone
    zone = np.zeros(n).astype(bool)

    # Get all vertices that already belong to a zone (occupied_n) and all free vertices (free_n)
    occupied_n = np.nonzero(already_in_zone)[0]
    free_n = set(range(n)) - set(occupied_n)
    i = random.sample(free_n, 1)[0]
    zone[i] = already_in_zone[i] = 1

    for _ in range(size-1):

        neighbours = get_neighbours(zone, already_in_zone, net['adj_mat'])
        # Add a neighbour to the zone
        i_new = random.choice(neighbours.nonzero()[0])
        zone[i_new] = already_in_zone[i_new] = 1

    return zone, already_in_zone


def sample_initial_zones(net, nr_zones, min_size=MIN_SIZE):
    """This function generates <nr_zones> initial zones by sampling random vertices and adding their
    neighbourhood.

    Args:
        net (dict): A dictionary comprising all sites of the network.
        nr_zones (int): Number of zones.
        min_size (int): The minimum size of a zone

    Returns:
        dict: A dictionary comprising all generated zones, each zone is a boolean np.array.
    """
    n = net['adj_mat'].shape[0]
    already_in_zone = np.zeros(n).astype(bool)
    init_zones = {}

    for i in range(nr_zones):
        g = grow_zone(min_size, net, already_in_zone)
        init_zones[i] = g[0]
        already_in_zone += g[1]

    return init_zones


def grow_step(zone, net, already_in_zone, max_size=MAX_SIZE):
    """This function increases the size of a contact zone.
    Args:
        zone (np.array: The current contact zone (boolean array)
        net (dict): A dictionary comprising all sites of the network.
        already_in_zone (np.array): All nodes in the network already assigned to a zone (boolean array)
        max_size (int): Maximum number of nodes in a zone
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
    i_new = random.choice(neighbours.nonzero()[0])

    new_zone[i_new] = already_in_zone[i_new] = 1

    new_size = size + 1

    # Transition probability when growing
    q = 1 / np.count_nonzero(neighbours)
    if size > 1:
        q /= 2

    # Back-probability (-> shrinking)
    q_back = 1 / new_size

    if new_size < max_size:
        q_back /= 2

    return new_zone, q, q_back


def shrink_step(zone, net, already_in_zone, max_size=MAX_SIZE):
    """This function decreases the size of a contact zone.
    Args:
        zone (np.array: The current contact zone (boolean array)
        net (dict): A dictionary comprising all sites of the network
        already_in_zone (np.array): All nodes in the network already assigned to a zone (boolean array)
        max_size (int): the maximum number of nodes in a zone
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
    if size < max_size:
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
    i_new = random.choice(neighbours_idx)
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


def propose_contact_zone(zone, net, already_in_zone, p_swap=P_SWAP, min_size=MIN_SIZE, max_size=MAX_SIZE):
    """This function proposes a new candidate zone in the network. The new zone differs
    from the previous one by exactly one vertex. An exception are global update steps, which
    are performed with probability p_global and should avoid getting stuck in local
    modes.

    Args:
        zone (np.array): the input zone, which will be modified to generate the new one.
        net (dict): A dictionary comprising all sites of the network.
        already_in_zone (np.array): All nodes in the network already assigned to a zone (boolean array).
        p_swap(float): the frequency of 'swap' steps
        min_size: the minimum size of a zone
        max_size: the maximum size of a zone

    Returns:
        np.array, float, float: The proposed new contact zone, the proposal probability and the back-probability.

    """

    # Decide whether to swap
    if random.random() < p_swap:
        return swap_step(zone, net, already_in_zone)

    # Decide whether to grow or to shrink the zone:
    grow = (random.random() < 0.5)

    # Ensure we don't exceed the size limits
    prev_size = np.count_nonzero(zone)
    if prev_size <= min_size:
        grow = True
    if prev_size >= max_size:
        grow = False

    if grow:
        return grow_step(zone, net, already_in_zone)
    else:
        return shrink_step(zone, net, already_in_zone)


def plot_zones(zones, net):
    """ This function plots the contact zones proposed by the MCMC

    Args:
        zones (np.array): The current zone (boolean array).
        net (dict): The full network containing all sites.
    """

    # Initialize plot
    fig, ax = plt.subplots()
    col = get_colors()
    all_sites = net['locations']
    size = 4
    bg = ax.scatter(*all_sites.T, s=size, color=col['zones']['background_nodes'])
    zo = []

    for z in zones:
        zo.append(ax.scatter(*all_sites[zones[z]].T, s=size * 6, color=col['zones']['in_zones'][int(z)]))

    # Remove axes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend
    ax.legend([bg, zo[4]], ['All sites', 'Sites in proposed contact zones'], frameon=False, fontsize=10)
    plt.show()


network = get_network()

# Test initial zones ---- SUCCESSFUL
zones = sample_initial_zones(network, 5)
plot_zones(zones, network)
print(zones)

# Test: grow step -------- SUCCESSFUL
for z in zones:
    in_zone = sum(zones.values()).astype(bool)
    zones[z] = grow_step(zones[z], network, in_zone)[0]
plot_zones(zones, network)

# Test: shrink step ------ SUCCESSFUL
for z in zones:
    in_zone = sum(zones.values()).astype(bool)
    zones[z] = shrink_step(zones[z], network, in_zone)[0]
plot_zones(zones, network)

# Test: swap step ------ SUCCESSFUL
for z in zones:
    in_zone = sum(zones.values()).astype(bool)
    zones[z] = swap_step(zones[z], network, in_zone)[0]
plot_zones(zones, network)

# Test: propose_contact_zone ------
for z in zones:
    in_zone = sum(zones.values()).astype(bool)
    zones[z] = propose_contact_zone(zones[z], network, in_zone)[0]
plot_zones(zones, network)


def run_metropolis_hastings(net, n_samples, n_steps, feat, lh_lookup, plot_samples=False, n_zones=NUMBER_PARALLEL_ZONES,
                            geo_weight=GEO_LIKELIHOOD_WEIGHT, ecdf_geo):
    """ Generate samples according to the likelihood defined in lh_lookup and model.py.
    A sample is defined by a set of vertices (represented by a binary vector).

    args:
        net (dict): The full network containing all sites.
        n_samples (int): The number of samples to be generated.
        n_steps (int): The number of MCMC-steps for every sample.
        feat (np.array): The (binary) feature matrix.
        lh_lookup (dict): Lookup table for more efficient computation of the likelihood.
        n_zones (int): the number of parallel zones
        geo_weigth (float): the weigth of the geo-likelihood\
        ecdf_geo (dict): the empirical geo-likelihood

    kwargs:
        plot_samples (bool): Plot every sampled zone.

    return:
        np.array: generated samples
    """

    # This dictionary stores statistics and results of the MCMC.
    mcmc_stats = {'lh': {'per_zone': [],
                         'aggregated': []},
                  'samples': {'per_zone': [],
                              'aggregated': []},
                  'acceptance_ratio': 0.}

    # Each of the zones gets their own sub-dicts
    for n in n_zones:
        mcmc_stats['lh']['per_zone'][n] = []
        mcmc_stats['samples']['per_zone'][n] = []

    accepted = 0

    # Define the impact of the geo_weigth
    geo_weight = geo_weight* feat.shape[1]

    # Generate random initial zones
    zones = sample_initial_zones(net, n_zones)
    lh = {}

    for z in zones:
        lh[z] = compute_feature_likelihood(zones[z], feat, lh_lookup)
        lh[z] += geo_weight * compute_geo_likelihood_particularity(zones[z], net, ecdf_geo, subgraph_type="mst")

    for i_sample in range(n_samples):

        if RESTART_CHAIN:
            # Restart the chain from a random location after every sample
            zones = sample_initial_zones(net, n_zones)

            for z in zones:
                lh[z] = compute_feature_likelihood(zones[z], feat, lh_lookup)
                lh[z] += geo_weight * compute_geo_likelihood_particularity(zones[z], net, ecdf_geo, subgraph_type="mst")

        for k in range(n_steps):
            for z in zones:

                # Propose a candidate zone
                in_zone = sum(zones.values()).astype(bool)
                candidate_zone, q, q_back = propose_contact_zone(zones[z], net, in_zone)

                # Compute the likelihood of the candidate zone
                lh_cand = compute_feature_likelihood(candidate_zone, feat, lh_lookup)
                lh_cand += geo_weight * compute_geo_likelihood_particularity(candidate_zone, net, ecdf_geo, subgraph_type="mst")

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
            mcmc_stats['sample']['per_zone'][z].append(zones[z])

            mcmc_stats['lh']['aggregated'] += lh[z]
            mcmc_stats['sample']['aggregated'] += zones[z]

        print('Log-Likelihood:        %.2f' % mcmc_stats['lh']['aggregated'])
        print('Size:                  %i' % np.count_nonzero(mcmc_stats['sample']['aggregated']))
        print('Acceptance ratio:  %.2f' % (accepted / ((i_sample + 1) * n_steps)))
        print()

        if plot_samples:
             plot_zones(zones, net)

    mcmc_stats['acceptance_ratio'] = accepted / (n_steps * n_samples)
    return mcmc_stats
