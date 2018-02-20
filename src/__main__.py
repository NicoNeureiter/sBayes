# In this document we test an algorithm to identify linguistic contact zones in simulated data.
# We import the data from a PostgreSQL database.
import random
import math

import numpy as np
from igraph import Graph

from src.preprocessing import get_network, \
                              compute_feature_prob, \
                              get_contact_zones, \
                              simulate_background_distribution, \
                              simulate_contact
from src.model import lookup_log_likelihood, \
                      compute_geo_likelihood
from src.plotting import plot_zone, plot_posterior
from src.util import timeit, dump, load_from, get_neighbours
from src.config import *

if LL_MODE == 'generative':
    from src.model import compute_likelihood_generative as compute_likelihood
else:
    from src.model import compute_likelihood


def sample_initial_zone(net, lookup_table=None):
    """Generate an initial zone by sampling a random vertex and adding its
    neighbourhood.
        :In
        - net (dict): The dictionary describing the network graph
        :Out
        - zone (np.array): The generated zone as a bool array.
    """
    if lookup_table is not None:
        zone_as_list = random.choice(lookup_table)
        zone = np.array(zone_as_list)
    else:
        zone = np.zeros(net['n'])

        v = random.randrange(net['n'])
        zone[v] = 1
        zone += net['adj_mat'].dot(zone)
        zone = zone.astype(bool)

    return zone


def grow_step(zone, net, max_size):
    """Sample a nei

    Args:
        zone:
        net:
        max_size:

    Returns:

    """
    adj_mat = net['adj_mat']
    new_zone = zone.copy()

    zone_idx = zone.nonzero()[0]
    size = len(zone_idx)

    neighbours = get_neighbours(zone, adj_mat)

    # Add a neighbour to the zone.
    i_new = random.choice(neighbours.nonzero()[0])
    new_zone[i_new] = 1
    new_size = size + 1

    # Transition probability when growing.
    q = 1 / np.count_nonzero(neighbours)
    if size > 1:
        q /= 2

    # Back-probability (-> shrinking)
    q_back = 1 / new_size

    if new_size < max_size:
        q_back /= 2

    return new_zone, q, q_back


def shrink_step(zone, net, max_size):
    G = net['graph']
    adj_mat = net['adj_mat']
    new_zone = zone.copy()

    zone_idx = zone.nonzero()[0]
    size =len(zone_idx)

    # G_zone = G.induced_subgraph(zone_idx)
    # assert G_zone.is_connected()
    #
    # cut_vs_idx = G_zone.cut_vertices()
    # if len(cut_vs_idx) == size:
    #     return None
    #
    # cut_vertices = G_zone.vs[cut_vs_idx]['name']
    # removal_candidates = [v for v in zone_idx if v not in cut_vertices]

    removal_candidates = zone_idx

    # Remove a vertex from the zone.
    i_out = random.choice(removal_candidates)
    new_zone[i_out] = 0
    new_size = size - 1

    # Transition probability when shrinking.
    q = 1 / size
    if size < max_size:
        q /= 2

    # Back-probability (-> growing)
    back_neighbours = get_neighbours(new_zone, adj_mat)
    q_back = 1 / np.count_nonzero(back_neighbours)

    if new_size > 1:
        q_back /= 2

    return new_zone, q, q_back


def swap_step(zone, net, max_swaps=5):
    G = net['graph']
    adj_mat = net['adj_mat']
    new_zone = zone.copy()


    # Compute the neighbourhood
    neighbours = get_neighbours(zone, adj_mat)
    neighbours_idx = neighbours.nonzero()[0]
    n_neighbours = len(neighbours_idx)

    # Add a neighbour to the zone
    i_new = random.choice(neighbours_idx)
    new_zone[i_new] = 1


    zone_idx = new_zone.nonzero()[0]
    size = len(zone_idx)

    # # # Compute cut_vertices (can be removed while keeping zone connected)
    # G_zone = G.induced_subgraph(zone_idx)
    # assert G_zone.is_connected()
    #
    # cut_vs_idx = G_zone.cut_vertices()
    # if len(cut_vs_idx) == size:
    #     return None
    #
    # cut_vertices = G_zone.vs[cut_vs_idx]['name']
    # removal_candidates = [v for v in zone_idx if v not in cut_vertices]

    removal_candidates = zone_idx

    # Remove a vertex from the zone.
    i_out = random.choice(removal_candidates)
    new_zone[i_out] = 0

    if size-1 != np.count_nonzero(new_zone):
        print('Zone:', )
        print('   ', i_new)


    back_neighbours = get_neighbours(zone, adj_mat)
    q = 1. / np.count_nonzero(neighbours)
    q_back = 1. / np.count_nonzero(back_neighbours)

    return new_zone, q, q_back


def propose_contact_zone(net, prev_zone, max_size, p_global=0.):
    """This function proposes a new candidate zone in the network. The new zone differs
    from the previous one by exactly on vertex. An exception are global update steps
    the are performed with probability p_global and should avoid getting stuck in local
    modes.

    :In
        - net (dict): Network graph.
        - prev_zone (np.array): the previous zone, which will be modified to generate
            the new one.
        - max_size (int): upper bound on the number of languages in a zone.
        - p_global (float): probability (frequency) of global update steps. (in [0,1])
    :Out
        - new_zone (np.array): The proposed new contact zone.
        - q (float): The proposal probability.
        - q_back (float): The probability to transition back (new_zone -> prev_zone)
    """
    if random.random() < 0.5:
        return swap_step(prev_zone, net)

    # Continue with local mcmc steps...
    # Decide whether to grow or to shrink the zone:
    grow = (random.random() < 0.5)

    # Ensure we don't exceed size limits
    prev_size = np.count_nonzero(prev_zone)
    if prev_size <= MIN_SIZE:
        grow = True
    if prev_size >= max_size:
        grow = False

    if grow:
        return grow_step(prev_zone, net, max_size)
    else:
        result = shrink_step(prev_zone, net, max_size)
        if result:
            return result
        else:
            # No way to shrink while keeping the zone connected
            return grow_step(prev_zone, net, max_size)


@timeit
def run_metropolis_hastings(net, n_samples, n_steps, feat, lh_lookup, max_size,
                            plot_samples=False):
    """ Generate samples according to the likelihood defined in lh_lookup and model.py.
     A sample is defined by a set of vertices (represented by a binary vector).

    args:
        net (dict): Network dictionary containing infos about the graph.
        n_samples (int): The number of samples to be generated.
        n_steps (int): The number of MCMC-steps for every sample.
        feat (np.array): The (binary) feature matrix.
        lh_lookup (dict): Lookup table for more efficient computation of the likelihood.
        max_size (int): The maximum size of a cluster.

    kwargs:
        plot_samples (bool): Plot every sampled zone (many plots!).
    return:
        np.array: generated samples
    """

    # This dictionary stores statistics and results of the MCMC.
    mcmc_stats = {'likelihoods': [], 'acceptance_ratio': 0.}

    accepted = 0
    samples = np.zeros((n_samples, net['n'])).astype(bool)
    temperature = 1.

    # Generate a random starting zone and comput its log-likelihood
    zone = sample_initial_zone(net)
    ll = compute_likelihood(zone, feat, lh_lookup)
    if USE_GEO_LL:
        ll += compute_geo_likelihood(zone, net)

    for i_sample in range(n_samples):

        # zone = sample_initial_zone(net)
        # # llh = compute_likelihood(zone, feat, lh_lookup)
        # llh = 1.

        for k in range(n_steps):
            candidate_zone, q, q_back = propose_contact_zone(net, zone,
                                                             max_size=max_size)

            # q = q_back = 1.
            # for _ in range(5):
            #
            #     # Propose a new candidate for the start location
            #     candidate_zone, q_, q_back_ = propose_contact_zone(net, prev_zone,
            #                                                        max_size=max_size)
            #     prev_zone = candidate_zone
            #     q *= q_
            #     q_back *= q_back_

            # Compute the likelihood of the candidate zone

            ll_cand = compute_likelihood(candidate_zone, feat, lh_lookup)

            if USE_GEO_LL:
                # Assign a likelihood to the geographic distribution of the languages
                # in the zone. Should ensure more connected zones.
                ll_cand += compute_geo_likelihood(candidate_zone, net)

            if SIMULATED_ANNEALING:
                # Simulated annealing: Scale all LL values to one in the beginning,
                # then converge to the true likelihood.
                temperature = ((k + 1) / n_steps) ** 2

            # This is the core of the MCMC: We compare the candidate to the current zone
            # Usually, we go for the better of the two zones,
            # but sometimes we decide for the candidate, even if it's worse
            a = (ll_cand - ll) * temperature + math.log(q_back / q)

            if math.log(random.random()) < a:
                zone = candidate_zone
                ll = ll_cand
                accepted += 1

        samples[i_sample] = zone
        mcmc_stats['likelihoods'].append(ll)

        print('Log-Likelihood:        %.2f' % ll)
        print('Size:                  %i' % np.count_nonzero(zone))
        print('Acceptance ratio:  %.2f' % (accepted / ((i_sample+1) * n_steps)))
        print()

        if plot_samples:
            plot_zone(zone, net)

    mcmc_stats['acceptance_ratio'] = accepted / (n_steps * n_samples)

    return samples, mcmc_stats


if __name__ == "__main__":
    if RELOAD_DATA:

        # Get all necessary data
        # Retrieve the network from the DB
        network = get_network()
        dump(network, NETWORK_PATH)

        # Retrieve the contact zones
        contact_zones = get_contact_zones()
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

        # Compute lookup tables for likelihood
        # this speeds up the processing time of the algorithm
        lh_lookup = lookup_log_likelihood(1, MAX_SIZE, feature_prob)
        dump(lh_lookup, LOOKUP_TABLE_PATH)

    else:
        # Load preprocessed data from dump files
        network = load_from(NETWORK_PATH)
        contact_zones = load_from(CONTACT_ZONES_PATH)
        features_bg = load_from(FEATURES_BG_PATH)
        features = load_from(FEATURES_PATH)
        features_prob = load_from(FEATURE_PROB_PATH)
        lh_lookup = load_from(LOOKUP_TABLE_PATH)

    vertices = network['vertices']
    locations = network['locations']
    adj_mat = network['adj_mat']

    G = Graph(network['n'])
    G.vs['name'] = list(range(network['n']))
    for u, v in zip(*adj_mat.nonzero()):
        G.add_edge(u, v)

    network['graph'] = G

    print(network['n'])

    samples, mcmc_stats = run_metropolis_hastings(network, N_SAMPLES, N_STEPS,
                                                  features, lh_lookup,
                                                  max_size=MAX_SIZE,
                                                  plot_samples=True )

    likelihood = np.asarray(mcmc_stats['likelihoods'])

    # n_clusters = 12
    # k_means = KMeans(n_clusters)
    #
    # # sample_means = (samples / samples.sum(axis=1)[:, None]).dot(locations)
    # # cluster_idx = k_means.fit_predict(sample_means)
    #
    # cluster_idx = k_means.fit_predict(samples / samples.sum(axis=1)[:, None])
    #
    # cluster_weights = np.ones(N_SAMPLES)
    #
    # plt.scatter(*locations.T, lw=0, alpha=0.2, s=4)
    # for i in range(n_clusters):
    #     mean_lh = np.mean(likelihood[cluster_idx == i])
    #     freq = np.sum(cluster_idx == i)
    #     cluster_weights[cluster_idx == i] = (mean_lh / freq)
    #     print(np.sum(cluster_idx==i))
    #     cluster_vertices = samples[cluster_idx==i].sum(axis=0).astype(bool)
    #
    #     plt.scatter(*locations[cluster_vertices].T, lw=0, s=10)
    #
    # plt.show()
    #
    # cluster_weights /= np.mean(cluster_weights)
    # print(np.unique(cluster_weights))
    #
    #
    # sizes = samples.astype(bool).sum(axis=1)
    # plt.hist(sizes, bins=20)
    # plt.show()

    print('Acceptance Ratio: %.2f' % mcmc_stats ['acceptance_ratio'])

    plot_posterior(samples, adj_mat, locations)

    # likelihood /= np.mean(likelihood)
    # plot_posterior(samples, adj_mat, locations, weights=likelihood)
