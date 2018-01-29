# In this document we test an algorithm to identify linguistic contact zones in simulated data.
# We import the data from a PostgreSQL database.

import random
import numpy as np
import matplotlib.pyplot as plt
import csv

from src.model import compute_likelihood, lookup_log_likelihood
from src.preprocessing import get_network, get_features, compute_feature_prob
from src.util import timeit, dump, load_from
from src.config import *


def generate_initial_zone(net):
    """Generate an initial zone by sampling a random vertex and adding its
    neighbourhood.
        :In
        - net (dict): The dictionary describing the network graph
        :Out
        - zone (np.array): The generated zone as a bool array.
    """
    zone = np.zeros(net['n'])

    i_start = random.randrange(net['n'])
    zone[i_start] = 1
    zone += net['adj_mat'].dot(zone)

    return zone.astype(bool)


def get_neighbours(zone, adj_mat):
    """Compute the neighbourhood of a zone (excluding vertices from the zone itself)."""
    return (adj_mat.dot(zone) - zone).astype(bool)


def propose_global_step(net, prev_zone):
    """Globally sample a new zone to escape local modes.

    :In
        - net (dict): Network graph.
        - prev_zone (np.array): The previous zone (needed to compute the back-transition
            probability)
    :Out
        - new_zone (np.array): The proposed new contact zone.
        - q (float): The proposal probability.
        - q_back (float): The probability to transition back (new_zone -> prev_zone).
    """
    adj_mat = net['adj_mat']

    # Propose new zone like in initialization
    new_zone = generate_initial_zone(net)

    # Transition is uniform over all edges
    q = 1 / net['n']

    # Global back-probability depends on the shape of the zone
    adj_mat_zone = adj_mat[prev_zone, prev_zone]
    degree_in_zone = np.sum(adj_mat_zone, axis=0)

    if np.max(degree_in_zone) == (np.sum(prev_zone) - 1):
        q_back = q
    else:
        q_back = 0

    return new_zone, q, q_back


def propose_contact_zone(net, prev_zone, max_size, p_global=0.2):
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

    # From time to time resample globally
    if random.random() < p_global:
        return propose_global_step(net, prev_zone)

    # ...continue with local mcmc steps
    new_zone = prev_zone.copy()

    # Decide whether to grow or to shrink the zone:
    grow = (random.random() < 0.5)

    # Ensure we don't exceed size limits
    prev_size = np.sum(prev_zone)
    if prev_size <= 1:
        grow = True
    if prev_size >= max_size:
        grow = False

    if grow:
        neighbours = get_neighbours(prev_zone, net['adj_mat'])

        # Add a neighbour to the zone.
        i_new = np.random.choice(neighbours.nonzero()[0])
        new_zone[i_new] = 1

        # Transition probability when growing.
        q = 0.5 / np.sum(neighbours)

        # Back-probability (-> shrinking)
        q_back = 0.5 / (prev_size + 1)

    else:
        # Remove a vertex from the zone.
        i_out = random.choice(prev_zone.nonzero()[0])
        new_zone[i_out] = 0

        # Transition probability when shrinking.
        q = 0.5 / prev_size

        # Back-probability (-> growing)
        back_neighbours = get_neighbours(new_zone, net['adj_mat'])
        q_back = 0.5 / np.sum(back_neighbours)

    return new_zone, q, q_back


@timeit
def run_metropolis_hastings(n_iter, net, feat, lh_lookup, max_size, p_global):
    # Generate a random starting zone.
    zone = generate_initial_zone(net)
    size = np.sum(zone)

    # Compute the likelihood of the starting zone.
    lh = compute_likelihood(zone, feat, lh_lookup)

    # This dictionary stores statistics and results of the MCMC.
    mcmc_stats = {'posterior': [],
                  'acceptance_ratio': 0.}

    # The number of accepted moves in the current MCMC run.
    accepted = 0

    for i in range(1, n_iter):
        # Propose a new candidate for the start location
        candidate_zone, q, q_back = propose_contact_zone(net, zone, max_size=max_size,
                                                         p_global=p_global)

        # Compute the likelihood of the candidate zone
        lh_cand = compute_likelihood(candidate_zone, feat, lh_lookup)

        # This is the core of the MCMC: We compare the candidate to the current zone
        # Usually, we go for the better of the two zones,
        # but sometimes we decide for the candidate, even if it's worse
        a = (q_back + lh_cand) - (lh + q)

        if np.log(random.uniform(0, 1)) < a:
            zone = candidate_zone
            lh = lh_cand
            size = np.sum(zone)
            accepted += 1

        mcmc_stats['posterior'].append({'iteration': i,
                                        'languages': zone.astype(bool),
                                        'size': size,
                                        'log_likelihood': lh})

    mcmc_stats['acceptance_ratio'] = accepted / n_iter
    return mcmc_stats


if __name__ == "__main__":
    if RELOAD_DATA:
        # Get all necessary data
        # Retrieve the network from the DB
        network = get_network()
        dump(network, NETWORK_PATH)

        # Retrieve the features for all languages in the sample
        features = get_features()
        dump(features, FEATURES_PATH)

        # Compute the probability of a feature to be present/absent
        feature_prob = compute_feature_prob(features)
        dump(feature_prob, FEATURE_PROB_PATH)

        # Compute lookup tables for likelihood and direction,
        # this speeds up the processing time of the algorithm
        lh_lookup = lookup_log_likelihood(1, MAX_SIZE, feature_prob)
        dump(lh_lookup, LOOKUP_TABLE_PATH)
    else:
        # Load preprocessed data from dump files
        network = load_from(NETWORK_PATH)
        features = load_from(FEATURES_PATH)
        feature_prob = load_from(FEATURE_PROB_PATH)
        lh_lookup = load_from(LOOKUP_TABLE_PATH)

    mcmc = run_metropolis_hastings(N_ITER, network, features, lh_lookup,
                                   max_size=MAX_SIZE, p_global=P_GLOBAL)

    print('Acceptance Ration: %.2f' % mcmc['acceptance_ratio'])

    locations = network['locations']
    posterior = mcmc['posterior']

    #  Write the results to a csv file
    plt.scatter(*locations.T, s=5, alpha=0.2)
    with open(PATH_MCMC_RESULTS, 'w') as csvfile:

        result_writer = csv.writer(csvfile, delimiter=';', quotechar='"',
                                   quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        result_writer.writerow(["Iterations", "Languages", "Log-lh", "size"])

        for i, sample in enumerate(posterior[::PLOT_INTERVAL]):
            result_writer.writerow([sample['iteration'], sample['languages'],
                                    sample['log_likelihood'], sample['size']])

            zone = sample['languages']
            plt.scatter(*locations[zone].T, s=10, c=COLOR_WHEEL[i])

    plt.show()
