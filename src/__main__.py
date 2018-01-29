# In this document we test an algorithm to identify linguistic contact zones in simulated data.
# We import the data from a PostgreSQL database.

import random
import numpy as np
import scipy.stats
import time
import pickle
import matplotlib.pyplot as plt
import csv
import scipy.sparse as sps

from src.model import compute_likelihood
from src.preprocessing import get_network, get_features, compute_feature_prob
from src.util import timeit
from src.config import *


def lookup_lh(min_size, max_size, feat_prob):
    """This function generates a lookup table of likelihoods
    :In
    - min_size: the minimum number of languages in a diffusion
    - max_size: the maximum number of languages in a diffusion
    - feat_prob: the probability of a feature to be present
    :Out
    - lookup_dict: the lookup table of likelihoods for a specific feature, sample size and presence
    """

    lookup_dict = {}
    for f in range(0, len(feat_prob)):
        lookup_dict[f] = {}
        for s in range(min_size, max_size + 1):
            lookup_dict[f][s] = {}
            for p in range(s + 1):
                lookup_dict[f][s][p] = -np.log(scipy.stats.binom_test(p, s, feat_prob[f],
                                               alternative='two-sided'))

    return lookup_dict


def compute_neighbours(zone, adj_mat):
    # return np.clip(adj_mat.dot(zone) - zone, 0, 1)
    return (adj_mat.dot(zone) - zone).astype(bool)


def propose_contact_zone(net, prev_zone, max_size):
    """This function proposes a new candidate zone in the network. The new zone differs
    from the previous one by exactly on vertex.

        :In
        - prev_zone (np.array): the previous zone, which will be modified to generate
            the new one.
        :Out
        - zone: a list of vertices that are part of the zone.
        - proposal_prob: the probability of the transition under the proposal
            distribution.
    """
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
        neighbours = compute_neighbours(prev_zone, net['adj_mat'])
        # neighbours = net['adj_mat'].dot(prev_zone)
        # neighbours -= prev_zone
        # neighbours = np.clip(neighbours, 0, 1)
        # print(sorted(neighbours))

        # Add a neigbhour to the zone.
        i_new = np.random.choice(neighbours.nonzero()[0])
        new_zone[i_new] = 1

        # Transition probability when growing
        q = 0.5 / np.sum(neighbours)

    else:
        # Remove a vertex from the zone.
        i_out = random.choice(prev_zone.nonzero()[0])
        new_zone[i_out] = 0

        # Transition probability when shrinking
        q = 0.5 / prev_size

    return new_zone, q


def transition_probability(net, prev_zone, new_zone):
    """Get the probability to move from prev_zone to new_zone under proposal
    distribution defined above."""
    prev_size = np.sum(prev_zone)
    if prev_size < np.sum(new_zone):
        # Zone was growing

        neighbours = compute_neighbours(prev_zone, net['adj_mat'])
        return 0.5 / np.sum(neighbours > 0)

    else:
        # Zone was shrinking
        return 0.5 / prev_size


@timeit
def run_metropolis_hastings(n_iter, net, feat, lh_lookup, max_size):
    # Generate a random starting point
    i_start = random.randrange(net['n'])
    zone = np.zeros(net['n'])
    zone[i_start] = 1

    zone += net['adj_mat'].dot(zone)
    zone = np.clip(zone, 0, 1)

    size = np.sum(zone)

    # Compute the likelihood of the starting zone
    lh = compute_likelihood(zone, feat, lh_lookup)

    # This dictionary stores statistics and results of the MCMC
    mcmc_stats = {'posterior': [],
                  'acceptance_ratio': []}

    # The number of accepted moves in the MCMC
    acc = 0

    # Propose a candidate diffusion
    # The candidate diffusion differs from the current diffusion
    # The proposal distribution defines how much the candidate diffusion differs

    for i in range(1, n_iter):
        # Propose a new candidate for the start location
        candidate_zone, q = propose_contact_zone(net, zone, max_size=max_size)

        # Compute back-transition probability
        q_back = transition_probability(net, zone, candidate_zone)

        # Compute the likelihood of the candidate diffusion
        lh_cand = compute_likelihood(candidate_zone, feat, lh_lookup)

        # This is the core of the MCMC: We compare the candidate to the current diffusion
        # Usually, we go for the better of the two diffusions,
        # but sometimes we decide for the candidate, even if it's worse
        a = (q_back + lh_cand) - (lh + q)

        if np.log(random.uniform(0, 1)) < a:
            zone = candidate_zone
            lh = lh_cand
            size = np.sum(zone)
            acc += 1

        mcmc_stats['posterior'].append(dict([('iteration', i),
                                             ('languages', zone.astype(bool)),
                                             ('size', size),
                                             ('log_likelihood', lh)]))

    mcmc_stats['acceptance_ratio'] = acc / n_iter
    return mcmc_stats


def dump(data, path):
    with open(path, 'wb') as dump_file:
        pickle.dump(data, dump_file)


def load_from(path):
    with open(path, 'rb') as dump_file:
        return pickle.load(dump_file)


if __name__ == "__main__":

    start_time = time.time()

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
        lh_lookup = lookup_lh(1, MAX_SIZE, feature_prob)
        dump(lh_lookup, LOOKUP_TABLE_PATH)

    else:
        # Load preprocessed data from dump files
        network = load_from(NETWORK_PATH)
        features = load_from(FEATURES_PATH)
        feature_prob = load_from(FEATURE_PROB_PATH)
        lh_lookup = load_from(LOOKUP_TABLE_PATH)

    # Tune the MCMC
    # Number of iterations of the Markov Chain while tuning
    N_ITER = 4000
    PLOT_INTERVAL = 1000

    mcmc = run_metropolis_hastings(N_ITER, network, features, lh_lookup,
                                   max_size=MAX_SIZE)

    print('Acceptance Ration: %.2f' % mcmc['acceptance_ratio'])

    locations = network['locations']
    posterior = mcmc['posterior']

    #  Write the results to a csv file
    plt.scatter(*locations.T, s=5, alpha=0.2)
    COLOR_WHEEL = [
        (0.05, 0.05, 0.1),
        (0.05, 0.4, 0.15),
        (0.15, 0.6, 0.15),
        (0.7, 0.6, 0.0),
        (0.7, 0.3, 0.15),
        (1., 0.4, 0.3),
        (0.8, 0.1, 0.6),
    ] + [(0.9, 0.0, 0.7)]*20
    with open(PATH_MCMC_RESULTS, 'w') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=';', quotechar='"',
                                   quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        result_writer.writerow(["Iterations", "Languages", "Log-lh", "size"])

        for i, sample in enumerate(posterior[::PLOT_INTERVAL]):
            result_writer.writerow([sample['iteration'], sample['languages'],
                                    sample['log_likelihood'], sample['size']])

            zone = sample['languages']
            plt.scatter(*locations[zone].T, s=20, c=COLOR_WHEEL[i])

    plt.show()
