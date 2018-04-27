#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import itertools
import datetime

from src.util import dump
from src.preprocessing import (get_network,
                               generate_ecdf_geo_likelihood,
                               get_contact_zones,
                               simulate_background_distribution,
                               simulate_contact, compute_feature_prob,
                               precompute_feature_likelihood)
from src.sampling.zone_sampling import ZoneMCMC


if __name__ == '__main__':

    # Configuration (see config file)
    now = datetime.datetime.now().__str__().rsplit('.')[0]
    TEST_RESULTS_SAMPLING_DIRECTORY = 'data/results/test/{experiment}/'.format(experiment=now)
    TEST_RESULTS_SAMPLING_PATH = TEST_RESULTS_SAMPLING_DIRECTORY + 'sampling_e{e}_a{a}_m{m}_{run}.pkl'

    # TODO mkdir

    MIN_SIZE = 5
    MAX_SIZE = 100
    SAMPLES_PER_ZONE_SIZE = 10000

    TOTAL_N_FEATURES = 30
    P_SUCCESS_MIN = 0.05
    P_SUCCESS_MAX = 0.7
    GEO_LIKELIHOOD_WEIGHT = 1
    P_TRANSITION_MODE = {
        'swap': 0.5,
        'grow': 0.75,
        'shrink': 1.}

    # Retrieve the network from the DB
    network = get_network()
    # Generate an empirical distribution for estimating the geo-likelihood
    ecdf_geo = generate_ecdf_geo_likelihood(net=network, min_n=MIN_SIZE, max_n=MAX_SIZE,
                                            nr_samples=SAMPLES_PER_ZONE_SIZE, plot=False)

    # MCMC Parameters
    BURN_IN = 0
    N_STEPS = 1
    N_SAMPLES = 1000
    N_RUNS = 20

    z = 6  # That's the flamingo zone
    i = [0.9, 0.6]
    f = [0.9, 0.4]

    # Two settings: [1] Easy: Favourable zones  with high intensity and many features affected by contact
    #               [2] Hard: Unfavourable zones with low intensity and few features affected by contact
    test_ease = [0, 1]
    test_annealing = [True, False]
    test_model = ['particularity', 'generative']
    r = float('inf')

    sampling_param_grid = itertools.product(test_ease,
                                            test_annealing,
                                            test_model)

    for run in range(N_RUNS):

        # Test ease, annealing and likelihood modes
        for e, a, m in sampling_param_grid:

            features_bg = simulate_background_distribution(m_feat=TOTAL_N_FEATURES,
                                                           n_sites=len(network['vertices']),
                                                           p_min=P_SUCCESS_MIN, p_max=P_SUCCESS_MAX,
                                                           load_from_dump=False)
            features = simulate_contact(r_feat=f[e], features=features_bg, p=i[e],
                                        contact_zones=get_contact_zones(z), load_from_dump=False)
            feature_prob = compute_feature_prob(features, load_from_dump=False)
            lh_lookup = precompute_feature_likelihood(1, MAX_SIZE, feature_prob, load_from_dump=False)

            zone_sampler = ZoneMCMC(network=network, features=features, min_size=MIN_SIZE,
                                    max_size=MAX_SIZE, p_transition_mode=P_TRANSITION_MODE,
                                    geo_weight=GEO_LIKELIHOOD_WEIGHT, lh_lookup=lh_lookup, n_zones=1,
                                    ecdf_geo=ecdf_geo, restart_interval=r, ecdf_type="mst",
                                    geo_ll_mode=m, feature_ll_mode=m, simulated_annealing=a,
                                    plot_samples=False)

            samples = zone_sampler.generate_samples(N_SAMPLES, N_STEPS, BURN_IN)

            # Store the results
            path = TEST_RESULTS_SAMPLING_PATH.format(e=e, a=a, m=m, run=run)

            dump(samples, path)
