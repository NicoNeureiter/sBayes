#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import logging
import itertools
import datetime
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

from src.util import dump
from src.preprocessing import (get_network,
                               generate_ecdf_geo_likelihood,
                               get_contact_zones,
                               simulate_background_distribution,
                               simulate_contact, compute_feature_prob,
                               precompute_feature_likelihood)
from src.sampling.zone_sampling import ZoneMCMC




if __name__ == '__main__':

    now = datetime.datetime.now().__str__().rsplit('.')[0]

    TEST_SAMPLING_DIRECTORY = 'data/results/test/{experiment}/'.format(experiment=now)
    TEST_SAMPLING_RESULTS_PATH = TEST_SAMPLING_DIRECTORY + 'sampling_e{e}_a{a}_m{m}_{run}.pkl'
    TEST_SAMPLING_LL_PLOT_PATH = TEST_SAMPLING_DIRECTORY + 'sampling_ll_e{e}_a{a}_m{m}.pdf'
    TEST_SAMPLING_ZONE_PLOT_PATH = TEST_SAMPLING_DIRECTORY + 'sampling_zone_e{e}_a{a}_m{m}_{run}.pdf'
    TEST_SAMPLING_LOG_PATH = TEST_SAMPLING_DIRECTORY + 'sampling.log'

    # Make result directory if it doesn't exist yet
    if not os.path.exists(TEST_SAMPLING_DIRECTORY):
        os.mkdir(TEST_SAMPLING_DIRECTORY)

    logging.basicConfig(filename=TEST_SAMPLING_LOG_PATH, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    MIN_SIZE = 5
    MAX_SIZE = 100
    SAMPLES_PER_ZONE_SIZE = 5000

    TOTAL_N_FEATURES = 30

    P_SUCCESS_MIN = 0.05
    P_SUCCESS_MAX = 0.7
    GEO_LIKELIHOOD_WEIGHT = 1.

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
    N_STEPS = 1000
    N_SAMPLES = 2
    N_RUNS = 2

    z = 6  # That's the flamingo zone
    i = [0.9, 0.6]
    f = [0.9, 0.4]
    r = float('inf')

    # Two settings: [1] Easy: Favourable zones  with high intensity and many features affected by contact
    #               [2] Hard: Unfavourable zones with low intensity and few features affected by contact
    test_ease = [0, 1]
    test_annealing = [0, 1]
    # test_model = ['particularity', 'generative']
    # test_model = ['particularity']
    test_model = ['generative']

    def evaluate_sampling_parameters(params):
        m, e, a = params

        # Retrieve the network from the DB
        network = get_network(reevaluate=False)
        # Generate an empirical distribution for estimating the geo-likelihood
        ecdf_geo = generate_ecdf_geo_likelihood(net=network, min_n=MIN_SIZE, max_n=MAX_SIZE,
                                                nr_samples=SAMPLES_PER_ZONE_SIZE, plot=False,
                                                reevaluate=False)

        stats = []
        samples = []

        contact_zones_idxs = get_contact_zones(z)
        n_zones = len(contact_zones_idxs)
        contact_zones = np.zeros((n_zones, network['n']), bool)
        for k, cz_idxs in enumerate(contact_zones_idxs.values()):
            contact_zones[k, cz_idxs] = True

        for run in range(N_RUNS):

            # Simulation

            features_bg = simulate_background_distribution(m_feat=TOTAL_N_FEATURES,
                                                           n_sites=len(network['vertices']),
                                                           p_min=P_SUCCESS_MIN, p_max=P_SUCCESS_MAX,
                                                           reevaluate=True)
            features = simulate_contact(r_feat=f[e], features=features_bg, p=i[e],
                                        contact_zones=contact_zones_idxs, reevaluate=True)
            feature_prob = compute_feature_prob(features, reevaluate=True)
            lh_lookup = precompute_feature_likelihood(MIN_SIZE, MAX_SIZE, feature_prob,
                                                      reevaluate=True)

            # Sampling

            zone_sampler = ZoneMCMC(network=network, features=features, min_size=MIN_SIZE,
                                    max_size=MAX_SIZE, p_transition_mode=P_TRANSITION_MODE,
                                    geo_weight=GEO_LIKELIHOOD_WEIGHT, lh_lookup=lh_lookup, n_zones=1,
                                    ecdf_geo=ecdf_geo, restart_interval=r, ecdf_type="mst",
                                    geo_ll_mode=m, feature_ll_mode=m, simulated_annealing=a,
                                    plot_samples=False, print_logs=False)

            run_samples = zone_sampler.generate_samples(N_SAMPLES, N_STEPS, BURN_IN, return_steps=True)

            # Collect statistics

            run_stats = zone_sampler.statistics
            run_stats['true_zones_ll'] = [zone_sampler.log_likelihood(cz) for cz in contact_zones]

            # Add plot
            fig, axes = plt.subplots(1, N_SAMPLES)
            axes = [zone_sampler.plot_sample(z, ax=ax) for z, ax in zip(run_samples, axes)]
            run_stats['plot'] = fig

            stats.append(run_stats)
            samples.append(run_samples)

            # Store the results
            path = TEST_SAMPLING_RESULTS_PATH.format(e=e, a=a, m=m, run=run)
            dump((run_samples, run_stats), path)

        return samples, stats


    sampling_param_grid = list(itertools.product(test_model,
                                                 test_ease,
                                                 test_annealing))

    # Test ease, annealing and likelihood modes
    with Pool(4) as pool:
        all_stats = pool.map(evaluate_sampling_parameters, sampling_param_grid)

    for params, param_results in zip(sampling_param_grid, all_stats):
        param_samples, param_stats = param_results
        m, e, a = params
        print(m, e, a)

        plt.close()

        fig_ll, ax_ll = plt.subplots()

        for i_run, run_stats in enumerate(param_stats):
            lls = run_stats['step_likelihoods']
            if m == 'generative':
                lls = - (abs(np.asarray(lls)) ** 0.01)

            ax_ll.plot(lls, c='darkred', alpha=0.8)

            plot_path_zone = TEST_SAMPLING_ZONE_PLOT_PATH.format(e=e, a=a, m=m, run=i_run)
            fig_zone = run_stats['plot']
            fig_zone.show()
            fig_zone.savefig(plot_path_zone, format='pdf')


        true_zones_ll = run_stats['true_zones_ll']
        true_zones_ll = - abs(np.asarray(true_zones_ll)) ** 0.01
        plt.axhline(true_zones_ll, color='grey', linestyle='--')

        plot_path_ll = TEST_SAMPLING_LL_PLOT_PATH.format(e=e, a=a, m=m)
        fig_ll.savefig(plot_path_ll, format='pdf')
        fig_ll.show()
