#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import itertools

from src.util import dump, load_from
from src.preprocessing import (get_network,
                               generate_ecdf_geo_likelihood,
                               get_contact_zones,
                               simulate_background_distribution,
                               simulate_contact, compute_feature_prob,
                               precompute_feature_likelihood)
from src.sampling.zone_sampling import ZoneMCMC


if __name__ == '__main__':

    # Configuration (see config file)
    TEST_RESULTS_ZONES_PATH = 'data/results/test/zones/'

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
    N_STEPS = 100
    N_SAMPLES = 10000
    SIMULATED_ANNEALING = True
    RESTART_INTERVAL = True
    REPEAT = 10

    # Zones
    # 6 ... flamingo, 10 ... banana, 8 ... small, 3 ... medium, 5 ... large
    test_zones = [6, 10, 8, 3, 5]

    # Intensity: proportion of sites, which are indicative of contact
    test_intensity = [0.6, 0.75, 0.9]

    # Features: proportion of features affected by contact
    test_feature_ratio = [0.4, 0.65, 0.9]

    # Models: either GM or PM
    test_models = ["generative", "particularity"]

    # Test different zones
    for z in test_zones:

        # Test different intensity
        for i in test_intensity:

            # Test different proportion of features
            for f in test_feature_ratio:

                features_bg = simulate_background_distribution(m_feat=TOTAL_N_FEATURES,
                                                               n_sites=len(network['vertices']),
                                                               p_min=P_SUCCESS_MIN, p_max=P_SUCCESS_MAX)

                features = simulate_contact(r_feat=f, features=features_bg, p=i, contact_zones=get_contact_zones(z))
                feature_prob = compute_feature_prob(features)
                lh_lookup = precompute_feature_likelihood(1, MAX_SIZE, feature_prob)

                # Test different model (PM, GM)
                for m in test_models:

                    zone_sampler = ZoneMCMC(network=network, features=features, min_size=MIN_SIZE,
                                            max_size=MAX_SIZE, p_transition_mode=P_TRANSITION_MODE,
                                            geo_weight=GEO_LIKELIHOOD_WEIGHT, lh_lookup=lh_lookup, n_zones=1,
                                            ecdf_geo=ecdf_geo, restart_interval=RESTART_INTERVAL, ecdf_type="mst",
                                            geo_ll_mode=m, feature_ll_mode=m, simulated_annealing=SIMULATED_ANNEALING,
                                            plot_samples=False)

                    # Repeatedly run the MCMC
                    samples = []
                    for r in range(0, REPEAT):
                        samples.append(zone_sampler.generate_samples(N_SAMPLES, N_STEPS, BURN_IN))

                    path = TEST_RESULTS_ZONES_PATH + '_z' + str(z)
                    path += '_i' + str(i - int(i))[2:]
                    path += '_f' + str(f - int(f))[2:]
                    path += '_m' + str(m)
                    path += '.pkl'
                    dump(samples, path)