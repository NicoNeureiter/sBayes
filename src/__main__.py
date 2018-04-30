# In this document we test an algorithm to identify linguistic contact zones in simulated data.
# We import the data from a PostgreSQL database.
import logging

import numpy as np
from src.sampling.zone_sampling import ZoneMCMC
from src.preprocessing import (get_network,
                               compute_feature_prob,
                               get_contact_zones,
                               simulate_background_distribution,
                               simulate_contact,
                               generate_ecdf_geo_likelihood,
                               estimate_random_walk_covariance, precompute_feature_likelihood)
from src.util import dump
from src.config import *


logging.basicConfig(filename='contact_zones.log',level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logging.info('\n============================   NEW RUN   ============================')

    # Retrieve the network from the DB
    network = get_network()
    logging.info('Loaded Network...')

    # Retrieve the contact zones
    all_zones = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    contact_zones = get_contact_zones(all_zones)
    logging.info('Loaded Contact zones...')

    # Simulate distribution of features
    features_bg = simulate_background_distribution(TOTAL_N_FEATURES, len(network['vertices']),
                                                   p_min=P_SUCCESS_MIN, p_max=P_SUCCESS_MAX)
    logging.info('Simulated background features...')

    # Simulate contact zones
    features = simulate_contact(R_CONTACT_FEATURES, features_bg, P_CONTACT, contact_zones)
    logging.info('Simulated features...')

    # Compute the probability of a feature to be present or absent
    feature_prob = compute_feature_prob(features)
    logging.info('Computed feature probability...')

    # Compute a lookup table for the likelihood
    lh_lookup = precompute_feature_likelihood(1, MAX_SIZE, feature_prob)
    logging.info('Built ll lookup table...')

    # Generate an empirical distribution for estimating the geo-likelihood
    ecdf_geo = generate_ecdf_geo_likelihood(network, min_n=MIN_SIZE, max_n=MAX_SIZE,
                                            nr_samples=SAMPLES_PER_ZONE_SIZE, plot=False)
    logging.info('Computed empirical CDF...')

    random_walk_cov = estimate_random_walk_covariance(network)
    logging.info('Estimated random walk covariance...')

    logging.info('\nPreprocessing completed...\n')

    # Initialize the sampler. All parameters should be set through the config.py file.
    zone_sampler = ZoneMCMC(network, features, MIN_SIZE, MAX_SIZE, P_TRANSITION_MODE,
                            GEO_LIKELIHOOD_WEIGHT, lh_lookup, n_zones=NUMBER_PARALLEL_ZONES,
                            ecdf_geo=ecdf_geo, random_walk_cov=random_walk_cov,
                            restart_interval=RESTART_INTERVAL, simulated_annealing=SIMULATED_ANNEALING,
                            plot_samples=PLOT_SAMPLES)

    # Run the sampler
    samples = zone_sampler.generate_samples(N_SAMPLES, N_STEPS, BURN_IN_STEPS)

    # Export statistics
    mcmc_results = {'stats': zone_sampler.statistics,
                    'samples': samples,
                    'n_features': np.size(zone_sampler.features, 1),
                    'n_zones': zone_sampler.n_zones}

    # Dump the results
    dump(mcmc_results, MCMC_RESULTS_PATH)
    stats = zone_sampler.statistics
    print()
    print('Number of zones:      %r' % zone_sampler.n_zones)
    print('Acceptance Ratio:     %.2f' % stats['acceptance_ratio'])
    print('Log-Likelihood:       %.2f' % stats['sample_likelihoods'][-1])
    print('Zone sizes:           %r' % [np.sum(x) for x in samples[-1]])
    print('Time per sample:      %.3f s' % stats['time_per_sample'])


