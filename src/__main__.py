# In this document we test an algorithm to identify linguistic contact zones in simulated data.
# We import the data from a PostgreSQL database.
import numpy as np

from src.sampling.zone_sampling import ZoneMCMC
from src.preprocessing import (get_network,
                               compute_feature_prob,
                               get_contact_zones,
                               simulate_background_distribution,
                               simulate_contact,
                               generate_ecdf_geo_likelihood)
from src.model import lookup_log_likelihood
from src.util import dump, load_from
from src.config import *


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

        # Compute a lookup table for likelihood
        # this speeds up the processing time of the algorithm
        lh_lookup = lookup_log_likelihood(1, MAX_SIZE, feature_prob)
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

    # Initialize the sampler. All parameters should be set through the config.py file.
    zone_sampler = ZoneMCMC(network, features, N_STEPS, MIN_SIZE, MAX_SIZE, P_TRANSITION_MODE,
                            GEO_LIKELIHOOD_WEIGHT, lh_lookup, n_zones=NUMBER_PARALLEL_ZONES,
                            ecdf_geo=ecdf_geo, restart_chain=RESTART_CHAIN,
                            simulated_annealing=SIMULATED_ANNEALING, plot_samples=PLOT_SAMPLES)

    # Run the sampler
    samples = zone_sampler.generate_samples(N_SAMPLES)

    # Print statistics
    stats = zone_sampler.statistics
    print('Acceptance Ratio:     %.2f' % stats['acceptance_ratio'])
    print('Log-Likelihood:       %.2f' % stats['sample_likelihoods'][-1])
    print('Size:                 %r' % np.count_nonzero(samples[-1], axis=-1))
