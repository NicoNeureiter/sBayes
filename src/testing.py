from src.config import *
from src.util import dump, load_from
from src.preprocessing import (get_network,
                               generate_ecdf_geo_likelihood,
                               get_contact_zones,
                               simulate_background_distribution,
                               simulate_contact, compute_feature_prob)
from src.model import lookup_log_likelihood
from src.sampling.zone_sampling import ZoneMCMC

# Test fuer Parameter des MCMC (simulated_annealing, nr_steps=1, nr_samples=50000), 10 iterations, both modes, n hoch, p joch, flamingo
# Test fuer Parameter des MCMC (simulated_annealing, nr_steps=1, nr_samples=50000), 10 iterations, both modes, n niedrig, p niedrig, flamingo

# Likelihood plot

if __name__ == "__main__":

    if RELOAD_DATA:
        # Get all necessary data

        # Retrieve the network from the DB
        network = get_network()
        dump(network, NETWORK_PATH)

        # Generate an empirical distribution for estimating the geo-likelihood
        ecdf_geo = generate_ecdf_geo_likelihood(network, min_n=MIN_SIZE, max_n=MAX_SIZE,
                                                nr_samples=SAMPLES_PER_ZONE_SIZE, plot=False)
        dump(ecdf_geo, ECDF_GEO_PATH)

    else:
        # Load preprocessed data from dump files
        network = load_from(NETWORK_PATH)

    # Define Nr. of steps, nr. of samples and Nr. of repetitions

    # TODO: DEFINE IN CONFIG AND AGREE ON REALISTIC NR. OF ITERATIONS
    # TODO: IS it possible to run the MCMC in a specific mode?
    burn_in = empirisch, alle Sanples speichern
    nr_steps = 10   # 100
    nr_samples = 200 # 2000
    RESTART = for restart in [True, False]:
    simulated_annealing = JA

    repeat = 10

    # Background distribution
    features_bg = simulate_background_distribution(m_feat=TOTAL_N_FEATURES,
                                                   n_sites=len(network['vertices']),
                                                   p_min=P_SUCCESS_MIN, p_max=P_SUCCESS_MAX)

    # Zones
    # 6 ... flamingo, 10 ... banana, 8 ... small, 3 ... medium, 5 ... large
    test_zones = [6, 10, 8, 3, 5]

    # Intensity: proportion of sites, which are indicative of contact
    test_intensity = [0.6, 0.75, 0.9]

    # Features: proportion of features affected by contact
    test_feature_ratio = [0.4, 0.65, 0.9]

    # Models: either GM or PM
    test_models = ["GM", "PM"]

    # First loop: test different zones
    for z in test_zones:

        # Second loop: test different intensity
        for i in test_intensity:

            # Third loop: test different proportion of features
            for f in test_feature_ratio:
                features = simulate_contact(r_feat=f, features=features_bg, p=i, contact_zones=get_contact_zones(z))
                feature_prob = compute_feature_prob(features)
                lh_lookup = lookup_log_likelihood(1, MAX_SIZE, feature_prob)

                # Fourth loop: test different model (PM, GM)
                for m in test_models:

                    feature_ll_mode
                    geo_ll_mode
                    ecdf_type = mst

                    zone_sampler = ZoneMCMC(network, features, N_STEPS, MIN_SIZE, MAX_SIZE, P_TRANSITION_MODE,
                                            GEO_LIKELIHOOD_WEIGHT, lh_lookup, n_zones=NUMBER_PARALLEL_ZONES,
                                            ecdf_geo=ecdf_geo, restart_chain=RESTART_CHAIN,
                                            simulated_annealing=SIMULATED_ANNEALING, plot_samples=PLOT_SAMPLES)

                    # Repeatedly run the MCMC
                    samples = []
                    for r in range(0, repeat):
                        samples.append(zone_sampler.generate_samples(N_SAMPLES))

                    path = TEST_RESULTS_PATH + '_z' + str(z)
                    path += '_i' + str(i-int(i))[2:]
                    path += '_f' + str(f-int(f))[2:]
                    path += '_m' + str(m)

                    dump(samples, path)


    # Define Statistics to evaluate the performance
    # Parallel processing?
