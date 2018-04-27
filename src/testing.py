from src.util import dump, load_from
from src.preprocessing import (get_network,
                               generate_ecdf_geo_likelihood,
                               get_contact_zones,
                               simulate_background_distribution,
                               simulate_contact, compute_feature_prob,
                               precompute_feature_likelihood)
from src.sampling.zone_sampling import ZoneMCMC


if __name__ == "__main__":

    test_type = "parallel zones"  # "zone parameters"# "sampling parameters" #

    # Configuration (see config file)
    RELOAD_DATA = False
    NETWORK_PATH = 'data/processed/network.pkl'
    ECDF_GEO_PATH = 'data/processed/ecdf_geo.pkl'
    TEST_RESULTS_PARALLEL_PATH = 'data/results/test/parallel/'
    TEST_RESULTS_SAMPLING_PATH = 'data/results/test/sampling/'
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

    # TODO: IS it possible to run the MCMC in a specific mode?

    if test_type == "sampling parameters":

        # MCMC Parameters
        BURN_IN = 0
        N_STEPS = 1
        N_SAMPLES = 100000

        z = 6  # That's the flamingo zone
        i = [0.9, 0.6]
        f = [0.9, 0.4]
        m = "particularity"

        # Two settings: [1] Easy: Favourable zones  with high intensity and many features affected by contact
        #               [2] Hard: Unfavourable zones with low intensity and few features affected by contact
        test_ease = [0, 1]
        test_restart = [True, False]
        test_annealing = [True, False]

        # Test ease
        for e in test_ease:

            # Test restart
            for r in test_restart:

                # Test annealing
                for a in test_annealing:
                    features_bg = simulate_background_distribution(m_feat=TOTAL_N_FEATURES,
                                                                   n_sites=len(network['vertices']),
                                                                   p_min=P_SUCCESS_MIN, p_max=P_SUCCESS_MAX)
                    features = simulate_contact(r_feat=f[e], features=features_bg, p=i[e], contact_zones=get_contact_zones(z))
                    feature_prob = compute_feature_prob(features)
                    lh_lookup = precompute_feature_likelihood(1, MAX_SIZE, feature_prob)

                    zone_sampler = ZoneMCMC(network=network, features=features, min_size=MIN_SIZE,
                                            max_size=MAX_SIZE, p_transition_mode=P_TRANSITION_MODE,
                                            geo_weight=GEO_LIKELIHOOD_WEIGHT, lh_lookup=lh_lookup, n_zones=1,
                                            ecdf_geo=ecdf_geo, restart_chain=r, ecdf_type="mst",
                                            geo_ll_mode=m, feature_ll_mode=m, simulated_annealing=a,
                                            plot_samples=False)

                    samples = zone_sampler.generate_samples(N_SAMPLES, N_STEPS, BURN_IN)

                    # Discard the burn-in
                    samples = samples[BURN_IN:]

                    # Store the results
                    path = TEST_RESULTS_SAMPLING_PATH + '_e' + str(e)
                    path += '_r' + str(r)
                    path += '_a' + str(a)
                    path += '.pkl'
                    dump(samples, path)

    elif test_type == "zone parameters":
        # MCMC Parameters
        BURN_IN = 0
        N_STEPS = 100
        N_SAMPLES = 10000
        SIMULATED_ANNEALING = True
        RESTART_CHAIN = True
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
                                                ecdf_geo=ecdf_geo, restart_chain=RESTART_CHAIN, ecdf_type="mst",
                                                geo_ll_mode=m, feature_ll_mode=m, simulated_annealing=SIMULATED_ANNEALING,
                                                plot_samples=False)

                        # Repeatedly run the MCMC
                        samples = []
                        for r in range(0, REPEAT):
                            samples.append(zone_sampler.generate_samples(N_SAMPLES, N_STEPS, BURN_IN))

                        path = TEST_RESULTS_ZONES_PATH + '_z' + str(z)
                        path += '_i' + str(i-int(i))[2:]
                        path += '_f' + str(f-int(f))[2:]
                        path += '_m' + str(m)
                        path += '.pkl'
                        dump(samples, path)

    elif test_type == "parallel zones":

        # MCMC Parameters
        BURN_IN = 0
        N_STEPS = 100
        N_SAMPLES = 10000
        SIMULATED_ANNEALING = True
        RESTART_CHAIN = True

        i = [0.9, 0.6]
        f = [0.9, 0.4]

        test_zones = [(6, 2), (6, 3, 8, 5), (6, 3, 8, 5, 4, 9), (6, 3, 8, 5, 4, 7, 1, 10)]
        # Two settings: [1] Easy: Favourable zones  with high intensity and many features affected by contact
        #               [2] Hard: Unfavourable zones with low intensity and few features affected by contact
        test_ease = [0, 1]
        test_models = ["generative", "particularity"]
        test_n_zones = list(range(1, 11))

        # Test different zones
        for z in test_zones:

            # Test different intensity
            for e in test_ease:
                features_bg = simulate_background_distribution(m_feat=TOTAL_N_FEATURES,
                                                               n_sites=len(network['vertices']),
                                                               p_min=P_SUCCESS_MIN, p_max=P_SUCCESS_MAX)

                features = simulate_contact(r_feat=f[e], features=features_bg, p=i[e], contact_zones=get_contact_zones(z))
                feature_prob = compute_feature_prob(features)
                lh_lookup = precompute_feature_likelihood(1, MAX_SIZE, feature_prob)

                # Test different models
                for m in test_models:

                    samples = {}
                    # Run the MCMC for different nr. of zones
                    for n in test_n_zones:

                        zone_sampler = ZoneMCMC(network=network, features=features, min_size=MIN_SIZE,
                                                max_size=MAX_SIZE, p_transition_mode=P_TRANSITION_MODE,
                                                geo_weight=GEO_LIKELIHOOD_WEIGHT, lh_lookup=lh_lookup, n_zones=n,
                                                ecdf_geo=ecdf_geo, restart_chain=RESTART_CHAIN, ecdf_type="mst",
                                                geo_ll_mode=m, feature_ll_mode=m,
                                                simulated_annealing=SIMULATED_ANNEALING,
                                                plot_samples=False)

                        samples[n] = zone_sampler.generate_samples(N_SAMPLES, N_STEPS, BURN_IN)

                    path = TEST_RESULTS_PARALLEL_PATH + 'size_z' + str(len(z))
                    path += '_e' + str(e)
                    path += '_m' + str(m)
                    path += '.pkl'
                    dump(samples, path)

    else:
        print("choose a valid test type")

    # TODO: Define Statistics to evaluate the performance
    # TODO: posterior mit unterschiedlicher Temperatur
