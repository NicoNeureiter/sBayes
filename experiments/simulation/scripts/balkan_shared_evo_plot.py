if __name__ == '__main__':
    from src.util import load_from, transform_weights_from_log, transform_p_from_log, read_languages_from_csv
    from src.preprocessing import compute_network
    from src.postprocessing import compute_dic
    from src.plotting import plot_posterior_frequency, plot_trace_lh, plot_trace_recall_precision, \
        plot_zone_size_over_time, plot_dics, plot_correlation_weights, plot_histogram_weights, plot_correlation_p

    import numpy as np

    TEST_ZONE_DIRECTORY = 'results/2019-10-02_18-41'

    # MAP SETTINGS

    PROJ4_STRING = '+proj=lcc +lat_1=43 +lat_2=62 +lat_0=30 +lon_0=10 +x_0=0 +y_0=0 +ellps=intl +units=m +no_defs'
    GEOJSON_MAP_PATH = 'data/map/ne_50m_land.geojson'
    GEOJSON_RIVER_PATH = 'data/map/ne_50m_rivers_lake_centerlines_scale_rank.geojson'

    # Zone, ease and number of runs
    n_zone = 1
    n_zone_file = 3
    run = 0
    features = 'less_salient'

    # Load the MCMC results
    sample_path = TEST_ZONE_DIRECTORY + '/shared_evo_' + features + '_nz' \
                  + str(n_zone_file) + '_' + str(run) + '.pkl'

    samples = load_from(sample_path)

    n_zones = samples['sample_zones'][0].shape[0]
    if samples['sample_p_families'][0] is not None:
        n_families = samples['sample_p_families'][0].shape[0]
    else:
        n_families = 0

    # Define output format for estimated samples
    mcmc_res = {'lh': [],
                'prior': [],
                'recall': [],
                'precision': [],
                'posterior': [],
                'zones': [[] for _ in range(n_zones)],
                'weights': [],
                'p_zones': [[] for _ in range(n_zones)],
                'p_families': [[] for _ in range(n_families)]}

    for t in range(len(samples['sample_zones'])):

            # Zones and p_zones
            for z in range(n_zones):
                mcmc_res['zones'][z].append(samples['sample_zones'][t][z])
                mcmc_res['p_zones'][z].append(transform_p_from_log(samples['sample_p_zones'][t])[z])

            # Weights
            mcmc_res['weights'].append(transform_weights_from_log(samples['sample_weights'][t]))

            # p_families
            for fam in range(n_families):
                mcmc_res['p_families'][fam].append(transform_p_from_log(samples['sample_p_families'][t])[fam])

            # Likelihood, prior and posterior
            mcmc_res['lh'].append(samples['sample_likelihood'][t])
            mcmc_res['prior'].append(samples['sample_prior'][t])

            posterior = samples['sample_likelihood'][t] + samples['sample_prior'][t]
            mcmc_res['posterior'].append(posterior)

    np.set_printoptions(suppress=True)

    # Retrieve the sites from the csv and transform into a network
    sites, site_names, _, _, _, _, _ = \
        read_languages_from_csv(file="data/" + 'features_balkan_' + features + '.csv')

    network = compute_network(sites)

    # Compute the dic
    dic = compute_dic(mcmc_res, burn_in=0.6)

    # Plot posterior frequency
    plot_posterior_frequency(mcmc_res, net=network, nz=n_zone, burn_in=0.6,  family=False, bg_map=True, proj4=PROJ4_STRING,
                             geojson_map=GEOJSON_MAP_PATH, geo_json_river=GEOJSON_RIVER_PATH, offset_factor=0.1)
    #plot_trace_lh(mcmc_res, burn_in=0.3, true_lh=False)

    # n_zone_file = 1
    # dics = {}
    # while True:
    #     print("here")
    #     try:
    #         # Load the MCMC results
    #         sample_path = TEST_ZONE_DIRECTORY + '/shared_evo_' + features + '_nz' \
    #                       + str(n_zone_file) + '_' + str(run) + '.pkl'
    #
    #         samples = load_from(sample_path)
    #
    #     except FileNotFoundError:
    #         break
    #
    #     # Define output format
    #     n_zones = samples['sample_zones'][0].shape[0]
    #
    #     # Define output format
    #     mcmc_res_all = {'lh': []}
    #
    #     for t in range(len(samples['sample_zones'])):
    #
    #         # Likelihood, prior and posterior
    #         mcmc_res_all['lh'].append(samples['sample_likelihood'][t])
    #
    #     dics[n_zone_file] = compute_dic(mcmc_res_all, 0.5)
    #     n_zone_file += 1
    #
    # plot_dics(dics)



