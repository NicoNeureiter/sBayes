if __name__ == '__main__':
    from src.util import load_from, transform_weights_from_log, transform_p_from_log, \
        read_languages_from_csv
    from src.preprocessing import compute_network
    from src.postprocessing import compute_dic, match_zones, rank_zones
    from src.plotting import plot_posterior_frequency, plot_trace_lh, plot_trace_recall_precision, \
        plot_zone_size_over_time, plot_dics, plot_correlation_weights, plot_histogram_weights, plot_correlation_p, \
        plot_posterior_frequency_map_new, plot_mst_posterior_map

    import numpy as np
    import os

    import warnings

    warnings.filterwarnings("ignore")

    PATH = '../../../../' # relative path to contact_zones_directory
    PATH_BK = f'{PATH}/src/experiments/balkan/'


    TEST_ZONE_DIRECTORY = 'results/shared_evolution/prior_p_global/2019-12-13_21-43/'

    # plotting directories
    PLOT_PATH = f'{PATH}plots/balkan/'
    if not os.path.exists(PLOT_PATH): os.makedirs(PLOT_PATH)

    # MAP SETTINGS
    PROJ4_STRING =  "+proj=lcc +lat_1=43 +lat_2=62 +lat_0=30 +lon_0=10 +x_0=0 +y_0=0 +ellps=intl +units=m +no_defs"
    GEOJSON_MAP_PATH = f'{PATH_BK}data/map/ne_50m_land.geojson'
    GEOJSON_RIVER_PATH = f'{PATH_BK}data/map/ne_50m_rivers_lake_centerlines_scale_rank.geojson'

    # parameters for area to be shown on the map and in overview of map
    extend_params = {
        'lng_offsets': (800000, 500000),
        'lat_offsets': (300000, 300000),
        'lng_extend_overview': (-4800000, 3900000),
        'lat_extend_overview': (-3000000, 6200000)
    }

    # Zone, ease and number of runs
    run = 0
    n_zones = [1, 2, 3, 4, 5, 6]
    # n_zones = [5]

    # general parameters
    ts_posterior_freq = 0.6
    ts_lower_freq = 0.8
    burn_in = 0.8


    for n_zone in n_zones:

        scenario_plot_path = f'{PLOT_PATH}nz{n_zone}_{run}/'
        if not os.path.exists(scenario_plot_path):
            os.makedirs(scenario_plot_path)

        # Load the MCMC results
        # sample_path = TEST_ZONE_DIRECTORY + '/sa_contact_zones_nz' + str(n_zone_file) + '_' + str(run) + '.pkl'
        sample_path = f'{PATH_BK}{TEST_ZONE_DIRECTORY}bk_prior_p_global_nz{n_zone}_{run}.pkl'

        samples = load_from(sample_path)
        n_zones = samples['sample_zones'][0].shape[0]
        # n_families = samples['sample_p_families'][0].shape[0]
        n_families = 0

        # Define output format for estimated samples
        mcmc_res = {'lh': [],
                    'prior': [],
                    'recall': [],
                    'precision': [],
                    'posterior': [],
                    'zones': [[] for _ in range(n_zones)],
                    'lh_single_zones': [[] for _ in range(n_zones)],
                    'posterior_single_zones': [[] for _ in range(n_zones)],
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

            # Likelihood and posterior of single zones
            for z in range(n_zones):
                mcmc_res['lh_single_zones'][z].append(samples['sample_lh_single_zones'][t][z])
                posterior_single_zone = samples['sample_lh_single_zones'][t][z] + samples['sample_prior_single_zones'][t][z]
                mcmc_res['posterior_single_zones'][z].append(posterior_single_zone)

            mcmc_res['posterior'].append(posterior)

        np.set_printoptions(suppress=True)

        # Retrieve the sites from the csv and transform into a network
        sites, site_names, _, _, _, families, family_names = \
            read_languages_from_csv(f'{PATH_BK}data/features/features.csv')
        # print(families, family_names)
        # read_features_from_csv(file_location="data/features/", log=True)

        # print(site_names)
        network = compute_network(sites)

        # Change order and rank
        mcmc_res = match_zones(mcmc_res)
        mcmc_res, p_per_zone = rank_zones(mcmc_res, rank_by="lh", burn_in=0.8)

        plot_mst_posterior_map(
            mcmc_res,
            sites,
            labels=['Arabela', 'Achuar', 'Tapiete', 'Chipaya'],
            families=families,
            family_names=family_names,
            family_alpha_shape=None,
            ts_posterior_freq=ts_posterior_freq,
            lh = p_per_zone,
            bg_map=True,
            proj4=PROJ4_STRING,
            geojson_map=GEOJSON_MAP_PATH,
            geo_json_river=GEOJSON_RIVER_PATH,
            burn_in=burn_in,
            show_axes=False,
            x_extend = (-500000, 1700000),
            y_extend = (600000, 2000000),
            x_extend_overview = (-2000000, 2500000),
            y_extend_overview = (400000, 4300000),
            fname=f'{scenario_plot_path}mst_posterior_nz{n_zones}_{run}'
        )

        # Compute the dic
        # dic = compute_dic(mcmc_res, burn_in=burn_in)

    """
        # Plot posterior frequency
        plot_posterior_frequency_map_new(
            mcmc_res,
            net=network,
            labels = [],
            families = families,
            family_names = family_names,
            family_alpha_shape = 0.002,
            nz=-1,
            burn_in=burn_in,
            ts_posterior_freq = ts_posterior_freq,
            bg_map=True,
            proj4=PROJ4_STRING,
            geojson_map=GEOJSON_MAP_PATH,
            geo_json_river=GEOJSON_RIVER_PATH,
            extend_params = extend_params,
            fname=f'{scenario_plot_path}sa_contact_zones_nz{n_zones}_{run}'
        )
    """
    n_zone_file = 1
    dics = {}
    while True:
        try:
            # Load the MCMC results
            sample_path = f'{PATH_BK}{TEST_ZONE_DIRECTORY}bk_prior_p_global_nz{n_zone_file}_{run}.pkl'
            # sample_path = TEST_ZONE_DIRECTORY + '/sa_shared_evolution_nz' + str(n_zone_file) + '_' + str(run) + '.pkl'
            samples = load_from(sample_path)

        except FileNotFoundError:
            break

        # Define output format
        n_zones = samples['sample_zones'][0].shape[0]

        # Define output format
        mcmc_res_all = {'lh': []}

        for t in range(len(samples['sample_zones'])):
            # Likelihood, prior and posterior
            mcmc_res_all['lh'].append(samples['sample_likelihood'][t])

        dics[n_zone_file] = compute_dic(mcmc_res_all, 0.5)
        n_zone_file += 1

    plot_dics(dics, 5, f'{PLOT_PATH}DICs')
