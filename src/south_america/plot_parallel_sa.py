if __name__ == '__main__':
    from src.util import load_from
    from src.preprocessing import get_network
    from src.plotting import plot_posterior_frequency, plot_trace_mcmc, \
        plot_zone_size_over_time

    import numpy as np

    # Get Data
    TABLE = 'sbayes_south_america.languages'
    TEST_ZONE_DIRECTORY = 'src/south_america/2019-02-26_14-41-59/'

    # MAP
    PROJ4_STRING = '+proj=eqdc +lat_0=-32 +lon_0=-60 +lat_1=-5 +lat_2=-42 +x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs '
    GEOJSON_MAP_PATH = 'src/south_america/map/ne_50m_land.geojson'
    GEOJSON_RIVER_PATH = 'src/south_america/map/ne_50m_rivers_lake_centerlines_scale_rank.geojson'

    # Zone, ease and number of runs
    n_runs = 5
    z = 1
    mcmc_res = {'lh': [[] for _ in range(n_runs)],
                'prior': [[] for _ in range(n_runs)],
                'recall': [[] for _ in range(n_runs)],
                'precision': [[] for _ in range(n_runs)],
                'zones': [[] for _ in range(n_runs)],
                'posterior': [[] for _ in range(n_runs)]}

    for r in range(n_runs):

        # Load the MCMC results
        sample_path = TEST_ZONE_DIRECTORY + 'single_zone_run_' + str(r) + '.pkl'

        samples = load_from(sample_path)

        for t in range(len(samples['sample_zones'])):

            # Zones, likelihoods and priors
            zones = np.asarray(samples['sample_zones'][t])

            mcmc_res['zones'][r].append(zones)
            mcmc_res['lh'][r].append(samples['sample_likelihoods'][t])
            mcmc_res['prior'][r].append(samples['sample_priors'][t])

            # Normalized likelihood and posterior
            posterior = [x + y for x, y in zip(samples['sample_likelihoods'][t], samples['sample_priors'][t])]
            mcmc_res['posterior'][r].append(posterior)

    network = get_network(reevaluate=True, table=TABLE)


    # Posterior frequency
    plot_posterior_frequency(mcmc_res['zones'], net=network, pz=-1, r=0, burn_in=0, map=True, proj4=PROJ4_STRING,
                             geojson_map=GEOJSON_MAP_PATH, geo_json_river=GEOJSON_RIVER_PATH, offset_factor=0.4)


    # Trace, precision and recall
    plot_trace_mcmc(mcmc_res, r=0, burn_in=0.2, recall=False, precision=False, normalized=False)

    # Zone size over time
    plot_zone_size_over_time(mcmc_res, r=0, burn_in=0.2, true_zone=False)
