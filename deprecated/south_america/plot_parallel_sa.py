if __name__ == '__main__':
    from src.util import load_from
    from src.preprocessing import get_network
    from src.plotting import plot_posterior_frequency, plot_trace_mcmc, \
        plot_zone_size_over_time, plot_parallel_posterior
    from src.postprocessing import match_chains, apply_matching
    import numpy as np

    # Get Data
    TABLE = 'sbayes_south_america.languages'
    TEST_ZONE_DIRECTORY = 'src/south_america/2019-02-26_17-10-44/'

    # MAP
    PROJ4_STRING = '+proj=eqdc +lat_0=-32 +lon_0=-60 +lat_1=-5 +lat_2=-42 +x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs '
    GEOJSON_MAP_PATH = 'src/south_america/map/ne_50m_land.geojson'
    GEOJSON_RIVER_PATH = 'src/south_america/map/ne_50m_rivers_lake_centerlines_scale_rank.geojson'

    # Zone, ease and number of runs
    n_runs = 1
    z = 4
    mcmc_res = {'lh': [[] for _ in range(n_runs)],
                'prior': [[] for _ in range(n_runs)],
                'recall': [[] for _ in range(n_runs)],
                'precision': [[] for _ in range(n_runs)],
                'zones': [[] for _ in range(n_runs)],
                'posterior': [[] for _ in range(n_runs)]}

    for r in range(n_runs):

        # Load the MCMC results
        sample_path = TEST_ZONE_DIRECTORY + 'parallel_zone_z_' +str(z) + '_run_' + str(r) + '.pkl'

        samples = load_from(sample_path)

        # Match clusters
        matching = match_chains(samples['sample_zones'])
        mcmc_res['zones'][r] = apply_matching(samples['sample_zones'], matching)
        mcmc_res['lh'][r] = apply_matching(samples['sample_likelihoods'], matching)
        mcmc_res['prior'][r] = apply_matching(samples['sample_priors'], matching)
        mcmc_res['posterior'][r] = mcmc_res['prior'][r] + mcmc_res['lh'][r]

    network = get_network(reevaluate=True, table=TABLE)


    # Posterior frequency
    for pz in range(4):
         plot_posterior_frequency(mcmc_res['zones'], net=network, pz=pz, r=0, burn_in=0, map=True, proj4=PROJ4_STRING,
                                  geojson_map=GEOJSON_MAP_PATH, geo_json_river=GEOJSON_RIVER_PATH, offset_factor=0.4)

    plot_parallel_posterior(mcmc_res['lh'])

