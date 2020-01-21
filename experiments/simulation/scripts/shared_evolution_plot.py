if __name__ == '__main__':
    from src.util import load_from, samples2res, transform_weights_from_log,transform_p_from_log
    from src.preprocessing import compute_network, get_sites
    from src.plotting import plot_posterior_frequency, plot_trace_lh, plot_trace_recall_precision, \
        plot_zone_size_over_time, plot_minimum_spanning_tree, plot_mst_posterior
    import itertools
    import os


    PATH = '../../../../' # relative path to contact_zones_directory
    PATH_SIMULATION = f'{PATH}/src/experiments/simulation/'

    # data directories
    TEST_ZONE_DIRECTORY = 'results/shared_evolution/2019-10-20_17-57/'

    # plotting directories
    PLOT_PATH = f'{PATH}plots/shared_evolution/'
    if not os.path.exists(PLOT_PATH): os.makedirs(PLOT_PATH)


    # Zone, ease and number of runs

    # Zone [3, 4, 6, 8]
    zones = [3, 4, 6, 8]
    # zones = [6]
    zone = 3

    # Ease [0, 1, 2]
    eases = [0, 1, 2]
    # eases = [1]
    ease = 0

    # Run [0]
    runs = [0]
    run = 0

    # general parameters
    ts_posterior_freq = 0.6
    ts_lower_freq = 0.5
    burn_in = 0.2






    scenarios = [zones, eases, runs]
    scenarios = list(itertools.product(*scenarios))
    # print(scenarios)

    for scenario in scenarios:

        zone, ease, run = scenario
        print(f'Scenario: {zone} (zone), {ease} (ease), {run} (run)')

        scenario_plot_path = f'{PLOT_PATH}z{zone}_e{ease}_{run}/'


        if not os.path.exists(scenario_plot_path):
            os.makedirs(scenario_plot_path)

        # Load the MCMC results
        sample_path = f'{PATH_SIMULATION}{TEST_ZONE_DIRECTORY}shared_evolution_z{zone}_e{ease}_{run}.pkl'
        samples = load_from(sample_path)
        mcmc_res = samples2res(samples)

        zones = mcmc_res['zones']
        # print(f'Number of zones: {len(zones)}')
        # print(type(mcmc_res['zones']))
        # print(len(mcmc_res['zones'][0][0]))

        # Retrieve the sites from the csv and transform into a network
        sites, site_names = get_sites(f'{PATH_SIMULATION}data/sites_simulation.csv')
        network = compute_network(sites)

        plot_mst_posterior(
            mcmc_res,
            sites,
            ts_posterior_freq=ts_posterior_freq,
            burn_in=burn_in,
            show_zone_boundaries=True,
            show_axes=False,
            x_extend = (2510, 10000), # (1750, 10360)
            y_extend = (700, 10000), # (400, 11950)
            fname=f'{scenario_plot_path}mst_posterior_z{zone}_e{ease}_{run}'
        )


        """
        # Plot posterior frequency
        plot_posterior_frequency(
            mcmc_res,
            net = network,
            nz = 1,
            ts_posterior_freq = ts_posterior_freq,
            burn_in = burn_in,
            show_zone_bbox = True,
            show_axes = False,
            fname = f'{scenario_plot_path}posterior_frequency_z{zone}_e{ease}_{run}'
        )


        # Plot minimum spanning tree
        plot_minimum_spanning_tree(
            mcmc_res,
            network,
            z = 1,
            ts_posterior_freq = ts_posterior_freq,
            burn_in = burn_in,
            show_axes = False,
            annotate = True,
            fname = f'{scenario_plot_path}minimum spanning tree_z{zone}_e{ease}_{run}'
        )

        
        

        # Plot trace of likelihood, recall and precision
        plot_trace_lh(
            mcmc_res,
            burn_in = burn_in,
            true_lh = True,
            fname = f'{scenario_plot_path}trace_likelihood_z{zone}_e{ease}_{run}'
        )
        """


        plot_trace_recall_precision(
            mcmc_res,
            burn_in = burn_in,
            fname = f'{scenario_plot_path}trace_recall_precision_z{zone}_e{ease}_{run}'
        )

        # Plot zone size over time
        plot_zone_size_over_time(
            mcmc_res,
            r = 0,
            burn_in = burn_in,
            fname = f'{scenario_plot_path}zone_size_over_time_z{zone}_e{ease}_{run}'
        )




