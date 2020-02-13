if __name__ == '__main__':
    from src.util import load_from, transform_weights_from_log, samples2res
    from src.preprocessing import get_sites, compute_network
    from src.plotting import plot_trace_recall_precision, plot_trace_lh, \
        plot_posterior_frequency, plot_dics, plot_zone_size_over_time, plot_posterior_frequency, \
        plot_trace_lh_with_prior, plot_mst_posterior
    from src.postprocessing import match_zones, compute_dic
    import numpy as np
    import os

    PATH = '../../../' # relative path to contact_zones_directory
    PATH_SIMULATION = f'{PATH}/experiments/simulation/'

    # data directories
    TEST_ZONE_DIRECTORY = 'results/prior/geo_prior/2019-10-25_21-04/'

    # plotting directories
    PLOT_PATH = f'{PATH}plots/geo_prior/'
    if not os.path.exists(PLOT_PATH): os.makedirs(PLOT_PATH)


    # Number of zones number of runs
    pg = 1
    run = 0

    # general parameters
    ts_posterior_freq = 0.6
    ts_lower_freq = 0.5
    burn_in = 0.2


    for pg in [0, 1]:

        scenario_plot_path = f'{PLOT_PATH}pg{pg}_{run}/'

        if not os.path.exists(scenario_plot_path):
            os.makedirs(scenario_plot_path)

        # Load the MCMC results
        sample_path = f'{PATH_SIMULATION}{TEST_ZONE_DIRECTORY}geo_prior_pg{pg}_{run}.pkl'
        samples = load_from(sample_path)

        mcmc_res = samples2res(samples)
        np.set_printoptions(suppress=True)

        # Retrieve the sites from the csv and transform into a network
        sites, site_names = get_sites(f'{PATH_SIMULATION}data/sites_simulation.csv')
        network = compute_network(sites=sites)

        """
        # Plot posterior frequency
        plot_posterior_frequency(
            mcmc_res,
            net=network,
            nz=1,
            ts_posterior_freq=ts_posterior_freq,
            burn_in=burn_in,
            show_zone_bbox=False,
            show_axes=False,
            fname=f'{scenario_plot_path}posterior_frequency_pg{pg}_{run}'
        )
        """

        plot_mst_posterior(
            mcmc_res,
            sites,
            ts_posterior_freq=ts_posterior_freq,
            burn_in=burn_in,
            show_zone_boundaries=False,
            show_axes=False,
            frame_offset=None,
            x_extend=(1450, 10000),
            y_extend=(-200, 12700),
            fname=f'{scenario_plot_path}mst_posterior_pg{pg}_{run}'
        )

        # Plot trace of likelihood, recall and precision
        plot_trace_lh(
            mcmc_res,
            burn_in=burn_in,
            true_lh=True,
            steps_per_sample=2000,
            fname=f'{scenario_plot_path}trace_likelihood_pg{pg}_{run}'
        )

        plot_trace_lh_with_prior(
            mcmc_res,
            lh_range = (-35060, -34760),
            prior_range = (-140, 10),
            burn_in=burn_in,
            steps_per_sample=2000,
            labels = ('Log-likelihood', 'Geo prior'),
            fname=f'{scenario_plot_path}trace_likelihood_with_prior_pg{pg}_{run}'
        )



        plot_trace_recall_precision(
            mcmc_res,
            steps_per_sample=2000,
            burn_in=burn_in,
            fname=f'{scenario_plot_path}trace_recall_precision_pg{pg}_{run}'
        )

        # Plot zone size over time
        plot_zone_size_over_time(
            mcmc_res,
            r=0,
            burn_in=burn_in,
            fname=f'{scenario_plot_path}zone_size_over_time_pg{pg}_{run}'
        )

        # Plot posterior frequency
        # plot_posterior_frequency(mcmc_res, net=network, nz=1, burn_in=0.6)

        # Plot trace of likelihood, recall and precision
        # plot_trace_lh(mcmc_res, burn_in=0.8, true_lh=True)
        # plot_trace_recall_precision(mcmc_res, burn_in=0.4)

