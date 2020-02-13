if __name__ == '__main__':
    from src.util import load_from, transform_weights_from_log, samples2res
    from src.preprocessing import get_sites, compute_network
    from src.plotting import plot_trace_recall_precision, plot_trace_lh, \
        plot_posterior_frequency, plot_dics, \
        plot_zone_size_over_time, plot_trace_lh_with_prior, plot_mst_posterior
    from src.postprocessing import match_zones, compute_dic, rank_zones
    import numpy as np
    import os


    PATH = '../../../' # relative path to contact_zones_directory
    PATH_SIMULATION = f'{PATH}/experiments/simulation/'

    # data directories
    TEST_ZONE_DIRECTORY = 'results/prior/prior_p_global/2019-10-25_21-07/'

    # plotting directories
    PLOT_PATH = f'{PATH}plots/prior_p_global/'
    if not os.path.exists(PLOT_PATH): os.makedirs(PLOT_PATH)


    # Number of zones number of runs
    pg = 1
    run = 0


    # general parameters
    ts_posterior_freq = 0.1
    ts_lower_freq = 0.5
    burn_in = 0.4



    for pg in [0, 1]:

        scenario_plot_path = f'{PLOT_PATH}pg{pg}_{run}/'
        if not os.path.exists(scenario_plot_path):
            os.makedirs(scenario_plot_path)

        # Load the MCMC results
        sample_path = f'{PATH_SIMULATION}{TEST_ZONE_DIRECTORY}prior_p_global_pg{pg}_{run}.pkl'
        print(sample_path, "path")
        samples = load_from(sample_path)
        mcmc_res = samples2res(samples)
        n_zones = samples['sample_zones'][0].shape[0]

        # Retrieve the sites from the csv and transform into a network
        sites, site_names = get_sites(f'{PATH_SIMULATION}data/sites_simulation.csv', subset=True)
        # get_subset=True
        network = compute_network(sites)

        # plot_mst_posterior(
        #     mcmc_res,
        #     sites,
        #     subset = True,
        #     ts_posterior_freq=ts_posterior_freq,
        #     burn_in=burn_in,
        #     show_zone_boundaries=True,
        #     show_axes=False,
        #     x_extend=(1750, 10360), # (1750, 10360)
        #     y_extend=(400, 11950), # (400, 11950)
        #     fname=f'{scenario_plot_path}mst_posterior_pg{pg}_{run}'
        # )
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
        # Plot trace of likelihood, recall and precision
        plot_trace_lh(
            mcmc_res,
            burn_in = burn_in,
            true_lh = True,
            steps_per_sample=2000,
            fname = f'{scenario_plot_path}trace_likelihood_pg{pg}_{run}'
        )

        plot_trace_lh_with_prior(
            mcmc_res,
            lh_range = (-3200, -2600),
            prior_range = (-9000, 1000),
            burn_in=burn_in,
            steps_per_sample=2000,
            labels = ('Log-likelihood', 'Prior p global'),
            fname=f'{scenario_plot_path}trace_likelihood_with_prior_pg{pg}_{run}'
        )

        plot_trace_recall_precision(
            mcmc_res,
            burn_in = burn_in,
            steps_per_sample= 2000,
            fname = f'{scenario_plot_path}trace_recall_precision_pg{pg}_{run}'
        )

        # Plot zone size over time
        plot_zone_size_over_time(
            mcmc_res,
            r = 0,
            burn_in = burn_in,
            fname = f'{scenario_plot_path}zone_size_over_time_pg{pg}_{run}'
        )

