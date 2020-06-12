# for Olga: could you rename plotting.py to plotting_old.py and create a new plotting.py with a plotting Class
# end then start to transfer functions from plotting_old.py to the new file as they appear in the code below?
# Some old plotting functions might not be relevant anymore.

if __name__ == '__main__':
    from src.util import load_from, samples2res
    from src.preprocessing import read_sites, compute_network
    from src.postprocessing import rank_zones
    from src.plotting import plot_trace_recall_precision, plot_trace_lh, \
        plot_posterior_frequency, plot_dics, plot_traces, plot_zone_size_over_time, \
        plot_posterior_map

    from src.postprocessing import match_zones, compute_dic
    import numpy as np
    import os

    # for Olga: this part could be read from the config file instead
    # I just noticed that for simulations, the data folder is called simulation, could you change that back to data?
    path_results = '../results/'
    path_data = '../data'
    path_plots = path_results + 'plots'

    if not os.path.exists(path_plots):
        os.makedirs(path_plots)

    run = 0
    scenarios = [1]

    # general parameters for plots
    post_freq_lines = [0.7, 0.5, 0.3]
    burn_in = 0.4

    # for Olga: This reads all the relevant files from the results folder.

    for sc in scenarios:

        n_zones = sc
        pth = f'{path_plots}/nz{n_zones}_{run}/'

        if not os.path.exists(pth):
            os.makedirs(pth)

        # This could be the first function of the plotting class: read_results
        sample_path = f'{path_results}number_zones_n{n_zones}_{run}.pkl'
        samples = load_from(sample_path)

        # for Olga: this could also be a function of the plotting class: convert_data
        mcmc_res = samples2res(samples, ground_truth=True, per_zone=True)

        # print(samples.keys())
        # test, p_per_zone = rank_zones(mcmc_res, 'lh', burn_in)
        # print(p_per_zone)

        # for Olga: This should go into the read_results function
        # Retrieve the sites from the csv and transform into a network
        sites, site_names, _ = read_sites(f'{path_data}/sites_simulation.csv')
        network = compute_network(sites)

        # for Olga: This is the first real plotting function
        # I would rather not collect the arguments that are passed to the plotting functions in config,
        # but pass them directly
        plot_posterior_map(
            mcmc_res,
            sites,
            size_line=3,
            post_freq_lines=post_freq_lines,
            burn_in=burn_in,
            lh_single_zones=False,
            simulated_data=True,
            show_axes=False,
            x_extend=(1750, 10360), # (1750, 10360)
            y_extend=(400, 11950), # (400, 11950)
            fname=f'{pth}mst_posterior_nz{n_zones}_{run}')

    # for Olga: This is relevant for later
    # nz = 0
    # dics = {}
    # list_precision = []
    # list_recall = []

    # while True:
    #
    #     nz += 1
    #     try:
    #         # Load the MCMC results
    #         sample_path = f'{path_results}number_zones_nz{nz}_{run}.pkl'
    #         samples = load_from(sample_path)
    #
    #     except FileNotFoundError:
    #         break
    #
    #     # Define output format
    #     n_zones = samples['sample_zones'][0].shape[0]
    #
    #     # Define output format
    #     mcmc_res = {'lh': [],
    #                 'prior': [],
    #                 'recall': [],
    #                 'precision': [],
    #                 'posterior': [],
    #                 'zones': [[] for _ in range(n_zones)],
    #                 'weights': [],
    #                 'true_zones': [],
    #                 'true_weights': [],
    #                 'true_lh': []}
    #
    #     # True sample
    #     true_z = np.any(samples['true_zones'], axis=0)
    #     mcmc_res['true_zones'].append(true_z)
    #     mcmc_res['true_weights'] = samples['true_weights']
    #
    #     # True likelihood
    #     mcmc_res['true_lh'] = samples['true_ll']
    #     true_posterior = samples['true_ll'] + samples['true_prior']
    #
    #     for t in range(len(samples['sample_zones'])):
    #
    #         # Zones
    #         for z in range(n_zones):
    #             mcmc_res['zones'][z].append(samples['sample_zones'][t][z])
    #         mcmc_res['weights'].append(samples['sample_weights'][t])
    #
    #         # Likelihood, prior and posterior
    #         mcmc_res['lh'].append(samples['sample_likelihood'][t])
    #         mcmc_res['prior'].append(samples['sample_prior'][t])
    #
    #         posterior = samples['sample_likelihood'][t] + samples['sample_prior'][t]
    #         mcmc_res['posterior'].append(posterior)
    #
    #         # Recall and precision
    #         sample_z = np.any(samples['sample_zones'][t], axis=0)
    #         n_true = np.sum(true_z)
    #
    #         intersections = np.minimum(sample_z, true_z)
    #         total_recall = np.sum(intersections, axis=0) / n_true
    #         mcmc_res['recall'].append(total_recall)
    #         list_recall.append(total_recall)
    #
    #         precision = np.sum(intersections, axis=0) / np.sum(sample_z, axis=0)
    #         mcmc_res['precision'].append(precision)
    #         list_precision.append(precision)
    #
    #     dics[nz] = compute_dic(mcmc_res, 0.5)

    #plot_dics(dics, fname=f'{path_plots}/DICs_all_{run}')

    # plot_traces(
    #     list_recall,
    #     list_precision,
    #     fname=f'{path_plots}/traces_all_{run}'
    #     )

    """
        

        # Plot minimum spanning tree
        for z in range(1, n_zones + 1):
            print(f'MST Zone {z}')
            # Plot minimum spanning tree
            plot_minimum_spanning_tree(
                mcmc_res,
                network,
                z=z,
                ts_posterior_freq=ts_posterior_freq,
                burn_in=burn_in,
                show_axes=False,
                annotate=True,
                fname=f'{scenario_plot_path}minimum_spanning_tree_nz{n_zones}_{run}_z{z}'
            )

        # Plot trace of likelihood, recall and precision
        plot_trace_lh(
            mcmc_res,
            burn_in=burn_in,
            true_lh=True,
            fname=f'{scenario_plot_path}trace_likelihood_nz{n_zones}_{run}'
        )

        plot_trace_recall_precision(
            mcmc_res,
            burn_in=burn_in,
            fname=f'{scenario_plot_path}trace_recall_precision_nz{n_zones}_{run}'
        )

    """

    """
        # Plot zone size over time
        plot_zone_size_over_time(
            mcmc_res,
            r = 0,
            burn_in = burn_in,
            fname = f'{scenario_plot_path}zone_size_over_time_nz{n_zones}'
        )
    """


