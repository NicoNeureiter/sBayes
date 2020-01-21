if __name__ == '__main__':
    from src.util import load_from, transform_weights_from_log, samples2res
    from src.preprocessing import get_sites, compute_network
    from src.postprocessing import rank_zones
    from src.plotting import plot_trace_recall_precision, plot_trace_lh, \
        plot_posterior_frequency, plot_dics, plot_traces, plot_zone_size_over_time, \
        plot_minimum_spanning_tree, plot_mst_posterior

    from src.postprocessing import match_zones, compute_dic
    import numpy as np
    import os


    PATH = '../../../../' # relative path to contact_zones_directory
    PATH_SIMULATION = f'{PATH}/src/experiments/simulation/'

    # data directories
    TEST_ZONE_DIRECTORY = 'results/number_zones/2019-11-23_15-04/'

    # plotting directories
    PLOT_PATH = f'{PATH}plots/number_zones/'
    if not os.path.exists(PLOT_PATH): os.makedirs(PLOT_PATH)



    run = 0
    # scenarios = [1, 2, 3, 4, 5, 6, 7]
    scenarios = [6] # fix for more than 4 zones

    # general parameters for plots
    ts_posterior_freq = 0.6
    ts_low_frequency = 0.5
    burn_in =  0.4


    for n_zones in scenarios:

        scenario_plot_path = f'{PLOT_PATH}nz{n_zones}_{run}/'
        if not os.path.exists(scenario_plot_path):
            os.makedirs(scenario_plot_path)

        # Load the MCMC results
        sample_path = f'{PATH_SIMULATION}{TEST_ZONE_DIRECTORY}number_zones_nz{n_zones}_{run}.pkl'
        samples = load_from(sample_path)
        mcmc_res = samples2res(samples)

        # print(samples.keys())
        # test, p_per_zone = rank_zones(mcmc_res, 'lh', burn_in)
        # print(p_per_zone)

        zones = mcmc_res['zones']
        print(f'Number of zones: {len(zones)}')
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
            fname=f'{scenario_plot_path}mst_posterior_nz{n_zones}_{run}'
        )



        """
        # Plot posterior frequency
        plot_posterior_frequency(
            mcmc_res,
            net=network,
            nz=-1,
            ts_posterior_freq=ts_posterior_freq,
            burn_in=burn_in,
            show_zone_bbox=True,
            show_axes=False,
            fname=f'{scenario_plot_path}posterior_frequency_nz{n_zones}_{run}'
        )

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




"""

nz = 0
dics = {}
list_precision = []
list_recall = []
while True:

    nz += 1

    try:
        # Load the MCMC results
        # sample_path = TEST_ZONE_DIRECTORY + 'number_zones_nz' + str(nz) + '_' + str(run) + '.pkl'
        sample_path = f'{PATH_SIMULATION}{TEST_ZONE_DIRECTORY}number_zones_nz{nz}_{run}.pkl'
        samples = load_from(sample_path)

    except FileNotFoundError:
        break

    # Define output format
    n_zones = samples['sample_zones'][0].shape[0]

    # Define output format
    mcmc_res = {'lh': [],
                'prior': [],
                'recall': [],
                'precision': [],
                'posterior': [],
                'zones': [[] for _ in range(n_zones)],
                'weights': [],
                'true_zones': [],
                'true_weights': [],
                'true_lh': []}

    # True sample
    true_z = np.any(samples['true_zones'], axis=0)
    mcmc_res['true_zones'].append(true_z)
    mcmc_res['true_weights'] = transform_weights_from_log(samples['true_weights'])

    # True likelihood
    mcmc_res['true_lh'] = samples['true_ll']
    true_posterior = samples['true_ll'] + samples['true_prior']

    for t in range(len(samples['sample_zones'])):

        # Zones
        for z in range(n_zones):
            mcmc_res['zones'][z].append(samples['sample_zones'][t][z])
        mcmc_res['weights'].append(transform_weights_from_log(samples['sample_weights'][t]))

        # Likelihood, prior and posterior
        mcmc_res['lh'].append(samples['sample_likelihood'][t])
        mcmc_res['prior'].append(samples['sample_prior'][t])

        posterior = samples['sample_likelihood'][t] + samples['sample_prior'][t]
        mcmc_res['posterior'].append(posterior)

        # Recall and precision
        sample_z = np.any(samples['sample_zones'][t], axis=0)
        n_true = np.sum(true_z)

        intersections = np.minimum(sample_z, true_z)
        total_recall = np.sum(intersections, axis=0) / n_true
        mcmc_res['recall'].append(total_recall)
        list_recall.append(total_recall)

        precision = np.sum(intersections, axis=0) / np.sum(sample_z, axis=0)
        mcmc_res['precision'].append(precision)
        list_precision.append(precision)

    dics[nz] = compute_dic(mcmc_res, 0.5)


plot_dics(dics, fname=f'{PLOT_PATH}DICs_all_{run}')

plot_traces(
    list_recall,
    list_precision,
    fname=f'{PLOT_PATH}traces_all_{run}'
)

"""