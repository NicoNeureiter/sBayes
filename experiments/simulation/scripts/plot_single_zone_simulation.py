if __name__ == '__main__':
    from src.util import load_from, transform_weights_from_log,transform_p_from_log
    from src.preprocessing import compute_network, get_sites
    from src.postprocessing import compute_dic
    from src.plotting import plot_posterior_frequency, plot_trace_lh, plot_trace_recall_precision, \
        plot_zone_size_over_time, plot_dics, plot_correlation_weights, plot_histogram_weights, plot_correlation_p

    import numpy as np
    #sampling
    TEST_ZONE_DIRECTORY = 'results/single_zone/2019-07-30_19-00/'
    # No sampling
    #TEST_ZONE_DIRECTORY = 'results/single_zone/2019-07-29_11-04/'

    # Zone, ease and number of runs
    zone = 4
    ease = 0
    run = 0

    # Load the MCMC results
    sample_path = TEST_ZONE_DIRECTORY + 'zone_z' + str(zone) + '_e' + str(ease) + '_' + str(run) + '.pkl'
    samples = load_from(sample_path)

    n_zones = samples['sample_zones'][0].shape[0]
    if samples['sample_p_families'][0] is not None:
        n_families = samples['sample_p_families'][0].shape[0]
    else:
        n_families = 0
    # todo: samples to mcmc_res
    # Define output format for estimated samples
    mcmc_res = {'lh': [],
                'prior': [],
                'recall': [],
                'precision': [],
                'posterior': [],
                'zones': [[] for _ in range(n_zones)],
                'weights': [],
                'p_zones': [[] for _ in range(n_zones)],
                'p_families': [[] for _ in range(n_families)],
                'true_zones': []}

    # Collect true sample
    true_z = np.any(samples['true_zones'], axis=0)
    mcmc_res['true_zones'].append(true_z)

    mcmc_res['true_families'] = samples['true_families']

    mcmc_res['true_p_global'] = samples['true_p_global']

    mcmc_res['true_weights'] = samples['true_weights']
    mcmc_res['true_p_zones'] = samples['true_p_zones']
    mcmc_res['true_p_families'] = samples['true_p_families']

    mcmc_res['true_lh'] = samples['true_ll']
    true_posterior = samples['true_ll'] + samples['true_prior']
    mcmc_res['true_posterior'] = true_posterior

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

            # Recall and precision
            sample_z = samples['sample_zones'][t][0]
            n_true = np.sum(true_z)

            intersections = np.minimum(sample_z, true_z)
            total_recall = np.sum(intersections, axis=0) / n_true
            mcmc_res['recall'].append(total_recall)

            precision = np.sum(intersections, axis=0) / np.sum(sample_z, axis=0)
            mcmc_res['precision'].append(precision)
    np.set_printoptions(suppress=True)

    # Retrieve the sites from the csv and transform into a network
    sites, site_names = get_sites("data/sites_simulation.csv")
    network = compute_network(sites)

    # todo: move to plot_multiple_zones
    # Compute the dic
    # dic = compute_dic(mcmc_res, burn_in=0.6)
    #
    # dics = {'model_1': dic,
    #         'model_2': dic}
    # # plot_dics(dics)

    # todo plot distribution of estimated

    #plot_correlation_weights(mcmc_res, burn_in=0.4, which_weight="global")
    #plot_correlation_weights(mcmc_res, burn_in=0.4, which_weight="contact")
    #plot_correlation_weights(mcmc_res, burn_in=0.1, which_weight="inheritance")

    plot_correlation_p(mcmc_res, burn_in=0.5, which_p="p_zones", which_nr=0, which_cat=0)
    plot_correlation_p(mcmc_res, burn_in=0.5, which_p="p_zones", which_nr=0, which_cat=1)
    plot_correlation_p(mcmc_res, burn_in=0.1, which_p="p_families", which_nr=0, which_cat=1)

    # Plot posterior frequency
    plot_posterior_frequency(mcmc_res, net=network, pz=0, burn_in=0.8, family=False)

    # Plot trace of recall and precision
    plot_trace_lh(mcmc_res, burn_in=0.4, true_lh=True)
    plot_trace_recall_precision(mcmc_res, burn_in=0.4)

    # Zone size over time
    plot_zone_size_over_time(mcmc_res, r=0, burn_in=0.2)


# todo: Possible reasons:
# - caching
# - assignemnt of families
# - p_transform_from_log




