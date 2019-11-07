if __name__ == '__main__':
    from src.util import load_from, transform_weights_from_log
    from src.preprocessing import get_sites, compute_network
    from src.plotting import plot_trace_recall_precision, plot_trace_lh, \
        plot_posterior_frequency, plot_dics
    from src.postprocessing import match_zones, compute_dic
    import numpy as np

    TEST_SAMPLING_DIRECTORY = 'results/number_zones/2019-10-24_14-27/'

    # Number of zones number of runs
    nz = 6
    run = 0

    # Load the MCMC results
    sample_path = TEST_SAMPLING_DIRECTORY + 'number_zones_nz' + str(nz) + '_' + str(run) + '.pkl'

    samples = load_from(sample_path)

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

        precision = np.sum(intersections, axis=0) / np.sum(sample_z, axis=0)
        mcmc_res['precision'].append(precision)

    np.set_printoptions(suppress=True)

    # Retrieve the sites from the csv and transform into a network
    sites, site_names = get_sites("data/sites_simulation.csv")
    network = compute_network(sites)

    # Plot posterior frequency
    plot_posterior_frequency(mcmc_res, net=network, nz=3, burn_in=0.8)

    # Plot trace of likelihood, recall and precision
    plot_trace_lh(mcmc_res, burn_in=0.4, true_lh=True)
    plot_trace_recall_precision(mcmc_res, burn_in=0.4)

    nz = 0
    dics = {}
    while True:

        nz += 1

        try:
            # Load the MCMC results
            sample_path = TEST_SAMPLING_DIRECTORY + 'number_zones_nz' + str(nz) + '_' + str(run) + '.pkl'

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

            precision = np.sum(intersections, axis=0) / np.sum(sample_z, axis=0)
            mcmc_res['precision'].append(precision)

        dics[nz] = compute_dic(mcmc_res, 0.5)

    plot_dics(dics)

