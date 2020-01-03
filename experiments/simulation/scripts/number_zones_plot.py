if __name__ == '__main__':
    from src.util import load_from, transform_weights_from_log
    from src.preprocessing import get_sites, compute_network
    from src.plotting import plot_trace_recall_precision, plot_trace_lh, \
        plot_posterior_frequency, plot_dics
    from src.postprocessing import match_zones, compute_dic, rank_zones
    import numpy as np

    TEST_SAMPLING_DIRECTORY = 'results/number_zones/2019-11-23_15-04/'

    # Number of zones number of runs
    nz = 4
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
                'lh_single_zones': [[] for _ in range(n_zones)],
                'posterior_single_zones': [[] for _ in range(n_zones)],
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

        # Likelihood and posterior of single zones
        for z in range(n_zones):
            mcmc_res['lh_single_zones'][z].append(samples['sample_lh_single_zones'][t][z])
            posterior_single_zone = samples['sample_lh_single_zones'][t][z] + samples['sample_prior_single_zones'][t][z]
            mcmc_res['posterior_single_zones'][z].append(posterior_single_zone)

        # Recall and precision
        sample_z = np.any(samples['sample_zones'][t], axis=0)
        n_true = np.sum(true_z)

        intersections = np.minimum(sample_z, true_z)
        total_recall = np.sum(intersections, axis=0) / n_true
        mcmc_res['recall'].append(total_recall)

        precision = np.sum(intersections, axis=0) / np.sum(sample_z, axis=0)
        mcmc_res['precision'].append(precision)

    np.set_printoptions(suppress=True)

    # Change order and rank
    mcmc_res = match_zones(mcmc_res)
    mcmc_res, p_per_zone = rank_zones(mcmc_res, rank_by="lh", burn_in=0.8)
    # Retrieve the sites from the csv and transform into a network
    sites, site_names = get_sites("data/sites_simulation.csv")
    network = compute_network(sites)

    # Plot posterior frequency
    plot_posterior_frequency(mcmc_res, net=network, nz=1, burn_in=0.8)

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

