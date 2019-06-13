if __name__ == '__main__':
    from src.util import load_from, transform_weights_from_log
    from src.preprocessing import get_network
    from src.plotting import plot_posterior_frequency, plot_trace_mcmc, \
        plot_zone_size_over_time, plot_histogram_weights, plot_correlation_weights

    import numpy as np

    TEST_ZONE_DIRECTORY = 'data/results/test/zones/2019-06-11_23-26-09/'

    # Zone, ease and number of runs
    zone = 6
    ease = 0
    run = 0

    # Load the MCMC results
    sample_path = TEST_ZONE_DIRECTORY + 'zone_z' + str(zone) + '_e' + str(ease) + '_' + str(run) + '.pkl'
    samples = load_from(sample_path)

    n_zones = samples['sample_zones'][0].shape[0]

    # Define output format
    mcmc_res = {'lh': [],
                'prior': [],
                'recall': [],
                'precision': [],
                'posterior': [],
                'lh_norm': [],
                'posterior_norm': [],
                'zones': [[] for _ in range(n_zones)],
                'weights': [],
                'true_zones': [],
                'true_weights': []}

    # True sample
    true_z = np.any(samples['true_zones'], axis=0)
    mcmc_res['true_zones'].append(true_z)
    mcmc_res['true_weights'] = transform_weights_from_log(samples['true_weights'])

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

            # Normalized likelihood and posterior
            true_posterior = samples['true_zones_ll'] + samples['true_zones_prior']

            lh_norm = samples['true_zones_ll'] / samples['sample_likelihood'][t]
            mcmc_res['lh_norm'].append(lh_norm)

            posterior_norm = np.asarray(posterior) / true_posterior
            mcmc_res['posterior_norm'].append(posterior_norm)

            # Recall and precision
            sample_z = samples['sample_zones'][t][0]
            n_true = np.sum(true_z)

            intersections = np.minimum(sample_z, true_z)
            total_recall = np.sum(intersections, axis=0) / n_true
            mcmc_res['recall'].append(total_recall)

            precision = np.sum(intersections, axis=0) / np.sum(sample_z, axis=0)
            mcmc_res['precision'].append(precision)
    np.set_printoptions(suppress=True)

    print(mcmc_res['true_weights'])
    #print(samples['true_zones_ll'])
    # Network for visualization
    network = get_network(reevaluate=False)


    #plot_histogram_weights(mcmc_res, 3)
    plot_correlation_weights(mcmc_res, burn_in=0.8)

    # Plot posterior frequency
    plot_posterior_frequency(mcmc_res['zones'], net=network, pz=0, burn_in=0.6)

    # Plot trace, precision and recall
    plot_trace_mcmc(mcmc_res, r=0, burn_in=0.1, lh=True, recall=False, normalized=True, precision=False)

    # Zone size over time
    plot_zone_size_over_time(mcmc_res, r=0, burn_in=0.2)

