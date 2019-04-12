if __name__ == '__main__':
    from src.util import load_from
    from src.preprocessing import get_network
    from src.plotting import plot_posterior_frequency, plot_trace_mcmc, \
        plot_zone_size_over_time

    import numpy as np

    TEST_ZONE_DIRECTORY = 'data/results/test/zones/2019-02-23_23-04-50/'

    # Zone, ease and number of runs
    zone = 2
    ease = 3
    n_runs = 1

    mcmc_res = {'lh': [[] for _ in range(n_runs)],
                'prior': [[] for _ in range(n_runs)],
                'recall': [[] for _ in range(n_runs)],
                'precision': [[] for _ in range(n_runs)],
                'zones': [[] for _ in range(n_runs)],
                'posterior': [[] for _ in range(n_runs)],
                'lh_norm': [[] for _ in range(n_runs)],
                'posterior_norm': [[] for _ in range(n_runs)],
                'true_zones': [[] for _ in range(n_runs)]}

    for r in range(n_runs):

        # Load the MCMC results
        sample_path = TEST_ZONE_DIRECTORY + 'zone_z' + str(zone) + '_e' + \
                      str(ease) + '_' + str(r) + '.pkl'

        samples = load_from(sample_path)
        #print(samples['sample_likelihoods'])
        for t in range(len(samples['sample_zones'])):

            # Zones, likelihoods and priors
            zones = np.asarray(samples['sample_zones'][t])

            mcmc_res['zones'][r].append(zones)
            mcmc_res['lh'][r].append(samples['sample_likelihoods'][t])
            mcmc_res['prior'][r].append(samples['sample_priors'][t])

            # Normalized likelihood and posterior
            posterior = [x + y for x, y in zip(samples['sample_likelihoods'][t], samples['sample_priors'][t])]
            true_posterior = samples['true_zones_lls'][t] + samples['true_zones_priors'][t]
            mcmc_res['posterior'][r].append(posterior)
            lh_norm = np.asarray(samples['sample_likelihoods'][t]) / samples['true_zones_lls'][t]
            posterior_norm = np.asarray(posterior) / true_posterior

            # Recall and precision
            true_z = samples['true_zones'][t]
            n_true = np.sum(true_z)

            intersections = np.minimum(zones, true_z)
            total_recall = np.sum(intersections, axis=1) / n_true
            precision = np.sum(intersections, axis=1) / np.sum(zones, axis=1)

            # Store to dict
            mcmc_res['lh_norm'][r].append(lh_norm)
            mcmc_res['posterior_norm'][r].append(posterior_norm)
            mcmc_res['recall'][r].append(total_recall)
            mcmc_res['precision'][r].append(precision)
            mcmc_res['true_zones'][r].append(true_z)

    network = get_network(reevaluate=False)

    print(len(mcmc_res['zones'][0]))

    # Posterior frequency
    plot_posterior_frequency(mcmc_res['zones'], net=network, pz=-1, r=0, burn_in=0.2)


    # Trace, precision and recall
    plot_trace_mcmc(mcmc_res, r=0, burn_in=0.2)

    # Zone size over time
    plot_zone_size_over_time(mcmc_res, r=0, burn_in=0.2)
