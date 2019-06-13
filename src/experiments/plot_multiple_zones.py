if __name__ == '__main__':
    from src.util import load_from, transform_weights_from_log
    from src.preprocessing import get_network
    from src.plotting import plot_parallel_posterior, plot_trace_recall_precision, plot_trace_lh
    #from src.postprocessing import match_chains, apply_matching
    import numpy as np

    TEST_SAMPLING_DIRECTORY = 'data/results/test/multiple_zones/2019-06-12_19-31-05/'

    # Zones, ease, number of runs, parallel zones
    # Zone, ease and number of runs
    zone = 1
    ease = 0
    run = 0
    pz = 2

    # Load the MCMC results
    sample_path = TEST_SAMPLING_DIRECTORY + 'multiple_z' + str(zone) + '_e' + \
                  str(ease) + '_pz' + str(pz) + '_' + str(run) + '.pkl'

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
                'true_weights': [],
                'true_lh': []}

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
        mcmc_res['true_lh'] = samples['true_zones_ll']
        true_posterior = samples['true_zones_ll'] + samples['true_zones_prior']

        lh_norm = samples['true_zones_ll'] / samples['sample_likelihood'][t]
        mcmc_res['lh_norm'].append(lh_norm)

        posterior_norm = np.asarray(posterior) / true_posterior
        mcmc_res['posterior_norm'].append(posterior_norm)

        # Recall and precision
        sample_z = np.any(samples['sample_zones'][t], axis=0)
        n_true = np.sum(true_z)

        intersections = np.minimum(sample_z, true_z)
        total_recall = np.sum(intersections, axis=0) / n_true
        mcmc_res['recall'].append(total_recall)

        precision = np.sum(intersections, axis=0) / np.sum(sample_z, axis=0)
        mcmc_res['precision'].append(precision)

    np.set_printoptions(suppress=True)

    # Match clusters
    #matching = match_chains(samples['sample_zones'])

    #mcmc_res['zones'][r] = apply_matching(samples['sample_zones'], matching)

        # for t in range(len(samples['sample_zones'])):
        #
        #     # Zones, likelihoods and priors
        #     zones = np.asarray(samples['sample_zones'][t])
        #
        #     mcmc_res['zones'][r].append(zones)
        #     mcmc_res['lh'][r].append(samples['sample_likelihoods'][t])
        #     mcmc_res['prior'][r].append(samples['sample_priors'][t])
        #
        #     # Normalized likelihood and posterior
        #     posterior = [x + y for x, y in zip(samples['sample_likelihoods'][t], samples['sample_priors'][t])]
        #     #true_posterior = samples['true_zones_lls'][t] + samples['true_zones_priors'][t]
        #     mcmc_res['posterior'][r].append(posterior)
        #     #lh_norm = np.asarray(samples['sample_likelihoods'][t]) / samples['true_zones_lls'][t]
        #     #posterior_norm = np.asarray(posterior) / true_posterior
        #
        #     # Recall and precision
        #     #true_z = samples['true_zones'][t]
        #     #n_true = np.sum(true_z)
        #
        #     # zones = zones[:, 0, :]
        #     #intersections = np.minimum(zones, true_z)
        #     #total_recall = np.sum(intersections, axis=1)/n_true
        #     #precision = np.sum(intersections, axis=1)/np.sum(zones, axis=1)
        #
        #     # Store to dict
        #     #mcmc_res['lh_norm'][r].append(lh_norm)
        #     #mcmc_res['posterior_norm'][r].append(posterior_norm)
        #     #mcmc_res['recall'][r].append(total_recall)
        #     #mcmc_res['precision'][r].append(precision)
        #     #mcmc_res['true_zones'][r].append(true_z)

        # Reorder samples according to matching

    # Network for visualization
    network = get_network(reevaluate=False)

    # Plot posterior frequency
    #plot_posterior_frequency(mcmc_res['zones'], net=network, pz=0, burn_in=0.3)
    #plot_posterior_frequency(mcmc_res['zones'], net=network, pz=1, burn_in=0.3)

    # Plot trace of likelihood
    plot_trace_lh(mcmc_res, burn_in=0.6, true_lh=True)
    plot_trace_recall_precision(mcmc_res, burn_in=0.4)