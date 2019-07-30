if __name__ == '__main__':
    from src.util import load_from
    from src.preprocessing import get_network
    from src.plotting import plot_parallel_posterior, plot_posterior_frequency, plot_geo_prior_vs_feature_lh, plot_trace_mcmc, \
        plot_zone_size_over_time
    from src.postprocessing import match_chains, apply_matching
    import numpy as np

    TEST_SAMPLING_DIRECTORY = 'data/results/test/parallel/2018-12-16_17-17-19/'

    # Zones, ease, number of runs, parallel zones

    zones = 2
    ease = 1
    n_runs = 1
    pz = 6

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
        sample_path = TEST_SAMPLING_DIRECTORY + 'parallel_z' + str(zones) + '_e' + \
                      str(ease) + '_pz' + str(pz) + '_' + str(r) + '.pkl'

        samples = load_from(sample_path)

        # Match clusters
        matching = match_chains(samples['sample_zones'])
        mcmc_res['zones'][r] = apply_matching(samples['sample_zones'], matching)
        mcmc_res['lh'][r] = apply_matching(samples['sample_likelihoods'], matching)
        mcmc_res['prior'][r] = apply_matching(samples['sample_priors'], matching)
        mcmc_res['posterior'][r] = mcmc_res['prior'][r] + mcmc_res['lh'][r]

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

    network = get_network(reevaluate=False)
    for pz in range(6):
        plot_posterior_frequency(mcmc_res['zones'], network, pz=pz, r=0, burn_in=0.8)
    plot_parallel_posterior(mcmc_res['posterior'])
