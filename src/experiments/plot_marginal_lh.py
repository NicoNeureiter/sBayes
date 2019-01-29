if __name__ == '__main__':
    from src.util import load_from
    from src.preprocessing import get_network
    from src.plotting import plot_posterior_frequency, plot_trace_mcmc
    from src.postprocessing import stepping_stone_sampler, unnest_marginal_lh
    import numpy as np

    TEST_SAMPLING_DIRECTORY = 'data/results/test/marginal_lh/2018-12-27_00-33-57/'

    # Zones, ease, number of runs, parallel zones
    zones = 102
    ease = 0
    n_runs = 0

    # Load the MCMC results
    sample_path = TEST_SAMPLING_DIRECTORY + 'marginal_lh_z' + str(zones) + '_e' + str(ease) + '_' + str(n_runs) + '.pkl'
    samples = load_from(sample_path)

    n_temp = len(samples['temperatures'])

    mcmc_res = {'lh': [[] for _ in range(n_temp)],
                'prior': [[] for _ in range(n_temp)],
                'zones': [[] for _ in range(n_temp)],
                'posterior': [[] for _ in range(n_temp)]}

    for r in range(n_temp):

            for t in range(len(samples['m_lh_full'][r]['sample_zones'])):

                # Zones, likelihoods and priors
                zones = np.asarray(samples['m_lh_full'][r]['sample_zones'][t])
                mcmc_res['zones'][r].append(zones)

                mcmc_res['lh'][r].append(samples['m_lh_full'][r]['sample_likelihoods'][t])
                mcmc_res['prior'][r].append(samples['m_lh_full'][r]['sample_priors'][t])

                # Posterior
                posterior = [x + y for x, y in zip(samples['m_lh_full'][r]['sample_likelihoods'][t],
                                                   samples['m_lh_full'][r]['sample_priors'][t])]

                mcmc_res['posterior'][r].append(posterior)
        #print(samples['temperatures'])

    #samples['m_lh_full'] = unnest_marginal_lh(samples['m_lh_full'])
    #samples['m_lh_areal_prior'] = unnest_marginal_lh(samples['m_lh_areal_prior'])

        #temperatures = np.linspace(0, 1, 100)
        #step_full = stepping_stone_sampler(lh=samples['m_lh_full'][1:], temp=temperatures[1:])
        #step_areal_prior = stepping_stone_sampler(lh=samples['m_lh_areal_prior'][1:], temp=temperatures[1:])

        #print(samples['m_lh_full'][2])
        #print(samples['m_lh_full'][3])
    #print(mcmc_res[1]['zones'])
    print(mcmc_res['lh'][0])
    network = get_network(reevaluate=True)
    #plot_posterior_frequency(mcmc_res['zones'], net=network, pz=0, r=1, burn_in=0)
    plot_trace_mcmc(mcmc_res, r=0, burn_in=0, normalized=False, precision=False, recall=False)

# Remap network
# Plot Zone 101 and 102
