if __name__ == '__main__':
    from src.util import load_from
    from src.preprocessing import get_network
    from src.plotting import plot_posterior_frequency, plot_trace_mcmc
    from src.postprocessing import stepping_stone_sampler, compute_bayes_factor
    import numpy as np

    TEST_SAMPLING_DIRECTORY = 'data/results/test/marginal_lh/2019-02-20_22-16-11/'

    # Zones, ease, number of runs, parallel zones
    areal_prior = 101
    ease = 1
    n_runs = 0

    # Load the MCMC results
    sample_path = TEST_SAMPLING_DIRECTORY + 'marginal_lh_a' + str(areal_prior) + '_e' + str(ease) + '_' + str(n_runs) + '.pkl'
    samples = load_from(sample_path)

    # Compute the marginal likelihood for each temperature
    m_lh_f = []
    m_lh_ap = []
    for size, mlh_samples in sorted(samples['m_lh_full'].items()):
        m_lh_f.append(stepping_stone_sampler(mlh_samples))

    for size, mlh_samples in sorted(samples['m_lh_areal_prior'].items()):
        m_lh_ap.append(stepping_stone_sampler(mlh_samples))

    bf = compute_bayes_factor(m_lh_ap, m_lh_f)
    print('Bayes factor in favor of m1:', bf)

    network = get_network(reevaluate=True)
    zone_plot = [[[]]]
    #plot_posterior_frequency(zone_plot, net=network, pz=-1, r=0, burn_in=0)

