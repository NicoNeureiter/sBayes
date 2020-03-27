from simulation import ContactZonesSimulator
from mcmc_setup import MCMCSetup

if __name__ == '__main__':

    # Simulation
    czs = ContactZonesSimulator()
    czs.simulation()
    czs.logging_setup()
    czs.logging_simulation()

    # MCMC
    # ms = MCMCSetup(czs.network_sim, czs.zones_sim, czs.features_sim, czs.categories_sim,
    #                czs.families_sim, czs.weights_sim, czs.p_global_sim,
    #                czs.p_zones_sim, czs.p_families_sim)
    # ms.logging_mcmc()
    #
    # for inheritance_val in ms.config['mcmc']['INHERITANCE_TEST']:
    #
    #     # Sampling setup, output operators
    #     operators = ms.sampling_setup(inheritance_val)
    #
    #     # Logging sampling
    #     ms.logging_sampling()
    #
    #     # Rerun experiment to check for consistency
    #     for run in range(ms.config['mcmc']['N_RUNS']):
    #
    #         # Sampling run
    #         zone_sampler = ms.sampling_run(inheritance_val, operators)
    #         run_stats = ms.stats(zone_sampler, operators)
    #
    #         # True sample
    #         # true_sample has to be passed from simulation to mcmc (?)
    #         true_sample, weights_sim_normed, p_global_sim_padded = ms.true_sample_eval(inheritance_val)
    #         ms.true_sample_stats(zone_sampler, inheritance_val, run,
    #                              true_sample, run_stats,
    #                              weights_sim_normed, p_global_sim_padded)
