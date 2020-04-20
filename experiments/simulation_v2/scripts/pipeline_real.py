from import_data import DataImporter
from mcmc_setup import MCMCSetup

if __name__ == '__main__':

    # Real world data
    di = DataImporter()
    di.logging_setup()
    di.get_data_features()
    di.get_prior_information()

    # MCMC (with the real world data)
    ms = MCMCSetup(log_path=di.TEST_SAMPLING_LOG_PATH,
                   results_path=di.TEST_SAMPLING_RESULTS_PATH, network=di.network,
                   features=di.features, families=di.families)

    ms.set_prior_parameters(di.p_global_dirichlet, di.p_global_categories,
                            di.p_families_dirichlet, di.p_families_categories)
    ms.logging_mcmc()
    ms.logging_sampling()
    initial_sample = ms.initialize_sample(None, None, None, None, None)

    for run in range(ms.config['mcmc']['N_RUNS']):
        for N in range(1, ms.config['mcmc']['N_ZONES'] + 1):
            operators = ms.sampling_setup()
            zone_sampler = ms.sampling_run(operators, initial_sample, n_zones=N)

            # Collect statistics
            run_stats = zone_sampler.statistics

            # Use last sample as new initial sample for next run
            initial_sample = ms.initialize_sample(run_stats['last_sample'].zones,
                                                  run_stats['last_sample'].weights,
                                                  run_stats['last_sample'].p_global,
                                                  run_stats['last_sample'].p_zones,
                                                  run_stats['last_sample'].p_families)

            # Save stats to file
            ms.save_stats(cur_run=run, cur_run_stats=run_stats, n_zones=N)
