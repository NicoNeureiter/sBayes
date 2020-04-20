from simulation import ContactZonesSimulator
from mcmc_setup import MCMCSetup

if __name__ == '__main__':

    # Simulation
    czs = ContactZonesSimulator()
    czs.simulation()
    czs.logging_setup()
    czs.logging_simulation()

    # MCMC (with simulated data)
    ms = MCMCSetup(log_path=czs.TEST_SAMPLING_LOG_PATH,
<<<<<<< HEAD
                   results_path=czs.TEST_SAMPLING_RESULTS_PATH,
                   network=czs.network, features=czs.features, families=czs.families, label_results="i")
    ms.logging_mcmc()

    for inheritance in (True, False):

        # Sampling setup, output operators
        operators = ms.sampling_setup(inheritance=inheritance)
=======
                   results_path=czs.TEST_SAMPLING_RESULTS_PATH, is_real=False,
                   network=czs.network, features=czs.features, families=czs.families)
    ms.logging_mcmc()

    for inheritance_val in ms.config['mcmc']['INHERITANCE_TEST']:

        # Sampling setup, output operators
        operators = ms.sampling_setup(inheritance_val)
>>>>>>> 183793eafdd752df55b8fed2ffc499315e4107f9

        # Logging sampling
        ms.logging_sampling()

        # Rerun experiment to check for consistency
        for run in range(ms.config['mcmc']['N_RUNS']):
            # Sampling run
            initial_sample = ms.initialize_sample(None, None, None, None, None)
<<<<<<< HEAD
            zone_sampler = ms.sampling_run(operators, initial_sample, inheritance=inheritance)
=======
            zone_sampler = ms.sampling_run(inheritance_val, operators, initial_sample)
>>>>>>> 183793eafdd752df55b8fed2ffc499315e4107f9

            # Collect statistics
            run_stats = zone_sampler.statistics
            ms.print_op_stats(run_stats, operators)

            # True sample
            true_sample, weights_normed, p_global_padded = \
<<<<<<< HEAD
                ms.true_sample_eval(inheritance, czs.zones, czs.p_zones,
=======
                ms.true_sample_eval(inheritance_val, czs.zones, czs.p_zones,
>>>>>>> 183793eafdd752df55b8fed2ffc499315e4107f9
                                    czs.weights, czs.p_global, czs.p_families)
            run_stats = ms.true_sample_stats(zone_sampler, true_sample, run_stats,
                                             weights_normed, p_global_padded,
                                             czs.zones, czs.p_zones, czs.p_families)

            # Save stats to file
<<<<<<< HEAD
            ms.save_stats(inheritance=inheritance, cur_run=run, cur_run_stats=run_stats)
=======
            ms.save_stats(inheritance_value=inheritance_val, cur_run=run, cur_run_stats=run_stats)
>>>>>>> 183793eafdd752df55b8fed2ffc499315e4107f9
