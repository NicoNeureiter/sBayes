from sbayes.experiment_setup import Experiment
from sbayes.simulation import Simulation
from sbayes.mcmc_setup import MCMC


if __name__ == '__main__':

    # 1. Initialize the experiment
    exp = Experiment()
    exp.load_config(config_file='config.json')
    exp.log_experiment()

    # 2. Simulate contact areas
    sim = Simulation(experiment=exp)
    sim.run_simulation()
    sim.log_simulation()

    # Iterate over different setups (priors)
    PRIOR_UNIVERSAL = [False, True]

    for P in PRIOR_UNIVERSAL:
        exp.config['mcmc']['PRIOR']['universal'] = "from_simulated_counts" if P else "uniform"

        # 3. Configure MCMC
        mc = MCMC(data=sim, experiment=exp)
        mc.log_setup()

        # Rerun experiment to check for consistency
        for run in range(exp.config['mcmc']['N_RUNS']):
            # 4. Sample from posterior
            mc.sample()

            # 5. Evaluate ground truth
            mc.eval_ground_truth()

            # 6. Log sampling statistics and save samples to file
            mc.log_statistics()
            mc.save_samples(run=run)
