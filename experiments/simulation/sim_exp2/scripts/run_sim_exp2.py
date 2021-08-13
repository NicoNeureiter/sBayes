from sbayes.experiment_setup import Experiment
from sbayes.simulation import Simulation
from sbayes.mcmc_setup import MCMC

if __name__ == '__main__':

    # When performing the MCMC iterate over different setups (inheritance)
    INHERITANCE = [False, True]

    for IN in INHERITANCE:
        # 1. Initialize the experiment
        exp = Experiment()
        exp.load_config(config_file='experiments/simulation/sim_exp2/config.json',
                        custom_settings={'model': {'inheritance': IN}})
        exp.log_experiment()

        # 2. Simulate contact areas
        sim = Simulation(experiment=exp)
        sim.run_simulation()
        sim.log_simulation()

        # 3. Define MCMC
        mc = MCMC(data=sim, experiment=exp)
        mc.log_setup()

        # Rerun experiment to check for consistency
        for run in range(exp.config['mcmc']['n_runs']):

            # 4. Sample from posterior
            mc.warm_up()
            mc.sample()

            # 5. Evaluate ground truth
            mc.eval_ground_truth()

            # 6. Log sampling statistics and save samples to file
            mc.log_statistics()
            mc.save_samples(run=run)
