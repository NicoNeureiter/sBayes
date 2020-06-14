from src.experiment_setup import Experiment
from src.simulation import Simulation
from src.mcmc_setup import MCMC

if __name__ == '__main__':

    # 1. Initialize the experiment
    exp = Experiment()
    exp.load_config(config_file='../config/config.json')
    exp.log_experiment()

    # 2. Simulate contact areas
    simca = Simulation(experiment=exp)
    simca.run_simulation()
    simca.log_simulation()

    # Iterate over different setups (different number of areas)
    NUMBER_AREAS = range(1, 8)

    for N in NUMBER_AREAS:
        # Update config information according to the current setup
        exp.config['mcmc']['N_AREAS'] = N

        # 3. Define MCMC
        mc = MCMC(data=simca, experiment=exp)
        mc.log_setup()

        # Rerun experiment to check for consistency
        for run in range(exp.config['mcmc']['N_RUNS']):
            # 4. Sample from posterior
            mc.sample()

            # 5. Evaluate ground truth
            mc.eval_ground_truth()

            # 6. Log sampling statistics and save samples to file
            mc.log_statistics()
            mc.save_samples()
