from sbayes.experiment_setup import Experiment
from sbayes.simulation import Simulation
from sbayes.mcmc_setup import MCMC

if __name__ == '__main__':

    # 1. Initialize the experiment
    exp = Experiment()
    exp.load_config(config_file='experiments/simulation/sim_exp2/config.json')
    exp.log_experiment()

    # 2. Simulate contact areas
    sim = Simulation(experiment=exp)
    sim.run_simulation()
    sim.log_simulation()

    # When performing the MCMC iterate over different setups (inheritance)
    INHERITANCE = [False, True]

    for IN in INHERITANCE:

        # Update config information according to the current setup
        exp.config['model']['INHERITANCE'] = IN
        exp.config['mcmc']['PROPOSAL_PRECISION'] = {
            "weights": 20, "universal": 30, "contact": 10, "inheritance": 10} if IN else {
            "weights": 20, "universal": 30, "contact": 10, "inheritance": None}
        exp.config['mcmc']['STEPS'] = {
            "area": 0.1, "weights": 0.4, "universal": 0.1, "contact": 0.3, "inheritance": 0.1} if IN else {
            "area": 0.1, "weights": 0.4, "universal": 0.1, "contact": 0.4, "inheritance": 0.0}

        # 3. Define MCMC
        mc = MCMC(data=sim, experiment=exp)
        mc.log_setup()

        # Rerun experiment to check for consistency
        for run in range(exp.config['mcmc']['N_RUNS']):

            # 4. Sample from posterior
            mc.warm_up()
            mc.sample()

            # 5. Evaluate ground truth
            mc.eval_ground_truth()

            # 6. Log sampling statistics and save samples to file
            mc.log_statistics()
            mc.save_samples(run=run)
