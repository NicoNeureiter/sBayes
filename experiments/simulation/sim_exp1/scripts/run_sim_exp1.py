from sbayes.experiment_setup import Experiment
from sbayes.simulation import Simulation
from sbayes.mcmc_setup import MCMC
import itertools


if __name__ == '__main__':


    # When simulating iterate over different setups (different areas and strengths of contact)
    I_CONTACT = [2, 3, 4]
    E_CONTACT = [0.75, 0.5, 0.25]
    STRENGTH = range(len(E_CONTACT))
    AREA = [4, 6, 3, 8]

    SETUP = list(itertools.product(STRENGTH, AREA))

    for S in SETUP:

        # Update config information according to the current setup
        custom_settings = {
            'I_CONTACT': I_CONTACT[S[0]],
            'E_CONTACT': E_CONTACT[S[0]],
            'STRENGTH': S[0],
            'AREA': S[1]
        }

        # 1. Initialize the experiment
        exp = Experiment()
        exp.load_config(config_file='experiments/simulation/sim_exp1/config.json',
                        custom_settings=custom_settings)
        exp.log_experiment()

        # 2. Simulate contact areas
        sim = Simulation(experiment=exp)
        sim.run_simulation()
        sim.log_simulation()

        # 3. Define MCMC
        mc = MCMC(data=sim, experiment=exp)
        mc.log_setup()

        # Rerun experiment to check for consistency
        for run in range(exp.config['mcmc']['N_RUNS']):

            # 4. Warm-up sampler and sample from posterior
            mc.warm_up()
            mc.sample()

            # 5. Evaluate ground truth
            mc.eval_ground_truth()

            # 6. Log sampling statistics and save samples to file
            mc.log_statistics()
            mc.save_samples(run=run)
