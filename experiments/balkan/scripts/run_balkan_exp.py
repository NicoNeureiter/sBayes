from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMC

if __name__ == '__main__':

    # 1. Initialize the experiment
    exp = Experiment()
    exp.load_config(config_file='experiments/balkan/config.json')
    exp.log_experiment()

    # 2. Load Data
    dat = Data(experiment=exp)
    # Features
    dat.load_features()
    # Counts for priors
    dat.load_universal_counts()
    dat.load_inheritance_counts()

    # Log
    dat.log_loading()

    NUMBER_AREAS = range(1, 8)
    initial_sample = None

    # Rerun experiment to check for consistency
    for run in range(exp.config['model']['N_RUNS']):

        for N in NUMBER_AREAS:
            # Update config information according to the current setup
            exp.config['model']['N_AREAS'] = N

            # 3. Configure MCMC
            mc = MCMC(data=dat, experiment=exp)
            mc.log_setup()

            # 4. Sample from posterior
            mc.sample(initial_sample=initial_sample)

            # 5. Log sampling statistics and save samples to file
            mc.log_statistics()
            mc.save_samples(run=run)

            # 6. Use the last sample as the new initial sample
            initial_sample = mc.samples['last_sample']

        initial_sample = None
