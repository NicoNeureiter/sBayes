from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMC

if __name__ == '__main__':

    # 1. Initialize the experiment
    exp = Experiment()
    exp.load_config(config_file='config.json')
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

    initial_sample = None

    # Rerun experiment to check for consistency
    for run in range(exp.config['mcmc']['N_RUNS']):

        # Update config information according to the current setup

        # 3. MCMC
        mc = MCMC(data=dat, experiment=exp)
        mc.log_setup()

        # Sample
        mc.sample(initial_sample=initial_sample)

        # 4. Save samples to file
        mc.log_statistics()
        mc.save_samples(run=run)

