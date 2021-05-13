from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMC

if __name__ == '__main__':

    # Initialize the experiment
    exp = Experiment()
    exp.load_config(config_file='experiments/balkan/config.json')
    exp.log_experiment()

    # Load Data
    dat = Data(experiment=exp)
    # Features
    dat.load_features()
    # Counts for priors
    dat.load_universal_counts()
    dat.load_inheritance_counts()

    # Log
    dat.log_loading()

    NUMBER_AREAS = range(1, 8)

    # Rerun experiment to check for consistency
    for run in range(exp.config['mcmc']['n_runs']):

        for N in NUMBER_AREAS:
            # Update config information according to the current setup
            exp.config['model']['n_areas'] = N

            # Configure MCMC
            mc = MCMC(data=dat, experiment=exp)

            mc.log_setup()

            # Sample from posterior
            mc.warm_up()
            mc.sample()

            # Log sampling statistics and save samples to file
            mc.log_statistics()
            mc.save_samples(run=run)

