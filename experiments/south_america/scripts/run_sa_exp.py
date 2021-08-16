from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMC


if __name__ == '__main__':

    # Initialize the experiment
    exp = Experiment()
    exp.load_config(config_file='experiments/south_america/config.json')
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

    if isinstance(exp.config['model']['areas'], list):
        NUMBER_AREAS = exp.config['model']['areas']
    else:
        NUMBER_AREAS = [exp.config['model']['areas']]

    # Rerun experiment to check for consistency
    for run in range(exp.config['mcmc']['runs']):
        for N in NUMBER_AREAS:

            # Update config information according to the current setup
            exp.config['model']['areas'] = N

            # MCMC
            mc = MCMC(data=dat, experiment=exp)
            mc.log_setup()

            # Sample
            mc.warm_up()
            mc.sample()

            # Save samples to file
            mc.log_statistics()
            mc.save_samples(run=run)

