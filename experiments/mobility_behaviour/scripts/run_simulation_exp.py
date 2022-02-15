from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMC

from pathlib import Path

if __name__ == '__main__':

    # Initialize the experiment
    exp = Experiment()
    exp.load_config(config_file=Path("experiments/mobility_behaviour/config/config.json"))
    exp.log_experiment()

    # Load Data
    dat = Data(experiment=exp)
    # Features
    dat.load_features()

    # Log
    dat.log_loading()

    mc = MCMC(data=dat, experiment=exp)
    mc.log_setup()

    # Sample
    mc.warm_up()
    mc.sample()

    # Save samples to file
    mc.log_statistics()
    mc.save_samples()
    #
