from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMCSetup

from pathlib import Path

if __name__ == '__main__':

    # Initialize the experiment
    exp = Experiment(config_file="experiments/mobility_behaviour/config.json")
    exp.log_experiment()

    # Load Data
    dat = Data.from_experiment(experiment=exp)
    mc = MCMCSetup(data=dat, experiment=exp)
    mc.log_setup()

    # Sample
    mc.warm_up()
    mc.sample()

    # Save samples to file
    mc.log_statistics()
    mc.save_samples()
