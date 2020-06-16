import argparse
from pathlib import Path

from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMC


def main():
    parser = argparse.ArgumentParser(
        description="An MCMC algorithm to identify contact zones")
    parser.add_argument("config", nargs="?", type=Path,
                        help="The JSON configuration file")
    args = parser.parse_args()

    # 1. Initialize the experiment
    exp = Experiment()
    exp.load_config(config_file=args.config)
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
    for run in range(exp.config['mcmc']['N_RUNS']):

        for N in NUMBER_AREAS:
            # Update config information according to the current setup
            exp.config['mcmc']['N_AREAS'] = N

            # 3. MCMC
            mc = MCMC(data=dat, experiment=exp)
            mc.log_setup()

            # Sample
            mc.sample(initial_sample=initial_sample)

            # 4. Save samples to file
            mc.log_statistics()
            mc.save_samples(run=run)

            # Use the last sample as the new initial sample
            initial_sample = mc.samples['last_sample']

        initial_sample = None
