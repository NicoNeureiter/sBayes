from src.experiment_setup import Experiment
from src.load_data import Data
from src.mcmc_setup import MCMC

if __name__ == '__main__':

    # 1. Initialize the experiment
    exp = Experiment()
    exp.load_config()
    exp.log()

    # 2. Load Data
    dat = Data(experiment=exp)
    # Features
    dat.load_features()
    # Counts for priors
    dat.load_universal_counts()
    dat.load_inheritance_counts()
    dat.log()

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
