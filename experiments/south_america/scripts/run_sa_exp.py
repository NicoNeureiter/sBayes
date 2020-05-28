from src.experiment_setup import InitializeExperiment
from src.load_data import DataLoader
from src.mcmc_setup import MCMCSetup

if __name__ == '__main__':

    # 1. Initialize the experiment
    exp = InitializeExperiment()
    exp.load_config()
    exp.log_experiment()

    # 2. Load Data
    dat = DataLoader(experiment=exp)
    # Features
    dat.load_features()
    # Counts for priors
    dat.load_universal_counts()
    dat.load_inheritance_counts()
    dat.log_loading()

    NUMBER_AREAS = range(1, 8)
    initial_sample = None

    # Rerun experiment to check for consistency
    for run in range(exp.config['mcmc']['N_RUNS']):

        for N in NUMBER_AREAS:
            # Update config information according to the current setup
            exp.config['mcmc']['N_AREAS'] = N

            # 3. Configure MCMC
            mc = MCMCSetup(data=dat, experiment=exp)
            mc.log_setup()

            # 4. Sample from posterior
            mc.sample(initial_sample=initial_sample)

            # 5. Log sampling statistics and save samples to file
            mc.log_statistics()
            mc.save_samples(run=run)

            # 6. Use the last sample as the new initial sample
            initial_sample = mc.samples['last_sample']

        initial_sample = None
