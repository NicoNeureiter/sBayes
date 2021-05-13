import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from sbayes.experiment_setup import Experiment
from sbayes.load_data import Data
from sbayes.mcmc_setup import MCMC
from sbayes.simulation import Simulation
from sbayes.plot import Plot


def run_experiment(experiment, data, run):
    mcmc = MCMC(data=data, experiment=experiment)
    mcmc.log_setup()

    # Warm-up
    mcmc.warm_up()

    # Sample from posterior
    mcmc.sample()

    # Save samples to file
    mcmc.log_statistics()
    mcmc.save_samples(run=run)

    # Use the last sample as the new initial sample
    return mcmc.samples['last_sample']


def main(config=None, experiment_name=None):
    if config is None:
        parser = argparse.ArgumentParser(
            description="An MCMC algorithm to identify contact zones")
        parser.add_argument("config", nargs="?", type=Path,
                            help="The JSON configuration file")
        args = parser.parse_args()
        config = args.config

    # 0. Ask for config file via files-dialog, if not provided as argument.
    if config is None:
        tk.Tk().withdraw()
        config = filedialog.askopenfilename(
            title='Select a config file in JSON format.',
            initialdir='..',
            filetypes=(('json files', '*.json'),('all files', '*.*'))
        )

    # Initialize the experiment
    experiment = Experiment(experiment_name=experiment_name,
                            config_file=config, log=True)

    if experiment.is_simulation():
        # The data is defined by a ´Simulation´ object
        data = Simulation(experiment=experiment)
        data.run_simulation()
        data.log_simulation()

    else:
        # Experiment based on a specified (in config) data-set
        data = Data(experiment=experiment)
        data.load_features()

        # Counts for priors
        data.load_universal_counts()
        data.load_inheritance_counts()

        # Log
        data.log_loading()

    # Rerun experiment to check for consistency
    for run in range(experiment.config['mcmc']['n_runs']):
        n_areas = experiment.config['model']['n_areas']
        if isinstance(n_areas, list):
            # Run the experiment multiple times to determine the number of areas.
            for N in n_areas:
                # Update config information according to the current setup
                experiment.config['model']['n_areas'] = N

                # Run the experiment with the specified number of areas
                run_experiment(experiment, data, run)
        else:
            # Run the experiment once, with the specified settings
            assert isinstance(n_areas, int)
            run_experiment(experiment, data, run)


def plot(config=None, experiment_name=None):
    # TODO adapt paths according to experiment_name (if provided)
    # TODO add argument defining which plots to generate (all [default], map, weights, ...)

    if config is None:
        parser = argparse.ArgumentParser(
            description="plot results of a sBayes analysis")
        parser.add_argument("config", type=Path,
                            help="The JSON configuration file")
        args = parser.parse_args()
        config = args.config

    results_per_model = {}
    plot = Plot(simulated_data=False)
    plot.load_config(config_file=config)
    plot.read_data()

    # Get model names
    names = plot.get_model_names()

    for m in names:

        # Read results for each model
        plot.read_results(model=m)

        print(plot.results['feature_names'])
        exit()

        print('Plotting model', m)

        # How often does a point have to be in the posterior to be visualized in the map?
        min_posterior_frequency = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        mpf_counter = 1
        print('Plotting results for ' + str(len(min_posterior_frequency)) + ' different mpf values')

        for mpf in min_posterior_frequency:

            print('Current mpf: ' + str(mpf) + ' (' + str(mpf_counter) + ' out of ' +
                  str(len(min_posterior_frequency)) + ')')

            # Assign new mpf values
            plot.config['map']['content']['min_posterior_frequency'] = mpf

            # Plot maps
            try:
                plot.posterior_map(file_name='posterior_map_' + m + '_' + str(mpf), return_correspondence=True)
            except ValueError:
                pass

            mpf_counter += 1

        # Plot weights
        plot.plot_weights_grid(file_name='weights_grid_' + m)

        # Plot probability grids
        parameter = ["gamma_a1", "gamma_a2", "gamma_a3", "gamma_a4", "gamma_a5"]
        for p in parameter:
            try:
                plot.config['probabilities_plot']['parameter'] = p
                plot.plot_probability_grid(file_name='prob_grid_' + m + '_' + p)
            except ValueError:
                pass

        # Collect all models for DIC plot
        results_per_model[m] = plot.results

    # Plot DIC over all models
    plot.plot_dic(results_per_model, file_name='dic')


if __name__ == '__main__':
    main()
