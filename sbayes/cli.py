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
        iterate_or_run(
            x=n_areas,
            config_setter=lambda x: experiment.config['model'].__setitem__('n_areas', x),
            function=lambda x: run_experiment(experiment, data, run)
        )


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
        # Read results for model ´m´
        plot.read_results(model=m)
        print('Plotting model', m)

        # Plot the reconstructed areas on a map
        # TODO (NN) For now we always plot the map, since the other plotting functions
        #  depend on the preprocessing done in plot_map. I suggest we resolve this when
        #  separating area summarization from plotting.
        # if 'map' in plot.config:
        plot_map(plot, m)

        # Plot the reconstructed mixture weights in simplex plots
        if 'weights_plot' in plot.config:
            plot.plot_weights_grid(file_name='weights_grid_' + m)

        # Plot the reconstructed probability vectors in simplex plots
        if 'probabilities_plot' in plot.config:
            iterate_or_run(
                x=plot.config['probabilities_plot']['parameter'],
                config_setter=lambda x: plot.config['probabilities_plot'].__setitem__('parameter', x),
                function=lambda x: plot.plot_probability_grid(file_name=f'prob_grid_{m}_{x}')
            )

        # Plot the reconstructed areas in pie-charts
        # (one per language, showing how likely the language is to be in each area)
        if 'pie_plot' in plot.config:
            plot.plot_pies(file_name= 'plot_pies_' + m)

        results_per_model[m] = plot.results

    # Plot DIC over all models
    if 'dic_plot' in plot.config:
        plot.plot_dic(results_per_model, file_name='dic')


def plot_map(plot, m):
    map_type = plot.config['map']['content']['type']

    if map_type == plot.config['map']['content']['type'] == 'density_map':
        plot.posterior_map(file_name='posterior_map_' + m)

    elif map_type == plot.config['map']['content']['type'] == 'density_map':
        iterate_or_run(
            x=plot.config['map']['content']['min_posterior_frequency'],
            config_setter=lambda x: plot.config['map']['content'].__setitem__('min_posterior_frequency', x),
            function=lambda x: plot.posterior_map(file_name=f'posterior_map_{m}_{x}'),
            print_message='Current mpf: {value} ({i} out of {len(mpf_values)})'
        )
    else:
        raise ValueError(f'Unknown map type: {map_type}  (in the config file "map" -> "content" -> "type")')


def iterate_over_parameter(values, config_setter, function, print_message=None):
    """Iterate over each value in ´values´, apply ´config_setter´ (to update the config
    dictionary) and run ´function´."""
    for i, value in enumerate(values):
        if print_message is not None:
            print(print_message.format(value=value, i=i))
        config_setter(value)
        yield function(value)


def iterate_or_run(x, config_setter, function, print_message=None):
    """If ´x´ is list, iterate over all values in ´x´ and run ´function´ for each value.
    Otherwise directly apply ´function´ to ´x´."""
    if type(x) in [tuple, list, set]:
        yield from iterate_over_parameter(x, config_setter, function, print_message)
    else:
        return function(x)


if __name__ == '__main__':
    main()
