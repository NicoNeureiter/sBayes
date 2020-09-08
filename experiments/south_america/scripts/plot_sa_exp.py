import numpy as np

from sbayes.plotting.map import Map
from sbayes.plotting.plot import Plot
import os

if __name__ == '__main__':

    plt = Map(simulated_data=False)
    plt.load_config(config_file='../config_plot.json')

    for scenario in plt.config['input']['scenarios']:
        print('Reading input data...')

        # Read sites, sites_names, network
        plt.read_data()

        # Read the results
        plt.read_results(scenario)
        plt.posterior_map(
            post_freq_lines=[0.8, 0.6, 0.4],
            burn_in=0.8,
            plot_families=True,
            plot_single_zones_stats=True,
            add_overview=True,
            fname='/posterior_map.pdf')

    # # Weights  and Probabilities
    # Initialize Plot class
    plt = Plot(simulated_data=False)
    plt.load_config(config_file='../config_plot.json')

    for scenario in plt.config['input']['scenarios']:
        # Set a path for the resulting plots for the current run
        # current_path = plt.set_scenario_path(scenario)

        print('Reading input data...')

        # Read sites, sites_names, network
        plt.read_data()

        # Read the results
        plt.read_results(scenario)

        print('Plotting...')

        # Plot weights for feature 1
        # feature = 1
        # samples, _ = plt.transform_input_weights(feature)
        # plt.plot_weights(samples, feature)

        # Plot a grid for all features
        labels = ['U', 'C', 'I']
        plt.plot_probability_grid(burn_in=0.5)
        plt.plot_weights_grid(labels=labels, burn_in=0.5)
