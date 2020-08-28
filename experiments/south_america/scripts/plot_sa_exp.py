import numpy as np

from sbayes.plotting.plot import Plot
import os

if __name__ == '__main__':

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
        plt.plot_probability_grid()
        plt.plot_weights_grid(labels=labels)
