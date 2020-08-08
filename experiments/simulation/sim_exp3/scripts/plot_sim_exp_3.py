import numpy as np
from sbayes.plotting_old import plot_weights, plot_parameters_ridge

from sbayes.plotting.plot import Plot


if __name__ == '__main__':
    # labels = list(map(str, range(3)))
    # weight_samples = np.random.dirichlet([1, 3, 5], size=(100,))
    # plot_weights(weight_samples, labels)
    #
    # parameter_samples = np.random.dirichlet([1, 6], size=(100, ))
    # print(parameter_samples)
    # plot_parameters_ridge(parameter_samples)



    # Initialize Plot class
    plt = Plot(simulation=False)
    plt.load_config(config_file='../config/plot.json')

    for scenario in plt.config['input']['scenarios']:
        # Set a path for the resulting plots for the current run
        # current_path = plt.set_scenario_path(scenario)

        print('Reading input data...')

        # Read sites, sites_names, network
        plt.read_data('../data/sites.csv')

        # Read the results
        plt.read_results(scenario)

        print('Plotting...')

        # Plot weights for feature 1
        # feature = 1
        # samples, _ = plt.transform_input_weights(feature)
        # plt.plot_weights(samples, feature)

        # Plot a grid for all features
        labels = ['U', 'C', 'I']
        plt.plot_weights_grid(labels=labels)
