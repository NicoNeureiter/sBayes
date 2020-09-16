""" Testing the new Plot class on the results of the sim_exp3 experiment """

from sbayes.plotting.map import Map


if __name__ == '__main__':

    # Initialize Plot class
    plt = Map(simulated_data=True)
    plt.load_config(config_file='../config_map.json')

    for scenario in plt.config['input']['scenarios']:
        # Set a path for the resulting plots for the current run
        current_path = plt.set_scenario_path(scenario)

        print('Reading input data...')

        # Read sites, sites_names, network
        plt.read_data()

        # Read the results
        plt.read_results(scenario)

        print('Plotting...')

        # Make a number zones plot (plot_posterior_map)
        plt.posterior_map(
            burn_in=0.4,
            fname='/posterior_map')
