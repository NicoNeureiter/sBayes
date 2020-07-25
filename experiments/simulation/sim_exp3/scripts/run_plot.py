""" Testing the new Plot class on the results of the sim_exp3 experiment """

from sbayes.plotting.map import Map


if __name__ == '__main__':

    # Initialize Plot class
    plt = Map(simulation=True)
    plt.load_config(config_file='../config/plot.json')

    for scenario in plt.config['input']['scenarios']:
        # Set a path for the resulting plots for the current run
        current_path = plt.set_scenario_path(scenario)

        print('Reading input data...')

        # Read sites, sites_names, network
        plt.read_data('../data/sites.csv')

        # Read the results
        plt.read_results(scenario)

        print('Plotting...')

        # Make a number zones plot (plot_posterior_map)
        plt.number_zones(
            plt.results,
            plt.sites,
            post_freq_lines=plt.config['general']['post_freq_lines'],
            burn_in=plt.config['general']['burn_in'],
            lh_single_zones=False,
            simulated_data=True,
            fname=current_path + '/number_zones_plot')
