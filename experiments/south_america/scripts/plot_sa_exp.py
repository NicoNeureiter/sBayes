""" Testing class Plot (plot.py) on south_america results """

import warnings
from sbayes.plot import Plot

# warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=RuntimeWarning)

if __name__ == '__main__':
    results_per_model = {}
    plot = Plot(simulated_data=False)
    plot.load_config(config_file='experiments/south_america/config_plot.json')
    plot.read_data()

    # Get model names
    names = plot.get_model_names()

    for m in names:

        # Read results for each model
        plot.read_results(model=m)

        print('Plotting model', m)

        # How often does a point have to be in the posterior to be visualized in the map?
        # min_posterior_frequency = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        min_posterior_frequency = [0.9]
        mpf_counter = 1
        print('Plotting results for ' + str(len(min_posterior_frequency)) + ' different mpf values')

        for mpf in min_posterior_frequency:

            print('Current mpf: ' + str(mpf) + ' (' + str(mpf_counter) + ' out of ' +
                  str(len(min_posterior_frequency)) + ')')

            # Assign new mpf values
            plot.config['map']['content']['min_posterior_frequency'] = mpf

            # Plot maps
            plot.posterior_map(file_name='posterior_density_' + m + '_' + str(mpf))

            mpf_counter += 1

        # plot.plot_pies(file_name='pie_plot')

        # # Plot weights
        # plot.plot_weights_grid(file_name='weights_grid_' + m)
        #
        # # Plot probability grids
        # parameter = ["gamma_a1", "gamma_a2", "gamma_a3", "gamma_a4", "gamma_a5"]
        # for p in parameter:
        #     try:
        #         plot.config['probabilities_plot']['parameter'] = p
        #         plot.plot_probability_grid(file_name='prob_grid_' + m + '_' + p)
        #     except ValueError:
        #         pass
        #
        # # Collect all models for DIC plot
        results_per_model[m] = plot.results

    # # Plot DIC over all models
    # plot.plot_dic(results_per_model, file_name='dic')
