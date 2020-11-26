""" Testing class Plot (plot.py) on balkan results """

import warnings
from sbayes.plot import Plot

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

if __name__ == '__main__':
    results_per_model = {}
    plot = Plot(simulated_data=False)
    plot.load_config(config_file='../config_plot.json')
    plot.read_data()

    # Get model names
    names = plot.get_model_names()

    for m in names:

        # Read results for each model
        plot.read_results(model=m)

        print('Plotting for the model', m)

        # Config has a value 0.5 (just to avoid errors)
        min_posterior_frequency = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

        mpf_counter = 1

        for mpf in min_posterior_frequency:

            print('Testing mpf value: ' + str(mpf_counter) + ' out of ' + str(len(min_posterior_frequency)))
            print('Current mpf:', mpf)

            # Assign new mpf values
            plot.config['map']['content']['min_posterior_frequency'] = mpf

            # Plot maps
            try:
                plot.posterior_map(file_name='posterior_map_' + m + '_' + str(mpf), return_correspondence=True)
            except ValueError:
                pass

            mpf_counter += 1

        results_per_model[m] = plot.results

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

    # Plot DIC over all models
    plot.plot_dic(results_per_model, file_name='dic')
