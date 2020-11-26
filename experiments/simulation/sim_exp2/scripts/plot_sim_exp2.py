""" Testing class Plot (plot.py) on sim_exp2 results """

import warnings
from sbayes.plot import Plot

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

if __name__ == '__main__':
    plot = Plot(simulated_data=True)
    plot.load_config(config_file='../results/results_server/config_plot.json')
    plot.read_data()

    # Get model names
    names = plot.get_model_names()

    for m in names:

        # Read results for each model
        plot.read_results(model=m)

        # Plot map
        try:
            plot.posterior_map(file_name='posterior_map_' + m, return_correspondence=True)
        except ValueError:
            pass

        # Weights
        try:
            plot.plot_weights_grid(file_name='weights_grid_' + m)
        except ValueError:
            pass

        # Probabilities
        parameter = ["alpha", "beta_fam1", "gamma_a1"]
        for p in parameter:
            try:
                plot.config['probabilities_plot']['parameter'] = p
                plot.plot_probability_grid(file_name='prob_grid_' + m + '_' + p)
            except ValueError:
                pass
