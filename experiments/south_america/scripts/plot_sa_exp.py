""" Testing class Plot (plot.py) on south_america results """

import warnings
from sbayes.plot import Plot

# warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=RuntimeWarning)

if __name__ == '__main__':
    results_per_model = {}

    plot = Plot()
    plot.load_config(config_file='../results/scaled_prior/config_plot_new.json')

    plot.read_data()

    # Get model names
    names = plot.get_model_names()
    for m in names:

        # Read results for each model
        plot.read_results(model=m)

        print('Plotting map for model', m)

        plot.posterior_map(file_name='posterior_density_' + m)
        plot.plot_weights(file_name='weights_' + m)
        plot.plot_preferences(file_name='preferences_' + m)

        # # Collect all models for DIC plot
        results_per_model[m] = plot.results

    # # Plot DIC over all models
    # plot.plot_dic(results_per_model, file_name='dic')
