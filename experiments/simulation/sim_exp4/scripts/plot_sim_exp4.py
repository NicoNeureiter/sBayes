import numpy as np

from sbayes.plotting.map import Map
from sbayes.plotting.general_plot import GeneralPlot
import os

if __name__ == '__main__':
    results_per_model = {}
    models = GeneralPlot(simulated_data=True)
    models.load_config(config_file='../config_plot.json')

    # Get model names
    names = models.get_model_names()

    for m in names:
        plt = Map(simulated_data=True)
        plt.load_config(config_file='../config_plot.json')
        # Read sites, sites_names, network
        plt.read_data()
        # Read results for each model
        plt.read_results(model=m)
        results_per_model[m] = plt.results

        # Plot Maps
        try:
            plt.posterior_map(
                post_freq_legend=[1, 0.75, 0.5],
                post_freq=0.7,
                burn_in=0.2,
                plot_families=False,
                plot_area_stats=False,
                add_overview=False,
                label_languages=False,
                fname='/posterior_map_' + m)
        except ValueError:
            pass

        # Plot weights  and probabilities
        # labels = ['U', 'C', 'I']
        # plt.plot_probability_grid(burn_in=0.5, fname='/prob_grid_' + m + '_.pdf')
        # plt.plot_weights_grid(labels=labels, burn_in=0.5, fname='/weights_grid_' + m + '_.pdf')

    # Plot DIC over all models
    models.plot_dic(results_per_model, burn_in=0.5, fname='/dic.pdf')
