import numpy as np

from sbayes.plotting.map import Map
from sbayes.plotting.plot import Plot
import os

if __name__ == '__main__':
    results_per_model = {}
    models = Map()
    models.load_config(config_file='../config_plot.json')

    # Get model names
    names = models.get_model_names()

    for m in names:
        plt = Map()
        plt.load_config(config_file='../config_plot.json')
        # Read sites, sites_names, network
        plt.read_data()
        # Read results for each model
        plt.read_results(model=m)
        results_per_model[m] = plt.results

        # Plot Maps
        # plt.posterior_map(
        #     post_freq_legend=[0.6, 0.4, 0.2],
        #     post_freq=0.8,
        #     burn_in=0.6,
        #     plot_families=True,
        #     plot_single_zones_stats=True,
        #     add_overview=True,
        #     fname='/posterior_map_' + m + '_.pdf')

        # Plot weights  and probabilities
        labels = ['U', 'C', 'I']
        plt.plot_probability_grid(burn_in=0.5, fname='/prob_grid_' + m + '_.pdf')
        plt.plot_weights_grid(labels=labels, burn_in=0.5, fname='/weights_grid_' + m + '_.pdf')

    # Plot DIC over all models
    models.plot_dic(results_per_model, burn_in=0.5, fname='/dic.pdf')
