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
        map = Map(simulated_data=True)
        map.load_config(config_file='../config_plot.json')
        # Read sites, sites_names, network
        map.read_data()
        # Read results for each model
        map.read_results(model=m)
        results_per_model[m] = map.results
        #
        # # Plot Maps
        # try:
        #     map.posterior_map(
        #         post_freq_legend=[1, 0.75, 0.5],
        #         post_freq=0.95,
        #         burn_in=0.2,
        #         plot_families=False,
        #         plot_area_stats=False,
        #         add_overview=False,
        #         label_languages=False,
        #         fname='/posterior_map_' + m)
        #
        #     map.config['graphic']['x_extend'] = [6800, 9200]
        #     map.config['graphic']['y_extend'] = [7700, 9700]
        #
        #     map.posterior_map(
        #         post_freq_legend=[1, 0.75, 0.5],
        #         post_freq=0.95,
        #         burn_in=0.2,
        #         plot_families=False,
        #         plot_area_stats=False,
        #         add_overview=False,
        #         label_languages=False,
        #         fname='/posterior_map_zoom_' + m)
        #
        # except ValueError:
        #     pass
        # plt = GeneralPlot(simulated_data=True)
        # plt.load_config(config_file='../config_plot.json')
        #
        # # Read sites, sites_names, network
        # plt.read_data()
        # # Read results for each model
        # plt.read_results(model=m)
        # results_per_model[m] = plt.results
        # plt.plot_trace_lh_prior(burn_in=0.2, fname="/trace_likelihood_prior_" + m, prior_lim=(-2000, 0))
        # plt.plot_trace(burn_in=0.2, parameter='precision',
        #                fname="/trace_precision_" + m, ylim=(0, 1))
        # plt.plot_trace(burn_in=0.2, parameter='likelihood', ground_truth=True,
        #                fname="/trace_lh_" + m)
        # plt.plot_trace(burn_in=0.2, parameter='recall_and_precision', ylim=(0, 1),
        #                fname="/trace_recall_precision_" + m)



