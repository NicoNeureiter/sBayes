import warnings
from sbayes.plotting.map import Map
from sbayes.plotting.general_plot import GeneralPlot

warnings.simplefilter(action='ignore', category=UserWarning)

if __name__ == '__main__':
    results_per_model = {}
    models = GeneralPlot()
    models.load_config(config_file='../config_plot.json')

    # Get model names
    names = models.get_model_names()

    for m in names:
        plot_map = Map()
        plot_map.load_config(config_file='../config_plot.json')
        plot_map.add_config_default()

        # Read sites, sites_names, network
        plot_map.read_data()
        # Read results for each model
        plot_map.read_results(model=m)
        results_per_model[m] = plot_map.results

        # Plot Maps
        plot_map.posterior_map(
            post_freq_legend=[0.8, 0.6, 0.4],
            post_freq=0.5,
            burn_in=0.4,
            plot_families=True,
            add_overview=True,
            fname='/posterior_map_' + m)

        # Plot weights  and probabilities
        labels = ['U', 'C', 'I']

        plot_general = GeneralPlot()
        plot_general.load_config(config_file='../config_plot.json')
        # In this case, we don't need to use load_results
        plot_general.results = plot_map.results
        plot_general.plot_probability_grid(burn_in=0.5, fname='/prob_grid_' + m)
        plot_general.plot_weights_grid(labels=labels, burn_in=0.5, fname='/weights_grid_' + m)

    # Plot DIC over all models
    models.plot_dic(results_per_model, burn_in=0.5, fname='/dic.pdf')


