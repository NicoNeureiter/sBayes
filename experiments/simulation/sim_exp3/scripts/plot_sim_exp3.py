from sbayes.plotting.map import Map
from sbayes.plotting.general_plot import GeneralPlot


if __name__ == '__main__':
    results_per_model = {}
    models = GeneralPlot(simulated_data=True)
    models.load_config(config_file='../config_generalplot.json')

    # Get model names
    names = models.get_model_names()

    for m in names:
        plt = Map(simulated_data=True)
        plt.load_config(config_file='../config_map.json')
        plt.add_config_default()

        # Read sites, sites_names, network
        plt.read_data()
        # Read results for each model
        plt.read_results(model=m)
        results_per_model[m] = plt.results

        # Plot Maps
        try:
            plt.posterior_map(
                post_freq_legend=[1, 0.75, 0.5],
                post_freq=0.9,
                burn_in=0.7,
                plot_families=False,
                plot_area_stats=False,
                add_overview=False,
                label_languages=False,
                fname='/posterior_map_' + m)
        except ValueError:
            pass

