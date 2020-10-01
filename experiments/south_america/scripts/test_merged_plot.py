import warnings
from sbayes.plot import Plot

warnings.simplefilter(action='ignore', category=UserWarning)

if __name__ == '__main__':

    plot = Plot()

    results_per_model = {}
    plot.load_config(config_file='../config_plot.json')

    # Get model names
    names = plot.get_model_names()

    for m in names:
        # Read sites, sites_names, network
        plot.read_data()

        # Read results for each model
        plot.read_results(model=m)
        results_per_model[m] = plot.results

        # Plot Maps
        try:
            plot.posterior_map(
                post_freq_legend=[0.8, 0.6, 0.4],
                post_freq=0.7,
                burn_in=0.2,
                plot_families=True,
                plot_area_stats=True,
                add_overview=True,
                label_languages=True,
                fname='/posterior_map_' + m)
        except ValueError:
            pass

        # Plot weights  and probabilities
        labels = ['U', 'C', 'I']
        plot.plot_probability_grid(burn_in=0.5, fname='/prob_grid_' + m)
        plot.plot_weights_grid(labels=labels, burn_in=0.5, fname='/weights_grid_' + m)

    # Plot DIC over all models
    plot.plot_dic(results_per_model, burn_in=0.5)
