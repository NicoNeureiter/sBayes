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

        plot.config['plot_trace']['parameter'] = 'recall_and_precision'
        plot.config['plot_trace']['ground_truth']['add'] = False
        plot.plot_trace(file_name="trace_recall_precision_" + m)

        plot.config['plot_trace']['parameter'] = 'likelihood'
        plot.config['plot_trace']['ground_truth']['add'] = True
        plot.plot_trace(file_name="trace_likelihood_" + m)

