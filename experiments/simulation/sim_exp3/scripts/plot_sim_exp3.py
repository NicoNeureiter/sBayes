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
    results_per_model = {}

    for m in names:

        # Read results for each model
        plot.read_results(model=m)
        
        # Plot map
        try:
            plot.posterior_map(file_name='posterior_map_' + m, return_correspondence=False)
        except ValueError:
            pass
        results_per_model[m] = plot.results

    plot.plot_dic(results_per_model, file_name='dic')
    plot.plot_recall_precision_over_all_models(results_per_model, file_name='recall_precision_over_all_models')
