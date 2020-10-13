from sbayes.plotting.map import Map
from sbayes.plotting.general_plot import GeneralPlot


import warnings
from sbayes.plotting.map import Map
from sbayes.plotting.general_plot import GeneralPlot

warnings.simplefilter(action='ignore', category=UserWarning)

if __name__ == '__main__':
    results_per_model = {}
    models = GeneralPlot()
    models.load_config(config_file='../results/results_server/config_plot.json')

    # Get model names
    names = models.get_model_names()
    map = None
    for m in names:

        map = Map(simulated_data=True)
        map.load_config(config_file='../results/results_server/config_plot.json')
        # Read sites, sites_names, network
        map.read_data()
        # Read results for each model
        map.read_results(model=m)
        try:
            map.posterior_map(file_name='posterior_map_' + m, return_correspondence=False)
        except ValueError:
            pass
        results_per_model[m] = map.results

    models.plot_dic(results_per_model, file_name='dic')
    models.plot_recall_precision_over_all_models(results_per_model, file_name='recall_precision_over_all_models')

