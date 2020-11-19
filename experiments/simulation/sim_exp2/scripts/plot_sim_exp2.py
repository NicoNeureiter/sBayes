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
        # try:
        #     map.posterior_map(file_name='posterior_map_' + m, return_correspondence=True)
        # except ValueError:
        #     pass
        results_per_model[m] = map.results

        plt = GeneralPlot()
        plt.load_config(config_file='../results/results_server/config_plot.json')

        # # In this case, we don't need to use load_results
        plt.results = map.results
        plt.is_simulation = map.is_simulation
        try:
            plt.plot_weights_grid(file_name='weights_grid_' + m)
        except ValueError:
            pass
        parameter = ["alpha", "beta_fam1", "gamma_a1"]
        for p in parameter:
            try:
                plt.config['probabilities_plot']['parameter'] = p
                plt.plot_probability_grid(file_name='prob_grid_' + m + '_' + p)
            except ValueError:
                pass