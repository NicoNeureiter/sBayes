import warnings
from sbayes.plotting.map import Map
from sbayes.plotting.general_plot import GeneralPlot

warnings.simplefilter(action='ignore', category=UserWarning)

if __name__ == '__main__':
    results_per_model = {}
    models = GeneralPlot()
    models.load_config(config_file='../results/no_prior/config_plot.json')

    # Get model names
    names = models.get_model_names()
    map = None
    for m in names:

        # min_posterior_frequency = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        # for mpf in min_posterior_frequency:
        #
        map = Map()
        map.load_config(config_file='../results/no_prior/config_plot.json')
        #     # map.add_config_default()
        #     # Read sites, sites_names, network
        map.read_data()
        #     # Read results for each model
        map.read_results(model=m)
        #
        #     map.config['map']['content']['min_posterior_frequency'] = mpf
        #     try:
        #         map.posterior_map(file_name='posterior_map_' + m + '_' + str(mpf), return_correspondence=True)
        #     except ValueError:
        #         pass
        #
        # results_per_model[m] = map.results
        #
        plt = GeneralPlot()
        plt.load_config(config_file='../results/no_prior/config_plot.json')
        #
        # # # In this case, we don't need to use load_results
        plt.results = map.results

        plt.plot_weights_grid(file_name='weights_grid_small' + m)

        # parameter = ["gamma_a1", "gamma_a2", "gamma_a3", "gamma_a4", "gamma_a5"]
        # for p in parameter:
        #     try:
        #         plt.config['probabilities_plot']['parameter'] = p
        #         plt.plot_probability_grid(file_name='prob_grid_' + m + '_' + p)
        #     except ValueError:
        #         pass

    # Plot DIC over all models
    models.plot_dic(results_per_model, file_name='dic')



