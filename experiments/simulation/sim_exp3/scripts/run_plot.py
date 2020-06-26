""" Testing the new Plot class on the results of the sim_exp3 experiment """

from src.plot import Plot


if __name__ == '__main__':
    plt = Plot()
    plt.load_config(config_file='../config/plot.json')

    for scenario in plt.config['input']['scenarios']:
        plt.set_scenario_path(scenario)