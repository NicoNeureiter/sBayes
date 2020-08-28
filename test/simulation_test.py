#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as sps

from sbayes.experiment_setup import Experiment
from sbayes.simulation import Simulation
from sbayes.cli import run_experiment
from sbayes.util import decode_area


def load_and_run():
    # Create Experiment object from config file
    experiment = Experiment('prior_simulation', config_file='config.json', logging=True)

    # Simulate data
    data = Simulation(experiment=experiment)
    data.run_simulation()
    data.log_simulation()

    # Run experiments
    for run in range(experiment.config['mcmc']['N_RUNS']):
        run_experiment(experiment, data, run)


def stack_columns_starting_with(prefix: str) -> np.array:
    column_names = [c for c in stats.columns if c.startswith(prefix)]
    return stats[column_names].to_numpy().flatten()


def read_areas(path):
    with open(path, 'r') as areas_file:
        areas_decoded = map(lambda line: decode_area(line.strip()), areas_file.readlines())
        return np.array(list(areas_decoded))


def evaluate_results(stats, areas):
    w_universal = stack_columns_starting_with('w_universal')
    w_contact = stack_columns_starting_with('w_contact')

    alpha = stack_columns_starting_with('alpha')
    gamma = stack_columns_starting_with('gamma')

    size = stack_columns_starting_with('size')

    print(f'KS-Test w_univ:', sps.kstest(w_universal, 'uniform'))
    print(f'KS-Test w_cont:', sps.kstest(w_contact, 'uniform'))

    print(f'KS-Test alpha:', sps.kstest(alpha, 'uniform'))
    print(f'KS-Test gamma:', sps.kstest(gamma, 'uniform'))

    size_distr = sps.binom(10, 0.5)
    # print(f'KS-Test size:', sps.binom_test(size, )

    n_samples, n_languages = areas.shape
    for i_language in range(n_languages):
        print(f'KS-Test languages {i_language}:', sps.binom_test(np.sum(areas[:, i_language]), n=n_samples, p=0.5))

    # sns.distplot(w_universal, kde=False, norm_hist=True, bins=40)
    # sns.distplot(w_contact, kde=False, norm_hist=True, bins=40)
    # sns.distplot(alpha[alpha>0], kde=False, norm_hist=True, bins=40)
    # sns.distplot(gamma[gamma>0], kde=False, norm_hist=True, bins=40)
    sns.distplot(size, kde=False, norm_hist=True, bins=40)

    # t = np.linspace(0, 1, 1000)
    t = np.arange(0, 11)
    plt.plot(t, 4.5*size_distr.pmf(t))

    plt.show()


if __name__ == '__main__':
    # x = np.random.binomial(10, 0.5, size=100)
    # print(x)
    # print(f'KS-Test:', sps.binom_test(x, n=10, p=0.5))
    # # print(sps.kstest(sps.binom(10, 0.5).rvs(size=100), sps.binom(10, 0.5).cdf))
    # exit()


    # Run the experiment
    load_and_run()

    # Load the results
    stats = pd.read_csv('results/prior_simulation/n1/stats_n1_0.txt', sep='\t')
    areas = read_areas('results/prior_simulation/n1/areas_n1_0.txt')

    # Evaluate and plot results
    evaluate_results(stats, areas)
