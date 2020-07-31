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


def evaluate_results(stats):
    w_universal = stack_columns_starting_with('w_universal')
    w_contact = stack_columns_starting_with('w_contact')

    alpha = stack_columns_starting_with('alpha')
    gamma = stack_columns_starting_with('gamma')

    size = stack_columns_starting_with('size')

    print(sps.kstest(w_universal, 'uniform'))
    print(sps.kstest(w_contact, 'uniform'))

    print(sps.kstest(alpha, 'uniform'))
    print(sps.kstest(gamma, 'uniform'))

    size_distr = sps.binom(10, 0.5)
    print(sps.kstest(size, size_distr.cdf))

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
    # Run the experiment
    load_and_run()

    # Load the results
    stats = pd.read_csv('results/prior_simulation/n1/stats_n1_0.txt', sep='\t')

    # Evaluate and plot results
    evaluate_results(stats)
