#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import namedtuple
import itertools

import numpy as np
import matplotlib.pyplot as plt

from sbayes.experiment_setup import Experiment
from sbayes.simulation import Simulation
from sbayes.mcmc_setup import MCMC


Rectangle = namedtuple('Rectangle', ['left', 'bottom', 'right', 'top'])
"""Basic rectangle class."""


def generate_background_locations(n_samples: int, bounding_box: Rectangle) -> np.array:
    return np.random.uniform(low=[bounding_box.left, bounding_box.bottom],
                             high=[bounding_box.right, bounding_box.top],
                             size=(n_samples, 2))


def generate_area_locations(n_samples, loc, scale):
    return np.random.normal(loc, scale, size=(n_samples, 2))


def plot_locations(background_locations, area_locations):
    # Do the plotting
    plt.scatter(*background_locations.T, c='grey')
    plt.scatter(*area_locations.T, fc='orange', ec='red')

    # Plot configuration
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Number of samples
    N_BACKGROUND = 20
    N_AREA = 10

    # Spatial distribution
    BBOX = Rectangle(left=0, bottom=0, right=100, top=100)
    AREA_MEAN = np.array([35, 60])
    AREA_STD = 10

    # Generate spatial locations
    background_locations = generate_background_locations(N_BACKGROUND, BBOX)
    area_locations = generate_area_locations(N_AREA, AREA_MEAN, AREA_STD)
    plot_locations(background_locations, area_locations)

    # 1. Initialize the experiment
    exp = Experiment()
    exp.load_config(config_file='../config.json')
    exp.log_experiment()

    # When simulating iterate over different setups (different areas and strengths of contact)
    I_CONTACT = [1.5, 2, 2.5]
    E_CONTACT = [1.25, 0.75, 0.25]
    STRENGTH = range(len(E_CONTACT))
    AREA = [4, 6, 3, 8]
    SETUP = list(itertools.product(STRENGTH, AREA))

    for S in SETUP:

        # Update config information according to the current setup
        exp.config['simulation']['I_CONTACT'] = I_CONTACT[S[0]]
        exp.config['simulation']['E_CONTACT'] = E_CONTACT[S[0]]
        exp.config['simulation']['STRENGTH'] = S[0]
        exp.config['simulation']['AREA'] = S[1]

        # 2. Simulate contact areas
        sim = Simulation(experiment=exp)
        sim.run_simulation()
        sim.log_simulation()

        # 3. Define MCMC
        mc = MCMC(data=sim, experiment=exp)
        mc.log_setup()

        # Rerun experiment to check for consistency
        for run in range(exp.config['mcmc']['N_RUNS']):

            # 4. Sample from posterior
            mc.sample()

            # 5. Evaluate ground truth
            mc.eval_ground_truth()

            # 6. Log sampling statistics and save samples to file
            mc.log_statistics()
            mc.save_samples(run=run)
