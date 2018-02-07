#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

# Sampling parameters
N_STEPS = 1000
"""int: Number of MCMC steps."""

N_SAMPLES = 200
"""int: Number of generated samples."""

PLOT_INTERVAL = 1000
"""int: Number of steps between plotting samples."""

MAX_SIZE = 50
"""int: The maximum size for the contact zones."""

P_GLOBAL = 0.0
"""float: Probability at which the new sample is generated from global distribution."""

# Config flags
RELOAD_DATA = False
"""bool: Reload the data from the DB, preprocess is and dump it."""

# Paths for dump files
NETWORK_PATH = 'data/processed/network.pkl'
FEATURES_PATH = 'data/processed/features.pkl'
FEATURE_PROB_PATH = 'data/processed/feature_prob.pkl'
LOOKUP_TABLE_PATH = 'data/processed/lookup_table.pkl'

# Path for results
PATH_MCMC_RESULTS = 'data/results/mcmc_results.csv'


# Ploting
COLOR_WHEEL = [
    (0., 0., 0.),
    (0.4, 0.0, 0.2),
    (0.75, 0.6, 0.),
    (0.35, 0.55, 0.),
] + [(0.25, 0.7, 0.2),] * 20