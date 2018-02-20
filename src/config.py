#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

# Simulate contact zones
TOTAL_N_FEATURES = 23
"""The total number of simulated features"""

N_CONTACT_FEATURES = 20
"""The number of features for which the algorithm simulates contact"""

P_CONTACT = 0.8
"""The probability of 0/1 in the contact zones"""


# Sampling parameters
N_STEPS = 1000
"""int: Number of MCMC steps."""

N_SAMPLES = 200
"""int: Number of generated samples."""

PLOT_INTERVAL = 1000
"""int: Number of steps between plotting samples."""

MIN_SIZE = 3
"""int: The minimum size for the contact zones."""

MAX_SIZE = 100
"""int: The maximum size for the contact zones."""

P_GLOBAL = 0.0
"""float: Probability at which the new sample is generated from global distribution."""


# Config flags
LIKELIHOOD_MODES = ['generative', 'binom_test', 'binom_test_2']
"""list: All implemented modes for the likelihood function."""

LL_MODE = LIKELIHOOD_MODES[2]
"""str: The switch for which likelihood to use."""

USE_GEO_LL = False
"""bool: Use the geo-likelihood (more connected regions)."""

RESTART_CHAIN = False
"""bool: Restart the chain from a random point after every sample."""

SIMULATED_ANNEALING = True
"""bool: Slowly increase a temerature parameter to smoothly blend from sampling from a 
uniform distribution to the actual likelihood. Should help with separated modes."""

RELOAD_DATA = False
"""bool: Reload the data from the DB, pre-process it and dump it."""


# Paths for dump files
NETWORK_PATH = 'data/processed/network.pkl'
FEATURES_PATH = 'data/processed/features.pkl'
FEATURES_BG_PATH = 'data/processed/features_bg.pkl'
FEATURE_PROB_PATH = 'data/processed/feature_prob.pkl'
LOOKUP_TABLE_PATH = 'data/processed/lookup_table.pkl'
CONTACT_ZONES_PATH = 'data/processed/contact_zones.pkl'

# Path for results
PATH_MCMC_RESULTS = 'data/results/mcmc_results.csv'
