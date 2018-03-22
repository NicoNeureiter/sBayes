#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

# Simulate contact zones
TOTAL_N_FEATURES = 30
"""The total number of simulated features"""

N_CONTACT_FEATURES = 20
"""The number of features for which the algorithm simulates contact"""

P_CONTACT = 0.8
"""The probability of 0/1 in the contact zones"""


# Sampling parameters
N_STEPS = 3000
"""int: Number of MCMC steps."""

N_SAMPLES = 10
"""int: Number of generated samples."""

MIN_SIZE = 5
"""int: The minimum size for the contact zones."""

MAX_SIZE = 100
"""int: The maximum size for the contact zones."""

P_SWAP = 0.5  # TODO remove (replaced by P_TRANSITION_MODE)
"""float: Frequency of 'swap' steps, where a node gets replaced by another (size 
remains constant)."""

ALPHA_ANNEALING = 1.5
"""float: The parameter controlling the cooling schedule for the simulated annealing 
temperature."""

P_TRANSITION_MODE = {
    'swap': 0.5,
    'grow': 0.75,
    'shrink': 1.}
assert P_TRANSITION_MODE['swap'] <= P_TRANSITION_MODE['grow'] \
                                 <= P_TRANSITION_MODE['shrink'] == 1.
"""list: Frequency of performing a 'swap', 'grow' or 'shrink' step, respectively (has 
to sum up to 1)."""


# Config flags

RESTART_CHAIN = True
"""bool: Restart the chain from a random point after every sample."""

SIMULATED_ANNEALING = False
"""bool: Slowly increase a temperature parameter to smoothly blend from sampling from a
uniform distribution to the actual likelihood. Should help with separated modes."""

RELOAD_DATA = True
"""bool: Reload the data from the DB, pre-process it and dump it."""

NUMBER_PARALLEL_ZONES = 3
"""int: Number of parallel contact zones."""

PLOT_SAMPLES = True
"""bool: Plot the every sample during MCMC run."""


# Parameters for likelihood

FEATURE_LIKELIHOOD_MODES = ['generative', 'binom_test', 'binom_test_2']
FEATURE_LL_MODE = FEATURE_LIKELIHOOD_MODES[1]
"""str: Switch for which feature-likelihood to use."""

GEO_LIKELIHOOD_MODES = ['none', 'gaussian', 'empirical']
GEO_LL_MODE = GEO_LIKELIHOOD_MODES[2]
"""str: Switch for the geo-likelihood to use."""

GEO_ECDF_TYPES = ['mst', 'delaunay', 'complete']
GEO_ECDF_TYPE = GEO_ECDF_TYPES[1]
"""str: Switch for the type of sub-graph to use for the empirical geo-likelihood."""

GEO_LIKELIHOOD_WEIGHT = 1
"""float: The weight of the geo-likelihood as compared to the feature likelihood."""

SAMPLES_PER_ZONE_SIZE = 1000
"""int: The number of samples for generating the empirical geo-likelihood."""


# Names of DB tables
DB_ZONE_TABLE = 'cz_sim.contact_zones_raw'
DB_EDGE_TABLE = 'cz_sim.delaunay_edge_list'

# Paths for dump files
NETWORK_PATH = 'data/processed/network.pkl'
FEATURES_PATH = 'data/processed/features.pkl'
FEATURES_BG_PATH = 'data/processed/features_bg.pkl'
FEATURE_PROB_PATH = 'data/processed/feature_prob.pkl'
LOOKUP_TABLE_PATH = 'data/processed/lookup_table.pkl'
CONTACT_ZONES_PATH = 'data/processed/contact_zones.pkl'
ECDF_GEO_PATH = 'data/processed/ecdf_geo.pkl'

# Path for results
PATH_MCMC_RESULTS = 'data/results/mcmc_results.csv'



