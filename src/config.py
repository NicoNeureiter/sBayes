#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals


# Simulate contact zones and background distribution
TOTAL_N_FEATURES = 30
"""The total number of simulated features"""

R_CONTACT_FEATURES = 2/3
"""The ratio of features for which the algorithm simulates contact"""

P_CONTACT = 0.8
"""The probability of 1 in the contact zones"""

P_SUCCESS_MIN = 0.05
P_SUCCESS_MAX = 0.95
"""The minimum and maximum probability of 1 in the background distribution"""

# Sampling parameters
N_SAMPLES = 10
"""int: Number of generated samples."""

N_STEPS = 4000
"""int: Number of MCMC steps."""

BURN_IN_STEPS = 100
"""int: Number of steps before the first sample."""

MIN_SIZE = 5
"""int: The minimum size for the contact zones."""

MAX_SIZE = 100
"""int: The maximum size for the contact zones."""

ALPHA_ANNEALING = 1.
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

RESTART_INTERVAL = 2
"""bool: Restart the chain from a random point after RESTART_INTERVAL sample.
To deactivate restarting completely, set it to float('inf')"""

SIMULATED_ANNEALING = 1
"""bool: Slowly increase a temperature parameter to smoothly blend from sampling from a
uniform distribution to the actual likelihood. Should help with separated modes."""

RELOAD_DATA = 1
"""bool: Reload the data from the DB, pre-process it and dump it."""

NUMBER_PARALLEL_ZONES = 4
"""int: Number of parallel contact zones."""

PLOT_SAMPLES = True
"""bool: Plot the every sample during MCMC run."""


# Parameters for likelihood
m = 0

FEATURE_LIKELIHOOD_MODES = ['generative', 'particularity']
FEATURE_LL_MODE = FEATURE_LIKELIHOOD_MODES[m]
"""str: Switch for which feature-likelihood to use."""

GEO_PRIOR_MODES = ['generative', 'particularity', 'none']
GEO_PRIOR_MODE = GEO_PRIOR_MODES[m]
"""str: Switch for the geo-prior to use."""

GEO_ECDF_TYPES = ['mst', 'delaunay', 'complete']
GEO_ECDF_TYPE = GEO_ECDF_TYPES[0]
"""str: Switch for the type of sub-graph to use for the empirical geo-prior."""

GEO_PRIOR_WEIGHT = 8.
"""float: The weight of the geo-prior as compared to the feature likelihood."""

SAMPLES_PER_ZONE_SIZE = 8000
"""int: The number of samples for generating the empirical geo-prior."""


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
RANDOM_WALK_COV_PATH = 'data/processed/random_walk_cov.pkl'

# Path for results
MCMC_RESULTS_PATH = 'data/results/mcmc_results.pkl'
TEST_RESULTS_PATH = 'data/results/test'


