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
N_STEPS = 5000
"""int: Number of MCMC steps."""

N_SAMPLES = 200
"""int: Number of generated samples."""

MIN_SIZE = 5
"""int: The minimum size for the contact zones."""

MAX_SIZE = 40
"""int: The maximum size for the contact zones."""

P_SWAP = 1.0
"""float: Frequency of 'swap' steps, where a node gets replaced by another (size 
remains constant)."""

ALPHA_ANEALING = 1.5
"""float: The parameter controlling the cooling schedule for the simulated annealing 
temperature."""


# Config flags
LIKELIHOOD_MODES = ['generative', 'binom_test', 'binom_test_2']
"""list: All implemented modes for the likelihood function."""

LL_MODE = LIKELIHOOD_MODES[2]
"""str: The switch for which likelihood to use."""

GEO_LIKELIHOOD_MODES = ['None', 'Gaussian', 'Empirical']
"""list: All implemented modes for the geo-likelihood"""

GEO_LL_MODE = GEO_LIKELIHOOD_MODES[1]
"""str: The switch for the geo-likelihood to use."""

RESTART_CHAIN = True
"""bool: Restart the chain from a random point after every sample."""

SIMULATED_ANNEALING = True
"""bool: Slowly increase a temperature parameter to smoothly blend from sampling from a
uniform distribution to the actual likelihood. Should help with separated modes."""

RELOAD_DATA = False
"""bool: Reload the data from the DB, pre-process it and dump it."""

# Parameters for geo-likelihood
GEO_LIKELIHOOD_WEIGHT = 5.
"""float: The weight of the geo-likelihood as compared to the feature likelihood"""

SAMPLES_PER_ZONE_SIZE = 10000
"""int: The number of samples for generating the empirical geo-likelihood"""


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



