#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

# Configuration
MAX_SIZE = 50
"""int: The maximum size for the contact zones."""

RELOAD_DATA = False
"""bool: Reload the data from the DB, preprocess is and dump it."""

# Paths for dump files
NETWORK_PATH = 'data/processed/network.pkl'
FEATURES_PATH = 'data/processed/features.pkl'
FEATURE_PROB_PATH = 'data/processed/feature_prob.pkl'
LOOKUP_TABLE_PATH = 'data/processed/lookup_table.pkl'

# Path for results
PATH_MCMC_RESULTS = 'data/results/mcmc_results.csv'