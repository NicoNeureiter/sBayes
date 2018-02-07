#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
from scipy.stats import binom_test


def compute_likelihood(zone, features, ll_lookup):

    """This function computes the likelihood of a zone.
    The function performs a two-sided binomial test. The test computes the probability of
    the observed presence/absence in the zone given the presence/absence in the data.
    Then the function takes the logarithm of the binomial test and multiplies it by -1.
    If the data in the zone follow the general trend, the zone is not indicative of
    language contact. When the binomial test yields a high value, the negative logarithm
    is small."""

    # Retrieve all languages in the zone
    idx = zone.nonzero()[0]
    zone_size = len(idx)

    # Count the presence and absence
    present = features[idx].sum(axis=0)
    log_lh = 0

    for f_idx, f_freq in enumerate(present):
        log_lh += ll_lookup[f_idx][zone_size][f_freq]

    return log_lh


def lookup_log_likelihood(min_size, max_size, feat_prob):
    """This function generates a lookup table of likelihoods
    :In
    - min_size: the minimum number of languages in a zone
    - max_size: the maximum number of languages in a zone
    - feat_prob: the probability of a feature to be present
    :Out
    - lookup_dict: the lookup table of likelihoods for a specific feature, sample size and presence
    """

    lookup_dict = {}
    for i_feat, p_global in enumerate(feat_prob):
        lookup_dict[i_feat] = {}
        for s in range(min_size, max_size + 1):
            lookup_dict[i_feat][s] = {}
            for p_zone in range(s + 1):
                lookup_dict[i_feat][s][p_zone] = -np.log(binom_test(p_zone, s, p_global))

    return lookup_dict
