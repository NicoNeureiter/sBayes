#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np


def compute_likelihood(diffusion, features, ll_lookup):

    """This function computes the likelihood of a diffusion.
    The function performs a two-sided binomial test. The test computes the probability of the observed presence/absence
    in the diffusion given the presence/absence in the data. Then the function takes the logarithm of the binomial
    test and multiplies it by -1. If the data in the diffusion follow the general trend, the diffusion is not indicative
    of language contact. When the binomial test yields a high value, the negative logarithm is small."""

    # Retrieve all languages in the diffusion
    idx = diffusion
    diffusion_size = len(diffusion)

    # Count the presence and absence
    present = np.sum(features[idx], axis=0)
    log_lh = 0

    for f_idx, f_freq in enumerate(present):
        log_lh += ll_lookup[f_idx][diffusion_size][f_freq]

    return log_lh
