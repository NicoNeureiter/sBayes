def lookup_log_likelihood(min_size, max_size, feat_prob, mode):
    """This function generates a lookup table of likelihoods
    Args:
        min_size (int): the minimum number of languages in a zone.
        max_size (int): the maximum number of languages in a zone.
        feat_prob (np.array): the probability of a feature to be present.
    Returns:
        dict: the lookup table of likelihoods for a specific feature,
            sample size and observed presence.
    """
    if FEATURE_LL_MODE == 'binom_test_2':
        # The binomial test computes the p-value of having k or more (!) successes out of n trials,
        # given a specific probability of success
        # For a two-sided binomial test, simply remove "greater"
        def ll(p_zone, s, p_global):
            return math.log(1 - binom_test(p_zone, s, p_global, 'greater') + EPS)

    else:
        # This version of the ll is more sensitive to exceptional observations.
        def ll(p_zone, s, p_global):
            p = binom_test(p_zone, s, p_global, 'greater')
            try:
                return - math.log(p)
            except Exception as e:
                print(p_zone, s, p_global, p)
                raise e

    lookup_dict = {}
    for i_feat, p_global in enumerate(feat_prob):
        lookup_dict[i_feat] = {}
        for s in range(min_size, max_size + 1):
            lookup_dict[i_feat][s] = {}
            for p_zone in range(s + 1):
                lookup_dict[i_feat][s][p_zone] = ll(p_zone, s, p_global)

    return lookup_dict