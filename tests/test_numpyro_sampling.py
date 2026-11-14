"""Tests for sbayes/sampling/numpyro_sampling.py."""

import numpy as np

from sbayes.sampling.numpyro_sampling import MAX_TREE_DEPTH, summarize_steps


def test_summarize_steps_counts_divergences_and_saturated_trees():
    max_steps = 2 ** MAX_TREE_DEPTH - 1
    diverging = np.array([False] * 199 + [True])
    num_steps = np.array([max_steps, max_steps] + [10] * 198)
    lines = summarize_steps(diverging, num_steps)
    assert lines[0] == "Divergent steps: 1 / 200 (0.5%)"
    assert lines[1] == "Steps at the maximum tree depth: 2 / 200 (1.0%)"
    assert lines[2].startswith("Mean number of leapfrog steps:")
    assert len(lines) == 3   # 0.5% divergent: no warning


def test_summarize_steps_warns_about_many_divergences():
    lines = summarize_steps(np.array([True, False, False]), np.ones(3))
    assert lines[-1].startswith("Warning:")
