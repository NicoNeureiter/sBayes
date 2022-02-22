#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import unittest

from sbayes.plot import Plot, main


class TestPlot(unittest.TestCase):

    """
    Test cases of plotting functions in ´sbayes/plot.py´.
    """

    @staticmethod
    def test_example():
        main(
            config='test/plot_test_files/config_plot.json',
            plot_types=['map'],
        )
