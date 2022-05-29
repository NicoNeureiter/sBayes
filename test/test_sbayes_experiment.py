#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import deepcopy
import unittest
from pathlib import Path

from sbayes.cli import main as sbayes_main
from sbayes.simulation import main as simulation_main


class TestExperiment(unittest.TestCase):

    """
    Test cases covering the general pipeline of sbayes experiments on simple and short runs.
    This should include data loading, configuration, mcmc setup and running the actual analysis.
    """

    CUSTOM_SETTINGS = {
        'mcmc': {
            'steps': 40,
            'samples': 20,
            'warmup': {
                'warmup_steps': 5,
                'warmup_chains': 2
            },
        },
    }

    @staticmethod
    def test_mobility_simulation():
        """Test whether mobility simulation is running without errors."""
        simulation_main('experiments/mobility_behaviour/simulation/config_simulation.json')
        print('Mobility simulation passed\n')

    @staticmethod
    def test_mobility_run():
        """Test whether mobility behaviour analysis on simulated data is running
        without errors."""
        custom_settings = TestExperiment.CUSTOM_SETTINGS
        sbayes_main(
            config='experiments/mobility_behaviour/config/config.json',
            custom_settings=custom_settings,
            experiment_name='test_mobility_run'
        )
        print('Mobility analysis passed\n')

    @staticmethod
    def test_sample_prior():
        """Test whether sampling from prior is running without errors."""
        custom_settings = deepcopy(TestExperiment.CUSTOM_SETTINGS)
        custom_settings['mcmc']['sample_from_prior'] = True
        sbayes_main(
            config='experiments/mobility_behaviour/config/config.json',
            custom_settings=custom_settings,
            experiment_name='test_mobility_run_prior',
        )
        print('Sample prior passed\n')


if __name__ == '__main__':
    unittest.main()
