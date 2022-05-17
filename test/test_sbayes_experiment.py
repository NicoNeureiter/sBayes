#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import deepcopy
import unittest
from pathlib import Path

from sbayes.experiment_setup import Experiment
from sbayes.simulation import Simulation
from sbayes.mcmc_setup import MCMCSetup
from sbayes.cli import main


class TestExperiment(unittest.TestCase):

    """
    Test cases covering the general pipeline of sbayes experiments on simple and short runs.
    This should include data loading, configuration, mcmc setup and running the actual analysis.
    """

    CUSTOM_SETTINGS = {
        'simulation': {
            'n_features': 5,
            'i_contact': 3,
            'e_contact': 0.5,
            'clusters': 3,
            'correlation_threshold': 0.8,
        },
        'model': {},
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
    def test_sim_exp1():
        """Test whether simulation experiment 1 is running without errors."""
        custom_settings = TestExperiment.CUSTOM_SETTINGS
        main(
            config=Path('experiments/simulation/sim_exp1/config.json'),
            custom_settings=custom_settings,
            experiment_name='test_sim_exp1',
        )

        print('Experiment 1 passed\n')

    @staticmethod
    def test_sim_exp2():
        """Test whether simulation experiment 2 is running without errors."""
        custom_settings = deepcopy(TestExperiment.CUSTOM_SETTINGS)
        custom_settings['model']['inheritance'] = True
        main(
            config=Path('experiments/simulation/sim_exp2/config.json'),
            custom_settings=custom_settings,
            experiment_name='test_sim_exp2',
        )

        print('Experiment 2 passed\n')

    @staticmethod
    def test_sim_exp3():
        """Test whether simulation experiment 3 is running without errors."""
        custom_settings = deepcopy(TestExperiment.CUSTOM_SETTINGS)
        custom_settings['model']['clusters'] = 2
        main(
            config=Path('experiments/simulation/sim_exp3/config.json'),
            custom_settings=custom_settings,
            experiment_name='test_sim_exp3',
        )
        print('Experiment 3 passed\n')

    @staticmethod
    def test_sample_prior():
        """Test whether sampling from prior is running without errors."""
        custom_settings = deepcopy(TestExperiment.CUSTOM_SETTINGS)
        custom_settings['mcmc']['sample_from_prior'] = True
        main(
            config=Path('experiments/simulation/sim_exp1/config.json'),
            custom_settings=custom_settings,
            experiment_name='test_sample_prior',
        )

        print('Sample prior passed\n')


if __name__ == '__main__':
    unittest.main()
