#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from copy import deepcopy
import unittest
import jax.random as random

from sbayes.cli import main as sbayes_main
from sbayes.simulate.config import load_config
from sbayes.simulate.simulator import Simulator

SIMULATE_CONFIG_PATH = "experiments/simulation/config_simulate.yaml"
RNG_SEED = 0

class TestExperiment(unittest.TestCase):

    """
    Test cases covering the general pipeline of sbayes experiments on simple and short runs.
    This should include data loading, configuration, mcmc setup and running the actual analysis.
    """

    CUSTOM_SETTINGS = {
        "mcmc": {
            "steps": 40,
            "samples": 20,
            "runs": 2,
            "warmup": {"warmup_steps": 5, "warmup_chains": 2},
        },
    }

    @staticmethod
    def test_simulation_and_run():
        """Simulation and subsequent inference run without errors on simulated data."""
        sim_config = load_config(SIMULATE_CONFIG_PATH)
        sim = Simulator(sim_config)
        sim.prepare_simulation()
        sim.simulate(random.PRNGKey(RNG_SEED))
        sim.write_simulation(write_parameters=False)
        sim.infer()
        print("Simulation and inference passed\n")


    @staticmethod
    def test_south_america_run():
        """Test whether south america case study is running without errors."""
        custom_settings = deepcopy(TestExperiment.CUSTOM_SETTINGS)
        sbayes_main(
            config="experiments/south_america/config.yaml",
            custom_settings=custom_settings,
            experiment_name="test_south_america_run",
        )
        print("South america analysis passed\n")


    @staticmethod
    def test_custom_settings_as_args():
        """Test whether south america case study is running without errors."""
        custom_settings = deepcopy(TestExperiment.CUSTOM_SETTINGS)
        custom_settings["mcmc"]["runs"] = 1
        sbayes_main(
            config="experiments/south_america/config.yaml",
            custom_settings=custom_settings,
            experiment_name="test_south_america_run",
            i_run=7,
            n_clusters=[2, 3],
        )

        print("South america analysis passed\n")

    @staticmethod
    def test_sample_prior():
        """Test whether sampling from prior is running without errors."""
        custom_settings = deepcopy(TestExperiment.CUSTOM_SETTINGS)
        custom_settings["mcmc"]["sample_from_prior"] = True

        sbayes_main(
            config="experiments/mobility_behaviour/config.yaml",
            custom_settings=custom_settings,
            experiment_name="test_mobility_run_prior",
        )
        print("Sample prior passed\n")


if __name__ == "__main__":
    unittest.main()
svi