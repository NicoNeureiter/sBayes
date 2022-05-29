from pathlib import Path
from sbayes.simulation import Simulation


if __name__ == '__main__':
    sim = Simulation()
    sim.load_config_simulation(config_file=Path("experiments/mobility_behaviour/simulation/config_simulation.json"))

    # Simulate mobility behaviour
    sim.run_simulation()
    sim.write_to_csv()
