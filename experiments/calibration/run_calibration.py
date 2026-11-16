import subprocess

from pathlib import Path
from sbayes.simulate.calibration import (
    SimulateConfigFile,
    run_no_data_check, run_variant
)

VARIANTS: dict[str, dict[str, int]] = {
    "categorical": {"categorical_2": 10, "categorical_3": 20, "categorical_4": 10},
    "gaussian": {"gaussian": 40},
    "poisson": {"poisson": 40},
    "logitnormal": {"logitnormal": 40},
    "mixed": {"categorical_3": 10, "gaussian":10, "poisson":10, "logitnormal":10},
}
"""The feature types of each study variant. Everything else is shared."""


def are_we_on_gce():
    try:
        with open("/sys/class/dmi/id/product_name") as f:
            return "Google Compute Engine" in f.read()
    except OSError:
        return False


if __name__ == "__main__":
    config = SimulateConfigFile.from_config_file(Path("config_simulate.yaml"))
    on_gce = are_we_on_gce()
    bucket = "sbayes-bucket" if on_gce else None
    # run_no_data_check(config, n_replicates=100, bucket=bucket)

    for variant, feature_types in VARIANTS.items():
        if variant == "poisson":
            run_variant(config, name=variant, feature_types=feature_types,
                        n_replicates=100, bucket=bucket)

    # Reached only if the run and upload succeeded; a failure raises above
    # and leaves the VM up so the results can be retrieved from disk.
    if on_gce:
        print("Shutting down VM to save costs.")
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)