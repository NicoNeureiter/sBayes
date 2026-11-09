import subprocess

from pathlib import Path


from sbayes.simulate.calibration import (
    run_study,
    SimulateConfigFile,
    prepare_replicates,
    collect_ranks,
    gate_replicates,
    plot_rank_ecdf,
    write_study_report,
    run_no_data_check
)

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
    run_no_data_check(config, name="test", n_replicates=1, bucket=bucket)


    # Reached only if the run and upload succeeded; a failure raises above
    # and leaves the VM up so the results can be retrieved from disk.
    if on_gce:
        print("Shutting down VM to save costs.")
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
