import subprocess

from pathlib import Path
from ruamel import yaml

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
    output_path = Path("output/no_data")
    run_no_data_check(config, output_path,
                      n_replicates=1, bucket=bucket)


    # Reached only if the run and upload succeeded; a failure raises above
    # and leaves the VM up so the results can be retrieved from disk.
    if on_gce:
        print("Shutting down VM to save costs.")
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)

    # study = run_study(config=config, output_path=Path("output"), n_replicates=3)
    #
    # prepared = prepare_replicates(study)
    # gate = gate_replicates({i: p.diagnostics for i, p in prepared.items()})
    # ranks = collect_ranks(prepared, gate, master_seed=config.simulation.seed)
    #
    # print(ranks["rank"].value_counts().head(10))
    # print(ranks.groupby("kind")["rank"].agg(["min", "max", "nunique"]))
    #
    # ranks.to_csv(study.path / "ranks.tsv", sep="\t", index=False)
    # plot_rank_ecdf(ranks, study.path / "rank_ecdf.png")
    # plot_rank_ecdf(ranks, study.path / "rank_ecdf_all.png", passed_only=False)
    # write_study_report(study, prepared, gate, ranks, study.path / "report.md")
