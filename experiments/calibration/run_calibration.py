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


if __name__ == "__main__":
    config = SimulateConfigFile.from_config_file(
        Path("config_simulate.yaml"), )

    # Validate the pipeline: no likelihood on either side, so the ranks must be
    # uniform by construction.
    run_no_data_check(config, Path("output/no_data"), n_replicates=100)

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
