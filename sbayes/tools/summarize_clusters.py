"""Write a cluster summary table for every run in a results directory.

The summary of a run holds the mean membership of each object in each cluster, after
the burn-in, as a TSV table (`cluster_summary_K{K}_{run}.tsv`, next to the samples
file). It has the same format as the cluster reference tables (see
`sbayes.cluster_table`), so it is easy to read without the samples file, and any
summary can serve as a reference. The object IDs and names are taken from the samples
file; files written by earlier versions have neither, and their tables only list the
object indices.

With --reference, each summary is aligned to a cluster reference, taking its cluster
names and order. Without it, the clusters keep the labels of the run, which are not
comparable across runs.

Usage:
    sbayesSummarizeClusters results/my_experiment
    sbayesSummarizeClusters results/exp_a results/exp_b --reference results/exp_a
"""
from __future__ import annotations

import argparse

from pathlib import Path
from sbayes.cluster_table import (
    ClusterReference, find_reference_files, match_to_reference, summary_file_name
)
from sbayes.results import read_h5_samples
from sbayes.tools.align_clusters import Run, find_runs


def summarize_run(
    run: Run,
    burn_in: float = 0.1,
    reference: ClusterReference | None = None,
    warn_below: float = 0.5,
) -> Path:
    """Write the cluster summary of one run.

    Args:
        run: the run to summarize
        burn_in: proportion of the chain to discard before averaging
        reference: if given, align the clusters to this reference
        warn_below: report clusters whose overlap with the reference is below this

    Returns:
        The path of the summary file.

    Raises:
        ValueError: if `reference` does not match the run.
    """
    metadata, arrays = read_h5_samples(run.samples_path, include_derived=False)
    clusters = arrays["clusters"]
    mean_clusters = clusters[int(burn_in * len(clusters)):].mean(axis=0)

    message = ""
    cluster_names = None
    if reference is not None:
        permutation, overlap = match_to_reference(
            mean_clusters, reference, metadata.get("object_ids")
        )
        mean_clusters = mean_clusters[permutation]
        cluster_names = reference.cluster_names
        message = "  overlap: " + "  ".join(
            f"{name} {o:.2f}" for name, o in zip(cluster_names, overlap)
        )
        poor = [name for name, o in zip(cluster_names, overlap) if o < warn_below]
        if poor:
            message += (
                f"\n  WARNING: {', '.join(poor)} overlap less than {warn_below:.2f} "
                f"with the reference."
            )

    summary = ClusterReference.from_metadata(mean_clusters, metadata, cluster_names)
    path = run.samples_path.parent / summary_file_name(run.n_clusters, run.run_id)
    summary.write(path)
    print(f"Cluster summary: {path}" + (f"\n{message}" if message else ""))
    return path


def main(
    paths: list[Path],
    k: int | None = None,
    burn_in: float = 0.1,
    reference_path: Path | None = None,
    warn_below: float = 0.5,
) -> list[Path]:
    """Write the cluster summaries of all runs found in `paths`.

    Args:
        paths: directories with runs (see `find_runs`)
        k: only summarize the runs with this number of clusters
        burn_in: proportion of each chain to discard before averaging
        reference_path: a cluster reference file, or a directory with one file per
            number of clusters, to align the summaries to. Runs without a reference
            for their number of clusters keep their own labels.
        warn_below: report clusters whose overlap with the reference is below this

    Returns:
        The paths of the summary files.
    """
    if not 0.0 <= burn_in < 1.0:
        raise ValueError(f"`burn_in` must be in [0, 1), but was {burn_in}.")

    references = {}
    if reference_path is not None:
        references = {
            n: ClusterReference.read(p)
            for n, p in find_reference_files(reference_path).items()
        }
        if not references:
            raise ValueError(f"No cluster reference found in {reference_path}.")

    written = []
    for path in paths:
        if not path.is_dir():
            raise ValueError(f"{path} is not a directory.")
        for run in find_runs(path):
            if k is not None and run.n_clusters != k:
                continue
            reference = references.get(run.n_clusters)
            if references and reference is None:
                print(f"No cluster reference with K={run.n_clusters}: {run} keeps its labels.")
            written.append(summarize_run(
                run, burn_in=burn_in, reference=reference, warn_below=warn_below,
            ))

    if not written:
        print("No runs found.")
    return written


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Write the mean cluster memberships of every sBayes run as a TSV "
                    "table (cluster_summary_K{K}_{run}.tsv, next to the samples file)."
    )
    parser.add_argument(
        "paths", type=Path, nargs="+",
        help="Results directories, either with K* subdirectories or with samples "
             "files directly.",
    )
    parser.add_argument("-k", type=int, default=None,
                        help="Only summarize the runs with this number of clusters.")
    parser.add_argument("--burn-in", type=float, default=0.1,
                        help="Proportion of each chain to discard (default: 0.1).")
    parser.add_argument(
        "--reference", type=Path, default=None,
        help="Cluster reference file, or a directory with cluster_reference_K{K}.tsv "
             "files, to align the summaries to.",
    )
    parser.add_argument("--warn-below", type=float, default=0.5,
                        help="Warn about clusters whose overlap with the reference is "
                             "below this (default: 0.5).")
    args = parser.parse_args()
    main(
        paths=args.paths,
        k=args.k,
        burn_in=args.burn_in,
        reference_path=args.reference,
        warn_below=args.warn_below,
    )


if __name__ == "__main__":
    cli()
