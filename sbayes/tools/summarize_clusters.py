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

With --combine, the runs with the same number of clusters in one directory are
summarized together (`cluster_summary_K{K}.tsv`): each run is aligned to the reference,
or to the first run without one, and the memberships are averaged over the samples of
all runs. Runs in different directories (e.g. different experiments) are never combined.

Usage:
    sbayesSummarizeClusters results/my_experiment
    sbayesSummarizeClusters results/my_experiment --combine
    sbayesSummarizeClusters results/exp_a results/exp_b --reference results/exp_a
"""
from __future__ import annotations

import argparse
import textwrap

from collections import defaultdict
from numpy.typing import NDArray
from pathlib import Path
from sbayes.cluster_table import (
    ClusterReference, find_reference_files, match_to_reference, summary_file_name
)
from sbayes.results import read_h5_samples
from sbayes.tools.align_clusters import Run, find_runs

import numpy as np


def read_mean_clusters(run: Run, burn_in: float) -> tuple[dict, NDArray, int]:
    """Read the mean cluster memberships of one run.

    Returns:
        The metadata of the samples file, the mean membership of each object in each
        cluster after the burn-in (shape: (n_clusters, n_objects)) and the number of
        samples averaged over.
    """
    metadata, arrays = read_h5_samples(run.samples_path, include_derived=False)
    clusters = arrays["clusters"]
    clusters = clusters[int(burn_in * len(clusters)):]
    return metadata, clusters.mean(axis=0), len(clusters)


def align_to(
    mean_clusters: NDArray,
    metadata: dict,
    reference: ClusterReference,
    warn_below: float,
) -> tuple[NDArray, str]:
    """Align mean memberships to a reference.

    Returns:
        The aligned memberships, and a report of the overlap of each cluster with the
        reference.
    """
    permutation, overlap = match_to_reference(
        mean_clusters, reference, metadata.get("object_ids")
    )
    names = reference.cluster_names
    message = "  overlap: " + "  ".join(f"{n} {o:.2f}" for n, o in zip(names, overlap))
    poor = [n for n, o in zip(names, overlap) if o < warn_below]
    if poor:
        message += (
            f"\n  WARNING: {', '.join(poor)} overlap less than {warn_below:.2f} "
            f"with the reference."
        )
    return mean_clusters[permutation], message


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
    metadata, mean_clusters, _ = read_mean_clusters(run, burn_in)

    message = ""
    cluster_names = None
    if reference is not None:
        mean_clusters, message = align_to(mean_clusters, metadata, reference, warn_below)
        cluster_names = reference.cluster_names

    summary = ClusterReference.from_metadata(mean_clusters, metadata, cluster_names)
    path = run.samples_path.parent / summary_file_name(run.n_clusters, run.run_id)
    summary.write(path)
    print(f"Cluster summary: {path}" + (f"\n{message}" if message else ""))
    return path


def summarize_combined(
    runs: list[Run],
    burn_in: float = 0.1,
    reference: ClusterReference | None = None,
    warn_below: float = 0.5,
) -> Path:
    """Write one cluster summary of several runs with the same number of clusters.

    Each run is aligned to `reference`, or to the first run without one, and the
    aligned memberships are averaged, weighted by the number of samples of each run.
    That is the mean over the samples of all runs together.

    Args:
        runs: the runs to combine, all in one directory
        burn_in: proportion of each chain to discard before averaging
        reference: if given, align the clusters to this reference
        warn_below: report clusters whose overlap with the target is below this

    Returns:
        The path of the summary file.

    Raises:
        ValueError: if the runs do not describe the same objects, or `reference` does
            not match them.
    """
    first_metadata, first_mean, _ = read_mean_clusters(runs[0], burn_in)
    target = reference or ClusterReference.from_metadata(first_mean, first_metadata)
    object_ids = first_metadata.get("object_ids")
    print(
        f"Combining {len(runs)} runs, aligned to "
        f"{'the cluster reference' if reference else runs[0]}:"
    )

    total = np.zeros_like(first_mean)
    n_total = 0
    for run in runs:
        metadata, mean_clusters, n_samples = read_mean_clusters(run, burn_in)
        if mean_clusters.shape != first_mean.shape or metadata.get("object_ids") != object_ids:
            raise ValueError(
                f"{run} does not describe the same clusters and objects as {runs[0]}."
            )
        aligned, message = align_to(mean_clusters, metadata, target, warn_below)
        print(f"  {run}\n" + textwrap.indent(message, "  "))
        total += n_samples * aligned
        n_total += n_samples

    summary = ClusterReference.from_metadata(
        total / n_total, first_metadata, target.cluster_names
    )
    path = runs[0].samples_path.parent / summary_file_name(runs[0].n_clusters)
    summary.write(path)
    print(f"Cluster summary: {path}")
    return path


def main(
    paths: list[Path],
    k: int | None = None,
    burn_in: float = 0.1,
    reference_path: Path | None = None,
    warn_below: float = 0.5,
    combine: bool = False,
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
        combine: write one summary of all runs with the same number of clusters in a
            directory, instead of one per run

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

    runs = []
    for path in paths:
        if not path.is_dir():
            raise ValueError(f"{path} is not a directory.")
        runs += [r for r in find_runs(path) if k is None or r.n_clusters == k]

    def get_reference(n_clusters: int, label: object) -> ClusterReference | None:
        reference = references.get(n_clusters)
        if references and reference is None:
            print(f"No cluster reference with K={n_clusters}: {label} keeps its labels.")
        return reference

    written = []
    if combine:
        groups: dict[tuple[Path, int], list[Run]] = defaultdict(list)
        for run in runs:
            groups[(run.samples_path.parent, run.n_clusters)].append(run)
        for (directory, n_clusters), group in groups.items():
            reference = get_reference(n_clusters, directory)
            written.append(summarize_combined(
                group, burn_in=burn_in, reference=reference, warn_below=warn_below,
            ))
    else:
        for run in runs:
            written.append(summarize_run(
                run, burn_in=burn_in, reference=get_reference(run.n_clusters, run),
                warn_below=warn_below,
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
    parser.add_argument(
        "--combine", action="store_true",
        help="Write one summary (cluster_summary_K{K}.tsv) of all runs with the same "
             "number of clusters in a directory, aligned to the reference or to the "
             "first run.",
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
        combine=args.combine,
    )


if __name__ == "__main__":
    cli()
