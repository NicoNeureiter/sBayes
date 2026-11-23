"""Align the cluster labels of several runs and rewrite their stats files.

Cluster labels are arbitrary, so independent runs of the same analysis find the same
clusters under different labels. This tool relabels every run to match a reference
run and rewrites the stats files (`stats_K{K}_{run}.tsv`) from the samples files, so
that the traces of all runs can be compared directly, e.g. in Tracer. The samples
files are not changed: the stats files can be rebuilt from them at any time.

With --write-reference, the mean cluster memberships of the reference run are also
written to `cluster_reference_K{K}.tsv` (see `sbayes.cluster_table`). sBlot can
align results to this table, and it can be edited by hand, e.g. to reorder clusters.

Usage:
    sbayesAlignClusters results/my_experiment
    sbayesAlignClusters results/my_experiment/K5 --reference 2
    sbayesAlignClusters results/exp_a results/exp_b -k 5
    sbayesAlignClusters results/exp_a results/exp_b --write-reference
"""
from __future__ import annotations

import argparse
import json
import re
import tables

from collections import defaultdict
from dataclasses import dataclass
from numpy.typing import NDArray
from pathlib import Path
from sbayes.cluster_table import ClusterReference, reference_file_name
from sbayes.results import alignment_permutation, cluster_overlap
from sbayes.sampling.loggers import (
    build_stats_table, permute_clusters, read_stats_samples, write_stats_table
)

SAMPLES_FILE_PATTERN = re.compile(r"samples_(?:K\d+_)?(\d+)\.h5")
"""Samples file names: `samples_{run}.h5`, or `samples_K{K}_{run}.h5` in some versions."""


@dataclass(frozen=True)
class Run:
    """One run of an analysis: its samples file and the stats file written from it."""

    samples_path: Path
    run_id: int
    n_clusters: int

    @property
    def stats_path(self) -> Path:
        return self.samples_path.parent / f"stats_K{self.n_clusters}_{self.run_id}.tsv"

    def __str__(self) -> str:
        return str(self.samples_path)


def read_n_clusters(samples_path: Path) -> int:
    """Read the number of clusters from the metadata of a samples file.

    Raises:
        ValueError: if the file has no metadata.
    """
    with tables.open_file(str(samples_path), mode="r") as f:
        try:
            metadata = json.loads(f.get_node_attr(f.root, "metadata"))
        except AttributeError:
            raise ValueError(
                f"No metadata in {samples_path}. Run "
                f"'sbayesMigrateResults <results_dir>' first."
            ) from None
    return len(metadata["cluster_names"])


def find_runs(path: Path) -> list[Run]:
    """Find the runs in a directory of samples files, or in its `K*` subdirectories.

    Args:
        path: a directory holding `samples_*.h5` files, or the results directory of
            an experiment, with one `K{K}` subdirectory per number of clusters

    Returns:
        The runs, sorted by number of clusters and run ID.
    """
    directories = [path, *sorted(d for d in path.glob("K*") if d.is_dir())]
    runs = []
    for directory in directories:
        for samples_path in sorted(directory.glob("samples_*.h5")):
            match = SAMPLES_FILE_PATTERN.fullmatch(samples_path.name)
            if match is None:
                continue
            runs.append(Run(
                samples_path=samples_path,
                run_id=int(match.group(1)),
                n_clusters=read_n_clusters(samples_path),
            ))
    return sorted(runs, key=lambda r: (r.n_clusters, r.run_id))


def mean_clusters(arrays: dict[str, NDArray], burn_in: float) -> NDArray:
    """The cluster samples after the burn-in. shape: (n_samples, n_clusters, n_objects)"""
    clusters = arrays["clusters"]
    return clusters[int(burn_in * len(clusters)):]


def align_runs(
    runs: list[Run],
    reference: Run,
    burn_in: float = 0.1,
    warn_below: float = 0.5,
    reference_dir: Path | None = None,
) -> dict[Run, NDArray]:
    """Align the clusters of `runs` to `reference` and rewrite their stats files.

    The runs are processed one at a time, so that only the reference and one other run
    are held in memory.

    Args:
        runs: the runs to align, all with the same number of clusters
        reference: the run whose cluster labels the others are matched to (its stats
            file is rewritten as well, with the labels of the within-run matching)
        burn_in: proportion of each chain ignored when matching the clusters. The
            stats files keep all samples.
        warn_below: report clusters whose overlap with the reference is below this
        reference_dir: if given, write the mean cluster memberships of the reference
            run (after the burn-in) to `cluster_reference_K{K}.tsv` in this directory

    Returns:
        The overlap of each cluster with the reference, per run.

    Raises:
        ValueError: if the runs do not describe the same number of clusters and objects.
    """
    ref_metadata, ref_arrays = read_stats_samples(reference.samples_path)
    ref_clusters = mean_clusters(ref_arrays, burn_in)
    cluster_names = ref_metadata["cluster_names"]

    write_stats_table(build_stats_table(ref_metadata, ref_arrays), reference.stats_path)
    print(f"Reference: {reference.stats_path}")
    del ref_arrays

    if reference_dir is not None:
        write_reference(ref_clusters.mean(axis=0), ref_metadata, reference_dir)

    overlaps = {}
    for run in runs:
        if run == reference:
            continue

        metadata, arrays = read_stats_samples(run.samples_path)
        clusters = mean_clusters(arrays, burn_in)
        if clusters.shape[1:] != ref_clusters.shape[1:]:
            raise ValueError(
                f"{run} has {clusters.shape[1]} clusters and {clusters.shape[2]} "
                f"objects, but the reference has {ref_clusters.shape[1]} clusters and "
                f"{ref_clusters.shape[2]} objects."
            )

        permutation = alignment_permutation(ref_clusters, clusters)
        arrays = permute_clusters(metadata, arrays, permutation)
        overlap = cluster_overlap(ref_clusters.mean(axis=0), clusters[:, permutation].mean(axis=0))
        overlaps[run] = overlap

        write_stats_table(build_stats_table(metadata, arrays), run.stats_path)

        mapping = ", ".join(
            f"{cluster_names[j]}->{name}"
            for name, j in zip(cluster_names, permutation)
            if cluster_names[j] != name
        )
        print(f"Aligned:   {run.stats_path}  ({mapping or 'labels unchanged'})")
        print("           overlap: " + "  ".join(
            f"{name} {o:.2f}" for name, o in zip(cluster_names, overlap)
        ))
        poor = [name for name, o in zip(cluster_names, overlap) if o < warn_below]
        if poor:
            print(
                f"           WARNING: {', '.join(poor)} overlap less than "
                f"{warn_below:.2f} with the reference: the runs may have found "
                f"different clusters, so their traces are not comparable."
            )

    return overlaps


def write_reference(mean_clusters: NDArray, metadata: dict, directory: Path) -> Path:
    """Write the cluster reference table for one number of clusters.

    The object IDs and names are taken from the metadata of the samples file.
    """
    reference = ClusterReference.from_metadata(mean_clusters, metadata)
    path = directory / reference_file_name(reference.n_clusters)
    reference.write(path)
    print(f"Cluster reference: {path}")
    return path


def main(
    paths: list[Path],
    k: int | None = None,
    reference: int | None = None,
    burn_in: float = 0.1,
    warn_below: float = 0.5,
    reference_dir: Path | None = None,
) -> None:
    """Align the runs found in `paths`, separately for each number of clusters.

    Args:
        paths: directories with runs (see `find_runs`). The reference run is taken
            from the first one.
        k: only align the runs with this number of clusters
        reference: run ID of the reference run. None = the first run.
        burn_in: proportion of each chain ignored when matching the clusters
        warn_below: report clusters whose overlap with the reference is below this
        reference_dir: if given, write a cluster reference table per number of
            clusters to this directory (see `align_runs`)
    """
    if not 0.0 <= burn_in < 1.0:
        raise ValueError(f"`burn_in` must be in [0, 1), but was {burn_in}.")

    # Runs per number of clusters, keeping the order of `paths`
    runs_by_k: dict[int, list[Run]] = defaultdict(list)
    for path in paths:
        if not path.is_dir():
            raise ValueError(f"{path} is not a directory.")
        for run in find_runs(path):
            runs_by_k[run.n_clusters].append(run)

    if k is not None:
        runs_by_k = {k: runs_by_k[k]} if k in runs_by_k else {}
    if not runs_by_k:
        print("No runs found.")
        return

    for n_clusters, runs in sorted(runs_by_k.items()):
        print(f"\nK={n_clusters}: {len(runs)} runs")
        if reference is None:
            ref_run = runs[0]
        else:
            candidates = [
                r for r in runs
                if r.run_id == reference and r.samples_path.is_relative_to(paths[0])
            ]
            if not candidates:
                print(f"No run {reference} with K={n_clusters} in {paths[0]}, skipped.")
                continue
            ref_run = candidates[0]

        align_runs(
            runs, ref_run, burn_in=burn_in, warn_below=warn_below,
            reference_dir=reference_dir,
        )


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Align the cluster labels of several sBayes runs and rewrite their "
                    "stats files from the samples files, so the traces can be compared "
                    "(e.g. in Tracer)."
    )
    parser.add_argument(
        "paths", type=Path, nargs="+",
        help="Results directories, either with K* subdirectories or with samples "
             "files directly. The reference run is taken from the first one.",
    )
    parser.add_argument("-k", type=int, default=None,
                        help="Only align the runs with this number of clusters.")
    parser.add_argument("--reference", type=int, default=None,
                        help="Run ID of the reference run (default: the first run).")
    parser.add_argument("--burn-in", type=float, default=0.1,
                        help="Proportion of each chain ignored when matching the "
                             "clusters (default: 0.1). The stats files keep all samples.")
    parser.add_argument("--warn-below", type=float, default=0.5,
                        help="Warn about clusters whose overlap with the reference is "
                             "below this (default: 0.5).")
    parser.add_argument(
        "--write-reference", nargs="?", type=Path, const=True, default=None,
        metavar="DIR",
        help="Write the mean cluster memberships of the reference run to "
             "cluster_reference_K{K}.tsv in DIR (default: the first results "
             "directory). sBlot can align results to these tables.",
    )
    args = parser.parse_args()
    reference_dir = args.paths[0] if args.write_reference is True else args.write_reference
    main(
        paths=args.paths,
        k=args.k,
        reference=args.reference,
        burn_in=args.burn_in,
        warn_below=args.warn_below,
        reference_dir=reference_dir,
    )


if __name__ == "__main__":
    cli()
