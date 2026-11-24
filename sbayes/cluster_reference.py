"""Cluster reference tables: a fixed cluster order to align results to.

A cluster reference is a TSV table with the mean membership of each object (rows) in
each cluster (columns), one file per number of clusters (`cluster_reference_K{K}.tsv`).
Results are aligned to it by matching their mean memberships to the reference columns,
so the cluster labels (and the colours derived from them) stay the same across
experiments and across separate plotting sessions.

The table is meant to be human-readable and editable:
- the column order sets the cluster order (e.g. the colours on a map)
- the column names set the cluster names
- the `object` column holds the object index in the data, and the optional `id` and
  `name` columns identify the objects. If `id` is given, objects are matched by ID.
"""
from __future__ import annotations

import re

from dataclasses import dataclass
from numpy.typing import NDArray
from pathlib import Path
from sbayes.results import Results, cluster_overlap, match_mean_clusters
from sbayes.util import PathLike

import numpy as np
import pandas as pd

OBJECT_COLUMNS = ("object", "id", "name")
"""Columns that identify the objects. All other columns are clusters."""

FILE_PATTERN = re.compile(r"cluster_reference_K(\d+)\.tsv")


def reference_file_name(n_clusters: int) -> str:
    return f"cluster_reference_K{n_clusters}.tsv"


@dataclass
class ClusterReference:
    """The reference memberships for one number of clusters.

    Attributes:
        cluster_names: the cluster names, in reference order
        mean_clusters: mean membership of each object in each cluster.
            shape: (n_clusters, n_objects)
        object_ids: the ID of each object, if known
        object_names: the name of each object, if known
    """

    cluster_names: list[str]
    mean_clusters: NDArray
    object_ids: list[str] | None = None
    object_names: list[str] | None = None

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_names)

    @property
    def n_objects(self) -> int:
        return self.mean_clusters.shape[1]

    @classmethod
    def from_results(
        cls,
        results: Results,
        object_ids: list | None = None,
        object_names: list | None = None,
    ) -> ClusterReference:
        return cls(
            cluster_names=list(results.cluster_names),
            mean_clusters=results.mean_clusters(),
            object_ids=None if object_ids is None else [str(i) for i in object_ids],
            object_names=None if object_names is None else [str(n) for n in object_names],
        )

    def write(self, path: PathLike, decimals: int = 3) -> None:
        """Write the reference as a TSV table."""
        table = pd.DataFrame({"object": np.arange(self.n_objects)})
        if self.object_ids is not None:
            table["id"] = self.object_ids
        if self.object_names is not None:
            table["name"] = self.object_names
        for name, column in zip(self.cluster_names, self.mean_clusters):
            table[name] = np.round(column, decimals)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, sep="\t", index=False)

    @classmethod
    def read(cls, path: PathLike) -> ClusterReference:
        """Read a reference table, with the rows sorted by the `object` column.

        Raises:
            ValueError: if the table has no cluster columns or duplicate objects.
        """
        table = pd.read_csv(path, sep="\t", dtype={"id": str, "name": str})
        cluster_names = [c for c in table.columns if c not in OBJECT_COLUMNS]
        if not cluster_names:
            raise ValueError(f"{path} has no cluster columns.")
        if "object" in table:
            if table["object"].duplicated().any():
                raise ValueError(f"{path} lists some objects more than once.")
            table = table.sort_values("object")
        if "id" in table and table["id"].duplicated().any():
            raise ValueError(f"{path} lists some object IDs more than once.")

        return cls(
            cluster_names=cluster_names,
            mean_clusters=table[cluster_names].to_numpy(dtype=float).T,
            object_ids=table["id"].tolist() if "id" in table else None,
            object_names=table["name"].tolist() if "name" in table else None,
        )

    def mean_clusters_for(self, object_ids: list | None, n_objects: int) -> NDArray:
        """The reference memberships, with columns in the order of `object_ids`.

        Objects are matched by ID if both the reference and `object_ids` have them,
        otherwise by their position.

        Raises:
            ValueError: if the objects do not match.
        """
        if self.object_ids is not None and object_ids is not None:
            index = {object_id: i for i, object_id in enumerate(self.object_ids)}
            missing = [str(o) for o in object_ids if str(o) not in index]
            if missing:
                raise ValueError(
                    f"{len(missing)} objects are not in the cluster reference: "
                    f"{', '.join(missing[:10])}{', ...' if len(missing) > 10 else ''}"
                )
            return self.mean_clusters[:, [index[str(o)] for o in object_ids]]

        if self.n_objects != n_objects:
            raise ValueError(
                f"The cluster reference has {self.n_objects} objects, but the results "
                f"have {n_objects}."
            )
        return self.mean_clusters


def find_reference_files(path: PathLike) -> dict[int, Path]:
    """The reference files in `path` (a file or a directory), by number of clusters."""
    path = Path(path)
    if path.is_file():
        return {ClusterReference.read(path).n_clusters: path}
    files = {}
    for file in sorted(path.glob("cluster_reference_K*.tsv")):
        match = FILE_PATTERN.fullmatch(file.name)
        if match is not None:
            files[int(match.group(1))] = file
    return files


def align_to_reference(
    results: Results,
    reference: ClusterReference,
    object_ids: list | None = None,
) -> tuple[Results, NDArray]:
    """Relabel the clusters of `results` to match `reference`.

    Args:
        results: the results to align
        reference: the reference, with the same number of clusters
        object_ids: the ID of each object in `results`, used to match the objects to
            the reference. None = match by position.

    Returns:
        The aligned results, which take the reference cluster names, and the overlap of
        each cluster with its reference cluster (see `cluster_overlap`).

    Raises:
        ValueError: if the number of clusters or the objects do not match.
    """
    if reference.n_clusters != results.n_clusters:
        raise ValueError(
            f"The cluster reference has {reference.n_clusters} clusters, but the "
            f"results have {results.n_clusters}."
        )
    ref_mean = reference.mean_clusters_for(object_ids, results.n_objects)
    mean = results.mean_clusters()
    permutation = match_mean_clusters(ref_mean, mean)
    overlap = cluster_overlap(ref_mean, mean[permutation])
    return results.permute_clusters(permutation, reference.cluster_names), overlap
