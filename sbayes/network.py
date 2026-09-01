#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pyproj
import scipy.spatial as spatial

from collections import Counter
from dataclasses import dataclass
from logging import Logger
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Self, Sequence

from sbayes.util import PathLike
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from sbayes.load_data import Objects


@dataclass(eq=False)
class Network:

    """A graph over a set of objects, together with their pairwise distances.

    The edges are derived from a Delaunay triangulation of the object locations. The
    distances are either Euclidean (in the units of the given locations) or geodesic in
    metres, depending on whether a coordinate reference system was provided.

    Use `Network.from_objects` to build a network from an `Objects` instance; the
    constructor itself only assembles precomputed parts.
    """

    vertices: list[str]
    """The IDs of the objects, one per vertex."""

    names: list[str]
    """The names of the objects, one per vertex."""

    locations: NDArray[np.float64]
    """The coordinates of each object. shape: (n_objects, 2)"""

    edges: NDArray[np.int_]
    """The edge list of the Delaunay graph, as pairs of vertex indices. shape: (n_edges, 2)"""

    adj_mat: csr_matrix
    """The adjacency matrix of the Delaunay graph. shape: (n_objects, n_objects)"""

    dist_mat: NDArray[np.float64]
    """The pairwise distances between objects. shape: (n_objects, n_objects)"""

    lat_lon: NDArray[np.float64] | None = None
    """The locations transformed to WGS84 lon/lat, or None if no CRS was used."""

    @property
    def n(self) -> int:
        """The number of objects (vertices) in the network."""
        return len(self.vertices)

    @property
    def m(self) -> int:
        """The number of edges in the network."""
        return self.edges.shape[0]

    @classmethod
    def from_objects(cls, objects: Objects, crs: pyproj.CRS | None = None) -> Self:
        """Build a network from a set of objects.

        Args:
            objects: the objects to connect, providing IDs, names and locations
            crs: the coordinate reference system the locations are given in. If given,
                distances are geodesic (in metres) instead of Euclidean.

        Returns:
            The network over the given objects.
        """
        locations = np.asarray(objects.locations, dtype=float)

        # Delaunay triangulation
        delaunay = compute_delaunay(locations)
        adj_mat = delaunay.tocsr()
        coo = delaunay.tocoo()
        edges: NDArray[np.intp] = np.column_stack((coo.row, coo.col)).astype(np.intp)

        if crs is None:
            dist_mat = euclidean_distance_matrix(locations)
            lat_lon = None
        else:
            dist_mat, lat_lon = geodesic_distance_matrix(locations, crs)

        return cls(
            vertices=objects.id,
            names=objects.names,
            locations=locations,
            edges=edges,
            adj_mat=adj_mat,
            dist_mat=dist_mat,
            lat_lon=lat_lon,
        )


def read_geo_costs_from_csv(file: PathLike, logger: Logger | None = None) -> pd.DataFrame:
    """Read a geographic cost matrix from a CSV file.

    The first column is used as the index, so the file is expected to contain one row
    and one column per object, both labelled by object name.

    Args:
        file: path to the CSV file containing the cost matrix
        logger: logger to report which file was read (optional)

    Returns:
        The cost matrix, indexed by object name along both axes. Rows and columns are
        in file order, not in the order used by the analysis.
    """
    data = pd.read_csv(file, dtype=float, index_col=0)
    if logger:
        logger.info(f"Geographical cost matrix read from {file}.")
    return data


def parse_geo_cost_matrix(
    object_names: Sequence[str],
    file: PathLike,
    logger: Logger | None = None,
) -> NDArray[np.float64]:
    """Read a geographic cost matrix from a CSV file.

    The file is expected to contain one row and one column per object, labelled by
    object name. Rows and columns are reordered to match `object_names`, so the returned
    matrix is aligned with the objects of the analysis.

    Args:
        object_names: the names of the objects, in the order used by the analysis
        file: path to the CSV file containing the cost matrix
        logger: logger to report on the file contents (optional)

    Returns:
        The symmetric cost matrix between objects. shape: (n_objects, n_objects)

    Raises:
        ValueError: if the file's objects do not match `object_names`, if labels are
            duplicated, or if the matrix contains missing values.
    """
    costs = read_geo_costs_from_csv(file, logger=logger)

    validate_cost_matrix_labels(costs.columns, object_names, file, axis="column")
    validate_cost_matrix_labels(costs.index, object_names, file, axis="row")

    names = list(object_names)
    cost_matrix = costs.loc[names, names].to_numpy(dtype=float)

    if np.isnan(cost_matrix).any():
        raise ValueError(f"The cost matrix in {file} contains missing values.")

    # Check if matrix is symmetric, if not make symmetric
    if not np.allclose(cost_matrix, cost_matrix.T):
        cost_matrix = (cost_matrix + cost_matrix.T) / 2
        if logger:
            logger.info("The cost matrix is not symmetric. It was made symmetric by "
                        "averaging the original costs in the upper and lower triangle.")

    return cost_matrix


def validate_cost_matrix_labels(
    labels: Sequence[str],
    object_names: Sequence[str],
    file: PathLike,
    axis: str,
) -> None:
    """Check that the labels of one axis of a cost matrix match the analysis' objects.

    Args:
        labels: the row or column labels found in the cost matrix file
        object_names: the names of the objects, as used by the analysis
        file: path to the cost matrix file (used in error messages)
        axis: name of the axis being checked, e.g. "row" (used in error messages)

    Raises:
        ValueError: if labels are duplicated, missing or unexpected.
    """
    duplicates = [name for name, count in Counter(labels).items() if count > 1]
    if duplicates:
        raise ValueError(
            f"Duplicate {axis} labels in the cost matrix in {file}: "
            f"{sorted(duplicates)}."
        )

    missing = set(object_names) - set(labels)
    unexpected = set(labels) - set(object_names)
    if missing or unexpected:
        raise ValueError(
            f"The {axis} labels of the cost matrix in {file} do not match the objects "
            f"of the analysis. Missing: {sorted(missing) or 'none'}. "
            f"Unexpected: {sorted(unexpected) or 'none'}."
        )


def euclidean_distance_matrix(locations: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the pairwise Euclidean distances between locations.

    Args:
        locations: the coordinates of each object, shape: (n_objects, 2)

    Returns:
        The symmetric distance matrix, shape: (n_objects, n_objects)
    """
    diff = locations[:, None] - locations
    return np.linalg.norm(diff, axis=-1)


def geodesic_distance_matrix(
    locations: NDArray[np.float64],
    crs: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the pairwise geodesic distances between locations.

    The locations are first transformed from `crs` into WGS84 (epsg:4326), since the
    geodesic distances are computed on the ellipsoid.

    Args:
        locations: the coordinates of each object in `crs`, shape: (n_objects, 2)
        crs: the coordinate reference system the locations are given in

    Returns:
        The symmetric distance matrix in metres, shape: (n_objects, n_objects), and the
        locations transformed to WGS84 lon/lat, shape: (n_objects, 2)
    """
    try:
        from cartopy.geodesic import Geodesic
    except ImportError as e:
        raise ImportError(
            "Using a coordinate reference system (crs) requires the `cartopy` library: "
            "pip install cartopy"
        ) from e

    # always_xy=True forces (lon, lat) order for both input and output, independently of
    # the axis order declared by the CRS (epsg:4326 declares lat/lon).
    transformer = pyproj.Transformer.from_crs(crs, "epsg:4326", always_xy=True)
    lons, lats = transformer.transform(locations[:, 0], locations[:, 1])

    lon_lat = np.vstack((lons, lats)).T.astype(np.float64)

    geod = Geodesic()
    dist_mat = np.array(
        [geod.inverse(loc, lon_lat)[:, 0] for loc in lon_lat], dtype=np.float64
    )
    # noinspection PyTypeChecker
    return dist_mat, lon_lat


def compute_delaunay(locations: NDArray[np.float64]) -> csr_matrix:
    """Compute the Delaunay triangulation between a set of point locations.

    Args:
        locations: the coordinates of each object. shape: (n_objects, n_spatial_dims = 2)

    Returns:
        The symmetric adjacency matrix of the triangulation, i.e. both (i, j) and (j, i)
        are set for each edge. shape: (n_objects, n_objects)
    """
    n = len(locations)

    if n < 4:
        # Qhull lifts the points into 3D and needs 4 points for the initial simplex, so
        # it fails for n < 4. Up to 3 points are their own triangulation anyway, i.e.
        # the fully connected graph.
        return csr_matrix(~np.eye(n, dtype=bool))

    # QJ joggles the input to resolve degenerate cases (e.g. collinear locations),
    # Pp suppresses the resulting precision warnings.
    delaunay = spatial.Delaunay(locations, qhull_options="QJ Pp")

    indptr, indices = delaunay.vertex_neighbor_vertices
    data = np.ones_like(indices, dtype=bool)
    return csr_matrix((data, indices, indptr), shape=(n, n))