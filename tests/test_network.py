"""Tests for sbayes/network.py."""
import numpy as np
import pytest
from pyproj import Transformer

from sbayes.load_data import Objects
from sbayes.network import (
    compute_delaunay, euclidean_distance_matrix, geodesic_distance_matrix, Network,
)

# A unit square: the Delaunay triangulation has 4 sides + 1 diagonal = 5 edges.
SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def make_objects(locations: np.ndarray) -> Objects:
    """Build a minimal Objects instance from locations alone."""
    n = len(locations)
    ids = [f"o{i}" for i in range(n)]
    return Objects(id=ids, locations=locations, names=[f"name{i}" for i in range(n)])


class TestComputeDelaunay:

    def test_square(self):
        adj = compute_delaunay(SQUARE).toarray()
        # No self-loops, and every vertex is connected to at least two others.
        assert not adj.diagonal().any()
        assert adj.sum(axis=1).min() >= 2
        # 5 undirected edges, stored in both directions.
        assert adj.sum() == 10

    def test_matrix_is_symmetric(self):
        # The rest of the code (edge list, m) depends on this.
        adj = compute_delaunay(SQUARE).toarray()
        assert np.array_equal(adj, adj.T)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_fewer_than_four_objects_is_fully_connected(self, n):
        # Qhull needs 4 points; up to 3 points are their own triangulation.
        adj = compute_delaunay(SQUARE[:n]).toarray()
        assert np.array_equal(adj, ~np.eye(n, dtype=bool))

    def test_collinear_locations(self):
        # Degenerate input must not raise: this is what qhull's QJ option is for.
        locations = np.array([[float(i), 0.0] for i in range(5)])
        adj = compute_delaunay(locations).toarray()
        assert np.array_equal(adj, adj.T)
        assert adj.sum(axis=1).min() >= 1


class TestDistanceMatrices:

    def test_euclidean_distances(self):
        dist = euclidean_distance_matrix(SQUARE)
        assert np.allclose(dist.diagonal(), 0.0)
        assert np.allclose(dist, dist.T)
        assert dist[0, 1] == pytest.approx(1.0)
        assert dist[0, 3] == pytest.approx(np.sqrt(2))

    def test_geodesic_distances_are_metres(self):
        # Two points one degree of latitude apart, i.e. roughly 111 km.
        lon_lat = np.array([[8.55, 47.37], [8.55, 48.37]])
        dist, transformed = geodesic_distance_matrix(lon_lat, "epsg:4326")
        assert np.allclose(dist.diagonal(), 0.0)
        assert np.allclose(dist, dist.T, atol=1e-6)
        assert dist[0, 1] == pytest.approx(111_000, rel=0.01)
        assert np.allclose(transformed, lon_lat)

    def test_geodesic_distances_are_independent_of_the_crs(self):
        """Regression test: the same points in a projected CRS must give the same
        distances. This fails if the lon/lat axis order is not forced, because
        epsg:4326 declares (lat, lon)."""
        lon_lat = np.array([[8.55, 47.37], [9.53, 46.85], [7.45, 46.95]])
        to_utm = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)
        x, y = to_utm.transform(lon_lat[:, 0], lon_lat[:, 1])
        utm = np.column_stack((x, y))

        from_lon_lat, _ = geodesic_distance_matrix(lon_lat, "epsg:4326")
        from_utm, transformed = geodesic_distance_matrix(utm, "epsg:32632")

        assert np.allclose(from_utm, from_lon_lat, rtol=1e-6)
        assert np.allclose(transformed, lon_lat, atol=1e-6)


class TestNetwork:

    def test_from_objects_euclidean(self):
        objects = make_objects(SQUARE)
        net = Network.from_objects(objects)

        assert net.n == 4
        assert net.m == net.edges.shape[0]
        assert net.edges.shape[1] == 2
        assert net.lat_lon is None
        assert net.dist_mat.shape == (4, 4)
        assert net.dist_mat[0, 1] == pytest.approx(1.0)
        assert list(net.vertices) == list(objects.id)
        assert list(net.names) == list(objects.names)

    def test_from_objects_geodesic(self):
        objects = make_objects(np.array([[8.55, 47.37], [9.53, 46.85], [7.45, 46.95]]))
        net = Network.from_objects(objects, crs="epsg:4326")

        assert net.lat_lon is not None
        # Geodesic distances are in metres, so much larger than the degree differences.
        assert net.dist_mat[0, 1] > 10_000

    @pytest.mark.parametrize("n", [2, 3])
    def test_small_networks(self, n):
        # compute_delaunay falls back to a fully connected graph; nothing should raise.
        net = Network.from_objects(make_objects(SQUARE[:n]))
        assert net.n == n
        assert net.m == n * (n - 1)

    def test_equality_is_identity(self):
        # eq=False: comparing numpy fields would raise "truth value is ambiguous".
        objects = make_objects(SQUARE)
        net = Network.from_objects(objects)
        assert net == net
        assert net != Network.from_objects(objects)
        assert hash(net) is not None