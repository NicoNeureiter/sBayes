from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ModelShapes:
    n_clusters: int
    n_objects: int
    n_features: int
    n_confounders: int
    n_groups: dict[str, int]

    @property
    def n_components(self) -> int:
        """Number of mixture components per object: the cluster and each confounder."""
        return self.n_confounders + 1

    @property
    def n_components_expanded(self) -> int:
        """Number of mixture components after expanding the single cluster component
        into one component per cluster: `n_clusters + n_confounders`."""
        return self.n_clusters + self.n_confounders
