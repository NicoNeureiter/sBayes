from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class ModelShapes:
    n_clusters: int
    n_objects: int
    n_features: int
    n_confounders: int
    n_groups: dict[str, int]

    @property
    def n_components(self):
        return self.n_confounders + 1

    def __getitem__(self, key):
        """Getter for backwards compatibility with dict-notation."""
        return getattr(self, key)
