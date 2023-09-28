from __future__ import annotations

from dataclasses import dataclass

from numpy._typing import NDArray


@dataclass(frozen=True)
class ModelShapes:
    n_clusters: int
    n_sites: int
    n_features: int
    n_states: int
    states_per_feature: NDArray[bool]
    n_confounders: int
    n_groups: dict[str, int]

    @property
    def n_states_per_feature(self):
        return [sum(applicable) for applicable in self.states_per_feature]

    @property
    def n_components(self):
        return self.n_confounders + 1

    def __getitem__(self, key):
        """Getter for backwards compatibility with dict-notation."""
        return getattr(self, key)
