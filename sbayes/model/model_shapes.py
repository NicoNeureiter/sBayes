from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class ModelShapes:
    n_groups: dict[str, int]
    n_confounders: int
    n_features: int
    n_clusters: int
    n_objects: int
    n_features: int
    n_features_categorical: int
    n_states_categorical: int
    states_per_feature: NDArray[bool]
    n_features_gaussian: int
    n_features_poisson: int
    n_features_logitnormal: int
    n_states_per_feature: list[int] = None
    confounder_index: dict[str, int] = None

    def __post_init__(self):
        self.n_states_per_feature = [sum(applicable) for applicable in self.states_per_feature]
        self.confounder_index = {conf: i for i, conf in enumerate(self.n_groups.keys())}

    @property
    def n_components(self):
        return self.n_confounders + 1

    def get_component_index(self, component: str):
        if component == 'cluster':
            return 0
        else:
            return self.confounder_index[component] + 1

    def __getitem__(self, key):
        """Getter for backwards compatibility with dict-notation."""
        return getattr(self, key)
