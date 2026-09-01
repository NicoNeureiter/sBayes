import numpy as np
import pandas as pd
import pytest

from sbayes.load_data import (Objects, CategoricalFeatures, GaussianFeatures,
                              PoissonFeatures, LogitNormalFeatures, Features,
                              Confounder, select_columns_of_type, parse_features)


def test_objects_from_dataframe_parses_and_defaults_names():
    df = pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0], "y": [3.0, 4.0]})
    objs = Objects.from_dataframe(df)
    assert objs.id == ["a", "b"]
    assert objs.locations.shape == (2, 2)
    assert objs.names == ["a", "b"]
    assert list(objs.indices) == [0, 1]

def test_objects_from_dataframe_missing_column_raises():
    df = pd.DataFrame({"id": ["a"], "x": [1.0]})  # no y column
    with pytest.raises(KeyError):
        Objects.from_dataframe(df)

def test_select_columns_of_type():
    data = pd.DataFrame({"g1": [1.0], "p1": [2.0], "g2": [3.0]})
    feature_types = {
        "g1": {"type": "gaussian"},
        "p1": {"type": "poisson"},
        "g2": {"type": "gaussian"},
    }
    indices, names = select_columns_of_type(data, feature_types, "gaussian")
    assert list(indices) == [0, 2]
    assert list(names) == ["g1", "g2"]

def test_select_columns_of_type_none_match():
    data = pd.DataFrame({"g1": [1.0]})
    feature_types = {"g1": {"type": "gaussian"}}
    indices, names = select_columns_of_type(data, feature_types, "poisson")
    assert len(indices) == 0

def test_categorical_partitions_by_nstates():
    """Categorical features split into partitions by number of states."""
    data = pd.DataFrame({
        "f1": ["a", "b", "a"],   # 2 states
        "f2": ["x", "y", "z"],   # 3 states
        "f3": ["p", "q", "p"],   # 2 states
    })
    feature_types = {
        "f1": {"type": "categorical", "states": ["a", "b"]},
        "f2": {"type": "categorical", "states": ["x", "y", "z"]},
        "f3": {"type": "categorical", "states": ["p", "q"]},
    }
    partitions = CategoricalFeatures.create_partitions_by_nstates(data, feature_types)
    assert sorted(p.n_states for p in partitions) == [2, 3]
    two = next(p for p in partitions if p.n_states == 2)
    assert set(two.names) == {"f1", "f3"}

def test_categorical_partitions_na_mapped():
    """Missing values are mapped to the NA sentinel."""
    data = pd.DataFrame({"f1": ["a", None, "b"]})
    feature_types = {"f1": {"type": "categorical", "states": ["a", "b"]}}
    p = CategoricalFeatures.create_partitions_by_nstates(data, feature_types)[0]
    assert p.na_values[1, 0]

def test_categorical_unknown_state_raises():
    """A value not among the declared states raises a clear error."""
    data = pd.DataFrame({"f1": ["a", "c"]})   # 'c' not declared
    feature_types = {"f1": {"type": "categorical", "states": ["a", "b"]}}
    with pytest.raises(ValueError, match="not declared"):
        CategoricalFeatures.create_partitions_by_nstates(data, feature_types)


def test_gaussian_features_from_dataframes():
    """Gaussian columns are parsed with NA positions from NaN values."""
    data = pd.DataFrame({"g1": [1.0, 2.0, np.nan], "g2": [4.0, 5.0, 6.0]})
    feature_types = {
        "g1": {"type": "gaussian"},
        "g2": {"type": "gaussian"},
    }
    feats = GaussianFeatures.from_dataframes(data, feature_types)
    assert feats.n_features == 2
    assert feats.na_values[2, 0]           # the NaN in g1
    assert not feats.na_values[0, 0]

def test_gaussian_features_none_when_absent():
    """Returns None when there are no Gaussian features."""
    data = pd.DataFrame({"c1": ["a", "b"]})
    feature_types = {"c1": {"type": "categorical", "states": ["a", "b"]}}
    assert GaussianFeatures.from_dataframes(data, feature_types) is None

def test_poisson_features_from_dataframes():
    """Poisson columns are parsed as counts with NA from NaN."""
    data = pd.DataFrame({"p1": [0.0, 3.0, np.nan], "p2": [1.0, 2.0, 5.0]})
    feature_types = {"p1": {"type": "poisson"}, "p2": {"type": "poisson"}}
    feats = PoissonFeatures.from_dataframes(data, feature_types)
    assert feats.n_features == 2
    assert feats.na_values[2, 0]

def test_poisson_features_none_when_absent():
    """Returns None when there are no Poisson features."""
    data = pd.DataFrame({"g1": [1.0, 2.0]})
    feature_types = {"g1": {"type": "gaussian"}}
    assert PoissonFeatures.from_dataframes(data, feature_types) is None

def test_logitnormal_features_transform_and_type():
    """Logit-normal proportions are logit-transformed and typed correctly."""
    data = pd.DataFrame({"ln1": [0.5, 0.2, 0.8]})
    feature_types = {"ln1": {"type": "logitnormal"}}
    feats = LogitNormalFeatures.from_dataframes(data, feature_types)
    assert feats is not None
    assert feats.name == "LogitNormal"
    assert isinstance(feats, GaussianFeatures)          # is-a Gaussian
    # logit(0.5) == 0
    assert np.isclose(feats.values[0, 0], 0.0)

def test_logitnormal_handles_boundary_values():
    """Values of 0 and 1 are nudged so the logit transform stays finite."""
    data = pd.DataFrame({"ln1": [0.0, 1.0]})
    feature_types = {"ln1": {"type": "logitnormal"}}
    feats = LogitNormalFeatures.from_dataframes(data, feature_types)
    assert np.all(np.isfinite(feats.values))            # no -inf/+inf

def test_logitnormal_none_when_absent():
    data = pd.DataFrame({"g1": [1.0, 2.0]})
    feature_types = {"g1": {"type": "gaussian"}}
    assert LogitNormalFeatures.from_dataframes(data, feature_types) is None

def test_features_from_dataframes_partitions_by_type():
    """Features are grouped into type-specific partitions, metadata excluded."""
    data = pd.DataFrame({
        "id": ["a", "b"],           # metadata, excluded
        "cat1": ["x", "y"],
        "gauss1": [1.0, 2.0],
    })
    feature_types = {
        "cat1": {"type": "categorical", "states": ["x", "y"]},
        "gauss1": {"type": "gaussian"},
    }
    features = Features.from_dataframes(data, feature_types)
    assert features.n_features == 2                    # id excluded
    assert "id" not in features.names
    types = {p.name for p in features.partitions}
    assert any("Categorical" in t for t in types)
    assert "Gaussian" in types

def test_features_consistency_checks():
    """n_features equals the sum across partitions."""
    data = pd.DataFrame({"c1": ["x", "y"], "g1": [1.0, 2.0]})
    feature_types = {"c1": {"type": "categorical", "states": ["x", "y"]},
                     "g1": {"type": "gaussian"}}
    features = Features.from_dataframes(data, feature_types)
    assert features.n_features == sum(p.n_features for p in features.partitions)

def test_confounder_from_dataframe_groups():
    """Objects are assigned to groups by their confounder column value."""
    data = pd.DataFrame({"family": ["A", "B", "A"]})
    conf = Confounder.from_dataframe(data, "family")
    assert conf.n_groups == 2
    assert set(conf.group_names) == {"A", "B"}
    # object 0 and 2 in group A, object 1 in group B
    a_idx = conf.group_names.index("A")
    assert list(conf.group_assignment[a_idx]) == [True, False, True]

def test_confounder_missing_column_defaults_to_all():
    """A confounder with no column applies uniformly to all objects."""
    data = pd.DataFrame({"id": ["x", "y"]})
    conf = Confounder.from_dataframe(data, "family")
    assert conf.group_names == ["<ALL>"]
    assert conf.group_assignment.shape == (1, 2)
    assert conf.group_assignment.all()

def test_confounder_na_object_in_no_group():
    """An object with a missing confounder value belongs to no group."""
    data = pd.DataFrame({"family": ["A", None, "B"]})
    conf = Confounder.from_dataframe(data, "family")
    # object 1 (None) is in no group
    assert not conf.any_group()[1]
    assert conf.any_group()[0] and conf.any_group()[2]

def test_parse_features_assembles_all():
    """parse_features builds objects, features, and confounders together."""
    data = pd.DataFrame({
        "id": ["a", "b"], "x": [1.0, 2.0], "y": [3.0, 4.0],
        "family": ["F1", "F2"],
        "f1": ["p", "q"],
    })
    feature_types = {"f1": {"type": "categorical", "states": ["p", "q"]}}
    objects, features, confounders = parse_features(
        data, feature_types, confounder_names=["family"]
    )
    assert objects.n_objects == 2
    assert features.n_features == 1
    assert "family" in confounders

def test_parse_features_unused_column_raises():
    """An unrecognised column raises a clear error."""
    data = pd.DataFrame({
        "id": ["a"], "x": [1.0], "y": [2.0],
        "mystery": ["?"],           # not id/name/x/y/confounder/feature
    })
    feature_types = {}
    with pytest.raises(ValueError, match="Unused column"):
        parse_features(data, feature_types, confounder_names=[])