import importlib
import numpy as np
import pandas as pd
import pytest

from pathlib import Path
from sbayes.load_data import FeatureType
from sbayes.tools._h5 import parse_run_location
from sbayes.tools.convert_prior_csv_to_json import read_prior_counts
from sbayes.tools.find_correlated_features import find_correlated_pairs
from sbayes.tools.guess_feature_types import guess_feature_type, describe_feature, merge_feature_states
from sbayes.tools.elpd import RunFile, find_run_files, resolve_na, to_log_likelihood_tree


@pytest.mark.parametrize("name", [
    "guess_feature_types", "migrate_results", "estimate_empirical_prior",
    "find_correlated_features", "elpd", "convert_prior_csv_to_json",
])
def test_tool_imports(name):
    importlib.import_module(f"sbayes.tools.{name}")

@pytest.mark.parametrize(
    "states, expected",
    [
        # Non-numeric states
        ({"red", "blue"}, FeatureType.categorical),
        ({"1", "a"}, FeatureType.categorical),
        ({"nan", "1"}, FeatureType.categorical),  # not finite
        ({"inf", "1"}, FeatureType.categorical),
        # Binary, regardless of formatting
        ({"0", "1"}, FeatureType.categorical),
        ({"0.0", "1.0"}, FeatureType.categorical),
        ({"1"}, FeatureType.categorical),
        # Non-negative integers
        ({"0", "7"}, FeatureType.poisson),
        ({"3.0", "7"}, FeatureType.poisson),
        # Proportions, bounds included
        ({"0", "0.5", "1"}, FeatureType.logitnormal),
        ({"0.0", "0.5"}, FeatureType.logitnormal),
        # Anything else numeric
        ({"-2", "5"}, FeatureType.gaussian),  # negative integers are not counts
        ({"1.5", "2"}, FeatureType.gaussian),
        ({"-0.5", "0.5"}, FeatureType.gaussian),  # outside [0, 1]
    ],
)
def test_guess_feature_type(states, expected):
    assert guess_feature_type(states) is expected


def test_guess_feature_type_without_states():
    with pytest.raises(ValueError):
        guess_feature_type(set())

def test_describe_feature():
    assert describe_feature({"b", "a"}) == {"type": "categorical", "states": ["a", "b"]}
    assert describe_feature({"3.0", "7"}) == {"type": "poisson", "states": {"min": 3, "max": 7}}
    assert describe_feature({"-1.5", "2"}) == {"type": "gaussian", "states": {"min": -1.5, "max": 2.0}}
    assert all(isinstance(v, int) for v in describe_feature({"0", "7"})["states"].values())


def test_merge_feature_states():
    merged = merge_feature_states({
        "a.csv": {"f1": {"x"}, "f2": {"1"}},
        "b.csv": {"f1": {"y"}, "f2": {"1", "2"}},
    })
    assert merged == {"f1": {"x", "y"}, "f2": {"1", "2"}}


def test_merge_feature_states_rejects_mismatch():
    with pytest.raises(ValueError, match="f2"):
        merge_feature_states({"a.csv": {"f1": set(), "f2": set()}, "b.csv": {"f1": set()}})

def write_csv(tmp_path, content: str):
    path = tmp_path / "counts.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_read_prior_counts(tmp_path):
    path = write_csv(tmp_path, "feature,a,b,c\nf1,3,1,\nf2,,2.5,4\n")
    counts = read_prior_counts(path)

    # Empty cells mean the state does not occur for that feature
    assert counts == {"f1": {"a": 3, "b": 1}, "f2": {"b": 2.5, "c": 4}}
    # Whole numbers stay integers, so the JSON reads as counts
    assert isinstance(counts["f1"]["a"], int)
    assert isinstance(counts["f2"]["b"], float)


def test_read_prior_counts_normalizes_names(tmp_path):
    path = write_csv(tmp_path, " feature , a \n  f1  ,2\n")
    assert read_prior_counts(path) == {"f1": {"a": 2}}


def test_read_prior_counts_without_feature_column(tmp_path):
    path = write_csv(tmp_path, "name,a\nf1,2\n")
    with pytest.raises(ValueError, match="feature"):
        read_prior_counts(path)


def test_read_prior_counts_with_duplicate_features(tmp_path):
    path = write_csv(tmp_path, "feature,a\nf1,2\nf1,3\n")
    with pytest.raises(ValueError, match="f1"):
        read_prior_counts(path)


def test_read_prior_counts_with_non_numeric_counts(tmp_path):
    path = write_csv(tmp_path, "feature,a\nf1,many\n")
    with pytest.raises(ValueError, match="Non-numeric"):
        read_prior_counts(path)

@pytest.fixture
def features() -> pd.DataFrame:
    """Three features over 40 objects: f1 and f2 are identical, f3 is independent."""
    rng = np.random.default_rng(0)
    f1 = np.array(["x", "y"] * 20)
    return pd.DataFrame({
        "f1": f1,
        "f2": np.where(f1 == "x", "p", "q"),  # perfectly dependent on f1
        "f3": rng.permutation(np.array(["m", "n"] * 20)),
        "f4": ["only_one_state"] * 40,  # cannot be tested
    })


def test_find_correlated_pairs(features):
    pairs, threshold = find_correlated_pairs(features, p_threshold=0.01)

    assert [(p.feature_1, p.feature_2) for p in pairs] == [("f1", "f2")]
    # 6 pairs are formed, of which f4's are skipped for having one state only
    assert threshold == pytest.approx(0.01 / 6)

    pair = pairs[0]
    assert pair.observed.to_numpy().tolist() == [[20, 0], [0, 20]]
    assert pair.expected.to_numpy().tolist() == [[10, 10], [10, 10]]
    assert list(pair.observed.index) == ["x", "y"]
    assert list(pair.observed.columns) == ["p", "q"]


def test_resolve_na_uses_the_stored_mask():
    likelihood = np.ones((5, 3))
    stored = np.array([True, False, False])

    # The stored mask wins, even where the likelihood would suggest otherwise
    np.testing.assert_array_equal(resolve_na(likelihood, stored), stored)


def test_resolve_na_rejects_a_mismatched_mask():
    with pytest.raises(ValueError, match="shape"):
        resolve_na(np.ones((5, 3)), np.array([True, False]))


def test_resolve_na_without_a_mask():
    # Without the flag, a missing mask is an error rather than a guess
    with pytest.raises(ValueError, match="assumeNaFromLikelihood"):
        resolve_na(np.ones((5, 3)), None)


def test_resolve_na_guesses_from_the_likelihood():
    likelihood = np.array([[1.0, 0.4, 1.0], [1.0, 0.6, 0.9]])

    with pytest.warns(UserWarning, match="1 observation"):
        is_na = resolve_na(likelihood, None, assume_from_likelihood=True)

    # Only the column that is 1.0 in every sample counts as missing
    np.testing.assert_array_equal(is_na, [True, False, False])


# --- to_log_likelihood_tree ---------------------------------------------------------------

def test_to_log_likelihood_tree():
    likelihood = np.tile(np.array([0.5, 0.25, 0.1]), (10, 1))
    is_na = np.array([False, False, True])

    data = to_log_likelihood_tree(likelihood, is_na, burnin=0.2)
    log_like = data.log_likelihood["y"]

    # NAs dropped, burn-in removed, and a chain axis of length 1 added
    assert log_like.shape == (1, 8, 2)
    np.testing.assert_allclose(log_like.to_numpy()[0, 0], np.log([0.5, 0.25]))


@pytest.mark.parametrize("burnin", [-0.1, 1.0, 1.5])
def test_to_log_likelihood_tree_rejects_an_invalid_burn_in(burnin):
    with pytest.raises(ValueError, match="burn-in"):
        to_log_likelihood_tree(np.ones((4, 2)), np.zeros(2, bool), burnin)


def test_to_log_likelihood_tree_without_observations():
    with pytest.raises(ValueError, match="Nothing left"):
        to_log_likelihood_tree(np.ones((4, 2)), np.ones(2, bool), burnin=0.0)


def test_to_log_likelihood_tree_warns_about_zeros():
    likelihood = np.array([[0.5, 0.0], [0.5, 0.5]])

    with pytest.warns(UserWarning, match="zero"):
        data = to_log_likelihood_tree(likelihood, np.zeros(2, bool), burnin=0.0)

    # Clipped rather than -inf, so arviz still gets finite values
    assert np.isfinite(data.log_likelihood["y"].to_numpy()).all()


# --- find_run_files ------------------------------------------------------------------

def make_results_tree(root):
    """Build a results tree with a migrated run, a legacy-only run and noise."""
    for relative in [
        "exp_a/K2/samples_0.h5",
        "exp_a/K2/likelihood_K2_0.h5",  # same run, legacy file left over
        "exp_a/K3/samples_0.h5",
        "exp_a/K3/samples_1.h5",
        "exp_b/K2/likelihood_K2_0.h5",  # not migrated
        "exp_b/K2/samples_0.chain1.h5",  # heated MC3 chain
        "exp_b/notK/samples_0.h5",  # not a K folder
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


def test_find_run_files(tmp_path):
    root = make_results_tree(tmp_path)

    with pytest.warns(UserWarning, match="notK"):
        run_files = find_run_files(root)

    assert [(r.experiment, r.k, r.run) for r in run_files] == [
        ("exp_a", 2, 0), ("exp_a", 3, 0), ("exp_a", 3, 1), ("exp_b", 2, 0),
    ]


def test_find_run_files_prefers_the_samples_file(tmp_path):
    root = make_results_tree(tmp_path)

    with pytest.warns(UserWarning):
        by_run = {(r.experiment, r.k, r.run): r for r in find_run_files(root)}

    # A migrated run is counted once, through its samples file
    assert by_run[("exp_a", 2, 0)].path.name == "samples_0.h5"
    # A run without a samples file is still found
    assert by_run[("exp_b", 2, 0)].path.name == "likelihood_K2_0.h5"


def test_find_run_files_in_an_empty_directory(tmp_path):
    assert find_run_files(tmp_path) == []


def test_run_file_is_a_named_tuple(tmp_path):
    run_file = RunFile(tmp_path / "samples_0.h5", "exp", 2, 0)
    assert (run_file.experiment, run_file.k, run_file.run) == ("exp", 2, 0)


@pytest.mark.parametrize(
    "path, expected",
    [
        # Current format
        ("results/exp/K3/samples_0.h5", (3, 0)),
        ("results/exp/K12/samples_7.h5", (12, 7)),
        # Some versions repeat the cluster count in the file name
        ("results/exp/K3/samples_K3_2.h5", (3, 2)),
        # Legacy likelihood files, read by the elpd tool
        ("results/exp/K3/likelihood_K3_1.h5", (3, 1)),
        ("results/exp/K3/likelihood_1.h5", (3, 1)),
        # Only the last two path components matter
        ("/absolute/anywhere/K2/samples_0.h5", (2, 0)),
    ],
)
def test_parse_run_location(path, expected):
    assert parse_run_location(Path(path)) == expected


@pytest.mark.parametrize(
    "path",
    [
        "results/exp/samples_0.h5",  # no K folder
        "results/exp/K/samples_0.h5",  # no cluster count
        "results/exp/Kthree/samples_0.h5",  # not a number
        "results/exp/K3/samples.h5",  # no run index
        "results/exp/K3/stats_K3_0.txt",  # not a samples file
        "results/exp/K3/samples_0_backup.h5",  # trailing text
    ],
)
def test_parse_run_location_rejects_other_paths(path):
    with pytest.raises(ValueError, match="K<k>"):
        parse_run_location(Path(path))