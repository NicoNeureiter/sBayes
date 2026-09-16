import datetime
import io
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from sbayes.util import (
    activate_verbose_warnings,
    cap_counts,
    cluster_agreement,
    decompose_config_path,
    default_experiment_name,
    fix_relative_path,
    format_cluster_columns,
    get_best_permutation,
    log_expit,
    normalize,
    normalize_str,
    onehot_to_integer_encoding,
    parse_cluster_columns,
    read_data_csv,
    sample_categorical,
    timeit,
    update_recursive,
    warn_with_traceback,
)


# --- Legacy .txt cluster format -------------------------------------------------------

def test_cluster_columns_round_trip():
    clusters = np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=bool)
    encoded = format_cluster_columns(clusters)
    assert encoded == "1010\t0110"
    np.testing.assert_array_equal(parse_cluster_columns(encoded), clusters)


# --- Names and paths ------------------------------------------------------------------

def test_default_experiment_name():
    name = default_experiment_name()
    # Raises if the format is wrong
    datetime.datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")


def test_decompose_config_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base, path = decompose_config_path("sub/config.yaml")
    assert path == tmp_path / "sub" / "config.yaml"
    assert base == tmp_path / "sub"


def test_fix_relative_path(tmp_path):
    assert fix_relative_path("data.csv", tmp_path) == tmp_path / "data.csv"
    assert fix_relative_path(str(tmp_path / "x.csv"), "/elsewhere") == tmp_path / "x.csv"


# --- CSV reading ----------------------------------------------------------------------

def test_normalize_str():
    assert normalize_str("  abc \t") == "abc"
    assert np.isnan(normalize_str(np.nan))


def test_read_data_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(" id ,f1\n1,  a \n2,\n3,   \n4,NA\n", encoding="utf-8")
    df = read_data_csv(path)

    assert list(df.columns) == ["id", "f1"]
    assert df["f1"].iloc[0] == "a"
    assert df["f1"].iloc[1:3].isna().all()  # empty and whitespace-only cells are missing
    assert df["f1"].iloc[3] == "NA"  # "NA" is a value, not a missing marker
    assert df["id"].tolist() == ["1", "2", "3", "4"]  # everything is read as a string


# --- Numerics -------------------------------------------------------------------------

def test_cap_counts():
    counts = np.array([[2.0, 6.0], [1.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(
        cap_counts(counts, cap_to=4.0),
        [[1.0, 3.0], [1.0, 1.0], [0.0, 0.0]],
    )
    with pytest.raises(ValueError):
        cap_counts(counts, cap_to=0.0)


def test_normalize():
    np.testing.assert_allclose(normalize(np.ones((2, 4))), np.full((2, 4), 0.25))
    np.testing.assert_allclose(normalize(np.ones((2, 4)), axis=0), np.full((2, 4), 0.5))
    assert np.isnan(normalize(np.zeros((1, 3)))).all()


def test_log_expit():
    x = jnp.array([-200.0, 0.0, 5.0], dtype=jnp.float32)
    out = log_expit(x)
    assert jnp.isfinite(out).all()
    np.testing.assert_allclose(out[0], -200.0, rtol=1e-6)
    np.testing.assert_allclose(out[1], np.log(0.5), rtol=1e-6)
    np.testing.assert_allclose(out[2], -np.log1p(np.exp(-5.0)), rtol=1e-6)


def test_onehot_to_integer_encoding():
    onehot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    np.testing.assert_array_equal(onehot_to_integer_encoding(onehot), [0, 1, 2, -1])
    np.testing.assert_array_equal(
        onehot_to_integer_encoding(onehot.T, none_index=9, axis=0), [0, 1, 2, 9]
    )


def test_sample_categorical():
    p = np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])

    # Reproducible from the global seed
    np.random.seed(0)
    s1 = sample_categorical(p)
    np.random.seed(0)
    s2 = sample_categorical(p)
    np.testing.assert_array_equal(s1, s2)

    # Zero-probability states are never drawn
    samples = sample_categorical(np.tile(p, (500, 1)))
    assert np.all(samples[0::2] == 1)
    assert set(samples[1::2]) == {0, 2}

    onehot = sample_categorical(p, binary_encoding=True)
    assert onehot.shape == (2, 3)
    assert onehot.sum(axis=-1).tolist() == [1, 1]


def test_sample_categorical_tolerates_float_drift():
    p = np.array([[0.5, 0.50027]])  # the drift seen in float32 model output
    assert sample_categorical(p).shape == (1,)


@pytest.mark.parametrize("p", [[[0.5, 0.2]], [[np.nan, 1.0]], [[-0.5, 1.5]]])
def test_sample_categorical_rejects_invalid(p):
    with pytest.raises(ValueError):
        sample_categorical(np.array(p))


# --- Cluster alignment ----------------------------------------------------------------

def test_cluster_agreement():
    a = np.array([[1, 1, 0], [0, 1, 1]])
    b = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1]])
    np.testing.assert_array_equal(cluster_agreement(a, b), [[1, 1, 2], [0, 2, 2]])


def test_get_best_permutation():
    prev = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]])
    shuffled = prev[[1, 2, 0]]
    perm = get_best_permutation(shuffled.astype(bool), prev)
    np.testing.assert_array_equal(shuffled[perm], prev)


def test_get_best_permutation_rejects_shape_mismatch():
    prev = np.ones((3, 4))
    with pytest.raises(ValueError):
        get_best_permutation(np.ones((2, 4), dtype=bool), prev)


# --- Config merging -------------------------------------------------------------------

def test_update_recursive():
    cfg = {0: 0, 1: {1: 0}, 2: {2: 1}}
    result = update_recursive(cfg, {1: {1: 1}, 2: {1: 1, 2: 2}})
    assert result == {0: 0, 1: {1: 1}, 2: {2: 2, 1: 1}}
    assert result is cfg  # updated in place

    # A dict override replaces a scalar, and vice versa
    assert update_recursive({1: 1}, {1: {1: 1}}) == {1: {1: 1}}
    assert update_recursive({1: {1: 1}}, {1: 2}) == {1: 2}


def test_update_recursive_copies_overrides():
    new = {"model": {"clusters": 3}}
    cfg = update_recursive({}, new)
    cfg["model"]["clusters"] = 5
    assert new["model"]["clusters"] == 3


# --- Diagnostics ----------------------------------------------------------------------

def test_timeit(capsys):
    @timeit("ms")
    def add(a, b):
        """Add two numbers."""
        return a + b

    assert add(1, 2) == 3
    assert "Runtime add:" in capsys.readouterr().out
    assert add.__name__ == "add"
    assert add.__doc__ == "Add two numbers."

    with pytest.raises(KeyError):
        timeit("days")


def test_warn_with_traceback():
    out = io.StringIO()
    warn_with_traceback("careful", UserWarning, "x.py", 1, file=out)
    text = out.getvalue()
    assert "UserWarning: careful" in text
    assert "test_warn_with_traceback" in text  # the caller's frame is in the stack


def test_activate_verbose_warnings(monkeypatch):
    monkeypatch.setattr(warnings, "showwarning", warnings.showwarning)
    activate_verbose_warnings()
    assert warnings.showwarning is warn_with_traceback