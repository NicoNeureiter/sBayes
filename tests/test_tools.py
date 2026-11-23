import importlib
import json
import numpy as np
import pandas as pd
import pytest
import tables

from pathlib import Path
from sbayes.load_data import FeatureType
from sbayes.sampling.loggers import write_stats_file
from sbayes.tools._h5 import parse_run_location
from sbayes.cluster_table import ClusterReference, align_to_reference, find_reference_files
from sbayes.results import Results, cluster_overlap
from sbayes.tools.summarize_clusters import main as summarize_main
from sbayes.tools.align_clusters import align_runs, find_runs, main as align_main
from sbayes.tools.convert_prior_csv_to_json import read_prior_counts
from sbayes.tools.find_correlated_features import find_correlated_pairs
from sbayes.tools.guess_feature_types import guess_feature_type, describe_feature, merge_feature_states
from sbayes.tools.elpd import RunFile, find_run_files, resolve_na, to_log_likelihood_tree


@pytest.mark.parametrize("name", [
    "guess_feature_types", "migrate_results", "estimate_empirical_prior",
    "find_correlated_features", "elpd", "convert_prior_csv_to_json", "align_clusters",
    "summarize_clusters",
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

# --- Aligning clusters ------------------------------------------------------------

# Two clusters over four objects: cluster A holds objects 0 and 1, cluster B objects 2
# and 3. Each has its own effect and geo-prior distance, which must move with its label.
MEMBERS = {"A": [0, 1], "B": [2, 3]}
EFFECT = {"A": [0.9, 0.1], "B": [0.2, 0.8]}
TOTAL_DIST = {"A": 1.0, "B": 2.0}
ALIGN_METADATA = {
    "cluster_names": ["a0", "a1"],
    "feature_names": ["f0"],
    "component_names": ["areal", "universal"],
    "confounders": {"universal": ["<ALL>"]},
    "partitions": [{
        "feature_names": ["f0"],
        "type": "categorical",
        "state_names": ["s0", "s1"],
        "cluster_effect_keys": ["cluster_effect_cat"],
        "confounder_effect_keys": {"universal": ["conf_effect_universal_cat"]},
    }],
}


def write_run(
    path: Path,
    labels: list[tuple[str, str]],
    object_ids: list[str] | None = None,
    object_names: list[str] | None = None,
    membership: float = 0.9,
) -> Path:
    """Write a samples file in which sample i labels the clusters `labels[i]`.

    The object IDs and names are stored in the metadata if given, as by the current
    sampler; without them, the file is like one written by earlier versions.
    """
    n_samples = len(labels)
    z = np.zeros((n_samples, 4, 3))
    effect = np.zeros((n_samples, 2, 1, 2))
    total_dist = np.zeros((n_samples, 2))
    for i, sample_labels in enumerate(labels):
        for k, cluster in enumerate(sample_labels):
            z[i, MEMBERS[cluster], k] = membership
            z[i, MEMBERS[cluster], 2] = 1 - membership
            effect[i, k, 0] = EFFECT[cluster]
            total_dist[i, k] = TOTAL_DIST[cluster]

    arrays = {
        "z": z,
        "w": np.full((n_samples, 1, 2), 0.5),
        "cluster_effect_cat": effect,
        "conf_effect_universal_cat": np.full((n_samples, 1, 1, 2), 0.5),
        "geoprior_total_dist": total_dist,
        "potential_energy": np.arange(n_samples, dtype=float),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tables.open_file(str(path), mode="w") as f:
        metadata = dict(ALIGN_METADATA)
        if object_ids is not None:
            metadata["object_ids"] = object_ids
        if object_names is not None:
            metadata["object_names"] = object_names
        f.set_node_attr(f.root, "metadata", json.dumps(metadata))
        for name, array in arrays.items():
            f.create_array(f.root, name, array)
        f.create_group(f.root, "derived")
        f.create_array(f.root.derived, "log_posterior", -np.arange(n_samples, dtype=float))
    return path


def test_stats_file_is_written_from_the_samples_file(tmp_path):
    samples_path = write_run(tmp_path / "samples_0.h5", [("A", "B")] * 3)
    write_stats_file(samples_path, tmp_path / "stats_K2_0.tsv")

    stats = pd.read_csv(tmp_path / "stats_K2_0.tsv", sep="\t")
    assert stats["Sample"].tolist() == [0, 1, 2]
    # The expected size: the sum of the membership probabilities
    assert stats["size_a0"].tolist() == pytest.approx([1.8] * 3)
    assert stats["areal_a0_f0_s0"].tolist() == pytest.approx([0.9] * 3)
    assert stats["geoprior_total_dist_a1"].tolist() == pytest.approx([2.0] * 3)
    assert stats["log_posterior"].tolist() == pytest.approx([0.0, -1.0, -2.0])
    assert not (tmp_path / "stats_K2_0.tsv.tmp").exists()


def test_stats_file_matches_clusters_within_the_run(tmp_path):
    """A label switch within the chain is undone, for every per-cluster column."""
    samples_path = write_run(tmp_path / "samples_0.h5", [("A", "B"), ("A", "B"), ("B", "A")])
    write_stats_file(samples_path, tmp_path / "stats_K2_0.tsv")

    stats = pd.read_csv(tmp_path / "stats_K2_0.tsv", sep="\t")
    assert stats["areal_a0_f0_s0"].tolist() == pytest.approx([0.9] * 3)
    assert stats["geoprior_total_dist_a0"].tolist() == pytest.approx([1.0] * 3)
    assert stats["geoprior_total_dist_a1"].tolist() == pytest.approx([2.0] * 3)


def test_align_runs_relabels_the_stats_files(tmp_path):
    write_run(tmp_path / "K2" / "samples_0.h5", [("A", "B")] * 4)
    write_run(tmp_path / "K2" / "samples_1.h5", [("B", "A")] * 4)
    runs = find_runs(tmp_path)
    assert [r.run_id for r in runs] == [0, 1]

    overlaps = align_runs(runs, reference=runs[0], burn_in=0.0)

    stats_0 = pd.read_csv(tmp_path / "K2" / "stats_K2_0.tsv", sep="\t")
    stats_1 = pd.read_csv(tmp_path / "K2" / "stats_K2_1.tsv", sep="\t")
    pd.testing.assert_frame_equal(stats_0, stats_1)
    assert overlaps[runs[1]] == pytest.approx([1.0, 1.0])


def test_align_main_uses_the_given_reference(tmp_path):
    write_run(tmp_path / "samples_0.h5", [("A", "B")] * 4)
    write_run(tmp_path / "samples_1.h5", [("B", "A")] * 4)

    align_main([tmp_path], reference=1, burn_in=0.0)

    stats_0 = pd.read_csv(tmp_path / "stats_K2_0.tsv", sep="\t")
    assert stats_0["areal_a0_f0_s0"].tolist() == pytest.approx([0.2] * 4)


def test_cluster_overlap():
    a = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    b = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    assert cluster_overlap(a, b) == pytest.approx([0.5, 1.0])


# --- Cluster reference tables -----------------------------------------------------

def test_permute_clusters_moves_the_cluster_effects(tmp_path):
    results = Results.from_h5(write_run(tmp_path / "samples_0.h5", [("A", "B")] * 4))
    permuted = results.permute_clusters(np.array([1, 0]), ["x", "y"])

    assert permuted.cluster_names == ["x", "y"]
    np.testing.assert_allclose(permuted.clusters[:, 0], results.clusters[:, 1])
    np.testing.assert_allclose(
        permuted.areal_effect["x"]["f0"], results.areal_effect["a1"]["f0"]
    )


def test_cluster_reference_round_trip(tmp_path):
    reference = ClusterReference(
        cluster_names=["a0", "a1"],
        mean_clusters=np.array([[0.9, 0.9, 0.0, 0.0], [0.0, 0.0, 0.9, 0.9]]),
        object_ids=["o0", "o1", "o2", "o3"],
        object_names=["zero", "one", "two", "three"],
    )
    reference.write(tmp_path / "cluster_reference_K2.tsv")
    read = ClusterReference.read(tmp_path / "cluster_reference_K2.tsv")

    assert read.cluster_names == reference.cluster_names
    assert read.object_ids == reference.object_ids
    assert read.object_names == reference.object_names
    np.testing.assert_allclose(read.mean_clusters, reference.mean_clusters)
    assert find_reference_files(tmp_path) == {2: tmp_path / "cluster_reference_K2.tsv"}


def test_align_to_an_edited_reference(tmp_path):
    """Swapping and renaming the columns of the table relabels the clusters."""
    results = Results.from_h5(write_run(tmp_path / "samples_0.h5", [("A", "B")] * 4))
    ClusterReference.from_results(results).write(tmp_path / "ref.tsv")

    table = pd.read_csv(tmp_path / "ref.tsv", sep="\t")
    table = table[["object", "a1", "a0"]].rename(columns={"a1": "east", "a0": "west"})
    table.to_csv(tmp_path / "ref.tsv", sep="\t", index=False)

    aligned, overlap = align_to_reference(results, ClusterReference.read(tmp_path / "ref.tsv"))
    assert aligned.cluster_names == ["east", "west"]
    np.testing.assert_allclose(aligned.clusters[:, 0], results.clusters[:, 1])
    assert aligned.areal_effect["east"]["f0"][0] == pytest.approx(EFFECT["B"])
    assert overlap == pytest.approx([1.0, 1.0])


def test_align_to_reference_matches_objects_by_id(tmp_path):
    results = Results.from_h5(write_run(tmp_path / "samples_0.h5", [("B", "A")] * 4))
    # The reference lists the objects in reverse order, so cluster A (objects 0 and 1)
    # is in its second column
    reference = ClusterReference(
        cluster_names=["a0", "a1"],
        mean_clusters=np.array([[0.9, 0.9, 0.0, 0.0], [0.0, 0.0, 0.9, 0.9]]),
        object_ids=["o3", "o2", "o1", "o0"],
    )
    aligned, _ = align_to_reference(results, reference, object_ids=["o0", "o1", "o2", "o3"])
    assert aligned.clusters[0, 0, :2] == pytest.approx([0.0, 0.0])
    assert aligned.clusters[0, 1, :2] == pytest.approx([0.9, 0.9])

    with pytest.raises(ValueError, match="not in the cluster reference"):
        align_to_reference(results, reference, object_ids=["o0", "o1", "o2", "o9"])


def test_align_to_reference_rejects_a_different_number_of_clusters(tmp_path):
    results = Results.from_h5(write_run(tmp_path / "samples_0.h5", [("A", "B")] * 4))
    reference = ClusterReference(["a0"], np.zeros((1, 4)))
    with pytest.raises(ValueError, match="1 clusters"):
        align_to_reference(results, reference)


def test_align_main_writes_the_reference(tmp_path):
    ids = ["o0", "o1", "o2", "o3"]
    write_run(tmp_path / "K2" / "samples_0.h5", [("B", "A")] * 4, object_ids=ids)
    write_run(tmp_path / "K2" / "samples_1.h5", [("A", "B")] * 4, object_ids=ids)

    align_main([tmp_path], burn_in=0.0, reference_dir=tmp_path)

    reference = ClusterReference.read(tmp_path / "cluster_reference_K2.tsv")
    assert reference.object_ids == ["o0", "o1", "o2", "o3"]
    assert reference.object_names is None
    # Run 0 is the reference, in which cluster a0 is B (objects 2 and 3)
    np.testing.assert_allclose(reference.mean_clusters, [[0, 0, 0.9, 0.9], [0.9, 0.9, 0, 0]])


# --- Cluster summaries ------------------------------------------------------------

def test_summarize_writes_one_table_per_run(tmp_path):
    write_run(tmp_path / "K2" / "samples_0.h5", [("A", "B")] * 4)
    write_run(tmp_path / "K2" / "samples_1.h5", [("B", "A")] * 4)

    paths = summarize_main([tmp_path], burn_in=0.0)

    assert [p.name for p in paths] == ["cluster_summary_K2_0.tsv", "cluster_summary_K2_1.tsv"]
    summary_0 = ClusterReference.read(paths[0])
    summary_1 = ClusterReference.read(paths[1])
    assert summary_0.cluster_names == ["a0", "a1"]
    np.testing.assert_allclose(summary_0.mean_clusters, [[0.9, 0.9, 0, 0], [0, 0, 0.9, 0.9]])
    # Without a reference, each run keeps its own labels
    np.testing.assert_allclose(summary_1.mean_clusters, summary_0.mean_clusters[::-1])


def test_summarize_aligns_to_a_reference(tmp_path):
    ids, names = ["o0", "o1", "o2", "o3"], list("pqrs")
    write_run(tmp_path / "K2" / "samples_0.h5", [("A", "B")] * 4, ids, names)
    write_run(tmp_path / "K2" / "samples_1.h5", [("B", "A")] * 4, ids, names)
    reference = ClusterReference(
        cluster_names=["east", "west"],
        mean_clusters=np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]),
        object_ids=["o0", "o1", "o2", "o3"],
    )
    reference.write(tmp_path / "cluster_reference_K2.tsv")

    paths = summarize_main([tmp_path], burn_in=0.0, reference_path=tmp_path)

    for path in paths:
        summary = ClusterReference.read(path)
        assert summary.cluster_names == ["east", "west"]
        assert summary.object_names == list("pqrs")
        np.testing.assert_allclose(summary.mean_clusters, [[0, 0, 0.9, 0.9], [0.9, 0.9, 0, 0]])


def test_a_summary_can_serve_as_reference(tmp_path):
    write_run(tmp_path / "a" / "samples_0.h5", [("A", "B")] * 4)
    write_run(tmp_path / "b" / "samples_0.h5", [("B", "A")] * 4)
    (summary_a,) = summarize_main([tmp_path / "a"], burn_in=0.0)

    (summary_b,) = summarize_main([tmp_path / "b"], burn_in=0.0, reference_path=summary_a)

    np.testing.assert_allclose(
        ClusterReference.read(summary_b).mean_clusters,
        ClusterReference.read(summary_a).mean_clusters,
    )


def test_summarize_matches_the_reference_by_object_id(tmp_path):
    """A reference with its rows in another order is matched by object ID."""
    write_run(tmp_path / "samples_0.h5", [("A", "B")] * 2, object_ids=["o0", "o1", "o2", "o3"])
    reference = ClusterReference(
        cluster_names=["east", "west"],
        # Objects o2, o3 (cluster B) first
        mean_clusters=np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]),
        object_ids=["o2", "o3", "o0", "o1"],
    )
    reference.write(tmp_path / "ref.tsv")

    (path,) = summarize_main([tmp_path], burn_in=0.0, reference_path=tmp_path / "ref.tsv")

    summary = ClusterReference.read(path)
    assert summary.object_ids == ["o0", "o1", "o2", "o3"]
    np.testing.assert_allclose(summary.mean_clusters, [[0, 0, 0.9, 0.9], [0.9, 0.9, 0, 0]])


def test_summary_of_an_older_samples_file_lists_object_indices(tmp_path):
    """Files written before the metadata held the object IDs still work."""
    write_run(tmp_path / "samples_0.h5", [("A", "B")] * 2)
    (path,) = summarize_main([tmp_path], burn_in=0.0)

    table = pd.read_csv(path, sep="\t")
    assert list(table.columns) == ["object", "a0", "a1"]



def test_summarize_combines_the_runs_of_a_directory(tmp_path):
    """Runs are aligned to the first run and averaged, weighted by their samples."""
    write_run(tmp_path / "K2" / "samples_0.h5", [("A", "B")] * 1, membership=0.9)
    write_run(tmp_path / "K2" / "samples_1.h5", [("B", "A")] * 3, membership=0.5)

    (path,) = summarize_main([tmp_path], burn_in=0.0, combine=True)

    assert path.name == "cluster_summary_K2.tsv"
    assert not list(tmp_path.rglob("cluster_summary_K2_*.tsv"))
    summary = ClusterReference.read(path)
    assert summary.cluster_names == ["a0", "a1"]
    # A is a0 in the first run: (1 * 0.9 + 3 * 0.5) / 4 = 0.6
    np.testing.assert_allclose(summary.mean_clusters, [[0.6, 0.6, 0, 0], [0, 0, 0.6, 0.6]])


def test_summarize_combines_each_directory_separately(tmp_path):
    write_run(tmp_path / "a" / "samples_0.h5", [("A", "B")] * 2)
    write_run(tmp_path / "a" / "samples_1.h5", [("B", "A")] * 2)
    write_run(tmp_path / "b" / "samples_0.h5", [("B", "A")] * 2)

    paths = summarize_main([tmp_path / "a", tmp_path / "b"], burn_in=0.0, combine=True)

    assert paths == [tmp_path / "a" / "cluster_summary_K2.tsv", tmp_path / "b" / "cluster_summary_K2.tsv"]
    # Without a reference, each directory keeps the labels of its first run
    np.testing.assert_allclose(
        ClusterReference.read(paths[1]).mean_clusters,
        ClusterReference.read(paths[0]).mean_clusters[::-1],
    )


def test_summarize_combined_runs_take_the_reference_labels(tmp_path):
    write_run(tmp_path / "samples_0.h5", [("A", "B")] * 2)
    write_run(tmp_path / "samples_1.h5", [("B", "A")] * 2)
    reference = ClusterReference(
        cluster_names=["east", "west"],
        mean_clusters=np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]),
    )
    reference.write(tmp_path / "ref.tsv")

    (path,) = summarize_main(
        [tmp_path], burn_in=0.0, combine=True, reference_path=tmp_path / "ref.tsv"
    )

    summary = ClusterReference.read(path)
    assert summary.cluster_names == ["east", "west"]
    np.testing.assert_allclose(summary.mean_clusters, [[0, 0, 0.9, 0.9], [0.9, 0.9, 0, 0]])


def test_summarize_does_not_combine_runs_of_other_objects(tmp_path):
    write_run(tmp_path / "samples_0.h5", [("A", "B")] * 2, object_ids=["o0", "o1", "o2", "o3"])
    write_run(tmp_path / "samples_1.h5", [("A", "B")] * 2, object_ids=["x0", "x1", "x2", "x3"])
    with pytest.raises(ValueError, match="same clusters and objects"):
        summarize_main([tmp_path], burn_in=0.0, combine=True)
