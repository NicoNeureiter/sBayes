from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Sequence, TypeVar, Callable

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import tables

from sbayes.util import PathLike

TResults = TypeVar("TResults", bound="Results")


class Results:

    """Class for reading, storing, summarizing results of a sBayes analysis.

    Attributes:
        clusters (NDArray[float]): Cluster assignment samples.
            shape: (n_clusters, n_samples, n_objects)
        weights (dict[str, NDArray]): Weights per feature.
            Each value has shape (n_samples, n_components).
        areal_effect (dict[str, dict[str, NDArray]]): Cluster effects.
            Nested as {cluster_name: {feature_name: array(n_samples, n_states)}}.
        confounding_effects (dict[str, dict[str, dict[str, NDArray]]]): Confounder effects.
            Nested as {confounder: {group: {feature: array(n_samples, n_states)}}}.
        groups_by_confounders (dict[str, list[str]]): Group names for each confounder.
    """

    def __init__(
        self,
        clusters: NDArray[float],
        weights: dict[str, NDArray],
        areal_effect: dict[str, dict[str, NDArray]],
        confounding_effects: dict[str, dict[str, dict[str, NDArray]]],
        cluster_names: list[str],
        feature_names: list[str],
        feature_states: list[list[str]],
        groups_by_confounders: dict[str, list[str]],
        sample_id: NDArray[int] = None,
        log_posterior: NDArray[float] = None,
        log_likelihood: NDArray[float] = None,
        likelihood_pointwise: NDArray[float] = None,
        parameters: pd.DataFrame = None,
    ):
        self.clusters = clusters
        self.cluster_names = cluster_names
        self.feature_names = feature_names
        self.feature_states = feature_states
        self.groups_by_confounders = groups_by_confounders

        self.sample_id = sample_id if sample_id is not None else np.arange(self.n_samples)

        self.weights = weights
        self.areal_effect = areal_effect
        self.confounding_effects = confounding_effects

        # Log-probabilities (named without log_ prefix for backward compatibility)
        self.posterior = log_posterior
        self.likelihood = log_likelihood
        if log_posterior is not None and log_likelihood is not None:
            self.prior = log_posterior - log_likelihood
        else:
            self.prior = None

        # Per-observation log-likelihoods; shape (n_samples, n_obs), NAs excluded.
        # Only available when loaded from the new h5 format that includes derived/likelihood.
        self.likelihood_pointwise = likelihood_pointwise

        # Legacy: raw parameters DataFrame (set by from_csv_files for align tools)
        self.parameters = parameters

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_clusters(self) -> int:
        return self.clusters.shape[0]

    @property
    def n_samples(self) -> int:
        return self.clusters.shape[1]

    @property
    def n_objects(self) -> int:
        return self.clusters.shape[2]

    @property
    def confounders(self) -> list[str]:
        return list(self.groups_by_confounders.keys())

    @property
    def n_confounders(self) -> int:
        return len(self.groups_by_confounders)

    def get_states_for_feature_name(self, f: str) -> list[str]:
        return self.feature_states[self.feature_names.index(f)]

    # ----------------------------------------------------------------
    # Combining multiple Results objects
    # ----------------------------------------------------------------

    @classmethod
    def concatenate(
        cls: type[TResults],
        results_list: list[TResults],
        align_clusters: bool = False,
    ) -> TResults:
        """Concatenate multiple Results objects along the samples axis.

        Args:
            results_list: List of Results objects to combine.
            align_clusters: If True, align cluster labels of results[1:]
                to match results[0] before concatenating (using mean cluster
                assignments and the Hungarian algorithm).

        Returns:
            A single Results object with concatenated samples.
        """
        # Validate compatibility
        if not results_list:
            raise ValueError("Cannot concatenate an empty list of Results.")
        ref = results_list[0]
        for i, r in enumerate(results_list[1:], 1):
            assert r.n_clusters == ref.n_clusters
            assert r.n_objects == ref.n_objects

        # Catch simple base case
        if len(results_list) == 1:
            return results_list[0]

        if align_clusters:
            results_list = cls.align_results_list(results_list)

        # Concatenate clusters: (n_clusters, n_samples, n_objects)
        clusters = np.concatenate([r.clusters for r in results_list], axis=1)


        # Concatenate weights, areal_effects and confounding_effects
        concat = lambda xs: np.concatenate(xs, axis=0)
        weights = concat_dicts_recursive([r.weights for r in results_list], concat)
        areal_effect = concat_dicts_recursive([r.areal_effect for r in results_list], concat)
        confounding_effects = concat_dicts_recursive([r.confounding_effects for r in results_list], concat)

        log_posterior = None
        if all(r.posterior is not None for r in results_list):
            log_posterior = np.concatenate([r.posterior for r in results_list])

        log_likelihood = None
        if all(r.likelihood is not None for r in results_list):
            log_likelihood = np.concatenate([r.likelihood for r in results_list])

        likelihood_pointwise = None
        if all(r.likelihood_pointwise is not None for r in results_list):
            likelihood_pointwise = np.concatenate([r.likelihood_pointwise for r in results_list])

        parameters = None
        if all(r.parameters is not None for r in results_list):
            parameters = pd.concat([r.parameters for r in results_list], ignore_index=True)

        return cls(
            clusters=clusters,
            weights=weights,
            areal_effect=areal_effect,
            confounding_effects=confounding_effects,
            cluster_names=ref.cluster_names,
            feature_names=ref.feature_names,
            feature_states=ref.feature_states,
            groups_by_confounders=ref.groups_by_confounders,
            sample_id=np.arange(clusters.shape[1]),
            log_posterior=log_posterior,
            log_likelihood=log_likelihood,
            likelihood_pointwise=likelihood_pointwise,
            parameters=parameters,
        )

    @classmethod
    def align_results_list(
        cls: type[TResults],
        results_list: list[TResults],
    ) -> list[TResults]:
        """Align cluster labels of results[1:] to match results[0].

        Uses mean cluster assignments and the Hungarian algorithm to find the
        best permutation for each subsequent Results object.
        """
        from scipy.optimize import linear_sum_assignment

        ref = results_list[0]
        ref_mean = np.mean(ref.clusters, axis=1)  # (n_clusters, n_objects)

        aligned = [ref]
        for r in results_list[1:]:
            r_mean = np.mean(r.clusters, axis=1)
            agreement = ref_mean @ r_mean.T  # (n_clusters, n_clusters)
            perm = linear_sum_assignment(agreement, maximize=True)[1]

            if np.all(perm == np.arange(len(perm))):
                aligned.append(r)
                continue

            # Permute clusters along axis 0 (n_clusters)
            clusters_aligned = r.clusters[perm]

            # Remap areal_effect: ref cluster i ← r cluster perm[i]
            areal_effect_aligned = {}
            for i, ref_name in enumerate(ref.cluster_names):
                r_name = r.cluster_names[perm[i]]
                areal_effect_aligned[ref_name] = r.areal_effect[r_name]

            aligned.append(cls(
                clusters=clusters_aligned,
                weights=r.weights,
                areal_effect=areal_effect_aligned,
                confounding_effects=r.confounding_effects,
                cluster_names=ref.cluster_names,
                feature_names=r.feature_names,
                feature_states=r.feature_states,
                groups_by_confounders=r.groups_by_confounders,
                sample_id=r.sample_id,
                log_posterior=r.posterior,
                log_likelihood=r.likelihood,
                likelihood_pointwise=r.likelihood_pointwise,
            ))

        return aligned

    # ----------------------------------------------------------------
    # Constructor: from consolidated h5 file (new format)
    # ----------------------------------------------------------------

    @classmethod
    def from_h5(
        cls: type[TResults],
        h5_path: PathLike,
        burn_in: float = 0.1,
        subsample_interval: int = 1,
        do_match_clusters: bool = True,
    ) -> TResults:
        """Load results from a consolidated samples h5 file.

        The h5 file contains all MCMC parameter samples and a JSON metadata
        attribute that describes feature names, partitions, confounders, etc.
        This makes the Results object fully independent of the stats TSV file.
        """
        with tables.open_file(str(h5_path), mode="r") as f:
            if not hasattr(f.root._v_attrs, "metadata"):
                raise ValueError(
                    f"No metadata in {h5_path}. Run "
                    f"'python -m sbayes.tools.migrate_results <results_dir>' first."
                )
            metadata = json.loads(f.root._v_attrs.metadata)

            cluster_names = metadata["cluster_names"]
            feature_names = metadata["feature_names"]
            confounders = metadata["confounders"]
            partitions = metadata["partitions"]

            # Read parameter arrays
            def _read(key):
                return np.array(f.root._v_children[key])

            z = _read('z')
            w = _read('w')

            # Old-format h5 files have a chain dim at axis 1: (n_samples, 1, ...)
            old_format = z.ndim == 4
            if old_format:
                z = z[:, 0]
                w = w[:, 0]

            cluster_effect_arrays = {}
            conf_effect_arrays = {}
            for p in partitions:
                for key in p["cluster_effect_keys"]:
                    arr = _read(key)
                    if old_format:
                        arr = arr[:, 0]
                    cluster_effect_arrays[key] = arr
                for conf_name, keys in p["confounder_effect_keys"].items():
                    for key in keys:
                        arr = _read(key)
                        if old_format:
                            arr = arr[:, 0]
                        conf_effect_arrays[key] = arr

            log_posterior = None
            if 'potential_energy' in f.root._v_children:
                arr = _read('potential_energy')
                if old_format and arr.ndim == 2:
                    arr = arr[:, 0]
                log_posterior = arr

            log_likelihood = None
            likelihood_pointwise = None
            if 'derived' in f.root._v_children:
                derived = f.root._v_children['derived']
                if 'likelihood' in derived._v_children:
                    lh = np.array(derived._v_children['likelihood'])
                    if 'na_values' in derived._v_children:
                        na = np.array(derived._v_children['na_values'])
                        lh_valid = lh[:, ~na]
                    else:
                        lh_valid = lh
                    likelihood_pointwise = lh_valid
                    log_likelihood = lh_valid.sum(axis=1)

        # Apply burn-in and subsampling
        n_total = z.shape[0]
        burn_in_idx = int(burn_in * n_total)
        indices = np.arange(burn_in_idx, n_total, max(1, subsample_interval))

        all_arrays = (
            [z, w]
            + list(cluster_effect_arrays.values())
            + list(conf_effect_arrays.values())
        )
        z, w, *rest = [a[indices] for a in all_arrays]
        n_cluster_eff = len(cluster_effect_arrays)
        for key, arr in zip(cluster_effect_arrays, rest[:n_cluster_eff]):
            cluster_effect_arrays[key] = arr
        for key, arr in zip(conf_effect_arrays, rest[n_cluster_eff:]):
            conf_effect_arrays[key] = arr

        if log_posterior is not None:
            log_posterior = log_posterior[indices]
        if log_likelihood is not None:
            log_likelihood = log_likelihood[indices]
        if likelihood_pointwise is not None:
            likelihood_pointwise = likelihood_pointwise[indices]

        # Cluster matching
        if do_match_clusters:
            match_clusters(z, cluster_effect_arrays)

        # Build clusters: (n_clusters, n_samples, n_sites)
        clusters = np.transpose(z[..., :-1], (2, 0, 1)).astype(float, copy=False)

        # Build feature_states mapping
        feature_to_states = {}
        for p in partitions:
            for fname in p["feature_names"]:
                feature_to_states[fname] = p["state_names"]
        feature_states = [feature_to_states[f] for f in feature_names]

        # Build weights: {feature -> (n_samples, n_components)}
        weights = {f: w[:, i, :] for i, f in enumerate(feature_names)}

        # Build areal_effect and confounding_effects
        areal_effect = _build_effect_dict(
            cluster_names, partitions, cluster_effect_arrays,
            get_keys=lambda p: p["cluster_effect_keys"],
        )
        confounding_effects = {
            conf_name: _build_effect_dict(
                group_names, partitions, conf_effect_arrays,
                get_keys=lambda p, cn=conf_name: p["confounder_effect_keys"][cn],
            )
            for conf_name, group_names in confounders.items()
        }

        return cls(
            clusters=clusters,
            weights=weights,
            areal_effect=areal_effect,
            confounding_effects=confounding_effects,
            cluster_names=cluster_names,
            feature_names=feature_names,
            feature_states=feature_states,
            groups_by_confounders=confounders,
            sample_id=np.arange(len(indices)),
            log_posterior=log_posterior,
            log_likelihood=log_likelihood,
            likelihood_pointwise=likelihood_pointwise,
        )

    # ----------------------------------------------------------------
    # Constructor: from CSV files (legacy format)
    # ----------------------------------------------------------------

    @classmethod
    def from_csv_files(
        cls: type[TResults],
        clusters_path: PathLike,
        parameters_path: PathLike,
        burn_in: float = 0.1,
        subsample_interval: int = 1,
        feature_names: list[str] = None,
        confounder_names: dict[str, list[str]] = None,
    ) -> TResults:
        """Load results from legacy cluster and stats text/CSV files."""
        clusters = cls.read_clusters(clusters_path, subsample_interval=subsample_interval)
        parameters = cls.read_stats(parameters_path, subsample_interval=subsample_interval)

        # Apply burn-in
        n_total = clusters.shape[1]
        burn_in_idx = int(burn_in * n_total)
        clusters = clusters[:, burn_in_idx:, :]
        parameters = parameters.iloc[burn_in_idx:]

        # Extract names from column headers
        cluster_names = cls.get_cluster_names(parameters.columns)

        if feature_names is None:
            feature_names = _extract_feature_names(parameters)

        if confounder_names is not None:
            groups_by_confounders = confounder_names
        else:
            groups_by_confounders = cls.get_groups_by_confounder(parameters.columns)

        feature_states = [
            _extract_state_names(parameters, prefix=f"areal_{cluster_names[0]}_{f}_")
            for f in feature_names
        ]

        # Parse weights
        components = ["areal"] + list(groups_by_confounders.keys())
        weights = {
            f: np.column_stack(
                [parameters[f"w_{c}_{f}"].to_numpy(dtype=float) for c in components]
            )
            for f in feature_names
        }

        # Parse areal effect
        areal_effect = {
            cluster: {
                f: np.column_stack(
                    [parameters[f"areal_{cluster}_{f}_{s}"].to_numpy(dtype=float)
                     for s in feature_states[i_f]]
                )
                for i_f, f in enumerate(feature_names)
            }
            for cluster in cluster_names
        }

        # Parse confounding effects
        confounding_effects = {
            conf: {
                g: {
                    f: np.column_stack(
                        [parameters[f"{conf}_{g}_{f}_{s}"].to_numpy(dtype=float)
                         for s in feature_states[i_f]]
                    )
                    for i_f, f in enumerate(feature_names)
                }
                for g in groups
            }
            for conf, groups in groups_by_confounders.items()
        }

        # Parse log posterior, likelihood
        log_posterior = None
        for col_name in ["log_posterior", "posterior"]:
            if col_name in parameters.columns:
                log_posterior = parameters[col_name].to_numpy(dtype=float)
                break
        log_likelihood = None
        for col_name in ["log_likelihood", "likelihood"]:
            if col_name in parameters.columns:
                log_likelihood = parameters[col_name].to_numpy(dtype=float)
                break

        sample_id = (
            parameters["Sample"].to_numpy(dtype=int)
            if "Sample" in parameters.columns
            else None
        )

        return cls(
            clusters=clusters,
            weights=weights,
            areal_effect=areal_effect,
            confounding_effects=confounding_effects,
            cluster_names=cluster_names,
            feature_names=feature_names,
            feature_states=feature_states,
            groups_by_confounders=groups_by_confounders,
            sample_id=sample_id,
            log_posterior=log_posterior,
            log_likelihood=log_likelihood,
            parameters=parameters,
        )

    # ----------------------------------------------------------------
    # Static utility methods (used by from_csv_files and migrate_results)
    # ----------------------------------------------------------------

    @staticmethod
    def read_clusters(txt_path: PathLike, subsample_interval: int = 1) -> NDArray[float]:
        """Read cluster samples from text or .npy file (legacy format)."""
        path = Path(txt_path)

        # Use .npy if it exists
        if path.suffix == ".txt":
            if path.with_suffix(".npy").exists():
                path = path.with_suffix(".npy")

        if path.suffix == ".npy":
            clusters = np.load(path)
            if clusters.ndim != 3:
                raise ValueError(
                    f"Expected a 3D array in {path}, got shape {clusters.shape}."
                )
            if subsample_interval > 1:
                clusters = clusters[::subsample_interval, :, :]
            if clusters.shape[2] < 2:
                raise ValueError(
                    f"Expected at least 2 components (including not-assigned) in {path}, "
                    f"got shape {clusters.shape}."
                )
            clusters = clusters[..., :-1]
            clusters = np.transpose(clusters, (2, 0, 1)).astype(float, copy=False)
        else:
            with open(txt_path, "r") as f_sample:
                samples_list = [
                    [list(c) for c in line.split('\t')]
                    for line in f_sample.read().split("\n")[::subsample_interval]
                    if line.strip()
                ]
                clusters = np.array(samples_list, dtype=int).astype(bool).transpose((1, 0, 2))

        return clusters

    @staticmethod
    def read_stats(stats_path: PathLike, subsample_interval: int = 1, use_pyarrow=True) -> pd.DataFrame:
        """Read stats from TSV or legacy TXT file."""
        path = Path(stats_path)
        # Fall back to .txt if .tsv doesn't exist (legacy support)
        if path.suffix == ".tsv" and not path.exists() and path.with_suffix(".txt").exists():
            path = path.with_suffix(".txt")

        read_args = {}
        if subsample_interval > 1:
            read_args["skiprows"] = lambda i: i % subsample_interval != 0
            use_pyarrow = False

        if use_pyarrow:
            try:
                return pd.read_csv(path, delimiter="\t", engine="pyarrow", **read_args)
            except Exception as e:
                warnings.warn(str(e))
                return Results.read_stats(path, subsample_interval, use_pyarrow=False)
        else:
            return pd.read_csv(path, delimiter="\t", engine="python", **read_args)

    @staticmethod
    def get_cluster_names(column_names) -> list[str]:
        area_names = []
        for key in column_names:
            if not key.startswith("areal_"):
                continue
            _, area, _ = key.split("_", maxsplit=2)
            if area not in area_names:
                area_names.append(area)
        return area_names

    @staticmethod
    def get_groups_by_confounder(
        column_names: Sequence[str],
    ) -> dict[str, list[str]]:
        """Extract confounder names and group names from parameter column names."""
        groups_by_confounder = {}

        for key in column_names:
            if not key.startswith("w_"):
                continue
            if key.startswith("w_concentration_"):
                continue
            _, conf, _ = key.split("_", maxsplit=2)
            if conf in ["areal", "cluster"]:
                continue
            if conf in groups_by_confounder:
                continue
            groups_by_confounder[conf] = []

        for conf in groups_by_confounder:
            for key in column_names:
                if not key.startswith(f"{conf}_"):
                    continue
                _, group, _ = key.split("_", maxsplit=2)
                if group in groups_by_confounder[conf]:
                    continue
                groups_by_confounder[conf].append(group)

        return groups_by_confounder


# ----------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------

def match_clusters(z, cluster_effect_arrays):
    """Apply cluster matching to z and cluster effect arrays in-place."""
    from sbayes.preprocessing import sample_categorical
    from sbayes.util import get_best_permutation

    clusters_binary = sample_categorical(z, binary_encoding=True)[:, :, :-1]
    # (n_samples, n_sites, n_clusters) -> (n_samples, n_clusters, n_sites)
    clusters_binary = clusters_binary.transpose(0, 2, 1)
    n_samples = clusters_binary.shape[0]
    clusters_sum = np.zeros(clusters_binary.shape[1:], dtype=int)

    for i in range(n_samples):
        perm = get_best_permutation(clusters_binary[i], clusters_sum)
        if not np.all(perm == np.arange(len(perm))):
            clusters_binary[i] = clusters_binary[i][perm]
            z[i, :, :-1] = z[i, :, :-1][:, perm]
            for arr in cluster_effect_arrays.values():
                arr[i] = arr[i][perm]
        clusters_sum += clusters_binary[i]


def _build_effect_dict(
    entity_names: list[str],
    partitions: list[dict],
    arrays: dict[str, NDArray],
    get_keys: Callable[[dict], list[str]],
) -> dict[str, dict[str, NDArray]]:
    """Build {entity_name: {feature_name: array}} from h5 arrays.

    Works for both cluster effects (entity=cluster) and confounding effects
    (entity=group). The `get_keys` callable extracts the relevant h5 keys
    from each partition dict.

    For categorical partitions: single key, array has states as last dim.
    For gaussian/poisson: multiple keys (one per state), no states dim.
    """
    effect = {name: {} for name in entity_names}
    for p in partitions:
        keys = get_keys(p)
        p_features = p["feature_names"]

        if p["type"] == "categorical":
            arr = arrays[keys[0]]  # (n_samples, n_entities, n_features, n_states)
            for i_e, name in enumerate(entity_names):
                for i_f, fname in enumerate(p_features):
                    effect[name][fname] = arr[:, i_e, i_f, :]
        else:
            # Gaussian/Poisson: one array per state (mean/variance or rate)
            state_arrays = [arrays[k] for k in keys]  # each (n_samples, n_entities, n_features)
            for i_e, name in enumerate(entity_names):
                for i_f, fname in enumerate(p_features):
                    effect[name][fname] = np.column_stack(
                        [a[:, i_e, i_f] for a in state_arrays]
                    )
    return effect


def _concat_nested_dict(
    dicts: list[dict[str, dict[str, NDArray]]],
    outer_keys: list[str],
    inner_keys: list[str],
) -> dict[str, dict[str, NDArray]]:
    """Concatenate matching arrays across a list of {outer: {inner: array}} dicts."""
    return {
        ok: {
            ik: np.concatenate([d[ok][ik] for d in dicts], axis=0)
            for ik in inner_keys
        }
        for ok in outer_keys
    }


def concat_dicts_recursive(
    dicts: list[dict[str, object]],
    concat: callable[list[object], object]
):
    assert len(dicts) > 0

    combined = {}
    for k in dicts[0].keys():
        values = [d[k] for d in dicts]
        if isinstance(values[0], dict):
            combined[k] = concat_dicts_recursive(values, concat)
        else:
            combined[k] = concat(values)
    return combined


def _extract_feature_names(parameters: pd.DataFrame) -> list[str]:
    prefix = "w_areal_"
    return [c[len(prefix):] for c in parameters.columns if c.startswith(prefix)]


def _extract_state_names(parameters: pd.DataFrame, prefix: str) -> list[str]:
    return [c[len(prefix):] for c in parameters.columns if c.startswith(prefix)]
