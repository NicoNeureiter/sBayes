from __future__ import annotations

import json
import logging
import numpy as np
import pandas as pd
import tables
import warnings

from functools import partial
from numpy.typing import NDArray
from pathlib import Path
from sbayes.util import PathLike, get_best_permutation
from scipy.optimize import linear_sum_assignment
from typing import Sequence, Callable, Literal, Self, Any

logger = logging.getLogger(__name__)

# Column names used for the log-probabilities, in order of preference. Earlier versions
# of sBayes wrote the columns without the `log_` prefix.
LOG_POSTERIOR_COLUMNS = ("log_posterior", "posterior")
LOG_LIKELIHOOD_COLUMNS = ("log_likelihood", "likelihood")


class Results:

    """Class for reading, storing, summarizing results of a sBayes analysis.

    Attributes:
        clusters: cluster assignment samples.
            shape: (n_samples, n_clusters, n_objects)
        cluster_names: the names of the clusters
        feature_names: the names of the features
        feature_states: the states of each feature, in the order of `feature_names`
        groups_by_confounders: the group names of each confounder
        sample_id: the index of each sample. shape: (n_samples,)
        weights: the mixture weights per feature, each of
            shape (n_samples, n_components)
        areal_effect: the cluster effects, as
            {cluster_name: {feature_name: array(n_samples, n_states)}}
        confounding_effects: the confounding effects, as
            {confounder: {group: {feature: array(n_samples, n_states)}}}
        posterior: the log-posterior of each sample. shape: (n_samples,)
        likelihood: the log-likelihood of each sample. shape: (n_samples,)
        prior: the log-prior of each sample, derived from the two above
        likelihood_pointwise: the log-likelihood per sample and observation, with
            missing values excluded. shape: (n_samples, n_observations). Only available
            when read from an h5 file that contains the derived likelihood.
        parameters: the raw parameters data frame, only set by `from_csv_files`
    """

    def __init__(
        self,
        clusters: NDArray,
        weights: dict[str, NDArray],
        areal_effect: dict[str, dict[str, NDArray]],
        confounding_effects: dict[str, dict[str, dict[str, NDArray]]],
        cluster_names: list[str],
        feature_names: list[str],
        feature_states: list[list[str]],
        groups_by_confounders: dict[str, list[str]],
        sample_id: np.ndarray | None = None,
        log_posterior: NDArray[np.float64] | None = None,
        log_likelihood: NDArray[np.float64] | None = None,
        likelihood_pointwise: NDArray[np.float64] | None = None,
        parameters: pd.DataFrame | None = None,
    ) -> None:

        self.clusters = clusters
        self.cluster_names = cluster_names
        self.feature_names = feature_names
        self.feature_states = feature_states
        self.groups_by_confounders = groups_by_confounders

        self.sample_id = (
            sample_id if sample_id is not None
            else np.arange(self.n_samples)
        )

        self.weights = weights
        self.areal_effect = areal_effect
        self.confounding_effects = confounding_effects

        # The log-probabilities are stored without the `log_` prefix for backwards
        # compatibility, but they are log-densities throughout.
        self.posterior = log_posterior
        self.likelihood = log_likelihood
        if log_posterior is not None and log_likelihood is not None:
            self.prior = log_posterior - log_likelihood
        else:
            self.prior = None

        self.likelihood_pointwise = likelihood_pointwise

        # Legacy: only set by `from_csv_files`, for the alignment tools
        self.parameters = parameters

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_clusters(self) -> int:
        return self.clusters.shape[1]

    @property
    def n_samples(self) -> int:
        return self.clusters.shape[0]

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
        cls,
        results_list: list[Self],
        align_clusters: bool = False,
    ) -> Self:
        """Concatenate multiple Results objects along the samples axis.

        Args:
            results_list: the results to combine, which must describe the same clusters,
                objects and features
            align_clusters: if True, align the cluster labels of `results_list[1:]` to
                those of the first result before concatenating

        Returns:
            One Results object holding the samples of all inputs. A list with a single
            entry is returned unchanged, not copied.

        Raises:
            ValueError: if the list is empty or the results are not compatible.
        """
        if not results_list:
            raise ValueError("Cannot concatenate an empty list of Results.")

        ref = results_list[0]
        for i, r in enumerate(results_list[1:], 1):
            if (r.n_clusters, r.n_objects) != (ref.n_clusters, ref.n_objects):
                raise ValueError(
                    f"Results {i} has {r.n_clusters} clusters and {r.n_objects} "
                    f"objects, but the first has {ref.n_clusters} clusters and "
                    f"{ref.n_objects} objects."
                )
            if r.feature_names != ref.feature_names:
                raise ValueError(
                    f"Results {i} describes different features than the first result."
                )

        if len(results_list) == 1:
            return ref

        if align_clusters:
            results_list = cls.align_results_list(results_list)

        # Concatenate along the samples axis of each parameter
        concat_samples = partial(np.concatenate, axis=0)
        clusters = concat_samples([r.clusters for r in results_list])
        weights = concat_dicts_recursive(
            [r.weights for r in results_list], concat_samples
        )
        areal_effect = concat_dicts_recursive(
            [r.areal_effect for r in results_list], concat_samples
        )
        confounding_effects = concat_dicts_recursive(
            [r.confounding_effects for r in results_list], concat_samples
        )

        # The log-probabilities and the raw parameters are optional, and are only
        # carried over if every result has them
        def concat_if_complete(
            values: list[NDArray | None],
        ) -> NDArray | None:
            if any(v is None for v in values):
                return None
            return np.concatenate(values)

        log_posterior = concat_if_complete([r.posterior for r in results_list])
        log_likelihood = concat_if_complete([r.likelihood for r in results_list])
        likelihood_pointwise = concat_if_complete(
            [r.likelihood_pointwise for r in results_list]
        )

        parameters = None
        if all(r.parameters is not None for r in results_list):
            parameters = pd.concat(
                [r.parameters for r in results_list], ignore_index=True
            )

        return cls(
            clusters=clusters,
            weights=weights,
            areal_effect=areal_effect,
            confounding_effects=confounding_effects,
            cluster_names=ref.cluster_names,
            feature_names=ref.feature_names,
            feature_states=ref.feature_states,
            groups_by_confounders=ref.groups_by_confounders,
            sample_id=np.arange(len(clusters)),
            log_posterior=log_posterior,
            log_likelihood=log_likelihood,
            likelihood_pointwise=likelihood_pointwise,
            parameters=parameters,
        )

    def mean_clusters(self) -> NDArray:
        """The mean membership of each object in each cluster. shape: (n_clusters, n_objects)"""
        return self.clusters.mean(axis=0)

    def permute_clusters(
        self,
        permutation: NDArray[np.intp],
        cluster_names: list[str] | None = None,
    ) -> Self:
        """Relabel the clusters, so that new cluster `i` is old cluster `permutation[i]`.

        Args:
            permutation: the old cluster index of each new cluster. shape: (n_clusters,)
            cluster_names: the names of the new clusters. None = keep the current names.

        Returns:
            The results with permuted clusters and cluster effects. The weights and
            confounding effects are unchanged, since neither is indexed by cluster. The
            results are returned unchanged if neither the order nor the names change.
        """
        if cluster_names is None:
            cluster_names = self.cluster_names
        if len(cluster_names) != self.n_clusters or len(permutation) != self.n_clusters:
            raise ValueError(
                f"Expected {self.n_clusters} clusters, got a permutation of length "
                f"{len(permutation)} and {len(cluster_names)} cluster names."
            )
        if (np.all(permutation == np.arange(self.n_clusters))
                and list(cluster_names) == self.cluster_names):
            return self

        areal_effect = {
            name: self.areal_effect[self.cluster_names[int(i)]]
            for name, i in zip(cluster_names, permutation)
        }
        return type(self)(
            clusters=self.clusters[:, permutation],
            weights=self.weights,
            areal_effect=areal_effect,
            confounding_effects=self.confounding_effects,
            cluster_names=list(cluster_names),
            feature_names=self.feature_names,
            feature_states=self.feature_states,
            groups_by_confounders=self.groups_by_confounders,
            sample_id=self.sample_id,
            log_posterior=self.posterior,
            log_likelihood=self.likelihood,
            likelihood_pointwise=self.likelihood_pointwise,
            parameters=self.parameters,
        )

    @classmethod
    def align_results_list(cls, results_list: list[Self]) -> list[Self]:
        """Align the cluster labels of `results_list[1:]` to those of the first result.

        Cluster labels are arbitrary, so independent runs may have found the same
        clusters under different labels. The best matching permutation is found from the
        mean cluster assignments, using the Hungarian algorithm.

        Args:
            results_list: the results to align, the first of which is the reference

        Returns:
            The results with permuted clusters and cluster effects. The weights and
            confounding effects are unchanged, since neither is indexed by cluster.

        Raises:
            ValueError: if the list is empty.
        """
        if not results_list:
            raise ValueError("Cannot align an empty list of Results.")

        ref = results_list[0]

        aligned = [ref]
        for r in results_list[1:]:
            permutation = alignment_permutation(ref.clusters, r.clusters)
            aligned.append(r.permute_clusters(permutation, ref.cluster_names))

        return aligned

    # ----------------------------------------------------------------
    # Constructor: from consolidated h5 file (new format)
    # ----------------------------------------------------------------

    @classmethod
    def from_h5(
        cls,
        h5_path: PathLike,
        burn_in: float = 0.1,
        subsample_interval: int = 1,
        do_match_clusters: bool = True,
    ) -> Self:
        """Load the results of one run from its samples h5 file.

        The h5 file contains the parameter samples of the run and a JSON metadata
        attribute describing the feature names, partitions and confounders, so the
        results can be read without the stats file.

        Args:
            h5_path: path to the samples h5 file
            burn_in: proportion of the samples to discard from the start of the chain
            subsample_interval: keep only every n-th sample after the burn-in
            do_match_clusters: if True, permute the clusters of each sample so that
                cluster labels are consistent across samples. The matching runs over
                the whole chain, before the burn-in is removed, so the labels do not
                depend on `burn_in` or `subsample_interval` and agree with the stats
                file.

        Returns:
            The results of the run.

        Raises:
            ValueError: if `burn_in` is not in [0, 1), `subsample_interval` is not
                positive, or the file contains no metadata.
        """
        if not 0.0 <= burn_in < 1.0:
            raise ValueError(f"`burn_in` must be in [0, 1), but was {burn_in}.")
        if subsample_interval < 1:
            raise ValueError(
                f"`subsample_interval` must be positive, but was {subsample_interval}."
            )

        metadata, arrays = read_h5_samples(h5_path, match=do_match_clusters)
        cluster_names = metadata["cluster_names"]
        feature_names = metadata["feature_names"]
        confounders = metadata["confounders"]
        partitions = metadata["partitions"]

        # Apply burn-in and subsampling
        n_total = len(arrays["z"])
        indices = np.arange(int(burn_in * n_total), n_total, subsample_interval)
        arrays = {key: array[indices] for key, array in arrays.items()}

        w = arrays["w"]
        log_posterior = arrays.get("log_posterior")
        likelihood_pointwise = arrays.get("likelihood_pointwise")
        log_likelihood = (
            None if likelihood_pointwise is None else likelihood_pointwise.sum(axis=1)
        )

        feature_to_states = {}
        for p in partitions:
            for feature_name in p["feature_names"]:
                feature_to_states[feature_name] = p["state_names"]
        feature_states = [feature_to_states[f] for f in feature_names]

        weights = {f: w[:, i, :] for i, f in enumerate(feature_names)}

        areal_effect = _build_effect_dict(
            cluster_names, partitions, arrays,
            get_keys=lambda p: p["cluster_effect_keys"],
        )
        confounding_effects = {
            conf_name: _build_effect_dict(
                group_names, partitions, arrays,
                get_keys=lambda p, cn=conf_name: p["confounder_effect_keys"][cn],
            )
            for conf_name, group_names in confounders.items()
        }

        return cls(
            clusters=arrays["clusters"],
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
        cls,
        clusters_path: PathLike,
        parameters_path: PathLike,
        burn_in: float = 0.1,
        subsample_interval: int = 1,
        feature_names: list[str] | None = None,
        confounder_names: dict[str, list[str]] | None = None,
    ) -> Self:
        """Load the results of one run from the legacy cluster and stats files.

        The parameters are parsed from the column names of the stats file, so this
        reader depends on the column naming of the version that wrote them. Prefer
        `from_h5`, which reads the parameter names from the file's own metadata.

        Args:
            clusters_path: path to the clusters file (text or .npy)
            parameters_path: path to the stats file (TSV)
            burn_in: proportion of the samples to discard from the start of the chain
            subsample_interval: keep only every n-th sample after the burn-in
            feature_names: the feature names; inferred from the column names if not given
            confounder_names: the groups of each confounder; inferred from the column
                names if not given

        Returns:
            The results of the run.

        Raises:
            ValueError: if `burn_in` is not in [0, 1) or `subsample_interval` is not
                positive.
        """
        if not 0.0 <= burn_in < 1.0:
            raise ValueError(f"`burn_in` must be in [0, 1), but was {burn_in}.")
        if subsample_interval < 1:
            raise ValueError(
                f"`subsample_interval` must be positive, but was {subsample_interval}."
            )

        clusters = cls.read_clusters(clusters_path).astype(float, copy=False)
        parameters = cls.read_stats(parameters_path)

        # Apply the burn-in before subsampling, as `from_h5` does
        n_total = len(clusters)

        indices = np.arange(int(burn_in * n_total), n_total, subsample_interval)

        clusters = clusters[indices]
        parameters = parameters.iloc[list(indices)].reset_index(drop=True)

        # Extract names from the column headers
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
            log_posterior=_read_first_column(parameters, LOG_POSTERIOR_COLUMNS),
            log_likelihood=_read_first_column(parameters, LOG_LIKELIHOOD_COLUMNS),
            parameters=parameters,
        )

    # ----------------------------------------------------------------
    # Static utility methods (used by from_csv_files and migrate_results)
    # ----------------------------------------------------------------

    @staticmethod
    def read_clusters(
        txt_path: PathLike, subsample_interval: int = 1
    ) -> NDArray[np.float64] | NDArray[np.bool_]:
        """Read the cluster samples of one run from the legacy cluster file.

        Two legacy formats are supported, and they carry different information: a `.npy`
        file holds the continuous cluster membership (including the not-assigned
        component, which is dropped here), while a `.txt` file holds the discrete
        assignments. The `.txt` format has one line per sample, with tab-separated
        clusters and one `0`/`1` character per object.

        If a `.txt` path is given and a `.npy` file of the same name exists, the `.npy`
        file is preferred, since it carries the continuous memberships.

        Args:
            txt_path: path to the cluster file
            subsample_interval: keep only every n-th sample

        Returns:
            The cluster samples, of shape (n_samples, n_clusters, n_objects). Float for
            the `.npy` format, bool for the `.txt` format.

        Raises:
            ValueError: if the file does not have the expected shape.
        """
        path = Path(txt_path)

        # Prefer the .npy file, which holds the continuous cluster memberships
        npy_path = path.with_suffix(".npy")
        if path.suffix == ".txt" and npy_path.exists():
            warnings.warn(
                f"Reading the cluster memberships from '{npy_path.name}' instead of "
                f"'{path.name}', since the .npy file holds the continuous values."
            )
            path = npy_path

        if path.suffix == ".npy":
            clusters = np.load(path)
            if clusters.ndim != 3:
                raise ValueError(
                    f"Expected a 3D array in {path}, got shape {clusters.shape}."
                )
            if clusters.shape[2] < 2:
                raise ValueError(
                    f"Expected at least 2 components (including not-assigned) in "
                    f"{path}, got shape {clusters.shape}."
                )

            if subsample_interval > 1:
                clusters = clusters[::subsample_interval]

            # Drop the not-assigned component and move the cluster axis before objects
            clusters = clusters[..., :-1]
            return np.transpose(clusters, (0, 2, 1)).astype(float, copy=False)

        # One line per sample, tab-separated clusters, one character per object:
        # "0110\t1001" is two clusters over four objects
        with open(path, "r") as clusters_file:
            lines = clusters_file.read().split("\n")[::subsample_interval]
            samples = [
                [list(cluster) for cluster in line.split("\t")]
                for line in lines
                if line.strip()
            ]

        # shape: (n_samples, n_clusters, n_objects)
        return np.array(samples, dtype=int).astype(bool)

    @staticmethod
    def read_stats(
        stats_path: PathLike,
        subsample_interval: int = 1,
        use_pyarrow: bool = True,
    ) -> pd.DataFrame:
        """Read the stats of one run from its TSV file.

        Args:
            stats_path: path to the stats file. If a `.tsv` path does not exist, a
                `.txt` file of the same name is read instead (legacy format).
            subsample_interval: keep only every n-th sample
            use_pyarrow: whether to use the faster pyarrow engine. Ignored when
                subsampling, which pyarrow does not support.

        Returns:
            The stats, one row per sample.
        """
        path = Path(stats_path)

        # Fall back to .txt if .tsv does not exist (legacy format)
        if path.suffix == ".tsv" and not path.exists():
            txt_path = path.with_suffix(".txt")
            if txt_path.exists():
                path = txt_path

        skip_rows = None
        if subsample_interval > 1:
            skip_rows = lambda i: i > 0 and (i - 1) % subsample_interval != 0
            # The pyarrow engine does not support `skiprows` callables
            use_pyarrow = False

        if use_pyarrow:
            try:
                return pd.read_csv(path, delimiter="\t", engine="pyarrow")

            except ImportError:
                pass

            except Exception as e:
                warnings.warn(
                    f"Could not read '{path.name}' with the pyarrow engine "
                    f"({type(e).__name__}: {e}). Falling back to the default engine."
                )

        # The python engine is only needed for the `skiprows` callable
        engine: Literal["c", "python"] = "python" if skip_rows is not None else "c"
        return pd.read_csv(path, delimiter="\t", engine=engine, skiprows=skip_rows)


    @staticmethod
    def get_cluster_names(column_names: Sequence[str]) -> list[str]:
        """Extract the cluster names from the columns of a stats file.

        The cluster effect columns are named `areal_<cluster>_<feature>_<state>`, so the
        cluster names are the second element of those column names, in the order in
        which they first appear.

        Args:
            column_names: the column names of the stats file

        Returns:
            The names of the clusters.
        """
        cluster_names = []
        for column_name in column_names:
            if not column_name.startswith("areal_"):
                continue
            _, cluster_name, _ = column_name.split("_", maxsplit=2)
            if cluster_name not in cluster_names:
                cluster_names.append(cluster_name)
        return cluster_names


    @staticmethod
    def get_groups_by_confounder(
        column_names: Sequence[str],
    ) -> dict[str, list[str]]:
        """Extract the confounders and their groups from the columns of a stats file.

        The confounders are read from the weights columns, which are named
        `w_<component>_<feature>` and cover the clusters and every confounder. The
        groups of each confounder are then read from the confounding effect columns,
        which are named `<confounder>_<group>_<feature>_<state>`.

        Args:
            column_names: the column names of the stats file

        Returns:
            The group names of each confounder, in the order in which they first appear.
        """
        groups_by_confounder: dict[str, list[str]] = {}

        # The components of the weights are the clusters and the confounders
        for column_name in column_names:
            if not column_name.startswith("w_"):
                continue
            # `w_concentration_*` is a hyperparameter, not a component
            if column_name.startswith("w_concentration_"):
                continue

            _, component, _ = column_name.split("_", maxsplit=2)
            if component in ("areal", "cluster"):
                continue
            groups_by_confounder.setdefault(component, [])

        for confounder, groups in groups_by_confounder.items():
            for column_name in column_names:
                if not column_name.startswith(f"{confounder}_"):
                    continue
                _, group, _ = column_name.split("_", maxsplit=2)
                if group not in groups:
                    groups.append(group)

        return groups_by_confounder


# ----------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------

PER_CLUSTER_KEYS = ("w_cluster", "geoprior_total_dist")
"""Sample arrays, besides the cluster effects, that are indexed by cluster along their
second axis, so that they are permuted with the cluster labels."""


def read_h5_samples(
    h5_path: PathLike,
    extra_keys: Sequence[str] = (),
    match: bool = True,
    include_derived: bool = True,
) -> tuple[dict[str, Any], dict[str, NDArray]]:
    """Read the metadata and the sample arrays of one run from its samples h5 file.

    Reads the cluster assignments, the weights, the effect parameters and the derived
    log-densities of the whole chain, with no burn-in removed.

    Args:
        h5_path: path to the samples h5 file
        extra_keys: further sample arrays to read, each only if present in the file
        match: if True, permute the clusters of each sample so that cluster labels are
            consistent across the chain (see `match_clusters`)
        include_derived: if False, skip the log-densities in `/derived`, e.g. to avoid
            reading the (large) pointwise likelihood when only the clusters are needed

    Returns:
        The metadata, and the arrays by h5 key. Besides the keys read from the file,
        the arrays hold `clusters`, the continuous cluster memberships of shape
        (n_samples, n_clusters, n_objects), and, if present in the file,
        `log_posterior` and `likelihood_pointwise` (with the missing values removed).

    Raises:
        ValueError: if the file contains no metadata.
    """
    with tables.open_file(str(h5_path), mode="r") as f:
        try:
            metadata = json.loads(f.get_node_attr(f.root, "metadata"))
        except AttributeError:
            raise ValueError(
                f"No metadata in {h5_path}. Run "
                f"'python -m sbayes.tools.migrate_results <results_dir>' first."
            ) from None
        partitions = metadata["partitions"]

        # Old-format h5 files have a chain dimension at axis 1
        old_format = np.array(f.get_node(f.root, "z")).ndim == 4

        def read(key: str) -> NDArray:
            """Read one parameter array, dropping the chain dimension if present."""
            array = np.array(f.get_node(f.root, key))
            if old_format and array.ndim > 1:
                array = array[:, 0]
            return array

        # The keys of the effect parameters are taken from the metadata, so that
        # files written by earlier versions remain readable
        cluster_effect_keys = [
            key for p in partitions for key in p["cluster_effect_keys"]
        ]
        conf_effect_keys = [
            key
            for p in partitions
            for keys in p["confounder_effect_keys"].values()
            for key in keys
        ]

        # All per-sample arrays, so that they are cropped and subsampled together
        keys = ["z", "w", *cluster_effect_keys, *conf_effect_keys]
        keys += [k for k in extra_keys if k not in keys and f"/{k}" in f]
        arrays = {key: read(key) for key in keys}

        if include_derived and "derived" in f.root:
            derived = f.get_node(f.root, "derived")

            if "log_posterior" in derived:
                log_posterior = f.get_node(derived, "log_posterior")
                arrays["log_posterior"] = np.array(log_posterior)

            if "likelihood" in derived:
                likelihood = np.array(f.get_node(derived, "likelihood"))
                if "na_values" in derived:
                    na_values = np.array(f.get_node(derived, "na_values"))
                    likelihood = likelihood[:, ~na_values]
                arrays["likelihood_pointwise"] = likelihood

        if (include_derived and "log_posterior" not in arrays
                and "potential_energy" in f.root):
            # Some older files may only the sampler's potential, which lives in
            # unconstrained space and is therefore not comparable to
            # `log_likelihood`; `log_prior` derived from it is meaningless.
            warnings.warn(
                f"{h5_path} has no `/derived/log_posterior`. Falling back to "
                f"`potential_energy`, which is not a log posterior: the log "
                f"densities of this run cannot be compared or decomposed."
            )
            arrays["log_posterior"] = read("potential_energy")

    arrays = _crop_to_common_length(arrays)

    z = arrays["z"]
    clusters = np.transpose(z[:, :, :-1], (0, 2, 1)).astype(np.float64)
    if match:
        # Permutes `z` and the per-cluster arrays in place
        per_cluster_keys = cluster_effect_keys + [
            k for k in PER_CLUSTER_KEYS if k in arrays
        ]
        match_clusters(clusters, z, {k: arrays[k] for k in per_cluster_keys})
    arrays["clusters"] = clusters

    return metadata, arrays


def alignment_permutation(
    ref_clusters: NDArray, clusters: NDArray
) -> NDArray[np.intp]:
    """Find the relabelling of `clusters` that best matches `ref_clusters`.

    The clusters are compared by their mean memberships and matched one-to-one with
    the Hungarian algorithm, maximizing the total shared membership.

    Args:
        ref_clusters: the reference cluster samples. shape: (n_samples, n_clusters, n_objects)
        clusters: the cluster samples to align, with the same number of clusters and
            objects (the number of samples may differ)

    Returns:
        `perm` such that cluster `perm[i]` of `clusters` corresponds to reference
        cluster `i`. shape: (n_clusters,)
    """
    return match_mean_clusters(np.mean(ref_clusters, axis=0), np.mean(clusters, axis=0))


def match_mean_clusters(ref_mean: NDArray, mean: NDArray) -> NDArray[np.intp]:
    """Find the relabelling of the clusters in `mean` that best matches `ref_mean`.

    Same as `alignment_permutation`, but on the mean memberships, which is all the
    matching depends on.

    Args:
        ref_mean: mean membership of each object in each reference cluster.
            shape: (n_clusters, n_objects)
        mean: mean memberships of the clusters to align, of the same shape

    Returns:
        `perm` such that cluster `perm[i]` of `mean` corresponds to reference cluster
        `i`. shape: (n_clusters,)
    """
    if ref_mean.shape != mean.shape:
        raise ValueError(
            f"Cannot match {mean.shape[0]} clusters over {mean.shape[1]} objects to "
            f"{ref_mean.shape[0]} clusters over {ref_mean.shape[1]} objects."
        )
    return linear_sum_assignment(ref_mean @ mean.T, maximize=True)[1]


def cluster_overlap(ref_mean: NDArray, mean: NDArray) -> NDArray:
    """Overlap of each reference cluster with the (aligned) cluster in the same position.

    The overlap is the weighted Jaccard index of the mean memberships: 1 for clusters
    with the same memberships, 0 for clusters that share no objects.

    Args:
        ref_mean: mean membership of each object in each reference cluster.
            shape: (n_clusters, n_objects)
        mean: mean memberships of the aligned clusters, of the same shape

    Returns:
        One overlap per cluster. shape: (n_clusters,)
    """
    union = np.maximum(ref_mean, mean).sum(axis=-1)
    intersection = np.minimum(ref_mean, mean).sum(axis=-1)
    return np.divide(intersection, union, out=np.ones_like(union), where=union > 0)


def match_clusters(
    clusters: NDArray,
    z: NDArray,
    cluster_effect_arrays: dict[str, NDArray],
) -> None:
    """Align the cluster labels across samples, in place.

    Cluster labels are arbitrary, so the same cluster may carry different labels in
    different samples. Each sample is permuted to best match the clusters of the
    samples before it, so that a label means the same thing throughout the chain.

    Args:
        clusters: the cluster memberships, permuted in place.
            shape: (n_samples, n_clusters, n_objects)
        z: the cluster assignments, with the not-assigned component last. Permuted in
            place. shape: (n_samples, n_objects, n_clusters + 1)
        cluster_effect_arrays: the cluster effect parameters, permuted in place. Each of
            shape (n_samples, n_clusters, ...)
    """
    # Running total of the assignments seen so far, which each sample is matched against
    clusters_sum = np.zeros(clusters.shape[1:], dtype=float)

    for i in range(len(clusters)):
        permutation = get_best_permutation(clusters[i], clusters_sum)

        if not np.all(permutation == np.arange(len(permutation))):
            clusters[i] = clusters[i][permutation]
            z[i, :, :-1] = z[i, :, :-1][:, permutation]
            for effect_array in cluster_effect_arrays.values():
                effect_array[i] = effect_array[i][permutation]

        clusters_sum += clusters[i]


def _crop_to_common_length(arrays: dict[str, NDArray]) -> dict[str, NDArray]:
    """Crop all per-sample arrays to the length of the shortest one.

    The arrays grow by appending samples, so the rows they share are complete and
    aligned. A file read during a run can be caught between the appends of two
    arrays, leaving the last chunk in only some of them.
    """
    n_samples = min(len(array) for array in arrays.values())
    if any(len(array) != n_samples for array in arrays.values()):
        logger.debug(f"Cropping the sample arrays to {n_samples} common samples.")
    return {key: array[:n_samples] for key, array in arrays.items()}


def _build_effect_dict(
    entity_names: Sequence[str],
    partitions: list[dict],
    arrays: dict[str, NDArray],
    get_keys: Callable[[dict], list[str]],
) -> dict[str, dict[str, NDArray]]:
    """Build {entity_name: {feature_name: array}} from the arrays of an h5 file.

    Works for both cluster effects (entity = cluster) and confounding effects
    (entity = group); `get_keys` selects the relevant keys from a partition's metadata.

    Categorical partitions have one array with the states as its last dimension, while
    Gaussian and Poisson partitions have one array per parameter (mean and variance, or
    rate), which are stacked into that last dimension here.

    Args:
        entity_names: the clusters or the groups of one confounder
        partitions: the partition metadata read from the h5 file
        arrays: the parameter arrays, by h5 key
        get_keys: returns the keys of one partition's effect parameters

    Returns:
        The effect samples per entity and feature, each of shape (n_samples, n_states).
    """
    effect = {name: {} for name in entity_names}

    for partition in partitions:
        keys = get_keys(partition)
        feature_names = partition["feature_names"]

        if partition["type"] == "categorical":
            # shape: (n_samples, n_entities, n_features, n_states)
            array = arrays[keys[0]]
            for i_entity, entity_name in enumerate(entity_names):
                for i_feature, feature_name in enumerate(feature_names):
                    effect[entity_name][feature_name] = array[:, i_entity, i_feature, :]
        else:
            # One array per parameter, each of shape (n_samples, n_entities, n_features)
            parameter_arrays = [arrays[key] for key in keys]
            for i_entity, entity_name in enumerate(entity_names):
                for i_feature, feature_name in enumerate(feature_names):
                    effect[entity_name][feature_name] = np.column_stack(
                        [a[:, i_entity, i_feature] for a in parameter_arrays]
                    )

    return effect


def concat_dicts_recursive(
    dicts: list[dict[str, Any]],
    concat: Callable[[list], Any],
) -> dict[str, Any]:
    """Combine a list of nested dictionaries by concatenating their leaf values.

    The dictionaries must have the same structure. Nested dictionaries are combined
    recursively; every other value is passed to `concat` as a list of the values found
    under that key.

    Args:
        dicts: the dictionaries to combine, all with the same keys and nesting
        concat: combines the values found under one key, e.g. `np.concatenate`

    Returns:
        One dictionary with the same structure and the combined values.

    Raises:
        ValueError: if the list is empty.
    """
    if not dicts:
        raise ValueError("Cannot combine an empty list of dictionaries.")

    combined = {}
    for key in dicts[0]:
        values = [d[key] for d in dicts]
        if isinstance(values[0], dict):
            combined[key] = concat_dicts_recursive(values, concat)
        else:
            combined[key] = concat(values)

    return combined


def _extract_feature_names(parameters: pd.DataFrame) -> list[str]:
    """Extract the feature names from the columns of a legacy stats file.

    The weights columns are named `w_<component>_<feature>` and cover every feature, so
    the features are read from the columns of the cluster component.

    Args:
        parameters: the stats of one run

    Returns:
        The feature names, in the order in which they appear in the file.
    """
    return _strip_prefix(parameters.columns, prefix="w_areal_")


def _extract_state_names(parameters: pd.DataFrame, prefix: str) -> list[str]:
    """Extract the state names of one feature from the columns of a legacy stats file.

    The cluster effect columns are named `areal_<cluster>_<feature>_<state>`, so the
    states of a feature are found by stripping the prefix up to and including the
    feature name.

    Args:
        parameters: the stats of one run
        prefix: the column prefix of one cluster and feature, e.g. `areal_a0_F1_`

    Returns:
        The state names, in the order in which they appear in the file.
    """
    return _strip_prefix(parameters.columns, prefix=prefix)


def _strip_prefix(column_names: Sequence[str], prefix: str) -> list[str]:
    """Return the remainder of every column name that starts with `prefix`."""
    return [
        column_name[len(prefix):]
        for column_name in column_names
        if column_name.startswith(prefix)
    ]


def _read_first_column(
    parameters: pd.DataFrame, column_names: Sequence[str]
) -> NDArray[np.float64] | None:
    """Read the first of several alternative columns that is present in the stats.

    Used for values whose column name changed between versions, so that the current
    name is preferred and the older ones are still readable.

    Args:
        parameters: the stats of one run
        column_names: the candidate column names, in order of preference

    Returns:
        The values of the first column that is present, or None if none of them is.
    """
    for column_name in column_names:
        if column_name in parameters.columns:
            return parameters[column_name].to_numpy(dtype=float)
    return None