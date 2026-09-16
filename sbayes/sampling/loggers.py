from __future__ import annotations

import json
import logging
import numpy as np
import pandas as pd
import pickle
import tables
import warnings

from itertools import product
from numpyro import handlers
from numpyro.infer import log_likelihood
from pathlib import Path
from sbayes.load_data import Data, CategoricalFeatures, GaussianFeatures, PoissonFeatures, GenericTypeFeatures
from sbayes.model import Model
from sbayes.util import PathLike, sample_categorical, get_best_permutation
from typing import Sequence, Any, Self


logger = logging.getLogger(__name__)

# Suppress warnings from PyTables about natural names
warnings.simplefilter("ignore", category=tables.NaturalNameWarning)


def get_cluster_names(n_clusters: int) -> list[str]:
    """Create the names of the clusters, as used in the results files.

    Args:
        n_clusters: the number of clusters in the model

    Returns:
        One name per cluster, of the form `a0`, `a1`, ... (`a` for "area", the earlier
        term for a cluster).
    """
    return [f"a{i}" for i in range(n_clusters)]


def get_cluster_effect_names(partitions: list[GenericTypeFeatures]) -> list[str]:
    """Create the names of the cluster effect parameters, one or two per partition.

    The names must match the sample sites created by the `Model.add_partition_*`
    methods: Gaussian features have a mean and a variance, Poisson features a rate, and
    categorical features a single set of state probabilities.

    Args:
        partitions: the feature partitions of the data

    Returns:
        The parameter names, in the order of `partitions`.

    Raises:
        ValueError: if a partition has an unsupported feature type.
    """
    names = []
    for p in partitions:
        # Note: LogitNormalFeatures subclasses GaussianFeatures and is handled by the
        # Gaussian branch.
        if isinstance(p, CategoricalFeatures):
            names.append(f"cluster_effect_{p.name}")
        elif isinstance(p, GaussianFeatures):
            names.append(f"cluster_effect_{p.name}_mean")
            names.append(f"cluster_effect_{p.name}_variance")
        elif isinstance(p, PoissonFeatures):
            names.append(f"cluster_effect_{p.name}_rate")
        else:
            raise ValueError(
                f"Partition type {type(p).__name__} is not supported."
            )
    return names


def write_samples(
    run: int,
    base_path: Path,
    samples: dict[str, np.ndarray],
    data: Data,
    model: Model,
    match_clusters: bool = True,
) -> None:
    """Write the samples of one run to a stats file and the samples file.

    Two files are written: a TSV with one row per sample, holding the cluster sizes,
    the weights, the effect parameters and the log-densities (formatted for Tracer),
    and the metadata plus the pointwise likelihood, appended to the samples HDF5 file
    that the sampler wrote during the run.

    Args:
        run: index of this run, used in the file names
        base_path: directory to write the files to
        samples: the samples of this run, per parameter
        data: the data the model was fitted to
        model: the model that was sampled
        match_clusters: if True, permute the clusters of each sample so that cluster
            labels are consistent across samples
    """
    clusters_samples = sample_categorical(
        np.array(samples['z']), binary_encoding=True
    )[:, :, :-1].transpose(0, 2, 1)
    n_samples, n_clusters, n_objects = clusters_samples.shape

    partitions = data.features.partitions

    # Cast all parameters from jax.numpy to numpy
    samples = {k: np.array(v) for k, v in samples.items()}

    if match_clusters:
        # Cluster labels are arbitrary, so the clusters of each sample are permuted to
        # match the clusters seen so far. Note: only the binarized clusters and the
        # cluster effects are permuted - `z` in the samples file stays unpermuted.
        clusters_sum = np.zeros(clusters_samples.shape[1:], dtype=int)

        for i, clusters in enumerate(clusters_samples):
            permutation = get_best_permutation(clusters, clusters_sum)

            clusters = clusters[permutation]
            for param_name in get_cluster_effect_names(partitions):
                samples[param_name][i] = samples[param_name][i, permutation]

            clusters_sum += clusters
            clusters_samples[i] = clusters

    cluster_names = get_cluster_names(n_clusters)
    feature_names = data.features.names
    n_states = max(
        (p.n_states for p in data.features.categorical_partitions()), default=0
    )
    state_names = [f"s{s}" for s in range(n_states)]

    param_dfs_list: list[pd.DataFrame] = []

    cluster_sizes = np.sum(clusters_samples, axis=-1)

    sample_id_df = pd.DataFrame(data=np.arange(n_samples), columns=['Sample'])
    cluster_sizes_df = pd.DataFrame(
        data=cluster_sizes, columns=[f'size_{c}' for c in cluster_names]
    )
    param_dfs_list += [sample_id_df, cluster_sizes_df]

    # Transform the weights samples to a data frame
    # w shape: (n_samples, n_features, n_components)
    component_names = ['areal', *data.confounders.keys()]
    weights_df = samples_array_to_df(
        param_samples=samples['w'].transpose((0, 2, 1)),
        names=[component_names, feature_names],
        prefix='w',
    )
    param_dfs_list.append(weights_df)

    # If using varying weights per cluster, add them to the data frame list
    if "w_cluster" in samples:
        w_cluster_df = samples_array_to_df(
            param_samples=samples['w_cluster'],
            names=[cluster_names, feature_names],
            prefix='w_cluster',
        )
        param_dfs_list.append(w_cluster_df)

    all_cluster_eff_dfs: list[pd.DataFrame] = []
    all_conf_eff_dfs: list[pd.DataFrame] = []

    for partition in partitions:
        # Note: LogitNormalFeatures subclasses GaussianFeatures and is handled by the
        # Gaussian branch.
        if isinstance(partition, CategoricalFeatures):
            all_cluster_eff_dfs.append(samples_array_to_df(
                param_samples=samples[f'cluster_effect_{partition.name}'],
                names=[cluster_names, partition.names, state_names[:partition.n_states]],
                prefix='areal',
            ))

            for conf in data.confounders.values():
                all_conf_eff_dfs.append(samples_array_to_df(
                    param_samples=samples[f"conf_effect_{conf.name}_{partition.name}"],
                    names=[
                        conf.group_names,
                        partition.names,
                        state_names[:partition.n_states],
                    ],
                    prefix=conf.name,
                ))

        elif isinstance(partition, GaussianFeatures):
            for suffix in ("mean", "variance"):
                all_cluster_eff_dfs.append(samples_array_to_df(
                    param_samples=samples[f'cluster_effect_{partition.name}_{suffix}'],
                    names=[cluster_names, partition.names],
                    prefix='areal',
                    suffix=suffix,
                ))

                for conf in data.confounders.values():
                    all_conf_eff_dfs.append(samples_array_to_df(
                        param_samples=samples[
                            f"conf_effect_{conf.name}_{partition.name}_{suffix}"
                        ],
                        names=[conf.group_names, partition.names],
                        prefix=conf.name,
                        suffix=suffix,
                    ))

        elif isinstance(partition, PoissonFeatures):
            all_cluster_eff_dfs.append(samples_array_to_df(
                param_samples=samples[f'cluster_effect_{partition.name}_rate'],
                names=[cluster_names, partition.names],
                prefix='areal',
                suffix='rate',
            ))

            for conf in data.confounders.values():
                all_conf_eff_dfs.append(samples_array_to_df(
                    param_samples=samples[
                        f"conf_effect_{conf.name}_{partition.name}_rate"
                    ],
                    names=[conf.group_names, partition.names],
                    prefix=conf.name,
                    suffix='rate',
                ))

        else:
            raise NotImplementedError(
                f"Partition type {type(partition).__name__} is not supported."
            )

    # Add cluster and confounding effects to the data frame list
    param_dfs_list += all_cluster_eff_dfs + all_conf_eff_dfs

    params_df = pd.concat(param_dfs_list, axis=1)

    # `potential_energy` is the negative log-posterior and is always requested from the
    # sampler, but may be missing when the samples come from an older results file.
    if "potential_energy" in samples:
        params_df["log_posterior"] = samples["potential_energy"]
    else:
        params_df["log_posterior"] = np.nan

    # The pointwise likelihood, also written into the samples file below
    likelihoods_flat = None
    if not model.config.sample_from_prior:
        likelihoods_by_partition = log_likelihood(
            handlers.seed(model.get_model, 0), samples
        )
        likelihoods_flat = np.empty((n_samples,) + data.features.all_features.shape)
        for p in partitions:
            likelihoods_flat[:, :, p.feature_indices] = (
                likelihoods_by_partition[f"x_{p.name}"]
            )

        # Zero out missing values before summing
        likelihoods_flat[:, data.features.missing] = 0.0

        params_df["log_likelihood"] = likelihoods_flat.sum(axis=(-1, -2))
        params_df["log_prior"] = (
            params_df["log_posterior"] - params_df["log_likelihood"]
        )

    # Parameters that are only sampled under some settings
    optional_parameters = [
        "w_cluster_factor_c",
        "w_concentration",
        "z_concentration",
        "z_concentration_nocluster",
        "cluster_mask",
        "geoprior_scale",
        "geoprior",
        "geoprior_total_dist",
        "highest_z_penalty",
        "z0_stretch_factor",
    ]
    for param in optional_parameters:
        if param not in samples:
            continue

        values = samples[param]
        if values.ndim == 1:
            params_df[param] = values
        elif values.ndim == 2:
            for i in range(values.shape[1]):
                params_df[f"{param}_{i}"] = values[:, i]
        else:
            raise ValueError(
                f"Cannot write parameter `{param}` with shape {values.shape} to the "
                f"stats file: at most 2 dimensions are supported."
            )

    # Write params file (TSV for Tracer compatibility)
    params_path = base_path / f'stats_K{n_clusters}_{run}.tsv'
    with open(params_path, "w") as params_file:
        params_df.to_csv(params_file, sep='\t', index=False)

    # Write metadata (and optionally the likelihood) into the samples h5 file
    samples_h5_path = base_path / f'samples_{run}.h5'
    metadata = _build_h5_metadata(data, cluster_names, feature_names, component_names)

    with tables.open_file(str(samples_h5_path), mode="a") as h5_file:
        h5_file.set_node_attr(h5_file.root, "metadata", json.dumps(metadata))

        if likelihoods_flat is not None:
            if "/derived" not in h5_file:
                h5_file.create_group(h5_file.root, "derived")

            # noinspection PyTypeChecker
            # `bitshuffle` requires a blosc-family compressor
            lh_filters = tables.Filters(
                complevel=9, complib="blosc:zlib", bitshuffle=True, fletcher32=True
            )
            h5_file.create_carray(
                where=h5_file.root.derived,
                name="likelihood",
                obj=likelihoods_flat.reshape(n_samples, -1),
                atom=tables.Float64Col(),
                filters=lh_filters,
            )
            h5_file.create_carray(
                where=h5_file.root.derived,
                name="na_values",
                obj=data.features.missing.ravel(),
                atom=tables.BoolCol(),
                filters=tables.Filters(complevel=9, fletcher32=True),
            )


def _build_h5_metadata(
    data: Data,
    cluster_names: Sequence[str],
    feature_names: Sequence[str],
    component_names: Sequence[str],
) -> dict:
    """Build the metadata stored as an h5 attribute, for `Results.from_h5()`.

    The effect keys must match the sample sites created by the `Model.add_partition_*`
    methods, since they are how the results reader finds the parameters in the file.
    """
    partition_meta = []
    for partition in data.features.partitions:
        p_meta: dict[str, Any]= {
            "feature_names": list(partition.names),
            "type": str(partition.FEATURE_TYPE),
        }
        confounders = data.confounders.values()

        # Note: LogitNormalFeatures subclasses GaussianFeatures and is handled by the
        # Gaussian branch, but keeps its own FEATURE_TYPE above.
        if isinstance(partition, CategoricalFeatures):
            p_meta["state_names"] = [f"s{s}" for s in range(partition.n_states)]
            p_meta["cluster_effect_keys"] = [f"cluster_effect_{partition.name}"]
            p_meta["confounder_effect_keys"] = {
                conf.name: [f"conf_effect_{conf.name}_{partition.name}"]
                for conf in confounders
            }

        elif isinstance(partition, GaussianFeatures):
            p_meta["state_names"] = ["mean", "variance"]
            p_meta["cluster_effect_keys"] = [
                f"cluster_effect_{partition.name}_mean",
                f"cluster_effect_{partition.name}_variance",
            ]
            p_meta["confounder_effect_keys"] = {
                conf.name: [
                    f"conf_effect_{conf.name}_{partition.name}_mean",
                    f"conf_effect_{conf.name}_{partition.name}_variance",
                ]
                for conf in confounders
            }

        elif isinstance(partition, PoissonFeatures):
            p_meta["state_names"] = ["rate"]
            p_meta["cluster_effect_keys"] = [f"cluster_effect_{partition.name}_rate"]
            p_meta["confounder_effect_keys"] = {
                conf.name: [f"conf_effect_{conf.name}_{partition.name}_rate"]
                for conf in confounders
            }

        else:
            raise NotImplementedError(
                f"Partition type {type(partition).__name__} is not supported."
            )

        partition_meta.append(p_meta)

    return {
        "cluster_names": list(cluster_names),
        "feature_names": list(feature_names),
        "component_names": list(component_names),
        "confounders": {
            conf.name: list(conf.group_names) for conf in data.confounders.values()
        },
        "partitions": partition_meta,
    }


def numpy_to_tables_dtype(dtype: np.dtype) -> tables.Col:
    """Map a numpy dtype to the pytables column type used to store it.

    Args:
        dtype: the dtype of the samples to be stored

    Returns:
         The pytables column type. Floats are stored as float32, matching the model's
        `FLOAT_TYPE`.

    Raises:
        ValueError: if the dtype is not supported.
    """
    if np.issubdtype(dtype, np.integer):
        return tables.Int32Col()
    elif np.issubdtype(dtype, np.floating):
        return tables.Float32Col()
    elif np.issubdtype(dtype, np.bool_):
        return tables.BoolCol()
    else:
        raise ValueError(f"Unsupported numpy dtype: {dtype.name}")


def samples_array_to_df(
    param_samples: np.ndarray,
    names: Sequence[Sequence[str]],
    prefix: str = "",
    suffix: str = "",
) -> pd.DataFrame:
    """Flatten an array of samples into a data frame with one column per element.

    The first dimension of `param_samples` is the sample dimension; every remaining
    dimension is named by the corresponding entry of `names`, and the column names are
    the combinations of those names.

    Args:
        param_samples: the samples, of shape (n_samples, *param_dims)
        names: the names along each dimension of the parameter, one group per dimension
        prefix: prepended to every column name
        suffix: appended to every column name

    Returns:
        A data frame with `n_samples` rows and one column per parameter element.

    Raises:
        ValueError: if the names do not match the shape of the samples.
    """
    param_dims = param_samples.shape[1:]
    expected_dims = tuple(len(group) for group in names)
    if param_dims != expected_dims:
        raise ValueError(
            f"The samples have shape {param_dims} after the sample dimension, but the "
            f"given names imply {expected_dims}."
        )

    column_names = ['_'.join(combination) for combination in product(*names)]
    if prefix:
        column_names = [f"{prefix}_{column}" for column in column_names]
    if suffix:
        column_names = [f"{column}_{suffix}" for column in column_names]

    flattened_samples = param_samples.reshape(param_samples.shape[0], -1)
    return pd.DataFrame(flattened_samples, columns=column_names)


class OnlineSampleLogger:

    """Continually writes the samples of one run to a pytables file (.h5).

    The samples are written in chunks during the run, so that a long run's results are
    available before it finishes and can be resumed after an interruption.
    """

    def __init__(self, base_path: PathLike, run: int, resume: bool) -> None:
        """
        Args:
            base_path: directory to write the samples and sampler state to
            run: index of this run, used in the file names
            resume: if True, append to an existing samples file instead of overwriting
        """
        self.base_path = Path(base_path)
        self.path = self.base_path / f'samples_{run}.h5'
        self.state_path = self.base_path / f'state_{run}.pkl'
        self.resume = resume
        self.file: tables.File | None = None
        self._header_written = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def open(self) -> None:
        """Open the samples file, appending to it when resuming."""
        if not self.resume:
            self.file = tables.open_file(str(self.path), mode="w")
            return

        try:
            self.file = tables.open_file(str(self.path), mode="a")
        except tables.exceptions.HDF5ExtError as e:
            logger.warning(
                f"Could not append to the existing sample file '{self.path.name}' "
                f"({type(e).__name__}). Overwriting the previous samples."
            )
            self.resume = False
            self.file = tables.open_file(str(self.path), mode="w")

    def close(self) -> None:
        """Close the samples file, if it is open."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def write_sample(self, sample: dict[str, np.ndarray]) -> None:
        """Write one chunk of samples, creating the file and its arrays if needed."""
        if self.file is None:
            self.open()

        file = self.file
        if file is None:
            raise ValueError(f"Could not open the samples file '{self.path}'.")

        if not self._header_written:
            self._write_header(file, sample)
            self._header_written = True

        for param_name, param_values in sample.items():
            file.root[param_name].append(np.array(param_values)[0])
        file.flush()

    def _write_header(self, file: tables.File, sample: dict[str, np.ndarray]) -> None:
        """Create one extendable array per parameter, sized from the first chunk."""
        if self.resume:
            return

        for param_name, param_value in sample.items():
            squeezed = np.array(param_value)[0]
            _, *param_dims = squeezed.shape
            file.create_earray(
                where=file.root,
                name=param_name,
                atom=numpy_to_tables_dtype(squeezed.dtype),
                filters=tables.Filters(complevel=5),
                shape=(0, *param_dims),
            )

    def read_samples(self) -> dict[str, np.ndarray]:
        """Read all samples written to the file so far."""
        if self.file is None:
            raise ValueError(
                f"The samples file '{self.path.name}' is not open. `write_sample` or "
                f"`open` must be called before reading."
            )
        # noinspection PyProtectedMember
        return {param._v_name: np.array(param) for param in self.file.root}

    def dump_state(self, state) -> None:
        """Write the sampler state to disk, so that the run can be resumed."""
        with open(self.state_path, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self):
        """Read the sampler state of a previous run."""
        with open(self.state_path, 'rb') as f:
            return pickle.load(f)
