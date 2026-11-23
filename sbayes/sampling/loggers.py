from __future__ import annotations

import jax
import jax.numpy as jnp
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import tables
import time
import warnings

from itertools import product
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.infer import log_likelihood
from numpyro.infer.util import log_density
from pathlib import Path
from sbayes.load_data import Data, CategoricalFeatures, GaussianFeatures, PoissonFeatures, GenericTypeFeatures
from sbayes.model import Model
from sbayes.util import PathLike, sample_categorical, get_best_permutation
from typing import Sequence, Any, Self


logger = logging.getLogger(__name__)

# Suppress warnings from PyTables about natural names
warnings.simplefilter("ignore", category=tables.NaturalNameWarning)

DENSITY_BATCH_SIZE = 10
"""Number of samples whose density is evaluated at once. Evaluating all samples in one
`vmap` holds the model's intermediates for every sample in memory at the same time."""

def density_params(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the sample entries to substitute when evaluating a density.

    Everything the model declares is substituted, so that nothing is re-drawn; only
    the sampler's own diagnostics are left out.
    """
    return {k: v for k, v in samples.items() if k != "potential_energy"}


class DerivedValues:

    """Computes the per-sample values stored in the `/derived` group of a samples file.

    These are the joint log density and, unless sampling from the prior, the pointwise
    log likelihood. The log density is log p(theta) + log p(x | theta), which is
    comparable to the likelihood. It is not `-potential_energy`: the sampler's potential
    lives in unconstrained space and includes the log Jacobian of the transforms.

    Samples are evaluated in blocks of `DENSITY_BATCH_SIZE`, the last one padded, so
    that each function is compiled once and memory stays bounded for any number of
    samples.
    """

    def __init__(self, model: Model, data: Data) -> None:
        """
        Args:
            model: the model the samples are drawn from
            data: the data the model is fitted to
        """
        self.data = data
        self.with_likelihood = not model.config.sample_from_prior

        seeded = handlers.seed(model.get_model, 0)
        self._log_posterior = jax.jit(
            jax.vmap(lambda sample: log_density(seeded, (), {}, sample)[0])
        )
        self._log_likelihood = jax.jit(lambda samples: log_likelihood(seeded, samples))

    def constants(self) -> dict[str, np.ndarray]:
        """Values that are the same for every sample, written once per file."""
        if not self.with_likelihood:
            return {}
        return {"na_values": self.data.features.missing.ravel()}

    def __call__(self, samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compute the derived values of every sample.

        Args:
            samples: the samples, with a leading sample axis, as drawn — before any
                cluster permutation, which would leave `z` and the cluster effects
                inconsistent

        Returns:
            `log_posterior` of shape (n_samples,) and, unless sampling from the prior,
            `likelihood` of shape (n_samples, n_objects * n_features), with missing
            values set to zero.
        """
        params = density_params(samples)
        n_samples = len(next(iter(params.values())))

        blocks: list[dict[str, np.ndarray]] = []
        for start in range(0, n_samples, DENSITY_BATCH_SIZE):
            block = _padded_block(params, start, DENSITY_BATCH_SIZE)
            n_valid = min(DENSITY_BATCH_SIZE, n_samples - start)
            values = {"log_posterior": np.asarray(self._log_posterior(block))}
            if self.with_likelihood:
                values["likelihood"] = self._flat_likelihood(block)
            blocks.append({k: v[:n_valid] for k, v in values.items()})

        return {
            key: np.concatenate([b[key] for b in blocks]).astype(np.float64)
            for key in blocks[0]
        }

    def _flat_likelihood(self, block: dict[str, np.ndarray]) -> np.ndarray:
        """Pointwise log likelihood of a block, flattened over objects and features."""
        by_partition = self._log_likelihood(block)
        n_block = len(next(iter(block.values())))

        features = self.data.features
        likelihood = np.empty((n_block,) + features.all_features.shape)
        for partition in features.partitions:
            likelihood[:, :, partition.feature_indices] = np.asarray(
                by_partition[f"x_{partition.name}"]
            )
        likelihood[:, features.missing] = 0.0
        return likelihood.reshape(n_block, -1)


def _padded_block(
    params: dict[str, np.ndarray], start: int, size: int
) -> dict[str, np.ndarray]:
    """Rows `start:start + size` of every array, padded by repeating the last row."""
    block = {}
    for key, values in params.items():
        rows = np.asarray(values[start:start + size])
        n_missing = size - len(rows)
        if n_missing > 0:
            rows = np.concatenate([rows, np.repeat(rows[-1:], n_missing, axis=0)])
        block[key] = rows
    return block


def _append_derived(
    file: tables.File, values: dict[str, np.ndarray], constants: dict[str, np.ndarray]
) -> None:
    """Append derived values to the `/derived` group, creating it if needed."""
    if "/derived" not in file:
        file.create_group(file.root, "derived")
        for name, value in constants.items():
            file.create_carray(
                where=file.root.derived,
                name=name,
                obj=value,
                filters=tables.Filters(complevel=9, fletcher32=True),
            )

    for name, value in values.items():
        if f"/derived/{name}" not in file:
            if name == "likelihood":
                # `bitshuffle` requires a blosc-family compressor
                filters = tables.Filters(
                    complevel=9, complib="blosc:zlib", bitshuffle=True, fletcher32=True
                )
            else:
                filters = tables.Filters(complevel=9, fletcher32=True)
            file.create_earray(
                where=file.root.derived,
                name=name,
                atom=tables.Float64Col(),
                shape=(0, *value.shape[1:]),
                filters=filters,
            )
        file.get_node("/derived", name).append(value)


def _has_derived(file: tables.File, n_samples: int, with_likelihood: bool) -> bool:
    """Whether `/derived` holds extendable arrays for exactly `n_samples` samples."""
    names = ["log_posterior", "likelihood"] if with_likelihood else ["log_posterior"]
    for name in names:
        path = f"/derived/{name}"
        if path not in file:
            return False
        node = file.get_node(path)
        if not isinstance(node, tables.EArray) or node.nrows != n_samples:
            return False
    return True


def write_samples(
    run: int,
    base_path: Path,
    samples: dict[str, np.ndarray],
    data: Data,
    model: Model,
    match_clusters: bool = True,
) -> None:
    """Write the samples of one run to a stats file and the samples file.

    Writes a TSV with one row per sample, holding the cluster sizes, the weights, the
    effect parameters and the log-densities (formatted for Tracer), and adds the
    metadata to the samples HDF5 file that the sampler wrote during the run. The
    log-densities are taken from the file's `/derived` group, and computed and written
    there only if they are missing.

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

    # The derived values are usually written by the online logger during the run, and
    # only computed here if they are missing. They must be computed before the cluster
    # permutation below: that permutes the cluster effects but leaves `z` as it is, so
    # afterwards the two no longer describe the same parameter point. Both are
    # invariant to the permutation itself, so the values apply to the permuted samples.
    samples_h5_path = base_path / f'samples_{run}.h5'
    with_likelihood = not model.config.sample_from_prior
    with _open_h5(samples_h5_path, mode="a") as h5_file:
        h5_file.set_node_attr(
            h5_file.root, "metadata", json.dumps(h5_metadata(data, n_clusters))
        )
        if not _has_derived(h5_file, n_samples, with_likelihood):
            if "/derived" in h5_file:
                h5_file.remove_node("/derived", recursive=True)
            derive = DerivedValues(model, data)
            _append_derived(h5_file, derive(samples), derive.constants())

        log_posterior = h5_file.root.derived.log_posterior.read()
        log_likelihood = (
            h5_file.root.derived.likelihood.read().sum(axis=-1)
            if with_likelihood else None
        )

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

    params_df["log_posterior"] = log_posterior
    if log_likelihood is not None:
        params_df["log_likelihood"] = log_likelihood
        params_df["log_prior"] = (
            params_df["log_posterior"] - params_df["log_likelihood"]
        )

    # The sampler's own diagnostic, in unconstrained space — not the log posterior
    if "potential_energy" in samples:
        params_df["potential_energy"] = samples["potential_energy"]

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


def h5_metadata(data: Data, n_clusters: int) -> dict:
    """Metadata stored in the samples h5 file, for `Results.from_h5()`."""
    return _build_h5_metadata(
        data,
        cluster_names=get_cluster_names(n_clusters),
        feature_names=data.features.names,
        component_names=['areal', *data.confounders.keys()],
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


def _open_h5(
    path: Path, mode: str, timeout: float = 600.0, interval: float = 5.0
) -> tables.File:
    """Open an h5 file, waiting while another process (e.g. a plot) holds a lock on it.

    Raises:
        tables.HDF5ExtError: if the file cannot be opened for another reason, or is
            still locked after `timeout` seconds.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            return tables.open_file(str(path), mode=mode)
        except tables.HDF5ExtError as e:
            if not _is_lock_error(e) or time.monotonic() > deadline:
                raise
            time.sleep(interval)


def _is_lock_error(error: tables.HDF5ExtError) -> bool:
    """Whether opening an h5 file failed because another process has it open."""
    return "unable to lock the file" in str(error)


def _num_rows(file: tables.File) -> int:
    """Number of samples in the sample arrays at the root of a file (0 if none)."""
    arrays = list(file.iter_nodes(file.root, classname="EArray"))
    return min((int(a.nrows) for a in arrays), default=0)


def _num_readable_rows(array: tables.Leaf) -> int:
    """Number of leading rows of `array` that can be read, chunk by chunk."""
    step = array.chunkshape[0] if array.chunkshape else max(array.nrows, 1)
    for start in range(0, array.nrows, step):
        try:
            array.read(start, min(start + step, array.nrows))
        except tables.HDF5ExtError:
            return start
    return array.nrows


class OnlineSampleLogger:

    """Continually writes the samples of one run to a pytables file (.h5).

    The samples are written in chunks during the run, so that a long run's results are
    available before it finishes and can be resumed after an interruption. The file is
    only open while a chunk is written, so it can be read (e.g. plotted) during the run
    and a run that is killed between writes leaves it intact. With `derive`, the
    derived values of each chunk (e.g. the log posterior) are written alongside it to
    the `/derived` group.
    """

    def __init__(
        self,
        base_path: PathLike,
        run: int,
        resume: bool,
        metadata: dict | None = None,
        derive: DerivedValues | None = None,
    ) -> None:
        """
        Args:
            base_path: directory to write the samples and sampler state to
            run: index of this run, used in the file names
            resume: if True, append to an existing samples file instead of overwriting
            metadata: metadata for `Results.from_h5`, written when the file is created
                (see `h5_metadata`)
            derive: computes the derived values written with each chunk. None = no
                `/derived` group is written during the run
        """
        self.base_path = Path(base_path)
        self.path = self.base_path / f'samples_{run}.h5'
        self.state_path = self.base_path / f'state_{run}.pkl'
        self.resume = resume
        self.metadata = metadata
        self.derive = derive
        # Whether the file holds this run's samples, so that writes append to it
        self._file_ready = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        pass

    def _prepare(self) -> None:
        """On resume, check the existing file and repair it if needed (only once)."""
        if self._file_ready or not self.resume:
            return

        try:
            file = _open_h5(self.path, mode="a")
        except tables.exceptions.HDF5ExtError as e:
            if _is_lock_error(e):
                # Still in use (e.g. by a plot or another run): never overwrite it
                raise
            logger.warning(
                f"Could not append to the existing sample file '{self.path.name}' "
                f"({type(e).__name__}). Overwriting the previous samples."
            )
            self.resume = False
            return

        with file:
            self._truncate_to_readable(file)
            self._update_derived(file)
            if self.metadata is not None:
                file.set_node_attr(file.root, "metadata", json.dumps(self.metadata))
        self._file_ready = True

    def _truncate_to_readable(self, file: tables.File) -> None:
        """Cut the sample arrays back to the rows that all of them can read.

        A run that is killed while writing can leave unreadable chunks at the end of
        the arrays, or arrays of different lengths. Truncating them lets the resumed
        run sample the lost steps again.
        """
        # The sample arrays and the derived values written alongside them
        arrays = list(file.walk_nodes(file.root, classname="EArray"))
        if not arrays:
            return

        n_readable = min(_num_readable_rows(a) for a in arrays)
        n_rows = max(a.nrows for a in arrays)
        if n_readable == n_rows:
            return

        logger.warning(
            f"The sample file '{self.path.name}' is damaged after sample {n_readable} "
            f"(of {n_rows}), probably because an earlier run was killed while writing. "
            f"Discarding the damaged samples and sampling them again."
        )
        for array in arrays:
            array.truncate(n_readable)

    def _update_derived(self, file: tables.File) -> None:
        """Make the `/derived` group match the samples of a resumed file.

        Files written without `derive`, or by earlier versions that wrote `/derived` once
        at the end of the run, have derived values that do not extend with the samples.
        They are recomputed from the samples in the file.
        """
        n_samples = _num_rows(file)
        if self.derive is not None and _has_derived(
            file, n_samples, self.derive.with_likelihood
        ):
            return

        if "/derived" in file:
            file.remove_node("/derived", recursive=True)
        if self.derive is None or n_samples == 0:
            return

        logger.info(
            f"Computing the derived values of the {n_samples} samples in "
            f"'{self.path.name}'."
        )
        samples = {
            leaf.name: leaf.read()
            for leaf in file.iter_nodes(file.root, classname="Leaf")
        }
        _append_derived(file, self.derive(samples), self.derive.constants())

    def write_sample(self, sample: dict[str, np.ndarray]) -> None:
        """Append one chunk of samples, creating the file and its arrays if needed."""
        self._prepare()

        # Drop the chain dimension
        sample = {name: np.array(values)[0] for name, values in sample.items()}
        # Computed before opening the file, so that it is open as briefly as possible
        derived = self.derive(sample) if self.derive is not None else None

        mode = "a" if self._file_ready else "w"
        with _open_h5(self.path, mode=mode) as file:
            if mode == "w" and self.metadata is not None:
                file.set_node_attr(file.root, "metadata", json.dumps(self.metadata))

            for param_name, values in sample.items():
                if f"/{param_name}" not in file:
                    file.create_earray(
                        where=file.root,
                        name=param_name,
                        atom=numpy_to_tables_dtype(values.dtype),
                        filters=tables.Filters(complevel=5),
                        shape=(0, *values.shape[1:]),
                    )
                file.root[param_name].append(values)

            if derived is not None:
                _append_derived(file, derived, self.derive.constants())
        self._file_ready = True

    def read_samples(self) -> dict[str, np.ndarray]:
        """Read all samples written to the file so far."""
        self._prepare()
        if not self._file_ready:
            raise ValueError(f"No samples have been written to '{self.path.name}' yet.")

        with _open_h5(self.path, mode="r") as file:
            # Only the sample arrays at the root, not the `/derived` group
            return {
                param.name: np.array(param)
                for param in file.iter_nodes(file.root, classname="Leaf")
            }

    def num_written_samples(self) -> int:
        """Number of (thinned) samples in the file, 0 if it holds none."""
        self._prepare()
        if not self._file_ready:
            return 0

        with _open_h5(self.path, mode="r") as file:
            if "/potential_energy" not in file:
                return 0
            # `nrows` is a NumPy integer, which NumPyro does not accept as a count
            return int(file.root.potential_energy.nrows)

    def dump_state(self, state) -> None:
        """Write the sampler state to disk, so that the run can be resumed."""
        # Write to a temporary file first, so that a crash cannot leave a partial state
        tmp_path = self.state_path.with_suffix(".pkl.tmp")
        with open(tmp_path, 'wb') as f:
            pickle.dump(state, f)
        os.replace(tmp_path, self.state_path)

    def load_state(self):
        """Read the sampler state of a previous run."""
        with open(self.state_path, 'rb') as f:
            return pickle.load(f)
