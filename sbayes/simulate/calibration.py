
import arviz as az
import logging
import math
import numpy as np
import pandas as pd
import re
import warnings

from jax import random
from matplotlib import pyplot as plt
from pathlib import Path
from ruamel.yaml import YAML
from typing import NamedTuple, Sequence


from sbayes.results import Results
from sbayes.sampling.loggers import logger
from sbayes.simulate.config import SimulateConfigFile
from sbayes.simulate.simulator import Simulator
from sbayes.simulate.validate import Paths, build_result_paths
from sbayes.util import EPS

# For running on the server only
import subprocess

R_HAT_MAX = 1.02
"""The largest tolerable rank-normalised R-hat. 
Applied to the maximum over all monitored parameters, so looser than the 1.01"""

ESS_MIN = 400.0
"""The smallest tolerable effective sample size."""

MAX_FAILURE_RATE = 0.1
"""Above this share of failed replicates the study is re-run."""

MIN_REPLICATES_FOR_TOLERANCE = 20
"""Below this, the failure rate is too noisy to judge against the tolerance."""

AREAL_EFFECT = "areal_effect"
"""The `Results` attribute holding the cluster effects."""

CLUSTERS = "clusters"
"""The `Results` attribute holding the cluster memberships."""

MONITORED_PARAMETERS = ("weights", CLUSTERS, AREAL_EFFECT, "confounding_effects")

CLUSTER_INDEXED = ("clusters", "areal_effect")
"""Flattened parameters that index clusters, and so must be reordered."""

DIRECT_TARGET_PREFIXES = ("weights", AREAL_EFFECT, "confounding_effects")
"""Parameters tested by ranking the truth among the draws."""


class ReplicateSeeds(NamedTuple):
    """The seeds of one replicate, all derived from the study's master seed."""

    replicate: int
    """Index of the replicate, starting at 0."""

    assignment: int
    """Seeds NumPy's global generator for the hard cluster assignments."""

    simulation: int
    """Builds the JAX key that draws the parameters and simulates the features."""

    sampler: int
    """Builds the JAX key that runs the MCMC."""


class Replicate(NamedTuple):
    """The ground truth and the posterior of one replicate."""

    truth: Results
    """The simulated parameter set, a single draw."""

    chains: list[Results]
    """One entry per chain of the inference run, in file order."""


class Diagnostics(NamedTuple):
    """The convergence diagnostics of one replicate."""

    max_r_hat: float
    """The largest R-hat over all monitored parameters."""

    min_ess: float
    """The smallest effective sample size over all monitored parameters."""

    worst_r_hat_parameter: str
    """The parameter the largest R-hat belongs to."""

    worst_ess_parameter: str
    """The parameter the smallest effective sample size belongs to."""


class Gate(NamedTuple):
    """Which replicates of a study converged well enough to be used."""

    passed: list[int]
    """Indices of the replicates that met both thresholds."""

    failed: list[int]
    """Indices of the replicates that did not."""

    reasons: dict[int, str]
    """Per failed replicate, which threshold it missed and by how much."""

    failure_rate: float
    """The share of replicates that failed, out of those that ran."""

    within_tolerance: bool
    """Whether the failure rate is at or below the tolerance. Only meaningful when
    `tolerance_assessable` is True."""

    tolerance_assessable: bool
    """Whether there were enough replicates for the failure rate to mean anything."""


class PreparedReplicate(NamedTuple):
    """One replicate, ready to be turned into ranks."""

    truth: dict[str, np.ndarray]
    """The simulated parameter set, clusters reordered."""

    draws: dict[str, np.ndarray]
    """The posterior, clusters reordered and thinned. """

    diagnostics: Diagnostics
    """From the unthinned draws, after reordering."""

    decisiveness: float
    """Mean `ordering_decisiveness` over the draws. Low means clusters are har to distinguish."""

    cluster_names: Sequence[str]
    """The cluster names in the order they index `clusters`, from `Results`."""


class Study(NamedTuple):
    """A calibration study on disk, as run or as found again later."""

    path: Path
    """The study folder, holding one `replicate_###` folder per replicate."""

    config: SimulateConfigFile
    """The config the replicates were run with. Needed to locate their result files."""

    replicates: dict[int, Path]
    """The folder of each replicate, keyed by index."""

    seeds: list[ReplicateSeeds]
    """The seeds every replicate was run with. Empty if missing."""

    @classmethod
    def from_path(cls, path: Path, config: SimulateConfigFile) -> "Study":
        """Find a study that was run earlier, so it can be analysed on its own.

        Args:
            path: the study folder
            config: the config the study was run with

        Returns:
            The study, with whatever replicate folders and seeds are present.

        Raises:
            FileNotFoundError: if the folder holds no replicates.
        """
        folders = {
            int(p.name.removeprefix("replicate_")): p
            for p in path.iterdir()
            if p.is_dir() and re.fullmatch(r"replicate_\d+", p.name)
        }
        if not folders:
            raise FileNotFoundError(f"No replicate folders in '{path}'.")

        seed_log = path / "seeds.tsv"
        seeds = []
        if seed_log.exists():
            rows = pd.read_csv(seed_log, sep="\t")
            seeds = [
                ReplicateSeeds(*row)
                for row in rows[list(ReplicateSeeds._fields)].itertuples(index=False)
            ]
        else:
            warnings.warn(f"No seeds in '{path}'; replicates cannot be re-run.")

        return cls(
            path=path,
            config=config,
            replicates=dict(sorted(folders.items())),
            seeds=seeds,
        )


class EcdfBands(NamedTuple):
    """A confidence band for the ECDF of ranks under uniformity."""

    grid: np.ndarray
    """The possible rank values, 0..n_draws. shape: (n_draws + 1,)"""

    expected: np.ndarray
    """The uniform CDF at each grid point. shape: (n_draws + 1,)"""

    lower: np.ndarray
    """The lower edge of the band. shape: (n_draws + 1,)"""

    upper: np.ndarray
    """The upper edge of the band. shape: (n_draws + 1,)"""


def _seed_from(sequence: np.random.SeedSequence) -> int:
    """Draw one uint32 seed, valid for NumPy's generator and for `jax.random.PRNGKey`."""
    return int(sequence.generate_state(1, dtype=np.uint32)[0])

def _as_block(table: pd.DataFrame) -> str:
    """Format a table as a fenced code block, so no Markdown extras are needed."""
    return "```\n" + table.to_string(index=False) + "\n```"


def replicate_seeds(master_seed: int, n_replicates: int) -> list[ReplicateSeeds]:
    """Derive independent seeds for every replicate of a calibration study.

    Every seed follows deterministically from `master_seed`, so the whole study is
    reproducible from that one number and any single replicate can be re-run on its
    own.

    Args:
        master_seed: the study's single seed
        n_replicates: the number of replicates to derive seeds for

    Returns:
        One entry per replicate, in order.
    """
    sequences = np.random.SeedSequence(master_seed).spawn(n_replicates)

    seeds = []
    for i, sequence in enumerate(sequences):
        assignment, simulation, sampler = sequence.spawn(3)
        seeds.append(ReplicateSeeds(
            replicate=i,
            assignment=_seed_from(assignment),
            simulation=_seed_from(simulation),
            # `MCMCSetup.sample` adds the run index, so leave room under the limit
            sampler=_seed_from(sampler) % (2**32 - 64),
        ))

    return seeds

def replicate_paths(config: SimulateConfigFile, replicate_path: Path) -> Paths:
    """Build the samples paths of one replicate, checking the cluster counts agree.

    A calibration study has to infer the same number of clusters it simulated, so the
    inference config must resolve to that single value.

    Args:
        config: the study's simulation config
        replicate_path: the folder of this replicate

    Returns:
        The samples files of the simulated truth and of the inference.

    Raises:
        ValueError: if inference is configured for a different number of clusters than
            the simulation, or for more than one.
    """
    simulated_k = config.simulation.clusters
    inferred = config.inf_config.model.clusters
    inferred_ks = [inferred] if isinstance(inferred, int) else list(inferred)

    if inferred_ks != [simulated_k]:
        raise ValueError(
            f"A calibration study must infer the number of clusters it simulated: "
            f"simulated {simulated_k}, but inference is configured for {inferred_ks}."
        )

    return build_result_paths(replicate_path, simulated_k=simulated_k, k=simulated_k)

def write_seeds(
    seeds: list[ReplicateSeeds], master_seed: int, path: Path
) -> None:
    """Write the seeds of a calibration study to a file.

    Repeats the master seed on every row, so a single file is enough
    to reproduce the study.

    Args:
        seeds: the per-replicate seeds, as returned by `replicate_seeds`
        master_seed: the seed they were derived from
        path: the TSV file to write
    """
    log = pd.DataFrame(seeds)
    log.insert(0, "master_seed", master_seed)

    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, sep="\t", index=False)


def run_replicate(
    config: SimulateConfigFile,
    seeds: ReplicateSeeds,
    output_path: Path,
    sample_from_prior: bool = False
) -> Paths:
    """Simulate one data set from the prior and run inference on it.

    This is one replicate of a calibration study: draw a parameter set from the prior,
    simulate features conditional on it, write both, then infer with the same model and
    the same priors. The simulated parameters are the ground truth that the posterior
    is compared against.

    Args:
        config: the study's simulation config.
        seeds: the seeds of this replicate, from `replicate_seeds`
        output_path: the output folder of this replicate.
        sample_from_prior: whether to sample from the prior for no-data inference
    Returns:
        The samples files of the simulated truth and of the inference.
    """
    replicate_config = config.model_copy(deep=True)
    replicate_config.simulation.output.path = output_path
    replicate_config.simulation.datasets = 1
    replicate_config.simulation.seed = seeds.assignment
    replicate_config.inf_config.mcmc.seed = seeds.sampler

    simulator = Simulator(replicate_config)
    simulator.prepare_simulation()
    simulator.simulate(random.PRNGKey(seeds.simulation))
    simulator.write_simulation()
    simulator.infer(sample_from_prior=sample_from_prior)

    sim_k = replicate_config.simulation.clusters
    if not isinstance(sim_k, int):
        raise ValueError(f"A simulation has a single cluster count, got {sim_k}.")

    return replicate_paths(replicate_config, output_path)


def run_study(
    config: SimulateConfigFile,
    output_path: Path,
    n_replicates: int | None = None,
    sample_from_prior: bool = False,
) -> Study:
    """Run every replicate of a calibration study.

    Each replicate gets its own folder and its own seeds, all derived from the master
    seed in the config. Replicates whose inference has already finished are skipped, so
    an interrupted study can be resumed by calling this again with the same arguments.

    Args:
        config: the simulation config
        output_path: Receives one `replicate_###` folder per
            replicate, plus the seed file.
        n_replicates: the number of replicates. Defaults to `simulation.datasets`.
        sample_from_prior: whether to sample from the prior for no-data inference

    Returns:
        The samples files of each replicate, in order.
    """
    if n_replicates is None:
        n_replicates = config.simulation.datasets

    master_seed = config.simulation.seed
    seeds = replicate_seeds(master_seed, n_replicates)
    write_seeds(seeds, master_seed, output_path / "seeds.tsv")

    replicates = {}
    for replicate_seed in seeds:
        replicate_path = output_path / f"replicate_{replicate_seed.replicate:03d}"
        paths = replicate_paths(config, replicate_path)

        if paths.simulated.exists() and paths.inferred:
            logger.info(f"Skipping replicate {replicate_seed.replicate}, already done.")
        else:
            run_replicate(config, replicate_seed, replicate_path,
                          sample_from_prior=sample_from_prior)

        replicates[replicate_seed.replicate] = replicate_path

    return Study(
        path=output_path, config=config, replicates=replicates, seeds=seeds
    )



def load_replicate(
    replicate_path: Path,
    config: SimulateConfigFile,
    burn_in: float = 0.0,
) -> Replicate:
    """Read the ground truth and the posterior chains of one replicate.

    Args:
        replicate_path: the folder of this replicate
        config: the study's simulation config
        burn_in: the proportion of each chain to discard from the start.

    Returns:
        The ground truth and one `Results` per chain.

    Raises:
        FileNotFoundError: if the ground truth or the inference results are missing.
    """
    paths = replicate_paths(config, replicate_path)

    if not paths.simulated.exists():
        raise FileNotFoundError(
            f"No simulated ground truth at '{paths.simulated}'. "
            f"Has this replicate been run?"
        )
    truth = Results.from_h5(paths.simulated, burn_in=0, do_match_clusters=False)

    chains = [
        Results.from_h5(path, burn_in=burn_in, do_match_clusters=False)
        for path in paths.inferred
    ]

    return Replicate(truth=truth, chains=chains)


def convergence_diagnostics(draws: dict[str, np.ndarray]) -> Diagnostics:
    """Summarise the convergence of one replicate over all monitored parameters.

    Both numbers are worst-case not averages.

    Args:
        draws: one entry per monitored scalar parameter, each of shape
            (n_chains, n_draws).

    Returns:
        The worst R-hat and the worst effective sample size, each with the parameter it
        belongs to.

    Raises:
        ValueError: if `draws` is empty, or if the chains are fewer than two or of
            inconsistent shape.
    """
    if not draws:
        raise ValueError("No parameters to compute convergence diagnostics for.")

    shapes = {name: np.shape(values) for name, values in draws.items()}
    if any(len(shape) != 2 for shape in shapes.values()):
        raise ValueError(f"Expected (n_chains, n_draws) per parameter, got: {shapes}")

    n_chains = next(iter(shapes.values()))[0]
    if n_chains < 2:
        raise ValueError(
            f"R-hat compares chains against each other, so at least two are needed, "
            f"got {n_chains}. Set `mcmc.runs` to 2 or more."
        )

    r_hats = {name: float(np.asarray(az.rhat(values))) for name, values in draws.items()}
    ess = {name: float(np.asarray(az.ess(values))) for name, values in draws.items()}

    worst_r_hat_parameter, max_r_hat = max(r_hats.items(), key=lambda item: item[1])
    worst_ess_parameter, min_ess = min(ess.items(), key=lambda item: item[1])

    return Diagnostics(
        max_r_hat=max_r_hat,
        min_ess=min_ess,
        worst_r_hat_parameter=worst_r_hat_parameter,
        worst_ess_parameter=worst_ess_parameter,
    )


def gate_replicates(
    diagnostics: dict[int, Diagnostics],
    r_hat_max: float = R_HAT_MAX,
    ess_min: float = ESS_MIN,
    max_failure_rate: float = MAX_FAILURE_RATE,
    min_replicates: int = MIN_REPLICATES_FOR_TOLERANCE,
) -> Gate:
    """Decide which replicates converged well enough to be used.

    Replicates are excluded only for failing these diagnostics.

    Args:
        diagnostics: the diagnostics of every replicate that ran, keyed by index.
        r_hat_max: the largest tolerable R-hat
        ess_min: the smallest tolerable effective sample size
        max_failure_rate: the share of failures above which a study is not interpreted
        min_replicates: the fewest replicates for which the failure rate is judged

    Returns:
        The passed and failed replicates, why each failure failed, the failure rate and
        whether it can be judged against the tolerance.

    Raises:
        ValueError: if no replicate diagnostics are given.
    """
    if not diagnostics:
        raise ValueError("No replicates to gate.")

    passed, failed, reasons = [], [], {}
    for replicate, diagnostic in sorted(diagnostics.items()):
        problems = []
        if diagnostic.max_r_hat > r_hat_max:
            problems.append(
                f"R-hat {diagnostic.max_r_hat:.4f} > {r_hat_max} "
                f"({diagnostic.worst_r_hat_parameter})"
            )
        if diagnostic.min_ess < ess_min:
            problems.append(
                f"ESS {diagnostic.min_ess:.0f} < {ess_min:.0f} "
                f"({diagnostic.worst_ess_parameter})"
            )

        if problems:
            failed.append(replicate)
            reasons[replicate] = "; ".join(problems)
        else:
            passed.append(replicate)

    failure_rate = len(failed) / len(diagnostics)

    return Gate(
        passed=passed,
        failed=failed,
        reasons=reasons,
        failure_rate=failure_rate,
        within_tolerance=failure_rate <= max_failure_rate,
        tolerance_assessable=len(diagnostics) >= min_replicates,
    )


def thin_draws(
    draws: dict[str, np.ndarray], n_target: int = 100
) -> dict[str, np.ndarray]:
    """Reduce the posterior draws to a small set of near-independent ones.


    Every parameter is thinned at the same indices, so each surviving draw is a whole
    parameter set. Each chain contributes the same number of draws, and the chains are concatenated.

    Args:
        draws: one entry per parameter, each of shape (n_chains, n_draws, *dims)
        n_target: the number of draws to keep in total. Must be a multiple of the chain count.

    Returns:
        One entry per parameter, each of shape (n_target, *dims).

    Raises:
        ValueError: if `draws` is empty, if the parameters disagree on the chain and
            draw counts, if `n_target` is not a multiple of the chain count, or if
            there are fewer draws per chain than requested.
    """
    if not draws:
        raise ValueError("No draws to thin.")

    leading_shapes = {np.shape(values)[:2] for values in draws.values()}
    if len(leading_shapes) != 1:
        raise ValueError(
            f"All parameters must share the same (n_chains, n_draws), got "
            f"{sorted(leading_shapes)}."
        )
    n_chains, n_draws = leading_shapes.pop()

    per_chain, remainder = divmod(n_target, n_chains)
    if remainder:
        raise ValueError(
            f"n_target={n_target} is not a multiple of the {n_chains} chains, so the "
            f"chains could not contribute equally."
        )
    if per_chain > n_draws:
        raise ValueError(
            f"Cannot keep {per_chain} draws per chain: only {n_draws} are available."
        )

    # Evenly spaced, so the kept draws span the whole chain rather than one stretch
    indices = np.linspace(0, n_draws - 1, per_chain).round().astype(int)

    thinned = {}
    for name, values in draws.items():
        kept = np.asarray(values)[:, indices]  # (n_chains, per_chain, *dims)
        thinned[name] = kept.reshape(n_target, *kept.shape[2:])

    return thinned


def flatten_parameters(results: Results) -> dict[str, np.ndarray]:
    """Collect one chain's parameter arrays into a flat dict of named arrays.

    The effects are nested by partition, confounder, group, and they are flattened
    into dotted names.

    Args:
        results: the samples of one chain, or a simulated ground truth

    Returns:
        One entry per parameter array, keyed by its dotted name. Parameters the run
        does not carry are left out.
    """
    flat: dict[str, np.ndarray] = {}

    def add(name: str, value: object) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                add(f"{name}.{key}", inner)
        elif value is not None:
            flat[name] = np.asarray(value)

    for parameter in MONITORED_PARAMETERS:
        add(parameter, getattr(results, parameter, None))

    return flat


def expand_parameters(
    draws: dict[str, np.ndarray], n_leading: int = 2
) -> dict[str, np.ndarray]:
    """Expand the parameter arrays into one entry per scalar parameter each.
    The index becomes part of the name.

    Args:
        draws: one entry per parameter, each with `n_leading` sample axes first
        n_leading: the number of leading axes to keep, 2 for (n_chains, n_draws) and 1
            for pooled draws or for the ground truth

    Returns:
        One entry per scalar parameter, each keeping only the sampling axes.
    """
    expanded: dict[str, np.ndarray] = {}
    for name, values in draws.items():
        leading = (slice(None),) * n_leading
        for index in np.ndindex(*values.shape[n_leading:]):
            key = f"{name}[{','.join(str(i) for i in index)}]" if index else name
            expanded[key] = values[leading + index]

    return expanded


def _merge_chains(draws: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Merge the chain and draw axes into one, for operations that work per draw."""
    return {name: v.reshape(-1, *v.shape[2:]) for name, v in draws.items()}


def _split_chains(
    draws: dict[str, np.ndarray], n_chains: int
) -> dict[str, np.ndarray]:

    """Split the merged sample axis back into chains and draws."""
    return {name: v.reshape(n_chains, -1, *v.shape[1:]) for name, v in draws.items()}


def stack_chains(chains: list[Results]) -> dict[str, np.ndarray]:
    """Stack the parameter arrays of several chains on a leading chain axis.

    Args:
        chains: the chains of one replicate, as returned by `load_replicate`

    Returns:
        One entry per parameter, each of shape (n_chains, n_draws, *dims).

    Raises:
        ValueError: if no chains are given, if they carry different parameters, or if
            they differ in length.
    """
    if not chains:
        raise ValueError("No chains to stack.")

    per_chain = [flatten_parameters(chain) for chain in chains]

    names = set(per_chain[0])
    for i, chain in enumerate(per_chain[1:], start=1):
        if set(chain) != names:
            missing = sorted(names - set(chain))
            extra = sorted(set(chain) - names)
            raise ValueError(
                f"Chain {i} carries different parameters than chain 0. "
                f"Missing: {missing}. Unexpected: {extra}."
            )

    stacked = {}
    for name in sorted(names):
        arrays = [chain[name] for chain in per_chain]
        shapes = {array.shape for array in arrays}
        if len(shapes) != 1:
            raise ValueError(
                f"Chains disagree on the shape of '{name}': {sorted(shapes)}. "
                f"Chains of different length cannot be stacked."
            )
        stacked[name] = np.stack(arrays, axis=0)

    return stacked


def is_cluster_indexed(name: str) -> bool:
    """Whether a flattened parameter name indexes clusters.

    Args:
        name: a flattened parameter name, e.g. `areal_effect.Categorical[2]`

    Returns:
        True if the array's axis 1 runs over clusters.
    """
    return name.split(".")[0] in CLUSTER_INDEXED

def cluster_permutation(memberships: np.ndarray) -> np.ndarray:
    """Return the permutation that puts the clusters of one draw in a fixed order.

    Clusters are ordered by which objects they contain. The ordering object is the one
    whose membership differs most between clusters, i.e. Remaining clusters are separated
    by the next most decisive objects. The key depends only on this
    draw's parameters, never on the ground truth.

    Args:
        memberships: the continuous cluster memberships of one draw.
            shape: (n_clusters, n_objects)

    Returns:
        The cluster order, as indices into the input. shape: (n_clusters)
    """
    # How sharply each object distinguishes the clusters
    spread = memberships.max(axis=0) - memberships.min(axis=0)
    by_decisiveness = np.argsort(-spread, kind="stable")

    # `lexsort` takes its last key as the primary one, so the order is reversed
    keys = -memberships[:, by_decisiveness[::-1]].T
    return np.lexsort(keys)


def ordering_decisiveness(memberships: np.ndarray) -> float:
    """How sharply the ordering object separates the clusters, in [0, 1].

    A diagnostic for the ordering. Near 1, the clusters are cleanly
    distinguished and the order is stable across draws. Near 0, no object prefers one
    cluster over another.

    Args:
        memberships: the continuous cluster memberships of one draw.
            shape: (n_clusters, n_objects)

    Returns:
        The largest membership spread over all objects.
    """
    return float((memberships.max(axis=0) - memberships.min(axis=0)).max())


def reorder_clusters(
    draws: dict[str, np.ndarray], cluster_names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Put the clusters of every draw in a fixed, truth-independent order.

    Cluster labels are arbitrary. Before the ground truth can be compared against the
    posterior, both have to be put in order by the same rule, applied separately and
    without either consulting the other.

    Args:
        draws: one entry per parameter, each with the sample axis first. The truth is
            passed the same way, with a sample axis of length 1.
        cluster_names: the cluster names in the order they index `clusters`, from
            `Results.cluster_names`

    Returns:
        The same parameters, reordered per draw. Entries that do not depend on the
        cluster order are returned unchanged.

    Raises:
        ValueError: if `clusters` is missing, or if its cluster axis disagrees with
            `cluster_names`.
    """
    if "clusters" not in draws:
        raise ValueError(f"`clusters` is needed to order the clusters, not among: "
                         f"{sorted(draws)[:5]}...")

    memberships = draws["clusters"]
    n_samples, n_clusters = memberships.shape[:2]
    if n_clusters != len(cluster_names):
        raise ValueError(
            f"`clusters` has {n_clusters} clusters but {len(cluster_names)} names were "
            f"given: {list(cluster_names)}."
        )
    prefix = f"{AREAL_EFFECT}."

    suffixes = sorted({
        name.removeprefix(prefix).split(".", 1)[1]
        for name in draws if name.startswith(prefix)
    })

    affected = {CLUSTERS, *(
        f"{prefix}{name}.{suffix}"
        for name in cluster_names for suffix in suffixes
    )}


    reordered = {
        name: values.copy() if name in affected else values
        for name, values in draws.items()
    }

    permutations = np.array(
        [cluster_permutation(memberships[i]) for i in range(n_samples)]
    )  # shape: (n_samples, n_clusters)

    for i in range(n_samples):
        reordered["clusters"][i] = memberships[i][permutations[i]]

    # The effects move by key, in bulk: there are only n_clusters^2 source-target
    # pairs, so the draws needing each one are selected with a mask
    for target, target_name in enumerate(cluster_names):
        for source, source_name in enumerate(cluster_names):
            mask = permutations[:, target] == source
            if not mask.any():
                continue
            for suffix in suffixes:
                reordered[f"{prefix}{target_name}.{suffix}"][mask] = (
                    draws[f"{prefix}{source_name}.{suffix}"][mask]
                )

    return reordered


def prepare_replicate(
    replicate_path: Path,
    config: SimulateConfigFile,
    n_target: int = 100,
    burn_in: float = 0.0,
) -> PreparedReplicate:
    """Read one replicate and bring it into the form the rank statistics need.

    The clusters are reordered before the diagnostics, as each chain labels its clusters independently
    and R-hat could otherwise compare one chain's cluster 0 against another's cluster 1.
    Diagnostics are computed before thinning.

    Args:
        replicate_path: the folder of this replicate
        config: the study's simulation config
        n_target: the number of draws to keep for the ranks
        burn_in: the proportion of each chain to discard

    Returns:
        The ground truth and the posterior as named scalars, with the diagnostics.
    """
    replicate = load_replicate(replicate_path, config, burn_in=burn_in)
    draws = stack_chains(replicate.chains)

    n_chains = next(iter(draws.values())).shape[0]
    cluster_names = replicate.truth.cluster_names

    draws = _split_chains(
        reorder_clusters(_merge_chains(draws), cluster_names), n_chains
    )
    diagnostics = convergence_diagnostics(expand_parameters(draws, n_leading=2))

    decisiveness = float(np.mean([
        ordering_decisiveness(memberships)
        for chain in draws["clusters"]
        for memberships in chain
    ]))

    thinned = thin_draws(draws, n_target=n_target)
    truth = reorder_clusters(flatten_parameters(replicate.truth), cluster_names)

    return PreparedReplicate(
        truth=truth,
        draws=thinned,
        diagnostics=diagnostics,
        decisiveness=decisiveness,
        cluster_names=cluster_names
    )


def prepare_replicates(
    study: Study,
    n_target: int = 100,
    burn_in: float = 0.0,
) -> dict[int, PreparedReplicate]:
    """Read every replicate of a study that produced results.

    Replicates that did not run, or crashed, are reported as warnings and left out.
    They are a separate category from replicates that ran but failed the convergence
    gate, and are counted separately in the report.

    Args:
        study: a full calibration study
        n_target: the number of draws to keep for the ranks
        burn_in: the proportion of each chain to discard

    Returns:
        One entry per readable replicate, keyed by replicate index.
    """


    prepared: dict[int, PreparedReplicate] = {}
    for index, folder in study.replicates.items():
        try:
            prepared[index] = prepare_replicate(
                folder, study.config, n_target=n_target, burn_in=burn_in
            )
        except FileNotFoundError as e:
            warnings.warn(f"Skipping replicate {index}: {e}")

    return prepared


def rank_of_truth(
        truth: np.ndarray,
        draws: np.ndarray,
        rng: np.random.Generator | None = None
) -> int:
    """Rank the ground-truth value among the posterior draws of one parameter.

    The rank is the number of draws below the truth. It runs from 0 (the truth is
    smaller than every draw) to `n_draws` (larger than every draw). Under correct
    calibration this rank is uniform.

    Ties are broken at random, which matters for parameters that are not continuous -
    memberships saturating at 0 or 1, or a proportion over a finite set of pairs.
    Assigning every tie the same rank would pile mass at one position and look like
    miscalibration.

    Args:
        truth: the simulated value. A scalar, or an array with one element.
        draws: the posterior draws of the same parameter. shape: (n_draws,)
        rng: the generator used to break ties. Without one, ties all count as "above"
            the truth, which biases the ranks downwards.

    Returns:
        The rank, in 0..n_draws.

    Raises:
        ValueError: if `truth` is not a single value, or if `draws` is empty.
    """
    truth_value = np.asarray(truth).reshape(-1)
    if truth_value.size != 1:
        raise ValueError(f"Expected a single truth value, got {truth_value.size}.")

    draw_values = np.asarray(draws).reshape(-1)
    if draw_values.size == 0:
        raise ValueError("No draws to rank the truth among.")

    rank = int(np.sum(draw_values < truth_value))

    n_ties = int(np.sum(draw_values == truth_value))
    if n_ties and rng is not None:
        # The truth may sit anywhere within the tied block
        rank += int(rng.integers(n_ties + 1))

    return rank


def ranks_of_truth(
    truth: dict[str, np.ndarray],
    draws: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, int]:
    """Rank the ground truth among the posterior draws, for every parameter.

    Args:
        truth: one entry per scalar parameter, each a single value
        draws: the posterior draws of the same parameters, each of shape (n_draws,)
        rng: the generator used to break ties

    Returns:
        One rank per parameter.

    Raises:
        ValueError: if the two sides do not describe the same parameters.
    """
    missing = sorted(set(truth) - set(draws))
    extra = sorted(set(draws) - set(truth))
    if missing or extra:
        raise ValueError(
            f"The truth and the posterior describe different parameters. "
            f"Only in the truth: {missing[:5]}. Only in the posterior: {extra[:5]}."
        )

    return {
        name: rank_of_truth(truth[name], draws[name], rng=rng)
        for name in sorted(truth)
    }

def direct_targets(parameters: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Select the parameters that are tested by ranking the truth directly.

    Args:
        parameters: one entry per parameter, truth or draws

    Returns:
        The subset that is ranked directly, in the same form.
    """
    return {
        name: values for name, values in parameters.items()
        if name.split(".")[0] in DIRECT_TARGET_PREFIXES
    }


def state_counts(parameters: dict[str, np.ndarray]) -> dict[str, int]:
    """The size of each parameter's last axis, keyed by the unexpanded name.

    Args:
        parameters: one entry per parameter, before expansion, each with the sample
            axis first

    Returns:
        The last-axis size per parameter name. 1 for parameters with no trailing axis.
    """
    return {
        name: int(values.shape[-1]) if values.ndim > 1 else 1
        for name, values in parameters.items()
    }


def coassignment_proportion(memberships: np.ndarray) -> np.ndarray:
    """The probability that two objects share a cluster, averaged over all pairs.

    Memberships are probabilities, so two objects share a cluster with probability
    `sum_k z_ki z_kj`.

    Args:
        memberships: the cluster memberships. shape: (n_draws, n_clusters, n_objects)

    Returns:
        One proportion per draw, in [0, 1]. shape: (n_draws,)

    Raises:
        ValueError: if there are fewer than two objects, so there are no pairs.
    """
    n_objects = memberships.shape[-1]
    if n_objects < 2:
        raise ValueError(f"Need at least two objects to form a pair, got {n_objects}.")

    # sum over pairs i<j of sum_k z_ki z_kj, without building the n_objects^2 matrix
    total = np.square(memberships.sum(axis=-1)).sum(axis=-1)
    diagonal = np.square(memberships).sum(axis=(-2, -1))
    n_pairs = n_objects * (n_objects - 1) / 2

    return (total - diagonal) / 2 / n_pairs


def areal_mixture(
    draws: dict[str, np.ndarray], cluster_names: Sequence[str]
) -> dict[str, np.ndarray]:
    """The areal effect averaged over clusters, weighted by their membership mass.

    Args:
        draws: the parameters of one replicate, unexpanded, sample axis first
        cluster_names: the cluster names in the order they index `clusters`

    Returns:
        One entry per feature, each of shape (n_draws, n_states).
    """
    memberships = draws[CLUSTERS]                       # (n_draws, n_clusters, n_objects)
    mass = memberships.sum(axis=-1)                     # (n_draws, n_clusters)
    weights = mass / np.maximum(mass.sum(axis=-1, keepdims=True), EPS)

    prefix = f"{AREAL_EFFECT}."
    features = sorted({
        name.removeprefix(prefix).split(".", 1)[1]
        for name in draws if name.startswith(prefix)
    })

    mixture = {}
    for feature in features:
        by_cluster = np.stack(
            [draws[f"{prefix}{name}.{feature}"] for name in cluster_names], axis=1
        )                                               # (n_draws, n_clusters, n_states)
        mixture[feature] = np.einsum("nk,nks->ns", weights, by_cluster)

    return mixture


def summary_statistics(
    draws: dict[str, np.ndarray], cluster_names: Sequence[str]
) -> dict[str, np.ndarray]:
    """The order-independent statistics, one value per draw.

    Three statistics: how objects group together, where the cluster mixture sits, and
    how sharp it is.

    Args:
        draws: the parameters of one replicate, unexpanded, sample axis first
        cluster_names: the cluster names in the order they index `clusters`

    Returns:
        One entry per statistic, each of shape (n_draws,).
    """
    mixture = areal_mixture(draws, cluster_names)

    statistics: dict[str, np.ndarray] = {
        "coassignment": coassignment_proportion(draws[CLUSTERS]),
    }

    for feature, probabilities in mixture.items():
        # Normalised by the number of states, so features with 2, 3 and 4 states are
        # on the same scale and can be averaged
        n_states = probabilities.shape[-1]
        entropy = -np.sum(
            probabilities * np.log(np.maximum(probabilities, EPS)), axis=-1
        ) / np.log(n_states)

        statistics[f"mixture_probability.{feature}"] = probabilities[:, 0]
        statistics[f"mixture_entropy.{feature}"] = entropy

    for kind in ("mixture_probability", "mixture_entropy"):
        per_feature = [v for k, v in statistics.items() if k.startswith(f"{kind}.")]
        statistics[f"{kind}.mean"] = np.mean(per_feature, axis=0)

    return statistics


def replicate_ranks(
    prepared: PreparedReplicate, rng: np.random.Generator
) -> pd.DataFrame:
    """Rank the ground truth among the posterior draws, for one replicate.

    Two blocks of statistics, ranked the same way but meaning different things. The
    direct targets are individual parameters, and the cluster effects among them depend
    on the cluster ordering. The summaries are invariant to that ordering, so they stay
    informative when the ordering is decided by noise - see `decisiveness`.

    Args:
        prepared: the replicate, as returned by `prepare_replicate`
        rng: the generator used to break ties, shared across the whole study

    Returns:
        One row per statistic, with the rank and the columns needed to group the ranks
        in the plots.
    """
    counts = state_counts(direct_targets(prepared.draws))

    def n_states_of(name: str) -> int:
        """The state count a statistic belongs to, or 0 if it spans all features."""
        kind, _, rest = name.partition(".")
        if kind in DIRECT_TARGET_PREFIXES:
            return counts[name.split("[")[0]]
        if kind.startswith("mixture") and rest != "mean":
            # The feature's state count, read off its areal effect
            reference = f"{AREAL_EFFECT}.{prepared.cluster_names[0]}.{rest}"
            return counts.get(reference, 0)
        return 0

    # The individual parameters
    truth = expand_parameters(direct_targets(prepared.truth), n_leading=1)
    draws = expand_parameters(direct_targets(prepared.draws), n_leading=1)
    ranks = ranks_of_truth(truth, draws, rng=rng)

    # The order-independent summaries
    summary_truth = summary_statistics(prepared.truth, prepared.cluster_names)
    summary_draws = summary_statistics(prepared.draws, prepared.cluster_names)
    ranks |= ranks_of_truth(summary_truth, summary_draws, rng=rng)

    n_draws = len(next(iter(draws.values())))

    return pd.DataFrame([
        {
            "parameter": name,
            "kind": name.split(".")[0],
            "n_states": n_states_of(name),
            "rank": rank,
            "n_draws": n_draws,
        }
        for name, rank in ranks.items()
    ])


def collect_ranks(
    prepared: dict[int, PreparedReplicate],
    gate: Gate,
    master_seed: int,
) -> pd.DataFrame:
    """Collect the ranks of every replicate into one table.

    Args:
        prepared: the replicates, keyed by index
        gate: which replicates passed the convergence thresholds
        master_seed: the study's seed, from which the tie-breaking generator is derived

    Returns:
        One row per parameter per replicate.
    """
    rng = np.random.default_rng(master_seed)

    tables = []
    for replicate, one in sorted(prepared.items()):
        ranks = replicate_ranks(one, rng)
        ranks.insert(0, "replicate", replicate)
        ranks["passed_gate"] = replicate in set(gate.passed)
        ranks["decisiveness"] = one.decisiveness

        tables.append(ranks)

    return pd.concat(tables, ignore_index=True)


def rank_ecdf_bands(
    n_replicates: int,
    n_draws: int,
    prob: float = 0.95,
    n_simulations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> EcdfBands:
    """Build a confidence band for the ECDF of calibration ranks.

    If the ranks really were uniform, how far from the diagonal
    would their ECDF stray? It is simultaneous and the width is the
    distribution of the largest deviation anywhere along the curve, so a curve that
    leaves the band at any point is evidence against uniformity.

    The width is found by simulation: draw many sets of
    `n_replicates` uniform ranks, measure how far each set's ECDF strays, and take the
    `prob` quantile.

    Args:
        n_replicates: the number of ranks the ECDF is computed from - the replicate
            count, not the number of parameters
        n_draws: the number of posterior draws each rank was computed against, so the
            ranks run over 0..n_draws
        prob: the coverage of the band
        n_simulations: how many uniform rank sets to simulate
        rng: the generator used for the simulation

    Returns:
        The grid, the uniform CDF on it, and the two edges of the band.

    Raises:
        ValueError: if `n_replicates` or `n_draws` is not positive, or `prob` is not in
            (0, 1).
    """
    if n_replicates < 1 or n_draws < 1:
        raise ValueError(
            f"Need at least one replicate and one draw, got {n_replicates} and "
            f"{n_draws}."
        )
    if not 0.0 < prob < 1.0:
        raise ValueError(f"`prob` must be in (0, 1), but was {prob}.")

    if rng is None:
        rng = np.random.default_rng(0)

    n_values = n_draws + 1
    grid = np.arange(n_values)
    expected = np.asarray((grid + 1) / n_values, dtype=np.float64)

    # One row per simulated study, each holding `n_replicates` uniform ranks
    simulated = rng.integers(0, n_values, size=(n_simulations, n_replicates))

    # Counts per rank value, per simulation, in one pass
    offsets = simulated + np.arange(n_simulations)[:, None] * n_values
    counts = np.bincount(
        offsets.ravel(), minlength=n_simulations * n_values
    ).reshape(n_simulations, n_values)

    ecdfs = np.cumsum(counts, axis=1) / n_replicates
    width = float(np.quantile(np.abs(ecdfs - expected).max(axis=1), prob))

    return EcdfBands(
        grid=grid,
        expected=expected,
        lower=np.clip(expected - width, 0.0, 1.0),
        upper=np.clip(expected + width, 0.0, 1.0),
    )


def plot_rank_ecdf(
    ranks: pd.DataFrame,
    path: Path,
    prob: float = 0.95,
    passed_only: bool = True,
    group_by: Sequence[str] = ("kind", "n_states"),
) -> plt.Figure:
    """Plot the rank ECDFs of a calibration study, one panel per group of parameters.

    Each thin line is one parameter's ECDF over the replicates; the shaded band is
    where a uniform set of ranks would stay with probability `prob`. Under correct
    calibration the lines scatter around the diagonal and stay inside the band.
    The lines are drawn per parameter.

    Args:
        ranks: the table from `collect_ranks`
        path: the image file to write
        prob: the coverage of the band
        passed_only: whether to use only replicates that passed the convergence gate
        group_by: the columns whose combinations become panels

    Returns:
        The figure, already saved to `path`.

    Raises:
        ValueError: if the table is empty after filtering, or the draw count varies.
    """
    if passed_only:
        ranks = ranks[ranks["passed_gate"]]
    if ranks.empty:
        raise ValueError("No ranks left to plot.")

    draw_counts = sorted({int(value) for value in ranks["n_draws"].unique()})
    if len(draw_counts) != 1:
        raise ValueError(f"All ranks must share one draw count, got {draw_counts}.")
    n_draws = draw_counts[0]

    groups = list(ranks.groupby(list(group_by), sort=True))
    n_columns = min(3, len(groups))
    n_rows = math.ceil(len(groups) / n_columns)

    figure, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(4 * n_columns, 3.5 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )

    for axis, (key, group) in zip(axes.ravel(), groups):
        n_replicates = int(group.groupby("parameter").size().max())
        bands = rank_ecdf_bands(n_replicates, n_draws, prob=prob)

        axis.fill_between(
            bands.expected, bands.lower, bands.upper, alpha=0.2, label=f"{prob:.0%} band"
        )
        axis.plot(bands.expected, bands.expected, linewidth=1)

        for _, rows in group.groupby("parameter"):
            sorted_ranks = np.sort(rows["rank"].to_numpy())
            empirical = np.searchsorted(
                sorted_ranks, bands.grid, side="right"
            ) / len(sorted_ranks)
            axis.plot(bands.expected, empirical, linewidth=0.5, alpha=0.5)

        label = ", ".join(f"{column}={value}" for column, value in zip(group_by, np.atleast_1d(key)))
        axis.set_title(f"{label}  (n={n_replicates})", fontsize=9)

    for axis in axes.ravel()[len(groups):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("expected")
    for row in axes:
        row[0].set_ylabel("observed")

    figure.tight_layout()
    figure.savefig(path, dpi=150)

    return figure


def coverage_table(
    ranks: pd.DataFrame,
    levels: Sequence[float] = (0.5, 0.9, 0.95),
    group_by: Sequence[str] = ("kind", "n_states"),
) -> pd.DataFrame:
    """How often the ground truth fell inside the central credible interval.

    A  summary of the rank plots: coverage compresses the whole rank
    distribution into one number per level. The standard error is computed from the replicate count.

    Args:
        ranks: the table from `collect_ranks`
        levels: the nominal central probabilities to report
        group_by: the columns whose combinations become rows

    Returns:
        One row per group per level, with the observed and nominal coverage.

    Raises:
        ValueError: if `ranks` is empty or a level is not in (0, 1).
    """
    if ranks.empty:
        raise ValueError("No ranks to summarise.")
    for level in levels:
        if not 0.0 < level < 1.0:
            raise ValueError(f"Levels must be in (0, 1), but got {level}.")

    rows = []
    for key, group in ranks.groupby(list(group_by), sort=True):
        # The rank runs over 0..n_draws, so there are n_draws + 1 possible positions
        n_values = group["n_draws"].to_numpy() + 1
        position = (group["rank"].to_numpy() + 0.5) / n_values

        n_replicates = int(group["replicate"].nunique())
        n_parameters = int(group["parameter"].nunique())

        for level in levels:
            inside = np.abs(position - 0.5) <= level / 2
            rows.append({
                **dict(zip(group_by, np.atleast_1d(key))),
                "level": level,
                "coverage": float(inside.mean()),
                "standard_error": math.sqrt(level * (1 - level) / n_replicates),
                "n_parameters": n_parameters,
                "n_replicates": n_replicates,
            })

    return pd.DataFrame(rows)

def ecdf_excursions(
    ranks: pd.DataFrame,
    prob: float = 0.95,
    group_by: Sequence[str] = ("kind", "n_states"),
) -> pd.DataFrame:
    """How many parameters' rank ECDFs leave the confidence band.

    Each parameter's ECDF has probability `prob` of staying
    inside the band, so about `1 - prob` of them are expected to leave it by chance.

    Args:
        ranks: the table from `collect_ranks`
        prob: the coverage of the band
        group_by: the columns whose combinations become rows

    Returns:
        One row per group, with the observed and expected fraction outside the band.
    """
    rows = []
    for key, group in ranks.groupby(list(group_by), sort=True):
        n_draws = int(group["n_draws"].iloc[0])
        n_replicates = int(group.groupby("parameter").size().max())
        bands = rank_ecdf_bands(n_replicates, n_draws, prob=prob)

        outside = 0
        parameters = group.groupby("parameter")
        for _, rows_of_parameter in parameters:
            sorted_ranks = np.sort(rows_of_parameter["rank"].to_numpy())
            empirical = np.searchsorted(
                sorted_ranks, bands.grid, side="right"
            ) / len(sorted_ranks)
            if np.any(empirical < bands.lower) or np.any(empirical > bands.upper):
                outside += 1

        rows.append({
            **dict(zip(group_by, np.atleast_1d(key))),
            "n_parameters": parameters.ngroups,
            "n_replicates": n_replicates,
            "fraction_outside": outside / parameters.ngroups,
            "expected_outside": 1 - prob,
        })

    return pd.DataFrame(rows)


def write_study_report(
    study: Study,
    prepared: dict[int, PreparedReplicate],
    gate: Gate,
    ranks: pd.DataFrame,
    path: Path,
    levels: Sequence[float] = (0.5, 0.9, 0.95),
) -> None:
    """Write the summary of a calibration study as a Markdown file.

    The report states the settings, how many replicates survived which filter, and the
    coverage both with and without the replicates that failed the convergence gate.

    Args:
        study: the study the replicates belong to
        prepared: the replicates that could be read
        gate: the convergence gate applied to them
        ranks: the table from `collect_ranks`
        path: the Markdown file to write
        levels: the nominal coverage levels to report
    """
    simulation = study.config.simulation
    mcmc = study.config.inf_config.mcmc

    n_configured = len(study.seeds) or len(study.replicates)
    n_read = len(prepared)
    decisiveness = [one.decisiveness for one in prepared.values()]

    lines = [
        f"# Calibration study: {study.path.name}",
        "",
        "## Settings",
        "",
        f"- objects: {simulation.objects}",
        f"- features: {len(simulation.features)}",
        f"- clusters: {simulation.clusters} (simulated and inferred)",
        f"- confounders: {dict(simulation.confounders)}",
        f"- mcmc: {mcmc.steps} steps, {mcmc.samples} samples, {mcmc.runs} runs",
        f"- master seed: {simulation.seed}",
        "",
        "## Replicates",
        "",
        f"- configured: {n_configured}",
        f"- readable: {n_read}"
        + (f" ({n_configured - n_read} did not run or are incomplete)"
           if n_configured > n_read else ""),
        f"- passed the convergence gate: {len(gate.passed)}",
        f"- failed it: {len(gate.failed)} "
        f"({gate.failure_rate:.0%}, tolerance {MAX_FAILURE_RATE:.0%})",
        "",
        f"Gate: rank-normalised split R-hat < {R_HAT_MAX}, "
        f"minimum ESS >= {ESS_MIN:.0f}, over all monitored parameters.",
        "",
    ]

    if not gate.tolerance_assessable:
        lines += [
            f"> The failure rate is not assessable at {n_read} replicates "
            f"(at least {MIN_REPLICATES_FOR_TOLERANCE} are needed).",
            "",
        ]

    elif not gate.within_tolerance:
        lines += [
            "> **The failure rate is above tolerance, so these results are not "
            "interpretable.** Correct the sampler settings and run the study again.",
            "",
        ]

    if gate.failed:
        lines += ["### Why replicates failed", ""]
        lines += [f"- replicate {i}: {gate.reasons[i]}" for i in gate.failed]
        lines += [""]


    if decisiveness:
        lines += [
            "## Cluster ordering",
            "",
            f"Decisiveness of the ordering object: mean "
            f"{np.mean(decisiveness):.2f}, minimum {np.min(decisiveness):.2f}, "
            f"on a scale of 0 to 1.",
            "",
            "Near 1 the clusters are cleanly distinguished and the order is stable "
            "across draws. Near 0 the order is decided by noise, and the "
            "`areal_effect` results say little - the order-independent statistics "
            "(`coassignment`, `mixture_*`) carry the cluster findings instead.",
            "",
            "## Coverage",
            "",
            "The rank ECDF plots are the primary diagnostic; these numbers compress "
            "each of them to one value per level.",
            "",
            "## Band excursions",
            "",
            "How many parameters' rank ECDFs leave the confidence band. About "
            "`expected_outside` of them do so by chance; substantially more is "
            "evidence against uniformity.",
            "",
            _as_block(ecdf_excursions(ranks[ranks["passed_gate"]])),
            "",
            "### Replicates that passed the gate",
            "",
            _as_block(coverage_table(ranks[ranks["passed_gate"]], levels)),
            "",
            "### All readable replicates",
            "",
            _as_block(coverage_table(ranks, levels)),
            "",
            "## Files",
            "",
            "- `ranks.tsv` - one row per statistic per replicate",
            "- `rank_ecdf.png` - gate-passing replicates",
            "- `rank_ecdf_all.png` - all readable replicates",
            "- `seeds.tsv` - the seed of every replicate",
            "",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_no_data_check(
    config: SimulateConfigFile,
    output_path: Path,
    n_replicates: int = 100,
    prob: float = 0.95,
    bucket: str | None = None,
) -> None:
    """Validate the calibration pipeline on a problem whose answer is known.

    With the likelihood switched off on both sides there is nothing to learn from the
    data, so the posterior must equal the prior and the ranks must be uniform by
    construction.

    Args:
        config: the study's simulation config
        output_path: the folder for this check, kept separate from the real study
        n_replicates: the number of replicates. Cheap here, since no likelihood is
            evaluated, so prefer enough to read the plots - a hundred or more.
        prob: the coverage of the confidence bands
        bucket: the GCS bucket to upload the results to; None skips the upload

    Raises:
        Exception: if the upload fails, so the caller can keep the VM alive.
    """
    inf_config = config.inf_config

    if inf_config is None:
        raise ValueError(
            "The study config has no inference section, so there is nothing to "
            "calibrate against."
        )

    # Neither side reads this, but it makes the archived config reflect what ran
    inf_config.model.sample_from_prior = True
    archive_yaml(config, output_path)

    study = run_study(
        config=config,
        output_path=output_path,
        n_replicates=n_replicates,
        sample_from_prior=True,
    )
    try:
        prepared = prepare_replicates(study)
        gate = gate_replicates({i: p.diagnostics for i, p in prepared.items()})
        ranks = collect_ranks(prepared, gate, master_seed=config.simulation.seed)

        ranks.to_csv(study.path / "ranks.tsv", sep="\t", index=False)
        plot_rank_ecdf(ranks, study.path / "rank_ecdf.png", prob=prob)
        excursion = ecdf_excursions(ranks, prob=prob)
        write_study_report(study, prepared, gate, ranks, study.path / "report.md")
        write_excursions(excursion, study.path / "excursions.tsv")
    except Exception:
        logging.exception("The study failed; uploading whatever was written.")

    if bucket is not None:
        print(f"Uploading results from {study.path} to GCS bucket: {bucket}")
        try:
            upload_directory(study.path, bucket, "calibration")
        except Exception:
            logging.exception("Upload failed; results remain on disk.")
            raise
        print("Upload successful.")


def write_excursions(excursions: pd.DataFrame, path: Path) -> None:
    """Write the band excursions of a study or a check to a TSV file.

    Args:
        excursions: the table from `ecdf_excursions`
        path: the TSV file to write
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    excursions.to_csv(path, sep="\t", index=False)


def upload_directory(local_dir: Path, bucket: str, prefix: str) -> None:
    """Copy a directory to a GCS bucket, recursively.

    Args:
        local_dir: the local directory to copy
        bucket: the bucket to copy to
        prefix: the prefix to add to each copy
    """
    subprocess.run(
        ["gcloud", "storage", "rsync", "--recursive",
         str(local_dir), f"gs://{bucket}/{prefix}"],
        check=True,
    )


def archive_yaml(config: SimulateConfigFile, output_path: Path) -> None:
    """Archive the resolved configs, with all overrides applied.

    Two files, mirroring the inputs: the simulation settings and the inference
    settings.

    Args:
        config: the study's config, after any overrides
        output_path: the study folder

    Raises:
        ValueError: if the config has no inference section.
    """
    if config.inf_config is None:
        raise ValueError("The study config has no inference section to archive.")

    output_path.mkdir(parents=True, exist_ok=True)

    # The nested inference config is written separately, so neither file repeats it
    simulate = config.model_dump(mode="json", exclude={"inf_config"})
    inference = config.inf_config.model_dump(mode="json")

    yaml = YAML()
    yaml.default_flow_style = False

    for name, resolved in [
        ("config_simulate.yaml", simulate),
        ("config.yaml", inference),
    ]:
        with open(output_path / name, "w", encoding="utf-8") as f:
            yaml.dump(resolved, f)