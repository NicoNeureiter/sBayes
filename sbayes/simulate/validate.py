"""Compare simulated ground truth against the inference run on it.

A simulation draws parameters and features from the prior, so the true values are
known. Running inference on the simulated data and comparing the two shows whether the
sampler recovers what was simulated: well-calibrated inference puts the simulated value
inside the credible interval of its posterior in about 95% of cases.

Reads the samples files of both sides, aligns the arbitrary cluster labels, pairs the
parameters, and plots each parameter's simulated value against its inferred samples.
"""
from __future__ import annotations

import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, NamedTuple

import numpy as np
from numpy.typing import NDArray

from sbayes.results import Results

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class Paths(NamedTuple):
    """The sample files of the simulated ground truth and of the inference on it."""

    simulated: Path
    """`sim_{i}/sim_params/samples_0.h5`, the parameters the data were drawn from."""

    inferred: Path
    """`sim_{i}/results/simK{simulated}/K{k}/samples_0.h5`, the inferred samples."""


class SimulationComparison(NamedTuple):
    """The simulated ground truth of one data set and the inference on it."""

    simulated: Results
    """The parameters the data were drawn from. One sample."""

    inferred: Results
    """The inferred samples, with clusters permuted to match the simulated ones."""


def align_to_simulation(simulated: Results, inferred: Results) -> Results:
    """Permute the inferred clusters to match the simulated ones.

    Cluster labels are arbitrary, so the inferred cluster `i` need not correspond to
    the simulated cluster `i`. The permutation with the highest overlap between the
    mean cluster assignments is applied to the inferred results.

    Args:
        simulated: the simulated ground truth, used as the reference
        inferred: the inferred samples

    Returns:
        `inferred` with its clusters and cluster effects in the simulated order. Its
        `parameters` table is not permuted, so comparisons should use `clusters` and
        `areal_effect`.
    """
    return Results.align_results_list([simulated, inferred])[1]


def read_simulations(
    base_path: Path, simulated_k: int, k: int, burn_in: float = 0.1
) -> dict[str, SimulationComparison]:
    """Read every simulated data set together with the inference on it.

    Data sets whose inference has not been run, or is missing for this cluster count,
    are reported as warnings and left out.

    Args:
        base_path: the simulation output directory
        simulated_k: the number of clusters the data were simulated with
        k: the number of clusters used for inference
        burn_in: the proportion of inferred samples to discard from the start of the
            chain. The simulated side is a single draw and is never cut.

    Returns:
        One entry per data set, keyed by folder name, with the clusters aligned.
    """
    sim_folders = find_simulation_folders(base_path) or [base_path]

    comparisons: dict[str, SimulationComparison] = {}
    for sim_folder in sim_folders:
        paths = build_result_paths(sim_folder, simulated_k, k)
        if not paths.simulated.exists() or not paths.inferred.exists():
            missing = [str(p) for p in paths if not p.exists()]
            warnings.warn(f"Skipping {sim_folder.name}, missing: {missing}")
            continue

        # The simulated side holds one sample, so there is nothing to match it against
        simulated = Results.from_h5(paths.simulated, burn_in=0, do_match_clusters=False)
        inferred = Results.from_h5(paths.inferred, burn_in=burn_in)
        comparisons[sim_folder.name] = SimulationComparison(
            simulated=simulated, inferred=align_to_simulation(simulated, inferred)
        )

    return comparisons


def build_result_paths(sim_folder: Path, simulated_k: int, k: int) -> Paths:
    """Build the paths of the simulated and inferred samples of one data set.

    Args:
        sim_folder: the folder of the data set, e.g. `<output>/sim_0`
        simulated_k: the number of clusters the data were simulated with
        k: the number of clusters used for inference

    Returns:
        The two samples files.
    """
    return Paths(
        simulated=sim_folder / "sim_params" / "samples_0.h5",
        inferred=sim_folder / "results" / f"simK{simulated_k}" / f"K{k}" / "samples_0.h5",
    )


def find_simulation_folders(base_path: Path) -> list[Path]:
    """Find the folders of the simulated data sets, in order.

    Data sets are written to `sim_{index}` folders; anything else in the output
    directory, such as the feature types YAML, is ignored. Sorting by the index rather
    than by name keeps `sim_2` before `sim_10`.

    Args:
        base_path: the simulation output directory

    Returns:
        One folder per data set, ordered by index. Empty if the simulation wrote a
        single data set, which goes directly into `base_path`.
    """
    folders = [p for p in base_path.iterdir() if p.is_dir() and re.fullmatch(r"sim_\d+", p.name)]
    return sorted(folders, key=lambda p: int(p.name.removeprefix("sim_")))


class ParameterComparison(NamedTuple):
    """One parameter, as simulated and as inferred."""

    title: str
    """A readable description, e.g. `Feature f_3, s1 in cluster a0`."""

    simulated: float
    """The simulated value."""

    inferred: NDArray[np.floating]
    """The inferred samples of that value. shape: (n_samples,)"""


def state_names_of(results: Results, feature: str) -> list[str]:
    """Return the labels of the last axis of a feature's effect samples.

    The effect samples of a feature have shape (n_samples, n_states), where the last
    axis means something different per feature type: the states for categorical
    features, and the parameters of the distribution for the others — `mean` and
    `variance` for Gaussian features, `rate` for Poisson ones. The labels are used in
    the titles of the comparison plots.

    Args:
        results: the results holding the feature
        feature: the feature name

    Returns:
        One label per entry of the last axis.

    Raises:
        ValueError: if the feature is not in `results`.
    """
    if feature not in results.feature_names:
        raise ValueError(f"Unknown feature '{feature}'.")
    return list(results.feature_states[results.feature_names.index(feature)])


def component_names_of(results: Results) -> list[str]:
    """Return the mixture components, in the order of the weight samples.

    The weights of a feature have shape (n_samples, n_components); the first component
    is the cluster effect, followed by one component per confounder, in the order the
    confounders appear in the results. The names are used in the titles of the
    comparison plots.

    Args:
        results: the results whose weights are described

    Returns:
        The component names, e.g. `["cluster", "family", "contact"]`.
    """
    return ["cluster", *results.groups_by_confounders]


def compare_parameters(comparison: SimulationComparison) -> Iterator[ParameterComparison]:
    """Pair every simulated parameter with the samples inferred for it.

    Covers the weights, the cluster effects and the confounding effects. The clusters
    must already be aligned, so that a cluster name means the same on both sides.

    Args:
        comparison: the simulated and inferred results of one data set

    Yields:
        One entry per scalar parameter, titled from its position in the results.

    Raises:
        ValueError: if the simulated results hold more than one sample, or a parameter
            is missing from the inferred results.
    """
    simulated, inferred = comparison
    states = {f: state_names_of(simulated, f) for f in simulated.feature_names}
    components = component_names_of(simulated)

    def pairs(
        sim: dict[str, NDArray], inf: dict[str, NDArray], labels: Callable[[str], list[str]],
        title: Callable[[str, str], str],
    ) -> Iterator[ParameterComparison]:
        missing = sim.keys() - inf.keys()
        if missing:
            raise ValueError(f"Missing from the inferred results: {sorted(missing)}")

        for feature, sim_array in sim.items():
            if sim_array.shape[0] != 1:
                raise ValueError(
                    f"The simulated results hold {sim_array.shape[0]} samples; expected one."
                )
            for i, label in enumerate(labels(feature)):
                yield ParameterComparison(
                    title=title(feature, label),
                    simulated=float(sim_array[0, i]),
                    inferred=inf[feature][:, i],
                )

    yield from pairs(
        simulated.weights, inferred.weights,
        labels=lambda feature: components,
        title=lambda feature, label: f"Weight of {label} for feature {feature}",
    )

    for cluster in simulated.cluster_names:
        yield from pairs(
            simulated.areal_effect[cluster], inferred.areal_effect[cluster],
            labels=lambda feature: states[feature],
            title=lambda feature, label, c=cluster: f"Feature {feature}, {label} in cluster {c}",
        )

    for confounder, groups in simulated.confounding_effects.items():
        for group, sim_effect in groups.items():
            yield from pairs(
                sim_effect, inferred.confounding_effects[confounder][group],
                labels=lambda feature: states[feature],
                title=lambda feature, label, c=confounder, g=group:
                    f"Feature {feature}, {label} in {c}, group {g}",
            )


def stack_comparisons(
    comparisons: dict[str, SimulationComparison]
) -> dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]]:
    """Collect each parameter across all data sets.

    One point of a comparison plot is one data set, so the values of a parameter are
    gathered over the simulations that contain it.

    Args:
        comparisons: the data sets, as returned by `read_simulations`

    Returns:
        Per parameter title, the simulated values, shape (n_datasets,), and the
        inferred samples, shape (n_datasets, n_samples).

    Raises:
        ValueError: if a parameter has different numbers of samples across data sets.
    """
    simulated: dict[str, list[float]] = defaultdict(list)
    inferred: dict[str, list[NDArray]] = defaultdict(list)

    for comparison in comparisons.values():
        for parameter in compare_parameters(comparison):
            simulated[parameter.title].append(parameter.simulated)
            inferred[parameter.title].append(parameter.inferred)

    stacked = {}
    for title, values in simulated.items():
        sample_counts = {len(samples) for samples in inferred[title]}
        if len(sample_counts) > 1:
            raise ValueError(
                f"'{title}' has {sorted(sample_counts)} samples in different data sets."
            )
        stacked[title] = (np.array(values), np.stack(inferred[title]))

    return stacked


def plot_simulated_against_inferred(
    simulated: NDArray[np.floating],
    inferred: NDArray[np.floating],
    title: str | None = None,
    ax: plt.Axes | None = None,
    interval: float = 0.95,
) -> NDArray[np.bool_]:
    """Plot simulated parameter values against the distributions inferred for them.

    Each simulated value is drawn on the x-axis against its inferred samples on the
    y-axis. Values the inference missed, i.e. that fall outside the central credible
    interval of their samples, are drawn in red, and the count of recovered values is
    written into the corner. A well-calibrated model recovers about `interval` of them
    and scatters them along the diagonal.

    Args:
        simulated: the simulated values. shape: (n_parameters,)
        inferred: the inferred samples of each value. shape: (n_parameters, n_samples)
        title: the title of the plot
        ax: the axes to draw on. Defaults to the current axes.
        interval: the width of the credible interval, between 0 and 1

    Returns:
        Which values fall inside their interval. shape: (n_parameters,)

    Raises:
        ImportError: if matplotlib is not installed.
        ValueError: if the shapes do not match, or `interval` is out of range.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Plotting needs matplotlib: pip install sbayes[tools]") from e

    if simulated.shape[0] != inferred.shape[0]:
        raise ValueError(
            f"{simulated.shape[0]} simulated values but {inferred.shape[0]} "
            f"inferred distributions."
        )
    if not 0 < interval < 1:
        raise ValueError(f"`interval` must be between 0 and 1, got {interval}.")

    if ax is None:
        ax = plt.gca()

    tail = 100 * (1 - interval) / 2
    low = np.percentile(inferred, tail, axis=1)
    high = np.percentile(inferred, 100 - tail, axis=1)
    recovered = (low < simulated) & (simulated < high)

    # One call per group instead of per parameter, which matters for hundreds of them
    for mask, color in ((recovered, "lightgrey"), (~recovered, "red")):
        if not mask.any():
            continue
        x = np.repeat(simulated[mask], inferred.shape[1])
        ax.plot(x, inferred[mask].ravel(), "o", markersize=1, alpha=0.1, color=color)

    limits = (min(inferred.min(), simulated.min()), max(inferred.max(), simulated.max()))
    ax.axline((0, 0), slope=1, color="black")
    ax.text(0.99, 0.01, f"{recovered.sum()} / {len(simulated)}",
            ha="right", va="bottom", transform=ax.transAxes)
    ax.set_xlabel("Simulated")
    ax.set_ylabel("Inferred")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.grid(False)

    if title:
        ax.set_title(title)

    return recovered


def plot_all_parameters(
    comparisons: dict[str, SimulationComparison],
    output_path: Path,
    interval: float = 0.95,
    columns: int = 4,
) -> dict[str, float]:
    """Plot every parameter's simulated values against the inferred distributions.

    Args:
        comparisons: the data sets, as returned by `read_simulations`
        output_path: the file to write the figure to
        interval: the width of the credible interval
        columns: how many plots to put next to each other

    Returns:
        The proportion of data sets in which each parameter was recovered. With
        well-calibrated inference and enough data sets, these are close to `interval`.

    Raises:
        ImportError: if matplotlib is not installed.
        ValueError: if there is nothing to plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Plotting needs matplotlib: pip install sbayes[tools]") from e

    stacked = stack_comparisons(comparisons)
    if not stacked:
        raise ValueError("No parameters to compare.")

    rows = -(-len(stacked) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)

    recovery = {}
    for ax, (title, (simulated, inferred)) in zip(axes.ravel(), stacked.items()):
        recovered = plot_simulated_against_inferred(simulated, inferred, title, ax, interval)
        recovery[title] = float(recovered.mean())

    for ax in axes.ravel()[len(stacked):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Wrote {len(stacked)} parameter plots to {output_path}")
    return recovery