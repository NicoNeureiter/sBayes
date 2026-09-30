"""Compare sBayes runs by Bayesian cross validation (PSIS-LOO).

For every run, the pointwise likelihoods logged during sampling are turned into an
expected log pointwise predictive density (ELPD), estimated with Pareto-smoothed
importance sampling leave-one-out cross validation. Plotting the scores against the
number of clusters shows which cluster count predicts the data best.

Needs arviz, matplotlib and seaborn: pip install sbayes[tools]

Usage:
    python -m sbayes.tools.elpd results/ 0.1 --output elpd.pdf --csv elpd.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
import tables
import xarray as xr
from numpy.typing import NDArray

from sbayes.tools._h5 import parse_run_location, read_array
from sbayes.util import PathLike


class RunFile(NamedTuple):
    """One run's likelihood file, with the experiment, cluster count and run index."""

    path: Path
    experiment: str
    k: int
    run: int


# --- Reading -------------------------------------------------------------------------

def read_likelihood(likelihood_path: PathLike) -> tuple[NDArray, NDArray[np.bool_] | None]:
    """Read the pointwise likelihoods of one run from an h5 file.

    Reads the current format (`/derived/likelihood` in a samples file) and the legacy
    format (`/likelihood` in a separate likelihood file).

    Args:
        likelihood_path: the samples or likelihood h5 file

    Returns:
        The likelihoods, shape (n_samples, n_observations), and the NA mask,
        shape (n_observations,), or None if the file stores none.

    Raises:
        ValueError: if the file contains no likelihood data.
    """
    with tables.open_file(str(likelihood_path), mode="r") as f:
        if "/derived/likelihood" in f:
            group = "/derived"
        elif "/likelihood" in f:
            group = ""
        else:
            raise ValueError(f"No likelihood data found in {likelihood_path}.")

        likelihood = read_array(f, f"{group}/likelihood")
        na_path = f"{group}/na_values"
        is_na = read_array(f, na_path).astype(bool) if na_path in f else None

    return likelihood, is_na


def resolve_na(
    likelihood: NDArray,
    is_na: NDArray[np.bool_] | None,
    assume_from_likelihood: bool = False,
) -> NDArray[np.bool_]:
    """Determine which observations are missing.

    Args:
        likelihood: the pointwise likelihoods. shape: (n_samples, n_observations)
        is_na: the NA mask stored in the file, or None if it stores none
        assume_from_likelihood: if the file stores no mask, treat observations whose
            likelihood is 1.0 in every sample as missing. This is how missing values
            enter the likelihood, but an observation can also reach 1.0 legitimately,
            so it may drop real observations.

    Returns:
        The NA mask. shape: (n_observations,)

    Raises:
        ValueError: if the file stores no mask and `assume_from_likelihood` is False,
            or if the stored mask does not match the likelihood.
    """
    if is_na is not None:
        if is_na.shape != likelihood.shape[1:]:
            raise ValueError(
                f"The NA mask has shape {is_na.shape}, expected {likelihood.shape[1:]}."
            )
        return is_na

    if not assume_from_likelihood:
        raise ValueError(
            "The file stores no `na_values` array. Pass --assumeNaFromLikelihood to "
            "treat observations with a constant likelihood of 1.0 as missing."
        )

    guessed = np.all(np.isclose(likelihood, 1.0), axis=0)
    warnings.warn(
        f"No `na_values` array in the file: treating {guessed.sum()} observation(s) "
        f"with a constant likelihood of 1.0 as missing."
    )
    return guessed


def to_log_likelihood_tree(
    likelihood: NDArray, is_na: NDArray[np.bool_], burnin: float
) -> xr.DataTree:
    """Build the arviz input with the log-likelihoods of one run.

    Missing observations are dropped, the first `burnin` fraction of samples is
    discarded, and a leading chain axis of length 1 is added, since each run is a
    single chain. Zero likelihoods are clipped so that the log stays finite.

    Args:
        likelihood: the pointwise likelihoods. shape: (n_samples, n_observations)
        is_na: the NA mask. shape: (n_observations,)
        burnin: the fraction of samples to discard, in [0, 1)

    Returns:
        A tree with a single `log_likelihood` group holding the variable `y`, with
        dimensions (chain, draw, obs).

    Raises:
        ValueError: if `burnin` is out of range, or no samples or observations remain.
    """
    if not 0 <= burnin < 1:
        raise ValueError(f"The burn-in must be in [0, 1), got {burnin}.")

    likelihood = likelihood[:, ~is_na]
    likelihood = likelihood[int(burnin * likelihood.shape[0]):]
    if likelihood.shape[0] == 0 or likelihood.shape[1] == 0:
        raise ValueError(
            f"Nothing left to evaluate: {likelihood.shape[0]} sample(s), "
            f"{likelihood.shape[1]} observation(s) after dropping NAs and burn-in."
        )

    n_zero = int(np.sum(likelihood <= 0))
    if n_zero:
        warnings.warn(
            f"{n_zero} likelihood value(s) are zero and were clipped before taking the "
            f"log. The model assigns no probability to those observations."
        )
    log_likelihood = np.log(np.clip(likelihood, 1e-300, None))

    return xr.DataTree.from_dict({
        "log_likelihood": xr.Dataset(
            {"y": (("chain", "draw", "obs"), log_likelihood[np.newaxis])}
        )
    })


# --- Scoring -------------------------------------------------------------------------

def psis_loo(
    likelihood_path: PathLike, burnin: float, assume_na_from_likelihood: bool = False
) -> float:
    """Compute the PSIS-LOO score of one run.

    Warns if the Pareto k diagnostic exceeds arviz's threshold for any observation,
    which means the importance sampling did not converge there and the score is not
    trustworthy.

    The likelihood files hold no posterior draws, so the relative MCMC efficiency
    cannot be estimated and is set to 1, i.e. the draws are treated as independent.

    Args:
        likelihood_path: the samples or likelihood h5 file of one run
        burnin: the fraction of samples to discard, in [0, 1)
        assume_na_from_likelihood: infer the NA mask if the file stores none

    Returns:
        The expected log pointwise predictive density (ELPD).

    Raises:
        ImportError: if arviz is not installed.
    """
    try:
        import arviz as az
    except ImportError as e:
        raise ImportError("elpd needs arviz: pip install sbayes[tools]") from e

    likelihood, stored_na = read_likelihood(likelihood_path)
    is_na = resolve_na(likelihood, stored_na, assume_na_from_likelihood)
    data = to_log_likelihood_tree(likelihood, is_na, burnin)

    loo = az.loo(data, reff=1.0, pointwise=True)

    pareto_k = np.asarray(loo.pareto_k)
    bad = int(np.sum(pareto_k > loo.good_k))
    if bad:
        warnings.warn(
            f"{bad} of {pareto_k.size} observations have a Pareto k above "
            f"{loo.good_k:.2f} in {likelihood_path}. The ELPD is unreliable."
        )

    return float(loo.elpd)


def find_run_files(results_dir: Path) -> list[RunFile]:
    """Find the likelihood files of all runs below `results_dir`.

    Looks for samples files (current format, `.../{experiment}/K{k}/samples_{run}.h5`)
    and legacy likelihood files (`likelihood_K{k}_{run}.h5`) in the same place. Where
    both exist for a run, the samples file wins, so a migrated directory is not
    counted twice. Files of heated MC3 chains are skipped.

    Args:
        results_dir: the directory to search

    Returns:
        One entry per run, sorted by experiment, cluster count and run index.
    """
    by_run: dict[tuple[str, int, int], RunFile] = {}

    for pattern, is_legacy in (("samples_*.h5", False), ("likelihood_K*_*.h5", True)):
        for path in sorted(results_dir.rglob(pattern)):
            if ".chain" in path.name:
                continue
            try:
                k, run = parse_run_location(path)
            except ValueError as e:
                warnings.warn(f"Skipping {path}: {e}")
                continue

            experiment = path.parent.parent.name
            key = (experiment, k, run)
            if is_legacy and key in by_run:
                continue  # the samples file of this run was already found
            by_run[key] = RunFile(path, experiment, k, run)

    return [by_run[key] for key in sorted(by_run)]


def collect_elpd(
    run_files: Sequence[RunFile], burnin: float, assume_na_from_likelihood: bool = False
) -> pd.DataFrame:
    """Compute the PSIS-LOO score of every run.

    Runs whose likelihood cannot be evaluated are reported as warnings and left out.

    Args:
        run_files: the runs to evaluate
        burnin: the fraction of samples to discard, in [0, 1)
        assume_na_from_likelihood: infer the NA mask where a file stores none

    Returns:
        One row per run, with the columns `experiment`, `k`, `run` and `elpd_loo`.
    """
    rows = []
    for run_file in run_files:
        try:
            elpd = psis_loo(run_file.path, burnin, assume_na_from_likelihood)
        except Exception as e:
            warnings.warn(f"Skipping {run_file.path}: {e}")
            continue

        print(f"ELPD-LOO for {run_file.experiment} K{run_file.k} "
              f"run {run_file.run}: {elpd:.2f}")
        rows.append((run_file.experiment, run_file.k, run_file.run, elpd))

    return pd.DataFrame(rows, columns=["experiment", "k", "run", "elpd_loo"])


# --- Plotting ------------------------------------------------------------------------

def plot_elpd(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """Plot the ELPD per cluster count, or per experiment if there is only one K.

    Args:
        df: the scores, as returned by `collect_elpd`
        output_path: where to save the figure. If None, the figure is shown instead.

    Raises:
        ImportError: if matplotlib or seaborn are not installed.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        raise ImportError(
            "elpd needs matplotlib and seaborn: pip install sbayes[tools]"
        ) from e

    fig, ax = plt.subplots()
    if df["k"].nunique() == 1:
        sns.boxplot(df, x="experiment", y="elpd_loo", ax=ax)
    else:
        sns.lineplot(df, x="k", y="elpd_loo", hue="experiment", lw=0.5, ls="dashed", ax=ax)
        sns.scatterplot(df, x="k", y="elpd_loo", hue="experiment", s=10, alpha=0.5, ax=ax)

    fig.tight_layout(pad=0.5)
    if output_path is None:
        plt.show()
    else:
        fig.savefig(output_path)
        print(f"Wrote the plot to {output_path}")
    plt.close(fig)


# --- Entry point ---------------------------------------------------------------------

def main(
    results_dir: Path,
    burnin: float = 0.1,
    output_path: Path | None = None,
    csv_path: Path | None = None,
    assume_na_from_likelihood: bool = False,
) -> None:
    """Score all runs below `results_dir` with PSIS-LOO and plot the result.

    Finds the likelihood file of every run, computes its expected log pointwise
    predictive density, and plots the scores against the number of clusters, so that
    K values can be compared. Runs that cannot be evaluated are reported as warnings
    and left out; nothing is written if no run could be evaluated.

    Args:
        results_dir: directory containing the runs, searched recursively
        burnin: the fraction of samples to discard from each run, in [0, 1)
        output_path: where to save the plot. If None, the plot is shown instead.
        csv_path: where to write the scores as a CSV. If None, they are only printed.
        assume_na_from_likelihood: infer the NA mask for legacy files that store none
    """
    run_files = find_run_files(results_dir)
    if not run_files:
        warnings.warn(f"No likelihood files found in {results_dir}.")
        return

    df = collect_elpd(run_files, burnin, assume_na_from_likelihood)
    if df.empty:
        warnings.warn(
            f"None of the {len(run_files)} run(s) in {results_dir} could be evaluated."
        )
        return

    if csv_path is not None:
        df.to_csv(csv_path, index=False)
        print(f"Wrote the scores to {csv_path}")

    plot_elpd(df, output_path)


def cli() -> None:
    """Parse command-line arguments and score the runs."""
    parser = argparse.ArgumentParser(
        description="Compare sBayes runs by Bayesian cross validation (PSIS-LOO)."
    )
    parser.add_argument("results", type=Path,
                        help="Directory containing the sBayes results, searched recursively.")
    parser.add_argument("burnin", type=float, default=0.1, nargs="?",
                        help="Fraction of samples discarded as burn-in (default: 0.1).")
    parser.add_argument("--output", type=Path,
                        help="Save the plot to this file instead of showing it.")
    parser.add_argument("--csv", type=Path,
                        help="Also write the scores to this CSV file.")
    parser.add_argument("--assumeNaFromLikelihood", action="store_true",
                        help="For legacy files without an `na_values` array, treat "
                             "observations with a constant likelihood of 1.0 as missing.")
    args = parser.parse_args()

    if not 0 <= args.burnin < 1:
        parser.error("burnin must be in [0, 1).")
    if not args.results.is_dir():
        parser.error(f"Not a directory: {args.results}")

    main(
        results_dir=args.results,
        burnin=args.burnin,
        output_path=args.output,
        csv_path=args.csv,
        assume_na_from_likelihood=args.assumeNaFromLikelihood,
    )


if __name__ == "__main__":
    cli()