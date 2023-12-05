"""
Compute the PSIS-LOO score for logged observation likelihood values stored in a likelihood.h5 file.
"""
import warnings
from pathlib import Path

import tables
import numpy as np
import arviz as az
import argparse
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sbayes.util import activate_verbose_warnings


PathLike = Path | str
"""Convenience type for cases where `str` or `Path` are acceptable types."""


def read_likelihood_for_az(likelihood_path: PathLike, burnin: float) -> az.InferenceData:
    # Load the data into a numpy array (shape = n_samples*n_observations)
    likelihood_table = tables.open_file(likelihood_path, mode='r')
    likelihood_np = likelihood_table.root.likelihood[:]
    if "na_values" in likelihood_table.root:
        is_na = likelihood_table.root.na_values[:]
    else:
        warnings.warn(f"No `na_values` array found in the likelihood file `{likelihood_path}`. "
                      f"Assuming all observations with a constant likelihood of 1.0 to be NAs.")
        is_na = np.all(np.isclose(likelihood_np, 1), axis=0)
    likelihood_table.close()

    # drop NA values
    likelihood_np = likelihood_np[:, ~is_na]

    # drop burn-in
    burnin_int = int(burnin * len(likelihood_np))
    likelihood_np = likelihood_np[burnin_int:, :]

    # arviz interprets the first dimension as chains and the second as samples, but the
    # likelihood in the file is only for one chain, i.e. dimensions start with samples.
    # => Append a new dimension for chains!
    likelihood_np = likelihood_np[np.newaxis, ...]

    # Create an InferenceData object
    return az.convert_to_inference_data(np.log(likelihood_np))


def sbayes_psis_loo(likelihood_path: Path, burnin: float) -> float:
    """Load likelihood arrays for a sBayes run and evaluate the PSIS_LOO."""
    data = read_likelihood_for_az(likelihood_path, burnin)

    # Per default, the data is stored in the group "posterior", but az.loo() expects InferenceData with a "log_likelihood" group.
    # We can manually add that group (as a copy of the posterior):
    data.add_groups({'log_likelihood': data.posterior})

    # Now az.loo() should work:
    loo = az.loo(data)
    # waic = az.waic(data)
    return loo.elpd_loo


def main(results_dir: Path, burnin: float = 0.1):
    if __debug__:
        activate_verbose_warnings()

    df = pd.DataFrame(columns=["experiment", "k", "run", "elpd_loo"])\
           .set_index(["experiment", "k", "run"])

    for run_path in results_dir.rglob("likelihood_K*_*.h5"):
        *head, experiment, k_folder, file_name = run_path.parts

        if ".chain" in file_name:
            # Skip results of hot chains (for MC3 results)
            continue

        # Parse the run index and k (number of areas)
        run_id = int(run_path.stem.rpartition("_")[-1])
        k = int(k_folder[1:])

        try:
            loo = sbayes_psis_loo(run_path, burnin)
            print("ELPD-LOO for", (experiment, k, run_id), ":", loo)
            df.loc[(experiment, k, run_id)] = [loo]
        except Exception as e:
            msg = f"Error in likelihood file '{run_path}'. Will be skipped in model comparison."
            msg += "".join(["\n\t| " + l for l in str(e).split("\n")])
            warnings.warn(msg)

    if len(df) == 0:
        warnings.warn(f"No results with valid likelihood files were found in directory '{results_dir}'.")
        return

    df = df.reset_index()
    if len(df.k.unique()) == 1:
        sn.boxplot(df, x="experiment", y="elpd_loo")
    else:
        sn.lineplot(df, x="k", y="elpd_loo", hue="experiment", lw=0.5, ls="dashed")

    plt.tight_layout(pad=0.5)
    plt.show()


def cli():
    """Read the results directory as a command line argument and pass it to the main function."""
    parser = argparse.ArgumentParser(description="Bayesian cross validation of sBayes runs using PSIS-LOO.")
    parser.add_argument("results", type=Path, help="The path to a directory with sBayes likelihood files.")
    parser.add_argument("burnin", type=float, default=0.1, nargs="?",
                        help="Fraction of samples that are discarded as burn-in.")
    args = parser.parse_args()
    return main(args.results, args.burnin)


if __name__ == '__main__':
    cli()
