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

import h5py

PathLike = Path | str
"""Convenience type for cases where `str` or `Path` are acceptable types."""


def read_likelihood_for_az(likelihood_path: PathLike, burnin: float) -> az.InferenceData:
    """Read pointwise likelihoods from a samples h5 file (under /derived group)
    or from a legacy separate likelihood h5 file."""
    likelihood_table = tables.open_file(likelihood_path, mode='r')

    # New format: likelihood stored under /derived group in samples_?.h5
    if hasattr(likelihood_table.root, 'derived') and hasattr(likelihood_table.root.derived, 'likelihood'):
        likelihood_np = likelihood_table.root.derived.likelihood[:]
        if hasattr(likelihood_table.root.derived, 'na_values'):
            is_na = likelihood_table.root.derived.na_values[:]
        else:
            warnings.warn(f"No `na_values` array found in `{likelihood_path}`. "
                          f"Assuming all observations with a constant likelihood of 1.0 to be NAs.")
            is_na = np.all(np.isclose(likelihood_np, 1), axis=0)
    # Legacy format: likelihood at root level in separate likelihood_K?_?.h5
    elif hasattr(likelihood_table.root, 'likelihood'):
        likelihood_np = likelihood_table.root.likelihood[:]
        if hasattr(likelihood_table.root, 'na_values'):
            is_na = likelihood_table.root.na_values[:]
        else:
            warnings.warn(f"No `na_values` array found in `{likelihood_path}`. "
                          f"Assuming all observations with a constant likelihood of 1.0 to be NAs.")
            is_na = np.all(np.isclose(likelihood_np, 1), axis=0)
    else:
        likelihood_table.close()
        raise ValueError(f"No likelihood data found in `{likelihood_path}`.")

    likelihood_table.close()

    # drop NA values
    likelihood_np = likelihood_np[:, ~is_na]

    # drop burn-in
    burnin_int = int(burnin * len(likelihood_np))
    likelihood_np = likelihood_np[burnin_int:, :]

    l = np.exp(likelihood_np)
    print("MIN LH", np.min(l))
    print("MAX LH", np.max(l))

    # arviz interprets the first dimension as chains and the second as samples, but the
    # likelihood in the file is only for one chain, i.e. dimensions start with samples.
    # => Append a new dimension for chains!
    likelihood_np = likelihood_np[np.newaxis, ...]

    # Create an InferenceData object
    return az.convert_to_inference_data((likelihood_np))


def sbayes_psis_loo(likelihood_path: Path, burnin: float) -> float:
    """Load likelihood arrays for a sBayes run and evaluate the PSIS_LOO."""
    data = read_likelihood_for_az(likelihood_path, burnin)

    # Per default, the data is stored in the group "posterior", but az.loo() expects InferenceData with a "log_likelihood" group.
    # We can manually add that group (as a copy of the posterior):
    data.add_groups({'log_likelihood': data.posterior})

    # Now az.loo() should work:
    loo = az.loo(data, pointwise=True)

    print(loo)

    # pareto_k = loo.pareto_k.to_numpy()
    # pareto_k = np.sort(pareto_k)
    # print(pareto_k[-10:])  # Print the last 10 Pareto k values (should be < 0.5)
    # print(loo.p_loo)  # Print the last 10 Pareto k values (should be < 0.5)

    # waic = az.waic(data)
    return loo.elpd_loo


def main(results_dir: Path, burnin: float = 0.1):
    # if __debug__:
    #     activate_verbose_warnings()

    df = pd.DataFrame(columns=["experiment", "k", "run", "elpd_loo"])\
           .set_index(["experiment", "k", "run"])

    # Find samples h5 files (new format) and legacy likelihood files
    h5_paths = list(results_dir.rglob("samples_*.h5")) + list(results_dir.rglob("likelihood_K*_*.h5"))
    for run_path in h5_paths:
        *head, experiment, k_folder, file_name = run_path.parts

        if ".chain" in file_name:
            # Skip results of hot chains (for MC3 results)
            continue

        # Parse the run index and k (number of areas)
        run_id = int(run_path.stem.rpartition("_")[-1])
        k = int(k_folder[1:])

        # CODE FOR PREPARING FILES THAT ARE PARSEABLE BY RHDF5
        # with tables.open_file(run_path, 'r') as f_old, h5py.File(run_path.with_suffix('.compat.h5'), 'w') as f_new:
        #     f_new.create_dataset('likelihood', data=f_old.root.likelihood[:])
        #     if "na_values" in f_old.root:
        #         f_new.create_dataset('na_values', data=f_old.root.na_values[:])
        # continue

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
        sn.scatterplot(df, x="k", y="elpd_loo", hue="experiment", s=10, alpha=0.5)


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
