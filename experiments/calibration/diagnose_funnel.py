#!/usr/bin/env python3
"""Tell a funnel apart from two modes, for replicates that failed on a confounder effect.

The gate failures in the Poisson and Gaussian studies are concentrated in the
confounding effects, with a reproducible R-hat around 1.83 and an effective sample size
of 3. Two explanations fit those symptoms and call for different remedies:

* a funnel - the plausible region changes width along some direction, so one step size
  cannot serve all of it. The chains drift slowly through overlapping values. Careful
  stepping (a higher acceptance target, deeper trees) may rescue it.
* two modes - two self-consistent explanations of the data, with each chain settling in
  one. The chains sit at flat, separated levels. Careful stepping cannot help.

The trace distinguishes them, and the scatter against the component weight shows
whether the geometry really does narrow as the weight grows. The simulated values
printed alongside test the second hypothesis: that the group effect and the cluster
effects for that feature are close enough to be confusable.

Reads the table written by `diagnose_weights.py`, so it needs no second pass over the
whole study.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sbayes.simulate.calibration import (
    Study, flatten_parameters, load_replicate, stack_chains, CLUSTERS,
)
from sbayes.simulate.config import SimulateConfigFile

STUDY_PATH = Path("output/gaussian")
# STUDY_PATH =  Path("output/poisson")
CONFIG_PATH = Path("config_simulate.yaml")
N_REPLICATES = 4


def failing_on_a_confounder(study_path: Path, n: int) -> list[tuple[int, str]]:
    """Pick replicates that failed the gate on a confounding effect.

    Args:
        study_path: the study folder, holding `weight_diagnosis.tsv`
        n: how many replicates to return

    Returns:
        The replicate index and its hardest parameter, for up to `n` replicates.
    """
    table = pd.read_csv(study_path / "weight_diagnosis.tsv", sep="\t")
    failing = table[
        ~table["passed"]
        & table["strained_parameter"].str.startswith("confounding_effects")
    ]
    return list(
        failing[["replicate", "strained_parameter"]].head(n).itertuples(index=False, name=None)
    )


def inspect(study: Study, replicate: int, parameter: str, axes) -> dict:
    """Plot the chains of one strained parameter and report what surrounds it.

    Three diagnostics decide between the competing explanations:

    * the per-chain posterior density. Similar values mean two genuinely competing
      explanations of the data; a large gap means one chain is simply in a worse place
      and "multimodality" is the wrong word for it.
    * the per-chain weights for the feature. If the alternative explanation pays for a
      wrong effect by redistributing the weights, the chains must disagree about them.
    * the per-chain values of the neighbouring parameters - the other groups and the
      clusters for the same feature. This shows what the alternative explanation
      actually is, or that there isn't a coherent one.

    Args:
        study: the study the replicate belongs to
        replicate: the replicate index
        parameter: the expanded parameter name, e.g.
            `confounding_effects.social_group.group_A.f_39[0]`
        axes: two matplotlib axes, for the trace and the scatter

    Returns:
        The simulated values and the per-chain summaries, for the printed report.
    """
    folder = study.replicates[replicate]
    loaded = load_replicate(folder, study.config)

    # Chains kept separate and unthinned: the question is how each one moved
    draws = stack_chains(loaded.chains)
    truth = flatten_parameters(loaded.truth)

    base, _, index_text = parameter.partition("[")
    index = int(index_text.rstrip("]")) if index_text else 0
    _, confounder, group, feature = base.split(".")

    effect = draws[base][..., index]                       # (n_chains, n_draws)
    components = ["areal", *study.config.simulation.confounders]
    weights = draws[f"weights.{feature}"]                  # (n_chains, n_draws, n_comp)
    conf_weight = weights[..., components.index(confounder)]

    for chain, (values, chain_weights) in enumerate(zip(effect, conf_weight)):
        axes[0].plot(values, linewidth=0.8, label=f"run {chain}")
        axes[1].scatter(chain_weights, values, s=2, alpha=0.3, label=f"run {chain}")

    axes[0].set_title(f"replicate {replicate}: {group}, {feature}", fontsize=9)
    axes[0].set_xlabel("draw")
    axes[0].set_ylabel("effect")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel(f"weight on {confounder}")
    axes[1].set_ylabel("effect")

    group_names = loaded.truth.groups_by_confounders[confounder]
    cluster_names = loaded.truth.cluster_names

    # The simulated values around this parameter: if they are close, the data cannot
    # say which component produced the observations
    simulated_groups = {
        other: float(
            truth[f"confounding_effects.{confounder}.{other}.{feature}"].ravel()[index]
        )
        for other in group_names
    }
    simulated_clusters = {
        cluster: float(truth[f"areal_effect.{cluster}.{feature}"].ravel()[index])
        for cluster in cluster_names
    }

    # The same quantities per chain, so the alternative explanation becomes visible
    inferred_groups = {
        other: draws[f"confounding_effects.{confounder}.{other}.{feature}"][..., index]
        .mean(axis=1).tolist()
        for other in group_names
    }
    inferred_clusters = {
        cluster: draws[f"areal_effect.{cluster}.{feature}"][..., index]
        .mean(axis=1).tolist()
        for cluster in cluster_names
    }

    # `posterior` is absent in files written without `/derived/log_posterior`
    log_posteriors = [
        None if chain.posterior is None else float(np.mean(chain.posterior))
        for chain in loaded.chains
    ]

    features_csv = pd.read_csv(folder / "sim_data" / "features.csv")
    group_sizes = features_csv[confounder].value_counts().to_dict()


    in_group = (features_csv[confounder] == group).to_numpy(dtype=float)
    memberships = draws[CLUSTERS]  # (n_chains, n_draws, n_clusters, n_objects)
    cluster_group_overlap = {
        cluster: [
            float(np.corrcoef(memberships[chain, :, k].mean(axis=0), in_group)[0, 1])
            for chain in range(memberships.shape[0])
        ]
        for k, cluster in enumerate(cluster_names)
    }
    counts = features_csv[feature].to_numpy(dtype=float)
    cluster_count_overlap = {
        cluster: [
            float(np.corrcoef(memberships[chain, :, k].mean(axis=0), counts)[0, 1])
            for chain in range(memberships.shape[0])
        ]
        for k, cluster in enumerate(cluster_names)
    }

    return {
        "replicate": replicate,
        "parameter": parameter,
        "group": group,
        "feature": feature,
        "group_sizes": group_sizes,

        # What was simulated
        "simulated_effect": simulated_groups[group],
        "simulated_groups": simulated_groups,
        "simulated_clusters": simulated_clusters,
        "simulated_weights": dict(
            zip(components, truth[f"weights.{feature}"].ravel().tolist())
        ),

        # Where each chain went
        "chain_means": effect.mean(axis=1).tolist(),
        "chain_sds": effect.std(axis=1).tolist(),
        # Two separated blobs give about 2.0; overlapping chains give well under 1
        "chain_separation": float(
            abs(effect[0].mean() - effect[1].mean()) / max(effect.std(), 1e-12)
        ),

        # Are both regions equally good explanations?
        "chain_log_posteriors": log_posteriors,
        # Do the chains pay for a different effect with different weights?
        "chain_cluster_group_overlap": cluster_group_overlap,
        "cluster_count_overlap": cluster_count_overlap,
        "chain_weights": {
            component: weights[..., i].mean(axis=1).tolist()
            for i, component in enumerate(components)
        },
        # What does the alternative explanation look like?
        "chain_groups": inferred_groups,
        "chain_clusters": inferred_clusters,
    }


def main() -> None:
    config = SimulateConfigFile.from_config_file(CONFIG_PATH)
    config.simulation.features = {"gaussian": 40}
    # config.simulation.features = {"poisson": 40}

    study = Study.from_path(STUDY_PATH, config)
    selected = failing_on_a_confounder(STUDY_PATH, N_REPLICATES)
    if not selected:
        raise SystemExit("No failing replicate was strained on a confounding effect.")

    figure, axes = plt.subplots(
        len(selected), 2, figsize=(9, 3 * len(selected)), squeeze=False
    )

    summaries = []
    for row, (replicate, parameter) in enumerate(selected):
        summaries.append(inspect(study, replicate, parameter, axes[row]))

    figure.tight_layout()
    figure.savefig(STUDY_PATH / "chain_shapes.png", dpi=150)

    for summary in summaries:
        print()
        for key, value in summary.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()