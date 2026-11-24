#!/usr/bin/env python3
"""Test whether gate failures coincide with near-zero component weights.

The hypothesis: when a feature's weight on a component is near zero, that component's
effect for that feature is barely constrained by the data. For Poisson and Gaussian
features the effect is unbounded, so the unconstrained region is large and the sampler
stalls in it.

If the hypothesis holds, the replicates that failed the gate should have systematically
more extreme simulated weights than those that passed.

"""
from pathlib import Path

import numpy as np
import pandas as pd

from sbayes.simulate.calibration import (
    Study, gate_replicates, prepare_replicates,
)
from sbayes.simulate.config import SimulateConfigFile

STUDY_PATH = Path("output/poisson")
CONFIG_PATH = Path("config_simulate.yaml")


def component_of(parameter: str, confounders: list[str]) -> tuple[str, int] | None:
    """Which feature and weight component a strained parameter belongs to.

    The weights run over components in the order `['areal', *confounders]`, matching
    how `write_samples` builds them.

    Args:
        parameter: an expanded parameter name, e.g.
            `confounding_effects.social_group.group_A.f_39[0]`
        confounders: the confounder names, in the order they index the weights

    Returns:
        The feature name and the component index, or None for statistics that belong
        to no single feature.
    """
    name = parameter.split("[")[0]
    parts = name.split(".")
    kind = parts[0]

    if kind == "weights":
        # The failing component is the index in the name itself
        index = int(parameter.split("[")[1].rstrip("]")) if "[" in parameter else 0
        return parts[1], index

    if kind == "areal_effect":
        return parts[2], 0

    if kind == "confounding_effects":
        conf = parts[1]
        return parts[3], 1 + confounders.index(conf)

    return None


def main() -> None:
    config = SimulateConfigFile.from_config_file(CONFIG_PATH)
    # Apply the same variant overrides the study was run with
    config.simulation.features = {"poisson": 40}

    study = Study.from_path(STUDY_PATH, config)
    prepared = prepare_replicates(study)
    gate = gate_replicates({i: p.diagnostics for i, p in prepared.items()})
    passed = set(gate.passed)

    confounders = list(config.simulation.confounders)

    rows = []
    for replicate, one in sorted(prepared.items()):
        # The most strained parameter, whether or not it failed, so passing and failing
        # replicates are compared on the same footing
        strained = one.diagnostics.worst_r_hat_parameter
        placed = component_of(strained, confounders)

        weights = {
            name: values for name, values in one.truth.items()
            if name.startswith("weights.")
        }
        all_weights = np.concatenate([v.ravel() for v in weights.values()])

        rows.append({
            "replicate": replicate,
            "passed": replicate in passed,
            "max_r_hat": one.diagnostics.max_r_hat,
            "min_ess": one.diagnostics.min_ess,
            "strained_parameter": strained,
            "strained_feature": placed[0] if placed else "",
            "strained_component": placed[1] if placed else -1,
            # The simulated weight of the component the strained parameter belongs to
            "weight_at_strained": (
                float(weights[f"weights.{placed[0]}"].ravel()[placed[1]])
                if placed else np.nan
            ),
            # The most extreme weight anywhere in this replicate, which needs no
            # parameter-name parsing and is what the hypothesis really predicts
            "min_weight": float(all_weights.min()),
        })

    table = pd.DataFrame(rows)
    table.to_csv(STUDY_PATH / "weight_diagnosis.tsv", sep="\t", index=False)

    print(table.groupby("passed")[["weight_at_strained", "min_weight"]].describe().T)
    print()
    print("Strained parameter families:")
    print(table.groupby("passed")["strained_parameter"]
          .apply(lambda s: s.str.split(".").str[0].value_counts()))


if __name__ == "__main__":
    main()