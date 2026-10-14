import numpy as np
import pandas as pd

from pathlib import Path
from sbayes.load_data import Data, CategoricalFeatures, GenericTypeFeatures
from sbayes.model import Model
from sbayes.sampling.loggers import write_samples
from typing import Sequence


def write_simulated_data(
    partitions: Sequence[GenericTypeFeatures],
    sample: dict[str, np.ndarray],
    features_csv: pd.DataFrame,
    base_path: Path,
) -> None:
    """Write one simulated data set to `features.csv`.

    The feature columns of `features_csv` are filled with the simulated values;
    categorical values are converted from state indices back to state names.

    Args:
        partitions: the feature partitions of the model, which define the column names
            and, for categorical partitions, the state names
        sample: the prior sample of one data set, keyed by site name. The arrays keep
            their leading sample axis of length 1.
        features_csv: the metadata columns (id, name, x, y, confounders) with empty
            feature columns. Modified in place.
        base_path: the folder to write `features.csv` to
    """
    for partition in partitions:
        # Index 0: the data set was sliced earlier, keeping the sample axis
        simulated = sample[f"x_{partition.name}"][0]

        if isinstance(partition, CategoricalFeatures):
            column_indices = np.arange(simulated.shape[1])[None, :]
            simulated = partition.state_names[column_indices, simulated]

        features_csv[partition.names] = features_csv[partition.names].astype(str)
        features_csv.update(pd.DataFrame(simulated, columns=partition.names))

    features_csv.to_csv(base_path / "features.csv", index=False)


def write_simulated_parameters(
    sample: dict[str, np.ndarray],
    data: Data,
    model: Model,
    base_path: Path,
) -> None:
    """Write the parameters of one simulated data set to file.

    Uses the same writer as the sampler, so the simulated ground truth is stored in
    the same format as inferred samples and the two can be compared directly.

    Args:
        sample: the prior sample of one data set, keyed by site name. The arrays keep
            their leading sample axis of length 1, which the writer reads as one sample.
        data: the data the model was built on
        model: the model the sample was drawn from
        base_path: the folder to write the parameter files to
    """
    write_samples(
        run=0,
        base_path=base_path,
        samples=sample,
        data=data,
        model=model,
        match_clusters=False,  # a prior sample has nothing to match against
    )


def prepare_output_folders(
    results_path: Path, sample_index: int | None = None
) -> tuple[Path, Path]:
    """Create the output folders of one simulated data set.

    Several data sets are kept apart in `sim_{index}` folders. Within each, the
    simulated parameters and data go into `sim_params` and `sim_data`, so the layout
    is the same whether one or both are written.

    Args:
        results_path: the base output directory
        sample_index: the index of the data set, or None for a single data set

    Returns:
        The parameters folder and the data folder.
    """
    sim_folder = results_path if sample_index is None else results_path / f"sim_{sample_index}"
    params_folder = sim_folder / "sim_params"
    data_folder = sim_folder / "sim_data"
    params_folder.mkdir(parents=True, exist_ok=True)
    data_folder.mkdir(parents=True, exist_ok=True)
    return params_folder, data_folder

