"""Functions for reading h5 files shared by tools"""
import re

import tables
import numpy as np

from pathlib import Path

def read_array(h5_file: tables.File, path: str) -> np.ndarray:
    """Read the array stored at `path`.

    Args:
        h5_file: an open h5 file
        path: the absolute node path, e.g. `/likelihood`

    Returns:
        The array contents.

    Raises:
        ValueError: if the node is not an array.
    """
    node = h5_file.get_node(path)
    if not isinstance(node, tables.Array):
        raise ValueError(f"Expected an array at {path}, found {type(node).__name__}.")
    return node.read()


def parse_run_location(samples_h5_path: Path) -> tuple[int, int]:
    """Read the number of clusters and the run index from a samples file path.

    Legacy results are stored as `.../K{k}/samples_{run}.h5`, where some versions
    repeat the number of clusters in the file name (`samples_K{k}_{run}.h5`). Both
    forms are accepted. The values are needed to find the companion stats and
    likelihood files.

    Args:
        samples_h5_path: the path of a legacy samples file

    Returns:
        The number of clusters `k` and the run index.

    Raises:
        ValueError: if the path matches neither form.
    """
    k_match = re.fullmatch(r"K(\d+)", samples_h5_path.parent.name)
    run_match = re.fullmatch(r"(?:samples|likelihood)_(?:K\d+_)?(\d+)", samples_h5_path.stem)
    if k_match is None or run_match is None:
        raise ValueError(
            f"Expected a path of the form .../K<k>/samples_<run>.h5, got {samples_h5_path}"
        )
    return int(k_match.group(1)), int(run_match.group(1))

