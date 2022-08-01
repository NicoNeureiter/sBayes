import numpy as np
from pathlib import Path
import json
import argparse

import pandas as pd


def main(args):
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Tool to convert dirichlet prior parameters from CSV to JSON.")
    parser.add_argument("--csv", type=Path, required=True, help="The input CSV file")
    parser.add_argument("--output", type=Path, required=True, help="The output JSON file")
    args = parser.parse_args(args)
    csv_path = args.csv
    output_path = args.output

    # Load counts from CSV file into a pandas data-frame
    counts_df = pd.read_csv(csv_path, index_col='feature')

    # Convert data-frame into a nested dictionary
    counts_dict = {}
    for feature, row in counts_df.iterrows():
        row_dict = {
            k: v for k, v in row.to_dict().items()
            if not np.isnan(v)
        }
        counts_dict[feature] = row_dict

    # Write nested dictionary to a JSON file
    with open(output_path, 'w') as json_file:
        json.dump(counts_dict, json_file, indent=4)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
