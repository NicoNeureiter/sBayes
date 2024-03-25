import sys
from pathlib import Path


def main(paths: list[Path], interval: int) -> None:
    for path in paths:
        with open(path, "r") as in_file, \
                open(path[:-4] + "_subsampled.txt", "w") as out_file:
            lines = in_file.readlines()

            if Path(path).name.startswith("stats_"):
                header_line = lines.pop(0)
                out_file.write(header_line)

            for i, line in enumerate(lines):
                if i % interval == 0:
                    out_file.write(line)


def cli():
    """Read the results directory as a command line argument and pass it to the main function."""
    import argparse
    parser = argparse.ArgumentParser(description="Subsample sBayes results.")
    parser.add_argument("-f", "--files", nargs="*", type=Path,
                        help="The sBayes results files (stats_*.txt or clusters_*.txt).")
    parser.add_argument("interval", type=int, default=2,
                        help="Interval at which the results are subsampled.")
    args = parser.parse_args()
    return main(args.results, args.interval)


if __name__ == '__main__':
    main()
