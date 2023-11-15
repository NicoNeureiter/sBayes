import sys
from pathlib import Path


def main():
    print(sys.argv)

    paths = sys.argv[1:-1]
    interval = int(sys.argv[-1])

    for path in paths:
        # path = Path(path)
        with open(path, "r") as in_file, \
                open(path[:-4] + "_subsampled.txt", "w") as out_file:
            lines = in_file.readlines()

            if Path(path).name.startswith("stats_"):
                header_line = lines.pop(0)
                out_file.write(header_line)

            for i, line in enumerate(lines):
                if i % interval == 0:
                    out_file.write(line)


if __name__ == '__main__':
    main()