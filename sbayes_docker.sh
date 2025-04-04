#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 path/to/config.yaml [additional sbayes CLI arguments...]"
  exit 1
fi

CONFIG_PATH="$(realpath "$1")"
CONFIG_DIR="$(dirname "$CONFIG_PATH")"
shift  # Remove the config path from the arguments
CONFIG_FILE="$(basename "$CONFIG_PATH")"

docker run --rm -v "$CONFIG_DIR":/data sbayes /data/"$CONFIG_FILE" "$@"
