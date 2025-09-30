#!/usr/bin/env bash
set -euo pipefail

# usage
usage() {
    echo "Usage: $0 <workdir>" >&2
}

# args: <workdir>
if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

CONTAINER_WORK_DIR="$1"

# lhotse
source "${CONTAINER_WORK_DIR}/.venv/bin/activate"
lhotse install-sph2pipe # for tedlium dataset
