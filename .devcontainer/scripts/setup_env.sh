#!/usr/bin/env bash
set -euo pipefail

# usage
usage() {
    echo "Usage: $0 <env-file>" >&2
    exit 1
}

# args: <env-file>
if [[ $# -lt 1 ]]; then
    usage
fi

ENV_FILE="${1}"

if [ ! -f "$ENV_FILE" ]; then
    echo "[ERROR] '$ENV_FILE' not found" >&2
    return 1
fi

set -a
. "$ENV_FILE"
set +a

echo "[INFO] Completed loading environment variables from '$ENV_FILE'"
