#!/usr/bin/env bash
set -euo pipefail

# usage
usage() {
    echo "Usage: $0 <group> <workdir> <home>" >&2
    exit 1
}

# args: <group> <workdir>
if [[ $# -lt 3 ]]; then
    usage
fi

# args
CONTAINER_UV_GROUP="${1}"
CONTAINER_WORK_DIR="${2}"
CONTAINER_HOME="${3}"
CONTAINER_VENV_DIR="${CONTAINER_WORK_DIR}/.venv"

if [[ ! -d "${CONTAINER_HOME}" ]]; then
    echo "[ERROR] home directory '${CONTAINER_HOME}' does not exist" >&2
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "[ERROR] uv is not installed" >&2
    exit 1
fi

cd "${CONTAINER_WORK_DIR}"

echo "[info] uv sync"
if uv sync --group dev; then
    echo "[info] uv sync succeeded."
else
    echo "[info] uv sync failed."
    exit 1
fi

echo "[INFO] Done setup uv."

