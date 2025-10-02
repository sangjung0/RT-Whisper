#!/usr/bin/env bash
set -euo pipefail

CONTAINER_USER="${1:-$CONTAINER_USER}"
CONTAINER_HOME="${2:-$CONTAINER_HOME}"
CONTAINER_WORK_DIR="${3:-$CONTAINER_WORK_DIR}"

if ! id "${CONTAINER_USER}" &>/dev/null; then
    echo "User ${CONTAINER_USER} does not exist" >&2
    exit 1
elif [[ ! -d "${CONTAINER_HOME}" ]]; then
    echo "Home directory ${CONTAINER_HOME} does not exist" >&2
    exit 1
elif [[ ! -d "${CONTAINER_WORK_DIR}" ]]; then
    echo "Work directory ${CONTAINER_WORK_DIR} does not exist" >&2
    exit 1
fi


echo "[INFO] step 1/3: change ownership to ${CONTAINER_USER}"
bash "${CONTAINER_WORK_DIR}/.devcontainer/scripts/change_owner.sh" "${CONTAINER_USER}" \
    --target "${CONTAINER_HOME}" \
    --target "${CONTAINER_WORK_DIR}:${CONTAINER_WORK_DIR}/.datasets" \
    --target "${CONTAINER_WORK_DIR}/.datasets:${CONTAINER_WORK_DIR}/.datasets/pills:${CONTAINER_WORK_DIR}/.datasets/ILSVRC:${CONTAINER_WORK_DIR}/.datasets/asr-rankformer-datasets"

echo "[INFO] step 2/3: sync uv"
cd "${CONTAINER_WORK_DIR}" && uv sync --group dev

echo "[INFO] step 3/3: setup lhotse"
bash "${CONTAINER_WORK_DIR}/.devcontainer/scripts/setup_lhotse.sh" "${CONTAINER_WORK_DIR}"

