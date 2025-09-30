#!/usr/bin/env bash
set -euo pipefail

# usage
usage() {
    echo "Usage: $0 <username> <uid>" >&2
    exit 1
}

# args: <username> <uid>
if [[ $# -lt 2 ]]; then
    usage
fi

CONTAINER_USER="${1}"
CONTAINER_UID="${2}"

if [[ "${CONTAINER_USER}" != "root" && "${CONTAINER_UID}" == "0" ]]; then
    echo "[ERROR] Non-root user cannot have UID 0" >&2
    exit 1
fi

if [[ "${CONTAINER_UID}" == "0" ]]; then
    echo "[INFO] root user detected → skip useradd/sudo setup"
    exit 0
fi

if id "${CONTAINER_USER}" &>/dev/null; then
    echo "[INFO] user '${CONTAINER_USER}' already exists, skipping creation."
else
    echo "[INFO] creating user '${CONTAINER_USER}' with UID ${CONTAINER_UID}"
    useradd -u "${CONTAINER_UID}" -m -s /usr/bin/zsh "${CONTAINER_USER}"
fi

echo "${CONTAINER_USER} ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/${CONTAINER_USER}"
chmod 0440 "/etc/sudoers.d/${CONTAINER_USER}"

echo "[INFO] user '${CONTAINER_USER}' configured with sudo rights."
