#!/usr/bin/env bash
set -euo pipefail

# args: <username> <uid>
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <username> <uid>" >&2
    exit 1
fi

USER_NAME="$1"
USER_UID="$2"

if [[ "$USER_NAME" == "root" || "$USER_UID" == "0" ]]; then
    echo "[INFO] root user detected → skip useradd/sudo setup"
    exit 0
fi

if id "$USER_NAME" &>/dev/null; then
    echo "[INFO] user '$USER_NAME' already exists, skipping creation."
else
    echo "[INFO] creating user '$USER_NAME' with UID $USER_UID"
    useradd -u "$USER_UID" -m -s /usr/bin/zsh "$USER_NAME"
fi

echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/$USER_NAME"
chmod 0440 "/etc/sudoers.d/$USER_NAME"

echo "[INFO] user '$USER_NAME' configured with sudo rights."
