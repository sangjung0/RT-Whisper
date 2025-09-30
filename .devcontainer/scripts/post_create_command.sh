#!/usr/bin/env bash
set -euo pipefail

USERNAME="${1:-$USER}"

if ! id "$USERNAME" &>/dev/null; then
    echo "User $USERNAME does not exist" >&2
    exit 1
fi

# UID=$(id -u "$USERNAME")
HOME=$(getent passwd "$USERNAME" | cut -d: -f6)

echo "[INFO] step 1/2: change ownership to $USERNAME"
bash /workspaces/dev/.devcontainer/scripts/change_owner.sh "$USERNAME"\
    --target "$HOME" \
    --target /workspaces/dev:/workspaces/dev/.datasets \
    --target /workspaces/dev/.datasets/asr-rankformer-datasets \
    --target /workspaces/dev/.datasets/ami \
    --target /workspaces/dev/.datasets/vox_populi \
    --target /workspaces/dev/.datasets/tedlium \
    --target /workspaces/dev/.datasets/libri_speech

# echo "[INFO] step 2/3: setup uv for $USERNAME"
# bash /workspaces/dev/.devcontainer/scripts/setup_uv.sh
echo "[INFO] step 2/2: setup lhotse"
bash /workspaces/dev/.devcontainer/scripts/setup_lhotse.sh

