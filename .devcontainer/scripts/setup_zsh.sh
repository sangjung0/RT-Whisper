#!/usr/bin/env bash
set -euo pipefail

# args: <username> <uid>
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <username> <uid>" >&2
    exit 1
fi

CONTAINER_USER="${1:-$CONTAINER_USER}"
CONTAINER_UID="${2:-$CONTAINER_UID}"
CONTAINER_HOME="${3:-$CONTAINER_HOME}"

if [[ -z "${CONTAINER_USER}" ]]; then
    echo "[ERROR] CONTAINER_USER is not set" >&2
    exit 1
elif [[ -z "${CONTAINER_UID}" ]]; then
    echo "[ERROR] CONTAINER_UID is not set" >&2
    exit 1
elif [[ -z "${CONTAINER_HOME}" ]]; then
    echo "[ERROR] CONTAINER_HOME is not set" >&2
    exit 1
elif [[ "${CONTAINER_USER}" == "root" && "${CONTAINER_UID}" == "0" && "${CONTAINER_HOME}" != "/root" ]]; then
    echo "[ERROR] HOME is not /root for root user: ${CONTAINER_HOME}" >&2
    exit 1
elif [[ "${CONTAINER_USER}" != "root" && "${CONTAINER_UID}" == "0" ]]; then
    echo "[ERROR] Non-root user cannot have UID 0" >&2
    exit 1
elif [[ "${CONTAINER_HOME}" != "/home/${CONTAINER_USER}" && "${CONTAINER_USER}" != "root" ]]; then
    echo "[ERROR] HOME is not /home/${CONTAINER_USER} for non-root user: ${CONTAINER_HOME}" >&2 exit 1
fi

echo "[INFO] installing oh-my-zsh for ${CONTAINER_USER} at ${CONTAINER_HOME}"
RUN_AS="sudo -u ${CONTAINER_USER}"
[[ "${CONTAINER_USER}" == "root" ]] && RUN_AS=""

$RUN_AS sh -c "curl -LsSf https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | sh || true"

# Install powerlevel10k theme
OHMYZSH_THEME="${CONTAINER_HOME}/.oh-my-zsh/custom/themes"
mkdir -p "${OHMYZSH_THEME}"
$RUN_AS git clone --depth=1 https://github.com/romkatv/powerlevel10k.git "${OHMYZSH_THEME}/powerlevel10k" || true

ZSHRC="${CONTAINER_HOME}/.zshrc"
if [[ -f "${ZSHRC}" ]]; then
    sed -i 's|^ZSH_THEME=.*|ZSH_THEME="powerlevel10k/powerlevel10k"|' "${ZSHRC}"
    echo 'POWERLEVEL9K_DISABLE_CONFIGURATION_WIZARD=true' >> "${ZSHRC}"
    echo 'alias ls="ls -lsaF"' >> "${ZSHRC}"
fi

chown -R "${CONTAINER_USER}":"${CONTAINER_USER}" "${CONTAINER_HOME}/.oh-my-zsh" "${CONTAINER_HOME}/.zshrc" || true
chsh -s "$(which zsh)" "${CONTAINER_USER}" || true
