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
    echo "[INFO] root user detected → zsh config will be applied to /root"
    HOME_DIR="/root"
else
    HOME_DIR="/home/$USER_NAME"
fi

echo "[INFO] installing oh-my-zsh for $USER_NAME at $HOME_DIR"
RUN_AS="sudo -u $USER_NAME"
[[ "$USER_NAME" == "root" ]] && RUN_AS=""

$RUN_AS sh -c "curl -LsSf https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | sh || true"

# Install powerlevel10k theme
OHMYZSH_THEME="$HOME_DIR/.oh-my-zsh/custom/themes"
mkdir -p "$OHMYZSH_THEME"
$RUN_AS git clone --depth=1 https://github.com/romkatv/powerlevel10k.git "$OHMYZSH_THEME/powerlevel10k" || true

ZSHRC="$HOME_DIR/.zshrc"
if [[ -f "$ZSHRC" ]]; then
    sed -i 's|^ZSH_THEME=.*|ZSH_THEME="powerlevel10k/powerlevel10k"|' "$ZSHRC"
    echo 'POWERLEVEL9K_DISABLE_CONFIGURATION_WIZARD=true' >> "$ZSHRC"
    echo 'alias ls="ls -lsaF"' >> "$ZSHRC"
fi

chown -R "$USER_NAME":"$USER_NAME" "$HOME_DIR/.oh-my-zsh" "$HOME_DIR/.zshrc" || true
chsh -s "$(which zsh)" "$USER_NAME" || true
