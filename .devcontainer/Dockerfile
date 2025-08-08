FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ARG USER=vscode
ARG USER_UID=1001

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    git \
    sudo \
    zsh \
    curl \
    ffmpeg \
    sctk \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -u ${USER_UID} -m -s /usr/bin/zsh ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

USER ${USER}
WORKDIR /home/${USER}

RUN curl -LsSf https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | sh
ARG OHMYZSH_THEME=/home/${USER}/.oh-my-zsh/custom/themes

RUN <<EOT
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${OHMYZSH_THEME}/powerlevel10k
    sed -i 's|ZSH_THEME="devcontainers"|ZSH_THEME="powerlevel10k/powerlevel10k"|g' .zshrc
    echo 'POWERLEVEL9K_DISABLE_CONFIGURATION_WIZARD=true' >> .zshrc
EOT

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN echo '. $HOME/.local/bin/env' >> .zshrc

ENV PATH="${PATH}:/usr/lib/sctk/bin"
