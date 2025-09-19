FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV USER="root"
ENV USER_UID=0

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    git \
    sudo \
    zsh \
    curl \
    ffmpeg \
    sctk \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY .devcontainer/scripts /tmp/scripts
RUN chmod +x /tmp/scripts/*.sh

RUN /tmp/scripts/add_user_and_setup_sudo.sh ${USER} ${USER_UID}
RUN /tmp/scripts/setup_zsh.sh ${USER} ${USER_UID}

RUN rm -rf /tmp/scripts
