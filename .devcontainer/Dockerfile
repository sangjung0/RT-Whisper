FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG CONTAINER_USER
ARG CONTAINER_UID
ARG CONTAINER_HOME
ARG CONTAINER_WORK_DIR
ARG CONTAINER_UV_GROUP

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    git \
    sudo \
    zsh \
    curl \
    ffmpeg \
    sctk \
    python3 \
    python3-pip \
    autoconf \
    automake \
    build-essential \
    libtool \
    && rm -rf /var/lib/apt/lists/*

# LOAD SCRIPT
COPY .devcontainer/scripts /tmp/scripts
RUN chmod +x /tmp/scripts/*.sh

# ADD USER AND SETUP ZSH
RUN /tmp/scripts/add_user_and_setup_sudo.sh ${CONTAINER_USER} ${CONTAINER_UID}
RUN /tmp/scripts/setup_zsh.sh ${CONTAINER_USER} ${CONTAINER_UID}

RUN sudo -u ${CONTAINER_USER} curl -LsSf https://astral.sh/uv/install.sh | sudo -u ${CONTAINER_USER} sh
RUN sudo -u ${CONTAINER_USER} -H env PATH="${CONTAINER_HOME}/.local/bin:$PATH"

WORKDIR ${CONTAINER_WORK_DIR}
COPY pyproject.toml uv.lock ${CONTAINER_WORK_DIR}/

RUN install -d -o ${CONTAINER_UID} -g ${CONTAINER_UID} ${CONTAINER_WORK_DIR}/.venv ${CONTAINER_HOME}

RUN chown -R ${CONTAINER_UID}:${CONTAINER_UID} ${CONTAINER_WORK_DIR} ${CONTAINER_HOME}
RUN sudo -u ${CONTAINER_USER} -H bash -lc "/tmp/scripts/setup_uv.sh ${CONTAINER_UV_GROUP} ${CONTAINER_WORK_DIR} ${CONTAINER_HOME}"
RUN sudo -u ${CONTAINER_USER} -H bash -lc "/tmp/scripts/setup_mecab_kor.sh"

RUN rm -f ${CONTAINER_WORK_DIR}/pyproject.toml ${CONTAINER_WORK_DIR}/uv.lock /tmp/* || true

USER ${CONTAINER_USER}
ENV VIRTUAL_ENV="${CONTAINER_WORK_DIR}/.venv"
ENV PATH="${CONTAINER_WORK_DIR}/.venv/bin:${PATH}"

