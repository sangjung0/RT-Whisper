FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 AS base

ARG USER="ksj"
ARG UID=1001
ARG HOME="/home/${USER}"
ARG WORKDIR="/workspaces/dev"
ARG UV_GROUP="dev"

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
    && rm -rf /var/lib/apt/lists/*

COPY .devcontainer/scripts /tmp/scripts
RUN chmod +x /tmp/scripts/*.sh

# BUILDER
FROM base AS builder

RUN /tmp/scripts/add_user_and_setup_sudo.sh ${USER} ${UID} \
 && /tmp/scripts/setup_zsh.sh ${USER} ${UID}

WORKDIR ${WORKDIR}
COPY pyproject.toml uv.lock ${WORKDIR}/

RUN install -d -o ${UID} -g ${UID} ${WORKDIR}/.venv ${WORKDIR}/.vendor ${HOME}

RUN chown -R ${UID}:${UID} ${WORKDIR} ${HOME} \
 && sudo -u ${USER} -H /tmp/scripts/setup_uv.sh ${UV_GROUP} ${WORKDIR}

RUN rm -f ${WORKDIR}/pyproject.toml ${WORKDIR}/uv.lock || true

# RUN
FROM base AS final

RUN /tmp/scripts/add_user_and_setup_sudo.sh ${USER} ${UID} \
 && /tmp/scripts/setup_zsh.sh ${USER} ${UID}

WORKDIR ${WORKDIR}

COPY --from=builder --chown=${UID}:${UID} ${WORKDIR}/.venv   ${WORKDIR}/.venv
COPY --from=builder --chown=${UID}:${UID} ${WORKDIR}/.vendor ${WORKDIR}/.vendor
COPY --from=builder --chown=${UID}:${UID} ${HOME}/.cache/uv  ${HOME}/.cache/uv
COPY --from=builder --chown=${UID}:${UID} ${HOME}/.local/share/uv  ${HOME}/.local/share/uv

RUN rm -rf /tmp/scripts

USER ${USER}
ENV VIRTUAL_ENV=${WORKDIR}/.venv
ENV PATH="${WORKDIR}/.venv/bin:${PATH}"

