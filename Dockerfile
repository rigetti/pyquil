# syntax=docker/dockerfile:1.3

FROM python:3.7 AS dev

USER root

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    software-properties-common \
    bash-completion \
    libblas-dev \
    libffi-dev \
    liblapack-dev \ 
    libzmq3-dev \
    sudo

# Create the docker user 'rigetti'
ENV USER=rigetti
ARG UID=1000
ARG GID=1000
RUN addgroup --gid ${GID} ${USER} \
    && adduser \
    --disabled-password \
    --gecos ${USER} \
    --gid ${GID} \
    --uid ${UID} \
    --home /home/rigetti/ \
    ${USER} \
    && usermod -aG sudo ${USER} \
    && passwd -d ${USER}

# Create our project directory and switch to it
ARG PROJECT_SLUG=pyquil
ARG SRC_DIR=.
ENV APP_DIR=/home/${USER}/${PROJECT_SLUG}
RUN mkdir -p ${APP_DIR}/${SRC_DIR} && chown -R ${USER}:${USER} ${APP_DIR}/${SRC_DIR}

WORKDIR ${APP_DIR}/${SRC_DIR}

USER ${USER}

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/home/${USER}/.local/bin"
RUN mkdir -p /home/${USER}/.config/pypoetry/

# Install the package
COPY --chown=${USER}:${USER} ${SRC_DIR}/README.md ${SRC_DIR}/pyproject.toml ${SRC_DIR}/poetry.lock ./
COPY --chown=${USER}:${USER} --chmod=775 ${SRC_DIR}/pyquil/__init__.py ./pyquil/__init__.py
RUN poetry install \
    --no-interaction \
    --no-ansi \
    --extras latex
RUN rm -r ./pyquil

# Set up the venv
RUN mkdir ${HOME}/.venv/ && ln -s $(poetry env info -p) /home/${USER}/.venv/${PROJECT_SLUG}
ENV VIRTUAL_ENV=/home/${USER}/.venv/${PROJECT_SLUG}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR ${APP_DIR}
ENV SHELL /bin/bash
