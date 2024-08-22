ARG D_USER="zizou"
ARG D_GROUP="zizou"
ARG D_UID="1001"
ARG D_GID="1001"

ARG ROOT_CONTAINER=condaforge/mambaforge:4.12.0-0
ARG BASE_CONTAINER=$ROOT_CONTAINER
FROM $BASE_CONTAINER AS base 
USER root

COPY  ./environment.yml .
RUN --mount=type=cache,target=$HOME/.conda/pkgs mamba env create -p /env -f environment.yml

FROM $BASE_CONTAINER AS ops 

ARG D_USER
ARG D_GROUP
ARG D_UID
ARG D_GID

# Configure environment
ENV SHELL=/bin/bash \
    D_USER=$D_USER \
    D_UID=$D_UID \
    D_GID=$D_GID \
    D_GROUP=$D_GROUP \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    HOME=/home/$D_USER


# Copy a script that we will use to correct permissions after running certain commands
COPY fix-permissions /usr/local/bin/fix-permissions

# Create D_USER with name linus user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN groupadd -g $D_GID $D_GROUP && \
    useradd -m -s /bin/bash -N -u $D_UID -G $D_GROUP $D_USER && \
    chmod a+rx /usr/local/bin/fix-permissions && \
    mkdir /env && \
    mkdir $HOME/data && \
    mkdir -p $HOME/.conda/pkgs && \
    fix-permissions $HOME && \
    fix-permissions /env && \
    fix-permissions $HOME/data && \
    fix-permissions $CONDA_DIR

USER $D_USER    

ARG WORKDIR=$HOME
WORKDIR $WORKDIR
COPY --from=BASE --chown=${D_USER}:${D_GID} /env /env
COPY --chown=$D_USER:users . .
RUN /env/bin/pip install -e .

FROM $BASE_CONTAINER AS dev 

ARG D_USER
ARG D_GROUP
ARG D_UID
ARG D_GID

# Configure environment
ENV SHELL=/bin/bash \
    D_USER=$D_USER \
    D_UID=$D_UID \
    D_GID=$D_GID \
    D_GROUP=$D_GROUP \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    HOME=/home/$D_USER


# Copy a script that we will use to correct permissions after running certain commands
COPY fix-permissions /usr/local/bin/fix-permissions

# Create D_USER with name linus user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN groupadd -g $D_GID $D_GROUP && \
    useradd -m -s /bin/bash -N -u $D_UID -G $D_GROUP $D_USER && \
    chmod a+rx /usr/local/bin/fix-permissions && \
    mkdir /env && \
    mkdir $HOME/data && \
    mkdir -p $HOME/.conda/pkgs && \
    fix-permissions $HOME && \
    fix-permissions /env && \
    fix-permissions $HOME/data && \
    fix-permissions $CONDA_DIR

USER $D_USER    

ARG WORKDIR=$HOME
WORKDIR $WORKDIR
COPY --from=BASE --chown=${D_USER}:${D_GID} /env /env
COPY --chown=$D_USER:users . .