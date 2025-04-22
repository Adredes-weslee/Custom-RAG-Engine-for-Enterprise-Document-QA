FROM python:3.11.9-slim

ARG DEBIAN_FRONTEND="noninteractive"

ARG NON_ROOT_USER="aisg"
ARG NON_ROOT_UID="2222"
ARG NON_ROOT_GID="2222" 
ARG HOME_DIR="/home/${NON_ROOT_USER}"

ARG REPO_DIR="."

RUN useradd -l -m -s /bin/bash -u ${NON_ROOT_UID} ${NON_ROOT_USER}

RUN apt update && \
    apt -y install curl git bash && \
    apt clean

ENV PYTHONIOENCODING utf8
ENV LANG "C.UTF-8"
ENV LC_ALL "C.UTF-8"
ENV PATH "./.local/bin:${PATH}"

USER ${NON_ROOT_USER}
WORKDIR ${HOME_DIR}

COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} ${REPO_DIR} visiera
# Install pip requirements, modify this line to install the requirements for the assignment
RUN pip install -r visiera/requirements.txt 