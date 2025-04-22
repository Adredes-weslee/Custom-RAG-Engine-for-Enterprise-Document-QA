# Use the base Ollama image
FROM ollama/ollama:latest

# Set up environment variables
ARG DEBIAN_FRONTEND="noninteractive"
ARG NON_ROOT_USER="aisg"
ARG NON_ROOT_UID="2222"
ARG NON_ROOT_GID="2222"
ARG HOME_DIR="/home/${NON_ROOT_USER}"
ARG REPO_DIR="."

# Create a non-root user
RUN useradd -l -m -s /bin/bash -u ${NON_ROOT_UID} ${NON_ROOT_USER}

# Install necessary packages
RUN apt update && \
    apt -y install curl git bash python3 python3-pip && \
    apt clean

# Set environment variables
ENV PYTHONIOENCODING=utf8
ENV LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8"
ENV PATH="${HOME_DIR}/.local/bin:${PATH}"
ENV OLLAMA_MODELS="${HOME_DIR}/visiera/.ollama/models"

# Switch to non-root user
USER ${NON_ROOT_USER}
WORKDIR ${HOME_DIR}

# Copy the source code
COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} ${REPO_DIR} visiera

# Copy the startup script
COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} start.sh start.sh
RUN chmod +x start.sh

# Install Python dependencies
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --user -r visiera/requirements.txt

# Expose the necessary ports for Ollama and Streamlit
EXPOSE 11434
EXPOSE 8501

# Set the entrypoint to the startup script
ENTRYPOINT ["./start.sh"]