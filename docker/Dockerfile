# Multi-stage Dockerfile for Bionic Arm Development Environment
# Includes Python, C++, CUDA support, and all dependencies

# Stage 1: Base image with system dependencies
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    pkg-config \
    # Python dependencies
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # C++ development
    g++-11 \
    clang-14 \
    clang-format-14 \
    clang-tidy-14 \
    # Libraries
    libboost-all-dev \
    libeigen3-dev \
    libopencv-dev \
    libhdf5-dev \
    libssl-dev \
    # Hardware interface
    can-utils \
    libsocketcan-dev \
    # Utilities
    vim \
    tmux \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Stage 2: Development environment
FROM base AS development

WORKDIR /workspace

# Create virtual environment inside container
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies in venv
RUN . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir \
    # BCI libraries
    mne \
    mne-features \
    pyedflib \
    pylsl \
    # Deep learning
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    tensorflow \
    # Robotics
    pybullet \
    mujoco \
    # Scientific computing
    numpy scipy scikit-learn \
    pandas matplotlib seaborn \
    # Development tools
    pytest pytest-cov pytest-timeout \
    black pylint flake8 mypy \
    ipython jupyter \
    sphinx sphinx-rtd-theme

# Install additional tools
RUN . /opt/venv/bin/activate && \
    pip install --no-cache-dir \
    pre-commit \
    tensorboard \
    wandb

# Set up development user (non-root)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} developer && \
    useradd -l -u ${USER_ID} -g developer -m -s /bin/bash developer && \
    echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Copy project files
COPY --chown=developer:developer . /workspace

# Switch to development user
USER developer

# Activate venv by default in bash
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]

# Stage 3: Testing environment
FROM development AS testing

USER root

# Install additional testing tools
RUN . /opt/venv/bin/activate && \
    pip install --no-cache-dir \
    pytest-xdist \
    pytest-benchmark \
    hypothesis \
    coverage[toml]

USER developer

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=html"]

# Stage 4: Production runtime (minimal)
FROM base AS production

WORKDIR /app

# Create venv for production
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only necessary files
COPY requirements.txt /app/
RUN . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121 \
    mne pylsl pybullet

# Copy source code
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Set production user
RUN useradd -m -s /bin/bash bionicarm
USER bionicarm

# Activate venv
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)"

# Default command runs the main system
CMD ["python", "src/integration/main.py"]
