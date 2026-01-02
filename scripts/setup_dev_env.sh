#!/usr/bin/env bash

###############################################################################
# Bionic Arm Project - Development Environment Setup Script
# 
# This script sets up the complete development environment including:
# - Python virtual environment with all dependencies
# - System packages (C++ compiler, CMake, etc.)
# - Git hooks and linting tools
# - Docker environment (optional)
#
# Usage: ./scripts/setup_dev_env.sh [--with-docker] [--skip-system]
###############################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
WITH_DOCKER=false
SKIP_SYSTEM=false

for arg in "$@"; do
    case $arg in
        --with-docker)
            WITH_DOCKER=true
            ;;
        --skip-system)
            SKIP_SYSTEM=true
            ;;
        --help)
            echo "Usage: $0 [--with-docker] [--skip-system]"
            echo "  --with-docker  : Also set up Docker environment"
            echo "  --skip-system  : Skip system package installation"
            exit 0
            ;;
    esac
done

# Utility functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Banner
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║          Bionic Arm Project Setup Script                  ║"
echo "║          Development Environment Initialization            ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    log_info "Detected OS: Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    log_info "Detected OS: macOS"
else
    log_error "Unsupported OS: $OSTYPE"
    exit 1
fi

# Check Python version
log_info "Checking Python version..."
if check_command python3.11; then
    PYTHON_CMD="python3.11"
elif check_command python3; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc) -eq 1 ]]; then
        PYTHON_CMD="python3"
    else
        log_error "Python 3.11+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    log_error "Python 3.11+ not found"
    exit 1
fi
log_success "Found $PYTHON_CMD"

# Install system packages
if [[ "$SKIP_SYSTEM" == false ]]; then
    log_info "Installing system packages..."
    
    if [[ "$OS" == "linux" ]]; then
        if check_command apt-get; then
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                git \
                pkg-config \
                libboost-all-dev \
                libeigen3-dev \
                libopencv-dev \
                libhdf5-dev \
                clang-format \
                clang-tidy
            log_success "System packages installed"
        else
            log_warning "apt-get not found, skipping system packages"
        fi
    elif [[ "$OS" == "macos" ]]; then
        if check_command brew; then
            brew install cmake boost eigen opencv hdf5 clang-format
            log_success "System packages installed"
        else
            log_warning "Homebrew not found, skipping system packages"
        fi
    fi
else
    log_info "Skipping system package installation"
fi

# Create virtual environment
log_info "Creating Python virtual environment..."
if [[ ! -d ".venv" ]]; then
    $PYTHON_CMD -m venv .venv
    log_success "Virtual environment created"
else
    log_warning "Virtual environment already exists"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel
log_success "pip upgraded"

# Install Python dependencies
log_info "Installing Python dependencies (this may take several minutes)..."
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
    log_success "Python dependencies installed"
else
    log_error "requirements.txt not found"
    exit 1
fi

# Install PyTorch with CUDA support (if available)
log_info "Checking for CUDA..."
if check_command nvidia-smi; then
    log_info "NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    log_success "PyTorch with CUDA installed"
else
    log_warning "No NVIDIA GPU detected, PyTorch will use CPU only"
fi

# Install pre-commit hooks
log_info "Setting up pre-commit hooks..."
if [[ -f ".pre-commit-config.yaml" ]]; then
    pre-commit install
    log_success "Pre-commit hooks installed"
else
    log_info "Creating .pre-commit-config.yaml..."
    cat > .pre-commit-config.yaml <<'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203']

  - repo: https://github.com/pycqa/pylint
    rev: v3.0.0
    hooks:
      - id: pylint
        args: ['--rcfile=.pylintrc']
EOF
    pre-commit install
    log_success "Pre-commit hooks created and installed"
fi

# Create .env file if it doesn't exist
if [[ ! -f ".env" ]]; then
    log_info "Creating .env file..."
    cat > .env <<'EOF'
# Bionic Arm Project Environment Variables

# BCI Configuration
BCI_DEVICE=simulation  # Options: simulation, openbci, emotiv
BCI_CHANNELS=32
BCI_SAMPLING_RATE=250

# Hardware Configuration
ARM_INTERFACE=simulation  # Options: simulation, can, serial
CAN_BITRATE=1000000

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
USE_GPU=true

# Logging
LOG_LEVEL=INFO
WANDB_PROJECT=bionic-arm
WANDB_ENTITY=your-username  # Replace with your W&B username

# Data Paths
DATA_DIR=./data
MODEL_DIR=./data/models
LOG_DIR=./logs
EOF
    log_success ".env file created"
else
    log_warning ".env file already exists"
fi

# Build CMake project (firmware)
log_info "Building firmware (if applicable)..."
if [[ -f "CMakeLists.txt" ]]; then
    if [[ ! -d "build" ]]; then
        mkdir build
    fi
    cd build
    cmake ..
    cmake --build . --config Debug -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
    cd ..
    log_success "Firmware built"
else
    log_info "No CMakeLists.txt found, skipping firmware build"
fi

# Docker setup (if requested)
if [[ "$WITH_DOCKER" == true ]]; then
    log_info "Setting up Docker environment..."
    
    if check_command docker; then
        cd docker
        docker-compose build
        log_success "Docker images built"
        cd ..
    else
        log_error "Docker not found, skipping Docker setup"
    fi
fi

# Create directory structure
log_info "Ensuring directory structure..."
mkdir -p data/{raw,processed,calibration,models}
mkdir -p logs
mkdir -p assets/{images,videos,cad}
mkdir -p docs/{api,architecture,research,user-guides}
log_success "Directory structure created"

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║          Setup Complete! 🎉                                ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
log_success "Development environment is ready!"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     ${GREEN}source .venv/bin/activate${NC}"
echo ""
echo "  2. Run tests to verify setup:"
echo "     ${GREEN}pytest tests/ -v${NC}"
echo ""
echo "  3. Start simulation:"
echo "     ${GREEN}python src/simulation/demo.py${NC}"
echo ""
echo "  4. Check documentation:"
echo "     ${GREEN}open docs/project-plan.md${NC}"
echo ""
log_info "For more information, see README.md"
echo ""
