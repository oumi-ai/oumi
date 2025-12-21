#!/bin/bash
# Oumi Installation Script
# Usage:
#   curl -LsSf https://oumi.ai/install.sh | bash
#   curl -LsSf https://oumi.ai/install.sh | bash -s -- --gpu
#   curl -LsSf https://oumi.ai/install.sh | bash -s -- --python 3.12

set -euo pipefail

# Colors (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Default options
GPU=false
CPU_ONLY=false
VERSION=""
EXTRAS=""
PYTHON_VERSION=""
CURRENT_ENV=false

# Detect GPU availability
detect_gpu() {
    if command -v nvidia-smi > /dev/null 2>&1; then
        echo "nvidia"
    elif command -v rocm-smi > /dev/null 2>&1 || [ -d "/opt/rocm" ]; then
        echo "amd"
    elif [ -d "/usr/local/cuda" ] || [ -n "${CUDA_HOME:-}" ]; then
        echo "nvidia"
    else
        echo "none"
    fi
}

# Print functions
info() { printf "${BLUE}%s${NC}\n" "$1"; }
success() { printf "${GREEN}%s${NC}\n" "$1"; }
warn() { printf "${YELLOW}%s${NC}\n" "$1"; }
error() { printf "${RED}%s${NC}\n" "$1" >&2; }

# Parse arguments
while [ $# -gt 0 ]; do
    case $1 in
        --gpu)
            GPU=true
            shift
            ;;
        --cpu)
            CPU_ONLY=true
            shift
            ;;
        --version)
            [ -z "${2:-}" ] && { error "--version requires a value"; exit 1; }
            VERSION="$2"
            shift 2
            ;;
        --extras)
            [ -z "${2:-}" ] && { error "--extras requires a value"; exit 1; }
            EXTRAS="$2"
            shift 2
            ;;
        --python)
            [ -z "${2:-}" ] && { error "--python requires a value"; exit 1; }
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --current-env)
            CURRENT_ENV=true
            shift
            ;;
        -h|--help)
            cat << 'EOF'
Oumi Installation Script

Usage:
  curl -LsSf https://oumi.ai/install.sh | bash
  curl -LsSf https://oumi.ai/install.sh | bash -s -- [OPTIONS]

Options:
  --gpu             Install with GPU support (default if GPU detected)
  --cpu             Install CPU-only version (skip GPU auto-detection)
  --version VER     Install a specific version (e.g., 0.6.0)
  --extras EXTRA    Install with additional extras (e.g., 'evaluation')
  --python VER      Use specific Python version (e.g., 3.12). uv will download if needed.
  --current-env     Install in current environment instead of as a uv tool.
                    Requires an active virtual environment or conda environment.
  -h, --help        Show this help message

Examples:
  # Basic installation (uv manages the environment)
  curl -LsSf https://oumi.ai/install.sh | bash

  # With GPU support
  curl -LsSf https://oumi.ai/install.sh | bash -s -- --gpu

  # With specific Python version
  curl -LsSf https://oumi.ai/install.sh | bash -s -- --python 3.12

  # Install in current virtual environment
  source .venv/bin/activate
  curl -LsSf https://oumi.ai/install.sh | bash -s -- --current-env
EOF
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            echo "    Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate mutually exclusive options
if [ "$GPU" = true ] && [ "$CPU_ONLY" = true ]; then
    error "--gpu and --cpu are mutually exclusive"
    exit 1
fi

cat << 'EOF'

   ____  _    _ __  __ _____
  / __ \| |  | |  \/  |_   _|
 | |  | | |  | | \  / | | |
 | |  | | |  | | |\/| | | |
 | |__| | |__| | |  | |_| |_
  \____/ \____/|_|  |_|_____|

EOF

# Detect OS
OS="$(uname -s)"

# Check platform support
case "$OS" in
    Linux|Darwin)
        ;;
    MINGW*|MSYS*|CYGWIN*)
        error "Windows detected. Please use WSL (Windows Subsystem for Linux):"
        echo "    1. Install WSL: wsl --install"
        echo "    2. Open WSL terminal and run this script again"
        exit 1
        ;;
    *)
        warn "Unknown OS: $OS. Proceeding anyway..."
        ;;
esac

# Check for curl
if ! command -v curl > /dev/null 2>&1; then
    error "curl is required but not installed."
    echo ""
    echo "Please install curl and try again:"
    case "$OS" in
        Linux)
            echo "    Ubuntu/Debian: sudo apt install curl"
            echo "    Fedora/RHEL:   sudo dnf install curl"
            echo "    Arch:          sudo pacman -S curl"
            ;;
        Darwin)
            echo "    brew install curl"
            ;;
    esac
    exit 1
fi

# Install uv if not present
if command -v uv > /dev/null 2>&1; then
    UV_STATUS="yes"
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"

    if command -v uv > /dev/null 2>&1; then
        UV_STATUS="just installed"
    else
        error "Failed to install uv. Please install manually:"
        echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
fi

# Detect GPU and determine install variant
GPU_TYPE=$(detect_gpu)
if [ "$CPU_ONLY" = true ] || { [ "$GPU" = false ] && [ "$GPU_TYPE" = "none" ]; }; then
    VARIANT="CPU"
else
    GPU=true
    VARIANT="GPU"
fi

info "Detected: OS: $OS | GPU: $GPU_TYPE | uv installed: $UV_STATUS"

# Build package specification
EXTRAS_LIST=""
[ "$GPU" = true ] && EXTRAS_LIST="gpu"
if [ -n "$EXTRAS" ]; then
    EXTRAS_LIST="${EXTRAS_LIST:+$EXTRAS_LIST,}$EXTRAS"
fi

if [ -n "$EXTRAS_LIST" ]; then
    PACKAGE="oumi[$EXTRAS_LIST]"
else
    PACKAGE="oumi"
fi

[ -n "$VERSION" ] && PACKAGE="${PACKAGE}==${VERSION}"

# Build command based on installation mode
if [ "$CURRENT_ENV" = true ]; then
    # Install in current environment - require active venv or conda env
    if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
        error "No virtual environment active."
        echo ""
        echo "Either activate a virtual environment first:"
        echo "    source .venv/bin/activate  # or: conda activate myenv"
        echo "    curl -LsSf https://oumi.ai/install.sh | bash -s -- --current-env"
        echo ""
        echo "Or install as a uv tool (recommended):"
        echo "    curl -LsSf https://oumi.ai/install.sh | bash"
        exit 1
    fi

    if [ -n "${CONDA_PREFIX:-}" ]; then
        info "Installing oumi ($VARIANT) in conda environment: $(basename "$CONDA_PREFIX")..."
    else
        info "Installing oumi ($VARIANT) in virtual environment..."
    fi
    [ -n "$PYTHON_VERSION" ] && warn "Note: --python is ignored with --current-env (uses current environment's Python)"
    INSTALL_CMD=(uv pip install "$PACKAGE" --prerelease=allow)
else
    # Install as uv tool (default) - uv manages everything
    info "Installing oumi ($VARIANT) as uv tool..."
    INSTALL_CMD=(uv tool install "$PACKAGE" --prerelease=allow)

    if [ -n "$PYTHON_VERSION" ]; then
        INSTALL_CMD+=(--python "$PYTHON_VERSION")
    fi
fi

# Run installation
if "${INSTALL_CMD[@]}"; then
    echo ""
    success "Oumi installed successfully!"
    echo ""
    echo "Quick start:"
    echo "    oumi --help                     # Show available commands"
    echo "    oumi env                        # Check your environment"
    echo "    oumi infer -c smollm-135m -i    # Interactive inference"
    echo "    oumi train -c smollm-135m       # Train a model"
    echo ""
    echo "To upgrade later:"
    if [ "$CURRENT_ENV" = true ]; then
        echo "    uv pip install --upgrade oumi"
    else
        echo "    uv tool upgrade oumi"
    fi
    echo ""
    echo "Documentation: https://oumi.ai/docs"
    echo "Discord:       https://discord.gg/oumi"
else
    echo ""
    error "Installation failed."
    echo ""
    echo "Try installing manually:"
    if [ "$CURRENT_ENV" = true ]; then
        echo "    uv pip install '$PACKAGE'"
    else
        echo "    uv tool install '$PACKAGE'"
    fi
    echo ""
    echo "Or with pip:"
    echo "    pip install '$PACKAGE'"
    echo ""
    echo "If you encounter issues, please report them at:"
    echo "    https://github.com/oumi-ai/oumi/issues"
    exit 1
fi
