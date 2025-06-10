#!/bin/bash
# Script to build the oumi docker image

set -e

# Default values
TAG="latest"
OUMI_VERSION=""

# Simple argument parsing
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [TAG] [OUMI_VERSION]"
    echo "  TAG: Docker image tag (default: latest)"
    echo "  OUMI_VERSION: Oumi version to install (default: latest)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Build with latest oumi"
    echo "  $0 my-tag             # Build with custom tag"
    echo "  $0 my-tag 0.1.0       # Build with specific oumi version"
    exit 0
fi

# Parse positional arguments
if [ -n "$1" ]; then
    TAG="$1"
fi
if [ -n "$2" ]; then
    OUMI_VERSION="$2"
fi

# Build the image
echo "Building oumi:${TAG}..."
if [ -n "$OUMI_VERSION" ]; then
    echo "Using oumi version: ${OUMI_VERSION}"
    docker build -t "oumi:${TAG}" --build-arg OUMI_VERSION="${OUMI_VERSION}" .
else
    echo "Using latest oumi version"
    docker build -t "oumi:${TAG}" .
fi

# Test the image
echo ""
echo "Testing image..."
docker run --rm "oumi:${TAG}" oumi --version

echo ""
echo "Build complete! To run: docker run -it --gpus all oumi:${TAG}"
