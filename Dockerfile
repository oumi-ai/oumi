ARG PYTORCH_VERSION=2.8.0
ARG CUDA_VERSION=12.8
ARG PYTHON_VERSION=3.11

# Use PyTorch base images for AMD64, minimal Python for ARM64
FROM --platform=linux/amd64 pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn9-runtime AS base-amd64
FROM --platform=linux/arm64 python:${PYTHON_VERSION}-slim-bookworm AS base-arm64

# Select base image based on build architecture
FROM base-${TARGETARCH} AS final

# ARG for oumi version - defaults to empty string which will install latest
ARG OUMI_VERSION=
ARG TARGETARCH
ARG CUDA_VERSION
ARG PYTHON_VERSION

WORKDIR /oumi_workdir

# Create oumi user
RUN groupadd -r oumi && useradd -r -g oumi -m -s /bin/bash oumi
RUN chown -R oumi:oumi /oumi_workdir

# Install system dependencies
# ARM64 needs build essentials for compiling some Python packages
ARG TARGETARCH
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        vim \
        htop \
        tree \
        screen \
        curl \
        ca-certificates \
        $([ "$TARGETARCH" = "arm64" ] && echo "build-essential" || echo "") && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install Oumi dependencies
# AMD64: Install with GPU support + CUDA-matched PyTorch wheels
# ARM64: Install CPU-only version
RUN pip install --no-cache-dir uv && \
    if [ "$TARGETARCH" = "arm64" ]; then \
        OUMI_EXTRAS=""; \
    else \
        OUMI_EXTRAS="[gpu]"; \
        # Extract CUDA version (e.g., 12.4 -> 124 for PyTorch index)
        CUDA_VERSION_SHORT=$(echo ${CUDA_VERSION} | cut -d. -f1,2 | tr -d .); \
        # Install torch and torchvision from PyTorch index first
        uv pip install --system --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT} \
            torch torchvision; \
    fi && \
    # Install oumi using regular pip (without PyTorch index)
    if [ -z "$OUMI_VERSION" ]; then \
        uv pip install --system --no-cache-dir --prerelease=allow "oumi${OUMI_EXTRAS}"; \
    else \
        uv pip install --system --no-cache-dir --prerelease=allow "oumi${OUMI_EXTRAS}==$OUMI_VERSION"; \
    fi

# Switch to oumi user
USER oumi

# Copy application code
COPY . /oumi_workdir
