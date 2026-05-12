ARG PYTORCH_VERSION=2.9.1
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
        ca-certificates \
        curl \
        findutils \
        git \
        grep \
        htop \
        iputils-ping \
        less \
        nano \
        net-tools \
        openssh-client \
        procps \
        rsync \
        screen \
        tree \
        unzip \
        vim \
        wget \
        tmux \
        zip \
        $([ "$TARGETARCH" = "arm64" ] && echo "build-essential" || echo "") && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install Oumi dependencies.
# AMD64: install with the recommended GPU stack and CUDA-matched PyTorch
# wheels via the `cudaNNN` extra (one resolver pass, no separate torch
# pre-install). See docs/get_started/installation.md.
# ARM64: install CPU-only.
RUN pip install --no-cache-dir uv && \
    if [ "$TARGETARCH" = "arm64" ]; then \
        OUMI_EXTRAS=""; \
    else \
        CUDA_VERSION_SHORT=$(echo ${CUDA_VERSION} | cut -d. -f1,2 | tr -d .); \
        case "$CUDA_VERSION_SHORT" in \
            126|128|129) ;; \
            *) echo "ERROR: CUDA_VERSION=${CUDA_VERSION} not supported. Use 12.6, 12.8, or 12.9." >&2; exit 1 ;; \
        esac; \
        OUMI_EXTRAS="[torch29,tf4,cuda${CUDA_VERSION_SHORT}]"; \
    fi && \
    if [ -z "$OUMI_VERSION" ]; then \
        uv pip install --system --no-cache-dir --prerelease=allow "oumi${OUMI_EXTRAS}"; \
    else \
        uv pip install --system --no-cache-dir --prerelease=allow "oumi${OUMI_EXTRAS}==$OUMI_VERSION"; \
    fi

# Switch to oumi user
USER oumi

# Copy application code
COPY . /oumi_workdir
