ARG TARGETPLATFORM=linux/amd64

# Use CUDA runtime for AMD64, CPU-only for ARM64
# AMD64: CUDA 12.4 with GPU support
# ARM64: CPU-only
FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS base-amd64
FROM --platform=linux/arm64 pytorch/pytorch:2.5.1-cpu AS base-arm64

# Select base image based on build architecture
FROM base-${TARGETARCH} AS final

# ARG for oumi version - defaults to empty string which will install latest
ARG OUMI_VERSION=

WORKDIR /oumi_workdir

# Create oumi user
RUN groupadd -r oumi && useradd -r -g oumi -m -s /bin/bash oumi
RUN chown -R oumi:oumi /oumi_workdir

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        vim \
        htop \
        tree \
        screen \
        curl \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install Oumi dependencies
# AMD64: Install with GPU support
# ARM64: Install CPU-only version (no CUDA support on Apple Silicon)
ARG TARGETARCH
RUN pip install --no-cache-dir uv && \
    if [ "$TARGETARCH" = "arm64" ]; then \
        OUMI_EXTRAS=""; \
    else \
        OUMI_EXTRAS="[gpu]"; \
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
