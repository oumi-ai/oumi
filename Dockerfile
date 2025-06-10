ARG TARGETPLATFORM=linux/amd64
FROM --platform=${TARGETPLATFORM} pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

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
# If OUMI_VERSION is provided, install that specific version, otherwise install latest
RUN pip install --no-cache-dir uv && \
    if [ -z "$OUMI_VERSION" ]; then \
        uv pip install --system --no-cache-dir "oumi[gpu]"; \
    else \
        uv pip install --system --no-cache-dir "oumi[gpu]==$OUMI_VERSION"; \
    fi

# Switch to oumi user
USER oumi

# Copy application code
COPY . /oumi_workdir
