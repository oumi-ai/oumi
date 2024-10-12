FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime # 3.87 GB
#FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel  # 7.49 GB

WORKDIR /oumi_workdir

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git vim htop tree screen \
    && rm -rf /var/lib/apt/lists/*

# Install Oumi dependencies
COPY pyproject.toml /oumi_workdir
RUN pip install uv && uv pip install --no-cache-dir -e ".[dev]"

# Copy application code
COPY . /oumi_workdir
