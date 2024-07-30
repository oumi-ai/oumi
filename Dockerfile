FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime # 3.87 GB
#FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel  # 7.49 GB

WORKDIR /lema_workdir

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git vim htop tree screen \
    && rm -rf /var/lib/apt/lists/*

# Install lema dependencies
COPY pyproject.toml /lema_workdir
RUN pip install --no-cache-dir -e ".[dev,train]"

# Copy application code
COPY . /lema_workdir
