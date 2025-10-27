# ATLAS-Q Docker Image (GPU - CUDA 12.2)
# Production-ready image with NVIDIA GPU support

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

LABEL maintainer="ATLAS-Q Development Team"
LABEL description="ATLAS-Q: GPU-accelerated quantum tensor network simulator"
LABEL version="0.6.1"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Set working directory
WORKDIR /opt/atlas-q

# Copy package files first (for layer caching)
COPY pyproject.toml README.md MANIFEST.in ./
COPY src/ ./src/
COPY models/ ./models/

# Install ATLAS-Q with GPU support
RUN pip install .[gpu]

# Set environment variables for GPU optimization
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0"
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create non-root user
RUN useradd -m -u 1000 atlasq && chown -R atlasq:atlasq /opt/atlas-q
USER atlasq

# Default command: Python REPL
CMD ["python3"]
