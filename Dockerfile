# ATLAS-Q Docker Image (GPU - CUDA 12.2)
# Production-ready image with NVIDIA GPU support and Rust backends

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

LABEL maintainer="ATLAS-Q Development Team"
LABEL description="ATLAS-Q: High-performance quantum simulator with Rust backends (9.3× faster than Qiskit Aer) and GPU CUDA backend (2-13× faster than CPU for 15+ qubits)"
LABEL version="0.7.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, Rust, and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (for high-performance backends)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

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
COPY atlas_q_core/ ./atlas_q_core/

# Build Rust backends (9.3× faster than Qiskit Aer)
WORKDIR /opt/atlas-q/atlas_q_core
RUN PYO3_PYTHON=/usr/bin/python3 cargo build --release && \
    cp target/release/libatlas_q_core.so ../atlas_q_core.so && \
    cp ../atlas_q_core.so ../src/

# Install ATLAS-Q with GPU support
WORKDIR /opt/atlas-q
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
