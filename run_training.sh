#!/bin/bash
# Production Training Script for AQED Ultra-Fast Transformer
# Includes GB10 torch.compile workaround
#
# Usage:
#   ./run_training.sh --seq_len 4096 --batch_size 8 --epochs 3 --compile
#
# Author: Quantum Hybrid Simulator Team
# Date: October 24, 2025

set -e  # Exit on error

# GB10 torch.compile workaround
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="12.0"

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found at venv/bin/activate"
    exit 1
fi

# Check PyTorch is available
python3 -c "import torch" 2>/dev/null || {
    echo "Error: PyTorch not installed in virtual environment"
    exit 1
}

# Print configuration
echo "========================================"
echo "AQED Ultra-Fast Transformer Training"
echo "========================================"
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "========================================"
echo ""

# Run training
python3 transformers/train_transformer_ultra_fast.py "$@"

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
