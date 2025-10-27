#!/bin/bash
# ATLAS-Q Triton Setup Script
# Universal setup for Triton GPU kernels - auto-detects GPU architecture

set -e

echo "=========================================="
echo "ATLAS-Q Triton GPU Kernel Setup"
echo "=========================================="
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: CUDA not found. Please install CUDA toolkit first."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9\.]*\).*/\1/p')
echo "✅ CUDA Version: $CUDA_VERSION"

# Auto-detect GPU compute capability
echo ""
echo "Detecting GPU architecture..."

# Try to use Python from venv if available, otherwise use system python3
if [ -f "../venv/bin/python" ]; then
    PYTHON_CMD="../venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ ERROR: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

GPU_ARCH=$($PYTHON_CMD << 'PYEOF'
import torch
import sys

if not torch.cuda.is_available():
    print("NO_GPU", file=sys.stderr)
    sys.exit(1)

# Get GPU name
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU: {gpu_name}", file=sys.stderr)

# Get compute capability
capability = torch.cuda.get_device_capability(0)
compute_cap = f"{capability[0]}.{capability[1]}"
print(f"Compute Capability: {compute_cap}", file=sys.stderr)

# Map to TORCH_CUDA_ARCH_LIST format
# Note: For Triton, we need to set the architecture properly
major, minor = capability

# Determine architecture list based on detected GPU
if major == 7:
    # V100, Titan V
    arch_list = "7.0"
elif major == 8:
    # A100, A10, RTX 30xx
    if minor == 0:
        arch_list = "8.0"
    else:
        arch_list = "8.6"
elif major == 9:
    # H100
    arch_list = "9.0"
elif major >= 10 and major <= 12:
    # GB100, GB200 (Blackwell), future architectures
    arch_list = "12.0"
else:
    # Default to detected capability
    arch_list = compute_cap

print(arch_list)
PYEOF
)

GPU_DETECT_RESULT=$?

if [ $GPU_DETECT_RESULT -ne 0 ]; then
    echo "⚠️  WARNING: No GPU detected, using default settings"
    echo "   Triton kernels will not be available"
    TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0"
else
    TORCH_CUDA_ARCH_LIST=$GPU_ARCH
    echo "✅ Auto-detected architecture: $TORCH_CUDA_ARCH_LIST"
fi

# Set Triton environment variables
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST

echo "✅ Set TRITON_PTXAS_PATH=$TRITON_PTXAS_PATH"
echo "✅ Set TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo ""

# Check if already in bashrc
if grep -q "TRITON_PTXAS_PATH" ~/.bashrc; then
    echo "ℹ️  Triton variables already in ~/.bashrc"
    echo "   To update, remove old lines and re-run this script"
else
    echo "Adding Triton environment variables to ~/.bashrc..."
    cat >> ~/.bashrc << EOF

# ATLAS-Q Triton GPU kernel setup (added by setup_triton.sh)
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"
EOF
    echo "✅ Added to ~/.bashrc (will persist in new shells)"
fi

echo ""
echo "Testing Triton kernel compilation..."

# Test if Triton works
$PYTHON_CMD << 'PYEOF'
import sys
try:
    from triton_kernels.modpow import batched_modpow_triton
    results = batched_modpow_triton(7, list(range(10)), 899, device='cuda')
    print("✅ SUCCESS: Triton kernels working!")
    sys.exit(0)
except ImportError as e:
    print(f"❌ ERROR: Cannot import triton_kernels: {e}")
    print("   Run: pip install triton>=2.0.0")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: Triton kernel test failed: {e}")
    sys.exit(1)
PYEOF

TEST_RESULT=$?

echo ""
echo "=========================================="
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ Triton setup complete!"
    echo ""
    echo "Environment variables set for this session."
    echo "Run 'source ~/.bashrc' to load in current shell."
else
    echo "❌ Triton setup incomplete - see errors above"
    exit 1
fi
echo "=========================================="
