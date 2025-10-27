#!/bin/bash
# ATLAS-Q Triton Setup Script
# Sets required environment variables for Triton GPU kernels on GB10/DGX systems

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

# Set Triton environment variables
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="12.0"

echo "✅ Set TRITON_PTXAS_PATH=$TRITON_PTXAS_PATH"
echo "✅ Set TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo ""

# Check if already in bashrc
if grep -q "TRITON_PTXAS_PATH" ~/.bashrc; then
    echo "ℹ️  Triton variables already in ~/.bashrc"
else
    echo "Adding Triton environment variables to ~/.bashrc..."
    cat >> ~/.bashrc << 'EOF'

# ATLAS-Q Triton GPU kernel setup (added by setup_triton.sh)
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="12.0"
EOF
    echo "✅ Added to ~/.bashrc (will persist in new shells)"
fi

echo ""
echo "Testing Triton kernel compilation..."

# Test if Triton works
python3 << 'PYEOF'
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
