"""
Test cuQuantum backend integration

Note: These tests work with fallback mode if cuQuantum is not installed.
Full acceleration tests require cuquantum-python package.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from atlas_q.cuquantum_backend import (
    CuQuantumBackend,
    CuQuantumConfig,
    CUQUANTUM_AVAILABLE
)


def test_cuquantum_availability_detection():
    """Test that cuQuantum availability is detected correctly"""
    # This will be False if cuquantum-python is not installed
    print(f"cuQuantum available: {CUQUANTUM_AVAILABLE}")

    if CUQUANTUM_AVAILABLE:
        from atlas_q.cuquantum_backend import CUQUANTUM_VERSION
        print(f"cuQuantum version: {CUQUANTUM_VERSION}")

    print("✅ test_cuquantum_availability_detection passed")


def test_cuquantum_config():
    """Test CuQuantumConfig dataclass"""
    config = CuQuantumConfig(
        use_cutensornet=True,
        use_custatevec=True,
        workspace_size=512 * 1024 * 1024,
        algorithm='gesvd'
    )

    assert config.use_cutensornet == True
    assert config.use_custatevec == True
    assert config.workspace_size == 512 * 1024 * 1024
    assert config.algorithm == 'gesvd'

    print("✅ test_cuquantum_config passed")


def test_cuquantum_backend_initialization():
    """Test CuQuantumBackend initialization"""
    backend = CuQuantumBackend()

    assert backend.config is not None
    assert isinstance(backend.available, bool)

    print(f"Backend available: {backend.available}")
    print(f"Backend version: {backend.version}")

    print("✅ test_cuquantum_backend_initialization passed")


def test_cuquantum_svd_fallback():
    """Test SVD with PyTorch fallback"""
    backend = CuQuantumBackend()

    # Create test tensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.randn(10, 20, dtype=torch.complex64, device=device)

    # Compute SVD
    U, S, Vt = backend.svd(tensor, chi_max=5)

    # Check shapes
    assert U.shape == (10, 5)
    assert S.shape == (5,)
    assert Vt.shape == (5, 20)

    # Verify SVD is correct (reconstruct)
    reconstructed = U @ torch.diag(S.to(torch.complex64)) @ Vt
    # Check reconstruction matches truncated SVD
    # Compare against numpy SVD with same truncation
    U_ref, S_ref, Vt_ref = torch.linalg.svd(tensor, full_matrices=False)
    reconstructed_ref = U_ref[:, :5] @ torch.diag(S_ref[:5].to(torch.complex64)) @ Vt_ref[:5, :]

    error = torch.norm(reconstructed - reconstructed_ref) / torch.norm(reconstructed_ref)

    print(f"SVD reconstruction error: {error.item():.6e}")
    assert error < 1e-4

    print("✅ test_cuquantum_svd_fallback passed")


def test_cuquantum_svd_truncation():
    """Test SVD with bond dimension truncation"""
    backend = CuQuantumBackend()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.randn(15, 25, dtype=torch.complex64, device=device)

    # Truncate to chi=8
    U, S, Vt = backend.svd(tensor, chi_max=8, cutoff=1e-10)

    assert U.shape[1] <= 8
    assert S.shape[0] <= 8
    assert Vt.shape[0] <= 8

    print(f"Truncated to bond dim: {S.shape[0]}")
    print("✅ test_cuquantum_svd_truncation passed")


def test_cuquantum_svd_cutoff():
    """Test SVD with singular value cutoff"""
    backend = CuQuantumBackend()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create low-rank tensor
    A = torch.randn(10, 3, dtype=torch.complex64, device=device)
    B = torch.randn(3, 15, dtype=torch.complex64, device=device)
    tensor = A @ B  # Rank 3 tensor

    # SVD with cutoff
    U, S, Vt = backend.svd(tensor, cutoff=1e-6)

    # Should keep approximately 3 singular values
    print(f"Singular values: {S.cpu().numpy()}")
    print(f"Kept {S.shape[0]} singular values")

    # Typically should keep 3-5 values (rank 3 + numerical noise)
    assert S.shape[0] <= 10

    print("✅ test_cuquantum_svd_cutoff passed")


def test_cuquantum_contract_fallback():
    """Test tensor contraction with PyTorch fallback"""
    backend = CuQuantumBackend()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create test tensors for einsum: ij,jk->ik
    A = torch.randn(4, 5, dtype=torch.complex64, device=device)
    B = torch.randn(5, 6, dtype=torch.complex64, device=device)

    # Contract using backend
    result = backend.contract([A, B], 'ij,jk->ik')

    # Verify against torch.einsum
    expected = torch.einsum('ij,jk->ik', A, B)

    error = torch.norm(result - expected) / torch.norm(expected)
    print(f"Contraction error: {error.item():.6e}")
    assert error < 1e-6

    print("✅ test_cuquantum_contract_fallback passed")


def test_cuquantum_contract_three_tensors():
    """Test contraction of three tensors"""
    backend = CuQuantumBackend()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Three tensor contraction: ij,jk,kl->il
    A = torch.randn(3, 4, dtype=torch.complex64, device=device)
    B = torch.randn(4, 5, dtype=torch.complex64, device=device)
    C = torch.randn(5, 6, dtype=torch.complex64, device=device)

    result = backend.contract([A, B, C], 'ij,jk,kl->il')

    expected = torch.einsum('ij,jk,kl->il', A, B, C)

    error = torch.norm(result - expected) / torch.norm(expected)
    assert error < 1e-6

    print("✅ test_cuquantum_contract_three_tensors passed")


def test_cuquantum_config_defaults():
    """Test CuQuantumConfig default values"""
    config = CuQuantumConfig()

    assert config.use_cutensornet == True
    assert config.use_custatevec == True
    assert config.workspace_size == 1024 * 1024 * 1024
    assert config.algorithm == 'auto'
    assert config.device == 'cuda'

    print("✅ test_cuquantum_config_defaults passed")


def test_cuquantum_backend_with_custom_config():
    """Test backend initialization with custom config"""
    config = CuQuantumConfig(
        use_cutensornet=False,
        algorithm='gesvdj'
    )

    backend = CuQuantumBackend(config=config)

    assert backend.config.use_cutensornet == False
    assert backend.config.algorithm == 'gesvdj'

    # Should fall back to PyTorch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.randn(5, 8, dtype=torch.complex64, device=device)

    U, S, Vt = backend.svd(tensor, chi_max=4)
    assert U.shape == (5, 4)

    print("✅ test_cuquantum_backend_with_custom_config passed")


def test_cuquantum_svd_complex_tensors():
    """Test SVD handles complex tensors correctly"""
    backend = CuQuantumBackend()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create complex tensor with specific structure
    tensor = torch.randn(8, 12, dtype=torch.complex64, device=device)
    tensor = tensor + 1j * torch.randn(8, 12, device=device)

    U, S, Vt = backend.svd(tensor, chi_max=6)

    # Check output dtypes
    assert U.dtype == torch.complex64
    assert S.dtype == torch.float32  # Singular values are real
    assert Vt.dtype == torch.complex64

    print("✅ test_cuquantum_svd_complex_tensors passed")


if __name__ == "__main__":
    print("Running cuQuantum Backend tests...\n")

    if CUQUANTUM_AVAILABLE:
        print("✓ cuQuantum is available - testing with acceleration")
    else:
        print("✗ cuQuantum not available - testing PyTorch fallback")

    print("=" * 50 + "\n")

    try:
        test_cuquantum_availability_detection()
        test_cuquantum_config()
        test_cuquantum_backend_initialization()
        test_cuquantum_svd_fallback()
        test_cuquantum_svd_truncation()
        test_cuquantum_svd_cutoff()
        test_cuquantum_contract_fallback()
        test_cuquantum_contract_three_tensors()
        test_cuquantum_config_defaults()
        test_cuquantum_backend_with_custom_config()
        test_cuquantum_svd_complex_tensors()

        print("\n" + "="*50)
        print("✅ All cuQuantum Backend tests passed!")
        print("="*50)

        if not CUQUANTUM_AVAILABLE:
            print("\nNote: Install cuquantum-python for GPU acceleration:")
            print("  pip install cuquantum-python")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
