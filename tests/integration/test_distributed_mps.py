"""
Test Distributed MPS for multi-GPU simulation

Note: These tests work with single-GPU/CPU mode.
Full multi-GPU testing requires multiple CUDA devices.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from atlas_q.distributed_mps import (
    DistributedMPS,
    DistributedConfig,
    DistMode,
    MPSPartition
)


def test_distributed_config():
    """Test DistributedConfig dataclass"""
    config = DistributedConfig(
        mode=DistMode.BOND_PARALLEL,
        backend='nccl',
        world_size=2
    )

    assert config.mode == DistMode.BOND_PARALLEL
    assert config.backend == 'nccl'
    assert config.world_size == 2
    assert config.overlap_comm == True  # Default

    print("✅ test_distributed_config passed")


def test_distributed_mps_single_gpu():
    """Test DistributedMPS in single-GPU mode (no actual distribution)"""
    config = DistributedConfig(mode=DistMode.NONE, world_size=1)

    dmps = DistributedMPS(num_qubits=8, bond_dim=4, config=config)

    assert dmps.num_qubits == 8
    assert dmps.bond_dim == 4
    assert dmps.rank == 0
    assert dmps.world_size == 1

    # Check partition
    assert dmps.partition.start_site == 0
    assert dmps.partition.end_site == 8
    assert len(dmps.partition.tensors) == 8

    print("✅ test_distributed_mps_single_gpu passed")


def test_distributed_mps_partition_creation():
    """Test MPS partition creation for single GPU"""
    config = DistributedConfig(mode=DistMode.NONE)
    dmps = DistributedMPS(num_qubits=10, bond_dim=8, config=config)

    partition = dmps.partition

    assert partition.rank == 0
    assert partition.start_site == 0
    assert partition.end_site == 10
    assert len(partition.tensors) == 10

    # Check tensor shapes
    # Left boundary: [1, d, χ]
    assert partition.tensors[0].shape[0] == 1

    # Right boundary: [χ, d, 1]
    assert partition.tensors[9].shape[2] == 1

    # Bulk: [χ, d, χ]
    for i in range(1, 9):
        assert partition.tensors[i].shape == (8, 2, 8)

    print("✅ test_distributed_mps_partition_creation passed")


def test_distributed_mps_initialization():
    """Test that DistributedMPS initializes in |0⟩ state"""
    config = DistributedConfig(mode=DistMode.NONE)
    dmps = DistributedMPS(num_qubits=5, bond_dim=4, config=config)

    # Check first tensor is initialized to |0⟩
    first_tensor = dmps.partition.tensors[0]
    assert first_tensor[0, 0, 0] == 1.0  # |0⟩ state

    print("✅ test_distributed_mps_initialization passed")


def test_mps_partition_dataclass():
    """Test MPSPartition dataclass"""
    device = torch.device('cpu')
    tensors = [
        torch.zeros(1, 2, 4, dtype=torch.complex64),
        torch.zeros(4, 2, 4, dtype=torch.complex64),
        torch.zeros(4, 2, 1, dtype=torch.complex64)
    ]

    partition = MPSPartition(
        rank=0,
        start_site=0,
        end_site=3,
        tensors=tensors,
        device=device
    )

    assert partition.rank == 0
    assert partition.start_site == 0
    assert partition.end_site == 3
    assert len(partition.tensors) == 3

    print("✅ test_mps_partition_dataclass passed")


def test_distributed_mps_gate_counter():
    """Test gate counter for checkpointing"""
    config = DistributedConfig(mode=DistMode.NONE, checkpoint_every=10)
    dmps = DistributedMPS(num_qubits=6, bond_dim=4, config=config)

    assert dmps.gate_counter == 0
    assert dmps.config.checkpoint_every == 10

    print("✅ test_distributed_mps_gate_counter passed")


def test_distributed_modes():
    """Test different distribution modes enum"""
    assert DistMode.NONE.value == "none"
    assert DistMode.DATA_PARALLEL.value == "data_parallel"
    assert DistMode.BOND_PARALLEL.value == "bond_parallel"

    print("✅ test_distributed_modes passed")


def test_distributed_mps_device_assignment():
    """Test device assignment in partition"""
    config = DistributedConfig(mode=DistMode.NONE)
    dmps = DistributedMPS(num_qubits=4, bond_dim=2, config=config)

    # All tensors should be on the same device
    device = dmps.partition.device
    for tensor in dmps.partition.tensors:
        assert tensor.device == device

    print(f"Tensors on device: {device}")
    print("✅ test_distributed_mps_device_assignment passed")


def test_distributed_mps_bond_dimensions():
    """Test that bond dimensions are respected"""
    config = DistributedConfig(mode=DistMode.NONE)
    bond_dim = 16
    dmps = DistributedMPS(num_qubits=6, bond_dim=bond_dim, config=config)

    # Bulk tensors should have bond_dim
    bulk_tensor = dmps.partition.tensors[3]
    assert bulk_tensor.shape == (bond_dim, 2, bond_dim)

    print("✅ test_distributed_mps_bond_dimensions passed")


def test_distributed_config_defaults():
    """Test DistributedConfig default values"""
    config = DistributedConfig()

    assert config.mode == DistMode.BOND_PARALLEL
    assert config.backend == 'nccl'
    assert config.world_size == 1
    assert config.overlap_comm == True
    assert config.checkpoint_every == 100
    assert config.checkpoint_dir == './checkpoints'

    print("✅ test_distributed_config_defaults passed")


if __name__ == "__main__":
    print("Running Distributed MPS tests...\n")
    print("Note: Testing in single-GPU mode (no torch.distributed)")
    print("=" * 50 + "\n")

    try:
        test_distributed_config()
        test_distributed_mps_single_gpu()
        test_distributed_mps_partition_creation()
        test_distributed_mps_initialization()
        test_mps_partition_dataclass()
        test_distributed_mps_gate_counter()
        test_distributed_modes()
        test_distributed_mps_device_assignment()
        test_distributed_mps_bond_dimensions()
        test_distributed_config_defaults()

        print("\n" + "="*50)
        print("✅ All Distributed MPS tests passed!")
        print("="*50)
        print("\nNote: Full multi-GPU tests require NCCL and multiple GPUs")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
