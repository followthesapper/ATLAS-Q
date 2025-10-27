#!/usr/bin/env python3
"""
ATLAS-Q Command-Line Interface

Usage:
    python -m atlas_q [command] [options]
    atlas-q [command] [options]  # If installed via pip

Commands:
    factor <N>          Factor integer N using quantum period-finding
    benchmark           Run all benchmark suites
    info                Display system and version information
    demo                Run interactive demo
    help                Show this help message

Options:
    -h, --help          Show help message
    -v, --version       Show version information
    --device DEVICE     Set device (cuda/cpu, default: cuda if available)
    --verbose           Enable verbose output

Examples:
    python -m atlas_q factor 221
    python -m atlas_q benchmark --device cpu
    python -m atlas_q info
"""

import argparse
import sys
from pathlib import Path

__version__ = '0.6.1'


def print_banner():
    """Print ATLAS-Q banner"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║        ATLAS-Q - Quantum Tensor Network Simulator         ║
║     GPU-Accelerated | Adaptive MPS | Molecular Chemistry  ║
║                     Version 0.6.1                         ║
╚═══════════════════════════════════════════════════════════╝
""")


def cmd_factor(args):
    """Factor an integer using quantum period-finding"""
    from atlas_q.quantum_hybrid_system import QuantumClassicalHybrid

    N = args.N
    print(f"Factoring {N} using quantum period-finding...")

    sim = QuantumClassicalHybrid()
    factors = sim.factor(N)

    if factors and len(factors) == 2:
        print(f"\n✅ Success: {N} = {factors[0]} × {factors[1]}")
    else:
        print(f"\n❌ Failed to factor {N}")
        print(f"Result: {factors}")

    return 0


def cmd_benchmark(args):
    """Run benchmark suites"""
    import subprocess

    print("Running ATLAS-Q benchmarks...")
    print("=" * 60)

    # Check if benchmark script exists
    benchmark_script = Path(__file__).parent.parent.parent / "scripts" / "benchmarks" / "validate_all_features.py"

    if benchmark_script.exists():
        result = subprocess.run([sys.executable, str(benchmark_script)],
                              capture_output=False)
        return result.returncode
    else:
        print("⚠️  Benchmark script not found")
        print("Running basic integration tests instead...")

        result = subprocess.run([sys.executable, "-m", "pytest",
                               "tests/integration/", "-v"],
                              capture_output=False)
        return result.returncode


def cmd_info(args):
    """Display system information"""
    import torch

    from atlas_q import __version__

    print("\n📊 ATLAS-Q System Information")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Python: {sys.version.split()[0]}")

    # PyTorch info
    print(f"\nPyTorch:")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Optional dependencies
    print(f"\nOptional Dependencies:")

    try:
        import cuquantum
        print(f"  cuQuantum: {cuquantum.__version__} ✅")
    except ImportError:
        print(f"  cuQuantum: Not installed (optional)")

    try:
        import pyscf
        print(f"  PySCF: {pyscf.__version__} ✅")
    except ImportError:
        print(f"  PySCF: Not installed (optional)")

    # Feature status
    print(f"\nAvailable Features:")
    print(f"  ✅ Adaptive MPS")
    print(f"  ✅ Noise Models")
    print(f"  ✅ Stabilizer Backend")
    print(f"  ✅ MPO Operations")
    print(f"  ✅ TDVP Time Evolution")
    print(f"  ✅ VQE/QAOA")
    print(f"  ✅ 2D Circuits")
    print(f"  ✅ Circuit Cutting")
    print(f"  ✅ PEPS 2D Networks")
    print(f"  ✅ Distributed MPS")

    if 'cuquantum' in sys.modules:
        print(f"  ✅ cuQuantum Backend (accelerated)")
    else:
        print(f"  ⚪ cuQuantum Backend (fallback mode)")

    print()
    return 0


def cmd_demo(args):
    """Run interactive demo"""
    print("\n🚀 ATLAS-Q Interactive Demo")
    print("=" * 60)
    print("\nThis will demonstrate key features of ATLAS-Q.")
    print("Press Ctrl+C to exit at any time.\n")

    try:
        # Demo 1: Period finding
        print("Demo 1: Quantum Period-Finding")
        print("-" * 60)
        from atlas_q.quantum_hybrid_system import QuantumClassicalHybrid
        sim = QuantumClassicalHybrid()
        factors = sim.factor(15)
        print(f"Factored 15 = {factors[0]} × {factors[1]}")

        # Demo 2: Adaptive MPS
        print("\nDemo 2: Adaptive MPS (10 qubits)")
        print("-" * 60)
        from atlas_q import get_adaptive_mps
        mps_mod = get_adaptive_mps()
        mps = mps_mod['AdaptiveMPS'](10, bond_dim=8, device='cpu')
        print(f"Created MPS with {mps.num_qubits} qubits")
        print(f"Bond dimensions: {[t.shape for t in mps.tensors]}")

        # Demo 3: Hamiltonian
        print("\nDemo 3: Ising Hamiltonian")
        print("-" * 60)
        from atlas_q import get_mpo_ops
        mpo_mod = get_mpo_ops()
        H = mpo_mod['MPOBuilder'].ising_hamiltonian(n_sites=6, J=1.0, h=0.5, device='cpu')
        energy = mpo_mod['expectation_value'](H, mps)
        print(f"Built Ising Hamiltonian for 6 sites")
        print(f"Energy: {energy.real:.6f}")

        print("\n✅ Demo complete!")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='atlas-q',
        description='ATLAS-Q: GPU-Accelerated Quantum Tensor Network Simulator',
        epilog='For more information, visit: https://github.com/followthesapper/ATLAS-Q',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-v', '--version', action='version',
                       version=f'ATLAS-Q {__version__}')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Computation device (default: cuda)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Factor command
    parser_factor = subparsers.add_parser('factor', help='Factor an integer')
    parser_factor.add_argument('N', type=int, help='Integer to factor')
    parser_factor.set_defaults(func=cmd_factor)

    # Benchmark command
    parser_bench = subparsers.add_parser('benchmark', help='Run benchmarks')
    parser_bench.set_defaults(func=cmd_benchmark)

    # Info command
    parser_info = subparsers.add_parser('info', help='Show system information')
    parser_info.set_defaults(func=cmd_info)

    # Demo command
    parser_demo = subparsers.add_parser('demo', help='Run interactive demo')
    parser_demo.set_defaults(func=cmd_demo)

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command
    if not args.command:
        print_banner()
        parser.print_help()
        return 0

    # Run command
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
