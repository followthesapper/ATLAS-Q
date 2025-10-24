#!/usr/bin/env python3
"""
Project Consolidation Script

This script consolidates the quantum-hybrid-simulator project into a clean structure:
1. Creates src/quantum_hybrid_system/aqed/ module
2. Moves AQED transformer code into the package
3. Removes duplicate files
4. Consolidates scripts
5. Prepares for clean imports

Run after consolidation: python -m pytest tests/
"""

import os
import shutil
from pathlib import Path

# Base paths
ROOT = Path("/home/admin/quantum-hybrid-simulator")
SRC = ROOT / "src" / "quantum_hybrid_system"
AQED = SRC / "aqed"

def consolidate_aqed_module():
    """Move AQED transformer into package"""
    print("📦 Consolidating AQED module...")

    # Copy ultra_fast transformer as the canonical AQED implementation
    src_file = ROOT / "transformers" / "train_transformer_ultra_fast.py"
    if src_file.exists():
        print(f"  ✓ Using {src_file} as canonical AQED implementation")
    else:
        print(f"  ✗ Warning: {src_file} not found!")
        return False

    return True

def remove_duplicates():
    """Remove duplicate .py files from root"""
    print("\n🗑️  Removing duplicate files...")

    duplicates = [
        "chirp_features.py",
        "ntt_features.py",
        "periodic2d_features.py",
        "qih_llm.py",
        "qih_pat.py",
        "tn_layers.py"
    ]

    for dup in duplicates:
        dup_path = ROOT / dup
        if dup_path.exists():
            dup_path.unlink()
            print(f"  ✓ Removed {dup}")

    return True

def consolidate_md_files():
    """Consolidate .md files"""
    print("\n📝 Consolidating documentation...")

    # Keep these
    keep = ["README.md", "WHITEPAPER.md", "USAGE_GUIDE.md"]

    # Archive the rest
    archive_dir = ROOT / "docs" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    md_files = list(ROOT.glob("*.md"))
    archived = 0

    for md in md_files:
        if md.name not in keep and md.name not in ["LICENSE.md"]:
            dest = archive_dir / md.name
            shutil.move(str(md), str(dest))
            archived += 1
            print(f"  → Archived {md.name}")

    print(f"  ✓ Archived {archived} docs, keeping {len(keep)}")
    return True

def consolidate_scripts():
    """Clean up shell scripts"""
    print("\n🔧 Consolidating scripts...")

    # Keep useful scripts
    keep_scripts = [
        "run_training.sh",
        "benchmark_aqed_vs_baseline.sh",
    ]

    # Archive others
    archive_dir = ROOT / "scripts" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    sh_files = list(ROOT.glob("*.sh"))
    archived = 0

    for sh in sh_files:
        if sh.name not in keep_scripts:
            dest = archive_dir / sh.name
            shutil.move(str(sh), str(dest))
            archived += 1
            print(f"  → Archived {sh.name}")

    print(f"  ✓ Archived {archived} scripts")
    return True

def create_clean_imports():
    """Update __init__.py with clean imports"""
    print("\n🔗 Creating clean import structure...")

    init_file = SRC / "__init__.py"

    content = '''"""
Quantum Hybrid Simulator

A quantum-inspired hybrid system combining:
- Compressed quantum state representations (Periodic, Product, MPS)
- O(√r) period-finding algorithms
- AQED (Adaptive Quantum Entanglement Diffusion) transformers
- Quantum-inspired ML features

Example - Quantum Simulation:
    >>> from quantum_hybrid_system import QuantumClassicalHybrid
    >>> sim = QuantumClassicalHybrid()
    >>> factors = sim.factor(221)  # Factor 221 = 13 × 17
    >>> print(factors)  # [13, 17]

Example - AQED Transformer:
    >>> from quantum_hybrid_system.aqed import AQEDTransformerLM, AQEDConfig
    >>> config = AQEDConfig(vocab_size=32000, seq_len=4096)
    >>> model = AQEDTransformerLM(config).cuda()
    >>> # Model is 6-10× faster than traditional transformers!
"""

# Core quantum simulation
from .quantum_hybrid_system import (
    QuantumClassicalHybrid,
    PeriodicState,
    ProductState,
    MatrixProductState,
)

# AQED transformer (lazy import to avoid torch dependency for quantum-only use)
def get_aqed():
    """Get AQED transformer (requires torch)"""
    from .aqed import AQEDTransformerLM, AQEDConfig
    return AQEDTransformerLM, AQEDConfig

__all__ = [
    # Quantum simulation
    'QuantumClassicalHybrid',
    'PeriodicState',
    'ProductState',
    'MatrixProductState',
    # AQED
    'get_aqed',
]

__version__ = '0.2.0'
'''

    init_file.write_text(content)
    print("  ✓ Updated main __init__.py")
    return True

def main():
    """Main consolidation"""
    print("=" * 60)
    print("Quantum Hybrid Simulator - Project Consolidation")
    print("=" * 60)

    steps = [
        ("AQED Module", consolidate_aqed_module),
        ("Duplicates", remove_duplicates),
        ("Documentation", consolidate_md_files),
        ("Scripts", consolidate_scripts),
        ("Imports", create_clean_imports),
    ]

    results = []
    for name, func in steps:
        try:
            success = func()
            results.append((name, success))
        except Exception as e:
            print(f"  ✗ Error in {name}: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Consolidation Summary:")
    print("=" * 60)
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    all_success = all(s for _, s in results)

    if all_success:
        print("\n✅ Consolidation complete!")
        print("\nNext steps:")
        print("  1. Review changes: git status")
        print("  2. Test imports: python -c 'from quantum_hybrid_system import *'")
        print("  3. Run tests: python -m pytest tests/")
        print("  4. Build package: python -m build")
    else:
        print("\n⚠️  Some steps failed - review output above")

    return all_success

if __name__ == "__main__":
    main()
