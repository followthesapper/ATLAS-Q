# Project Structure

```
quantum-hybrid-simulator/
│
├── 📚 Documentation
│   └── docs/
│       ├── README.md                          # Project overview & quick start
│       ├── AQED_WHITEPAPER.md                 # AQED technical documentation
│       ├── QUANTUM_SIMULATOR_WHITEPAPER.md    # Quantum simulator technical docs
│       ├── AQED_USAGE_GUIDE.md                # Practical AQED tutorials
│       ├── QUANTUM_SIMULATOR_USAGE_GUIDE.md   # Quantum simulation examples
│       ├── CONTRIBUTING.md                    # Contribution guidelines
│       ├── CHANGELOG.md                       # Version history
│       ├── LICENSE                            # MIT License
│       └── PROJECT_STRUCTURE.md               # This file
│
├── 📦 Source Code
│   └── src/quantum_hybrid_system/
│       ├── __init__.py                    # Package initialization
│       ├── quantum_hybrid_system.py       # Core quantum simulator (1535 lines)
│       ├── aqed/                          # AQED transformer module ⭐
│       │   ├── __init__.py
│       │   ├── config.py                  # AQEDConfig dataclass
│       │   └── model.py                   # AQEDTransformerLM
│       ├── hybrid_aqed_layer.py           # MPS-attention hybrid (advanced)
│       ├── lowrank_aqed_layer.py          # Low-rank AQED variant
│       ├── mps_*.py                       # MPS implementation modules
│       ├── entanglement_router.py         # Routing algorithms
│       ├── telemetry.py                   # Performance tracking
│       └── tools_qih/                     # Quantum-inspired ML tools
│           ├── tn_core.py                 # Tensor network backend
│           ├── ai_rank_predictor.py       # AI-assisted SVD
│           ├── qih_pat.py                 # Period-Aware Transformer
│           └── ...                        # (14 more modules)
│
├── 🧪 Tests (Consolidated)
│   └── tests/                             # Unified test suite (27 test files) ⭐
│       ├── conftest.py                    # Pytest configuration
│       ├── pytest.ini                     # Pytest settings
│       ├── README.md                      # Test documentation
│       ├── test_quantum_system.py         # Core quantum tests
│       ├── test_hybrid_aqed.py            # AQED tests
│       ├── test_mps_*.py                  # MPS tests
│       ├── test_period_finding.py         # Period-finding tests
│       └── ...                            # (22 more test files)
│
├── 🚀 Scripts
│   └── scripts/
│       └── train_aqed.py                  # Unified AQED training script ⭐
│
├── 📓 Examples
│   └── Notebooks/                         # 21 interactive Jupyter notebooks
│       ├── 01_getting_started.ipynb
│       ├── 02_period_finding_qft.ipynb
│       ├── 03_compressed_states.ipynb
│       └── ...
│
├── ⚡ Triton Kernels
│   ├── triton_kernels/                    # Quantum simulator kernels ⭐
│   │   ├── README.md                      # Purpose & documentation
│   │   ├── modpow.py                      # Modular exponentiation (period-finding)
│   │   ├── mps_ops.py                     # MPS tensor operations
│   │   ├── mps_complex.py                 # Complex MPS operations
│   │   └── linproj_bmm.py                 # Low-rank projection
│   │
│   └── transformers/triton_kernels/       # AQED transformer kernels ⭐
│       ├── README.md                      # Purpose & documentation
│       ├── fast_routing.py                # Token routing
│       ├── fused_mixer.py                 # Fused mixer ops
│       └── packed_attention.py            # Packed attention
│
├── 🏃 Transformers (Legacy)
│   └── transformers/                      # Legacy transformer implementations
│       ├── train_baseline_transformer_fast.py
│       ├── train_transformer_ultra_fast.py
│       └── triton_kernels/                # (see above)
│
├── 📊 Benchmark Results
│   └── runs/                              # Recent benchmark results (cleaned) ⭐
│       ├── README.md                      # Benchmark documentation
│       ├── aqed_benchmark/                # Tests 1-6 at L=4096
│       ├── aqed_L8192/                    # Tests 1-6 at L=8192
│       ├── benchmark_summary.csv          # High-level comparison
│       └── summary_*.csv/png              # Visualizations & comparisons
│
├── 📁 Archive
│   └── archive/                           # Root-level archived files ⭐
│       ├── README.md                      # Archive documentation
│       ├── Tests_old/                     # Old Tests/ directory (merged)
│       ├── profiling/                     # Old profiling data
│       ├── scripts/                       # Old benchmark scripts
│       ├── test_logs/                     # Historical test outputs
│       ├── diagrams/                      # Old diagrams (pre-v0.3.0)
│       ├── runs_experiments/              # Experimental run data (46MB)
│       │   ├── README.md                  # Experiments documentation
│       │   ├── ftdata/                    # Training datasets (12M)
│       │   ├── svd_logs/                  # SVD telemetry (32M)
│       │   └── ...                        # Alpha sweeps, diffusion, etc.
│       │
│       └── docs/archive/                  # Historical documentation
│           ├── README.md                  # Docs archive navigation
│           ├── old_status_docs/           # Status reports & summaries
│           ├── WHITEPAPER_OLD.md          # Old combined whitepaper
│           ├── USAGE_GUIDE_OLD.md         # Old usage guide
│           └── ...                        # Technical notes & plans
│
├── ⚙️ Configuration
│   ├── pyproject.toml                     # Package configuration
│   ├── requirements-dev.txt               # Development dependencies
│   ├── Makefile                           # Build automation
│   └── .gitignore                         # Git ignore rules
│
└── 🔧 Build & Temp
    ├── build/                             # Build artifacts
    ├── tmp/                               # Temporary files
    ├── __pycache__/                       # Python cache
    ├── .pytest_cache/                     # Pytest cache
    ├── venv/                              # Virtual environment
    └── .venv/                             # Alternative venv
```

## Key Changes (v0.3.0)

### ✅ Consolidated Test Suite
- **Merged** `Tests/` (23 files) + `tests/` (4 files) → `tests/` (27 files)
- **Archived** old `Tests/` → `archive/Tests_old/`
- **Standard** Python convention (lowercase `tests/`)

### ⭐ Clarified Triton Kernels
Two **separate** directories serving **different purposes**:

1. **`triton_kernels/`** (root) - Quantum simulator kernels
   - Modular exponentiation for period-finding
   - MPS tensor operations
   - Low-rank projection for AQED

2. **`transformers/triton_kernels/`** - AQED transformer kernels
   - Fast routing for hybrid attention
   - Fused mixer operations
   - Packed attention kernels

Both documented with purpose-specific READMEs.

## Directory Purposes

### For Users
- **`docs/` directory**: All documentation (whitepapers, guides, README)
- **`Notebooks/`**: Interactive examples (21 notebooks)
- **`scripts/train_aqed.py`**: Command-line training

### For Developers
- **`src/quantum_hybrid_system/`**: Main codebase
- **`tests/`**: Unified test suite (27 files)
- **`triton_kernels/`**: Quantum kernels
- **`transformers/triton_kernels/`**: AQED kernels

### For Researchers
- **Whitepapers**: Mathematical details & proofs (in `docs/`)
- **`docs/archive/`**: Historical experiments & documentation
- **`runs/`**: Recent benchmark results (Tests 1-6)
- **`archive/runs_experiments/`**: Historical experimental data (46MB)

## Version
Last updated: 2025-10-25 (v0.3.0)
