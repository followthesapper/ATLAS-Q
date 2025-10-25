# Archive Directory

This directory contains historical files from the development process.

## Structure

```
archive/
├── profiling/          # Profiling data and scripts
│   ├── trace_lowrank.json
│   └── profile_lowrank.py
├── scripts/            # Old benchmark and utility scripts
│   ├── benchmark_aqed_vs_baseline.sh
│   ├── benchmark_L8192.sh
│   ├── run_training.sh
│   ├── analyze_results.py
│   ├── check_env_and_setup.py
│   └── run_tests.py
└── test_logs/          # Historical test output logs
    ├── test5_chi8.log
    ├── test5_final.log
    ├── test5_output.log
    └── benchmark_output.log
```

## Purpose

These files are preserved for:
- Historical reference
- Debugging past issues
- Understanding development evolution
- Reproducing old experiments

## Current Tools

For current benchmarking and testing, see:
- **Scripts:** `scripts/` directory (main training/benchmark scripts)
- **Tests:** `tests/` directory (pytest test suite)
- **Profiling:** See usage guides for current profiling methods

## Cleanup

These files were moved from the root directory on 2025-10-25 during the documentation reorganization (v0.3.0).
