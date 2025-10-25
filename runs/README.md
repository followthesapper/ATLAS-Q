# Benchmark Results

This directory contains recent benchmark results for AQED transformer performance evaluation.

## Key Results

### Tests 1-6: Comprehensive Benchmark (Oct 24, 2025)

Recent systematic comparison of AQED variants vs baseline transformers at L=4096 and L=8192.

**Directories:**
- `aqed_benchmark/` - Tests 1-6 results at L=4096
  - Test 1: Baseline (no optimization)
  - Test 2: Baseline (optimized)
  - Test 3: AQED v1 (no optimization)
  - Test 4: AQED v1 (optimized) ⭐ **Winner**
  - Test 5: AQED LowRank (no optimization)
  - Test 6: AQED LowRank (optimized)

- `aqed_L8192/` - Tests 1-6 results at L=8192
  - Same configuration as L=4096
  - Tests extreme scaling behavior

### Summary Files

**CSV Results:**
- `benchmark_summary.csv` - High-level comparison
- `summary_compare.csv` - Detailed comparison table
- `summary_compare_with_rel.csv` - With relative speedups

**Visualizations:**
- `summary_tokps.png` - Tokens/sec comparison
- `summary_loss.png` - Loss quality comparison
- `summary_pareto.png` - Speed vs loss Pareto frontier
- `summary_speed_vs_loss.png` - Speed/loss tradeoff
- `summary_mem_trend.png` - Memory usage trends

## Key Findings

From the benchmark results:

### L=4096 Performance
- **AQED v1 (optimized)**: 6.18× speedup over baseline
  - 211k tokens/sec vs 34k tokens/sec
  - Loss difference: Δloss ≤ 0.02

### L=8192 Performance
- **AQED v1 (optimized)**: 10.59× speedup over baseline
  - 197k tokens/sec vs 19k tokens/sec
  - Maintained quality with extreme sequence lengths

### Architecture Insights
- **LowRank AQED**: Showed architectural bottlenecks
  - Dual-path attention caused kernel launch overhead
  - GPU utilization: 10-50% (sawtooth pattern)
  - Not recommended for production use

## Archived Results

Historical experimental results have been moved to:
- `archive/runs_experiments/`

This includes:
- Alpha sweep experiments
- Hybrid routing variants
- Phase 3 grid searches
- Diffusion experiments
- Large training datasets (ftdata, svd_logs)

## Reproducing Results

To reproduce these benchmarks:

```bash
# Run single test
python scripts/train_aqed.py --config aqed_benchmark --seq_len 4096

# Run full benchmark suite (Tests 1-6)
make benchmark  # See Makefile for details
```

## Version

Results correspond to project version **0.3.0** (2025-10-25).

See `CHANGELOG.md` for details on AQED implementation and optimizations.
