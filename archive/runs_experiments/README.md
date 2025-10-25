# Archived Experimental Runs

Historical experimental results from AQED development and optimization.

## Contents

### Large Datasets
- **`ftdata/`** (12M) - Training dataset for feature extraction experiments
- **`svd_logs/`** (32M) - SVD decomposition telemetry (76 compressed log files)

### Experiments by Category

#### Alpha/Controller Sweeps
- `aqed_alpha_sweep*.csv/png` - Hyperparameter sweeps for α parameter
- `aqed_controller_*.csv` - Controller behavior experiments
- `aqed_controller_knobs.png` - Visualization of control parameters

#### Hybrid Attention Variants
- `aqed_hybrid_*.csv` - Hybrid attention routing experiments
- `aqed_routed_*.csv` - Token routing optimization tests
- Various k-factor experiments (k2, k3)

#### Diffusion Experiments
- `diffusion_*.png` (20+ files) - Quantum diffusion visualizations
- `diffusion_*.csv` - Diffusion experiment data
- Grid size experiments: 4×4, 6×6, 8×8, 10×10, 12×12
- Heisenberg model experiments

#### Phase 3 Grid Search
- `p3_*.log` - Phase 3 hyperparameter grid search results
- `p3_coarse_leaderboard.txt` - Best configurations from grid search
- Gamma (G) and Beta (B) parameter sweeps

#### Baseline Comparisons
- `baseline_*.csv` - Historical baseline transformer benchmarks
- `ultra_*.csv` - Ultra-optimized baseline variants
- `bench_ultra_*.csv` - Compilation experiments

### Other Files
- `test_logs/` - Historical test execution logs
- `archive/` - Double-archived files from earlier cleanup
- `real_vs_synthetic_comparison.png` - Dataset comparison
- `power_samples.csv` - Power consumption measurements
- `qubit_probe_*.log` - Quantum simulation probe experiments

## Timeline

Most experiments conducted: **October 22-24, 2025**

These experiments led to the final AQED architecture documented in v0.3.0.

## Archived Date

**2025-10-25** - During documentation consolidation (v0.3.0)

## Notes

- All experiments preserved for historical reference
- Production results are in `runs/` directory
- Final architecture selections documented in `AQED_WHITEPAPER.md`
- Key findings:
  - AQED v1 (standard) outperforms LowRank and Hybrid variants
  - Optimal skip parameter: `attn_keep_every=8` for L=4096
  - torch.compile + SDPA provides best performance/compatibility balance

## Accessing Archived Data

If you need to review specific experiments:

```bash
# View SVD logs
zcat archive/runs_experiments/svd_logs/svd_events_*.jsonl.gz | head

# Review grid search results
cat archive/runs_experiments/p3_coarse_leaderboard.txt

# Compare diffusion experiments
ls archive/runs_experiments/diffusion_*.png
```
