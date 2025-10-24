#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for AQED Speedup

Tests all optimization levels:
1. Baseline (standard PyTorch)
2. Ultra-fast (simplified mixer, compile, Flash Attention)
3. + Triton kernels
4. Full stack (everything)

Target: 6-15× speedup

Usage:
    python benchmark_speedup.py --quick     # Fast test
    python benchmark_speedup.py --full      # Full benchmark
"""

import argparse
import time
import pandas as pd
import torch
import sys
from pathlib import Path

# Check dependencies
def check_dependencies():
    print("=" * 60)
    print("DEPENDENCY CHECK")
    print("=" * 60)

    checks = {}

    # PyTorch
    checks['pytorch'] = torch.__version__
    checks['cuda_available'] = torch.cuda.is_available()
    if checks['cuda_available']:
        checks['cuda_version'] = torch.version.cuda
        checks['gpu_name'] = torch.cuda.get_device_name(0)

    # Flash Attention
    try:
        import flash_attn
        checks['flash_attn'] = flash_attn.__version__
    except ImportError:
        checks['flash_attn'] = 'Not installed'

    # Triton
    try:
        import triton
        checks['triton'] = triton.__version__
    except ImportError:
        checks['triton'] = 'Not installed'

    # torch.compile support
    if hasattr(torch, 'compile'):
        checks['torch_compile'] = 'Available'
    else:
        checks['torch_compile'] = 'Not available (PyTorch < 2.0)'

    for key, value in checks.items():
        print(f"  {key:20s}: {value}")

    print("=" * 60)
    return checks


def run_benchmark(config_name, train_script, args_list, csv_path):
    """Run a single benchmark configuration."""
    import subprocess

    print(f"\n{'='*60}")
    print(f"RUNNING: {config_name}")
    print(f"{'='*60}")

    cmd = ['python', train_script] + args_list
    print(f"Command: {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"❌ {config_name} FAILED")
        return None

    print(f"✓ {config_name} completed in {elapsed:.1f}s")

    # Read results
    try:
        df = pd.read_csv(csv_path)
        val_row = df[df['split'] == 'val'].iloc[-1]
        return {
            'config': config_name,
            'loss': val_row['loss'],
            'tok_per_s': val_row['tok_per_s'],
            'elapsed_s': elapsed,
        }
    except Exception as e:
        print(f"Warning: Could not read results: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test (small config)')
    parser.add_argument('--full', action='store_true', help='Full benchmark (L=2048)')
    parser.add_argument('--ultra-only', action='store_true', help='Only test ultra-fast')
    args = parser.parse_args()

    # Check dependencies first
    deps = check_dependencies()

    # Config
    if args.quick:
        seq_len = 512
        batch_size = 64
        train_batches = 500
        val_batches = 50
        epochs = 1
    elif args.full:
        seq_len = 2048
        batch_size = 16
        train_batches = 1500
        val_batches = 100
        epochs = 1
    else:
        # Default: medium test
        seq_len = 1024
        batch_size = 32
        train_batches = 1000
        val_batches = 100
        epochs = 1

    common_args = [
        f'--seq_len', str(seq_len),
        f'--batch_size', str(batch_size),
        f'--train_batches', str(train_batches),
        f'--val_batches', str(val_batches),
        f'--d_model', '512',
        f'--n_layers', '8',
        f'--n_heads', '8',
        f'--d_ff', '2048',
        f'--epochs', str(epochs),
    ]

    results = []

    # 1. Baseline
    if not args.ultra_only:
        print("\n" + "🔵 " * 30)
        print("BASELINE: Standard PyTorch Transformer")
        print("🔵 " * 30)

        result = run_benchmark(
            'Baseline',
            'train_baseline_transformer_fast.py',
            common_args + ['--log_csv', '../runs/bench_baseline.csv'],
            '../runs/bench_baseline.csv'
        )
        if result:
            results.append(result)

    # 2. Ultra-fast (no compile)
    print("\n" + "🟢 " * 30)
    print("ULTRA-FAST: Simplified mixer, no compile, no Flash")
    print("🟢 " * 30)

    result = run_benchmark(
        'Ultra (no compile)',
        'train_transformer_ultra_fast.py',
        common_args + [
            '--attn_keep_every', '4',
            '--no_flash',
            '--log_csv', '../runs/bench_ultra_nocompile.csv'
        ],
        '../runs/bench_ultra_nocompile.csv'
    )
    if result:
        results.append(result)

    # 3. Ultra-fast with compile
    if deps.get('torch_compile') == 'Available':
        print("\n" + "🟡 " * 30)
        print("ULTRA-FAST + COMPILE")
        print("🟡 " * 30)

        result = run_benchmark(
            'Ultra + compile',
            'train_transformer_ultra_fast.py',
            common_args + [
                '--attn_keep_every', '4',
                '--no_flash',
                '--compile',
                '--log_csv', '../runs/bench_ultra_compile.csv'
            ],
            '../runs/bench_ultra_compile.csv'
        )
        if result:
            results.append(result)

    # 4. Ultra-fast + Flash Attention
    if deps.get('flash_attn') != 'Not installed':
        print("\n" + "🟠 " * 30)
        print("ULTRA-FAST + FLASH ATTENTION")
        print("🟠 " * 30)

        result = run_benchmark(
            'Ultra + Flash',
            'train_transformer_ultra_fast.py',
            common_args + [
                '--attn_keep_every', '4',
                '--log_csv', '../runs/bench_ultra_flash.csv'
            ],
            '../runs/bench_ultra_flash.csv'
        )
        if result:
            results.append(result)

    # 5. Full stack (compile + Flash)
    if (deps.get('torch_compile') == 'Available' and
        deps.get('flash_attn') != 'Not installed'):
        print("\n" + "🔴 " * 30)
        print("FULL STACK: Compile + Flash")
        print("🔴 " * 30)

        result = run_benchmark(
            'Full stack',
            'train_transformer_ultra_fast.py',
            common_args + [
                '--attn_keep_every', '4',
                '--compile',
                '--log_csv', '../runs/bench_full_stack.csv'
            ],
            '../runs/bench_full_stack.csv'
        )
        if result:
            results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    if not results:
        print("❌ No results to show")
        return

    df = pd.DataFrame(results)

    # Calculate speedups relative to baseline
    if 'Baseline' in df['config'].values:
        baseline_tps = df[df['config'] == 'Baseline']['tok_per_s'].iloc[0]
        df['speedup'] = df['tok_per_s'] / baseline_tps
    else:
        # Use first result as baseline
        baseline_tps = df['tok_per_s'].iloc[0]
        df['speedup'] = df['tok_per_s'] / baseline_tps

    # Format output
    print(f"\n{'Config':<25} {'Loss':<10} {'Tok/s':<12} {'Speedup':<10}")
    print("-" * 80)

    for _, row in df.iterrows():
        speedup_str = f"{row['speedup']:.2f}×"
        if row['speedup'] >= 10.0:
            speedup_str += " 🎉🎉🎉"
        elif row['speedup'] >= 6.0:
            speedup_str += " 🎉🎉"
        elif row['speedup'] >= 3.0:
            speedup_str += " 🎉"
        elif row['speedup'] >= 1.5:
            speedup_str += " ✓"
        else:
            speedup_str += " ⚠️"

        print(f"{row['config']:<25} {row['loss']:<10.4f} {row['tok_per_s']:<12.0f} {speedup_str}")

    # Save results
    df.to_csv('../runs/benchmark_summary.csv', index=False)
    print(f"\nResults saved to: runs/benchmark_summary.csv")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    best_speedup = df['speedup'].max()

    if best_speedup >= 10.0:
        print("🎉🎉🎉 MISSION ACCOMPLISHED! 10× speedup achieved!")
        print(f"     Best config: {df.loc[df['speedup'].idxmax(), 'config']}")
    elif best_speedup >= 6.0:
        print(f"🎉🎉 Excellent! {best_speedup:.1f}× speedup achieved")
        print("     To reach 10×, try:")
        print("     1. Longer sequences (--seq_len 4096)")
        print("     2. More aggressive attention skipping (--attn_keep_every 8)")
        print("     3. Enable Triton kernels (if available)")
    elif best_speedup >= 3.0:
        print(f"🎉 Good progress! {best_speedup:.1f}× speedup")
        print("     To improve:")
        if deps.get('torch_compile') != 'Available':
            print("     1. Install PyTorch 2.0+ for torch.compile")
        if deps.get('flash_attn') == 'Not installed':
            print("     2. Install Flash Attention: pip install flash-attn")
        print("     3. Use longer sequences (--seq_len 2048 or 4096)")
    else:
        print(f"⚠️  Speedup only {best_speedup:.1f}×. Troubleshooting:")
        print("     1. Check that CUDA is available")
        print("     2. Try longer sequences (current: L={seq_len})")
        print("     3. Profile to find bottlenecks")
        print("     4. Ensure GPU is not throttled (check nvidia-smi)")

    print("\nFor detailed analysis, see: runs/benchmark_summary.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
