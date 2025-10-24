#!/usr/bin/env python
"""
Benchmark script to compare different compression modes.
"""
import subprocess
import time
import sys
from pathlib import Path

def run_benchmark(rows, cols, depth, chi, entangler, mode_args, label):
    """Run a single benchmark configuration."""
    cmd = [
        'python', 'scripts/tn_grid8x8_shallow.py',
        '--rows', str(rows),
        '--cols', str(cols),
        '--depth', str(depth),
        '--chi', str(chi),
        '--entangler', entangler,
    ] + mode_args
    
    print(f"\n{'='*70}")
    print(f"Running: {label}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    # Parse output for peak chi and error
    output = result.stdout
    peak_chi = None
    avg_error = None
    
    for line in output.split('\n'):
        if 'Peak χ:' in line:
            peak_chi = int(line.split('Peak χ:')[1].split()[0])
        if 'Avg truncation error:' in line:
            avg_error = float(line.split('Avg truncation error:')[1].split()[0])
    
    return {
        'label': label,
        'time': elapsed,
        'peak_chi': peak_chi,
        'avg_error': avg_error,
        'output': output
    }

def main():
    print("\n" + "="*70)
    print("Tensor Network Simulator Benchmark Suite")
    print("="*70)
    
    # Test configurations
    configs = [
        (8, 8, 10, 512, 'cz'),   # Medium grid
        (10, 10, 12, 1024, 'cz'), # Large grid
    ]
    
    results = []
    
    for rows, cols, depth, chi, entangler in configs:
        print(f"\n\nConfiguration: {rows}×{cols} grid, depth={depth}, χ_max={chi}")
        print("="*70)
        
        # Test 1: Fixed truncation
        r1 = run_benchmark(rows, cols, depth, chi, entangler, [], "Fixed Truncation")
        results.append(r1)
        
        # Test 2: Adaptive truncation
        r2 = run_benchmark(rows, cols, depth, chi, entangler, 
                          ['--adaptive', '--tol', '1e-4'], "Adaptive Truncation")
        results.append(r2)
        
        # Test 3: AI-assisted (if model exists)
        model_path = Path('models/rank_predictor.pt')
        if model_path.exists():
            r3 = run_benchmark(rows, cols, depth, chi, entangler,
                              ['--ai-compression', '--adaptive', '--tol', '1e-4'],
                              "AI-Assisted Compression")
            results.append(r3)
    
    # Print summary
    print("\n\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Mode':<30} {'Time (s)':<12} {'Peak χ':<10} {'Avg Error':<12}")
    print("-"*70)
    
    for r in results:
        time_str = f"{r['time']:.2f}" if r['time'] else "N/A"
        chi_str = str(r['peak_chi']) if r['peak_chi'] else "N/A"
        error_str = f"{r['avg_error']:.2e}" if r['avg_error'] else "N/A"
        print(f"{r['label']:<30} {time_str:<12} {chi_str:<10} {error_str:<12}")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
