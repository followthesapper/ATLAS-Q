#!/usr/bin/env python
"""Analyze collected SVD data."""
import gzip, json
from pathlib import Path
import numpy as np

def main():
    log_dir = Path("runs/svd_logs")
    log_files = list(log_dir.glob("*.jsonl.gz"))
    
    if not log_files:
        print("❌ No log files found.")
        return
    
    print(f"📊 Analyzing {len(log_files)} log files...\n")
    
    all_spectra = []
    all_adaptive_keeps = []
    all_chi_maxes = []
    all_tols = []
    all_entanglers = {}
    corrupted = 0
    
    for log_file in log_files:
        try:
            with gzip.open(log_file, 'rt') as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if event.get('kind') == 'svd_spectrum':
                            spectrum = np.array(event['S_top128'])
                            # Remove padding zeros
                            spectrum = spectrum[spectrum > 0]
                            if len(spectrum) > 0:
                                all_spectra.append(spectrum)
                                all_adaptive_keeps.append(event['adaptive_keep'])
                                all_chi_maxes.append(event.get('chi_max', 1024))
                                all_tols.append(event.get('tol', 1e-4))
                                
                                ent = event.get('entangler', 'unknown')
                                all_entanglers[ent] = all_entanglers.get(ent, 0) + 1
                    except json.JSONDecodeError:
                        continue
        except (EOFError, gzip.BadGzipFile):
            corrupted += 1
            continue
    
    if corrupted > 0:
        print(f"⚠️  Skipped {corrupted} corrupted files\n")
    
    if not all_spectra:
        print("❌ No valid SVD events found.")
        return
    
    print(f"✅ Loaded {len(all_spectra):,} SVD events\n")
    
    # Statistics
    print("="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    print(f"\n📈 Entangler distribution:")
    for ent, count in sorted(all_entanglers.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(all_spectra)
        print(f"  {ent:10s}: {count:6,} events ({pct:5.1f}%)")
    
    spectrum_sizes = [len(s) for s in all_spectra]
    print(f"\n📏 Spectrum sizes (# of singular values):")
    print(f"  Min:    {min(spectrum_sizes):4d}")
    print(f"  Max:    {max(spectrum_sizes):4d}")
    print(f"  Mean:   {np.mean(spectrum_sizes):6.1f}")
    print(f"  Median: {int(np.median(spectrum_sizes)):4d}")
    print(f"  Std:    {np.std(spectrum_sizes):6.1f}")
    
    print(f"\n✂️  Adaptive truncation choices:")
    print(f"  Min:    {min(all_adaptive_keeps):4d}")
    print(f"  Max:    {max(all_adaptive_keeps):4d}")
    print(f"  Mean:   {np.mean(all_adaptive_keeps):6.1f}")
    print(f"  Median: {int(np.median(all_adaptive_keeps)):4d}")
    print(f"  Std:    {np.std(all_adaptive_keeps):6.1f}")
    
    # Decay rates
    decay_rates = []
    for spectrum in all_spectra:
        if len(spectrum) >= 10:
            if spectrum[0] > 0 and spectrum[9] > 0:
                decay = -np.log(spectrum[9] / spectrum[0]) / 9
                decay_rates.append(decay)
    
    if decay_rates:
        print(f"\n📉 Spectral decay rates (λ in exp(-λx)):")
        print(f"  Min:    {min(decay_rates):6.4f}")
        print(f"  Max:    {max(decay_rates):6.4f}")
        print(f"  Mean:   {np.mean(decay_rates):6.4f}")
        print(f"  Median: {np.median(decay_rates):6.4f}")
        print(f"  Std:    {np.std(decay_rates):6.4f}")
    
    # Truncation fractions
    fractions = []
    for keep, spectrum in zip(all_adaptive_keeps, all_spectra):
        frac = keep / len(spectrum)
        fractions.append(frac)
    
    print(f"\n🎯 Truncation fractions (keep/total):")
    print(f"  Min:    {min(fractions):.4f}")
    print(f"  Max:    {max(fractions):.4f}")
    print(f"  Mean:   {np.mean(fractions):.4f}")
    print(f"  Median: {np.median(fractions):.4f}")
    print(f"  Std:    {np.std(fractions):.4f}")
    
    # Distribution of fractions
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    hist, _ = np.histogram(fractions, bins=bins)
    print(f"\n📊 Fraction distribution:")
    for i in range(len(bins)-1):
        pct = 100 * hist[i] / len(fractions)
        bar = '█' * int(pct / 2)
        print(f"  {bins[i]:.2f}-{bins[i+1]:.2f}: {hist[i]:6,} ({pct:5.1f}%) {bar}")
    
    # Tolerance distribution
    unique_tols = sorted(set(all_tols))
    print(f"\n⚙️  Tolerance values used:")
    for tol in unique_tols:
        count = sum(1 for t in all_tols if t == tol)
        pct = 100 * count / len(all_tols)
        print(f"  {tol:.6f}: {count:6,} events ({pct:5.1f}%)")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Dataset size: {len(all_spectra):,} real SVD events")
    print(f"   • Spectrum sizes: {min(spectrum_sizes)}-{max(spectrum_sizes)} SVs")
    print(f"   • Mean truncation: {np.mean(fractions):.1%} of SVs kept")
    if decay_rates:
        print(f"   • Decay rates: {np.mean(decay_rates):.3f} ± {np.std(decay_rates):.3f}")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. This is REAL quantum circuit data!")
    print(f"   2. Use it to retrain AI predictor on actual spectra")
    print(f"   3. Compare real vs synthetic training data distribution")
    print(f"   4. Publish results showing AI learns from real circuits")

if __name__ == "__main__":
    main()
