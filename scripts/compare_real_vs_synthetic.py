#!/usr/bin/env python
"""Compare real circuit data vs synthetic training data."""
import gzip, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Load real data
print("Loading real circuit data...")
log_dir = Path("runs/svd_logs")
real_decay_rates = []
real_fractions = []

for log_file in log_dir.glob("*.jsonl.gz"):
    try:
        with gzip.open(log_file, 'rt') as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if event.get('kind') == 'svd_spectrum':
                        spectrum = np.array(event['S_top128'])
                        spectrum = spectrum[spectrum > 0]
                        
                        if len(spectrum) >= 10:
                            # Decay rate
                            if spectrum[0] > 0 and spectrum[9] > 0:
                                decay = -np.log(spectrum[9] / spectrum[0]) / 9
                                real_decay_rates.append(decay)
                            
                            # Truncation fraction
                            keep = event['adaptive_keep']
                            frac = min(keep / len(spectrum), 1.0)  # Cap at 1.0
                            real_fractions.append(frac)
                except:
                    continue
    except:
        continue

print(f"✅ Loaded {len(real_decay_rates):,} real events")

# Generate synthetic data (matching training distribution)
print("Generating synthetic comparison data...")
n_synthetic = len(real_decay_rates)
synthetic_decay_rates = np.random.uniform(0.3, 5.0, n_synthetic)
synthetic_fractions = []

for decay in synthetic_decay_rates:
    # Simulate spectrum
    rank = np.random.randint(20, 500)
    spectrum = np.exp(-decay * np.arange(rank) / rank)
    
    # Compute fraction for random tolerance
    tol = np.random.choice([1e-3, 1e-4, 1e-5])
    cumsum = np.cumsum(spectrum**2)
    total = cumsum[-1]
    keep_idx = np.where(cumsum >= (1-tol)*total)[0]
    if len(keep_idx) > 0:
        frac = (keep_idx[0] + 1) / rank
    else:
        frac = 1.0
    synthetic_fractions.append(min(frac, 1.0))

# Create comparison plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Decay rates
ax = axes[0]
bins = np.linspace(0, 2, 50)
ax.hist(real_decay_rates, bins=bins, alpha=0.6, label='Real Circuits', density=True, color='blue')
ax.hist(synthetic_decay_rates, bins=bins, alpha=0.6, label='Synthetic Training', density=True, color='orange')
ax.set_xlabel('Spectral Decay Rate (λ)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Decay Rate Distribution: Real vs Synthetic', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: Truncation fractions
ax = axes[1]
bins = np.linspace(0, 1, 30)
ax.hist(real_fractions, bins=bins, alpha=0.6, label='Real Circuits', density=True, color='blue')
ax.hist(synthetic_fractions, bins=bins, alpha=0.6, label='Synthetic Training', density=True, color='orange')
ax.set_xlabel('Truncation Fraction (keep/total)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Truncation Distribution: Real vs Synthetic', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('runs/real_vs_synthetic_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved comparison plot: runs/real_vs_synthetic_comparison.png")

# Statistics
print("\n" + "="*60)
print("COMPARISON STATISTICS")
print("="*60)
print(f"\nDecay Rates:")
print(f"  Real:      μ={np.mean(real_decay_rates):.3f}, σ={np.std(real_decay_rates):.3f}")
print(f"  Synthetic: μ={np.mean(synthetic_decay_rates):.3f}, σ={np.std(synthetic_decay_rates):.3f}")

print(f"\nTruncation Fractions:")
print(f"  Real:      μ={np.mean(real_fractions):.3f}, σ={np.std(real_fractions):.3f}")
print(f"  Synthetic: μ={np.mean(synthetic_fractions):.3f}, σ={np.std(synthetic_fractions):.3f}")

print("\n💡 KEY INSIGHT:")
if np.mean(real_decay_rates) < 1.0:
    print("   Real circuits have SLOWER decay than synthetic training data!")
    print("   → AI trained on synthetic may be suboptimal for real circuits")
    print("   → Retraining on real data could improve performance")
else:
    print("   Real and synthetic distributions are similar")
    print("   → Current AI training is well-calibrated")

plt.show()
