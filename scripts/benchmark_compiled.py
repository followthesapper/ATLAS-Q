#!/usr/bin/env python
"""Benchmark compiled vs uncompiled model."""
import sys, time
from pathlib import Path
import torch
import importlib.util

# Load modules
tn_core_path = Path(__file__).parent.parent / 'src' / 'atlas_q' / 'tools_qih' / 'tn_core.py'
spec = importlib.util.spec_from_file_location("tn_core", tn_core_path)
tn_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tn_core)

finetuned_path = Path(__file__).parent.parent / 'src' / 'atlas_q' / 'tools_qih' / 'finetuned_predictor.py'
spec_ft = importlib.util.spec_from_file_location("finetuned_predictor", finetuned_path)
finetuned_module = importlib.util.module_from_spec(spec_ft)
spec_ft.loader.exec_module(finetuned_module)

FinetunedPredictorWrapper = finetuned_module.FinetunedPredictorWrapper

def run_test(n_qubits, depth, chi_max, model_path, label):
    """Run simulation with given model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.complex64
    
    predictor = FinetunedPredictorWrapper(model_path=str(model_path), device=device)
    cores = tn_core.mps_init_plus(n_qubits, device=device, dtype=dtype)
    
    t0 = time.time()
    peak_chi = 1
    
    for layer in range(depth):
        theta = 0.3 + 0.1 * layer
        U = tn_core.build_entangler('cz', theta, device, dtype)
        
        for i in range(0, n_qubits-1, 2):
            cores, chi, _ = tn_core.mps_apply_2q(
                cores, i, i+1, U, chi_max=chi_max,
                svd_driver='gesvda', adaptive=True, tol=1e-4,
                ai_predictor=predictor
            )
            peak_chi = max(peak_chi, chi)
    
    elapsed = time.time() - t0
    
    print(f"{label:20s}: {elapsed:6.2f}s (peak χ={peak_chi})")
    return elapsed

print("="*70)
print("Compiled vs Uncompiled Model Benchmark")
print("="*70)
print(f"\nTest: 8×8 grid, depth=15, χ_max=128\n")

# Test uncompiled
t_uncompiled = run_test(64, 15, 128, 
                        "models/rank_predictor_ft.pt",
                        "Uncompiled")

# Test compiled
if Path("models/rank_predictor_ft_compiled.pt").exists():
    t_compiled = run_test(64, 15, 128,
                          "models/rank_predictor_ft_compiled.pt", 
                          "Compiled")
    
    speedup = t_uncompiled / t_compiled
    print(f"\n{'='*70}")
    print(f"Speedup: {speedup:.2f}×")
    print(f"Time saved: {t_uncompiled - t_compiled:.2f}s")
else:
    print("\n⚠️  Compiled model not found")

print("="*70)
