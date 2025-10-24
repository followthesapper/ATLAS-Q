"""Optimize AI model for faster inference."""
import torch
import torch.nn as nn
from pathlib import Path

# Define model inline to avoid import issues
class FinetunedMLP(nn.Module):
    def __init__(self, d_in=135, d_h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(),
            nn.Linear(d_h, d_h), nn.ReLU(),
            nn.Linear(d_h, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

# Load model
model = FinetunedMLP()
model.load_state_dict(torch.load("models/rank_predictor_ft.pt"))
model.eval()

# Compile with TorchScript
print("Compiling model with TorchScript...")
scripted = torch.jit.script(model)
scripted.save("models/rank_predictor_ft_compiled.pt")

print("✅ Compiled model saved to models/rank_predictor_ft_compiled.pt")
print("   Expected speedup: 2-3×")

# Quick test
import numpy as np
x_test = torch.randn(1, 135)

# Warmup
for _ in range(10):
    _ = model(x_test)
    _ = scripted(x_test)

# Benchmark
import time

n_iters = 1000
t0 = time.time()
for _ in range(n_iters):
    _ = model(x_test)
t_uncompiled = time.time() - t0

t0 = time.time()
for _ in range(n_iters):
    _ = scripted(x_test)
t_compiled = time.time() - t0

speedup = t_uncompiled / t_compiled
print(f"\nMicro-benchmark ({n_iters} iterations):")
print(f"  Uncompiled: {t_uncompiled*1000:.2f}ms")
print(f"  Compiled:   {t_compiled*1000:.2f}ms")
print(f"  Speedup:    {speedup:.2f}×")
