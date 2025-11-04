"""
Profile ATLAS-Q adapter to identify performance bottlenecks
"""

import cProfile
import pstats
from io import StringIO
import time

from qiskit import QuantumCircuit
from qiskit_aer import Aer
from atlas_q.adapters import ATLASQBackend

print("=" * 80)
print("ATLAS-Q Adapter Profiling")
print("=" * 80)
print()

# Create Bell state circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Profile ATLAS-Q execution
print("[1] Profiling ATLAS-Q stabilizer backend (Bell state, 1000 shots)...")
print("-" * 80)

atlas_backend = ATLASQBackend(enable_stabilizer=True)

profiler = cProfile.Profile()
profiler.enable()

start = time.time()
atlas_job = atlas_backend.run(qc, shots=1000)
atlas_result = atlas_job.result()
atlas_time = time.time() - start

profiler.disable()

print(f"Total time: {atlas_time*1000:.2f} ms")
print()

# Print top time consumers
s = StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(20)  # Top 20 functions
print(s.getvalue())

print()
print("=" * 80)
print("Key findings to analyze:")
print("  - Look for deepcopy, measure, or gate application calls")
print("  - Identify if bottleneck is in circuit building or sampling")
print("=" * 80)
