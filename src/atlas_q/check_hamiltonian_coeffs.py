#!/usr/bin/env python3
"""
Diagnostic script for molecular Hamiltonian coefficients.
"""

import sys

import numpy as np

sys.path.insert(0, '/home/admin/ATLAS-Q/src')

from atlas_q.mpo_ops import MPOBuilder, _jordan_wigner_transform

molecule = sys.argv[1] if len(sys.argv) > 1 else "H2"

print(f"Checking coefficients for {molecule}...")
from pyscf import ao2mo, gto, scf

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
mf = scf.RHF(mol); mf.kernel()
h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri = ao2mo.kernel(mol, mf.mo_coeff)
h2 = ao2mo.restore(1, eri, h1.shape[0])
e_nuc = mol.energy_nuc()

terms = _jordan_wigner_transform(h1, h2, e_nuc)
coeffs = np.array([abs(c) for c in terms.values()])
print(f"Num terms: {len(coeffs)}")
print(f"max={coeffs.max():.3e}, mean={coeffs.mean():.3e}, std={coeffs.std():.3e}")

import matplotlib.pyplot as plt

plt.hist(np.log10(coeffs[coeffs>1e-15]), bins=80)
plt.xlabel("log10(|coeff|)")
plt.ylabel("count")
plt.title(f"Hamiltonian coefficient magnitudes: {molecule}")
plt.show()
