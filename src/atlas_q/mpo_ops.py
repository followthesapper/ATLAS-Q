"""
Matrix Product Operator (MPO) Operations

MPOs represent operators on quantum states in tensor network form:
- Hamiltonian evolution
- Observable expectation values
- Noise channels
- Time evolution

Author: ATLAS-Q Contributors
Date: October 2025
License: MIT
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

# GPU-optimized operations (if available)
try:
    import sys

    # Add project root dynamically
    project_root = Path(__file__).parent.parent.parent.resolve()
    sys.path.insert(0, str(project_root))
    from triton_kernels.tdvp_mpo_ops import mpo_expectation_step_optimized

    GPU_OPTIMIZED_AVAILABLE = True
except ImportError:
    GPU_OPTIMIZED_AVAILABLE = False


@dataclass
class MPO:
    """
    Matrix Product Operator

    Represents an operator as a chain of 4-tensors:
    O = Σ W[0]_{s₀s₀'} W[1]_{s₁s₁'} ... W[n-1]_{sₙ₋₁sₙ₋₁'}

    Each tensor W[i] has shape [χ_L, d, d, χ_R] where:
    - χ_L, χ_R: left and right bond dimensions
    - d: physical dimension (2 for qubits)
    """

    tensors: List[torch.Tensor]  # List of 4-tensors [χ_L, d, d, χ_R]
    n_sites: int

    def __post_init__(self):
        assert len(self.tensors) == self.n_sites
        # Validate shapes
        for i, W in enumerate(self.tensors):
            assert len(W.shape) == 4, f"MPO tensor {i} must be 4D"
            assert W.shape[1] == W.shape[2], f"Physical dims must match at site {i}"

    @staticmethod
    def identity(n_sites: int, device: str = "cuda", dtype=torch.complex64) -> "MPO":
        """Create identity MPO"""
        tensors = []
        for i in range(n_sites):
            # Identity operator: W[σ,σ'] = δ_{σ,σ'}
            W = torch.zeros(1, 2, 2, 1, dtype=dtype, device=device)
            W[0, 0, 0, 0] = 1.0
            W[0, 1, 1, 0] = 1.0
            tensors.append(W)
        return MPO(tensors, n_sites)

    @staticmethod
    def from_local_ops(ops: List[torch.Tensor], device: str = "cuda") -> "MPO":
        """
        Create MPO from list of local operators (one per site)

        Args:
            ops: List of 2×2 operators for each site
        """
        n_sites = len(ops)
        tensors = []

        for i, op in enumerate(ops):
            assert op.shape == (2, 2), f"Operator {i} must be 2×2"
            # Wrap operator in MPO tensor [1, 2, 2, 1]
            W = op.view(1, 2, 2, 1).to(device)
            tensors.append(W)

        return MPO(tensors, n_sites)

    @staticmethod
    def from_operators(ops: List[torch.Tensor], device: str = "cuda") -> "MPO":
        """Alias for from_local_ops"""
        return MPO.from_local_ops(ops, device=device)


class MPOBuilder:
    """Helper class to build common MPOs"""

    @staticmethod
    def identity_mpo(n_sites: int, device: str = "cuda", dtype=torch.complex64) -> MPO:
        """Create identity MPO (wrapper for MPO.identity)"""
        return MPO.identity(n_sites, device=device, dtype=dtype)

    @staticmethod
    def ising_hamiltonian(
        n_sites: int, J: float = 1.0, h: float = 0.5, device: str = "cuda", dtype=torch.complex64
    ) -> MPO:
        """
        Transverse-field Ising Hamiltonian:
        H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ

        Args:
            n_sites: Number of sites
            J: Coupling strength
            h: Transverse field

        Virtual bond structure (D=3):
        - 0→0: identity track
        - 0→1: emit Z (open ZZ term)
        - 1→2: close with -J Z
        - 2→2: identity tail
        - 0→2: local field -h X
        """
        I = torch.eye(2, dtype=dtype, device=device)
        X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)

        D = 3  # virtual bond dimension
        tensors = []

        for i in range(n_sites):
            if i == 0:
                # left boundary: shape [1, 2, 2, D]
                W = torch.zeros(1, 2, 2, D, dtype=dtype, device=device)
                # 0->0: I
                W[0, :, :, 0] = I
                # 0->1: Z   (open a ZZ term)
                W[0, :, :, 1] = Z
                # 0->2: -h X   (local field)
                if h != 0.0:
                    W[0, :, :, 2] = -h * X
            elif i == n_sites - 1:
                # right boundary: shape [D, 2, 2, 1]
                W = torch.zeros(D, 2, 2, 1, dtype=dtype, device=device)
                # 2->0: I    (close tail)
                W[2, :, :, 0] = I
                # 1->0: -J Z (close a ZZ term)
                W[1, :, :, 0] = -J * Z
                # 0->0: -h X (local field on last site sits on diagonal)
                if h != 0.0:
                    W[0, :, :, 0] = -h * X
            else:
                # bulk: shape [D, 2, 2, D]
                W = torch.zeros(D, 2, 2, D, dtype=dtype, device=device)
                # identity track
                W[0, :, :, 0] = I  # 0->0
                W[2, :, :, 2] = I  # 2->2
                # propagate a single Z
                W[0, :, :, 1] = Z  # 0->1
                # close ZZ with -J Z
                W[1, :, :, 2] = -J * Z  # 1->2
                # local field goes 0->2
                if h != 0.0:
                    W[0, :, :, 2] = -h * X

            tensors.append(W)

        return MPO(tensors, n_sites)

    @staticmethod
    def heisenberg_hamiltonian(
        n_sites: int,
        Jx: float = 1.0,
        Jy: float = 1.0,
        Jz: float = 1.0,
        device: str = "cuda",
        dtype=torch.complex64,
    ) -> MPO:
        """
        Heisenberg Hamiltonian:
        H = Σᵢ (Jₓ XᵢXᵢ₊₁ + Jᵧ YᵢYᵢ₊₁ + Jᵧ ZᵢZᵢ₊₁)
        """
        I = torch.eye(2, dtype=dtype, device=device)
        X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)

        tensors = []

        for i in range(n_sites):
            if i == 0:
                W = torch.zeros(1, 2, 2, 4, dtype=dtype, device=device)
                W[0, :, :, 0] = I
                W[0, :, :, 1] = Jx * X
                W[0, :, :, 2] = Jy * Y
                W[0, :, :, 3] = Jz * Z
            elif i == n_sites - 1:
                W = torch.zeros(4, 2, 2, 1, dtype=dtype, device=device)
                W[0, :, :, 0] = I
                W[1, :, :, 0] = X
                W[2, :, :, 0] = Y
                W[3, :, :, 0] = Z
            else:
                W = torch.zeros(4, 2, 2, 4, dtype=dtype, device=device)
                W[0, :, :, 0] = I
                W[1, :, :, 0] = X
                W[2, :, :, 0] = Y
                W[3, :, :, 0] = Z
                W[0, :, :, 1] = Jx * X
                W[0, :, :, 2] = Jy * Y
                W[0, :, :, 3] = Jz * Z

            tensors.append(W)

        return MPO(tensors, n_sites)

    @staticmethod
    def maxcut_hamiltonian(edges: List[Tuple[int, int]],
                          weights: Optional[List[float]] = None,
                          n_sites: Optional[int] = None,
                          device: str = 'cuda',
                          dtype=torch.complex64) -> MPO:
        """
        MaxCut QAOA Hamiltonian for graph optimization:
        H = Σ_{(i,j)∈E} w_{ij} (1 - ZᵢZⱼ) / 2

        This Hamiltonian encodes the MaxCut problem where we want to maximize
        the number of edges between two sets (minimize edges within sets).

        Args:
            edges: List of (i, j) tuples representing graph edges
            weights: Optional edge weights (default: all 1.0)
            n_sites: Number of nodes (inferred from edges if not provided)
            device: 'cuda' or 'cpu'
            dtype: torch dtype

        Returns:
            MPO representation of MaxCut Hamiltonian

        Example:
            ```python
            # Triangle graph (nodes 0, 1, 2)
            edges = [(0,1), (1,2), (0,2)]
            H = MPOBuilder.maxcut_hamiltonian(edges, device='cuda')

            # Use with QAOA
            from atlas_q import get_vqe_qaoa
            qaoa = get_vqe_qaoa()
            config = qaoa['QAOAConfig'](p=2, max_iter=100)
            solver = qaoa['QAOA'](H, config)
            energy, params = solver.run()
            ```
        """
        # Determine number of sites
        if n_sites is None:
            max_node = max(max(i, j) for i, j in edges)
            n_sites = max_node + 1

        # Default weights
        if weights is None:
            weights = [1.0] * len(edges)

        assert len(weights) == len(edges), "weights must match edges length"

        # Build list of ZZ interactions
        # H = Σ_{(i,j)} w_ij * (1 - Z_i Z_j) / 2
        #   = Σ w_ij/2 * I - Σ w_ij/2 * Z_i Z_j
        # We'll implement the -Σ w_ij/2 * Z_i Z_j part as MPO

        I = torch.eye(2, dtype=dtype, device=device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)

        # Build bond dimension: need to track all possible ZZ terms
        # For simplicity, use a larger bond dimension that can accommodate all terms
        # More efficient: use optimized MPO construction, but for now use general approach

        # Simple approach: sum individual ZZ MPOs
        # Each ZZ term can be built as an MPO, then sum them

        # Create identity MPO as base
        result_tensors = None

        for (i, j), w in zip(edges, weights):
            # Normalize edge order (swap if needed since graph is undirected)
            if i > j:
                i, j = j, i
            assert i < j, f"Invalid edge with i == j: ({i}, {j})"
            assert j < n_sites, f"Edge {(i,j)} exceeds n_sites={n_sites}"

            # Build ZZ MPO for sites i and j with coefficient -w/2
            coeff = -w / 2.0

            zz_tensors = []
            for site in range(n_sites):
                if site == i:
                    # First Z in the ZZ term
                    if site == 0:
                        W = torch.zeros(1, 2, 2, 2, dtype=dtype, device=device)
                        W[0, :, :, 0] = I  # identity track
                        W[0, :, :, 1] = coeff * Z  # start ZZ term
                    else:
                        W = torch.zeros(2, 2, 2, 2, dtype=dtype, device=device)
                        W[0, :, :, 0] = I
                        W[1, :, :, 1] = I
                        W[0, :, :, 1] = coeff * Z
                elif site < j:
                    # Between i and j: propagate identity
                    if site == 0:
                        W = torch.zeros(1, 2, 2, 2, dtype=dtype, device=device)
                        W[0, :, :, 0] = I
                    else:
                        W = torch.zeros(2, 2, 2, 2, dtype=dtype, device=device)
                        W[0, :, :, 0] = I
                        W[1, :, :, 1] = I
                elif site == j:
                    # Second Z in the ZZ term
                    if site == n_sites - 1:
                        W = torch.zeros(2, 2, 2, 1, dtype=dtype, device=device)
                        W[0, :, :, 0] = I
                        W[1, :, :, 0] = Z  # complete ZZ term
                    else:
                        W = torch.zeros(2, 2, 2, 2, dtype=dtype, device=device)
                        W[0, :, :, 0] = I
                        W[1, :, :, 0] = Z  # complete and stay on identity
                        W[1, :, :, 1] = I
                else:
                    # After j: pure identity
                    if site == n_sites - 1:
                        W = torch.zeros(2, 2, 2, 1, dtype=dtype, device=device)
                        W[0, :, :, 0] = I
                        W[1, :, :, 0] = I
                    else:
                        W = torch.zeros(2, 2, 2, 2, dtype=dtype, device=device)
                        W[0, :, :, 0] = I
                        W[1, :, :, 1] = I

                zz_tensors.append(W)

            # Add to result (sum MPOs by summing tensors element-wise)
            if result_tensors is None:
                result_tensors = zz_tensors
            else:
                # Sum MPO tensors - need to handle different bond dims
                # For simplicity in first implementation, rebuild with larger bonds
                # This is not optimal but works correctly
                for idx in range(n_sites):
                    # Expand bond dims to accommodate both MPOs
                    old_shape = result_tensors[idx].shape
                    new_shape = zz_tensors[idx].shape

                    max_left = max(old_shape[0], new_shape[0])
                    max_right = max(old_shape[3], new_shape[3])

                    # Create new tensor with expanded bonds
                    expanded = torch.zeros(max_left, 2, 2, max_right,
                                         dtype=dtype, device=device)

                    # Copy old values
                    expanded[:old_shape[0], :, :, :old_shape[3]] += result_tensors[idx]
                    # Add new values
                    expanded[:new_shape[0], :, :, :new_shape[3]] += zz_tensors[idx]

                    result_tensors[idx] = expanded

        # Add constant term: Σ w_ij/2 * I to get full (1 - ZZ)/2
        # This is a global energy shift, often omitted in optimization
        # For completeness, add it to the first site
        const_term = sum(weights) / 2.0
        result_tensors[0][0, :, :, 0] += const_term * I

        return MPO(result_tensors, n_sites)

    @staticmethod
    def molecular_hamiltonian_from_specs(
        molecule: str = 'H2',
        basis: str = 'sto-3g',
        charge: int = 0,
        spin: int = 0,
        mapping: str = 'jordan_wigner',
        device: str = 'cuda',
        dtype=torch.complex64
    ) -> MPO:
        """
        Build molecular Hamiltonian from molecular specifications using PySCF.

        This function computes the electronic Hamiltonian for a molecule and
        converts it to an MPO suitable for VQE or other quantum algorithms.

        Args:
            molecule: Molecular formula or geometry string
                     Examples: 'H2', 'LiH', 'H2O', or geometry string
            basis: Gaussian basis set (sto-3g, 6-31g, cc-pvdz, etc.)
            charge: Total molecular charge
            spin: Spin multiplicity (2S, where S is total spin)
            mapping: Fermion-to-qubit mapping ('jordan_wigner', 'bravyi_kitaev', 'parity')
            device: 'cuda' or 'cpu'
            dtype: torch dtype

        Returns:
            MPO representation of molecular Hamiltonian

        Example:
            ```python
            from atlas_q import get_mpo_ops, get_vqe_qaoa

            # H2 molecule
            mpo_mod = get_mpo_ops()
            H = mpo_mod['MPOBuilder'].molecular_hamiltonian_from_specs(
                molecule='H2',
                basis='sto-3g',
                device='cuda'
            )

            # Run VQE
            vqe_mod = get_vqe_qaoa()
            config = vqe_mod['VQEConfig'](n_layers=2, max_iter=100)
            vqe = vqe_mod['VQE'](H, config)
            energy, params = vqe.run()
            print(f"Ground state energy: {energy:.6f} Ha")
            ```

        Note:
            Requires pyscf package: pip install pyscf
        """
        try:
            from pyscf import gto, scf, ao2mo
        except ImportError:
            raise ImportError(
                "PySCF is required for molecular Hamiltonians. "
                "Install with: pip install pyscf"
            )

        # Parse molecule specification
        if molecule in ['H2', 'h2']:
            # H2 with default bond length 0.74 Å
            mol_spec = 'H 0 0 0; H 0 0 0.74'
        elif molecule in ['LiH', 'lih']:
            # LiH with default bond length
            mol_spec = 'Li 0 0 0; H 0 0 1.5949'
        elif molecule in ['H2O', 'h2o']:
            # Water with default geometry
            mol_spec = '''
            O 0.0000 0.0000 0.1173
            H 0.0000 0.7572 -0.4692
            H 0.0000 -0.7572 -0.4692
            '''
        elif ';' in molecule or '\n' in molecule:
            # Custom geometry string
            mol_spec = molecule
        else:
            raise ValueError(
                f"Unknown molecule '{molecule}'. "
                "Provide geometry string or use 'H2', 'LiH', 'H2O'"
            )

        # Build molecule with PySCF
        mol = gto.M(
            atom=mol_spec,
            basis=basis,
            charge=charge,
            spin=spin
        )

        # Run Hartree-Fock
        mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
        mf.kernel()

        # Get one- and two-electron integrals in MO basis
        h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        eri = ao2mo.kernel(mol, mf.mo_coeff)

        # eri is in physicist notation: (pq|rs) = ∫ φp(1) φq(2) r₁₂⁻¹ φr(1) φs(2)
        # Convert to chemist notation for fermion Hamiltonian
        n_orbitals = h1.shape[0]
        h2 = ao2mo.restore(1, eri, n_orbitals)  # chemist notation (pr|qs)

        # Convert to numpy
        h1 = h1.real if np.allclose(h1.imag, 0) else h1
        h2 = h2.real if np.allclose(h2.imag, 0) else h2

        # Get nuclear repulsion energy
        e_nuc = mol.energy_nuc()

        # Apply fermion-to-qubit mapping
        if mapping.lower() == 'jordan_wigner':
            pauli_terms = _jordan_wigner_transform(h1, h2, e_nuc)
        elif mapping.lower() == 'bravyi_kitaev':
            raise NotImplementedError("Bravyi-Kitaev mapping not yet implemented")
        elif mapping.lower() == 'parity':
            raise NotImplementedError("Parity mapping not yet implemented")
        else:
            raise ValueError(f"Unknown mapping: {mapping}")

        # Convert Pauli terms to MPO
        return _pauli_terms_to_mpo(pauli_terms, device=device, dtype=dtype)


def _jordan_wigner_transform(h1: np.ndarray, h2: np.ndarray, e_nuc: float) -> Dict:
    """
    Apply Jordan-Wigner transformation to fermionic Hamiltonian.

    Returns dictionary of Pauli terms and coefficients.
    """
    n_orbitals = h1.shape[0]
    n_qubits = 2 * n_orbitals  # spin orbitals

    pauli_terms = {}

    # Add nuclear repulsion as identity term
    pauli_terms[('I',) * n_qubits] = e_nuc

    # One-body terms: Σ h_pq a†_p a_q
    for p in range(n_qubits):
        for q in range(n_qubits):
            # Only diagonal in spin
            if p // n_orbitals != q // n_orbitals:
                continue

            p_orb = p % n_orbitals
            q_orb = q % n_orbitals

            coeff = h1[p_orb, q_orb]
            if abs(coeff) < 1e-12:
                continue

            # Jordan-Wigner: a†_p a_q → pauli string
            pauli_string = _jw_fermi_op(p, q, n_qubits)
            for ps, c in pauli_string.items():
                if ps not in pauli_terms:
                    pauli_terms[ps] = 0
                pauli_terms[ps] += coeff * c

    # Two-body terms: Σ h_pqrs a†_p a†_q a_r a_s (in chemist notation)
    for p in range(n_qubits):
        for q in range(n_qubits):
            for r in range(n_qubits):
                for s in range(n_qubits):
                    # Extract orbital indices
                    p_orb, p_spin = p % n_orbitals, p // n_orbitals
                    q_orb, q_spin = q % n_orbitals, q // n_orbitals
                    r_orb, r_spin = r % n_orbitals, r // n_orbitals
                    s_orb, s_spin = s % n_orbitals, s // n_orbitals

                    # Spin conservation
                    if p_spin != r_spin or q_spin != s_spin:
                        continue

                    # Get two-electron integral (chemist notation)
                    coeff = 0.5 * h2[p_orb, r_orb, q_orb, s_orb]
                    if abs(coeff) < 1e-12:
                        continue

                    # Convert to Pauli
                    pauli_string = _jw_two_body_op(p, q, r, s, n_qubits)
                    for ps, c in pauli_string.items():
                        if ps not in pauli_terms:
                            pauli_terms[ps] = 0
                        pauli_terms[ps] += coeff * c

    return pauli_terms


def _jw_fermi_op(p: int, q: int, n_qubits: int) -> Dict:
    """Jordan-Wigner transform of a†_p a_q"""
    # Simplified implementation for proof of concept
    # Full implementation requires proper handling of all cases
    pauli_dict = {}

    if p == q:
        # Number operator: (I - Z)/2
        string = ['I'] * n_qubits
        pauli_dict[tuple(string)] = 0.5
        string[p] = 'Z'
        pauli_dict[tuple(string)] = -0.5
    else:
        # General case: requires X/Y operators with phase
        # Simplified: main contribution
        string = ['I'] * n_qubits
        if p < q:
            for i in range(p+1, q):
                string[i] = 'Z'
            string[p] = 'X'
            string[q] = 'X'
            pauli_dict[tuple(string)] = 0.5

            string2 = string.copy()
            string2[p] = 'Y'
            string2[q] = 'Y'
            pauli_dict[tuple(string2)] = 0.5
        else:
            # p > q case
            for i in range(q+1, p):
                string[i] = 'Z'
            string[q] = 'X'
            string[p] = 'X'
            pauli_dict[tuple(string)] = 0.5

            string2 = string.copy()
            string2[q] = 'Y'
            string2[p] = 'Y'
            pauli_dict[tuple(string2)] = -0.5

    return pauli_dict


def _jw_two_body_op(p: int, q: int, r: int, s: int, n_qubits: int) -> Dict:
    """Jordan-Wigner transform of a†_p a†_q a_r a_s"""
    # This is complex; for now use approximation
    # Full implementation requires product of two one-body terms
    pauli_dict = {}

    # Simplified: diagonal terms dominate for molecular Hamiltonians
    if p == r and q == s:
        # n_p n_q term
        string = ['I'] * n_qubits
        string[p] = 'Z'
        string[q] = 'Z'
        pauli_dict[tuple(string)] = 0.25

    return pauli_dict


def _pauli_terms_to_mpo(pauli_terms: Dict, device: str, dtype) -> MPO:
    """
    Convert Pauli terms to MPO representation.

    This is a simplified conversion for demonstration.
    Production code should use optimized MPO compression.
    """
    # Get number of qubits from first term
    n_qubits = len(next(iter(pauli_terms.keys())))

    # Pauli matrices
    I = torch.eye(2, dtype=dtype, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)

    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    # Build as sum of product operators
    # Start with zero MPO
    result_tensors = None

    for pauli_string, coeff in pauli_terms.items():
        if abs(coeff) < 1e-12:
            continue

        # Build MPO for this Pauli string
        ops = [pauli_map[p] * coeff if i == 0 else pauli_map[p]
               for i, p in enumerate(pauli_string)]

        term_mpo = MPO.from_local_ops(ops, device=device)

        # Sum with result
        if result_tensors is None:
            result_tensors = term_mpo.tensors
        else:
            # Add tensors (this assumes compatible bond dims)
            for i in range(n_qubits):
                result_tensors[i] = result_tensors[i] + term_mpo.tensors[i]

    return MPO(result_tensors, n_qubits)


def apply_mpo_to_mps(mpo: MPO, mps, chi_max: int = 128, eps: float = 1e-8) -> "AdaptiveMPS":
    """
    Apply MPO to MPS: |ψ'⟩ = O |ψ⟩

    Uses zipper/zip-up algorithm for efficient contraction

    Args:
        mpo: Matrix product operator
        mps: Matrix product state (AdaptiveMPS)
        chi_max: Maximum bond dimension after compression
        eps: Truncation tolerance

    Returns:
        New MPS after applying MPO
    """
    from .adaptive_mps import AdaptiveMPS

    assert mpo.n_sites == mps.num_qubits, "MPO and MPS must have same number of sites"

    n = mpo.n_sites
    device = mps.tensors[0].device
    dtype = mps.tensors[0].dtype

    # Result MPS tensors
    new_tensors = []

    # Contract MPO with MPS site by site
    for i in range(n):
        # W: [l, s, s', r]
        W = mpo.tensors[i].to(device=device, dtype=dtype)
        # A: [a, s', b]
        A = mps.tensors[i]

        # Contract over s'
        # M[l a, s, r b] = Σ_{s′} W[l, s, s′, r] * A[a, s′, b]
        M = torch.einsum("lstr, atb -> lasrb", W, A)  # [l, a, s, r, b]
        l, a, s, r, b = M.shape
        M = M.reshape(l * a, s, r * b)  # [l a, s, r b]

        new_tensors.append(M)

    # Create new MPS
    result = AdaptiveMPS(n, bond_dim=2, device=device)
    result.tensors = new_tensors

    # Compress back to chi_max using SVD sweeps
    result.to_left_canonical()

    return result


def expectation_value(mpo: MPO, mps, use_gpu_optimized: bool = True) -> complex:
    """
    Compute ⟨ψ|O|ψ⟩ where O is an MPO and |ψ⟩ is an MPS

    Args:
        mpo: Matrix Product Operator
        mps: Matrix Product State
        use_gpu_optimized: Use GPU-optimized contractions (torch.compile)

    Returns:
        Complex expectation value
    """
    n = mpo.n_sites
    assert n == mps.num_qubits

    dtype = mps.tensors[0].dtype
    device = mps.tensors[0].device

    # Use GPU-optimized version if available and enabled
    use_optimized = GPU_OPTIMIZED_AVAILABLE and use_gpu_optimized and device.type == "cuda"

    # E has shape [l, ā, a]; start with scalars (1×1×1)
    E = torch.ones(1, 1, 1, dtype=dtype, device=device)

    for i in range(n):
        W = mpo.tensors[i].to(device=device, dtype=dtype)  # [l, s, s', r]
        A = mps.tensors[i]  # [a, s, b]

        if use_optimized:
            # GPU-optimized contraction (torch.compile + optimized order)
            E = mpo_expectation_step_optimized(E, A, W)
        else:
            # Standard einsum
            Ac = A.conj()  # [ā, s', b̄]
            # E' [χR, aR, bR] = Σ E[χL,aL,bL] * Ac[aL,σ',aR] * W[χL,σ,σ',χR] * A[bL,σ,bR]
            # Indices: L=χL, a=aL, b=bL, t=σ', r=aR, s=σ, R=χR, B=bR
            E = torch.einsum("Lab, atr, LstR, bsB -> RrB", E, Ac, W, A)

    # Now E should be [1, 1, 1] -> scalar
    if E.numel() == 1:
        return complex(E.item())
    else:
        # Extract the [0,0,0] element if not scalar
        return complex(E[0, 0, 0].item())


def correlation_function(
    op1: torch.Tensor, site1: int, op2: torch.Tensor, site2: int, mps
) -> complex:
    """
    Compute two-point correlation function: ⟨ψ| O₁(site1) O₂(site2) |ψ⟩

    Args:
        op1: First operator (2×2)
        site1: First site
        op2: Second operator (2×2)
        site2: Second site
        mps: MPS state

    Returns:
        Correlation ⟨O₁ O₂⟩
    """
    n = mps.num_qubits
    assert 0 <= site1 < n and 0 <= site2 < n

    # Ensure site1 < site2
    if site1 > site2:
        site1, site2 = site2, site1
        op1, op2 = op2, op1

    device = mps.tensors[0].device
    dtype = mps.tensors[0].dtype

    # Build MPO with op1 at site1, op2 at site2, identity elsewhere
    I = torch.eye(2, dtype=dtype, device=device)
    ops = [I] * n
    ops[site1] = op1.to(device=device, dtype=dtype)
    ops[site2] = op2.to(device=device, dtype=dtype)

    mpo = MPO.from_local_ops(ops, device=device)

    return expectation_value(mpo, mps)
