# ATLAS-Q Research Documentation

This directory contains research documentation and validation results for ATLAS-Q's coherence-aware quantum computing framework.

## Key Documents

### Primary Documentation

- **COHERENCE_AWARE_VQE.md**: Main research documentation for coherence-aware VQE
- **COHERENCE_AWARE_VQE_BREAKTHROUGH.md**: Detailed breakthrough achievements and validation
- **VQE_VRA_INTEGRATION_COMPLETE.md**: Integration completion summary

### Hardware Validation

- **IBM_QUANTUM_DEPLOYMENT_RESULTS.md**: IBM Brisbane deployment results
- **IBM_QUANTUM_SETUP.md**: Setup guide for IBM Quantum hardware
- **QUICK_START_IBM_QUANTUM.md**: Quick start guide for IBM Quantum

### VRA Integration Analysis

- **ATLAS_Q_VRA_INTEGRATION_SUMMARY.md**: VRA integration summary
- **ATLAS_Q_VRA_ANALYSIS_AND_NEXT_STEPS.md**: Analysis and future directions
- **VRA_COMPLETE_INTEGRATION_SUMMARY.md**: Complete integration details
- **VRA_INTEGRATION_COMPLETE.md**: Integration completion notes
- **VRA_INTEGRATION_OPPORTUNITIES.md**: Future integration opportunities
- **VRA_VQE_SUMMARY.md**: VQE-specific VRA summary

### Validation Results

- **VRA_COMMUTATIVITY_SUMMARY.md**: Commutativity analysis
- **VRA_FINAL_SCALING_RESULTS.md**: Scaling validation results
- **VRA_LARGER_MOLECULES_SUMMARY.md**: Results for larger molecules
- **VRA_INTEGRATION_SUMMARY.md**: Overall integration summary

## Key Results

### Hardware Validation (IBM Brisbane)

| Molecule | Qubits | R̄ (Coherence) | Classification | Runtime |
|----------|--------|---------------|----------------|---------|
| H2       | 4      | 0.891         | GO             | 29s     |
| LiH      | 12     | 0.980         | GO             | 43s     |
| H2O      | 14     | 0.988         | GO             | 96s     |

### VRA Measurement Compression

- H2: 15 → 15 groups (1.0×)
- LiH: 631 → 127 groups (4.97×)
- H2O: 1086 → 219 groups (4.96×)

## For Researchers

These documents provide comprehensive technical details for:
- Reproducing results on IBM Quantum hardware
- Understanding the coherence-aware framework
- Extending VRA integration to new algorithms
- Publishing research based on ATLAS-Q

## For Users

Start with:
1. `COHERENCE_AWARE_VQE.md` for overview
2. `IBM_QUANTUM_SETUP.md` for hardware setup
3. `../user_guide/coherence_aware_vqe.rst` for API documentation

## Status

**Production Ready** - All results hardware-validated and peer-review ready.

**Version**: 0.6.2
**Last Updated**: November 2025
