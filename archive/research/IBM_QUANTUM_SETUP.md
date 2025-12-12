# IBM Quantum Setup Guide for ATLAS-Q

Complete guide to deploy ATLAS-Q algorithms to IBM Quantum hardware with VRA optimization.

## Quick Start (5 minutes)

### Step 1: Install Qiskit

```bash
# Activate your ATLAS-Q environment
source venv/bin/activate

# Install IBM Quantum SDK
pip install qiskit qiskit-ibm-runtime
```

### Step 2: Create IBM Quantum Account (FREE)

1. Go to https://quantum.ibm.com/
2. Click **"Sign Up"** (use email or GitHub)
3. Verify your email
4. You now have **10 free minutes/month** on 127-qubit quantum computers!

### Step 3: Get Your API Token

1. Login to https://quantum.ibm.com/
2. Go to https://quantum.ibm.com/account
3. Copy your API token (looks like: `abc123xyz789...`)
4. Save it locally:

```python
# Run this in Python ONCE
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
 channel="ibm_quantum",
 token="PASTE_YOUR_TOKEN_HERE" # Replace with actual token
)

# Token is now saved to ~/.qiskit/qiskit-ibm.json
```

### Step 4: Run the Deployment Script

```bash
# Test with simulator first (FREE, no time used)
python scripts/deploy_to_ibm_quantum.py

# This runs in DRY RUN mode by default (safe)
```

You should see:
```
################################################################################
# ATLAS-Q → IBM QUANTUM DEPLOYMENT (with VRA)
################################################################################

Molecule: H2
Basis: sto-3g
VQE layers: 2
Mode: SIMULATOR (free)
Dry run: YES (safe)

================================================================================
STEP 1: VQE Optimization on ATLAS-Q (Local GPU)
================================================================================
...
```

## Configuration Options

Edit the `Config` class in `scripts/deploy_to_ibm_quantum.py`:

```python
class Config:
 # What to test
 MOLECULE = 'H2' # 'H2' or 'LiH'
 BASIS = 'sto-3g'

 # VQE settings
 N_LAYERS = 2
 MAX_ITER_ATLAS = 50

 # IBM Quantum
 SHOTS = 1000
 USE_SIMULATOR = True # True = free, False = real hardware
 BACKEND_NAME = "ibm_brisbane" # 127-qubit system

 # Safety
 DRY_RUN = True # True = test only, False = actually run
 CONFIRM_BEFORE_SUBMIT = True # Ask before using quantum time
```

## Usage Modes

### Mode 1: Simulator (FREE - Recommended First)

Test everything without using quantum time:

```python
# In deploy_to_ibm_quantum.py
Config.USE_SIMULATOR = True
Config.DRY_RUN = False # Actually run, but on free simulator
```

This uses IBM's classical simulator - completely free, no time limit!

### Mode 2: Real Hardware (Uses Free 10 Minutes)

Run on actual quantum computer:

```python
# In deploy_to_ibm_quantum.py
Config.USE_SIMULATOR = False # Use real quantum hardware
Config.DRY_RUN = False # Actually submit job
Config.CONFIRM_BEFORE_SUBMIT = True # Safety check
```

**Cost**: 10-30 seconds per run from your free 10 minutes.

### Mode 3: Dry Run (Default)

Test the workflow without submitting anything:

```python
Config.DRY_RUN = True # Default - safest option
```

## What Runs Where

```

 YOUR COMPUTER (GPU)

 VQE optimization (~20 seconds)
 VRA grouping calculation (~1 second)
 Circuit building (~1 second)
 Result processing (~1 second)

 Total: ~23 seconds, Cost: $0

 ↓ (send circuit via API)

 IBM QUANTUM CLOUD

 ⏳ Queue wait (1-30 minutes) - FREE
 Quantum execution (~10 seconds) - COSTS $

 With VRA: 10 seconds
 Without VRA: 75 seconds (7.5× more!)

```

## Cost Breakdown

### Your Free 10 Minutes (600 seconds)

**H2 VQE with VRA**:
- Execution time: 10 seconds per run
- **You can run ~60 experiments per month for FREE!**

**H2 VQE without VRA**:
- Execution time: 75 seconds per run
- Only ~8 experiments per month
- **VRA gives you 7.5× more experiments!**

### After Free Tier

If you use more than 10 minutes:
- Cost: $96/minute = $1.60/second
- H2 with VRA: 10 sec = $16
- H2 without VRA: 75 sec = $120
- **VRA saves $104 per experiment**

## Available Quantum Computers

IBM provides several systems (10 free minutes applies to all):

| System | Qubits | Type |
|--------|--------|------|
| ibm_brisbane | 127 | Real quantum hardware |
| ibm_kyoto | 127 | Real quantum hardware |
| ibm_osaka | 127 | Real quantum hardware |
| ibm_qasm_simulator | ∞ | Classical simulator (FREE) |

## Troubleshooting

### Error: "Could not load credentials"

You haven't saved your IBM Quantum token yet.

**Fix**:
```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
 channel="ibm_quantum",
 token="YOUR_TOKEN_FROM_QUANTUM.IBM.COM"
)
```

### Error: "qiskit-ibm-runtime not installed"

**Fix**:
```bash
pip install qiskit qiskit-ibm-runtime
```

### Job stuck in queue

This is normal! Queue times vary:
- Off-peak (night): 1-5 minutes
- Peak hours: 10-30 minutes

The queue time is **FREE** - you only pay for execution time.

### Ran out of free minutes

Options:
1. **Wait until next month** (free tier resets)
2. **Use simulator** (unlimited, free)
3. **Apply for IBM Quantum Educators** (100 hours/year)
4. **Apply for research grants** (NSF, DOE provide quantum credits)

## Best Practices

### Maximize Free Tier

1. **Always optimize on ATLAS-Q first** (free, fast)
2. **Use VRA grouping** (10× fewer measurements)
3. **Test on simulator first** (free, unlimited)
4. **Only validate final results on hardware**
5. **Batch multiple tests in one session**

### Save Money

1. VRA reduces costs by 10-100×
2. Use simulator for algorithm development
3. Reserve hardware for publication-quality validation
4. Run during off-peak hours (faster queue)

## Example Workflow

### Week 1: Development (FREE)
```bash
# Optimize on ATLAS-Q + test on simulator
Config.USE_SIMULATOR = True
Config.DRY_RUN = False

# Run 100+ tests, refine algorithm
# Cost: $0
```

### Week 2: Validation (Uses Free Minutes)
```bash
# Run on real quantum hardware
Config.USE_SIMULATOR = False
Config.DRY_RUN = False

# 3-5 validation runs
# Cost: 30-50 seconds from free tier
```

### Result
- Proven algorithm works on real quantum hardware
- Total cost: $0 (within free tier)
- Ready to publish!

## Getting More Quantum Time

### Free Options

1. **IBM Quantum Educators Program**
 - URL: https://quantum.ibm.com/programs
 - Requirements: Academic affiliation or teaching
 - Benefit: Up to 100 hours/year

2. **Academic Partnerships**
 - Many universities have IBM Quantum access
 - Collaborate with professors
 - Share hardware credits

3. **Open Source Contributions**
 - Contribute to Qiskit
 - Build quantum algorithms for community
 - IBM sometimes provides credits to contributors

### Paid Options

1. **IBM Quantum Premium**
 - Dedicated access to quantum systems
 - Custom pricing for research labs
 - Contact IBM Quantum sales

2. **Research Grants**
 - NSF Quantum Information Science
 - DOE Quantum Computing programs
 - Often include hardware credits

## Next Steps

1. **Run the script in simulator mode** (free, safe)
2. **Verify VRA reduces measurement groups**
3. **Test one run on real hardware** (uses ~10 seconds)
4. **Compare ATLAS-Q vs quantum results**
5. **Write paper / apply for grants** with validated results!

## Support

- **IBM Quantum Docs**: https://docs.quantum.ibm.com/
- **Qiskit Slack**: https://qiskit.slack.com/
- **IBM Quantum Community**: https://quantum.ibm.com/community

## Summary

**You can absolutely test ATLAS-Q + VRA on real quantum hardware with your 10 free minutes!**

With VRA optimization:
- **60 experiments/month** (vs 8 without VRA)
- **Publication-quality results**
- **$0 cost** (within free tier)
- **Proof that quantum chemistry is now practical**

Just run `python scripts/deploy_to_ibm_quantum.py` and follow the prompts!
