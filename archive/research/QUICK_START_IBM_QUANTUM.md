# Quick Start: ATLAS-Q on IBM Quantum (10 Minutes Setup)

Follow this checklist to deploy ATLAS-Q + VRA to IBM Quantum hardware.

## Checklist

### Setup (One-Time - 5 minutes)

- [ ] **Install Qiskit**
 ```bash
 pip install qiskit qiskit-ibm-runtime
 ```

- [ ] **Create IBM Quantum Account**
 - Go to: https://quantum.ibm.com/
 - Click "Sign Up" (free)
 - Verify email
 - You now have 10 free minutes/month!

- [ ] **Get API Token**
 - Login to: https://quantum.ibm.com/account
 - Copy your API token
 - Save it:
 ```python
 from qiskit_ibm_runtime import QiskitRuntimeService
 QiskitRuntimeService.save_account(
 channel="ibm_quantum",
 token="PASTE_YOUR_TOKEN_HERE"
 )
 ```
 - Token saved to `~/.qiskit/qiskit-ibm.json`

### First Run (2 minutes)

- [ ] **Test with Simulator (FREE)**
 ```bash
 python scripts/deploy_to_ibm_quantum.py
 ```
 - Should show: `Dry run: YES (safe)`
 - Should complete: ` DEPLOYMENT COMPLETE!`

- [ ] **Enable Simulator Mode**
 - Edit `scripts/deploy_to_ibm_quantum.py`
 - Change: `Config.DRY_RUN = False`
 - Keep: `Config.USE_SIMULATOR = True`
 - Run again - uses free simulator

- [ ] **Verify VRA Savings**
 - Look for: `VRA groups: 2 (from 15 terms)`
 - Look for: `Savings: 7.5× faster`
 - VRA is working!

### Real Quantum Hardware (Your Free 10 Minutes)

- [ ] **Enable Real Hardware**
 - Edit `scripts/deploy_to_ibm_quantum.py`
 - Change: `Config.USE_SIMULATOR = False`
 - Keep: `Config.DRY_RUN = False`
 - Keep: `Config.CONFIRM_BEFORE_SUBMIT = True`

- [ ] **Run on Quantum Computer**
 ```bash
 python scripts/deploy_to_ibm_quantum.py
 ```
 - Wait for prompt: `Continue? (yes/no):`
 - Type: `yes`
 - Job submitted to quantum computer!

- [ ] **Wait for Results**
 - Queue: 1-30 minutes (FREE)
 - Execution: ~10 seconds (uses your free time)
 - Results received!

- [ ] **Check Usage**
 - Go to: https://quantum.ibm.com/account
 - View: Time used this month
 - Should show ~10-30 seconds used

## You Did It!

You just:
- Ran VQE on ATLAS-Q (local GPU)
- Applied VRA grouping (10× fewer measurements)
- Deployed to IBM Quantum
- Validated on real quantum hardware
- Used only ~10 seconds of your free 10 minutes

**Result**: You can run ~60 experiments per month for FREE with VRA!

## What Next?

### Publish Results
- Compare ATLAS-Q vs quantum hardware energies
- Show VRA reduces measurement time by 7.5×
- Write paper: "VRA on IBM Quantum Hardware"

### Get More Time
- Apply to IBM Quantum Educators (100 hours/year)
- Collaborate with university professors
- Apply for NSF/DOE quantum grants

### Scale Up
- Test LiH molecule (12 qubits)
- Try larger VQE circuits
- Benchmark QAOA on graphs

## Troubleshooting

### "Could not load credentials"
→ Run the `save_account()` command again with your token

### "qiskit not found"
→ `pip install qiskit qiskit-ibm-runtime`

### Job takes forever
→ Queue times are 1-30 min (normal, FREE)
→ Execution time is 10-30 sec (uses your quota)

### Out of free minutes
→ Use simulator (unlimited, FREE)
→ Wait for next month reset
→ Apply for IBM Quantum Educators

## Files Created

- `scripts/deploy_to_ibm_quantum.py` - Main deployment script
- `IBM_QUANTUM_SETUP.md` - Detailed documentation
- `benchmarks/vra_quantum_hardware_calculator.py` - Cost calculator

## Support

Questions? Check:
- `IBM_QUANTUM_SETUP.md` - Full documentation
- https://docs.quantum.ibm.com/ - IBM Quantum docs
- https://qiskit.slack.com/ - Qiskit community

---

**Bottom Line**: You have everything you need to test ATLAS-Q + VRA on real quantum hardware for FREE!
