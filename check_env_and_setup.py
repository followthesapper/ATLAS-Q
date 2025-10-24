#!/usr/bin/env python3
"""
Environment inspector + safe setup for NVIDIA DGX AI Mini PCs.

- Prints GPU info (CUDA, cuDNN, PyTorch device status)
- Shows key Python package versions
- Lists all installed packages (short summary)
- Creates a virtual environment .venv/ for isolated development
"""

import os
import sys
import subprocess
import venv
import platform
from importlib import import_module

def run(cmd, check=False):
    """Run a shell command and return (stdout, stderr)."""
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return out.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# 1) Python and system info
print_header("SYSTEM & PYTHON INFO")
print(f"Python executable: {sys.executable}")
print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")

# 2) GPU / CUDA info
print_header("GPU & CUDA INFO")
try:
    out = run("nvidia-smi")
    print(out.splitlines()[0:10] if out else "nvidia-smi not found")
except Exception as e:
    print(f"GPU check error: {e}")

# 3) Library checks
print_header("KEY LIBRARY VERSIONS")
for mod in ["torch", "tensorflow", "numpy", "cupy", "numba", "scikit-learn"]:
    try:
        m = import_module(mod)
        ver = getattr(m, "__version__", "unknown")
        print(f"{mod:<15} {ver}")
        if mod == "torch":
            print(f"  CUDA available: {m.cuda.is_available()}")
            if m.cuda.is_available():
                print(f"  Device count: {m.cuda.device_count()}")
                print(f"  Device name: {m.cuda.get_device_name(0)}")
    except ModuleNotFoundError:
        print(f"{mod:<15} not installed")

# 4) All installed packages
print_header("INSTALLED PACKAGES (summary)")
out = run(f"{sys.executable} -m pip list | head -n 50")
print(out)

# 5) Create a virtual environment
print_header("VIRTUAL ENVIRONMENT SETUP")
venv_dir = os.path.join(os.getcwd(), ".venv")
if not os.path.exists(venv_dir):
    print(f"Creating venv at {venv_dir} ...")
    venv.create(venv_dir, with_pip=True, clear=False)
else:
    print(f"venv already exists at {venv_dir}")

# Activate instructions
activate_cmd = "source .venv/bin/activate" if os.name != "nt" else ".venv\\Scripts\\activate"
print(f"\nTo activate your environment, run:\n  {activate_cmd}")

print("\nTo install project deps inside venv:\n  pip install -r requirements-dev.txt")

print("\n✅ Environment inspection complete.")
