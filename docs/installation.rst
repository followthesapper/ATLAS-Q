Installation
============

Prerequisites
-------------

ATLAS-Q requires:

- Python 3.9 or higher
- PyTorch 2.0 or higher
- NumPy 1.22 or higher
- SciPy 1.10 or higher

Optional dependencies:

- Triton 2.0+ for GPU acceleration (Linux only)
- cuQuantum 23.0+ for NVIDIA GPU optimization
- PySCF 2.0+ for molecular Hamiltonians
- OpenFermion 1.5+ for fermionic operators

Hardware:

- CPU: Any modern x86_64 processor
- GPU: NVIDIA GPU with CUDA support (compute capability 7.0+, recommended 8.0+)

  - Tested on: V100, A100, H100, RTX 4090
  - GPU memory: 4GB minimum, 16GB+ recommended for large simulations

Installation Methods
--------------------

PyPI Installation (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Standard installation:

.. code-block:: bash

   pip install atlas-quantum

With GPU support (includes Triton kernels):

.. code-block:: bash

   pip install atlas-quantum[gpu]

With molecular chemistry support:

.. code-block:: bash

   pip install atlas-quantum[chemistry]

With all optional dependencies:

.. code-block:: bash

   pip install atlas-quantum[all]

Virtual environment recommended:

.. code-block:: bash

   python -m venv atlas-env
   source atlas-env/bin/activate  # Linux/macOS
   atlas-env\Scripts\activate     # Windows
   pip install atlas-quantum[gpu]

Docker Installation
^^^^^^^^^^^^^^^^^^^

Pull the GPU-enabled image:

.. code-block:: bash

   docker pull ghcr.io/followthesapper/atlas-q:cuda

Run interactive session:

.. code-block:: bash

   docker run --rm -it --gpus all ghcr.io/followthesapper/atlas-q:cuda python3

Run with volume mounting for persistent data:

.. code-block:: bash

   docker run --rm -it --gpus all \
     -v $(pwd)/data:/data \
     ghcr.io/followthesapper/atlas-q:cuda python3

CPU-only image:

.. code-block:: bash

   docker pull ghcr.io/followthesapper/atlas-q:cpu
   docker run --rm -it ghcr.io/followthesapper/atlas-q:cpu python3

Run benchmarks in Docker:

.. code-block:: bash

   docker run --rm --gpus all ghcr.io/followthesapper/atlas-q:cuda \
     python3 /opt/atlas-q/scripts/benchmarks/validate_all_features.py

Building from Source
^^^^^^^^^^^^^^^^^^^^

Clone the repository:

.. code-block:: bash

   git clone https://github.com/followthesapper/ATLAS-Q.git
   cd ATLAS-Q

Install in development mode:

.. code-block:: bash

   pip install -e .

With GPU support:

.. code-block:: bash

   pip install -e .[gpu]

Configure GPU acceleration (auto-detects your GPU):

.. code-block:: bash

   ./setup_triton.sh

This script:

- Detects GPU architecture (V100, A100, H100, etc.)
- Sets ``TORCH_CUDA_ARCH_LIST`` environment variable
- Configures ``TRITON_PTXAS_PATH``
- Adds settings to ``~/.bashrc`` for persistence

Run tests:

.. code-block:: bash

   pytest tests/ -v

Run benchmarks:

.. code-block:: bash

   python scripts/benchmarks/validate_all_features.py

Build requirements:

- GCC 9+ or Clang 10+ (Linux)
- MSVC 2019+ (Windows)
- CUDA Toolkit 11.8+ (for GPU support)
- Git

Google Colab / Jupyter
^^^^^^^^^^^^^^^^^^^^^^^

Open the interactive notebook directly in Google Colab:

`Open ATLAS_Q_Demo.ipynb in Colab <https://colab.research.google.com/github/followthesapper/ATLAS-Q/blob/ATLAS-Q/ATLAS_Q_Demo.ipynb>`_

Or download and run locally:

.. code-block:: bash

   wget https://github.com/followthesapper/ATLAS-Q/raw/ATLAS-Q/ATLAS_Q_Demo.ipynb
   jupyter notebook ATLAS_Q_Demo.ipynb

Install in Colab session:

.. code-block:: python

   !pip install atlas-quantum[gpu]

Conda-forge (Coming Soon)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Conda-forge package is under review. Once approved:

.. code-block:: bash

   conda install -c conda-forge atlas-quantum

APT Installation (Future)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Debian/Ubuntu package planned for future release.

UV Installation (Future)
^^^^^^^^^^^^^^^^^^^^^^^^^

UV package manager support planned.

Verification
------------

Verify installation:

.. code-block:: python

   import atlas_q
   print(atlas_q.__version__)

Check GPU availability:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA device: {torch.cuda.get_device_name(0)}")

Check Triton availability:

.. code-block:: python

   try:
       import triton
       print(f"Triton available: {triton.__version__}")
   except ImportError:
       print("Triton not installed (GPU kernels disabled)")

Check cuQuantum availability:

.. code-block:: python

   from atlas_q import get_cuquantum
   cuq = get_cuquantum()
   print(f"cuQuantum available: {cuq['is_cuquantum_available']()}")
   if cuq['is_cuquantum_available']():
       print(f"cuQuantum version: {cuq['get_cuquantum_version']()}")

Run basic test:

.. code-block:: python

   from atlas_q import get_quantum_sim

   QCH, _, _, _ = get_quantum_sim()
   sim = QCH()
   factors = sim.factor_number(21)
   assert factors[0] * factors[1] == 21
   print("Installation verified")

Troubleshooting
---------------

CUDA/GPU Issues
^^^^^^^^^^^^^^^

If CUDA is not detected:

1. Verify NVIDIA driver installation:

   .. code-block:: bash

      nvidia-smi

2. Check PyTorch CUDA installation:

   .. code-block:: python

      import torch
      print(torch.version.cuda)
      print(torch.cuda.is_available())

3. Reinstall PyTorch with CUDA support:

   .. code-block:: bash

      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Triton Compilation Errors
^^^^^^^^^^^^^^^^^^^^^^^^^^

If Triton kernels fail to compile:

1. Check CUDA Toolkit installation:

   .. code-block:: bash

      nvcc --version

2. Set architecture manually:

   .. code-block:: bash

      export TORCH_CUDA_ARCH_LIST="8.0"  # For A100
      export TORCH_CUDA_ARCH_LIST="9.0"  # For H100

3. Verify ``ptxas`` path:

   .. code-block:: bash

      export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

Memory Errors
^^^^^^^^^^^^^

If encountering out-of-memory errors:

1. Reduce bond dimension:

   .. code-block:: python

      mps = AdaptiveMPS(num_qubits=20, bond_dim=32, chi_max_per_bond=64)

2. Enable memory budgets:

   .. code-block:: python

      mps = AdaptiveMPS(num_qubits=20, bond_dim=32, budget_global_mb=4096)

3. Use mixed precision:

   .. code-block:: python

      from atlas_q.adaptive_mps import DTypePolicy
      policy = DTypePolicy(default=torch.complex64)
      mps = AdaptiveMPS(num_qubits=20, bond_dim=32, dtype_policy=policy)

Import Errors
^^^^^^^^^^^^^

If optional dependencies are missing:

.. code-block:: bash

   pip install pyscf openfermion openfermionpyscf  # For chemistry
   pip install cuquantum-python                     # For cuQuantum
   pip install triton                               # For GPU kernels

cuQuantum Issues
^^^^^^^^^^^^^^^^

cuQuantum is optional. If unavailable, ATLAS-Q automatically falls back to PyTorch implementations. To install:

.. code-block:: bash

   pip install cuquantum-python

Requires CUDA Toolkit and compatible NVIDIA GPU.

Version Compatibility
---------------------

Tested configurations:

- Python 3.9, 3.10, 3.11, 3.12
- PyTorch 2.0, 2.1, 2.2, 2.3
- CUDA 11.8, 12.0, 12.1
- Triton 2.0, 2.1, 2.2
- cuQuantum 23.0, 24.0, 25.0

Minimum versions:

- Python: 3.9
- PyTorch: 2.0
- NumPy: 1.22
- SciPy: 1.10

Platform support:

- Linux: Full support (CUDA + Triton + cuQuantum)
- macOS: CPU only (no CUDA/Triton)
- Windows: CUDA support (Triton limited)
