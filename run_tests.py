#!/usr/bin/env python3
"""
Universal test runner for the Quantum Hybrid Simulator repo.

Features
- Optional: install dev dependencies (pytest, pytest-cov, pytest-xdist, sklearn, torch-cpu)
- Skip GPU tests with -k "not gpu"
- Parallel workers via pytest-xdist
- Coverage (terminal + XML + optional HTML)
- JUnit XML for CI systems
- Clear exit code mirroring pytest’s result

Usage examples
-------------
# Simple
python run_tests.py

# Skip GPU tests and produce coverage + HTML + JUnit
python run_tests.py --skip-gpu --coverage --html-report --xml-report

# Install deps first and run in 8 workers
python run_tests.py --install-deps --workers 8
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sh(cmd: list[str], check: bool = True) -> int:
    print(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check).returncode


def have_module(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def ensure_pytest(python: str = sys.executable):
    try:
        subprocess.run([python, "-m", "pytest", "--version"], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("pytest not found; installing...")
        sh([python, "-m", "pip", "install", "--upgrade", "pip"])
        sh([python, "-m", "pip", "install", "pytest"])


def maybe_install(packages: list[str], python: str = sys.executable):
    if not packages:
        return
    print(f"Installing: {', '.join(packages)}")
    sh([python, "-m", "pip", "install", *packages])


def main():
    ap = argparse.ArgumentParser(description="Run project tests with optional reports.")
    ap.add_argument("--install-deps", action="store_true", help="Install dev/test dependencies before running.")
    ap.add_argument("--skip-gpu", action="store_true", help="Skip tests marked with 'gpu'.")
    ap.add_argument("--coverage", action="store_true", help="Generate coverage (terminal + XML).")
    ap.add_argument("--html-report", action="store_true", help="Also generate HTML coverage report.")
    ap.add_argument("--xml-report", action="store_true", help="Emit JUnit XML test results.")
    ap.add_argument("--workers", type=int, default=0, help="Number of parallel workers (requires pytest-xdist). 0 = no parallel.")
    ap.add_argument("--tests-path", default="tests", help="Path to tests directory (default: tests).")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Optionally install deps (best-effort; DGX has internet via pip mirror, typically)
    if args.install_deps:
        pkgs = ["pytest>=7.0"]
        if args.coverage or args.html_report:
            pkgs.append("pytest-cov>=4.0")
        if args.workers and args.workers > 0:
            pkgs.append("pytest-xdist>=3.0")
        # Useful optional extras for demos/tests
        pkgs.extend([
            "numpy>=1.22",
            "matplotlib>=3.6",
            "scikit-learn>=1.2"
        ])
        # Torch CPU is fine even on DGX if you don't want CUDA in tests
        # (comment out if you prefer your system CUDA wheels)
        pkgs.append("torch")
        maybe_install(pkgs, python=args.python)

    # Ensure pytest exists
    ensure_pytest(args.python)

    # Assemble pytest args
    pytest_args: list[str] = ["-q"]

    # Skip GPU tests via keyword expression
    if args.skip_gpu:
        pytest_args += ["-k", "not gpu"]

    # Parallel workers
    if args.workers and args.workers > 0:
        if have_module("xdist"):
            pytest_args += ["-n", str(args.workers)]
        else:
            print("pytest-xdist not installed; running without parallel workers.")

    # Reports
    junit_xml = reports_dir / "junit.xml"
    cov_xml = reports_dir / "coverage.xml"
    htmlcov_dir = reports_dir / "htmlcov"

    if args.xml_report:
        pytest_args += [f"--junitxml={junit_xml}"]

    if args.coverage or args.html_report:
        if not have_module("pytest_cov"):
            # Try to install on the fly (no fatal if fails)
            try:
                maybe_install(["pytest-cov>=4.0"], python=args.python)
            except Exception:
                print("WARNING: pytest-cov not available; skipping coverage reports.")
        if have_module("pytest_cov"):
            pytest_args += ["--cov=.", "--cov-report=term-missing", f"--cov-report=xml:{cov_xml}"]
            if args.html_report:
                pytest_args += [f"--cov-report=html:{htmlcov_dir}"]
        else:
            # If still not present, drop coverage flags
            args.coverage = False
            args.html_report = False

    # Test path
    tests_path = repo_root / args.tests_path
    pytest_args.append(str(tests_path if tests_path.exists() else repo_root))

    # Set PYTHONPATH to include repo (src layout + root)
    env = os.environ.copy()
    extra_paths = [str(repo_root), str(repo_root / "src")]
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + [env.get("PYTHONPATH", "")])

    # Print final command
    print("=" * 60)
    print(f"{args.python} -m pytest {' '.join(pytest_args)}")
    print("=" * 60)

    # Run tests
    try:
        rc = subprocess.run([args.python, "-m", "pytest", *pytest_args], env=env).returncode
    except KeyboardInterrupt:
        rc = 130

    # Summary
    print("\n" + "=" * 60)
    if rc == 0:
        print("✅ All tests passed.")
    else:
        print(f"❌ Some tests failed. Exit code: {rc}")
    if args.xml_report:
        print(f"JUnit XML: {junit_xml}")
    if args.coverage:
        print(f"Coverage XML: {cov_xml}")
    if args.html_report:
        index_html = htmlcov_dir / "index.html"
        print(f"HTML coverage: {index_html}")
    print("=" * 60 + "\n")

    sys.exit(rc)


if __name__ == "__main__":
    main()
