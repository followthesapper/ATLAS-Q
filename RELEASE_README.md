# ATLAS-Q Release Guide
**Quick Reference for Publishing to PyPI and Docker**

This document summarizes how to build, test, and publish ATLAS-Q as a Python package and Docker images.

---

## Prerequisites

### For PyPI Publishing
1. Create PyPI account: https://pypi.org/account/register/
2. Generate API token: https://pypi.org/manage/account/token/
3. Add token to GitHub secrets:
   - Go to: https://github.com/followthsapper/ATLAS-Q/settings/secrets/actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Your PyPI API token

### For Docker
- Docker installed and running
- For GPU images: NVIDIA Docker runtime

---

## Quick Commands

### Building Python Package

```bash
# Build package
make build

# Test build locally
pip install dist/atlas_q-0.5.0-py3-none-any.whl

# Publish to TestPyPI (test first!)
make publish-test

# Publish to PyPI (production)
make publish
```

### Building Docker Images

```bash
# Build images
make docker-build-gpu    # CUDA 12.2 GPU image
make docker-build-cpu    # CPU-only image

# Test images
make docker-run-gpu
make docker-run-cpu
```

### Running Tests

```bash
make test                # All tests
make test-unit           # Unit tests only
make test-nogpu          # Skip GPU tests
make bench               # Run benchmarks
```

---

## Automated Release (Recommended)

The easiest way to release is through GitHub Releases, which triggers automatic publishing:

### Step 1: Prepare Release

```bash
# Update version numbers
vim pyproject.toml                    # version = "0.5.0"
vim src/atlas_q/__init__.py          # __version__ = '0.5.0'
vim README.md                         # **Version 0.5.0**
vim docs/CHANGELOG.md                 # Add [0.5.0] - 2025-10-27

# Test everything
make test
make build
make docker-build-gpu
make docker-build-cpu

# Commit
git add -A
git commit -m "Release version 0.5.0"
git push origin main
```

### Step 2: Create GitHub Release

1. Go to: https://github.com/followthsapper/ATLAS-Q/releases/new
2. Click "Choose a tag" → type `v0.5.0` → "Create new tag: v0.5.0 on publish"
3. Release title: `ATLAS-Q v0.5.0`
4. Description: Copy from CHANGELOG.md
5. Click "Publish release"

**GitHub Actions will automatically:**
- ✅ Build Python package
- ✅ Publish to PyPI
- ✅ Build Docker images (GPU and CPU)
- ✅ Publish to GitHub Container Registry

### Step 3: Verify (wait 5-10 minutes)

```bash
# Verify PyPI
pip install --upgrade atlas-q
python -c "import atlas_q; print(atlas_q.__version__)"

# Verify Docker
docker pull ghcr.io/followthsapper/atlas-q:cuda
docker run --rm ghcr.io/followthsapper/atlas-q:cuda python -c "import atlas_q; print(atlas_q.__version__)"

# Check pages
# PyPI: https://pypi.org/project/atlas-q/
# Docker: https://github.com/followthsapper/ATLAS-Q/pkgs/container/atlas-q
```

---

## Manual Release

If you need to publish manually without GitHub Actions:

### PyPI

```bash
# Build
make build

# Upload (requires ~/.pypirc with API token)
make publish
```

### Docker

```bash
# Build
make docker-build-gpu
make docker-build-cpu

# Tag for registry
docker tag atlas-q:cuda ghcr.io/followthsapper/atlas-q:cuda
docker tag atlas-q:cpu ghcr.io/followthsapper/atlas-q:cpu

# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u followthsapper --password-stdin

# Push
docker push ghcr.io/followthsapper/atlas-q:cuda
docker push ghcr.io/followthsapper/atlas-q:cpu
```

---

## How Users Will Install

### Python Package

```bash
# Basic installation (CPU)
pip install atlas-q

# With GPU support
pip install atlas-q[gpu]

# All optional features
pip install atlas-q[all]

# From source
git clone https://github.com/followthsapper/ATLAS-Q.git
cd ATLAS-Q
pip install -e .[gpu,dev]
```

### Docker

```bash
# GPU version
docker pull ghcr.io/followthsapper/atlas-q:cuda
docker run --rm -it --gpus all ghcr.io/followthsapper/atlas-q:cuda

# CPU version
docker pull ghcr.io/followthsapper/atlas-q:cpu
docker run --rm -it ghcr.io/followthsapper/atlas-q:cpu

# Run with mounted code
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  ghcr.io/followthsapper/atlas-q:cuda \
  python /workspace/my_script.py
```

---

## Files Created

This setup added/modified these files:

### Documentation
- `docs/API_GUIDE.md` - Complete API reference
- `docs/MAINTENANCE_GUIDE.md` - Updated with PyPI/Docker workflows
- `RELEASE_README.md` - This file

### Packaging
- `pyproject.toml` - Updated for production PyPI
- `.gitignore` - Added package/Docker artifacts

### Docker
- `Dockerfile` - GPU variant (CUDA 12.2)
- `Dockerfile.cpu` - CPU variant
- `.dockerignore` - Exclude unnecessary files

### CI/CD
- `.github/workflows/ci.yml` - Run tests on PRs/pushes
- `.github/workflows/publish-pypi.yml` - Auto-publish to PyPI
- `.github/workflows/publish-docker.yml` - Auto-publish Docker images

### Build Tools
- `Makefile` - Added `build`, `publish`, `docker-*` targets

---

## Troubleshooting

### PyPI Upload Fails

**Error:** `403 Forbidden`
**Solution:** Check API token is correct and has permissions for `atlas-q` project

**Error:** `400 Bad Request: File already exists`
**Solution:** Version already published. Bump version number and try again.

### Docker Build Fails

**Error:** `CUDA not found`
**Solution:** Using wrong Dockerfile. Use `Dockerfile.cpu` for non-CUDA builds.

**Error:** `Out of memory`
**Solution:** Increase Docker memory limit in Docker Desktop settings.

### GitHub Actions Fail

Check workflow logs at: https://github.com/followthsapper/ATLAS-Q/actions

**Common issues:**
- Missing `PYPI_API_TOKEN` secret
- Version number mismatch between files
- Tests failing (fix tests before releasing)

---

## Next Steps

1. **First Time Setup:**
   - Create PyPI account
   - Add `PYPI_API_TOKEN` to GitHub secrets
   - Test build process: `make build && make docker-build-cpu`

2. **Before Each Release:**
   - Update version in `pyproject.toml` and `src/atlas_q/__init__.py`
   - Update `CHANGELOG.md`
   - Run `make test` and `make bench`
   - Update performance numbers in docs if needed

3. **Release Process:**
   - Follow "Automated Release" steps above
   - Wait 5-10 minutes for GitHub Actions
   - Verify installation works
   - Announce release

---

## Support

- **GitHub Actions:** Check `.github/workflows/*.yml` for workflow details
- **PyPI:** https://pypi.org/project/atlas-q/
- **Docker Hub:** https://github.com/followthsapper/ATLAS-Q/pkgs/container/atlas-q
- **Full Guide:** See `docs/MAINTENANCE_GUIDE.md` Workflows 5-7

---

**Status:** ✅ Production-ready packaging setup complete
**Last Updated:** October 27, 2025
