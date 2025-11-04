# ATLAS-Q Maintenance Guide
**How to Keep Documentation and Code in Sync**

This guide explains what needs to be updated when you make changes to ATLAS-Q. Use this as a checklist to ensure consistency across all documentation.

---

## Quick Reference: What to Update When

### When You Add a New Feature
1. Update `CHANGELOG.md` with the feature description
2. Update `README.md` "What is ATLAS-Q?" section if it's a major feature
3. Update `WHITEPAPER.md` Section 4 (Core Features)
4. Update `OVERVIEW.md` relevant sections
5. Add usage example to `COMPLETE_GUIDE.md`
6. Update performance numbers if applicable
7. Add tests to appropriate test directory

### When You Improve Performance
1. Update performance numbers in:
 - `README.md` (Performance Highlights section)
 - `WHITEPAPER.md` (Abstract + Section 5)
 - `OVERVIEW.md` (Performance Numbers section)
 - `RESEARCH_PAPER.md` (Appendix benchmarks)
2. Update `CHANGELOG.md` with new benchmark results
3. Re-run and document benchmarks in `/benchmarks/`

### When You Fix a Bug
1. Add to `CHANGELOG.md` under "Fixed"
2. Add regression test if applicable
3. Update documentation if bug was due to unclear docs

### When You Release a New Version
1. Update version number in:
 - `README.md` (header)
 - `WHITEPAPER.md` (header)
 - `RESEARCH_PAPER.md` (header)
 - `OVERVIEW.md` (footer)
 - `src/atlas_q/__init__.py` (`__version__`)
 - `CHANGELOG.md` (add new version section)
2. Update "Roadmap" in `README.md`
3. Update "Current Status" sections across docs
4. Create git tag: `git tag -a v0.X.0 -m "Version 0.X.0"`

### When Project Structure Changes
1. Update `README.md` "Architecture" section
2. Update `WHITEPAPER.md` Section 2 (Architecture)
3. Update file paths in all documentation
4. Update import examples in usage guides

---

## Documentation File Inventory

### **Public-Facing Documents** (User-Oriented)

#### `README.md`
**Purpose:** First thing people see on GitHub. High-level overview and quick start.

**Key Sections:**
- Performance Highlights (update with new benchmarks)
- Quick Start (keep installation steps current)
- Performance vs Competition table
- What is ATLAS-Q? (update when adding major features)
- Architecture diagram and module structure
- Roadmap (update with each release)
- Citation (keep version current)

**When to Update:**
- Every release
- Major performance improvements
- New features that change what ATLAS-Q can do
- Links or usernames change

---

#### `OVERVIEW.md`
**Purpose:** Unified guide for all audiences - from friends/family to researchers. Progressive detail.

**Structure:**
- Simple intro (friends/family can stop after first few sections)
- Real-world examples (accessible to non-technical)
- Honest Q&A section (scaling, quantum computer questions)
- Technical details (for researchers, includes code examples)
- Comparisons and use cases

**Key Sections:**
- What Makes It Different (update with new capabilities)
- Honest Questions, Honest Answers (update with research findings)
- When to Use ATLAS-Q (update use case list)
- Real-World Use Cases (add new examples as they emerge)
- Technical Approach (keep analogies current, NO code until later sections)
- Performance Numbers (update with benchmarks)
- Comparison table (update when competitors change)
- What to Tell People (elevator pitches)

**When to Update:**
- Performance improvements
- New real-world applications
- Competitor landscape changes
- User feedback about clarity
- New research on scaling or hybrid approaches

---

#### `COMPLETE_GUIDE.md`
**Purpose:** Practical tutorials showing how to use ATLAS-Q features.

**Key Sections:**
- Installation instructions (keep current)
- Quick Start examples (test these work!)
- Feature-by-feature tutorials
- Troubleshooting (add common issues)

**When to Update:**
- New features added
- API changes
- Common user questions emerge
- Dependencies change

---

### **Technical Documents** (Research-Oriented)

#### `WHITEPAPER.md`
**Purpose:** Comprehensive technical documentation of architecture and implementation.

**Key Sections:**
- Abstract (update with version achievements)
- Section 2: Architecture (update diagrams and structure)
- Section 3: GPU Acceleration (update kernel details)
- Section 4: Core Features (add new features here)
- Section 5: Performance Benchmarks (update regularly)
- Section 6: Competitive Analysis (update comparisons)
- Section 7: Implementation examples

**When to Update:**
- Major architectural changes
- New features implemented
- Performance benchmarks change
- Competitive landscape shifts

---

#### `RESEARCH_PAPER.md`
**Purpose:** Academic-style paper with mathematical foundations and algorithms.

**Key Sections:**
- Abstract (keep technical achievements current)
- Section 3: Compressed Quantum State Representations
- Section 4: Adaptive MPS Framework
- Section 5: Period-Finding Algorithms
- Appendix: Latest version results

**When to Update:**
- New algorithms added
- Mathematical approaches change
- Benchmark results improve
- Publishing or citing

---

### **Meta Documents**

#### `CHANGELOG.md`
**Purpose:** Historical record of all changes by version.

**Format:**
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Modifications to existing features

### Fixed
- Bug fixes

### Performance
- Performance improvements
```

**When to Update:**
- Every notable change (features, fixes, performance)
- Keep as running log during development
- Finalize when releasing version

**Note:** Examples and demos live in `/scripts/demos/` not `/examples/`

---

## Standard Update Workflows

### Workflow 1: Adding a Major Feature (e.g., New Algorithm)

```bash
# 1. Implement feature
# 2. Add tests
# 3. Update documentation:

# Update core docs
- CHANGELOG.md: Add to [Unreleased] section
- README.md: Add to "What is ATLAS-Q?" if major
- WHITEPAPER.md: Add to Section 4 (Core Features)
- OVERVIEW.md: Update "What Makes It Different" or use cases

# Update tutorials
- COMPLETE_GUIDE.md: Add usage example
- Add demo script to /scripts/demos/

# Update technical docs
- RESEARCH_PAPER.md: Add algorithm description if research-worthy

# 4. Test all examples still work
# 5. Commit with descriptive message
```

---

### Workflow 2: Performance Improvement

```bash
# 1. Implement optimization
# 2. Run benchmarks and document results
# 3. Update all performance numbers:

- README.md: Performance Highlights
- OVERVIEW.md: Performance Numbers section
- WHITEPAPER.md: Section 5 (benchmarks)
- RESEARCH_PAPER.md: Appendix results
- CHANGELOG.md: Performance section

# 4. Save benchmark outputs to /benchmarks/ or /runs/
# 5. Commit with benchmark data
```

---

### Workflow 3: Releasing a New Version

```bash
# 1. Finalize all features for release
# 2. Run full test suite
# 3. Update version numbers:

# Edit these files:
vim README.md # Header: Version X.Y.Z
vim WHITEPAPER.md # Header: Version X.Y.Z
vim RESEARCH_PAPER.md # Header + Appendix
vim OVERVIEW.md # Footer: Version X.Y.Z
vim src/atlas_q/__init__.py # __version__ = 'X.Y.Z'
vim CHANGELOG.md # Move [Unreleased] to [X.Y.Z] - DATE

# 4. Update roadmap
vim README.md # Roadmap section

# 5. Commit and tag
git add -A
git commit -m "Release version X.Y.Z"
git tag -a vX.Y.Z -m "Version X.Y.Z"
git push origin main --tags
```

---

### Workflow 4: Fixing Documentation Links

```bash
# If you change:
# - GitHub username
# - Repository name
# - File locations
# - External links

# Search and replace in all docs:
grep -r "old_text" docs/
grep -r "old_text" *.md

# Files that commonly have links:
- README.md
- CONTRIBUTING.md
- docs/COMPLETE_GUIDE.md
- docs/WHITEPAPER.md
- docs/RESEARCH_PAPER.md
- docs/OVERVIEW.md
- docs/FEATURE_STATUS.md

# Test all links work!
```

---

## Common Pitfalls to Avoid

### Don't:
1. **Update version in one place but forget others** → Use the checklist above
2. **Add features without updating CHANGELOG** → Future you won't remember
3. **Update performance numbers without re-running benchmarks** → Must be reproducible
4. **Change API without updating all examples** → Users will get errors
5. **Forget to spell out acronyms** → MPS → Matrix Product States (MPS) first time
6. **Use marketing language** → Be factual about capabilities and limitations
7. **Reference removed features** → Check for AQED or old project names

### Do:
1. **Test examples after documentation updates** → Code changes break docs
2. **Keep CHANGELOG.md as running log** → Easier than reconstructing later
3. **Be honest about limitations** → Better to underpromise, overdeliver
4. **Update comparison tables** → Competitors improve too
5. **Version everything consistently** → All docs should match
6. **Spell out acronyms for general audience** → Not everyone knows QFT, VQE, etc.

---

## Documentation Principles

### **Consistency**
- Version numbers must match across all files
- GitHub username: `followthsapper` everywhere
- Terminology: Use "ATLAS-Q" consistently (not "ATLAS_Q" or "AtlasQ")
- Acronym style: "Matrix Product States (MPS)" first use, "MPS" after

### **Accuracy**
- Performance numbers must be reproducible
- Don't claim capabilities you haven't tested
- Update comparisons when competitors change
- Keep limitations section honest

### **Clarity**
- Spell out acronyms for non-experts
- Use analogies (regex pattern, data compression)
- Specify "low-to-moderate entanglement" not just "large scale"
- Explain caveats immediately after claims

### **Maintainability**
- Date all benchmarks and results
- Link to actual benchmark scripts
- Keep examples runnable
- Document dependencies and requirements

---

## Key Numbers to Keep Consistent

These numbers appear in multiple documents. If they change, update EVERYWHERE:

### Performance Numbers
- **Memory compression:** 626,000× (30 qubits)
- **Gate throughput:** 77,000+ ops/sec
- **Clifford speedup:** 20× over generic MPS
- **Triton kernel speedup:** 1.5-3× over PyTorch
- **Max demonstrated qubits:** 100,000 (with low-to-moderate entanglement)

### Version Info
- **Current version:** 0.6.0
- **Release date:** October 2025
- **Python requirement:** 3.8+
- **PyTorch requirement:** 2.0+

### Test Coverage
- **Integration tests passing:** 46/46
- **Total tests:** 75+
- **Test categories:** Unit, integration, performance, legacy

---

## Pre-Commit Checklist

Before committing major changes, verify:

- [ ] Version numbers consistent across all docs
- [ ] CHANGELOG.md updated with changes
- [ ] Performance numbers match latest benchmarks
- [ ] All code examples tested and working
- [ ] Links tested (especially GitHub links)
- [ ] Acronyms spelled out on first use
- [ ] No "Tier 1.5" or other marketing fluff
- [ ] Limitations clearly stated
- [ ] Comparisons with competitors fair and current

---

## Questions to Ask Yourself

**Before updating documentation:**
1. Is this change significant enough to document?
2. Which docs need updating? (Use checklist above)
3. Do performance claims need re-benchmarking?
4. Will this change break existing examples?
5. Should this go in user docs, technical docs, or both?

**After updating documentation:**
1. Are version numbers consistent?
2. Do all links work?
3. Do all code examples run?
4. Are acronyms explained?
5. Are limitations honestly stated?

---

### Workflow 5: Publishing to PyPI

```bash
# Prerequisites:
# 1. PyPI account at https://pypi.org
# 2. Set up API token at https://pypi.org/manage/account/token/
# 3. Store token in ~/.pypirc or GitHub secrets

# Option 1: Manual publish
make build # Build package (creates dist/)
make publish-test # Test on TestPyPI first
make publish # Publish to PyPI

# Option 2: Automatic (GitHub Actions)
# Publishing happens automatically when you create a GitHub release:
# 1. Go to GitHub → Releases → Create new release
# 2. Create tag: v0.5.0
# 3. Generate release notes
# 4. Publish release
# → GitHub Actions will automatically build and publish to PyPI

# Verify installation:
pip install atlas-quantum
python -c "import atlas_q; print(atlas_q.__version__)"
```

**PyPI Publishing Checklist:**
- [ ] Version number updated in `pyproject.toml` and `src/atlas_q/__init__.py`
- [ ] CHANGELOG.md updated with release notes
- [ ] All tests passing (`make test`)
- [ ] Package builds successfully (`make build`)
- [ ] README.md renders correctly on PyPI (check at https://pypi.org/project/atlas-quantum/)
- [ ] Dependencies are correct in `pyproject.toml`

---

### Workflow 6: Publishing Docker Images

```bash
# Prerequisites:
# Docker installed and running

# Option 1: Manual build and test locally
make docker-build-gpu # Build GPU image
make docker-build-cpu # Build CPU image

# Test images locally
make docker-run-gpu # Test GPU image
make docker-run-cpu # Test CPU image

# Option 2: Automatic (GitHub Actions)
# Docker images publish automatically when you create a GitHub release:
# 1. Create GitHub release (same as PyPI workflow)
# 2. GitHub Actions will build and push images to GitHub Container Registry
# 3. Images available at:
# - ghcr.io/followthsapper/atlas-quantum:cuda
# - ghcr.io/followthsapper/atlas-quantum:cpu
# - ghcr.io/followthsapper/atlas-quantum:latest (=cuda)

# Users can pull and run:
docker pull ghcr.io/followthsapper/atlas-quantum:cuda
docker run --rm -it --gpus all ghcr.io/followthsapper/atlas-quantum:cuda python
```

**Docker Publishing Checklist:**
- [ ] Dockerfiles build successfully
- [ ] Both GPU and CPU variants tested
- [ ] Package version matches Dockerfile labels
- [ ] Base images (nvidia/cuda, python:slim) are up to date
- [ ] .dockerignore excludes unnecessary files
- [ ] Images run successfully with test scripts

---

### Workflow 7: Complete Release Process

Full workflow for releasing a new version with all distribution channels:

```bash
# Step 1: Prepare release
# ----------------------
# Update version numbers:
vim pyproject.toml # version = "X.Y.Z"
vim src/atlas_q/__init__.py # __version__ = 'X.Y.Z'
vim README.md # **Version X.Y.Z**
vim docs/WHITEPAPER.md # Version X.Y.Z
vim docs/CHANGELOG.md # Add [X.Y.Z] - YYYY-MM-DD

# Run tests
make test
make bench # Run benchmarks if performance changed

# Build and test locally
make build
pip install dist/atlas_q-X.Y.Z-py3-none-any.whl
python -c "import atlas_q; print(atlas_q.__version__)"

# Build and test Docker images
make docker-build-gpu
make docker-build-cpu
make docker-run-gpu

# Step 2: Commit and tag
# ----------------------
git add -A
git commit -m "Release version X.Y.Z"
git tag -a vX.Y.Z -m "Version X.Y.Z

Release notes:
- Feature 1
- Feature 2
- Bug fixes"

git push origin main
git push origin vX.Y.Z

# Step 3: Create GitHub Release
# ------------------------------
# Go to: https://github.com/followthsapper/ATLAS-Q/releases/new
# - Tag: vX.Y.Z
# - Title: ATLAS-Q vX.Y.Z
# - Description: Copy from CHANGELOG.md
# - Publish release

# This triggers GitHub Actions to:
# Publish to PyPI automatically
# Build and publish Docker images
# Create build artifacts

# Step 4: Verify
# --------------
# Wait 5-10 minutes, then verify:
pip install --upgrade atlas-quantum
python -c "import atlas_q; print(atlas_q.__version__)" # Should be X.Y.Z

docker pull ghcr.io/followthsapper/atlas-quantum:cuda
docker run --rm ghcr.io/followthsapper/atlas-quantum:cuda python -c "import atlas_q; print(atlas_q.__version__)"

# Check PyPI page: https://pypi.org/project/atlas-quantum/
# Check GitHub Container Registry: https://github.com/followthsapper/ATLAS-Q/pkgs/container/atlas-quantum
```

---

## Secrets and Credentials

For automated publishing, configure these secrets in GitHub repository settings:

**Required GitHub Secrets:**
- `PYPI_API_TOKEN`: PyPI API token for publishing packages
 - Get from: https://pypi.org/manage/account/token/
 - Scope: Project-level token for atlas-quantum
 - Add at: GitHub → Settings → Secrets → Actions → New repository secret

**Docker Registry:**
- No additional secrets needed - uses `GITHUB_TOKEN` (automatically provided)

---

## Testing Package Installation

After publishing, test installation on fresh environments:

```bash
# Test CPU installation
docker run --rm -it python:3.10-slim bash
pip install atlas-quantum
python -c "import atlas_q; atlas_q.get_quantum_sim()"

# Test GPU installation
docker run --rm -it --gpus all nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 bash
apt-get update && apt-get install -y python3 python3-pip
pip install atlas-quantum[gpu]
python3 -c "import atlas_q; print(atlas_q.__version__)"
```

---

## Tips for Efficient Updates

1. **Use text editor multi-file search** to find all instances of numbers/terms
2. **Keep CHANGELOG as a scratchpad** during development
3. **Update docs immediately** after code changes (don't batch up)
4. **Test examples regularly** even if code didn't change
5. **Use git diff** to verify consistency across files

---

## Document Map Reference

Quick reference for where information lives:

| Information Type | Primary Location | Also Update |
|------------------|------------------|-------------|
| Version number | `__init__.py` | All MD files |
| Performance benchmarks | Whitepaper Section 5 | README, Overview, Research Paper |
| Feature descriptions | Whitepaper Section 4 | README, Overview |
| Usage examples | Usage Guide | README Quick Start |
| Architecture diagrams | Whitepaper Section 2 | README |
| Comparison tables | Overview | Whitepaper Section 6 |
| Algorithm descriptions | Research Paper | Whitepaper |
| Change history | CHANGELOG | N/A |
| Real-world use cases | Overview | COMPLETE_GUIDE |
| Installation steps | README, COMPLETE_GUIDE | N/A |

---

**Last Updated:** October 2025
**Maintainer:** Use this guide every time you make changes!
