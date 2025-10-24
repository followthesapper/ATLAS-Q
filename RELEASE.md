# Release checklist (v0.1.0)

- [ ] Update version in `pyproject.toml`
- [ ] Ensure core module is under `src/quantum_hybrid_system/__init__.py`
- [ ] Run tests locally: `pytest -q`
- [ ] (Optional) Train period head: `python scripts/qih_period_head_train.py`
- [ ] Tag release: `git tag v0.1.0 && git push --tags`
- [ ] Build: `python -m build`
- [ ] Upload: `python -m twine upload dist/*` (or TestPyPI first)
