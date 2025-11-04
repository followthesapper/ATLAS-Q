# Security Policy

## Supported Versions

Security updates are provided for the following versions:

| Version | Supported |
| ------- | ------------------ |
| 0.5.x | :white_check_mark: |
| < 0.5 | :x: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in ATLAS-Q, please follow these steps:

### For Non-Sensitive Security Issues

For issues that are not immediately exploitable or do not expose sensitive data:

1. Open a [GitHub Issue](https://github.com/followthesapper/ATLAS-Q/issues)
2. Label it with "security"
3. Provide a clear description of the issue
4. Include steps to reproduce if applicable

### For Sensitive Security Issues

For critical vulnerabilities or issues that could be exploited:

1. **Do NOT** open a public GitHub issue
2. Contact the maintainers directly through:
 - GitHub Security Advisory (preferred): Go to "Security" → "Report a vulnerability"
 - Or email the project maintainers (see GitHub profile)
3. Provide:
 - Description of the vulnerability
 - Potential impact
 - Steps to reproduce
 - Any proof-of-concept code (if applicable)
 - Suggested fix (if you have one)

### What to Expect

- **Acknowledgment:** Within 48 hours of your report
- **Assessment:** We'll assess severity and impact within 1 week
- **Fix:** Security fixes are prioritized and released as quickly as possible
- **Credit:** Security researchers who report valid issues will be credited (unless they prefer to remain anonymous)

## Security Best Practices for Users

When using ATLAS-Q:

### 1. Dependency Management
```bash
# Regularly update dependencies
pip install --upgrade atlas-quantum

# Check for vulnerabilities
pip-audit
```

### 2. Docker Security
```bash
# Pull specific versions, not 'latest'
docker pull ghcr.io/followthesapper/atlas-q:cuda

# Run as non-root (already configured in images)
docker run --user atlasq ...

# Use read-only filesystem when possible
docker run --read-only ...
```

### 3. Environment Security

**Never commit:**
- API keys or tokens
- Private keys (.pem, .key files)
- Environment files (.env)
- Credentials

These are already in `.gitignore`, but always verify before committing.

### 4. Input Validation

When using ATLAS-Q in applications that accept user input:
- Validate qubit counts and circuit parameters
- Sanitize file paths if loading external circuits
- Set resource limits for large simulations

## Security Audit

A comprehensive security audit was performed on October 27, 2025. See [SECURITY_AUDIT.md](SECURITY_AUDIT.md) for details.

**Key Findings:**
- No known vulnerabilities in dependencies
- No hardcoded secrets
- Secure Docker images (non-root user)
- Safe code practices (no eval/exec)
- Trusted publishing to PyPI

## Known Security Considerations

### GPU Memory Exhaustion

Large quantum simulations can exhaust GPU memory. This is expected behavior, not a vulnerability. Users should:
- Set appropriate qubit limits
- Monitor GPU memory usage
- Use bond dimension limits in MPS simulations

Example:
```python
# Limit bond dimension to prevent memory exhaustion
mps = AdaptiveMPS(n_qubits=50, chi_max=64)
```

### Numerical Stability

Quantum simulations involve complex floating-point operations. For production use:
- Validate simulation results against known benchmarks
- Use appropriate error thresholds
- Be aware of numerical precision limits

## Security Features

ATLAS-Q includes several security-conscious design choices:

1. **No External Network Calls:** Pure computational library, no network dependencies
2. **Type Safety:** Extensive type hints and validation
3. **Memory Safety:** PyTorch/NumPy handle memory management
4. **Reproducibility:** Deterministic simulations for verification

## Dependency Security

ATLAS-Q depends on:
- **NumPy/SciPy** - Industry standard, well-audited
- **PyTorch** - Security team at Meta
- **Triton** - Maintained by OpenAI
- **Matplotlib** - Long-standing, stable

All dependencies are actively maintained with security updates.

## CI/CD Security

Our build pipeline uses:
- **Trusted Publishing** to PyPI (OIDC, no API keys)
- **GitHub Actions** with minimal permissions
- **Dependency scanning** in CI
- **Code quality checks** (ruff, black)

## Updates and Patches

Security patches are released as:
- **Patch versions** (0.5.x) for security fixes
- **Minor versions** (0.x.0) for security improvements
- Released immediately upon discovery and verification

Subscribe to releases on GitHub to stay informed.

## Acknowledgments

We thank the security research community for helping keep ATLAS-Q safe. If you've reported a security issue, you'll be listed here (with your permission):

- (No security issues reported yet)

---

**Last Updated:** October 27, 2025
**Next Review:** Before v1.0.0 release
