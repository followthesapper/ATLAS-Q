# ATLAS-Q Documentation

This directory contains the source files for ATLAS-Q documentation, built using Sphinx with the pydata-sphinx-theme.

## Building Documentation Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install -r requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in a browser to view.

### Build Other Formats

```bash
make latexpdf # PDF via LaTeX
make epub # EPUB format
make man # Man pages
```

### Clean Build Directory

```bash
make clean
```

### Live Rebuild (Development)

For automatic rebuilding during development:

```bash
pip install sphinx-autobuild
sphinx-autobuild . _build/html
```

Then open http://127.0.0.1:8000 in your browser. The documentation will automatically rebuild when you save changes.

## Documentation Structure

The documentation follows the Diátaxis Framework:

- **Tutorials** (`user_guide/tutorials/`): Step-by-step learning paths
- **How-To Guides** (`user_guide/howtos/`): Problem-oriented solutions
- **Explanations** (`user_guide/explanations/`): Understanding-oriented discussions
- **Reference** (`reference/`): Technical API documentation

```
docs/
 conf.py # Sphinx configuration
 index.rst # Main landing page
 installation.rst # Installation guide
 quickstart.rst # Quick start guide
 user_guide/
 tutorials/ # Learning-oriented tutorials
 howtos/ # Task-oriented how-tos
 explanations/ # Understanding-oriented explanations
 reference/ # API reference (auto-generated)
 developer/ # Developer documentation
 examples/ # Example gallery
 faq.rst # Frequently asked questions
 citing.rst # Citation information
 _build/ # Generated documentation (gitignored)
```

## Writing Documentation

### reStructuredText Basics

Documentation is written in reStructuredText (.rst) format. Basic syntax:

```rst
Section Title
=============

Subsection
----------

Subsubsection
^^^^^^^^^^^^^

**Bold text** and *italic text*

.. code-block:: python

 # Python code
 from atlas_q import AdaptiveMPS
 mps = AdaptiveMPS(10, bond_dim=8)

- Bulleted
- List

1. Numbered
2. List

:doc:`cross-reference`
:class:`atlas_q.adaptive_mps.AdaptiveMPS`
:func:`atlas_q.mpo_ops.expectation_value`
```

### Docstring Format

ATLAS-Q uses numpydoc-style docstrings:

```python
def function_name(param1, param2, option='default'):
 """
 Brief one-line description.

 Extended description with details about what the function does
 and any important implementation considerations.

 Parameters
 ----------
 param1 : type
 Description of param1.
 param2 : type
 Description of param2.
 option : {'default', 'alternative'}, optional
 Description. Default is 'default'.

 Returns
 -------
 result : type
 Description of return value.

 Raises
 ------
 ValueError
 When param1 is invalid.

 See Also
 --------
 related_function : Brief description.

 Examples
 --------
 >>> result = function_name(1, 2)
 >>> print(result)
 expected_output
 """
```

### Cross-References

Link to other documentation:

- Documentation pages: `:doc:`installation``
- Sections: `:ref:`section-label``
- Classes: `:class:`atlas_q.adaptive_mps.AdaptiveMPS``
- Functions: `:func:`atlas_q.mpo_ops.expectation_value``
- Modules: `:mod:`atlas_q.mpo_ops``

### Mathematical Notation

Use LaTeX for math:

```rst
Inline math: :math:`E = mc^2`

Display math:

.. math::

 |\psi\rangle = \sum_{i=1}^n c_i |i\rangle
```

## Contributing to Documentation

1. Follow the existing structure and style
2. Use proper reStructuredText syntax
3. Include code examples that work
4. Cross-reference related documentation
5. Build locally to check for errors
6. Submit pull request with clear description

## Continuous Integration

Documentation is automatically built and deployed to GitHub Pages on every push to the main branch. The workflow is defined in `.github/workflows/docs.yml`.

Build status: Check the Actions tab in the GitHub repository.

## Viewing Online Documentation

Published documentation: https://followthsapper.github.io/ATLAS-Q/

## Troubleshooting

### Sphinx Build Errors

If you encounter build errors:

1. Check syntax: `rst-lint <file>.rst`
2. Clean build: `make clean && make html`
3. Check warnings: `make html 2>&1 | grep WARNING`
4. Verify imports: Ensure ATLAS-Q is installed (`pip install -e ..`)

### Missing Cross-References

If cross-references don't resolve:

1. Check spelling and capitalization
2. Verify target exists in source code
3. Ensure intersphinx mapping is configured (for external references)
4. Run `make html` and check warnings

### LaTeX/PDF Build Issues

PDF building requires LaTeX:

```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra

# macOS
brew install --cask mactex

# Then build
make latexpdf
```

## Documentation Standards

- Keep paragraphs concise (3-5 sentences)
- Use active voice
- Include working code examples
- Add cross-references liberally
- Follow numpydoc conventions for docstrings
- Test all code examples before committing
- Check for typos and grammatical errors
- Maintain consistent terminology

## Style Guide

- **Code**: Use `code` for inline code, functions, variables
- **File paths**: Use `path/to/file`
- **Modules**: Use `:mod:`atlas_q.module_name``
- **Classes**: Use `:class:`ClassName``
- **Functions**: Use `:func:`function_name``
- **Parameters**: Use *emphasis* for parameter names in prose
- **Terminology**: Be consistent (e.g., "bond dimension" not "bond dim")

## Getting Help

- Sphinx documentation: https://www.sphinx-doc.org/
- reStructuredText primer: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
- Numpydoc guide: https://numpydoc.readthedocs.io/
- pydata-sphinx-theme: https://pydata-sphinx-theme.readthedocs.io/

For questions about ATLAS-Q documentation, open an issue on GitHub.
