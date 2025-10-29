# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ATLAS-Q'
copyright = '2025, ATLAS-Q Development Team'
author = 'ATLAS-Q Development Team'
release = '0.6.1'
version = '0.6.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'numpydoc',
    'sphinx_copybutton',
]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Numpydoc settings
numpydoc_show_class_members = True
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    'Tensor': 'torch.Tensor',
    'ndarray': 'numpy.ndarray',
}
numpydoc_xref_ignore = {'optional', 'or', 'of', 'default'}

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': False,
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Intersphinx mapping for cross-references to other projects
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tmp', '*.md']

# The suffix(es) of source filenames.
source_suffix = '.rst'

# The master document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.png'

html_theme_options = {
    "logo": {
        "text": "ATLAS-Q",
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "show_prev_next": True,
    "navbar_align": "left",
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "show_toc_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "primary_sidebar_end": ["sidebar-ethical-ads"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/followthesapper/ATLAS-Q",
            "icon": "fab fa-github-square",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/atlas-quantum/",
            "icon": "fas fa-box",
        },
    ],
}

html_context = {
    "default_mode": "auto"
}

html_sidebars = {
    "**": ["sidebar-nav-bs"]
}

html_css_files = [
    "custom.css",
]

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'ATLAS-Q.tex', 'ATLAS-Q Documentation',
     'ATLAS-Q Development Team', 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'atlas-q', 'ATLAS-Q Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'ATLAS-Q', 'ATLAS-Q Documentation',
     author, 'ATLAS-Q', 'GPU-accelerated quantum tensor network simulator.',
     'Miscellaneous'),
]

# -- Options for copybutton --------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
