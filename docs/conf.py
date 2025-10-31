# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# --- Make your package importable for autodoc ---
# If you use a src/ layout, install the package or add src to path.
# Best practice: install your package; fallback path-insertion works locally.
sys.path.insert(0, os.path.abspath('../src'))    # src/ layout

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'llm_procedure_generation_ga'
copyright = '2025, Malia Barker'
author = 'Malia Barker'
release = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",           # Google/NumPy docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",                   # Markdown support
    "sphinx_copybutton",
]
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"

# Make unresolved heavy deps not break builds (e.g., C libs, GPUs, Ollama)
autodoc_mock_imports = [
    "ollama", "numpy", "pandas", "torch", "sklearn"  # add anything troublesome
]

# Optional: clean type hints in signatures
typehints_fully_qualified = False

# Optional intersphinx links
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
}

templates_path = ['_templates']
exclude_patterns = []

html_static_path = ['_static']