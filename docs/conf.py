# docs/conf.py
from pathlib import Path
import sys

CONF_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONF_DIR.parent
SRC_DIR = REPO_ROOT/"src"

# Make your package importable (works on any machine, any CWD)
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
else:
    sys.path.insert(0, str(REPO_ROOT))

project = "llm_procedure_generation_ga"
author = "Malia Barker"
copyright = "2025, Malia Barker"
release = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
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
typehints_fully_qualified = False
napoleon_use_ivar = True
napoleon_attr_annotations = True
autodoc_member_order = "bysource"


# Keep heavy libs mocked so autodoc doesn’t choke on other machines
autodoc_mock_imports = ["ollama", "numpy", "pandas", "torch", "sklearn", "pydantic", "typing_extensions"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# These paths are now relative to docs/ (not source/)
templates_path = ["source/_templates"] if (CONF_DIR / "source/_templates").exists() else []
html_static_path = ["source/_static"] if (CONF_DIR / "source/_static").exists() else []

html_theme = "sphinx_rtd_theme"
exclude_patterns = []

print("Using conf.py:", __file__)