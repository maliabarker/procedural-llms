from pathlib import Path
import sys

CONF_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONF_DIR.parent

# Point Sphinx at your package src dirs (unchanged)
CANDIDATE_SRC_DIRS = [
    REPO_ROOT / "projects" / "core" / "src",
    REPO_ROOT / "projects" / "procedures" / "src",
    REPO_ROOT / "src",
]

for p in CANDIDATE_SRC_DIRS:
    if p.is_dir():
        sys.path.insert(0, str(p))
        
print("conf.py sys.path additions:", [str(p) for p in CANDIDATE_SRC_DIRS if p.is_dir()])

# --- Project info ---
project = "EvoProc"
author = "Malia Barker"
copyright = "2025, Malia Barker"
release = "1.0"

# Tell Sphinx the master document is now docs/index.*
# root_doc = "index"

# --- Extensions ---
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_nb",
    "sphinx_design"
]
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# Mock heavy/optional deps
autodoc_mock_imports = [
    "ollama", "numpy", "pandas", "torch", "sklearn",
    "pydantic", "typing_extensions", "jsonschema"
]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# Now these are relative to docs/ (no "source/" prefix)
templates_path = ["_templates"] if (CONF_DIR / "_templates").exists() else []
def setup(app):
    if (CONF_DIR / "_static").exists():
        app.add_css_file("custom.css")
html_static_path = ["_static"] if (CONF_DIR / "_static").exists() else []

# Theme
html_theme = "furo"
html_title = "EvoProc"
html_logo = "_static/logo.png"       # if present
html_favicon = "_static/favicon.png" # if present
html_theme_options = {
    "light_logo": "logo.png",
    "dark_logo": "logo.png",
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
}

# MyST-NB execution (optional)
nb_execution_mode = "off"  # or "cache", "auto", etc.

# Napoleon/typehints polish
autodoc_typehints = "description"
autodoc_class_signature = "separated"
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_attr_annotations = True
typehints_fully_qualified = False
