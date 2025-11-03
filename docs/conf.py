# docs/conf.py
from pathlib import Path
import sys

CONF_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONF_DIR.parent

# Support both layouts:
#   - mono-repo: projects/core/src , projects/procedures/src
#   - classic:   src/
# Add src paths (mono-repo friendly)
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
html_title = "EvoProc"
author = "Malia Barker"
copyright = "2025, Malia Barker"
release = "1.0"

# --- Extensions / autodoc ---
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

# Mock heavy/optional deps so RTD/local builds don’t break
autodoc_mock_imports = [
    "ollama", "numpy", "pandas", "torch", "sklearn",
    "pydantic", "typing_extensions", "jsonschema"
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Paths relative to docs/
templates_path = ["source/_templates"] if (CONF_DIR / "source/_templates").exists() else []
def setup(app):
    app.add_css_file("custom.css")  # create this file if you want tweaks
html_static_path = ["source/_static"]   if (CONF_DIR / "source/_static").exists() else []

# Themes and styles
html_theme = "furo"
html_logo = "source/_static/logo.png"                 # optional
html_favicon = "source/_static/favicon.png"           # optional
html_theme_options = {
    "light_logo": "logo.png",                         # files under docs/source/_static
    "dark_logo": "logo.png",
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
}

# Nicer API pages
autodoc_typehints = "description"
autodoc_class_signature = "separated"
napoleon_use_param = True
napoleon_use_rtype = True

# Napoleon/typing polish
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_attr_annotations = True
typehints_fully_qualified = False

# --- MyST-NB configuration ----------------------------------------------------
# Do not execute notebooks during doc build (fast)
nb_execution_mode = "off"
nb_output_stderr = "show"
