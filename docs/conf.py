project = "pertTF"
copyright = "2026, Li Lab"
author = "Li Lab"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# perttf's runtime deps (torch, scanpy, jax, ...) are installed via the `docs`
# extra in pyproject.toml so autodoc can import real modules instead of mocks.
# flash_attn/flash_attn_interface are intentionally left uninstalled: they
# require a CUDA build unavailable on the doc build host, and
# perttf/model/modules.py already falls back gracefully via
# `try/except ImportError` when they're absent.

# Tutorial notebooks depend on GPU hardware, real data and trained checkpoints
# that aren't available during a doc build, so notebooks render their already
# stored outputs rather than being re-executed here.
nb_execution_mode = "off"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "furo"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
