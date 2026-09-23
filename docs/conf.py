project = "pertTF"
copyright = "2026, Li Lab"
author = "Li Lab"

extensions = [
    "myst_nb",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Tutorial notebooks depend on GPU hardware, real data and trained checkpoints
# that aren't available during a doc build, so notebooks render their already
# stored outputs rather than being re-executed here.
nb_execution_mode = "off"

html_theme = "furo"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
