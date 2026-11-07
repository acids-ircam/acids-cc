"""Sphinx configuration for cached_conv (built by Read the Docs)."""
import os
import sys
from importlib import metadata

sys.path.insert(0, os.path.abspath(".."))

project = "cached_conv"
author = "ACIDS, Axel Chemla--Romeu-Santos, Nils Demerlé"
copyright = "ACIDS - IRCAM"
try:
    release = metadata.version("cached-conv")
except metadata.PackageNotFoundError:
    release = "3.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    # do not repeat the (huge) torch.nn.Module documentation
    "inherited-members": False,
    "exclude-members": "forward, init_cache",
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

html_theme = "furo"
html_title = f"cached_conv {release}"
html_theme_options = {
    "source_repository": "https://github.com/acids-ircam/acids-cc/",
    "source_branch": "master",
    "source_directory": "docs/",
}

exclude_patterns = ["_build"]
