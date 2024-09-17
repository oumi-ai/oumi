import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Learning Machines"
copyright = "2024, Learning Machines"
author = "Learning Machines"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

main_doc = "index"
pygments_style = "default"  # see https://pygments.org/demo/
add_module_names = True

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.duration",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_parser",
    "nbsphinx",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".ipynb": "nbsphinx",
}

nbsphinx_execute = "never"
nbsphinx_allow_errors = True

napoleon_include_special_with_doc = True
napoleon_use_ivar = True
napoleon_numpy_docstring = False
napoleon_google_docstring = True

coverage_statistics_to_stdout = True
coverage_statistics_to_report = True
coverage_show_missing_items = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False
html_show_sphinx = False
html_static_path = ["_static"]
html_theme_options = {
    "navigation_with_keys": True,
    "repository_url": "https://github.com/oumi-ai/oumi",
    "use_repository_button": True,
    "repository_branch": "main",
    "show_toc_level": 3,
}
