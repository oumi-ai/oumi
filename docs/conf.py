# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Oumi"
copyright = "2025 - Oumi"
author = "Oumi Community"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

main_doc = "index"

extensions = [
    "myst_nb",  # implicitly enables myst_parser
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_rtd_theme",
    # "sphinx_term.termynal",  # Not currently used, but could be useful in the future
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.typer",
]

source_suffix = {
    ".rst": "restructuredtext",
}
nb_execution_mode = "off"
nitpick = True
myst_heading_anchors = 5

napoleon_use_admonition_for_examples = False
napoleon_include_special_with_doc = True
napoleon_use_ivar = True
napoleon_numpy_docstring = False
napoleon_google_docstring = True
napoleon_custom_sections = [
    ("Data Fields", "params_style"),
    ("License", "notes"),
    ("Citations", "admonition"),
    ("Citation", "admonition"),
]

coverage_statistics_to_stdout = True
coverage_statistics_to_report = True
coverage_show_missing_items = True

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "api/modules.rst",
    "api/oumi.models.experimental.*",
]

trim_doctest_flags = True

# Importing these modules causes errors in the docs build
autodoc_mock_imports = ["oumi.models.experimental"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# See https://sphinx-themes.org/#themes for theme options
# See https://sphinx-book-theme.readthedocs.io/en/stable/ for more information
# on the sphinx-book-theme
html_theme = "sphinx_book_theme"
html_show_sourcelink = True
html_show_sphinx = False
html_title = "Oumi"
html_favicon = "_static/logo/favicon.png"
html_static_path = ["_static"]

add_module_names = True

html_theme_options = {
    # Sphinx options
    "navigation_with_keys": True,
    # Sphinx-book-theme options
    "home_page_in_toc": False,  # Hide "Home" in the sidebar
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
    "path_to_docs": "docs",
    "repository_branch": "main",
    "repository_url": "https://github.com/oumi-ai/oumi",
    "show_toc_level": 3,
    "use_download_button": True,
    "use_edit_page_button": True,
    "use_fullscreen_button": False,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_source_button": True,
    "logo": {
        # Note: the light logo is intentionally used for the dark theme
        "image_light": "_static/logo/oumi_logo_dark.png",
        "image_dark": "_static/logo/oumi_logo_light.png",
        "alt_text": "Oumi Documentation - Home",
    },
    "analytics": {
        "google_analytics_id": "G-YZE0YFDLPT",
    },
    "switcher": {
        "json_url": "https://oumi.ai/docs/version.json",
        "version_match": os.environ.get("OUMI_VERSION", "latest"),
    },
    "navbar_start": ["version-switcher"],
}

# see https://pygments.org/demo/ for options
pygments_style = "github-dark"

# Mapping for intersphinx
# modeule name -> (url, inventory file)
intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en", None),
    "trl": ("https://huggingface.co/docs/trl/main/en", None),
    "datasets": ("https://huggingface.co/docs/datasets/main/en", None),
    "bitsandbytes": ("https://huggingface.co/docs/bitsandbytes/main/en", None),
}

# Disable all reftypes for intersphinx
# Reftypes need to be pre-fixed with :external: to be linked
intersphinx_disabled_reftypes = ["*"]

bibtex_bibfiles = ["citations.bib"]
bibtex_encoding = "utf-8"

myst_enable_extensions = [
    "colon_fence",  # Allows for directive blocks to be denoted by :::
    "tasklist",  # Enables GitHub-style task lists
    "fieldlist",  # Allows using rst-like field lists in markdown
    "deflist",  # Allows using definition lists
    "attrs_inline",  # Allows inline attributes
]

suppress_warnings = [
    # Ignore warning about non-consecutive header increase, e.g. H1 followed by H3
    "myst.header",
    # Ignore warnings from autodoc
    # "autodoc",
]

extlinks = {
    "gh": ("https://github.com/oumi-ai/oumi/blob/main/%s", None),
}
