# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
import shutil
from sphinx_polyversion import load
from sphinx_polyversion.git import GitRef

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "BayesFlow"
author = "The BayesFlow authors"
copyright = "2023, BayesFlow authors (lead maintainer: Stefan T. Radev)"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["references.bib"]

numpydoc_show_class_members = False
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ["http", "https", "mailto"]
autodoc_default_options = {
    "members": "var1, var2",
    "special-members": "__call__,__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "member-order": "bysource",
}

# Define shorthand for external links:
extlinks = {
    "mainbranch": ("https://github.com/bayesflow-org/bayesflow/blob/master/%s", None),
}

coverage_show_missing_items = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
autosummary_imported_members = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_title = "BayesFlow: Amortized Bayesian Inference"

# Add any paths that contain custom _static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin _static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_theme_options = {
    "repository_url": "https://github.com/bayesflow-org/bayesflow",
    "repository_branch": "master",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "logo": {"alt-text": "BayesFlow"},
}
html_logo = "_static/bayesflow_hex.png"
html_favicon = "_static/bayesflow_hex.ico"
html_baseurl = "https://www.bayesflow.org/"
html_js_files = ["https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"]
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "icon-links.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
        "versioning.html",
    ],
}

todo_include_todos = True

# do not execute jupyter notebooks when building docs
nb_execution_mode = "off"

# download notebooks as .ipynb and not as .ipynb.txt
html_sourcelink_suffix = ""

suppress_warnings = [
    f"autosectionlabel._examples/{filename.split('.')[0]}"
    for filename in os.listdir("../../examples")
    if os.path.isfile(os.path.join("../../examples", filename))
]  # Avoid duplicate label warnings for Jupyter notebooks.

remove_from_toctrees = ["_autosummary/*"]

autosummmary_generate = True

# sphinx-multiversion: select tags, branches and remotes
smv_tag_whitelist = r"^(v1.1.6)$"
# smv_branch_whitelist = r"^(|doc-autosummary|master|dev)$"
smv_branch_whitelist = r"^(master)$"
smv_remote_whitelist = None


# move files around if necessary
def copy_files_handler(app, config):
    print("TODO")
    return
    current_version = config["smv_current_version"]
    current_metadata = config["smv_metadata"][current_version]
    basedir = current_metadata["basedir"]
    sourcedir = current_metadata["sourcedir"]

    print("Current version:", current_version)
    print("Basedir:", basedir)
    print("Metadata:", current_metadata)

    examples_src = os.path.join(basedir, "examples")
    examples_dst = os.path.join(sourcedir, "_examples")
    if os.path.exists(examples_src):
        shutil.copytree(examples_src, examples_dst, dirs_exist_ok=True)
    examples_in_progress = os.path.join(examples_dst, "in_progress")
    if os.path.exists(examples_in_progress):
        shutil.rmtree(examples_in_progress)
    contributing_src = os.path.join(basedir, "CONTRIBUTING.md")
    contributing_dst = os.path.join(sourcedir, "contributing.md")
    if os.path.exists(contributing_src):
        shutil.copy2(contributing_src, contributing_dst)
    installation_src = os.path.join(basedir, "INSTALL.rst")
    installation_dst = os.path.join(sourcedir, "installation.rst")
    if os.path.exists(installation_src):
        shutil.copy2(installation_src, installation_dst)


def cleanup_handler(app, exception):
    print("Done, what now?")


def setup(app):
    app.connect("config-inited", copy_files_handler)
    app.connect("build-finished", cleanup_handler)


data = load(globals())
current: GitRef = data["current"]
