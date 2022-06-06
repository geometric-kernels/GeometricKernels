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

sys.path.insert(0, os.path.abspath('..'))  # Source code dir relative to this file

# -- Project information -----------------------------------------------------

project = 'GeometricKernels'
copyright = '2022, the GeometricKernels contributors'
author = 'The GeometricKernels Contributors'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.viewcode",
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
]

# autoapi
extensions.append("autoapi.extension")
autoapi_dirs = ["../geometric_kernels"]
autodoc_typehints = 'description'
autoapi_add_toctree_entry = False
autoapi_keep_files = True
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
# ignore these files to surpress warning multiple dispatch
autoapi_ignore = [f'**/lab_extras/{b}**' for b in ["torch", "jax", "tensorflow", "numpy"]]
autoapi_options = [
    "members",
    "private-members",
    "special-members",
    "imported-members",
    "show-inheritance",
]


# Add any paths that contain templates here, relative to this directory.
# The templates are as in https://stackoverflow.com/a/62613202
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/bootstrap_namespaced.css',
]

html_js_files = [
    'js/bootstrap.min.js',
]
