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

from inspect import getsourcefile


sys.path.insert(0, os.path.abspath('..'))  # Source code dir relative to this file

# -- Project information -----------------------------------------------------

project = 'GeometricKernels'
copyright = '2022-2024, the GeometricKernels Contributors'
author = 'The GeometricKernels Contributors'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'autoapi.extension',
    'sphinx.ext.viewcode',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx_math_dollar',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'nbsphinx',
    'nbsphinx_link',
    'sphinxcontrib.bibtex',
]

# autoapi
autoapi_dirs = ["../geometric_kernels"]
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autoapi_add_toctree_entry = False
autoapi_keep_files = True
autoapi_python_class_content = "class"  # we handle __init__ and __new__ below
autoapi_member_order = "groupwise"
# ignore these files to suppress warning multiple dispatch
autoapi_ignore = [f'**/lab_extras/{b}**' for b in ["torch", "jax", "tensorflow", "numpy"]]
autoapi_options = [
    "members",
    # "private-members",
    # "special-members",
    "imported-members",
    "show-inheritance",
]

# Only skip certain special members if they have an empty docstring.
privileged_special_members = ["__init__", "__new__", "__call__"]
def never_skip_init_or_new(app, what, name, obj, would_skip, options):
    if any(psm in name for psm in privileged_special_members):
        return not bool(obj._docstring)  # skip only if the docstring is empty
    return would_skip


# Add any paths that contain templates here, relative to this directory.
# The templates are as in https://stackoverflow.com/a/62613202
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

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
    'scripts/bootstrap.min.js',
    ( # require.js might be needed for nbspinx to render plotly plots.
        'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js',
        {
            'crossorigin': 'anonymous',
            'integrity': 'sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=',
        },
    ),
]

# Theme-specific options. See theme docs for more info
html_context = {
  'display_github': True,
  'github_user': 'geometric-kernels',
  'github_repo': 'GeometricKernels',
  'github_version': 'main/docs/'
}

# For sphinx_math_dollar (see https://www.sympy.org/sphinx-math-dollar/)

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}

# -- nbsphinx ----------------------------------------------------------------

nbsphinx_execute = 'never'

# Don't load require.js because it conflicts with bootstrap.min.js being
# loaded after it. If you need require.js, load it manually by adding it
# to the html_js_files list.
nbsphinx_requirejs_path = ""

# -- pandoc (required by nbsphinx) -------------------------------------------

# Get path to directory containing this file, conf.py.
DOCS_DIRECTORY = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))

def ensure_pandoc_installed(_):
    import pypandoc

    # Download pandoc if necessary. If pandoc is already installed and on
    # the PATH, the installed version will be used. Otherwise, we will
    # download a copy of pandoc into docs/bin/ and add that to our PATH.
    pandoc_dir = os.path.join(DOCS_DIRECTORY, "bin")
    # Add dir containing pandoc binary to the PATH environment variable
    if pandoc_dir not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + pandoc_dir
    pypandoc.ensure_pandoc_installed(
        targetfolder=pandoc_dir,
        delete_installer=True,
    )


# -- Bibtex ------------------------------------------------------------------

bibtex_bibfiles = ['references.bib']
bibtex_reference_style = "author_year"

import pybtex.plugin
from pybtex.style.formatting.plain import Style as UnsrtStyle
from pybtex.style.template import field, sentence

class GKUnsrtStyle(UnsrtStyle):

    def format_title(self, e, which_field, as_sentence=True):
        formatted_title = field(which_field)  # Leave the field exactly as is.
        if as_sentence:
            return sentence [ formatted_title ]
        else:
            return formatted_title

pybtex.plugin.register_plugin('pybtex.style.formatting', 'gkunsrt', GKUnsrtStyle)

# -- Connect stuff -----------------------------------------------------------

def setup(sphinx):
    sphinx.connect("builder-inited", ensure_pandoc_installed)
    sphinx.connect("autoapi-skip-member", never_skip_init_or_new)
