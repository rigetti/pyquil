# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import subprocess

import sphinx_rtd_theme
from sphinx.ext.doctest import doctest

from pyquil import __version__

project = "pyQuil"
copyright = "2021, Rigetti Computing"
author = "Rigetti Computing"
# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinxcontrib.jquery",
    "nbsphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []
source_suffix = [".rst", ".md"]

doctest_default_flags = (
    doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL | doctest.DONT_ACCEPT_TRUE_FOR_1 | doctest.NORMALIZE_WHITESPACE
)

root_doc = "index"
autosummary_generate = True
autoclass_content = "both"
pygments_style = "sphinx"
todo_include_todos = True
# intersphinx_mapping = { "python": ("https://docs.python.org/3/", None) }
autodoc_type_aliases = {"QPUCompilerAPIOptions": "pyquil.api._qpu_compiler.QPUCompilerAPIOptions"}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ["_static"]
html_css_files = ["theme_overrides.css"]  # override wide tables in RTD theme
htmlhelp_basename = "pyQuildoc"

latex_elements = {}
# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class], toctree_only).
latex_documents = [(root_doc, "pyQuil.tex", "pyQuil Documentation", "Rigetti Computing", "manual", False)]

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(root_doc, "pyquil", "pyQuil Documentation", [author], 1)]

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category, toctree_only)
texinfo_documents = [
    (
        root_doc,
        "pyQuil",
        "pyQuil Documentation",
        author,
        "pyQuil",
        "One line description of project.",
        "Miscellaneous",
        False,
    )
]

mathjax3_config = {
    "tex": {
        "macros": {
            "sket": ["\\left|\\left. #1 \\right\\rangle\\!\\right\\rangle", 1],
            "sbra": ["\\left\\langle\\!\\left\\langle #1 \\right.\\right|", 1],
            "sbraket": [
                "\\left\\langle\\!\\left\\langle #1 | #2 \\right\\rangle\\!\\right\\rangle",
                2,
            ],
            "ket": ["\\left| #1 \\right\\rangle", 1],
            "bra": ["\\left\\langle #1 \\right|", 1],
            "braket": ["\\left\\langle #1 | #2 \\right\\rangle", 2],
            "vect": ["\\text{vec}\\left(#1\\right)", 1],
            "tr": ["\\text{Tr}\\left(#1\\right)", 1],
        }
    }
}

suppress_warnings = [
    # TODO: Re-enable these warnings once Sphinx resolves this open issue:
    # https://github.com/sphinx-doc/sphinx/issues/4961
    "ref.python",
]

# fun little hack to always build the rst changelog from the markdown

dirname = os.path.dirname(__file__)

def builder_inited_handler(app):
    import pandoc

    infile = f"{dirname}/../../CHANGELOG.md"
    outfile = f"{dirname}/changes.rst"

    input = pandoc.read(source=None, file=infile, format="markdown")
    pandoc.write(input, file=outfile, format="rst")

    os.environ["SPHINX_APIDOC_OPTIONS"] = "members,undoc-members,show-inheritance,no-index"

    result = subprocess.run(
        [
            "sphinx-apidoc",
            "--module-first",
            "--force",
            "--append-syspath",
            "--separate",
            f"--output-dir={dirname}/apidocs",
            f"{dirname}/../../pyquil",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode == 0:
        print("sphinx-apidoc ran successfully.")
    else:
        print(f"sphinx-apidoc failed with return code {result.returncode}.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)


def setup(app):
    app.connect("builder-inited", builder_inited_handler)


# end: fun little hack
