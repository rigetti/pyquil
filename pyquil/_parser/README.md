The Quil Parser
===============

Introduction
------------

This package contains a number of items useful for parsing Quil
programs, both with or without pyQuil. It uses the ANTLR4 parser
generator framework.

`Quil.g4` - The reference grammar for Quil. This is a [symbolic
link](https://en.wikipedia.org/wiki/Symbolic_link) which points to a
[git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
found in `docs/quil`. Formerly, the grammar lived in this package,
but, to prevent duplication, was moved to a [canonical
location](https://github.com/rigetti/quil/tree/master/grammars). The
grammar is used to generate the Python parser, and may also be used to
generate parsers in any language that has support for ANTLR4 (see
below).

`PyQuilListener.py` - When the parser successfully consumes a Quil
program it invokes various callbacks after encountering different
parts of the AST (gates, expressions, etc). This is the code that
creates the pyQuil instructions.

`gen3/` - Generated parser code for Python 3. Should be checked in but
not hand modified.

Running ANTLR
-------------

1. Install [ANTLR4](http://www.antlr.org/)
2. Navigate to the top-level pyQuil directory
3. Generate the Python 3 parser code: `make generate-parser`

Step (3) will do the work of initializing and populating the
`docs/quil` submodule.

References
----------

Excellent ANTLR tutorial: https://tomassetti.me/antlr-mega-tutorial
