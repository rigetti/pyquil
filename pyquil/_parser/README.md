The Quil Parser
===============

Introduction
------------

This package contains a number of items useful for parsing Quil programs, both with or
without pyQuil. It uses [Lark](https://github.com/lark-parser/lark) for quick and
efficient parsing of Quil.

- `grammar.lark` defines the Quil grammar in the Lark grammar format.
- `parser.py` translates the Lark parse tree into pyQuil instruction objects.

References
----------

- Lark documentation: https://lark-parser.readthedocs.io/
