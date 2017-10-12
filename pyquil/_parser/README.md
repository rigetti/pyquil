# Quil Parser

## Introduction

This package contains a number of items useful for parsing Quil programs, both with or without PyQuil. It uses the 
ANTLR4 parser generator framework.

`Quil.g4` - This is the definitive reference grammar for Quil as described in the Quil paper. It is used to generate
the Python parser as well as parsers in other programming languages.

`PyQuilListener.py` - When the parser successfully consumes a Quil program it invokes various callbacks after
encountering different parts of the AST (gates, expressions, etc). This is the code that creates the PyQuil instructions.

`gen2/` - Generated parser code for Python 2. Should be checked in but not hand modified.

`gen3/` - Generated parser code for Python 3. Should be checked in but not hand modified.

## Running ANTLR

1. Install ANTLR4 and alias it to `antlr4`
2. cd to this directory
3. Generate the Python 2 parser code: `antlr4 -Dlanguage=Python2 -o gen2 Quil.g4` 
3. Generate the Python 3 parser code: `antlr4 -Dlanguage=Python3 -o gen3 Quil.g4` 

## References

Excellent ANTLR tutorial: https://tomassetti.me/antlr-mega-tutorial

ANTLR download: http://www.antlr.org/
