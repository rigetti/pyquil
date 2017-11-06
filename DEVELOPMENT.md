## Development and Testing

We use pytest (version > 3.0) and mock for testing. Tests can be run from the top-level directory using:
```
python setup.py test
```
If you want to test in multiple enviornments, such as Python 2.7 and Python 3.6, then you can use `tox`. This is done with:
```
pip install tox
tox
```

## Building the Docs

We use sphinx to build the documentation. To do this, navigate into pyQuil's top-level directory and run:

```
sphinx-build -b html docs/source docs/_build
```
To view the docs navigate to the newly-created `docs/_build` directory and open
the `index.html` file in a browser. Note that we use the Read the Docs theme for
our documentation, so this may need to be installed using `pip install sphinx_rtd_theme`.

## Working with the parser

Working with the ANTLR parser involves some extra steps, 
see [pyquil/_parser/README.md](pyquil/_parser/README.md).

Note that you only need to install ANTLR if you want to change the grammar, simply running the parser involves no extra
steps beyond installing PyQuil as usual.
