## Development and Testing

We use pytest (version > 3.0) and mock for testing. Tests can be run from the top-level directory using:
```
pytest
```

To run the complete suite of tests in their own environment, you can use `tox`. This is done with:
```
pip install tox
tox
```

When making considerable changes to `operator_estimation.py`, we recommend that you set
the `pytest` option `--use-seed` to `False` to make sure you have not broken anything:
```shell
pytest --use-seed=False pyquil
```

## Building the Docs

We use sphinx to build the documentation. Before building the docs, you must have
`pandoc` installed. Then, navigate into the `docs` directory and run:

```
make html
```

To view the docs navigate to the newly-created `docs/build` directory and open
the `index.html` file in a browser. Note that we use the Read the Docs theme for
our documentation, so this may need to be installed using `pip install sphinx_rtd_theme`.

## Working with the parser

Working with the ANTLR parser involves some extra steps, 
see [pyquil/_parser/README.md](pyquil/_parser/README.md).

Note that you only need to install ANTLR if you want to change the grammar, simply running the parser involves no extra
steps beyond installing PyQuil as usual.
