# PyQuil API usage examples

This directory contains a few select examples for what you are able to
do with the `pyquil` API.

## Requirements

To use these examples, we recommend that you actually clone the pyquil
git repository or [download it via this link.](https://github.com/rigetti/pyquil/archive/master.zip) Furthermore, you will
have to install pyquil and its requirements but also some additional
requirements specific to the examples:

```sh
cd YOUR/LOCAL/PYQUIL
pip install -r requirements.txt # install pyquil requirements
pip install -e . # install local version of pyquil
pip install -r examples/requirements.txt # install pyquil example requirements
```

where you must replace `YOUR/LOCAL/PYQUIL` with the path to where you
downloaded/cloned pyquil to.

## Running the examples

We provide a number of python scripts that can be invoked as follows:

``` sh
python examples/quantum_die.py
```

We also provide a number of jupyter notebooks. To run those you will
need to launch a jupyter notebook server:

``` sh
jupyter notebook examples/
```

Your web browser will present you with the notebooks -- enter one and
start evaluating the notebook cells.

