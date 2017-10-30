# pyQuil

A library for easily generating Quil programs to be executed using the Rigetti Forest platform.
pyQuil is licensed under the [Apache 2.0 license](https://github.com/rigetticomputing/pyQuil/blob/master/LICENSE).

[![Build Status](https://semaphoreci.com/api/v1/rigetti/pyquil/branches/master/badge.svg)](https://semaphoreci.com/rigetti/pyquil)
[![Documentation Status](https://readthedocs.org/projects/pyquil/badge/?version=latest)](http://pyquil.readthedocs.io/en/latest/?badge=latest)

## Documentation

Documentation is hosted at [http://pyquil.readthedocs.io/en/latest/](http://pyquil.readthedocs.io/en/latest/)

## Installation

You can install pyQuil directly from the Python package manager `pip` using:
```
pip install pyquil
```

To instead install pyQuil from source, clone this repository, `cd` into it, and run:
```
pip install -e .
```

This will also install pyQuil's dependencies (requests >= 2.4.2 and NumPy >= 1.10)
if you do not already have them.

## Connecting to the Rigetti Forest

pyQuil can be used to build and manipulate Quil programs without restriction. However, to run
programs (e.g., to get wavefunctions, get multishot experiment data), you will need an API key
for [Rigetti Forest](http://forest.rigetti.com). This will allow you to run your programs on the
Rigetti Quantum Virtual Machine (QVM) or on a real quantum processor (QPU).

Once you have your key, run the following command to automatically set up your config:
```
pyquil-config-setup
```

You can also create the configuration file manually if you'd like and place it at `~/.pyquil_config`.
The configuration file is in INI format and should contain all the information required to connect to Forest:

```
[Rigetti Forest]
key: <Rigetti Forest API key>
user_id: <Rigetti User ID>
```

You can change the location of this file by setting the `PYQUIL_CONFIG` environment variable.

If you encounter errors or warnings trying to connect to Forest then see the full
[Getting Started Guide](https://go.rigetti.com/getting-started)

## Examples using the Rigetti QVM

Here is how to construct a Bell state program and how to compute the amplitudes of its wavefunction:

```python
>>> import pyquil.quil as pq
>>> import pyquil.api as api
>>> from pyquil.gates import *
>>> qvm = api.SyncConnection()
>>> p = pq.Program(H(0), CNOT(0,1))
<pyquil.pyquil.Program object at 0x101ebfb50>
>>> qvm.wavefunction(p)[0]
[(0.7071067811865475+0j), 0j, 0j, (0.7071067811865475+0j)]
```

How to do a simulated multishot experiment measuring qubits 0 and 1 of a Bell state. (Of course,
each measurement pair will be `00` or `11`.)

```python
>>> import pyquil.quil as pq
>>> import pyquil.api as api
>>> from pyquil.gates import *
>>> qvm = api.SyncConnection()
>>> p = pq.Program()
>>> p.inst(H(0),
...        CNOT(0, 1),
...        MEASURE(0, 0),
...        MEASURE(1, 1))
<pyquil.pyquil.Program object at 0x101ebfc50>
>>> print p
H 0
CNOT 0 1
MEASURE 0 [0]
MEASURE 1 [1]

>>> qvm.run(p, [0, 1], 10)
[[0, 0], [1, 1], [1, 1], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
```

## Building the Docs

We use sphinx to build the documentation. To do this, navigate into pyQuil's top-level directory and run:

```
sphinx-build -b html docs/source docs/_build
```
To view the docs navigate to the newly-created `docs/_build` directory and open
the `index.html` file in a browser. Note that we use the Read the Docs theme for
our documentation, so this may need to be installed using `pip install sphinx_rtd_theme`.

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

## How to cite pyQuil and Forest

If you use pyQuil, Grove, or other parts of the Rigetti Forest stack in your research, please cite it as follows:

BibTeX:
```
@misc{1608.03355,
  title={A Practical Quantum Instruction Set Architecture},
  author={Smith, Robert S and Curtis, Michael J and Zeng, William J},
  journal={arXiv preprint arXiv:1608.03355},
  year={2016}
}
```

Text:
```
R. Smith, M. J. Curtis and W. J. Zeng, "A Practical Quantum Instruction Set Architecture," (2016), 
  arXiv:1608.03355 [quant-ph], https://arxiv.org/abs/1608.03355
```
