# pyQuil

A library for easily generating Quil programs to be executed using the Rigetti Forest platform.
pyQuil is licensed under the [Apache 2.0 license](https://github.com/rigetticomputing/pyQuil/blob/master/LICENSE).

[![Build Status]
(https://semaphoreci.com/api/v1/projects/044fb8f4-1f90-4e28-8459-95289d682b70/1124972/badge.svg)]
(https://semaphoreci.com/rigetti/pyquil) 
[![Documentation Status](https://readthedocs.org/projects/pyquil/badge/?version=latest)]
(http://pyquil.readthedocs.io/en/latest/?badge=latest)

## Documentation

Documentation is hosted at [http://pyquil.readthedocs.io/en/latest/]
(http://pyquil.readthedocs.io/en/latest/)

## Installation

Clone the git repository, `cd` into it, and run

```
pip install -e .
```
This will install pyQuil's dependencies (requests and numpy) if you do not already have them.

In order to connect to the QVM you need to set up configuration in the file `.pyquil_config` which
by default is looked for in your home directory. (You can change this location by setting the
environment variable `PYQUIL_CONFIG` to the path of the file.) Loading the `pyquil.forest` module
will print a warning if this is not found. The configuration file is in INI format and should
contain relevant information to connect.

```
[Rigetti Forest]
url: <URL to Rigetti Forest or QVM endpoint>
key: <Rigetti Forest API key>
```

If `url` is not specified, it will default to `https://api.rigetti.com/qvm`. In addition to the
above, the fields `https_cert` and `https_key` are supported for direct HTTPS connections to QVMs.

```
https_cert: <path to signed HTTPS certificate and key>
https_key: <path to separate key file, if different from the above>
```

## Examples using QVM

Here is how to construct a Bell state program and how to compute the amplitudes of its wavefunction:

```
>>> import pyquil.quil as pq
>>> import pyquil.forest as forest
>>> from pyquil.gates import *
>>> cxn = forest.Connection()
>>> p = pq.Program(H(0), CNOT(0,1))
<pyquil.pyquil.Program object at 0x101ebfb50>
>>> cxn.wavefunction(p)[0]
[(0.7071067811865475+0j), 0j, 0j, (0.7071067811865475+0j)]
```

How to do a simulated multishot experiment measuring qubits 0 and 1 of a Bell state. (Of course,
each measurement pair will be `00` or `11`.)

```
>>> import pyquil.quil as pq
>>> import pyquil.forest as forest
>>> cxn = forest.Connection()
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

>>> cxn.run(p, [0, 1], 10)
[[0, 0], [1, 1], [1, 1], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
```

## Rigetti Forest

pyQuil can be used to build and manipulate Quil programs without restriction. However, to run
programs (e.g., to get wavefunctions, get multishot experiment data), you will need an API key
for [Rigetti Forest](https://forest.rigetti.com).

## Building the docs

We use sphinx to build the documentation. This can be done with:

```
sphinx-build -b html docs/source docs/_build
```
To view the docs navigate to the `docs/_build` directory in the pyQuil root directory and open the `index.html` file a browser. Note that we use the readthedocs theme for our documentation, so this theme may need to be installed using `pip install sphinx_rtd_theme`.

## Development and Testing

We use pytest for testing. Tests can be run from the top level directory with

```
py.test --cov=pyquil
```
