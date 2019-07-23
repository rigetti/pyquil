PyQuil: Quantum programming in Python
=====================================

[![pipeline status](https://gitlab.com/rigetti/forest/pyquil/badges/master/pipeline.svg)](https://gitlab.com/rigetti/forest/pyquil/commits/master)
[![build status](https://semaphoreci.com/api/v1/rigetti/pyquil/branches/master/shields_badge.svg)](https://semaphoreci.com/rigetti/pyquil)
[![docs status](https://readthedocs.org/projects/pyquil/badge/?version=latest)](http://pyquil.readthedocs.io/en/latest/?badge=latest)
[![pypi downloads](https://img.shields.io/pypi/dm/pyquil.svg)](https://pypi.org/project/pyquil/)
[![pypi version](https://img.shields.io/pypi/v/pyquil.svg)](https://pypi.org/project/pyquil/)
[![conda-forge version](https://img.shields.io/conda/vn/conda-forge/pyquil.svg)](https://anaconda.org/conda-forge/pyquil)
[![slack workspace](https://img.shields.io/badge/slack-rigetti--forest-812f82.svg?)](https://join.slack.com/t/rigetti-forest/shared_invite/enQtNTUyNTE1ODg3MzE2LWExZWU5OTE4YTJhMmE2NGNjMThjOTM1MjlkYTA5ZmUxNTJlOTVmMWE0YjA3Y2M2YmQzNTZhNTBlMTYyODRjMzA)

PyQuil is a Python library for quantum programming using [Quil](https://arxiv.org/abs/1608.03355),
the quantum instruction language developed at [Rigetti Computing](https://www.rigetti.com/).
PyQuil serves three main functions:

- Easily generating Quil programs from quantum gates and classical operations
- Compiling and simulating Quil programs using the [Quil Compiler](https://github.com/rigetti/quilc)
  (quilc) and the [Quantum Virtual Machine](https://github.com/rigetti/qvm) (QVM)
- Executing Quil programs on real quantum processors (QPUs) using
  [Quantum Cloud Services](https://www.rigetti.com/qcs) (QCS)

PyQuil has a ton of other features, which you can learn more about in the
[docs](ttp://pyquil.readthedocs.io/en/latest/). However, you can also keep reading
below to get started with running your first quantum program!

Installation
------------

**Please Note: PyQuil, along with quilc, the QVM, and other libraries, make up what
is called the Forest SDK. To make full use of pyQuil's functionality, you will need
to additionally have installed [quilc](https://github.com/rigetti/quilc) and the
[QVM](https://github.com/rigetti/qvm). This can be done by following their respective
READMEs, or by downloading them as binaries from [here](https://rigetti.com/forest).**

PyQuil can be installed using `conda`, `pip`, or directly from source.

To install pyQuil as a `conda` package from the conda-forge channel (recommended), do the following:

```bash
conda install -c conda-forge pyquil
```

To instead install pyQuil as a PyPI package, do the following:

```bash
pip install pyquil
```

Finally, if you would prefer to install pyQuil directly from source, do the following
from within the repository after cloning it:

```bash
pip install -e .
```

If you choose to use `pip`, we highly recommend installing pyQuil within a virtual environment.

Getting Started
---------------

In just a few lines, we can use pyQuil with the Forest SDK to simulate a Bell state!

```python
from pyquil import get_qc, Program
from pyquil.gates import CNOT, H, MEASURE
 
qvm = get_qc('2q-qvm')
 
p = Program()
p += H(0)
p += CNOT(0, 1)
ro = p.declare('ro', 'BIT', 2)
p += MEASURE(0, ro[0])
p += MEASURE(1, ro[1])
p.wrap_in_numshots_loop(10)
 
qvm.run(p).tolist()
```

The output of the above program should look something like the following,
the statistics of which are consistent with a two-qubit entangled state.

```
[[0, 0],
 [1, 1],
 [1, 1],
 [1, 1],
 [1, 1],
 [0, 0],
 [0, 0],
 [1, 1],
 [0, 0],
 [0, 0]]
```

Running on the QPU
------------------

Using the Forest SDK, you can simulate the operation of a real quantum processor. If you
would like to run on the real QPUs in our lab in Berkeley, you can sign up for an account
on [Quantum Cloud Services](https://www.rigetti.com/qcs)!

Joining the Forest Community
----------------------------

Join the public Forest Slack channel at [http://slack.rigetti.com](https://join.slack.com/t/rigetti-forest/shared_invite/enQtNTUyNTE1ODg3MzE2LWExZWU5OTE4YTJhMmE2NGNjMThjOTM1MjlkYTA5ZmUxNTJlOTVmMWE0YjA3Y2M2YmQzNTZhNTBlMTYyODRjMzA).

The following projects have been contributed by community members:

- [Syntax Highlighting for Quil](https://github.com/JavaFXpert/quil-syntax-highlighter)
  contributed by [James Weaver](https://github.com/JavaFXpert)
- [Web Based Circuit Simulator](https://github.com/rasa97/quil-sim/tree/master)
  contributed by [Ravisankar A V](https://github.com/rasa97)
- [Quil in Javascript](https://github.com/mapmeld/jsquil)
  contributed by [Nick Doiron](https://github.com/mapmeld)
- [Quil in Java](https://github.com/QCHackers/jquil)
  contributed by [Victory Omole](https://github.com/vtomole)

Contributing to pyQuil
----------------------

To make changes to PyQuil itself see [DEVELOPMENT.md](DEVELOPMENT.md) for instructions on
development and testing.

Citing pyQuil and Forest
------------------------

If you use pyQuil, Grove, or other parts of the Rigetti Forest stack in your research,
please cite it as follows:

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

License
-------

PyQuil is licensed under the [Apache License 2.0](https://github.com/rigetti/pyQuil/blob/master/LICENSE).
