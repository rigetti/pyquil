# pyQuil

A library for easily generating Quil programs to be executed using the Rigetti Forest platform.
pyQuil is licensed under the [Apache 2.0 license](https://github.com/rigetti/pyQuil/blob/master/LICENSE).

[![Build Status](https://semaphoreci.com/api/v1/rigetti/pyquil/branches/master/badge.svg)](https://semaphoreci.com/rigetti/pyquil)
[![Documentation Status](https://readthedocs.org/projects/pyquil/badge/?version=latest)](http://pyquil.readthedocs.io/en/latest/?badge=latest)

**Please note: To make full use of our SDK, in addition to installing pyQuil, you'll need to download [quilc and the QVM](rigetti.com/forest), which 
will allow you to compile and simulate programs on your local machine. 
If you'd like to run programs on our quantum computers, you'll need to sign up for [Quantum Cloud Services](rigetti.com/qcs).**

## Documentation

Documentation is hosted at [http://pyquil.readthedocs.io/en/latest/](http://pyquil.readthedocs.io/en/latest/)

## Installation

You can install pyQuil as a conda package (recommended):

```bash
conda install -c -conda-forge pyquil
```

or using `pip`:

```
pip install pyquil
```

To instead install pyQuil from source, clone this repository, `cd` into it, and run:
```
pip install -e .
```

See the [Getting Started Guide](https://go.rigetti.com/getting-started) to start writing quantum programs!


## Community

Join the public Forest Slack channel at [http://slack.rigetti.com](https://join.slack.com/t/rigetti-forest/shared_invite/enQtNDI1MDQzOTk3OTI3LWZlNzVjZWNlMmYxMjAzYTNlZjRhNjlmNWE0MDc1MjNhYTRkODA5Mjg3NTY4YWRmYzJmN2Q1M2M0Nzg3YzhhYzI).

The following projects have been contributed by community members:

- [Syntax Highlighting for Quil](https://github.com/JavaFXpert/quil-syntax-highlighter)
  contributed by [James Weaver](https://github.com/JavaFXpert)
- [Web Based Circuit Simulator](https://github.com/rasa97/quil-sim/tree/master)
  contributed by [Ravisankar A V](https://github.com/rasa97)
- [Quil in Javascript](https://github.com/mapmeld/jsquil)
  contributed by [Nick Doiron](https://github.com/mapmeld)
- [Quil in Java](https://github.com/QCHackers/jquil)
  contributed by [Victory Omole](https://github.com/vtomole)

## Developing PyQuil

To make changes to PyQuil itself see [DEVELOPMENT.md](DEVELOPMENT.md) for instructions on development and testing.

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
