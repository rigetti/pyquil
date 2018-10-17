# pyQuil

A library for easily generating Quil programs to be executed using the Rigetti Forest platform.
pyQuil is licensed under the [Apache 2.0 license](https://github.com/rigetticomputing/pyQuil/blob/master/LICENSE).

[![Build Status](https://semaphoreci.com/api/v1/rigetti/pyquil/branches/master/badge.svg)](https://semaphoreci.com/rigetti/pyquil)
[![Documentation Status](https://readthedocs.org/projects/pyquil/badge/?version=latest)](http://pyquil.readthedocs.io/en/latest/?badge=latest)

## Documentation

Documentation is hosted at [http://pyquil.readthedocs.io/en/latest/](http://pyquil.readthedocs.io/en/latest/)

## Installation

Download the Forest SDK at [http://rigetti.com/forest](http://rigetti.com/forest), and sign up for Quantum Cloud Services at [http://rigetti.com/qcs](http://rigetti.com/qcs). 
You'll need to download the Forest SDK in order to be able to compile and run your pyQuil programs!

Once you've installed the SDK, you can install pyQuil using package manager `pip`:

```
pip install --pre pyquil
```

or install pyQuil from source by cloning this repository, `cd` into it, and run:
```
pip install -e .
``` 

## Community

Join the public Forest Slack channel at [http://slack.rigetti.com](http://slack.rigetti.com).

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
