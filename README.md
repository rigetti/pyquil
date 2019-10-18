PyQuil: Quantum programming in Python
=====================================

[![pipeline status](https://gitlab.com/rigetti/forest/pyquil/badges/master/pipeline.svg)](https://gitlab.com/rigetti/forest/pyquil/commits/master)
[![build status](https://semaphoreci.com/api/v1/rigetti/pyquil/branches/master/shields_badge.svg)](https://semaphoreci.com/rigetti/pyquil)
[![docs status](https://readthedocs.org/projects/pyquil/badge/?version=latest)](http://pyquil.readthedocs.io/en/latest/?badge=latest)
[![pypi downloads](https://img.shields.io/pypi/dm/pyquil.svg)](https://pypi.org/project/pyquil/)
[![pypi version](https://img.shields.io/pypi/v/pyquil.svg)](https://pypi.org/project/pyquil/)
[![conda-forge version](https://img.shields.io/conda/vn/conda-forge/pyquil.svg)](https://anaconda.org/conda-forge/pyquil)
[![slack workspace](https://img.shields.io/badge/slack-rigetti--forest-812f82.svg?)][slack_invite]

PyQuil is a Python library for quantum programming using [Quil](https://arxiv.org/abs/1608.03355),
the quantum instruction language developed at [Rigetti Computing](https://www.rigetti.com/).
PyQuil serves three main functions:

- Easily generating Quil programs from quantum gates and classical operations
- Compiling and simulating Quil programs using the [Quil Compiler](https://github.com/rigetti/quilc)
  (quilc) and the [Quantum Virtual Machine](https://github.com/rigetti/qvm) (QVM)
- Executing Quil programs on real quantum processors (QPUs) using
  [Quantum Cloud Services](https://www.rigetti.com/qcs) (QCS)

PyQuil has a ton of other features, which you can learn more about in the
[docs](http://pyquil.readthedocs.io/en/latest/). However, you can also keep reading
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

Joining the Forest Community
----------------------------

If you'd like to get involved with pyQuil and Forest, joining the [Rigetti Forest Slack
Workspace][slack_invite]
is a great place to start! You can do so by clicking the invite link in the previous sentence,
or in the badge at the top of this README. The Slack Workspace is a great place to ask general
questions, join high-level design discussions, and hear about updates to pyQuil and the Forest SDK.

To go a step further and start contributing to the development of pyQuil, good first steps are
[reporting a bug][bug], [requesting a feature][feature], or picking up one of the issues with the
[good first issue][first] or [help wanted][help] labels. Once you find an issue to work
on, make sure to [fork this repository][fork] and then [open a pull request][pr] once your changes
are ready. For more information on all the ways you can contribute to pyQuil (along with
some helpful tips for developers and maintainers) check out our
[Contributing Guide](CONTRIBUTING.md)!

To see what people have contributed in the past, check out the [Changelog](CHANGELOG.md) for
a detailed list of all announcements, improvements, changes, and bugfixes. The
[Releases](https://github.com/rigetti/pyquil/releases) page for pyQuil contains similar
information, but with links to the pull request for each change and its corresponding author.
Thanks for contributing to pyQuil! 🙂

[bug]: https://github.com/rigetti/pyquil/issues/new?assignees=&labels=bug+%3Abug%3A&template=BUG_REPORT.md&title=
[feature]: https://github.com/rigetti/pyquil/issues/new?assignees=&labels=enhancement+%3Asparkles%3A&template=FEATURE_REQUEST.md&title=
[first]: https://github.com/rigetti/pyquil/labels/good%20first%20issue%20%3Ababy%3A
[help]: https://github.com/rigetti/pyquil/labels/help%20wanted%20%3Awave%3A
[fork]: https://github.com/rigetti/pyquil/fork
[pr]: https://github.com/rigetti/pyquil/compare
[slack_invite]: https://join.slack.com/t/rigetti-forest/shared_invite/enQtNTUyNTE1ODg3MzE2LWQwNzBlMjZlMmNlN2M5MzQyZDlmOGViODQ5ODI0NWMwNmYzODY4YTc2ZjdjOTNmNzhiYTk2YjVhNTE2NTRkODY

Running on the QPU
------------------

Using the Forest SDK, you can simulate the operation of a real quantum processor. If you
would like to run on the real QPUs in our lab in Berkeley, you can sign up for an account
on [Quantum Cloud Services](https://www.rigetti.com/qcs)!

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

PyQuil is licensed under the
[Apache License 2.0](https://github.com/rigetti/pyQuil/blob/master/LICENSE).
