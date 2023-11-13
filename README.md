PyQuil: Quantum programming in Python
=====================================

[![binder](https://mybinder.org/badge_logo.svg)][binder]
[![docs][docs-badge]][docs-repo]
[![docker][docker-badge]][docker-repo]
[![pepy][pepy-badge]][pepy-repo]
[![pypi][pypi-badge]][pypi-repo]
[![slack][slack-badge]][slack-invite]

PyQuil is a Python library for quantum programming using [Quil](https://arxiv.org/abs/1608.03355),
the quantum instruction language developed at [Rigetti Computing](https://www.rigetti.com/).
PyQuil serves three main functions:

- Easily generating Quil programs from quantum gates and classical operations
- Compiling and simulating Quil programs using the [Quil Compiler](https://github.com/rigetti/quilc)
  (quilc) and the [Quantum Virtual Machine](https://github.com/rigetti/qvm) (QVM)
- Executing Quil programs on real quantum processors (QPUs) using
  [Quantum Cloud Services][qcs-paper] (QCS)

PyQuil has a ton of other features, which you can learn more about in the
[docs](http://pyquil.readthedocs.io/en/latest/). However, you can also keep reading
below to get started with running your first quantum program!

Quickstart with interactive tutorial notebooks
----------------------------------------------

Without installing anything, you can quickly get started with quantum programming by exploring
our interactive [Jupyter][jupyter] notebook tutorials and examples. To run them in a preconfigured
execution environment on [Binder][mybinder], click the "launch binder" badge at the top of the
README or the link [here][binder]! To learn more about the tutorials and how you can add your own,
visit the [rigetti/forest-tutorials][forest-tutorials] repository. If you'd rather set everything
up locally, or are interested in contributing to pyQuil, continue onto the next section for
instructions on installing pyQuil and the Forest SDK.

Installing pyQuil and the Forest SDK
------------------------------------

[![pypi][pypi-badge]][pypi-repo]
[![conda-forge][conda-forge-badge]][conda-forge-badge]
[![conda-rigetti][conda-rigetti-badge]][conda-rigetti-repo]

PyQuil can be installed using `conda`, `pip`, or from source. To install it from PyPI (via `pip`),
do the following:

```bash
pip install pyquil
```

To instead install pyQuil from source, do the following from within the repository after cloning it:

```bash
pip install -e .
```

If you choose to use `pip`, we highly recommend installing pyQuil within a virtual environment.

PyQuil, along with quilc, the QVM, and other libraries, make up what is called the Forest
SDK. To make full use of pyQuil, you will need to additionally have installed
[quilc](https://github.com/quil-lang/quilc) and the [QVM](https://github.com/quil-lang/qvm).
For more information, check out the docs!

Running your first quantum program
----------------------------------

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

qvm.run(p).get_register_map()['ro'].tolist()
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

Using the Forest SDK, you can simulate the operation of a real quantum processor (QPU). If you
would like to run on the real QPUs in our lab in Berkeley, you can sign up for an account
on [Quantum Cloud Services][qcs-request-access] (QCS)!

Joining the Forest community
----------------------------

If you'd like to get involved with pyQuil and Forest, joining the
[Rigetti Forest Slack Workspace][slack-invite] is a great place to start! You can do so by
clicking the invite link in the previous sentence, or in the badge at the top of this README.
The Slack Workspace is a great place to ask general questions, join high-level design discussions,
and hear about updates to pyQuil and the Forest SDK.

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

Citing pyQuil, Forest, and Quantum Cloud Services
-------------------------------------------------

[![zenodo][zenodo-badge]][zenodo-doi]

If you use pyQuil, Grove, or other parts of the Forest SDK in your research, please cite
the [Quil specification][quil-paper] using the following BibTeX snippet:

```bibtex
@misc{smith2016practical,
    title={A Practical Quantum Instruction Set Architecture},
    author={Robert S. Smith and Michael J. Curtis and William J. Zeng},
    year={2016},
    eprint={1608.03355},
    archivePrefix={arXiv},
    primaryClass={quant-ph}
}
```

Additionally, if your research involves taking data on Rigetti quantum processors (QPUs) via
the Quantum Cloud Services (QCS) platform, please reference the [QCS paper][qcs-paper] using the
following BibTeX snippet:

```bibtex
@article{Karalekas_2020,
    title = {A quantum-classical cloud platform optimized for variational hybrid algorithms},
    author = {Peter J Karalekas and Nikolas A Tezak and Eric C Peterson
              and Colm A Ryan and Marcus P da Silva and Robert S Smith},
    year = 2020,
    month = {apr},
    publisher = {{IOP} Publishing},
    journal = {Quantum Science and Technology},
    volume = {5},
    number = {2},
    pages = {024003},
    doi = {10.1088/2058-9565/ab7559},
    url = {https://doi.org/10.1088%2F2058-9565%2Fab7559},
}
```

The preprint of the QCS paper is available on [arXiv][qcs-arxiv], and the supplementary
interactive notebooks and datasets for the paper can be found in the [rigetti/qcs-paper][qcs-repo]
repository.

License
-------

PyQuil is licensed under the
[Apache License 2.0](https://github.com/rigetti/pyQuil/blob/master/LICENSE).

[binder]: https://mybinder.org/v2/gh/rigetti/forest-tutorials/master?urlpath=lab/tree/Welcome.ipynb
[conda-forge-badge]: https://img.shields.io/conda/vn/conda-forge/pyquil.svg
[conda-forge-repo]: https://anaconda.org/conda-forge/pyquil
[conda-rigetti-badge]: https://img.shields.io/conda/vn/rigetti/pyquil?label=conda-rigetti
[conda-rigetti-repo]: https://anaconda.org/rigetti/pyquil
[docker-badge]: https://img.shields.io/docker/pulls/rigetti/forest
[docker-repo]: https://hub.docker.com/r/rigetti/forest
[docs-badge]: https://readthedocs.org/projects/pyquil/badge/?version=latest
[docs-repo]: http://pyquil.readthedocs.io/en/latest/?badge=latest
[forest-tutorials]: https://github.com/rigetti/forest-tutorials
[jupyter]: https://jupyter.org/
[mybinder]: https://mybinder.org
[pepy-badge]: https://pepy.tech/badge/pyquil
[pepy-repo]: https://pepy.tech/project/pyquil
[pypi-badge]: https://img.shields.io/pypi/v/pyquil.svg
[pypi-repo]: https://pypi.org/project/pyquil/
[qcs-request-access]: https://qcs.rigetti.com/request-access
[slack-badge]: https://img.shields.io/badge/slack-rigetti--forest-812f82.svg?
[zenodo-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.3553165.svg
[zenodo-doi]: https://doi.org/10.5281/zenodo.3553165

[qcs-arxiv]: https://arxiv.org/abs/2001.04449
[qcs-paper]: https://dx.doi.org/10.1088/2058-9565/ab7559
[qcs-repo]: https://github.com/rigetti/qcs-paper
[quil-paper]: https://arxiv.org/abs/1608.03355

[bug]: https://github.com/rigetti/pyquil/issues/new?assignees=&labels=bug+%3Abug%3A&template=BUG_REPORT.md&title=
[feature]: https://github.com/rigetti/pyquil/issues/new?assignees=&labels=enhancement+%3Asparkles%3A&template=FEATURE_REQUEST.md&title=
[first]: https://github.com/rigetti/pyquil/labels/good%20first%20issue%20%3Ababy%3A
[help]: https://github.com/rigetti/pyquil/labels/help%20wanted%20%3Awave%3A
[fork]: https://github.com/rigetti/pyquil/fork
[pr]: https://github.com/rigetti/pyquil/compare
[slack-invite]: https://join.slack.com/t/rigetti-forest/shared_invite/enQtNTUyNTE1ODg3MzE2LWQwNzBlMjZlMmNlN2M5MzQyZDlmOGViODQ5ODI0NWMwNmYzODY4YTc2ZjdjOTNmNzhiYTk2YjVhNTE2NTRkODY
