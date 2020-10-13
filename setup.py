#!/usr/bin/env python
##############################################################################
# Copyright 2016-2019 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

import sys

from setuptools import setup, find_packages

VERSION_ERROR = """
PyQuil 2.0+ requires Python 3

To install the most recent version with support for Python 2, make sure you
have pip >= 9.0 as well as setuptools >= 24.2:

    pip install pip setuptools --upgrade

Then you can either:

    - Install an older version of PyQuil via pip install pyquil<2.0
    - Upgrade your system to use Python 3.6
"""

if sys.version_info < (3, 6):
    raise ImportError(VERSION_ERROR)

with open("VERSION.txt", "r") as f:
    __version__ = f.read().strip()

# Specify UTF-8 to guard against systems that default to an ASCII locale.
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# save the source code in version.py
with open("pyquil/version.py", "r") as f:
    version_file_source = f.read()

# overwrite version.py in the source distribution
with open("pyquil/version.py", "w") as f:
    f.write(f"__version__ = '{__version__}'\n")

setup(
    name="pyquil",
    version=__version__,
    author="Rigetti Computing",
    author_email="softapps@rigetti.com",
    description="A Python library to generate Quantum Instruction Language (Quil) Programs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rigetti/pyquil.git",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_data={"pyquil": ["py.typed"]},
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # The minimum spec for a working pyquil install.
        # note to developers: this should be a subset of requirements.txt
        "numpy",
        "antlr4-python3-runtime>=4.7.2,<4.8",
        "requests",
        "networkx>=2.0.0",
        "rpcq>=3.0.0",
        # dependency of contextvars, which we vendor
        "immutables==0.6",
    ],
    extras_require={"latex": ["ipython"],
                    "tutorials": ["forest-benchmarking", "jupyter", "matplotlib", "seaborn",
                                  "pandas", "scipy", "tqdm"]},
    keywords="quantum quil programming hybrid",
    python_requires=">=3.6",
)

# restore version.py to its previous state
with open("pyquil/version.py", "w") as f:
    f.write(version_file_source)
