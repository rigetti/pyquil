#!/usr/bin/env python
##############################################################################
# Copyright 2016-2017 Rigetti Computing
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

import os
import re
import sys

from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))

if sys.version_info < (3, 6):
    raise ImportError('\n'.join([
        'PyQuil 2.0+ requires Python 3'
        '',
        'To install the most recent version with support for Python 2, make sure you',
        'have pip >= 9.0 as well as setuptools >= 24.2:',
        '',
        ' $ pip install pip setuptools --upgrade',
        '',
        'Then you can either',
        '',
        '- install an older version of PyQuil:',
        '',
        " $ pip install 'pyquil<2.0'",
        '',
        '- Upgrade your system to use Python 3.', ]))


def read(*parts):
    with open(os.path.join(HERE, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pyquil",
    version=find_version('pyquil', '__init__.py'),
    author="Rigetti Computing",
    author_email="softapps@rigetti.com",
    description="A Python library to generate Quantum Instruction Language (Quil) Programs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rigetti/pyquil.git",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        # The minimum spec for a working pyquil install.
        # note to developers: this should be a subset of requirements.txt
        'numpy',
        'antlr4-python3-runtime',
        'requests',
        'six',
        'networkx',
        'rpcq>=2.2.1',

        # dependency of contextvars, which we vendor
        'immutables==0.6',
    ],
    keywords='quantum quil programming hybrid',
    python_requires=">=3.6",
)
