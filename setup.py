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

from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with open(os.path.join(HERE, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="pyquil",
    version=find_version('pyquil', '__init__.py'),
    author="Rigetti Computing",
    author_email="softapps@rigetti.com",
    description="A Python library to generate Quantum Instruction Language (Quil) Programs.",
    url="https://github.com/rigetticomputing/pyquil.git",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    license="LICENSE",
    install_requires=[
        'contextvars == 2.2',
        'numpy >= 1.10',
        'matplotlib >= 1.5',
        'requests >= 2.4.2',
        'typing >= 3.6',
        'urllib3 >= 1.21.1',
        "antlr4-python3-runtime>=4.7",
    ],
    setup_requires=['pytest-runner'],
    tests_require=[
        'pytest >= 3.0.0',
        'mock',
    ],
    test_suite='pyquil.tests',
    entry_points={
        'console_scripts': ['pyquil-config-setup=pyquil.setup.pyquil_config_setup:main']
    },
    keywords='quantum quil programming hybrid'
)
