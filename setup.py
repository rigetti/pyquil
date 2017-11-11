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

from setuptools import setup
from pyquil import __version__

setup(
    name="pyquil",
    version=__version__,
    author="Rigetti Computing",
    author_email="softapps@rigetti.com",
    description="A Python library to generate Quantum Instruction Language (Quil) Programs.",
    url="https://github.com/rigetticomputing/pyquil.git",
    packages=['pyquil', 'pyquil._parser', 'pyquil._parser.gen2', 'pyquil._parser.gen3', 'pyquil.setup'],
    license="LICENSE",
    install_requires=[
        'requests >= 2.4.2',
        'numpy >= 1.10',
        'matplotlib >= 1.5',
        'urllib3 >= 1.21.1',
        "antlr4-python2-runtime>=4.7; python_version < '3'",
        "antlr4-python3-runtime>=4.7; python_version >= '3'",
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
