#!/usr/bin/python
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

setup(
    name = "pyquil",
    version = "1.0.0",
    author = "Rigetti Computing",
    author_email = "softapps@rigetti.com",
    description = "A Python library to generate Quantum Instruction Language (Quil) Programs.",
    url="https://github.com/rigetticomputing/pyquil.git",
    download_url="https://github.com/rigetticomputing/pyquil/tarball/1.0.0",
    packages = ["pyquil"],
    license = "LICENSE",
    install_requires = [
        'requests >= 2.4.2',
        'numpy >= 1.10',
        'matplotlib >= 1.5',
    ],
    setup_requires = ['pytest-runner'],
    tests_require = [
        'pytest >= 3.0.0',
        'mock',
    ],
    keywords='quantum quil programming hybrid'
)
