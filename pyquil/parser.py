##############################################################################
# Copyright 2016-2018 Rigetti Computing
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
"""
Module for parsing Quil programs from text into PyQuil objects
"""
from pyquil.quil import Program

from pyquil._parser.PyQuilListener import run_parser


def parse_program(quil):
    """
    Parse a raw Quil program and return a PyQuil program.

    :param str quil: a single or multiline Quil program
    :return: PyQuil Program object
    """
    return Program(parse(quil))


def parse(quil):
    """
    Parse a raw Quil program and return a corresponding list of PyQuil objects.

    :param str quil: a single or multiline Quil program
    :return: list of instructions
    """
    return run_parser(quil)
