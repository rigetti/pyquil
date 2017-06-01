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
"""
Module for creating and defining parametric programs.
"""

import inspect
from copy import copy
from six.moves import range

from .quilbase import Slot
from .quil import Program


def argument_count(thing):
    """
    Get the number of arguments a callable has.

    :param thing: A callable.
    :return: The number of arguments it takes.
    :rtype: int
    """
    if not callable(thing):
        raise TypeError("should be callable")
    argspec = inspect.getargspec(thing)
    if argspec.varargs is not None:
        raise RuntimeError("callable should have no *args parameter")
    if argspec.keywords is not None:
        raise RuntimeError("callable should have no **kwargs parameter")
    return len(argspec.args)


class ParametricProgram(object):
    """
    .. note:: Experimental

    A class representing Programs with changeable gate parameters.
    """

    def __init__(self, program_constructor):
        self.num_arguments = argument_count(program_constructor)
        self.slots = [Slot() for _ in range(self.num_arguments)]
        self.instantiated_program = program_constructor(*self.slots)
        if not isinstance(self.instantiated_program, Program):
            raise TypeError("program_constructor should produce a Program object")

    def fuse(self, other):
        """
        .. note:: Experimental

        Fuse another program to this one.

        :param other: A Program or ParametricProgram.
        :return: A new ParametricProgram.
        :rtype: ParametricProgram
        """
        r = copy(self)   # shallow copy the object

        if isinstance(other, ParametricProgram):
            r.num_arguments += other.num_arguments
            r.slots.extend(other.slots)
            r.instantiated_program += other.instantiated_program
        elif isinstance(other, Program):
            r.instantiated_program += other
        else:
            raise TypeError("Can only fuse Programs and ParametricPrograms")

        return r

    def __call__(self, *values):
        if len(values) != self.num_arguments:
            raise RuntimeError("Invalid number of arguments provided"
                               " to a ParametricProgram instance")
        for slot, value in zip(self.slots, values):
            slot._value = value

        return self.instantiated_program


def parametric(decorated_function):
    """
    .. note:: Experimental

    A decorator to change a function into a ParametricProgram.

    :param decorated_function: The function taking parameters producing a Program object.
    :return: a callable ParametricProgram
    :rtype: ParametricProgram
    """
    return ParametricProgram(decorated_function)
