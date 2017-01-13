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
Module for creating and defining Quil programs.
"""

import copy

from pyquil.quilbase import (InstructionGroup,
                             Addr,
                             While,
                             If,
                             DefGate,
                             merge_resource_managers)

from pyquil.gates import MEASURE


class Program(InstructionGroup):
    def __init__(self, *instructions):
        super(Program, self).__init__()
        self.inst(*instructions)
        self.defined_gates = []

    def synthesize(self, resource_manager=None):
        self.resource_manager.reset()
        return super(Program, self).synthesize(resource_manager)

    def __add__(self, other):
        """
        Concatenate two programs together, returning a new one.

        :param other: Another program or instruction to concatenate to this one.
        :return: A newly concatenated program.
        """
        if isinstance(other, Program):
            self.resource_manager = merge_resource_managers(self.resource_manager,
                                                            other.resource_manager)
            p = Program()
            p.defined_gates = self.defined_gates + other.defined_gates
            p.actions = self.actions + other.actions
            return p
        else:
            return super(Program, self).__add__(other)

    def defgate(self, name, matrix):
        """
        Define a new static gate.

        :param name: (int) The name of the gate (str).
        :param matrix: List of lists or Numpy 2d array.
        :return: The Program instance.
        """
        self.defined_gates.append(DefGate(name, matrix))
        return self

    def measure(self, qubit_index, classical_reg):
        """
        Measures a qubit at qubit_index and puts the result in classical_reg

        :param qubit_index: (int) The address of the qubit to measure.
        :param classical_reg: (int) The address of the classical bit to store the result.

        :returns: The Quil Program with the appropriate measure instruction appended, e.g.
                  MEASURE 0 [1]
        """
        return self.inst(MEASURE(qubit_index, classical_reg))

    def while_do(self, classical_reg, q_program):
        """
        While a classical register at index classical_reg is 1, loop q_program

        :param classical_reg: (int) The classical register to check
        :param q_program: (Program) The Quil program to loop.

        :return: The Quil Program with the loop instructions added.
        """
        w_loop = While(Addr(classical_reg))
        w_loop.Body.inst(q_program)
        return self.inst(w_loop)

    def if_then(self, classical_reg, if_program, else_program=None):
        """
        If the classical register at index classical reg is 1, run if_program, else run
        else_program.

        :param classical_reg: (int) The classical register to check as the condition
        :param if_program: (Program) A Quil program to execute if classical_reg is 1
        :param else_program: (Program) A Quil program to execute if classical_reg is 0. This
            argument is optional and defaults to an empty Program.

        :returns: The Quil Program with the branching instructions added.
        """

        else_program = else_program if else_program is not None else Program()

        branch = If(Addr(classical_reg))
        branch.Then.inst(if_program)
        branch.Else.inst(else_program)
        return self.inst(branch)

    def out(self):
        """
        Converts the Quil program to a readable string.
        :return: (string)
        """
        s = ""
        for dg in self.defined_gates:
            s += dg.out()
            s += "\n"
        s += super(Program, self).out()
        return s
