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

from .quilbase import (InstructionGroup,
                       Addr,
                       While,
                       If,
                       DefGate,
                       Gate,
                       Measurement,
                       merge_resource_managers)

from .gates import MEASURE


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

        :param Program other: Another program or instruction to concatenate to this one.
        :return: A newly concatenated program.
        :rtype: Program
        """
        if isinstance(other, Program):
            p = Program()
            p.defined_gates = self.defined_gates + other.defined_gates
            p.actions = self.actions + other.actions
            p.resource_manager = merge_resource_managers(self.resource_manager,
                                                         other.resource_manager)
            return p
        else:
            return super(Program, self).__add__(other)

    def __getitem__(self, index):
        """
        Allows indexing into the program to get an action.
        :param index: The action at the specified index.
        :return:
        """
        return self.actions[index]

    def __iter__(self):
        """
        Allow built in iteration through a program's actions, e.g. [a for a in Program(X(0))]
        :return:
        """
        return self.actions.__iter__()

    def get_qubits(self):
        """
        :return: a set of all the qubit indices allocated in this program, synthesizing freely
        allocated qubits if neccessary.
        :rtype: set
        """
        allocated_qubits = set()
        self.synthesize()
        for ii, action in self:
            if isinstance(action, Gate):
                qubit_indices = {qq.index() for qq in action.arguments}
            elif isinstance(action, Measurement):
                qubit_indices = {action.arguments[0].index()}
            else:
                continue
            allocated_qubits = set.union(allocated_qubits, qubit_indices)
        return allocated_qubits

    def defgate(self, name, matrix):
        """
        Define a new static gate.

        :param string name: The name of the gate.
        :param array-like matrix: List of lists or Numpy 2d array.
        :return: The Program instance.
        :rtype: Program
        """
        self.defined_gates.append(DefGate(name, matrix))
        return self

    def measure(self, qubit_index, classical_reg):
        """
        Measures a qubit at qubit_index and puts the result in classical_reg

        :param int qubit_index: The address of the qubit to measure.
        :param int classical_reg: The address of the classical bit to store the result.
        :returns: The Quil Program with the appropriate measure instruction appended, e.g.
                  MEASURE 0 [1]
        :rtype: Program
        """
        return self.inst(MEASURE(qubit_index, classical_reg))

    def measure_all(self, *qubit_reg_pairs):
        """
        Measures many qubits into their specified classical bits, in the order
        they were entered.

        :param Tuple qubit_reg_pairs: Tuples of qubit indices paired with classical bits.
        :return: The Quil Program with the appropriate measure instructions appended, e.g.
                  MEASURE 0 [1]
                  MEASURE 1 [2]
                  MEASURE 2 [3]
        :rtype: Program
        """
        for qubit_index, classical_reg in qubit_reg_pairs:
            self.inst(MEASURE(qubit_index, classical_reg))
        return self

    def while_do(self, classical_reg, q_program):
        """
        While a classical register at index classical_reg is 1, loop q_program

        :param int classical_reg: The classical register to check
        :param Program q_program: The Quil program to loop.
        :return: The Quil Program with the loop instructions added.
        :rtype: Program
        """
        w_loop = While(Addr(classical_reg))
        w_loop.Body.inst(q_program)
        return self.inst(w_loop)

    def if_then(self, classical_reg, if_program, else_program=None):
        """
        If the classical register at index classical reg is 1, run if_program, else run
        else_program.

        :param int classical_reg: The classical register to check as the condition
        :param Program if_program: A Quil program to execute if classical_reg is 1
        :param Program else_program: A Quil program to execute if classical_reg is 0. This
            argument is optional and defaults to an empty Program.
        :returns: The Quil Program with the branching instructions added.
        :rtype: Program
        """

        else_program = else_program if else_program is not None else Program()

        branch = If(Addr(classical_reg))
        branch.Then.inst(if_program)
        branch.Else.inst(else_program)
        return self.inst(branch)

    def out(self):
        """
        Converts the Quil program to a readable string.

        :return: String form of a program
        :rtype: string
        """
        s = ""
        for dg in self.defined_gates:
            s += dg.out()
            s += "\n"
        s += super(Program, self).out()
        return s


def merge_programs(prog_list):
    """
    Merges a list of pyQuil programs into a single one by appending them in sequence

    :param list prog_list: A list of pyquil programs
    :return: a single pyQuil program
    :rtype: Program
    """
    return sum(prog_list, Program())
