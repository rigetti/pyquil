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
from math import pi

from six import integer_types

from pyquil.kraus import _check_kraus_ops, _create_kraus_pragmas
from .gates import MEASURE, STANDARD_GATES
from .quilbase import (InstructionGroup,
                       Instr,
                       Addr,
                       While,
                       If,
                       DefGate,
                       Gate,
                       Measurement,
                       AbstractQubit,
                       merge_resource_managers, Pragma)
import numpy as np


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

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.out() == other.out()

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_qubits(self):
        """
        Returns all of the qubit indices used in this program, including gate applications and
        allocated qubits. e.g.
            >>> p = Program()
            >>> p.inst(("H", 1))
            >>> p.get_qubits()
            {1}
            >>> q = p.alloc()
            >>> len(p.get_qubits())
            2

        :return: A set of all the qubit indices used in this program, synthesizing freely
         allocated qubits if neccessary.
        :rtype: set
        """
        qubits = set()
        self.synthesize()
        for ii, action in self:
            if isinstance(action, Gate):
                qubit_indices = {qq.index() for qq in action.arguments}
            elif isinstance(action, Measurement):
                qubit_indices = {action.arguments[0].index()}
            elif isinstance(action, Instr):
                qubit_indices = set()
                for arg in action.arguments:
                    if isinstance(arg, integer_types):
                        qubit_indices.add(arg)
                    elif isinstance(arg, AbstractQubit):
                        qubit_indices.add(arg.index())
            else:
                continue
            qubits = set.union(qubits, qubit_indices)
        return qubits

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

    def define_noisy_gate(self, name, qubit_indices, kraus_ops):
        """
        Overload a static ideal gate with a noisy one defined in terms of a Kraus map.

        :param str name: The name of the gate.
        :param tuple|list qubit_indices: The qubits it acts on.
        :param tuple|list kraus_ops: The Kraus operators.
        :return: The Program instance
        :rtype: Program
        """
        kraus_ops = [np.asarray(k, dtype=np.complex128) for k in kraus_ops]
        _check_kraus_ops(len(qubit_indices), kraus_ops)
        self.inst(_create_kraus_pragmas(name, tuple(qubit_indices), kraus_ops))
        return self

    def no_noise(self):
        """
        Prevent a noisy gate definition from being applied to the immediately following Gate
        instruction.

        :return: Program
        """
        self.inst(Pragma("NO-NOISE"))
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

    def dagger(self, inv_dict=None, suffix="-INV"):
        """
        Creates the conjugate transpose of the Quil program. The program must not
        contain any irreversible actions (measurement, control flow, qubit allocation).

        :return: The Quil program's inverse
        :rtype: Program

        """

        for action in self.actions:
            assert action[0] == 0, "Program must be valid Protoquil"
            gate = action[1]
            assert not isinstance(gate, Measurement), "Program cannot contain measurements"
            assert not isinstance(gate, While) and not isinstance(gate, If), \
                "Program cannot contain control flow"

        daggered = Program()

        for gate in self.defined_gates:
            if inv_dict is None or gate.name not in inv_dict:
                daggered.defgate(gate.name + suffix, gate.matrix.T.conj())

        for action in self.actions[::-1]:
            gate = action[1]
            if gate.operator_name in STANDARD_GATES:
                if gate.operator_name == "S":
                    daggered.inst(STANDARD_GATES["PHASE"](-pi / 2, *gate.arguments))
                elif gate.operator_name == "T":
                    daggered.inst(STANDARD_GATES["RZ"](pi / 4, *gate.arguments))
                elif gate.operator_name == "ISWAP":
                    daggered.inst(STANDARD_GATES["PSWAP"](pi / 2, *gate.arguments))
                else:
                    negated_params = list(map(lambda x: -1 * x, gate.parameters))
                    daggered.inst(STANDARD_GATES[gate.operator_name](*(negated_params + gate.arguments)))
            else:
                if inv_dict is None or gate.operator_name not in inv_dict:
                    gate_inv_name = gate.operator_name + suffix
                else:
                    gate_inv_name = inv_dict[gate.operator_name]

                daggered.inst(tuple([gate_inv_name] + gate.arguments))

        return daggered

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
