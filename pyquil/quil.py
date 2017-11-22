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
from itertools import count
from math import pi

import numpy as np
from six import string_types

from pyquil._parser.PyQuilListener import run_parser
from pyquil.kraus import _check_kraus_ops, _create_kraus_pragmas
from .gates import MEASURE, STANDARD_GATES, H
from .quilbase import (DefGate, Gate, Measurement, Pragma, AbstractInstruction, Qubit,
                       unpack_qubit, Jump, LabelPlaceholder, Label, JumpConditional, JumpTarget, JumpUnless, JumpWhen,
                       QubitPlaceholder, Addr)


class Program(object):
    def __init__(self, *instructions):
        self._defined_gates = []
        # Implementation note: the key difference between the private _instructions and the public instructions
        # property below is that the private _instructions list may contain placeholder values
        self._instructions = []

        # Performance optimization: as stated above _instructions may contain placeholder values so the program must
        # first be synthesized. _synthesized_instructions is simply a cache on the result of the _synthesize() method.
        # It is marked as None whenever new instructions are added.
        self._synthesized_instructions = None

        self.inst(*instructions)

    @property
    def defined_gates(self):
        """
        A list of defined gates on the program.
        """
        return self._defined_gates

    @property
    def instructions(self):
        """
        Fill in any placeholders and return a list of quil AbstractInstructions.
        """
        if self._synthesized_instructions is None:
            self._synthesized_instructions = self._synthesize()

        return self._synthesized_instructions

    def inst(self, *instructions):
        """
        Mutates the Program object by appending new instructions.

        This function accepts a number of different valid forms, e.g.
            >>> p = Program()
            >>> p.inst(H(0)) # A single instruction
            >>> p.inst(H(0), H(1)) # Multiple instructions
            >>> p.inst([H(0), H(1)]) # A list of instructions
            >>> p.inst(("H", 1)) # A tuple representing an instruction
            >>> p.inst("H 0") # A string representing an instruction
            >>> q = Program()
            >>> p.inst(q) # Another program

        It can also be chained:
            >>> p = Program()
            >>> p.inst(H(0)).inst(H(1))

        :param instructions: A list of Instruction objects, e.g. Gates
        :return: self for method chaining
        """
        for instruction in instructions:
            if isinstance(instruction, list):
                self.inst(*instruction)
            elif isinstance(instruction, tuple):
                if len(instruction) == 0:
                    raise ValueError("tuple should have at least one element")
                elif len(instruction) == 1:
                    self.inst(instruction[0])
                else:
                    op = instruction[0]
                    if op == "MEASURE":
                        if len(instruction) == 2:
                            self.measure(instruction[1])
                        else:
                            self.measure(instruction[1], instruction[2])
                    else:
                        params = []
                        possible_params = instruction[1]
                        rest = instruction[2:]
                        if isinstance(possible_params, list):
                            params = possible_params
                        else:
                            rest = [possible_params] + list(rest)
                        self.gate(op, params, rest)
            elif isinstance(instruction, string_types):
                self.inst(run_parser(instruction.strip()))
            elif isinstance(instruction, DefGate):
                self._defined_gates.append(instruction)
            elif isinstance(instruction, AbstractInstruction):
                self._instructions.append(instruction)
                self._synthesized_instructions = None
            elif isinstance(instruction, Program):
                if id(self) == id(instruction):
                    raise ValueError("Nesting a program inside itself is not supported")

                self._defined_gates.extend(list(instruction._defined_gates))
                self._instructions.extend(list(instruction._instructions))
                self._synthesized_instructions = None
            else:
                raise TypeError("Invalid instruction: {}".format(instruction))

        return self

    def gate(self, name, params, qubits):
        """
        Add a gate to the program.

        :param string name: The name of the gate.
        :param list params: Parameters to send to the gate.
        :param list qubits: Qubits that the gate operates on.
        :return: The Program instance
        :rtype: Program
        """
        return self.inst(Gate(name, params, [unpack_qubit(q) for q in qubits]))

    def defgate(self, name, matrix):
        """
        Define a new static gate.

        :param string name: The name of the gate.
        :param array-like matrix: List of lists or Numpy 2d array.
        :return: The Program instance.
        :rtype: Program
        """
        self._defined_gates.append(DefGate(name, matrix))
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
        return self.inst(_create_kraus_pragmas(name, tuple(qubit_indices), kraus_ops))

    def no_noise(self):
        """
        Prevent a noisy gate definition from being applied to the immediately following Gate
        instruction.

        :return: Program
        """
        return self.inst(Pragma("NO-NOISE"))

    def measure(self, qubit_index, classical_reg=None):
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

        Equivalent to the following construction:

        WHILE [c]:
           instr...
        =>
          LABEL @START
          JUMP-UNLESS @END [c]
          instr...
          JUMP @START
          LABEL @END

        :param int classical_reg: The classical register to check
        :param Program q_program: The Quil program to loop.
        :return: The Quil Program with the loop instructions added.
        :rtype: Program
        """
        label_start = LabelPlaceholder("START")
        label_end = LabelPlaceholder("END")
        self.inst(JumpTarget(label_start))
        self.inst(JumpUnless(target=label_end, condition=Addr(classical_reg)))
        self.inst(q_program)
        self.inst(Jump(label_start))
        self.inst(JumpTarget(label_end))
        return self

    def if_then(self, classical_reg, if_program, else_program=None):
        """
        If the classical register at index classical reg is 1, run if_program, else run
        else_program.

        Equivalent to the following construction:
        IF [c]:
           instrA...
        ELSE:
           instrB...
        =>
          JUMP-WHEN @THEN [c]
          instrB...
          JUMP @END
          LABEL @THEN
          instrA...
          LABEL @END

        :param int classical_reg: The classical register to check as the condition
        :param Program if_program: A Quil program to execute if classical_reg is 1
        :param Program else_program: A Quil program to execute if classical_reg is 0. This
            argument is optional and defaults to an empty Program.
        :returns: The Quil Program with the branching instructions added.
        :rtype: Program
        """
        else_program = else_program if else_program is not None else Program()

        label_then = LabelPlaceholder("THEN")
        label_end = LabelPlaceholder("END")
        self.inst(JumpWhen(target=label_then, condition=Addr(classical_reg)))
        self.inst(else_program)
        self.inst(Jump(label_end))
        self.inst(JumpTarget(label_then))
        self.inst(if_program)
        self.inst(JumpTarget(label_end))
        return self

    def alloc(self):
        """
        Get a new qubit.

        :return: A qubit.
        :rtype: Qubit
        """
        return QubitPlaceholder()

    def out(self):
        """
        Converts the Quil program to a readable string.

        :return: String form of a program
        :rtype: string
        """
        s = ""
        for dg in self._defined_gates:
            s += dg.out()
            s += "\n"
        for instr in self.instructions:
            s += instr.out() + "\n"
        return s

    def get_qubits(self):
        """
        Returns all of the qubit indices used in this program, including gate applications and
        allocated qubits. e.g.
            >>> p = Program()
            >>> p.inst(("H", 1))
            >>> p.get_qubits()
            {1}
            >>> q = p.alloc()
            >>> p.inst(H(q))
            >>> len(p.get_qubits())
            2

        :return: A set of all the qubit indices used in this program
        :rtype: set
        """
        qubits = set()
        for instr in self.instructions:
            if isinstance(instr, Gate):
                qubits |= {q.index for q in instr.qubits}
            elif isinstance(instr, Measurement):
                qubits.add(instr.qubit.index)
        return qubits

    def is_protoquil(self):
        """
        Protoquil programs may only contain gates, no classical instructions and no jumps.

        :return: True if the Program is Protoquil, False otherwise
        """
        for instr in self._instructions:
            if not isinstance(instr, Gate):
                return False
        return True

    def pop(self):
        """
        Pops off the last instruction.

        :return: The instruction that was popped.
        :rtype: tuple
        """
        res = self._instructions.pop()
        self._synthesized_instructions = None
        return res

    def dagger(self, inv_dict=None, suffix="-INV"):
        """
        Creates the conjugate transpose of the Quil program. The program must not
        contain any irreversible actions (measurement, control flow, qubit allocation).

        :return: The Quil program's inverse
        :rtype: Program

        """
        if not self.is_protoquil():
            raise ValueError("Program must be valid Protoquil")

        daggered = Program()

        for gate in self._defined_gates:
            if inv_dict is None or gate.name not in inv_dict:
                daggered.defgate(gate.name + suffix, gate.matrix.T.conj())

        for gate in reversed(self._instructions):
            if gate.name in STANDARD_GATES:
                if gate.name == "S":
                    daggered.inst(STANDARD_GATES["PHASE"](-pi / 2, *gate.qubits))
                elif gate.name == "T":
                    daggered.inst(STANDARD_GATES["RZ"](pi / 4, *gate.qubits))
                elif gate.name == "ISWAP":
                    daggered.inst(STANDARD_GATES["PSWAP"](pi / 2, *gate.qubits))
                else:
                    negated_params = list(map(lambda x: -1 * x, gate.params))
                    daggered.inst(STANDARD_GATES[gate.name](*(negated_params + gate.qubits)))
            else:
                if inv_dict is None or gate.name not in inv_dict:
                    gate_inv_name = gate.name + suffix
                else:
                    gate_inv_name = inv_dict[gate.name]

                daggered.inst(tuple([gate_inv_name] + gate.qubits))

        return daggered

    def _synthesize(self):
        """
        Takes a program which may contain placeholders and assigns them all defined values.

        For qubit placeholders:
        1. We look through the program to find all the known indexes of qubits and add them to a set
        2. We create a mapping from undefined qubits to their newly assigned index
        3. For every qubit placeholder in the program, if it's not already been assigned then look through the set of
            known indexes and find the lowest available one

        For label placeholders:
        1. Start a counter at 1
        2. For every label placeholder in the program, replace it with a defined label using the counter and increment
            the counter

        :return: List of AbstractInstructions with all placeholders removed
        """
        used_indexes = set()
        for instr in self._instructions:
            if isinstance(instr, Gate):
                for q in instr.qubits:
                    if not isinstance(q, QubitPlaceholder):
                        used_indexes.add(q.index)
            elif isinstance(instr, Measurement):
                if not isinstance(instr.qubit, QubitPlaceholder):
                    used_indexes.add(instr.qubit.index)

        def find_available_index():
            # Just do a linear search.
            for i in count(start=0, step=1):
                if i not in used_indexes:
                    return i

        qubit_mapping = dict()

        def remap_qubit(qubit):
            if not isinstance(qubit, QubitPlaceholder):
                return qubit
            if id(qubit) in qubit_mapping:
                return qubit_mapping[id(qubit)]
            else:
                available_index = find_available_index()
                used_indexes.add(available_index)
                remapped_qubit = Qubit(available_index)
                qubit_mapping[id(qubit)] = remapped_qubit
                return remapped_qubit

        label_mapping = dict()
        label_counter = 1

        def remap_label(placeholder):
            if id(placeholder) in label_mapping:
                return label_mapping[id(placeholder)]
            else:
                label = Label(placeholder.prefix + str(label_counter))
                label_mapping[id(placeholder)] = label
                return label

        result = []
        for instr in self._instructions:
            # Remap qubits on Gate and Measurement instructions
            if isinstance(instr, Gate):
                remapped_qubits = [remap_qubit(q) for q in instr.qubits]
                result.append(Gate(instr.name, instr.params, remapped_qubits))
            elif isinstance(instr, Measurement):
                result.append(Measurement(remap_qubit(instr.qubit), instr.classical_reg))

            # Remap any label placeholders on jump or target instructions
            elif isinstance(instr, Jump) and isinstance(instr.target, LabelPlaceholder):
                result.append(Jump(remap_label(instr.target)))
                label_counter += 1
            elif isinstance(instr, JumpTarget) and isinstance(instr.label, LabelPlaceholder):
                result.append(JumpTarget(remap_label(instr.label)))
                label_counter += 1
            elif isinstance(instr, JumpConditional) and isinstance(instr.target, LabelPlaceholder):
                new_label = remap_label(instr.target)
                if isinstance(instr, JumpWhen):
                    result.append(JumpWhen(new_label, instr.condition))
                elif isinstance(instr, JumpUnless):
                    result.append(JumpUnless(new_label, instr.condition))
                else:
                    raise TypeError("Encountered a JumpConditional that wasn't JumpWhen or JumpUnless: {} {}"
                                    .format(type(instr), instr))
                label_counter += 1

            # Otherwise simply add it to the result
            else:
                result.append(instr)

        return result

    def __add__(self, other):
        """
        Concatenate two programs together, returning a new one.

        :param Program other: Another program or instruction to concatenate to this one.
        :return: A newly concatenated program.
        :rtype: Program
        """
        p = Program()
        p.inst(self)
        p.inst(other)
        return p

    def __getitem__(self, index):
        """
        Allows indexing into the program to get an action.

        :param index: The action at the specified index.
        :return:
        """
        return self.instructions[index]

    def __iter__(self):
        """
        Allow built in iteration through a program's instructions, e.g. [a for a in Program(X(0))]

        :return:
        """
        return self.instructions.__iter__()

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.out() == other.out()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._instructions)

    def __str__(self):
        return self.out()


def merge_programs(prog_list):
    """
    Merges a list of pyQuil programs into a single one by appending them in sequence

    :param list prog_list: A list of pyquil programs
    :return: a single pyQuil program
    :rtype: Program
    """
    return sum(prog_list, Program())
