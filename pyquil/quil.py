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
import warnings
from itertools import count
import itertools
from math import pi
import types

import numpy as np
from six import string_types

from pyquil._parser.PyQuilListener import run_parser
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas
from pyquil.parameters import format_parameter
from pyquil.quilatom import LabelPlaceholder, QubitPlaceholder, unpack_qubit
from .gates import MEASURE, STANDARD_GATES, H
from .quilbase import (DefGate, Gate, Measurement, Pragma, AbstractInstruction, Qubit,
                       Jump, Label, JumpConditional, JumpTarget, JumpUnless, JumpWhen, Addr)


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
            self._synthesize()

        return self._synthesized_instructions

    def inst(self, *instructions):
        """
        Mutates the Program object by appending new instructions.

        This function accepts a number of different valid forms, e.g.

            >>> p = Program()
            >>> p.inst(H(0)) # A single instruction
            >>> p.inst(H(0), H(1)) # Multiple instructions
            >>> p.inst([H(0), H(1)]) # A list of instructions
            >>> p.inst(H(i) for i in range(4)) # A generator of instructions
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
            elif isinstance(instruction, types.GeneratorType):
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
            elif isinstance(instruction, Program):
                if id(self) == id(instruction):
                    raise ValueError("Nesting a program inside itself is not supported")

                for defgate in instruction._defined_gates:
                    self.inst(defgate)
                for instr in instruction._instructions:
                    self.inst(instr)

            # Implementation note: these two base cases are the only ones which modify the program

            elif isinstance(instruction, DefGate):
                defined_gate_names = [gate.name for gate in self._defined_gates]
                if instruction.name in defined_gate_names:
                    warnings.warn("Gate {} has already been defined in this program".format(instruction.name))

                self._defined_gates.append(instruction)
            elif isinstance(instruction, AbstractInstruction):
                self._instructions.append(instruction)
                self._synthesized_instructions = None
            else:
                raise TypeError("Invalid instruction: {}".format(instruction))

        return self

    def gate(self, name, params, qubits):
        """
        Add a gate to the program.

        .. note::

            The matrix elements along each axis are ordered by bitstring. For two qubits the order
            is ``00, 01, 10, 11``, where the the bits **are ordered in reverse** by the qubit index,
            i.e., for qubits 0 and 1 the bitstring ``01`` indicates that qubit 0 is in the state 1.
            See also :ref:`the related documentation section in the QVM Overview <basis-ordering>`.

        :param string name: The name of the gate.
        :param list params: Parameters to send to the gate.
        :param list qubits: Qubits that the gate operates on.
        :return: The Program instance
        :rtype: Program
        """
        return self.inst(Gate(name, params, [unpack_qubit(q) for q in qubits]))

    def defgate(self, name, matrix, parameters=None):
        """
        Define a new static gate.

        .. note::

            The matrix elements along each axis are ordered by bitstring. For two qubits the order
            is ``00, 01, 10, 11``, where the the bits **are ordered in reverse** by the qubit index,
            i.e., for qubits 0 and 1 the bitstring ``01`` indicates that qubit 0 is in the state 1.
            See also :ref:`the related documentation section in the QVM Overview <basis-ordering>`.


        :param string name: The name of the gate.
        :param array-like matrix: List of lists or Numpy 2d array.
        :param list parameters: list of parameters that are used in this gate
        :return: The Program instance.
        :rtype: Program
        """
        return self.inst(DefGate(name, matrix, parameters))

    def define_noisy_gate(self, name, qubit_indices, kraus_ops):
        """
        Overload a static ideal gate with a noisy one defined in terms of a Kraus map.

        .. note::

            The matrix elements along each axis are ordered by bitstring. For two qubits the order
            is ``00, 01, 10, 11``, where the the bits **are ordered in reverse** by the qubit index,
            i.e., for qubits 0 and 1 the bitstring ``01`` indicates that qubit 0 is in the state 1.
            See also :ref:`the related documentation section in the QVM Overview <basis-ordering>`.


        :param str name: The name of the gate.
        :param tuple|list qubit_indices: The qubits it acts on.
        :param tuple|list kraus_ops: The Kraus operators.
        :return: The Program instance
        :rtype: Program
        """
        kraus_ops = [np.asarray(k, dtype=np.complex128) for k in kraus_ops]
        _check_kraus_ops(len(qubit_indices), kraus_ops)
        return self.inst(_create_kraus_pragmas(name, tuple(qubit_indices), kraus_ops))

    def define_noisy_readout(self, qubit_index, p00, p11):
        """
        For this program define a classical bit flip readout error channel parametrized by
        ``p00`` and ``p11``. This models the effect of thermal noise that corrupts the readout
        signal **after** it has interrogated the qubit.

        :param int qubit_index: The qubit with noisy readout.
        :param float p00: The probability of obtaining the measurement result 0 given that the qubit
          is in state 0.
        :param float p11: The probability of obtaining the measurement result 1 given that the qubit
          is in state 1.
        :return: The Program with an appended READOUT-POVM Pragma.
        :rtype: Program
        """
        if not 0. <= p00 <= 1.:
            raise ValueError("p00 must be in the interval [0,1].")
        if not 0. <= p11 <= 1.:
            raise ValueError("p11 must be in the interval [0,1].")
        if not isinstance(qubit_index, int):
            raise TypeError("qubit_index must be a non-negative integer.")
        if not qubit_index >= 0:
            raise ValueError("qubit_index must be a non-negative integer.")
        p00 = float(p00)
        p11 = float(p11)
        aprobs = [p00, 1. - p11, 1. - p00, p11]
        aprobs_str = "({})".format(" ".join(format_parameter(p) for p in aprobs))
        pragma = Pragma("READOUT-POVM", [qubit_index], aprobs_str)
        return self.inst(pragma)

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

        .. code::

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

        .. code::

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

        .. code::

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
        warnings.warn("`alloc` is deprecated and will be removed in a future version of pyQuil. "
                      "Please create a `QubitPlaceholder` directly", DeprecationWarning)
        return QubitPlaceholder()

    def out(self):
        """
        Converts the Quil program to a readable string.

        :return: String form of a program
        :rtype: string
        """
        return '\n'.join(itertools.chain(
            (dg.out() for dg in self._defined_gates),
            (instr.out() for instr in self.instructions),
            [''],
        ))

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

        Changed in 1.9: Either all qubits must be defined or all undefined. If qubits are
        undefined, you are required to provide a qubit mapping via :py:func:`synthesize()`.
        All labels must be defined. These restrictions will be relaxed in future versions of
        pyQuil if necessary.

        Changed in 1.9: This function now returns ``self`` and updates
        ``self._synthesized_instructions``. See :py:func:`synthesize_program` for the old behavior.

        :return: This object with the ``_synthesized_instructions`` member set.
        """
        self._synthesized_instructions = instantiate_labels(self._instructions)
        return self

    def synthesize(self, qubit_mapping=None):
        """
        Takes a program which contains placeholders and assigns them all defined values.

        Either all qubits must be defined or all undefined. If qubits are
        undefined, you are required to provide a qubit mapping. This will be done automatically
        in future versions of pyQuil if necessary.

        All labels for classical control must be defined. These restrictions will be relaxed in
        future versions of pyQuil if necessary.

        This function will throw an error if the program was previously synthesized, as this
        indicates this function is being used incorrectly.

        :param qubit_mapping: A dictionary-like object that maps from :py:class:`QubitPlaceholder`
            to :py:class:`Qubit` or ``int`` (but not both).
        :return: This object with the ``_synthesized_instructions`` member set.
        """
        if self._synthesized_instructions is not None:
            raise RuntimeError("The program was already synthesized! "
                               "Make sure you haven't previously synthesized the program.")
        self._synthesized_instructions = synthesize_program(self._instructions, qubit_mapping)
        return self

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


def _what_type_of_qubit_does_it_use(instructions):
    has_placeholders = False
    has_real_qubits = False
    qubits = set()

    for instr in instructions:
        if isinstance(instr, Gate):
            for q in instr.qubits:
                qubits.add(q)
                if isinstance(q, QubitPlaceholder):
                    has_placeholders = True
                elif isinstance(q, Qubit):
                    has_real_qubits = True
                else:
                    raise ValueError("Unknown qubit type {}".format(q))
        elif isinstance(instr, Measurement):
            qubits.add(instr.qubit)
            if isinstance(instr.qubit, QubitPlaceholder):
                has_placeholders = True
            elif isinstance(instr.qubit, Qubit):
                has_real_qubits = True
            else:
                raise ValueError("Unknown qubit type {}".format(instr.qubit))

    if not (has_placeholders or has_real_qubits):
        warnings.warn("Your program doesn't use any qubits")

    if has_placeholders and has_real_qubits:
        raise ValueError("Your program mixes instatiated qubits with placeholders")

    return has_placeholders, has_real_qubits, len(qubits)


def _what_type_of_label_does_it_use(instructions):
    has_placeholders = False
    has_real_labels = False

    labels = set()

    for instr in instructions:
        if isinstance(instr, Jump):
            labels.add(instr.target)
            if isinstance(instr.target, LabelPlaceholder):
                has_placeholders = True
            elif isinstance(instr.target, Label):
                has_real_labels = True
            else:
                raise ValueError("Unknown label type {}".format(instr.target))

        elif isinstance(instr, JumpTarget):
            labels.add(instr.label)
            if isinstance(instr.label, LabelPlaceholder):
                has_placeholders = True
            elif isinstance(instr.label, Label):
                has_real_labels = True
            else:
                raise ValueError("Unknown label type {}".format(instr.label))

        elif isinstance(instr, JumpConditional) and isinstance(instr.target, LabelPlaceholder):
            labels.add(instr.target)
            if isinstance(instr.target, LabelPlaceholder):
                has_placeholders = True
            elif isinstance(instr.target, Label):
                has_real_labels = True
            else:
                raise ValueError("Unknown label type {}".format(instr.target))

    if has_placeholders and has_real_labels:
        raise ValueError("Your program mixes instantiated labels with placeholders")

    return has_placeholders, has_real_labels, len(labels)


def synthesize_program(instructions, qubit_mapping=None):
    """
    Takes a program which contains placeholders and assigns them all defined values.

    Either all qubits must be defined or all undefined. If qubits are
    undefined, you are required to provide a qubit mapping. This will be done automatically
    in future versions of pyQuil if necessary.

    All labels for classical control must be defined. These restrictions will be relaxed in
    future versions of pyQuil if necessary.

    :param qubit_mapping: A dictionary-like object that maps from :py:class:`QubitPlaceholder`
        to :py:class:`Qubit` or ``int`` (but not both).
    :return: list of instructions with all qubit placeholders assigned to real qubits.
    """
    fake_qubits, real_qubits, n_qubits = _what_type_of_qubit_does_it_use(instructions)
    if real_qubits:
        return instructions

    if qubit_mapping is None:
        raise NotImplementedError("Please specify a qubit mapping. "
                                  "At some point this may happen automatically")
    else:
        if all(isinstance(v, Qubit) for v in qubit_mapping.values()):
            pass  # we good
        elif all(isinstance(v, int) for v in qubit_mapping.values()):
            qubit_mapping = {k: Qubit(v) for k, v in qubit_mapping.items()}
        else:
            raise ValueError("Qubit mapping must map to type Qubit or int (but not both)")

    fake_labels, _, _ = _what_type_of_label_does_it_use(instructions)
    if fake_labels:
        raise NotImplementedError("Please use explicit labels for the time being")

    result = []
    for instr in instructions:
        # Remap qubits on Gate and Measurement instructions
        if isinstance(instr, Gate):
            remapped_qubits = [qubit_mapping[q] for q in instr.qubits]
            result.append(Gate(instr.name, instr.params, remapped_qubits))
        elif isinstance(instr, Measurement):
            result.append(Measurement(qubit_mapping[instr.qubit], instr.classical_reg))

        # Otherwise simply add it to the result
        else:
            result.append(instr)

    return result


def _get_label(placeholder, label_mapping, label_i):
    if placeholder in label_mapping:
        return label_mapping[placeholder], label_mapping, label_i

    new_target = Label("{}{}".format(placeholder.prefix, label_i))
    label_i += 1
    label_mapping[placeholder] = new_target
    return new_target, label_mapping, label_i


def instantiate_labels(instructions):
    """
    Takes a program which contains placeholders and assigns them all defined values.

    Either all qubits must be defined or all undefined. If qubits are
    undefined, you are required to provide a qubit mapping. This will be done automatically
    in future versions of pyQuil if necessary.

    All labels for classical control must be defined. These restrictions will be relaxed in
    future versions of pyQuil if necessary.

    :param qubit_mapping: A dictionary-like object that maps from :py:class:`QubitPlaceholder`
        to :py:class:`Qubit` or ``int`` (but not both).
    :return: list of instructions with all qubit placeholders assigned to real qubits.
    """
    has_target_label = (Jump, JumpConditional)
    has_label_label = (JumpTarget,)

    label_i = 1
    result = []
    label_mapping = dict()
    for instr in instructions:
        if isinstance(instr, Jump) and isinstance(instr.target, LabelPlaceholder):
            new_target, label_mapping, label_i = _get_label(instr.target, label_mapping, label_i)
            result.append(Jump(new_target))
        elif isinstance(instr, JumpConditional) and isinstance(instr.target, LabelPlaceholder):
            new_target, label_mapping, label_i = _get_label(instr.target, label_mapping, label_i)
            cls = instr.__class__  # Make the correct subclass
            result.append(cls(new_target, instr.condition))
        elif isinstance(instr, JumpTarget) and isinstance(instr.label, LabelPlaceholder):
            new_label, label_mapping, label_i = _get_label(instr.label, label_mapping, label_i)
            result.append(JumpTarget(new_label))
        else:
            result.append(instr)

    return result


def merge_programs(prog_list):
    """
    Merges a list of pyQuil programs into a single one by appending them in sequence

    :param list prog_list: A list of pyquil programs
    :return: a single pyQuil program
    :rtype: Program
    """
    return sum(prog_list, Program())


def shift_quantum_gates(program, shift_offset):
    """
    Shifts a quantum gates in a quil program so that all qubit indices change by a certain offset
    :param program: a pyquil Program
    :param shift_offset: integer
    :return: pyquil Program with shifted qubit indices
    """
    if not isinstance(shift_offset, int):
        raise ValueError("shift_offset must be an integer")
    if not isinstance(program, Program):
        raise ValueError("Program must be a pyquil Program instance")
    shifted_program = Program()
    shifted_program.inst(program)
    for instruct in shifted_program:
        if isinstance(instruct, Gate):
            for qubit in instruct.qubits:
                qubit.index += shift_offset
    return shifted_program
