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
Contains the core pyQuil objects that correspond to Quil instructions.
"""

import numpy as np
from copy import deepcopy
from .quil_atom import QuilAtom
from .slot import Slot
from .resource_manager import (AbstractQubit, DirectQubit, Qubit,
                               ResourceManager, check_live_qubit, merge_resource_managers)
from six import integer_types

allow_raw_instructions = True
"""
Allow constructing programs containing raw instructions.
"""


def issubinstance(x, cls):
    """
    Checks if class x is an instance or subclass of cls.
    """
    return isinstance(x, cls) or issubclass(x.__class__, cls)


# These are the first values to a 2-tuple.
# This indicates all regular Quil instructions except resource management.
ACTION_INSTALL_INSTRUCTION = 0
# These are for resource management.
ACTION_INSTANTIATE_QUBIT = 1
ACTION_RELEASE_QUBIT = 2


def action(type, obj):
    return (type, obj)


def action_install(obj):
    return action(ACTION_INSTALL_INSTRUCTION, obj)


def format_matrix_element(element):
    """
    Formats a parameterized matrix element.

    :param element: {int, float, complex, str} The parameterized element to format.
    """
    if isinstance(element, integer_types) or isinstance(element, (float, complex)):
        return format_parameter(element)
    elif isinstance(element, str):
        return element
    else:
        assert False, "Invalid matrix element: %r" % element


def format_parameter(element):
    """
    Formats a particular parameter.

    :param element: {int, float, long, complex, Slot} Formats a parameter for Quil output.
    """
    if isinstance(element, integer_types) or isinstance(element, float):
        return repr(element)
    elif isinstance(element, complex):
        r = element.real
        i = element.imag
        if i < 0:
            return repr(r) + "-" + repr(abs(i)) + "i"
        else:
            return repr(r) + "+" + repr(i) + "i"
    elif isinstance(element, Slot):
        return format_parameter(element.value())
    assert False, "Invalid parameter: %r" % element


class Addr(QuilAtom):
    """
    Representation of a classical bit address.

    :param int value: The classical address.
    """

    def __init__(self, value):
        if not isinstance(value, integer_types) or value < 0:
            raise TypeError("Addr value must be a non-negative int")
        self.address = value

    def __repr__(self):
        return "<Addr {0}>".format(self.address)

    def __str__(self):
        return "[{0}]".format(self.address)


class Label(QuilAtom):
    """
    Representation of a label.

    :param string label_name: The label name.
    """

    def __init__(self, label_name):
        self.name = label_name

    def __repr__(self):
        return "<Label {0}>".format(repr(self.name))

    def __str__(self):
        return "@" + self.name


class QuilAction(object):
    """
    Representation of some executable code, i.e., something that can be
    synthesized into final Quil instructions.
    """

    def synthesize(self, resource_manager=None):
        raise NotImplementedError()


class AbstractInstruction(QuilAction):
    """
    Abstract class for representing single instructionos.
    """

    def synthesize(self, resource_manager=None):
        return [self]

    def out(self):
        return NotImplementedError()


class DefGate(AbstractInstruction):
    """
    A DEFGATE directive.

    :param string name: The name of the newly defined gate.
    :param array-like matrix: {list, nparray, np.matrix} The matrix defining this gate.
    """

    def __init__(self, name, matrix):
        assert isinstance(name, str)
        assert isinstance(matrix, (list, np.ndarray, np.matrix))
        if isinstance(matrix, list):
            rows = len(matrix)
            assert all([len(row) == rows for row in matrix]), "Matrix must be square."
        elif isinstance(matrix, (np.ndarray, np.matrix)):
            rows, cols = matrix.shape
            assert rows == cols, "Matrix must be square."
        else:
            raise TypeError("Matrix argument must be a list or NumPy array/matrix")

        if 0 != rows & (rows - 1):
            raise AssertionError("Dimension of matrix must be a power of 2, got {0}"
                                 .format(rows))
        self.name = name
        self.matrix = np.asarray(matrix)

        is_unitary = np.allclose(np.eye(rows), self.matrix.dot(self.matrix.T.conj()))
        if not is_unitary:
            raise AssertionError("Matrix must be unitary.")


    def out(self):
        """
        Prints a readable Quil string representation of this gate.

        :returns: String representation of a gate
        :rtype: string
        """
        result = "DEFGATE %s:\n" % (self.name)
        for row in self.matrix:
            result += "    "
            fcols = [format_matrix_element(col) for col in row]
            result += ", ".join(fcols)
            result += "\n"
        return result

    def get_constructor(self):
        """
        :returns: A function that constructs this gate on variable qubit indices. E.g.
                  `mygate.get_constructor()(1) applies the gate to qubit 1.`
        """
        return lambda *qubits: Gate(name=self.name, params=[], qubits=qubits)

    def num_args(self):
        """
        :return: The number of qubit arguments the gate takes.
        :rtype: int
        """
        rows = len(self.matrix)
        return int(np.log2(rows))


class InstructionGroup(QuilAction):
    """
    Representation of a sequence of instructions that can be synthesized into a Quil program.
    """

    def __init__(self, resource_manager=None):
        self.actions = []
        if resource_manager is None:
            self.resource_manager = ResourceManager()
        else:
            self.resource_manager = resource_manager

    def synthesize(self, resource_manager=None):
        synthesized = []
        for action_type, obj in self.actions:
            if action_type == ACTION_INSTALL_INSTRUCTION:
                synthesized.extend(obj.synthesize(self.resource_manager))
            elif action_type == ACTION_INSTANTIATE_QUBIT:
                self.resource_manager.instantiate(obj)
            elif action_type == ACTION_RELEASE_QUBIT:
                self.resource_manager.uninstantiate_index(obj.assignment)
            else:
                raise RuntimeError("encountered invalid action")

        return synthesized

    def __str__(self):
        return self.out()

    def out(self):
        instrs = self.synthesize()
        s = ""
        for instr in instrs:
            s += instr.out() + "\n"
        return s

    def alloc(self):
        """
        Get a new qubit.

        :return: A qubit.
        :rtype: Qubit
        """
        qubit = self.resource_manager.allocate_qubit()
        self.actions.append(action(ACTION_INSTANTIATE_QUBIT, qubit))

        return qubit

    def free(self, qubit):
        """
        Free a qubit.

        :param AbstractQubit q: An AbstractQubit instance.
        """
        check_live_qubit(qubit)

        if qubit.resource_manager != self.resource_manager:
            raise RuntimeError("qubit is managed by a different instruction group")

        self.actions.append(action(ACTION_RELEASE_QUBIT, qubit))
        self.resource_manager.free_qubit(qubit)

    def inst(self, *instructions):
        """
        Mutates the Program object by appending new instructions.

        :param instructions: A list of Instruction objects, e.g. Gates
        :return: self
        """
        for instruction in instructions:
            if isinstance(instruction, list):
                self.inst(*instruction)
            elif isinstance(instruction, tuple):
                if len(instruction) == 0:
                    raise ValueError("tuple should have at least one element")
                elif len(instruction) == 1:
                    self.actions.append(action_install(Instr(instruction[0], [], [])))
                else:
                    op = instruction[0]
                    params = []
                    possible_params = instruction[1]
                    rest = instruction[2:]
                    if isinstance(possible_params, list):
                        params = possible_params
                    else:
                        rest = [possible_params] + list(rest)
                    self.actions.append(action_install(Instr(op, params, rest)))
            elif isinstance(instruction, str):
                self.actions.append(action_install(RawInstr(instruction)))
            elif issubinstance(instruction, QuilAction):
                self.actions.append(action_install(instruction))
            elif issubinstance(instruction, InstructionGroup):
                self.resource_manager = merge_resource_managers(self.resource_manager,
                                                                instruction.resource_manager)
                self.actions.extend(list(instruction.actions))
            else:
                raise TypeError("Invalid instruction: {}".format(instruction))
        # Return self for method chaining.
        return self

    def __add__(self, instruction):
        p = deepcopy(self)
        return p.inst(instruction)

    def pop(self):
        """
        Pops off the last instruction.

        :return: The (action, instruction) pair for the instruction that was popped.
        :rtype: tuple
        """
        if 0 != len(self.actions):
            return self.actions.pop()

    def extract_qubits(self):
        """
        Return all qubit addresses involved in the instruction group.
        :return: Set of qubits.
        :rtype: set
        """
        qubits = set()
        for jj, act_jj in self.actions:
            if jj == ACTION_INSTALL_INSTRUCTION:
                if isinstance(act_jj, Instr):
                    qubits = qubits | act_jj.qubits()
                elif isinstance(act_jj, If):
                    qubits = qubits | act_jj.Then.extract_qubits() | act_jj.Else.extract_qubits()
                elif isinstance(act_jj, While):
                    qubits = qubits | act_jj.Body.extract_qubits()
                elif isinstance(act_jj, InstructionGroup):
                    qubits = qubits | act_jj.extract_qubits()
                elif isinstance(act_jj, (JumpTarget, JumpConditional, SimpleInstruction,
                                         UnaryClassicalInstruction, BinaryClassicalInstruction,
                                         Jump)):
                    continue
                else:
                    raise ValueError(type(act_jj))
            elif jj in (ACTION_INSTANTIATE_QUBIT, ACTION_RELEASE_QUBIT):
                continue
        return qubits


class JumpTarget(AbstractInstruction):
    """
    Representation of a target that can be jumped to.
    """

    def __init__(self, label):
        if not isinstance(label, Label):
            raise TypeError("label must be a Label")
        self.label = label

    def __repr__(self):
        return "<JumpTarget {0}>".format(str(self.label))

    def out(self):
        return "LABEL {0}".format(str(self.label))


class JumpConditional(AbstractInstruction):
    """
    Abstract representation of an conditional jump instruction.
    """

    def __init__(self, target, condition):
        if not isinstance(target, Label):
            raise TypeError("target should be a Label")
        if not isinstance(condition, Addr):
            raise TypeError("condition should be an Addr")
        self.target = target
        self.condition = condition

    def __str__(self):
        return self.out()

    def out(self):
        return "%s %s %s" % (self.op, self.target, self.condition)


class JumpWhen(JumpConditional):
    """
    The JUMP-WHEN instruction.
    """
    op = "JUMP-WHEN"


class JumpUnless(JumpConditional):
    """
    The JUMP-UNLESS instruction.
    """
    op = "JUMP-UNLESS"


class SimpleInstruction(AbstractInstruction):
    """
    Abstract class for simple instructions with no arguments.
    """

    def __str__(self):
        return self.out()

    def out(self):
        return self.op


class Halt(SimpleInstruction):
    """
    The HALT instruction.
    """
    op = "HALT"


class Wait(SimpleInstruction):
    """
    The WAIT instruction.
    """
    op = "WAIT"


class Reset(SimpleInstruction):
    """
    The RESET instruction.
    """
    op = "RESET"


class Nop(SimpleInstruction):
    """
    The RESET instruction.
    """
    op = "NOP"


class UnaryClassicalInstruction(AbstractInstruction):
    """
    The abstract class for unary classical instructions.
    """

    def __init__(self, target):
        if not isinstance(target, Addr):
            raise TypeError("target operand should be an Addr")
        self.target = target

    def __str__(self):
        return self.out()

    def out(self):
        return "%s %s" % (self.op, self.target)


class ClassicalTrue(UnaryClassicalInstruction):
    op = "TRUE"


class ClassicalFalse(UnaryClassicalInstruction):
    op = "FALSE"


class ClassicalNot(UnaryClassicalInstruction):
    op = "NOT"


class BinaryClassicalInstruction(AbstractInstruction):
    """
    The abstract class for binary classical instructions.
    """

    def __init__(self, left, right):
        if not isinstance(left, Addr):
            raise TypeError("left operand should be an Addr")
        if not isinstance(right, Addr):
            raise TypeError("right operand should be an Addr")
        self.left = left
        self.right = right

    def __str__(self):
        return self.out()

    def out(self):
        return "%s %s %s" % (self.op, self.left, self.right)


class ClassicalAnd(BinaryClassicalInstruction):
    op = "AND"


class ClassicalOr(BinaryClassicalInstruction):
    op = "OR"


class ClassicalMove(BinaryClassicalInstruction):
    op = "MOVE"


class ClassicalExchange(BinaryClassicalInstruction):
    op = "EXCHANGE"


class Jump(AbstractInstruction):
    """
    Representation of an unconditional jump instruction (JUMP).
    """

    def __init__(self, target):
        if not isinstance(target, Label):
            raise TypeError("target should be a Label")
        self.target = target

    def __str__(self):
        return self.out()

    def out(self):
        return "JUMP %s" % self.target


label_counter = 0


def reset_label_counter():
    global label_counter
    label_counter = 0


def gen_label(prefix="L"):
    """
    Generate a fresh label.

    :param string prefix: An optional prefix for the label name.
    :return: A new Label instance.
    :rtype: Label
    """
    global label_counter
    label_counter += 1
    return Label(prefix + str(label_counter))


class RawInstr(AbstractInstruction):
    """
    A raw instruction represented as a string.
    """

    def __init__(self, instr_str):
        if not isinstance(instr_str, str):
            raise TypeError("Raw instructions require a string.")
        if not allow_raw_instructions:
            raise RuntimeError("Raw instructions are not allowed. Consider changing"
                               "the variable `allow_raw_instructions` to `True`.")
        self.instr = instr_str

    def out(self):
        return self.instr

    def __repr__(self):
        return '<RawInstr>'

    def __str__(self):
        return self.instr


class Instr(AbstractInstruction):
    """
    Representation of an instruction represented by an operator, parameters, and arguments.
    """

    def __init__(self, op, params, args):
        if not isinstance(op, str):
            raise TypeError("op must be a string")
        self.operator_name = op
        self.parameters = params
        self.arguments = []
        if 0 != len(args):
            if isinstance(args[0], list):
                self.parameters = None if 0 == len(args[0]) else args[0]
                self.arguments = args[1:]
            else:
                self.arguments = args

    def __str__(self):
        return "<Instr {0}>".format(self.operator_name)

    def __eq__(self, other):
        return self.out() == other.out()

    def __hash__(self):
        return hash(self.out())

    def synthesize(self, resource_manager=None):
        if resource_manager is not None:
            self.make_qubits_known(resource_manager)
        return [self]

    def out(self):
        def format_params(params):
            if not params:
                return ""
            else:
                return "(" + ",".join(map(format_parameter, params)) + ")"

        def format_args(args):
            if 0 == len(args):
                return ""
            else:
                return " " + " ".join([str(arg) for arg in args])

        if self.parameters:
            return self.operator_name + format_params(self.parameters) + format_args(self.arguments)
        else:
            return self.operator_name + format_args(self.arguments)

    def make_qubits_known(self, rm):
        """
        Make the qubits involved with this instruction known to a ResourceManager.

        :param ResourceManager rm: A ResourceManager object.
        """
        if not isinstance(rm, ResourceManager):
            raise TypeError("rm should be a ResourceManager")

        for arg in self.arguments:
            if isinstance(arg, DirectQubit):
                current = rm.in_use.get(arg.index(), False)
                rm.in_use[arg.index()] = arg
                # re-instantiate the qubit
                if current and isinstance(current, Qubit):
                    rm.instantiate(current)

    def qubits(self):
        """
        The qubits this instruction affects.

        :return: Set of qubit indexes.
        :rtype: set
        """
        qubits = set()
        for arg in self.arguments:
            if issubinstance(arg, AbstractQubit):
                qubits.add(arg.index())
        return qubits


class Gate(Instr):
    """
    This is the pyQuil object for a quantum gate instruction.
    """
    def __init__(self, name, params, qubits):
        for qubit in qubits:
            check_live_qubit(qubit)
        super(Gate, self).__init__(name, params, qubits)

    def __repr__(self):
        return "<Gate: " + self.out() + ">"

    def __str__(self):
        return self.out()


class Measurement(Instr):
    """
    This is the pyQuil object for a Quil measurement instruction.
    """
    def __init__(self, qubit, classical_reg=None):
        check_live_qubit(qubit)

        if classical_reg is None:
            args = (qubit,)
        elif isinstance(classical_reg, Addr):
            args = (qubit, classical_reg)
        else:
            raise TypeError("classical_reg should be None or an Addr instance")

        super(Measurement, self).__init__("MEASURE",
                                          params=None,
                                          args=args)

        self.classical_reg = classical_reg


class While(QuilAction):
    """
    Representation of a ``while`` construct. To use, initialize with an
    address to branch on, and use ``self.Body.inst()`` to add instructions to
    the body of the loop.
    """

    def __init__(self, condition):
        if not isinstance(condition, Addr):
            raise TypeError("condition must be an Addr")
        super(While, self).__init__()
        self.condition = condition
        self.Body = InstructionGroup()

    def synthesize(self, resource_manager=None):
        # WHILE [c]:
        #    instr...
        #
        # =>
        #
        #   LABEL @START
        #   JUMP-UNLESS @END [c]
        #   instr...
        #   JUMP @START
        #   LABEL @END
        label_start = gen_label("START")
        label_end = gen_label("END")
        insts = list()
        insts.append(JumpTarget(label_start))
        insts.append(JumpUnless(target=label_end, condition=self.condition))
        insts.extend(self.Body.synthesize())
        insts.append(Jump(target=label_start))
        insts.append(JumpTarget(label_end))
        return insts


class If(QuilAction):
    """
    Representation of an ``if`` construct. To use, initialize with an address
    to be branched on, and add instructions to ``self.Then`` and ``self.Else``
    for the corresponding branches.
    """

    def __init__(self, condition):
        if not isinstance(condition, Addr):
            raise TypeError("condition must be an Addr")
        super(If, self).__init__()
        self.condition = condition
        self.Then = InstructionGroup()
        self.Else = InstructionGroup()

    def synthesize(self, resource_manager=None):
        # IF [c]:
        #    instrA...
        # ELSE:
        #    instrB...
        #
        # =>
        #
        #   JUMP-WHEN @THEN [c]
        #   instrB...
        #   JUMP @END
        #   LABEL @THEN
        #   instrA...
        #   LABEL @END
        label_then = gen_label("THEN")
        label_end = gen_label("END")
        insts = list()
        insts.append(JumpWhen(target=label_then, condition=self.condition))
        insts.extend(self.Else.synthesize())
        insts.append(Jump(target=label_end))
        insts.append(JumpTarget(label_then))
        insts.extend(self.Then.synthesize())
        insts.append(JumpTarget(label_end))
        return insts
