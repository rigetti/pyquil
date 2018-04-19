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
from six import integer_types, string_types

from pyquil.parameters import Expression, _contained_parameters, format_parameter
from pyquil.quilatom import Qubit, Addr, Label, unpack_qubit, QubitPlaceholder, LabelPlaceholder


class AbstractInstruction(object):
    """
    Abstract class for representing single instructions.
    """

    def out(self):
        pass

    def __str__(self):
        return self.out()

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.out() == other.out()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.out())


RESERVED_WORDS = ['DEFGATE', 'DEFCIRCUIT', 'MEASURE',
                  'LABEL', 'HALT', 'JUMP', 'JUMP-WHEN', 'JUMP-UNLESS',
                  'RESET', 'WAIT', 'NOP', 'INCLUDE', 'PRAGMA',
                  'FALSE', 'TRUE', 'NOT', 'AND', 'OR', 'MOVE', 'EXCHANGE']


def _extract_qubit_index(qubit, index=True):
    if (not index) or isinstance(qubit, QubitPlaceholder):
        return qubit
    return qubit.index


class Gate(AbstractInstruction):
    """
    This is the pyQuil object for a quantum gate instruction.
    """

    def __init__(self, name, params, qubits):
        if not isinstance(name, string_types):
            raise TypeError("Gate name must be a string")

        if name in RESERVED_WORDS:
            raise ValueError("Cannot use {} for a gate name since it's a reserved word".format(name))

        if not isinstance(params, list):
            raise TypeError("Gate params must be a list")

        if not isinstance(qubits, list) or not qubits:
            raise TypeError("Gate arguments must be a non-empty list")
        for qubit in qubits:
            if not isinstance(qubit, (Qubit, QubitPlaceholder)):
                raise TypeError("Gate arguments must all be Qubits")

        self.name = name
        self.params = params
        self.qubits = qubits

    def get_qubits(self, indices=True):
        return {_extract_qubit_index(q, indices) for q in self.qubits}

    def out(self):
        def format_params(params):
            return "(" + ",".join(map(format_parameter, params)) + ")"

        def format_qubits(qubits):
            return " ".join([qubit.out() for qubit in qubits])

        if self.params:
            return "{}{} {}".format(self.name, format_params(self.params), format_qubits(self.qubits))
        else:
            return "{} {}".format(self.name, format_qubits(self.qubits))

    def __repr__(self):
        return "<Gate " + self.out() + ">"


class Measurement(AbstractInstruction):
    """
    This is the pyQuil object for a Quil measurement instruction.
    """

    def __init__(self, qubit, classical_reg=None):
        if not isinstance(qubit, (Qubit, QubitPlaceholder)):
            raise TypeError("qubit should be a Qubit")
        if classical_reg and not isinstance(classical_reg, Addr):
            raise TypeError("classical_reg should be None or an Addr instance")

        self.qubit = qubit
        self.classical_reg = classical_reg

    def out(self):
        if self.classical_reg:
            return "MEASURE {} {}".format(self.qubit, self.classical_reg)
        else:
            return "MEASURE {}".format(self.qubit)

    def get_qubits(self, indices=True):
        return {_extract_qubit_index(self.qubit, indices)}


class DefGate(AbstractInstruction):
    """
    A DEFGATE directive.

    :param string name: The name of the newly defined gate.
    :param array-like matrix: {list, nparray, np.matrix} The matrix defining this gate.
    :param list parameters: list of parameters that are used in this gate
    """

    def __init__(self, name, matrix, parameters=None):
        if not isinstance(name, string_types):
            raise TypeError("Gate name must be a string")

        if name in RESERVED_WORDS:
            raise ValueError("Cannot use {} for a gate name since it's a reserved word".format(name))

        if isinstance(matrix, list):
            rows = len(matrix)
            if not all([len(row) == rows for row in matrix]):
                raise ValueError("Matrix must be square.")
        elif isinstance(matrix, (np.ndarray, np.matrix)):
            rows, cols = matrix.shape
            if rows != cols:
                raise ValueError("Matrix must be square.")
        else:
            raise TypeError("Matrix argument must be a list or NumPy array/matrix")

        if 0 != rows & (rows - 1):
            raise ValueError("Dimension of matrix must be a power of 2, got {0}".format(rows))
        self.name = name
        self.matrix = np.asarray(matrix)

        if parameters:
            if not isinstance(parameters, list):
                raise TypeError("Paramaters must be a list")

            expressions = [elem for row in self.matrix for elem in row if isinstance(elem, Expression)]
            used_params = {param for exp in expressions for param in _contained_parameters(exp)}

            if set(parameters) != used_params:
                raise ValueError("Parameters list does not match parameters actually used in gate matrix:\n"
                                 "Parameters in argument: {}, Parameters in matrix: {}".format(parameters, used_params))
        else:
            is_unitary = np.allclose(np.eye(rows), self.matrix.dot(self.matrix.T.conj()))
            if not is_unitary:
                raise ValueError("Matrix must be unitary.")

        self.parameters = parameters

    def out(self):
        """
        Prints a readable Quil string representation of this gate.

        :returns: String representation of a gate
        :rtype: string
        """
        def format_matrix_element(element):
            """
            Formats a parameterized matrix element.

            :param element: {int, float, complex, str} The parameterized element to format.
            """
            if isinstance(element, integer_types) or isinstance(element, (float, complex, np.int_)):
                return format_parameter(element)
            elif isinstance(element, string_types):
                return element
            elif isinstance(element, Expression):
                return str(element)
            else:
                raise TypeError("Invalid matrix element: %r" % element)

        if self.parameters:
            result = "DEFGATE {}({}):\n".format(self.name, ', '.join(map(str, self.parameters)))
        else:
            result = "DEFGATE {}:\n".format(self.name)

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
        if self.parameters:
            return lambda *params: lambda *qubits: \
                Gate(name=self.name, params=list(params), qubits=list(map(unpack_qubit, qubits)))
        else:
            return lambda *qubits: Gate(name=self.name, params=[], qubits=list(map(unpack_qubit, qubits)))

    def num_args(self):
        """
        :return: The number of qubit arguments the gate takes.
        :rtype: int
        """
        rows = len(self.matrix)
        return int(np.log2(rows))


class JumpTarget(AbstractInstruction):
    """
    Representation of a target that can be jumped to.
    """

    def __init__(self, label):
        if not isinstance(label, (Label, LabelPlaceholder)):
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
        if not isinstance(target, (Label, LabelPlaceholder)):
            raise TypeError("target should be a Label")
        if not isinstance(condition, Addr):
            raise TypeError("condition should be an Addr")
        self.target = target
        self.condition = condition

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
        if not isinstance(target, (Label, LabelPlaceholder)):
            raise TypeError("target should be a Label")
        self.target = target

    def out(self):
        return "JUMP %s" % self.target


class Pragma(AbstractInstruction):
    """
    A PRAGMA instruction.

    This is printed in QUIL as::

        PRAGMA <command> <arg1> <arg2> ... <argn> "<freeform_string>"

    """

    def __init__(self, command, args=(), freeform_string=""):
        if not isinstance(command, string_types):
            raise TypeError("Pragma's require an identifier.")

        if not isinstance(args, (tuple, list)):
            raise TypeError("Pragma arguments must be a list: {}".format(args))
        for a in args:
            if not (isinstance(a, string_types) or isinstance(a, integer_types)):
                raise TypeError("Pragma arguments must be strings or integers: {}".format(a))

        if not isinstance(freeform_string, string_types):
            raise TypeError("The freeform string argument must be a string: {}".format(
                freeform_string))

        self.command = command
        self.args = args
        self.freeform_string = freeform_string

    def out(self):
        ret = "PRAGMA {}".format(self.command)
        if self.args:
            ret += " {}".format(" ".join(str(a) for a in self.args))
        if self.freeform_string:
            ret += " \"{}\"".format(self.freeform_string)
        return ret

    def __repr__(self):
        return '<PRAGMA {}>'.format(self.command)


class RawInstr(AbstractInstruction):
    """
    A raw instruction represented as a string.
    """

    def __init__(self, instr_str):
        if not isinstance(instr_str, string_types):
            raise TypeError("Raw instructions require a string.")
        self.instr = instr_str

    def out(self):
        return self.instr

    def __repr__(self):
        return '<RawInstr>'
