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
Contains the core pyQuil objects that correspond to Quil instructions.
"""
import numpy as np
from six import integer_types, string_types
from warnings import warn

from pyquil.parameters import Expression, _contained_parameters, format_parameter
from pyquil.quilatom import (Qubit, MemoryReference, Label, unpack_qubit, QubitPlaceholder,
                             LabelPlaceholder)


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
                  'DECLARE',
                  'NEG', 'NOT', 'AND', 'IOR', 'XOR',
                  'MOVE', 'EXCHANGE', 'CONVERT',
                  'ADD', 'SUB', 'MUL', 'DIV',
                  'EQ', 'GT', 'GE', 'LT', 'LE',
                  'LOAD', 'STORE',
                  # to be removed:
                  'TRUE', 'FALSE', 'OR'
                  ]


def _extract_qubit_index(qubit, index=True):
    if (not index) or isinstance(qubit, QubitPlaceholder):
        return qubit
    return qubit.index


def _format_qubit_str(qubit):
    if isinstance(qubit, QubitPlaceholder):
        return "{%s}" % str(qubit)
    return str(qubit)


def _format_qubits_str(qubits):
    return " ".join([_format_qubit_str(qubit) for qubit in qubits])


def _format_qubits_out(qubits):
    return " ".join([qubit.out() for qubit in qubits])


def _format_params(params):
    return "(" + ",".join(format_parameter(param) for param in params) + ")"


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
        if self.params:
            return "{}{} {}".format(self.name, _format_params(self.params),
                                    _format_qubits_out(self.qubits))
        else:
            return "{} {}".format(self.name, _format_qubits_out(self.qubits))

    def __repr__(self):
        return "<Gate " + str(self) + ">"

    def __str__(self):
        if self.params:
            return "{}{} {}".format(self.name, _format_params(self.params),
                                    _format_qubits_str(self.qubits))
        else:
            return "{} {}".format(self.name, _format_qubits_str(self.qubits))


class Measurement(AbstractInstruction):
    """
    This is the pyQuil object for a Quil measurement instruction.
    """

    def __init__(self, qubit, classical_reg=None):
        if not isinstance(qubit, (Qubit, QubitPlaceholder)):
            raise TypeError("qubit should be a Qubit")
        if classical_reg and not isinstance(classical_reg, MemoryReference):
            raise TypeError("classical_reg should be None or a MemoryReference instance")

        self.qubit = qubit
        self.classical_reg = classical_reg

    def out(self):
        if self.classical_reg:
            return "MEASURE {} {}".format(self.qubit.out(), self.classical_reg.out())
        else:
            return "MEASURE {}".format(self.qubit.out())

    def __str__(self):
        if self.classical_reg:
            return "MEASURE {} {}".format(_format_qubit_str(self.qubit), str(self.classical_reg))
        else:
            return "MEASURE {}".format(_format_qubit_str(self.qubit))

    def get_qubits(self, indices=True):
        return {_extract_qubit_index(self.qubit, indices)}


class ResetQubit(AbstractInstruction):
    """
    This is the pyQuil object for a Quil targeted reset instruction.
    """

    def __init__(self, qubit):
        if not isinstance(qubit, (Qubit, QubitPlaceholder)):
            raise TypeError("qubit should be a Qubit")
        self.qubit = qubit

    def out(self):
        return "RESET {}".format(self.qubit.out())

    def __str__(self):
        return "RESET {}".format(_format_qubit_str(self.qubit))

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
    op = NotImplemented

    def __init__(self, target, condition):
        if not isinstance(target, (Label, LabelPlaceholder)):
            raise TypeError("target should be a Label")
        if not isinstance(condition, MemoryReference):
            raise TypeError("condition should be an MemoryReference")
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
    The NOP instruction.
    """
    op = "NOP"


class UnaryClassicalInstruction(AbstractInstruction):
    """
    The abstract class for unary classical instructions.
    """

    def __init__(self, target):
        if not isinstance(target, MemoryReference):
            raise TypeError("target operand should be an MemoryReference")
        self.target = target

    def out(self):
        return "%s %s" % (self.op, self.target)


class ClassicalNeg(UnaryClassicalInstruction):
    """
    The NEG instruction.
    """
    op = "NEG"


class ClassicalNot(UnaryClassicalInstruction):
    """
    The NOT instruction.
    """
    op = "NOT"


class LogicalBinaryOp(AbstractInstruction):
    """
    The abstract class for binary logical classical instructions.
    """
    op = NotImplemented

    def __init__(self, left, right):
        if not isinstance(left, MemoryReference):
            raise TypeError("left operand should be an MemoryReference")
        if not isinstance(right, MemoryReference) and not isinstance(right, int):
            raise TypeError("right operand should be an MemoryReference or an Int")
        self.left = left
        self.right = right

    def out(self):
        return "%s %s %s" % (self.op, self.left, self.right)


class ClassicalAnd(LogicalBinaryOp):
    """
    WARNING: The operand order for ClassicalAnd has changed.  In pyQuil versions <= 1.9, AND had signature

        AND %source %target

    Now, AND has signature

        AND %target %source
    """

    op = "AND"


class ClassicalInclusiveOr(LogicalBinaryOp):
    """
    The IOR instruction.
    """
    op = "IOR"


class ClassicalExclusiveOr(LogicalBinaryOp):
    """
    The XOR instruction.
    """
    op = "XOR"


class ClassicalOr(ClassicalInclusiveOr):
    """
    Deprecated class.
    """

    def __init__(self, left, right):
        warn("ClassicalOr has been deprecated. Replacing with "
             "ClassicalInclusiveOr. Use ClassicalInclusiveOr instead. "
             "NOTE: The operands to ClassicalInclusiveOr are inverted from "
             "ClassicalOr.")
        super().__init__(right, left)


class ArithmeticBinaryOp(AbstractInstruction):
    """
    The abstract class for binary arithmetic classical instructions.
    """

    def __init__(self, left, right):
        if not isinstance(left, MemoryReference):
            raise TypeError("left operand should be an MemoryReference")
        if not isinstance(right, MemoryReference) and not isinstance(right, int) and not isinstance(right, float):
            raise TypeError("right operand should be an MemoryReference or a numeric literal")
        self.left = left
        self.right = right

    def out(self):
        return "%s %s %s" % (self.op, self.left, self.right)


class ClassicalAdd(ArithmeticBinaryOp):
    """
    The ADD instruction.
    """
    op = "ADD"


class ClassicalSub(ArithmeticBinaryOp):
    """
    The SUB instruction.
    """
    op = "SUB"


class ClassicalMul(ArithmeticBinaryOp):
    """
    The MUL instruction.
    """
    op = "MUL"


class ClassicalDiv(ArithmeticBinaryOp):
    """
    The DIV instruction.
    """
    op = "DIV"


class ClassicalMove(AbstractInstruction):
    """
    The MOVE instruction.

    WARNING: In pyQuil 2.0, the order of operands is as MOVE <target> <source>.
             In pyQuil 1.9, the order of operands was MOVE <source> <target>.
             These have reversed.
    """
    op = "MOVE"

    def __init__(self, left, right):
        if not isinstance(left, MemoryReference):
            raise TypeError("Left operand of MOVE should be an MemoryReference.  "
                            "Note that the order of the operands in pyQuil 2.0 has reversed from "
                            "the order of pyQuil 1.9 .")
        if not isinstance(right, MemoryReference) and not isinstance(right, int) and not isinstance(right, float):
            raise TypeError("Right operand of MOVE should be an MemoryReference "
                            "or a numeric literal")
        self.left = left
        self.right = right

    def out(self):
        return "%s %s %s" % (self.op, self.left, self.right)


class ClassicalFalse(ClassicalMove):
    """
    Deprecated class.
    """

    def __init__(self, target):
        super().__init__(target, 0)
        warn("ClassicalFalse is deprecated in favor of ClassicalMove.")


class ClassicalTrue(ClassicalMove):
    """
    Deprecated class.
    """

    def __init__(self, target):
        super().__init__(target, 1)
        warn("ClassicalTrue is deprecated in favor of ClassicalMove.")


class ClassicalExchange(AbstractInstruction):
    """
    The EXCHANGE instruction.
    """

    op = "EXCHANGE"

    def __init__(self, left, right):
        if not isinstance(left, MemoryReference):
            raise TypeError("left operand should be an MemoryReference")
        if not isinstance(right, MemoryReference):
            raise TypeError("right operand should be an MemoryReference")
        self.left = left
        self.right = right

    def out(self):
        return "%s %s %s" % (self.op, self.left, self.right)


class ClassicalConvert(AbstractInstruction):
    """
    The CONVERT instruction.
    """

    op = "CONVERT"

    def __init__(self, left, right):
        if not isinstance(left, MemoryReference):
            raise TypeError("left operand should be an MemoryReference")
        if not isinstance(right, MemoryReference):
            raise TypeError("right operand should be an MemoryReference")
        self.left = left
        self.right = right

    def out(self):
        return "%s %s %s" % (self.op, self.left, self.right)


class ClassicalLoad(AbstractInstruction):
    """
    The LOAD instruction.
    """

    op = "LOAD"

    def __init__(self, target, left, right):
        if not isinstance(target, MemoryReference):
            raise TypeError("target operand should be an MemoryReference")
        if not isinstance(right, MemoryReference):
            raise TypeError("right operand should be an MemoryReference")
        self.target = target
        self.left = left
        self.right = right

    def out(self):
        return "%s %s %s %s" % (self.op, self.target, self.left, self.right)


class ClassicalStore(AbstractInstruction):
    """
    The STORE instruction.
    """

    op = "STORE"

    def __init__(self, target, left, right):
        if not isinstance(left, MemoryReference):
            raise TypeError("left operand should be an MemoryReference")
        if not isinstance(right, MemoryReference):
            raise TypeError("right operand should be an MemoryReference")
        self.target = target
        self.left = left
        self.right = right

    def out(self):
        return "%s %s %s %s" % (self.op, self.target, self.left, self.right)


class ClassicalComparison(AbstractInstruction):
    """
    Abstract class for ternary comparison instructions.
    """

    def __init__(self, target, left, right):
        if not isinstance(target, MemoryReference):
            raise TypeError("target operand should be an MemoryReference")
        if not isinstance(left, MemoryReference):
            raise TypeError("left operand should be an MemoryReference")
        self.target = target
        self.left = left
        self.right = right

    def out(self):
        return "%s %s %s %s" % (self.op, self.target, self.left, self.right)


class ClassicalEqual(ClassicalComparison):
    """
    The EQ comparison instruction.
    """

    op = "EQ"


class ClassicalLessThan(ClassicalComparison):
    """
    The LT comparison instruction.
    """

    op = "LT"


class ClassicalLessEqual(ClassicalComparison):
    """
    The LE comparison instruction.
    """

    op = "LE"


class ClassicalGreaterThan(ClassicalComparison):
    """
    The GT comparison instruction.
    """

    op = "GT"


class ClassicalGreaterEqual(ClassicalComparison):
    """
    The GE comparison instruction.
    """

    op = "GE"


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
            if not (isinstance(a, string_types)
                    or isinstance(a, integer_types)
                    or isinstance(a, QubitPlaceholder)
                    or isinstance(a, Qubit)):
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


class Declare(AbstractInstruction):
    """
    A DECLARE directive.

    This is printed in Quil as::

        DECLARE <name> <memory-type> (SHARING <other-name> (OFFSET <amount> <type>)* )?

    """

    def __init__(self, name, memory_type, memory_size=1, shared_region=None, offsets=None):
        self.name = name
        self.memory_type = memory_type
        self.memory_size = memory_size
        self.shared_region = shared_region

        if offsets is None:
            offsets = []
        self.offsets = offsets

    def asdict(self):
        return {
            'name': self.name,
            'memory_type': self.memory_type,
            'memory_size': self.memory_size,
            'shared_region': self.shared_region,
            'offsets': self.offsets,
        }

    def out(self):
        ret = "DECLARE {} {}[{}]".format(self.name, self.memory_type, self.memory_size)
        if self.shared_region:
            ret += " SHARING {}".format(self.shared_region)
            for offset in self.offsets:
                ret += " OFFSET {} {}".format(offset[0], offset[1])

        return ret

    def __repr__(self):
        return '<DECLARE {}>'.format(self.name)


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
        return '<RawInstr {}>'.format(self.instr)
