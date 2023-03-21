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
import abc
import collections
import json

from numbers import Complex
from typing import (
    Any,
    Callable,
    ClassVar,
    Container,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
    cast,
)

from deprecation import deprecated
import numpy as np

from pyquil._version import pyquil_version
from pyquil.quilatom import (
    Expression,
    ExpressionDesignator,
    Label,
    LabelPlaceholder,
    MemoryReference,
    Parameter,
    ParameterDesignator,
    Frame,
    Waveform,
    Qubit,
    QubitDesignator,
    QubitPlaceholder,
    FormalArgument,
    _contained_parameters,
    _convert_to_py_qubit,
    _convert_to_py_qubits,
    _convert_to_rs_expression,
    _convert_to_rs_expressions,
    _convert_to_rs_qubit,
    _convert_to_rs_qubits,
    _convert_to_py_parameter,
    _convert_to_py_parameters,
    format_parameter,
    unpack_qubit,
    _complex_str,
)

if TYPE_CHECKING:
    from pyquil.paulis import PauliSum

from dataclasses import dataclass

import qcs_sdk.quil.instructions as quil_rs
import qcs_sdk.quil.expression as quil_rs_expr


class _InstructionMeta(abc.ABCMeta):
    """
    A metaclass that allows us to group all instruction types from quil-rs and pyQuil as an `AbstractInstruction`.
    As such, this should _only_ be used as a metaclass for `AbstractInstruction`.
    """

    def __init__(self, *args, **_):
        self.__name = args[0]
        try:
            self.__is_abstract_instruction = args[1][0] == AbstractInstruction
        except Exception:
            self.__is_abstract_instruction = False

    def __instancecheck__(self, __instance: Any) -> bool:
        # Already an Instruction, return True
        if isinstance(__instance, quil_rs.Instruction):
            return True

        # __instance is not an Instruction or AbstractInstruction, return False
        if not self.__name == "AbstractInstruction" and not self.__is_abstract_instruction:
            return False

        # __instance is a subclass of AbstractInstruction, do the normal check
        return super().__instancecheck__(__instance)


class AbstractInstruction(metaclass=_InstructionMeta):
    """
    Abstract class for representing single instructions.
    """

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(str(self))


def _convert_to_rs_instruction(instr: AbstractInstruction) -> quil_rs.Instruction:
    if isinstance(instr, quil_rs.Instruction):
        return instr
    if isinstance(instr, AbstractInstruction):
        return quil_rs.Instruction(instr)
    if isinstance(instr, quil_rs.Calibration):
        return quil_rs.Instruction.from_calibration_definition(instr)
    if isinstance(instr, quil_rs.Gate):
        return quil_rs.Instruction.from_gate(instr)
    if isinstance(instr, quil_rs.MeasureCalibrationDefinition):
        return quil_rs.Instruction.from_measure_calibration_definition(instr)
    if isinstance(instr, quil_rs.Measurement):
        return quil_rs.Instruction.from_measurement(instr)
    else:
        raise ValueError(f"{type(instr)} is not an Instruction")


def _convert_to_rs_instructions(instrs: Iterable[AbstractInstruction]) -> List[quil_rs.Instruction]:
    return [_convert_to_rs_instruction(instr) for instr in instrs]


def _convert_to_py_instruction(instr: quil_rs.Instruction) -> AbstractInstruction:
    if isinstance(instr, quil_rs.Instruction):
        # TODOV4: Will have to handle unit variants since they don't have inner data
        instr = instr.inner()
    if isinstance(instr, quil_rs.Declaration):
        return Declare._from_rs_declaration(instr)
    if isinstance(instr, quil_rs.Calibration):
        return DefCalibration._from_rs_calibration(instr)
    if isinstance(instr, quil_rs.Gate):
        return Gate._from_rs_gate(instr)
    if isinstance(instr, quil_rs.MeasureCalibrationDefinition):
        return DefMeasureCalibration._from_rs_measure_calibration_definition(instr)
    if isinstance(instr, quil_rs.Measurement):
        return Measurement._from_rs_measurement(instr)
    if isinstance(instr, quil_rs.Instruction):
        raise NotImplementedError(f"The {type(instr)} Instruction hasn't been mapped to an AbstractInstruction yet.")
    raise ValueError(f"{type(instr)} is not a valid Instruction type")


def _convert_to_py_instructions(instrs: Iterable[quil_rs.Instruction]) -> List[AbstractInstruction]:
    return [_convert_to_py_instruction(instr) for instr in instrs]


RESERVED_WORDS: Container[str] = [
    "DEFGATE",
    "DEFCIRCUIT",
    "MEASURE",
    "LABEL",
    "HALT",
    "JUMP",
    "JUMP-WHEN",
    "JUMP-UNLESS",
    "RESET",
    "WAIT",
    "NOP",
    "INCLUDE",
    "PRAGMA",
    "DECLARE",
    "NEG",
    "NOT",
    "AND",
    "IOR",
    "XOR",
    "MOVE",
    "EXCHANGE",
    "CONVERT",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "EQ",
    "GT",
    "GE",
    "LT",
    "LE",
    "LOAD",
    "STORE",
    # Quil-T additions:
    "DEFCAL",
    "DEFFRAME",
    "DEFWAVEFORM",
    "PULSE",
    "CAPTURE",
    "RAW-CAPTURE",
    "DELAY",
    "FENCE",
    "SET-FREQUENCY",
    "SET-PHASE",
    "SHIFT-PHASE",
    "SWAP-PHASES",
    "SET-SCALE",
    "SAMPLE-RATE",
    "INITIAL-FREQUENCY",
    # to be removed:
    "TRUE",
    "FALSE",
    "OR",
]


def _extract_qubit_index(qubit: Union[Qubit, QubitPlaceholder, FormalArgument], index: bool = True) -> QubitDesignator:
    if index and isinstance(qubit, Qubit):
        return qubit.index
    return qubit


def _get_frame_qubits(frame: Frame, index: bool = True) -> Set[QubitDesignator]:
    for q in frame.qubits:
        if isinstance(q, FormalArgument):
            raise ValueError("Attempted to extract FormalArgument where a Qubit is expected.")
    return {_extract_qubit_index(q, index) for q in cast(List[Qubit], frame.qubits)}


def _format_qubit_str(qubit: Union[Qubit, QubitPlaceholder, FormalArgument]) -> str:
    if isinstance(qubit, QubitPlaceholder):
        return "{%s}" % str(qubit)
    return str(qubit)


def _format_qubits_str(qubits: Iterable[Union[Qubit, QubitPlaceholder, FormalArgument]]) -> str:
    return " ".join([_format_qubit_str(qubit) for qubit in qubits])


def _format_qubits_out(qubits: Iterable[Union[Qubit, QubitPlaceholder, FormalArgument]]) -> str:
    return " ".join([qubit.out() for qubit in qubits])


def _format_params(params: Iterable[ParameterDesignator]) -> str:
    return "(" + ",".join(format_parameter(param) for param in params) + ")"


def _join_strings(*args: str) -> str:
    return " ".join(map(str, args))


class Gate(quil_rs.Gate, AbstractInstruction):
    """
    This is the pyQuil object for a quantum gate instruction.
    """

    def __new__(
        cls,
        name: str,
        params: Iterable[ParameterDesignator],
        qubits: Iterable[Union[Qubit, QubitPlaceholder, FormalArgument]],
        modifiers: Iterable[quil_rs.GateModifier] = [],
    ) -> "Gate":
        return super().__new__(cls, name, _convert_to_rs_expressions(params), _convert_to_rs_qubits(qubits), modifiers)

    @classmethod
    def _from_rs_gate(cls, gate: quil_rs.Gate) -> "Gate":
        return cls(gate.name, gate.parameters, gate.qubits, gate.modifiers)

    @deprecated(
        deprecated_in="4.0",
        removed_in="5.0",
        current_version=pyquil_version,
        details="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Set[QubitDesignator]:
        if indices:
            return self.get_qubit_indices()
        else:
            return set(_convert_to_py_qubits(super().qubits))

    @property
    def qubits(self):
        return list(self.get_qubits(indices=False))

    @qubits.setter
    def qubits(self, qubits: Iterable[Union[Qubit, QubitPlaceholder, FormalArgument]]):
        quil_rs.Gate.qubits.__set__(self, _convert_to_rs_qubits(qubits))

    @property
    def params(self):
        return _convert_to_py_parameters(super().parameters)

    @params.setter
    def params(self, params: Iterable[ParameterDesignator]):
        quil_rs.Gate.parameters.__set__(self, _convert_to_rs_expressions(params))

    def get_qubit_indices(self) -> Set[int]:
        return {qubit.as_fixed() for qubit in super().qubits}

    def controlled(self, control_qubit: Union[QubitDesignator, Sequence[QubitDesignator]]) -> "Gate":
        """
        Add the CONTROLLED modifier to the gate with the given control qubit or Sequence of control
        qubits.
        """
        if isinstance(control_qubit, Sequence):
            for qubit in control_qubit:
                self = super().controlled(_convert_to_rs_qubit(qubit))
        else:
            self = super().controlled(_convert_to_rs_qubit(control_qubit))

        return self

    def forked(self, fork_qubit: QubitDesignator, alt_params: List[ParameterDesignator]) -> "Gate":
        """
        Add the FORKED modifier to the gate with the given fork qubit and given additional
        parameters.
        """
        self = super().forked(_convert_to_rs_qubit(fork_qubit), _convert_to_rs_expressions(alt_params))
        return self

    def dagger(self) -> "Gate":
        """
        Add the DAGGER modifier to the gate.
        """
        self = super().dagger()
        return self

    def out(self) -> str:
        return str(self)


def _strip_modifiers(gate: Gate, limit: Optional[int] = None) -> Gate:
    """
    Remove modifiers from :py:class:`Gate`.

    This function removes up to ``limit`` gate modifiers from the given gate,
    starting from the leftmost gate modifier.

    :param gate: A gate.
    :param limit: An upper bound on how many modifiers to remove.
    """
    if limit is None:
        limit = len(gate.modifiers)

    # We walk the modifiers from left-to-right, tracking indices to identify
    # qubits/params introduced by gate modifiers.
    #
    # Invariants:
    #   - gate.qubits[0:qubit_index] are qubits introduced by gate modifiers
    #   - gate.params[param_index:] are parameters introduced by gate modifiers
    qubit_index = 0
    param_index = len(gate.params)
    for m in gate.modifiers[:limit]:
        if m == "CONTROLLED":
            qubit_index += 1
        elif m == "FORKED":
            if param_index % 2 != 0:
                raise ValueError("FORKED gate has an invalid number of parameters.")
            param_index //= 2
            qubit_index += 1
        elif m == "DAGGER":
            pass
        else:
            raise TypeError("Unsupported gate modifier {}".format(m))

    stripped = Gate(gate.name, gate.params[:param_index], gate.qubits[qubit_index:])
    stripped.modifiers = gate.modifiers[limit:]
    return stripped


class Measurement(quil_rs.Measurement, AbstractInstruction):
    """
    This is the pyQuil object for a Quil measurement instruction.
    """

    def __new__(
        cls,
        qubit: QubitDesignator,
        classical_reg: Optional[MemoryReference],
    ):
        classical_reg = cls._reg_to_target(classical_reg)
        return super().__new__(cls, _convert_to_rs_qubit(qubit), classical_reg)

    @classmethod
    def _reg_to_target(cls, classical_reg: Optional[MemoryReference]) -> Optional[quil_rs.MemoryReference]:
        if isinstance(classical_reg, quil_rs.MemoryReference):
            return classical_reg

        if classical_reg is not None:
            try:
                classical_reg = _convert_to_rs_expression(classical_reg).to_address()
            except ValueError:
                raise TypeError(f"classical_reg should be None or a MemoryReference instance")

        return classical_reg

    @classmethod
    def _from_rs_measurement(cls, measurement: quil_rs.Measurement):
        return cls(measurement.qubit, measurement.target)

    @property
    def qubit(self) -> QubitDesignator:
        return _convert_to_py_qubit(super().qubit)

    @qubit.setter
    def qubit(self, qubit):
        quil_rs.Measurement.qubit.__set__(self, _convert_to_rs_qubit(qubit))

    @property
    def classical_reg(self) -> Optional[MemoryReference]:
        target = super().target
        if target is None:
            return None
        return MemoryReference._from_rs_memory_reference(target)  # type: ignore

    @classical_reg.setter
    def classical_reg(self, classical_reg: Optional[MemoryReference]):
        classical_reg = self._reg_to_target(classical_reg)
        quil_rs.Measurement.target.__set__(self, classical_reg)

    @deprecated(
        deprecated_in="4.0",
        removed_in="5.0",
        current_version=pyquil_version,
        details="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Set[QubitDesignator]:
        if indices:
            return self.get_qubit_indices()
        else:
            return {_convert_to_py_qubit(super().qubit)}

    def get_qubit_indices(self) -> Set[int]:
        return {super().qubit.as_fixed()}

    def out(self) -> str:
        return str(self)


class ResetQubit(AbstractInstruction):
    """
    This is the pyQuil object for a Quil targeted reset instruction.
    """

    def __init__(self, qubit: Union[Qubit, QubitPlaceholder, FormalArgument]):
        if not isinstance(qubit, (Qubit, QubitPlaceholder, FormalArgument)):
            raise TypeError("qubit should be a Qubit")
        self.qubit = qubit

    def out(self) -> str:
        return "RESET {}".format(self.qubit.out())

    def __str__(self) -> str:
        return "RESET {}".format(_format_qubit_str(self.qubit))

    def get_qubits(self, indices: bool = True) -> Set[QubitDesignator]:
        return {_extract_qubit_index(self.qubit, indices)}


class DefGate(AbstractInstruction):
    """
    A DEFGATE directive.

    :param name: The name of the newly defined gate.
    :param matrix: The matrix defining this gate.
    :param parameters: list of parameters that are used in this gate
    """

    def __init__(
        self,
        name: str,
        matrix: Union[List[List[Any]], np.ndarray, np.matrix],
        parameters: Optional[List[Parameter]] = None,
    ):
        if not isinstance(name, str):
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
                raise ValueError(
                    "Parameters list does not match parameters actually used in gate matrix:\n"
                    "Parameters in argument: {}, Parameters in matrix: {}".format(parameters, used_params)
                )
        else:
            is_unitary = np.allclose(np.eye(rows), self.matrix.dot(self.matrix.T.conj()))
            if not is_unitary:
                raise ValueError("Matrix must be unitary.")

        self.parameters = parameters

    def out(self) -> str:
        """
        Prints a readable Quil string representation of this gate.

        :returns: String representation of a gate
        """

        def format_matrix_element(element: Union[ExpressionDesignator, str]) -> str:
            """
            Formats a parameterized matrix element.

            :param element: The parameterized element to format.
            """
            if isinstance(element, (int, float, complex, np.int_)):
                return format_parameter(element)
            elif isinstance(element, str):
                return element
            elif isinstance(element, Expression):
                return str(element)
            else:
                raise TypeError("Invalid matrix element: %r" % element)

        if self.parameters:
            result = "DEFGATE {}({}):\n".format(self.name, ", ".join(map(str, self.parameters)))
        else:
            result = "DEFGATE {}:\n".format(self.name)

        for row in self.matrix:
            result += "    "
            fcols = [format_matrix_element(col) for col in row]
            result += ", ".join(fcols)
            result += "\n"
        return result

    def get_constructor(self) -> Union[Callable[..., Gate], Callable[..., Callable[..., Gate]]]:
        """
        :returns: A function that constructs this gate on variable qubit indices. E.g.
                  `mygate.get_constructor()(1) applies the gate to qubit 1.`
        """
        if self.parameters:
            return lambda *params: lambda *qubits: Gate(
                name=self.name, params=list(params), qubits=list(map(unpack_qubit, qubits))
            )
        else:
            return lambda *qubits: Gate(name=self.name, params=[], qubits=list(map(unpack_qubit, qubits)))

    def num_args(self) -> int:
        """
        :return: The number of qubit arguments the gate takes.
        """
        rows = len(self.matrix)
        return int(np.log2(rows))


class DefPermutationGate(DefGate):
    def __init__(self, name: str, permutation: Union[List[Union[int, np.int_]], np.ndarray]):
        if not isinstance(name, str):
            raise TypeError("Gate name must be a string")

        if name in RESERVED_WORDS:
            raise ValueError(f"Cannot use {name} for a gate name since it's a reserved word")

        if not isinstance(permutation, (list, np.ndarray)):
            raise ValueError(f"Permutation must be a list or NumPy array, got value of type {type(permutation)}")

        permutation = np.asarray(permutation)

        ndim = permutation.ndim
        if 1 != ndim:
            raise ValueError(f"Permutation must have dimension 1, got {permutation.ndim}")

        elts = permutation.shape[0]
        if 0 != elts & (elts - 1):
            raise ValueError(f"Dimension of permutation must be a power of 2, got {elts}")

        self.name = name
        self.permutation = permutation
        self.parameters = None

    def out(self) -> str:
        body = ", ".join([str(p) for p in self.permutation])
        return f"DEFGATE {self.name} AS PERMUTATION:\n    {body}"

    def num_args(self) -> int:
        """
        :return: The number of qubit arguments the gate takes.
        """
        return int(np.log2(len(self.permutation)))


class DefGateByPaulis(DefGate):
    """
    Records a gate definition as the exponentiation of a PauliSum.
    """

    def __init__(
        self,
        gate_name: str,
        parameters: List[Parameter],
        arguments: List[QubitDesignator],
        body: "PauliSum",
    ):
        if not isinstance(gate_name, str):
            raise TypeError("Gate name must be a string")

        if gate_name in RESERVED_WORDS:
            raise ValueError(f"Cannot use {gate_name} for a gate name since it's a reserved word")

        self.name = gate_name
        self.parameters = parameters
        self.arguments = arguments
        self.body = body

    def out(self) -> str:
        out = f"DEFGATE {self.name}"
        if self.parameters is not None:
            out += f"({', '.join(map(str, self.parameters))}) "
        out += f"{' '.join(map(str, self.arguments))} AS PAULI-SUM:\n"
        for term in self.body:
            args = term._ops.keys()
            word = term._ops.values()
            out += f"    {''.join(word)}({term.coefficient}) " + " ".join(map(str, args)) + "\n"
        return out

    def num_args(self) -> int:
        return len(self.arguments)


class JumpTarget(AbstractInstruction):
    """
    Representation of a target that can be jumped to.
    """

    def __init__(self, label: Union[Label, LabelPlaceholder]):
        if not isinstance(label, (Label, LabelPlaceholder)):
            raise TypeError("label must be a Label")
        self.label = label

    def __repr__(self) -> str:
        return "<JumpTarget {0}>".format(str(self.label))

    def out(self) -> str:
        return "LABEL {0}".format(str(self.label))


class JumpConditional(AbstractInstruction):
    """
    Abstract representation of an conditional jump instruction.
    """

    op: ClassVar[str]

    def __init__(self, target: Union[Label, LabelPlaceholder], condition: MemoryReference):
        if not isinstance(target, (Label, LabelPlaceholder)):
            raise TypeError("target should be a Label")
        if not isinstance(condition, MemoryReference):
            raise TypeError("condition should be an MemoryReference")
        self.target = target
        self.condition = condition

    def out(self) -> str:
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

    op: ClassVar[str]

    def out(self) -> str:
        return self.op

    def __str__(self) -> str:
        return self.out()


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

    op: ClassVar[str]

    def __init__(self, target: MemoryReference):
        if not isinstance(target, MemoryReference):
            raise TypeError("target operand should be an MemoryReference")
        self.target = target

    def out(self) -> str:
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

    op: ClassVar[str]

    def __init__(self, left: MemoryReference, right: Union[MemoryReference, int]):
        if not isinstance(left, MemoryReference):
            raise TypeError("left operand should be an MemoryReference")
        if not isinstance(right, MemoryReference) and not isinstance(right, int):
            raise TypeError("right operand should be an MemoryReference or an Int")
        self.left = left
        self.right = right

    def out(self) -> str:
        return "%s %s %s" % (self.op, self.left, self.right)


class ClassicalAnd(LogicalBinaryOp):
    """
    WARNING: The operand order for ClassicalAnd has changed.  In pyQuil versions <= 1.9, AND had
    signature

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


class ArithmeticBinaryOp(AbstractInstruction):
    """
    The abstract class for binary arithmetic classical instructions.
    """

    op: ClassVar[str]

    def __init__(self, left: MemoryReference, right: Union[MemoryReference, int, float]):
        if not isinstance(left, MemoryReference):
            raise TypeError("left operand should be an MemoryReference")
        if not isinstance(right, MemoryReference) and not isinstance(right, int) and not isinstance(right, float):
            raise TypeError("right operand should be an MemoryReference or a numeric literal")
        self.left = left
        self.right = right

    def out(self) -> str:
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

    def __init__(self, left: MemoryReference, right: Union[MemoryReference, int, float]):
        if not isinstance(left, MemoryReference):
            raise TypeError(
                "Left operand of MOVE should be an MemoryReference.  "
                "Note that the order of the operands in pyQuil 2.0 has reversed from "
                "the order of pyQuil 1.9 ."
            )
        if not isinstance(right, MemoryReference) and not isinstance(right, int) and not isinstance(right, float):
            raise TypeError("Right operand of MOVE should be an MemoryReference or a numeric literal")
        self.left = left
        self.right = right

    def out(self) -> str:
        return "%s %s %s" % (self.op, self.left, self.right)


class ClassicalExchange(AbstractInstruction):
    """
    The EXCHANGE instruction.
    """

    op = "EXCHANGE"

    def __init__(self, left: MemoryReference, right: MemoryReference):
        if not isinstance(left, MemoryReference):
            raise TypeError("left operand should be an MemoryReference")
        if not isinstance(right, MemoryReference):
            raise TypeError("right operand should be an MemoryReference")
        self.left = left
        self.right = right

    def out(self) -> str:
        return "%s %s %s" % (self.op, self.left, self.right)


class ClassicalConvert(AbstractInstruction):
    """
    The CONVERT instruction.
    """

    op = "CONVERT"

    def __init__(self, left: MemoryReference, right: MemoryReference):
        if not isinstance(left, MemoryReference):
            raise TypeError("left operand should be an MemoryReference")
        if not isinstance(right, MemoryReference):
            raise TypeError("right operand should be an MemoryReference")
        self.left = left
        self.right = right

    def out(self) -> str:
        return "%s %s %s" % (self.op, self.left, self.right)


class ClassicalLoad(AbstractInstruction):
    """
    The LOAD instruction.
    """

    op = "LOAD"

    def __init__(self, target: MemoryReference, left: str, right: MemoryReference):
        if not isinstance(target, MemoryReference):
            raise TypeError("target operand should be an MemoryReference")
        if not isinstance(right, MemoryReference):
            raise TypeError("right operand should be an MemoryReference")
        self.target = target
        self.left = left
        self.right = right

    def out(self) -> str:
        return "%s %s %s %s" % (self.op, self.target, self.left, self.right)


class ClassicalStore(AbstractInstruction):
    """
    The STORE instruction.
    """

    op = "STORE"

    def __init__(self, target: str, left: MemoryReference, right: Union[MemoryReference, int, float]):
        if not isinstance(left, MemoryReference):
            raise TypeError("left operand should be an MemoryReference")
        if not (isinstance(right, MemoryReference) or isinstance(right, int) or isinstance(right, float)):
            raise TypeError("right operand should be an MemoryReference or an int or float.")
        self.target = target
        self.left = left
        self.right = right

    def out(self) -> str:
        return "%s %s %s %s" % (self.op, self.target, self.left, self.right)


class ClassicalComparison(AbstractInstruction):
    """
    Abstract class for ternary comparison instructions.
    """

    op: ClassVar[str]

    def __init__(
        self,
        target: MemoryReference,
        left: MemoryReference,
        right: Union[MemoryReference, int, float],
    ):
        if not isinstance(target, MemoryReference):
            raise TypeError("target operand should be an MemoryReference")
        if not isinstance(left, MemoryReference):
            raise TypeError("left operand should be an MemoryReference")
        if not (isinstance(right, MemoryReference) or isinstance(right, int) or isinstance(right, float)):
            raise TypeError("right operand should be an MemoryReference or an int or float.")
        self.target = target
        self.left = left
        self.right = right

    def out(self) -> str:
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

    def __init__(self, target: Union[Label, LabelPlaceholder]):
        if not isinstance(target, (Label, LabelPlaceholder)):
            raise TypeError("target should be a Label: {target}")
        self.target = target

    def out(self) -> str:
        return "JUMP %s" % self.target


class Pragma(AbstractInstruction):
    """
    A PRAGMA instruction.

    This is printed in QUIL as::

        PRAGMA <command> <arg1> <arg2> ... <argn> "<freeform_string>"

    """

    def __init__(
        self,
        command: str,
        args: Iterable[Union[QubitDesignator, str]] = (),
        freeform_string: str = "",
    ):
        if not isinstance(command, str):
            raise TypeError(f"Pragma's require an identifier: {command}")

        if not isinstance(args, collections.abc.Iterable):
            raise TypeError(f"Pragma arguments must be an Iterable: {args}")
        for a in args:
            if not (
                isinstance(a, str) or isinstance(a, int) or isinstance(a, QubitPlaceholder) or isinstance(a, Qubit)
            ):
                raise TypeError(f"Pragma arguments must be strings or integers: {a}")
        if not isinstance(freeform_string, str):
            raise TypeError(f"The freeform string argument must be a string: {freeform_string}")

        self.command = command
        self.args = tuple(args)
        self.freeform_string = freeform_string

    def out(self) -> str:
        ret = "PRAGMA {}".format(self.command)
        if self.args:
            ret += " {}".format(" ".join(str(a) for a in self.args))
        if self.freeform_string:
            ret += ' "{}"'.format(self.freeform_string)
        return ret

    def __repr__(self) -> str:
        return "<PRAGMA {}>".format(self.command)


class Declare(quil_rs.Declaration, AbstractInstruction):
    """
    A DECLARE directive.

    This is printed in Quil as::

        DECLARE <name> <memory-type> (SHARING <other-name> (OFFSET <amount> <type>)* )?

    """

    @staticmethod
    def __new__(
        cls,
        name: str,
        memory_type: str,
        memory_size: int = 1,
        shared_region: Optional[str] = None,
        offsets: Optional[Iterable[Tuple[int, str]]] = None,
    ):
        vector = quil_rs.Vector(Declare._memory_type_to_scalar_type(memory_type), memory_size)
        sharing = None
        if shared_region is not None:
            sharing = quil_rs.Sharing(shared_region, Declare._to_rs_offsets(offsets))
        return super().__new__(cls, name, vector, sharing)

    @classmethod
    def _from_rs_declaration(cls, declaration: quil_rs.Declaration) -> "Declare":
        return super().__new__(cls, declaration.name, declaration.size, declaration.sharing)

    @staticmethod
    def _memory_type_to_scalar_type(memory_type: str) -> quil_rs.ScalarType:
        memory_type = memory_type.upper()
        if memory_type == "BIT":
            return quil_rs.ScalarType.Bit
        if memory_type == "INTEGER":
            return quil_rs.ScalarType.Integer
        if memory_type == "REAL":
            return quil_rs.ScalarType.Real
        if memory_type == "OCTET":
            return quil_rs.ScalarType.Octet
        raise ValueError(f"{memory_type} is not a valid scalar type.")

    @staticmethod
    def _to_rs_offsets(offsets: Optional[Iterable[Tuple[int, str]]]):
        if offsets is None:
            return []
        return [
            quil_rs.Offset(offset, Declare._memory_type_to_scalar_type(memory_type)) for offset, memory_type in offsets
        ]

    @property
    def memory_type(self) -> str:
        return str(super().size.data_type)

    @memory_type.setter
    def memory_type(self, memory_type: str):
        vector = super().size
        vector.data_type = Declare._memory_type_to_scalar_type(memory_type)
        quil_rs.Declaration.size.__set__(self, vector)

    @property
    def memory_size(self) -> int:
        return super().size.length

    @memory_size.setter
    def memory_size(self, memory_size: int):
        vector = super().size
        vector.length = memory_size
        quil_rs.Declaration.size.__set__(self, vector)

    @property
    def shared_region(self) -> Optional[str]:
        sharing = super().sharing
        if sharing is None:
            return None
        return sharing.name

    @shared_region.setter
    def shared_region(self, shared_region: Optional[str]):
        sharing = super().sharing
        if sharing is None:
            if shared_region is None:
                return
            sharing = quil_rs.Sharing(shared_region, [])
        else:
            sharing.name = shared_region
        quil_rs.Declaration.sharing.__set__(self, sharing)

    @property
    def offsets(self) -> List[Tuple[int, str]]:
        sharing = super().sharing
        if sharing is None:
            return []
        return [(offset.offset, str(offset.data_type)) for offset in sharing.offsets]

    @offsets.setter
    def offsets(self, offsets: Optional[List[Tuple[int, str]]]):
        sharing = super().sharing
        if sharing is None:
            raise ValueError("DECLARE without a shared region cannot use offsets")
        sharing.offsets = Declare._to_rs_offsets(offsets)
        quil_rs.Declaration.sharing.__set__(self, sharing)

    def asdict(self) -> Dict[str, Union[Iterable[Tuple[int, str]], Optional[str], int]]:
        return {
            "name": self.name,
            "memory_type": self.memory_type,
            "memory_size": self.memory_size,
            "shared_region": self.shared_region,
            "offsets": self.offsets,
        }

    def out(self) -> str:
        return str(self)


class RawInstr(AbstractInstruction):
    """
    A raw instruction represented as a string.
    """

    def __init__(self, instr_str: str):
        if not isinstance(instr_str, str):
            raise TypeError("Raw instructions require a string.")
        self.instr = instr_str

    def out(self) -> str:
        return self.instr

    def __repr__(self) -> str:
        return "<RawInstr {}>".format(self.instr)


class Pulse(AbstractInstruction):
    def __init__(self, frame: Frame, waveform: Waveform, nonblocking: bool = False):
        self.frame = frame
        self.waveform = waveform
        self.nonblocking = nonblocking

    def out(self) -> str:
        result = "NONBLOCKING " if self.nonblocking else ""
        result += f"PULSE {self.frame} {self.waveform.out()}"
        return result

    def get_qubits(self, indices: bool = True) -> Set[QubitDesignator]:
        return _get_frame_qubits(self.frame, indices)


class SetFrequency(AbstractInstruction):
    def __init__(self, frame: Frame, freq: ParameterDesignator):
        self.frame = frame
        self.freq = freq

    def out(self) -> str:
        return f"SET-FREQUENCY {self.frame} {self.freq}"

    def get_qubits(self, indices: bool = True) -> Set[QubitDesignator]:
        return _get_frame_qubits(self.frame, indices)


class ShiftFrequency(AbstractInstruction):
    def __init__(self, frame: Frame, freq: ParameterDesignator):
        self.frame = frame
        self.freq = freq

    def out(self) -> str:
        return f"SHIFT-FREQUENCY {self.frame} {self.freq}"

    def get_qubits(self, indices: bool = True) -> Set[QubitDesignator]:
        return _get_frame_qubits(self.frame, indices)


class SetPhase(AbstractInstruction):
    def __init__(self, frame: Frame, phase: ParameterDesignator):
        self.frame = frame
        self.phase = phase

    def out(self) -> str:
        return f"SET-PHASE {self.frame} {self.phase}"

    def get_qubits(self, indices: bool = True) -> Set[QubitDesignator]:
        return _get_frame_qubits(self.frame, indices)


class ShiftPhase(AbstractInstruction):
    def __init__(self, frame: Frame, phase: ParameterDesignator):
        self.frame = frame
        self.phase = phase

    def out(self) -> str:
        return f"SHIFT-PHASE {self.frame} {self.phase}"

    def get_qubits(self, indices: bool = True) -> Set[QubitDesignator]:
        return _get_frame_qubits(self.frame, indices)


class SwapPhase(AbstractInstruction):
    def __init__(self, frameA: Frame, frameB: Frame):
        self.frameA = frameA
        self.frameB = frameB

    def out(self) -> str:
        return f"SWAP-PHASE {self.frameA} {self.frameB}"

    def get_qubits(self, indices: bool = True) -> Set[QubitDesignator]:
        return _get_frame_qubits(self.frameA, indices) | _get_frame_qubits(self.frameB, indices)


class SetScale(AbstractInstruction):
    def __init__(self, frame: Frame, scale: ParameterDesignator):
        self.frame = frame
        self.scale = scale

    def out(self) -> str:
        return f"SET-SCALE {self.frame} {self.scale}"

    def get_qubits(self, indices: bool = True) -> Set[QubitDesignator]:
        return _get_frame_qubits(self.frame, indices)


class Capture(AbstractInstruction):
    def __init__(
        self,
        frame: Frame,
        kernel: Waveform,
        memory_region: MemoryReference,
        nonblocking: bool = False,
    ):
        self.frame = frame
        self.kernel = kernel
        self.memory_region = memory_region
        self.nonblocking = nonblocking

    def out(self) -> str:
        result = "NONBLOCKING " if self.nonblocking else ""
        result += f"CAPTURE {self.frame} {self.kernel.out()}"
        result += f" {self.memory_region.out()}" if self.memory_region else ""
        return result

    def get_qubits(self, indices: bool = True) -> Set[QubitDesignator]:
        return _get_frame_qubits(self.frame, indices)


class RawCapture(AbstractInstruction):
    def __init__(
        self,
        frame: Frame,
        duration: float,
        memory_region: MemoryReference,
        nonblocking: bool = False,
    ):
        self.frame = frame
        self.duration = duration
        self.memory_region = memory_region
        self.nonblocking = nonblocking

    def out(self) -> str:
        result = "NONBLOCKING " if self.nonblocking else ""
        result += f"RAW-CAPTURE {self.frame} {self.duration} {self.memory_region.out()}"
        return result

    def get_qubits(self, indices: bool = True) -> Set[QubitDesignator]:
        return _get_frame_qubits(self.frame, indices)


class DelayFrames(AbstractInstruction):
    def __init__(self, frames: List[Frame], duration: float):
        # all frames should be on the same qubits
        if len(frames) == 0:
            raise ValueError("DELAY expected nonempty list of frames.")
        if len(set(tuple(f.qubits) for f in frames)) != 1:
            raise ValueError("DELAY with explicit frames requires all frames are on the same qubits.")

        self.frames = frames
        self.duration = duration

    def out(self) -> str:
        qubits = self.frames[0].qubits
        ret = "DELAY " + _format_qubits_str(qubits)
        for f in self.frames:
            ret += f' "{f.name}"'
        ret += f" {self.duration}"
        return ret


class DelayQubits(AbstractInstruction):
    def __init__(self, qubits: List[Union[Qubit, FormalArgument]], duration: float):
        self.qubits = qubits
        self.duration = duration

    def out(self) -> str:
        return f"DELAY {_format_qubits_str(self.qubits)} {self.duration}"


class FenceAll(SimpleInstruction):
    """
    The FENCE instruction.
    """

    op = "FENCE"


class Fence(AbstractInstruction):
    def __init__(self, qubits: List[Union[Qubit, FormalArgument]]):
        self.qubits = qubits

    def out(self) -> str:
        ret = "FENCE " + _format_qubits_str(self.qubits)
        return ret


class DefWaveform(AbstractInstruction):
    def __init__(
        self,
        name: str,
        parameters: List[Parameter],
        entries: List[Union[Complex, Expression]],
    ):
        self.name = name
        self.parameters = parameters
        self.entries = entries
        for e in entries:
            if not isinstance(e, (Complex, Expression)):
                raise TypeError(f"Unsupported waveform entry {e}")

    def out(self) -> str:
        ret = f"DEFWAVEFORM {self.name}"
        # TODO: simplify this
        if len(self.parameters) > 0:
            first_param, *params = self.parameters
            ret += f"({first_param}"
            for param in params:
                ret += f", {param}"
            ret += ")"
        ret += ":\n    "

        ret += ", ".join(map(_complex_str, self.entries))
        return ret


class DefCalibration(quil_rs.Calibration, AbstractInstruction):
    def __new__(
        cls,
        name: str,
        parameters: Iterable[ParameterDesignator],
        qubits: Iterable[Union[Qubit, FormalArgument]],
        instrs: Iterable[AbstractInstruction],
        modifiers: Iterable[quil_rs.GateModifier] = [],
    ):
        return super().__new__(
            cls,
            name,
            _convert_to_rs_expressions(parameters),
            _convert_to_rs_qubits(qubits),
            _convert_to_rs_instructions(instrs),
            modifiers,
        )

    @classmethod
    def _from_rs_calibration(cls, calibration: quil_rs.Calibration) -> "DefCalibration":
        return cls(
            calibration.name,
            calibration.parameters,
            calibration.qubits,
            calibration.instructions,
            calibration.modifiers,
        )

    @property
    def parameters(self):
        return _convert_to_py_parameters(super().parameters)

    @parameters.setter
    def parameters(self, parameters: Iterable[ParameterDesignator]):
        quil_rs.Calibration.parameters.__set__(self, _convert_to_rs_expressions(parameters))

    @property
    def qubits(self):
        return _convert_to_py_qubits(super().qubits)

    @qubits.setter
    def qubits(self, qubits: Iterable[QubitDesignator]):
        quil_rs.Calibration.qubits.__set__(self, _convert_to_rs_qubits(qubits))

    @property
    def instrs(self):
        return _convert_to_py_instructions(super().instructions)

    @instrs.setter
    def instrs(self, instrs: Iterable[AbstractInstruction]):
        quil_rs.Calibration.instructions.__set__(self, _convert_to_rs_instructions(instrs))

    def out(self) -> str:
        return str(self)


class DefMeasureCalibration(quil_rs.MeasureCalibrationDefinition, AbstractInstruction):
    def __new__(
        cls,
        qubit: Union[Qubit, FormalArgument],
        memory_reference: MemoryReference,
        instrs: List[AbstractInstruction],
    ) -> "DefMeasureCalibration":
        return super().__new__(
            cls,
            _convert_to_rs_qubit(qubit),
            memory_reference.name,
            _convert_to_rs_instructions(instrs),
        )

    @classmethod
    def _from_rs_measure_calibration_definition(
        cls, calibration: quil_rs.MeasureCalibrationDefinition
    ) -> "DefMeasureCalibration":
        return super().__new__(cls, calibration.qubit, calibration.parameter, calibration.instructions)

    @property
    def qubit(self) -> QubitDesignator:
        return _convert_to_py_qubit(super().qubit)

    @qubit.setter
    def qubit(self, qubit: QubitDesignator):
        quil_rs.MeasureCalibrationDefinition.qubit.__set__(self, _convert_to_rs_qubit(qubit))

    @property
    def memory_reference(self) -> Optional[MemoryReference]:
        return MemoryReference._from_parameter_str(super().parameter)

    @memory_reference.setter
    def memory_reference(self, memory_reference: MemoryReference):
        quil_rs.MeasureCalibrationDefinition.parameter.__set__(self, memory_reference.name)

    @property
    def instrs(self):
        return _convert_to_py_instructions(super().instructions)

    @instrs.setter
    def instrs(self, instrs: Iterable[AbstractInstruction]):
        quil_rs.MeasureCalibrationDefinition.instructions.__set__(self, _convert_to_rs_instructions(instrs))

    def out(self) -> str:
        return str(self)


class DefFrame(quil_rs.FrameDefinition, AbstractInstruction):
    @staticmethod
    def __new__(
        cls,
        frame: Frame,
        direction: Optional[str] = None,
        initial_frequency: Optional[float] = None,
        hardware_object: Optional[str] = None,
        sample_rate: Optional[float] = None,
        center_frequency: Optional[float] = None,
    ) -> "DefFrame":
        attributes = {
            key: DefFrame._to_attribute_value(value)
            for key, value in zip(
                ["DIRECTION", "INITIAL-FREQUENCY", "HARDWARE-OBJECT", "SAMPLE-RATE", "CENTER-FREQUENCY"],
                [direction, initial_frequency, hardware_object, sample_rate, center_frequency],
            )
            if value is not None
        }
        return super().__new__(cls, frame, attributes)

    @classmethod
    def _from_rs_frame_definition(cls, def_frame: quil_rs.FrameDefinition) -> "DefFrame":
        return super().__new__(def_frame.frame, def_frame.attributes)

    @classmethod
    def _from_rs_attribute_values(cls, frame: quil_rs.FrameIdentifier, attributes: Dict[str, quil_rs.AttributeValue]):
        return super().__new__(cls, frame, attributes)

    @staticmethod
    def _to_attribute_value(value: Union[str, float]) -> quil_rs.AttributeValue:
        if isinstance(value, str):
            return quil_rs.AttributeValue.from_string(value)
        if isinstance(value, float):
            return quil_rs.AttributeValue.from_expression(quil_rs_expr.Expression.from_number(complex(value)))
        raise ValueError(f"{type(value)} is not a valid AttributeValue")

    def out(self) -> str:
        return str(self)

    @property
    def frame(self) -> Frame:
        return Frame._from_rs_frame_identifier(super().identifier)

    @frame.setter
    def frame(self, frame: Frame):
        quil_rs.FrameDefinition.identifier.__set__(self, frame)

    def _set_attribute(self, name: str, value: Union[str, float]):
        updated = super().attributes
        updated.update({name: DefFrame._to_attribute_value(value)})
        quil_rs.FrameDefinition.attributes.__set__(self, updated)

    def _get_attribute(self, name: str) -> Optional[Union[str, float]]:
        value = super().attributes.get(name, None)
        if value is None:
            return None
        if value.is_string():
            return value.to_string()
        if value.is_expression():
            return value.to_expression().to_number().real

    @property
    def direction(self) -> Optional[str]:
        return self._get_attribute("DIRECTION")  # type: ignore

    @direction.setter
    def direction(self, direction: str):
        self._set_attribute("DIRECTION", direction)

    @property
    def initial_frequency(self) -> Optional[float]:
        return self._get_attribute("INITIAL-FREQUENCY")  # type: ignore

    @initial_frequency.setter
    def initial_frequency(self, initial_frequency: float):
        self._set_attribute("INITIAL-FREQUENCY", initial_frequency)

    @property
    def hardware_object(self) -> Optional[str]:
        return self._get_attribute("HARDWARE-OBJECT")  # type: ignore

    @hardware_object.setter
    def hardware_object(self, hardware_object: str):
        self._set_attribute("HARDWARE-OBJECT", hardware_object)

    @property
    def sample_rate(self) -> Frame:
        return self._get_attribute("SAMPLE-RATE")  # type: ignore

    @sample_rate.setter
    def sample_rate(self, sample_rate: float):
        self._set_attribute("SAMPLE-RATE", sample_rate)

    @property
    def center_frequency(self) -> Frame:
        return self._get_attribute("CENTER-FREQUENCY")  # type: ignore

    @center_frequency.setter
    def center_frequency(self, center_frequency: float):
        self._set_attribute("CENTER-FREQUENCY", center_frequency)
