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
    Type,
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
    _convert_to_py_expression,
    _convert_to_py_parameters,
    format_parameter,
    unpack_qubit,
    _complex_str,
)

if TYPE_CHECKING:  # avoids circular import
    from pyquil.paulis import PauliSum

import quil.instructions as quil_rs
import quil.expression as quil_rs_expr


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
    if isinstance(instr, quil_rs.Calibration):
        return quil_rs.Instruction.from_calibration_definition(instr)
    if isinstance(instr, quil_rs.Declaration):
        return quil_rs.Instruction.from_declaration(instr)
    if isinstance(instr, quil_rs.Delay):
        return quil_rs.Instruction.from_delay(instr)
    if isinstance(instr, quil_rs.Fence):
        return quil_rs.Instruction.from_fence(instr)
    if isinstance(instr, quil_rs.Gate):
        return quil_rs.Instruction.from_gate(instr)
    if isinstance(instr, quil_rs.MeasureCalibrationDefinition):
        return quil_rs.Instruction.from_measure_calibration_definition(instr)
    if isinstance(instr, quil_rs.Measurement):
        return quil_rs.Instruction.from_measurement(instr)
    if isinstance(instr, quil_rs.Pragma):
        return quil_rs.Instruction.from_pragma(instr)
    if isinstance(instr, quil_rs.Reset):
        return quil_rs.Instruction.from_reset(instr)
    if isinstance(instr, AbstractInstruction):
        return quil_rs.Instruction(instr)
    else:
        raise ValueError(f"{type(instr)} is not an Instruction")


def _convert_to_rs_instructions(instrs: Iterable[AbstractInstruction]) -> List[quil_rs.Instruction]:
    return [_convert_to_rs_instruction(instr) for instr in instrs]


def _convert_to_py_instruction(instr: quil_rs.Instruction) -> AbstractInstruction:
    if isinstance(instr, quil_rs.Instruction):
        # TODOV4: Will have to handle unit variants since they don't have inner data
        instr = instr.inner()
    if isinstance(instr, quil_rs.Calibration):
        return DefCalibration._from_rs_calibration(instr)
    if isinstance(instr, quil_rs.Declaration):
        return Declare._from_rs_declaration(instr)
    if isinstance(instr, quil_rs.Delay):
        return Delay._from_rs_delay(instr)
    if isinstance(instr, quil_rs.Fence):
        return Fence._from_rs_fence(instr)
    if isinstance(instr, quil_rs.Gate):
        return Gate._from_rs_gate(instr)
    if isinstance(instr, quil_rs.MeasureCalibrationDefinition):
        return DefMeasureCalibration._from_rs_measure_calibration_definition(instr)
    if isinstance(instr, quil_rs.Measurement):
        return Measurement._from_rs_measurement(instr)
    if isinstance(instr, quil_rs.Pragma):
        return Pragma._from_rs_pragma(instr)
    if isinstance(instr, quil_rs.Reset):
        return Reset._from_rs_reset(instr)
    if isinstance(instr, quil_rs.GateDefinition):
        return DefGate._from_rs_gate_definition(instr)
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
        return super().__new__(
            cls, name, _convert_to_rs_expressions(params), _convert_to_rs_qubits(qubits), list(modifiers)
        )

    @classmethod
    def _from_rs_gate(cls, gate: quil_rs.Gate) -> "Gate":
        return super().__new__(cls, gate.name, gate.parameters, gate.qubits, gate.modifiers)

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


class Reset(quil_rs.Reset, AbstractInstruction):
    """
    The RESET instruction.
    """

    def __new__(cls, qubit: Optional[Union[Qubit, QubitPlaceholder, FormalArgument]] = None):
        rs_qubit: Optional[quil_rs.Qubit] = None
        if qubit is not None:
            rs_qubit = _convert_to_rs_qubit(qubit)
        return super().__new__(cls, rs_qubit)

    @classmethod
    def _from_rs_reset(cls, reset: quil_rs.Reset) -> "Reset":
        return super().__new__(cls, reset.qubit)

    def out(self) -> str:
        return str(self)

    @deprecated(
        deprecated_in="4.0",
        removed_in="5.0",
        current_version=pyquil_version,
        details="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Optional[Set[QubitDesignator]]:
        if super().qubit is None:
            return None
        if indices:
            return self.get_qubit_indices()  # type: ignore
        return {_convert_to_py_qubit(super().qubit)}  # type: ignore

    def get_qubit_indices(self) -> Optional[Set[int]]:
        if super().qubit is None:
            return None
        return {super().qubit.to_fixed()}  # type: ignore

    @property
    def qubit(self) -> Optional[QubitDesignator]:
        if super().qubit:
            return _convert_to_py_qubit(super().qubit)  # type: ignore
        return super().qubit

    @qubit.setter
    def qubit(self, qubit: Optional[QubitDesignator]):
        rs_qubit: Optional[quil_rs.Qubit] = None
        if qubit is not None:
            rs_qubit = _convert_to_rs_qubit(qubit)
        quil_rs.Reset.qubit.__set__(self, rs_qubit)


class ResetQubit(Reset):
    """
    This is the pyQuil object for a Quil targeted reset instruction.
    """

    def __new__(cls, qubit: Union[Qubit, QubitPlaceholder, FormalArgument]):
        if qubit is None:
            raise TypeError("qubit should not be None")
        return super().__new__(cls, qubit)


class DefGate(quil_rs.GateDefinition, AbstractInstruction):
    """
    A DEFGATE directive.

    :param name: The name of the newly defined gate.
    :param matrix: The matrix defining this gate.
    :param parameters: list of parameters that are used in this gate
    """

    def __new__(
        cls,
        name: str,
        matrix: Union[List[List[Expression]], np.ndarray, np.matrix],
        parameters: Optional[List[Parameter]] = None,
    ) -> "DefGate":
        specification = DefGate._convert_to_matrix_specification(matrix)
        rs_parameters = [param.name for param in parameters or []]
        return super().__new__(cls, name, rs_parameters, specification)

    @classmethod
    def _from_rs_gate_definition(cls, gate_definition: quil_rs.GateDefinition) -> "DefGate":
        return super().__new__(cls, gate_definition.name, gate_definition.parameters, gate_definition.specification)

    @staticmethod
    def _convert_to_matrix_specification(
        matrix: Union[List[List[Expression]], np.ndarray, np.matrix]
    ) -> quil_rs.GateSpecification:
        to_rs_matrix = np.vectorize(_convert_to_rs_expression)
        return quil_rs.GateSpecification.from_matrix(to_rs_matrix(np.asarray(matrix)))

    def out(self) -> str:
        return str(self)

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

    @property
    def matrix(self) -> np.ndarray:
        to_py_matrix = np.vectorize(_convert_to_py_expression)
        return to_py_matrix(np.asarray(super().specification.to_matrix()))

    @matrix.setter
    def matrix(self, matrix: np.ndarray):
        quil_rs.GateDefinition.specification.__set__(self, DefGate._convert_to_matrix_specification(matrix))

    @property
    def parameters(self) -> List[Parameter]:
        return [Parameter(name) for name in super().parameters]

    @parameters.setter
    def parameters(self, parameters: Optional[List[Parameter]]):
        quil_rs.GateDefinition.parameters.__set__(self, [param.name for param in parameters or []])

    def __hash__(self) -> int:
        return hash(self.out())


class DefPermutationGate(DefGate):
    def __new__(cls, name: str, permutation: Union[List[int], np.ndarray]):
        specification = DefPermutationGate._convert_to_permutation_specification(permutation)
        gate_definition = quil_rs.GateDefinition(name, [], specification)
        return cls._from_rs_gate_definition(gate_definition)

    @staticmethod
    def _convert_to_permutation_specification(permutation: Union[List[int], np.ndarray]) -> quil_rs.GateSpecification:
        return quil_rs.GateSpecification.from_permutation([int(x) for x in permutation])

    @property
    def permutation(self) -> List[int]:
        return super().specification.to_permutation()

    @permutation.setter
    def permutation(self, permutation: List[int]):
        specification = DefPermutationGate._convert_to_permutation_specification(permutation)
        quil_rs.GateDefinition.specification.__set__(self, specification)

    def num_args(self) -> int:
        """
        :return: The number of qubit arguments the gate takes.
        """
        return int(np.log2(len(self.permutation)))


class DefGateByPaulis(DefGate):
    """
    Records a gate definition as the exponentiation of a PauliSum.
    """

    def __new__(
        cls,
        gate_name: str,
        parameters: List[Parameter],
        arguments: List[QubitDesignator],
        body: "PauliSum",
    ):
        specification = DefGateByPaulis._convert_to_pauli_specification(body, arguments)
        rs_parameters = [param.name for param in parameters]
        gate_definition = quil_rs.GateDefinition(gate_name, rs_parameters, specification)
        return cls._from_rs_gate_definition(gate_definition)

    @staticmethod
    def _convert_to_pauli_specification(body: "PauliSum", arguments: List[QubitDesignator]):
        return quil_rs.GateSpecification.from_pauli_sum(body._to_rs_pauli_sum(arguments))

    @property
    def arguments(self) -> List[FormalArgument]:
        return [FormalArgument(arg) for arg in super().specification.to_pauli_sum().arguments]

    @arguments.setter
    def arguments(self, arguments: List[QubitDesignator]):
        pauli_sum = super().specification.to_pauli_sum()
        pauli_sum.arguments = [str(arg) for arg in arguments]
        quil_rs.GateDefinition.specification.__set__(self, quil_rs.GateSpecification.from_pauli_sum(pauli_sum))

    @property
    def body(self) -> "PauliSum":
        from pyquil.paulis import PauliSum  # avoids circular import

        return PauliSum._from_rs_pauli_sum(super().specification.to_pauli_sum())

    @body.setter
    def body(self, body: "PauliSum"):
        specification = quil_rs.GateSpecification.from_pauli_sum(body._to_rs_pauli_sum())
        quil_rs.GateDefinition.specification.__set__(self, specification)

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


class Pragma(quil_rs.Pragma, AbstractInstruction):
    """
    A PRAGMA instruction.

    This is printed in QUIL as::

        PRAGMA <command> <arg1> <arg2> ... <argn> "<freeform_string>"

    """

    def __new__(
        cls,
        command: str,
        args: Iterable[Union[QubitDesignator, str]] = (),
        freeform_string: str = "",
    ) -> Type["Pragma"]:
        data = freeform_string or None
        return super().__new__(cls, command, Pragma._to_pragma_arguments(args), data)

    @classmethod
    def _from_rs_pragma(cls, pragma: quil_rs.Pragma) -> "Pragma":
        return super().__new__(cls, pragma.name, pragma.arguments, pragma.data)

    @staticmethod
    def _to_pragma_arguments(args: Iterable[Union[QubitDesignator, str]]) -> List[quil_rs.PragmaArgument]:
        pragma_arguments = []
        for arg in args:
            if isinstance(arg, Qubit):
                pragma_arguments.append(quil_rs.PragmaArgument.from_integer(arg.index))
            elif isinstance(arg, (str, FormalArgument)):
                pragma_arguments.append(quil_rs.PragmaArgument.from_identifier(str(arg)))
            else:
                raise TypeError(f"{type(arg)} isn't a valid QubitDesignator")
        return pragma_arguments

    @staticmethod
    def _to_py_arguments(args: List[quil_rs.PragmaArgument]) -> List[QubitDesignator]:
        arguments = []
        for arg in args:
            if arg.is_integer():
                arguments.append(Qubit(arg.to_integer()))
            else:
                arguments.append(FormalArgument(arg.to_identifier()))
        return arguments

    def out(self) -> str:
        return str(self)

    @property
    def command(self) -> str:
        return super().name

    @command.setter
    def command(self, command: str):
        quil_rs.Pragma.name.__set__(self, command)

    @property
    def args(self) -> Tuple[QubitDesignator]:
        return tuple(Pragma._to_py_arguments(super().arguments))

    @args.setter
    def args(self, args: str):
        quil_rs.Pragma.arguments.__set__(self, Pragma._to_pragma_arguments(args))

    @property
    def freeform_string(self) -> str:
        return super().data or ""

    @freeform_string.setter
    def freeform_string(self, freeform_string: str):
        quil_rs.Pragma.data.__set__(self, freeform_string)


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


class Delay(quil_rs.Delay, AbstractInstruction):
    def __new__(cls, frames: List[Frame], qubits: List[Union[int, Qubit, FormalArgument]], duration: float) -> "Delay":
        frame_names = [frame.name for frame in frames]
        rs_qubits = _convert_to_rs_qubits(Delay._join_frame_qubits(frames, qubits))
        expression = quil_rs_expr.Expression.from_number(complex(duration))
        return super().__new__(cls, expression, frame_names, rs_qubits)

    @classmethod
    def _from_rs_delay(cls, delay: quil_rs.Delay) -> "Delay":
        return super().__new__(cls, delay.duration, delay.frame_names, delay.qubits)

    @staticmethod
    def _join_frame_qubits(
        frames: List[Frame], qubits: List[Union[int, Qubit, FormalArgument]]
    ) -> List[Union[int, Qubit, FormalArgument]]:
        merged_qubits = set(qubits)
        for frame in frames:
            merged_qubits.update(frame.qubits)  # type: ignore
        return list(merged_qubits)

    def out(self) -> str:
        return str(self)

    @property
    def qubits(self) -> List[QubitDesignator]:
        return _convert_to_py_qubits(super().qubits)

    @qubits.setter
    def qubits(self, qubits: List[Union[int, Qubit, FormalArgument]]):
        quil_rs.Delay.qubits.__set__(self, _convert_to_rs_qubits(qubits))

    @property
    def frames(self) -> List[Frame]:
        return [Frame(self.qubits, name) for name in super().frame_names]

    @frames.setter
    def frames(self, frames: List[Frame]):
        new_qubits = Delay._join_frame_qubits(frames, [])
        frame_names = [frame.name for frame in frames]
        quil_rs.Delay.qubits.__set__(self, _convert_to_rs_qubits(new_qubits))
        quil_rs.Delay.frame_names.__set__(self, frame_names)

    @property
    def duration(self) -> float:
        return super().duration.to_real()

    @duration.setter
    def duration(self, duration: float):
        expression = quil_rs_expr.Expression.from_number(complex(duration))
        quil_rs.Delay.duration.__set__(self, expression)


class DelayFrames(Delay):
    def __new__(cls, frames: List[Frame], duration: float):
        return super().__new__(cls, frames, [], duration)


class DelayQubits(Delay):
    def __new__(cls, qubits: List[Union[Qubit, FormalArgument]], duration: float):
        return super().__new__(cls, [], qubits, duration)


class Fence(quil_rs.Fence, AbstractInstruction):
    def __new__(cls, qubits: List[Union[Qubit, FormalArgument]]):
        return super().__new__(cls, _convert_to_rs_qubits(qubits))

    @classmethod
    def _from_rs_fence(cls, fence: quil_rs.Fence) -> "Fence":
        return super().__new__(cls, fence.qubits)

    def out(self) -> str:
        return str(self)

    @property
    def qubits(self) -> List[Union[Qubit, FormalArgument]]:
        return _convert_to_py_qubits(super().qubits)

    @qubits.setter
    def qubits(self, qubits: List[Union[Qubit, FormalArgument]]):
        quil_rs.Fence.qubits.__set__(self, _convert_to_rs_qubits(qubits))


class FenceAll(Fence):
    """
    The FENCE instruction.
    """

    def __new__(cls):
        return super().__new__(cls, [])


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
