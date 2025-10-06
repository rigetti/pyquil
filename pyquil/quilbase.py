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
"""Contains the core pyQuil objects that correspond to Quil instructions."""

import abc
from collections.abc import Container, Iterable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Optional,
    Union,
)

import numpy as np
from deprecated.sphinx import deprecated
from typing_extensions import Self

from pyquil.quilatom import (
    Expression,
    FormalArgument,
    Frame,
    Label,
    LabelPlaceholder,
    MemoryReference,
    Parameter,
    ParameterDesignator,
    Qubit,
    QubitDesignator,
    QubitPlaceholder,
    Waveform,
    _convert_to_py_expression,
    _convert_to_py_expressions,
    _convert_to_py_qubit,
    _convert_to_py_qubits,
    _convert_to_py_waveform,
    _convert_to_rs_expression,
    _convert_to_rs_expressions,
    _convert_to_rs_qubit,
    _convert_to_rs_qubits,
    unpack_qubit,
)

if TYPE_CHECKING:  # avoids circular import
    from pyquil.paulis import PauliSum

import quil.expression as quil_rs_expr
import quil.instructions as quil_rs


class AbstractInstruction(metaclass=abc.ABCMeta):
    """Abstract class for representing single instructions."""


AbstractInstruction.register(quil_rs.Instruction)
for cls in quil_rs.Instruction.__subclasses__():
    AbstractInstruction.register(cls)


def _convert_to_rs_instruction(instr: Union[AbstractInstruction, quil_rs.Instruction]) -> quil_rs.Instruction:
    if isinstance(instr, quil_rs.Instruction):
        return instr
    if isinstance(instr, quil_rs.Arithmetic):
        return quil_rs.Instruction.Arithmetic(instr)
    if isinstance(instr, quil_rs.BinaryLogic):
        return quil_rs.Instruction.BinaryLogic(instr)
    if isinstance(instr, quil_rs.Call):
        return quil_rs.Instruction.Call(instr)
    if isinstance(instr, quil_rs.Capture):
        return quil_rs.Instruction.Capture(instr)
    if isinstance(instr, quil_rs.CircuitDefinition):
        return quil_rs.Instruction.CircuitDefinition(instr)
    if isinstance(instr, quil_rs.CalibrationDefinition):
        return quil_rs.Instruction.CalibrationDefinition(instr)
    if isinstance(instr, quil_rs.Convert):
        return quil_rs.Instruction.Convert(instr)
    if isinstance(instr, quil_rs.Declaration):
        return quil_rs.Instruction.Declaration(instr)
    if isinstance(instr, quil_rs.Delay):
        return quil_rs.Instruction.Delay(instr)
    if isinstance(instr, quil_rs.Exchange):
        return quil_rs.Instruction.Exchange(instr)
    if isinstance(instr, quil_rs.Fence):
        return quil_rs.Instruction.Fence(instr)
    if isinstance(instr, quil_rs.FrameDefinition):
        return quil_rs.Instruction.FrameDefinition(instr)
    if isinstance(instr, quil_rs.Gate):
        return quil_rs.Instruction.Gate(instr)
    if isinstance(instr, quil_rs.GateDefinition):
        return quil_rs.Instruction.GateDefinition(instr)
    if isinstance(instr, Halt):
        return quil_rs.Instruction.Halt()
    if isinstance(instr, Include):
        return quil_rs.Instruction.Include(instr)
    if isinstance(instr, quil_rs.Load):
        return quil_rs.Instruction.Load(instr)
    if isinstance(instr, quil_rs.MeasureCalibrationDefinition):
        return quil_rs.Instruction.MeasureCalibrationDefinition(instr)
    if isinstance(instr, quil_rs.Measurement):
        return quil_rs.Instruction.Measurement(instr)
    if isinstance(instr, Nop):
        return quil_rs.Instruction.Nop()
    if isinstance(instr, quil_rs.Pragma):
        return quil_rs.Instruction.Pragma(instr)
    if isinstance(instr, quil_rs.Pulse):
        return quil_rs.Instruction.Pulse(instr)
    if isinstance(instr, quil_rs.RawCapture):
        return quil_rs.Instruction.RawCapture(instr)
    if isinstance(instr, quil_rs.Reset):
        return quil_rs.Instruction.Reset(instr)
    if isinstance(instr, quil_rs.SetFrequency):
        return quil_rs.Instruction.SetFrequency(instr)
    if isinstance(instr, quil_rs.SetPhase):
        return quil_rs.Instruction.SetPhase(instr)
    if isinstance(instr, quil_rs.SetScale):
        return quil_rs.Instruction.SetScale(instr)
    if isinstance(instr, quil_rs.ShiftFrequency):
        return quil_rs.Instruction.ShiftFrequency(instr)
    if isinstance(instr, quil_rs.ShiftPhase):
        return quil_rs.Instruction.ShiftPhase(instr)
    if isinstance(instr, quil_rs.SwapPhases):
        return quil_rs.Instruction.SwapPhases(instr)
    if isinstance(instr, quil_rs.Store):
        return quil_rs.Instruction.Store(instr)
    if isinstance(instr, Wait):
        return quil_rs.Instruction.Wait()
    if isinstance(instr, quil_rs.WaveformDefinition):
        return quil_rs.Instruction.WaveformDefinition(instr)
    if isinstance(instr, quil_rs.Label):
        return quil_rs.Instruction.Label(instr)
    if isinstance(instr, quil_rs.Move):
        return quil_rs.Instruction.Move(instr)
    if isinstance(instr, quil_rs.Jump):
        return quil_rs.Instruction.Jump(instr)
    if isinstance(instr, quil_rs.JumpWhen):
        return quil_rs.Instruction.JumpWhen(instr)
    if isinstance(instr, quil_rs.JumpUnless):
        return quil_rs.Instruction.JumpUnless(instr)
    if isinstance(instr, quil_rs.UnaryLogic):
        return quil_rs.Instruction.UnaryLogic(instr)
    if isinstance(instr, quil_rs.Comparison):
        return quil_rs.Instruction.Comparison(instr)
    raise ValueError(f"{type(instr)} is not an Instruction")


def _convert_to_rs_instructions(instrs: Iterable[AbstractInstruction]) -> list[quil_rs.Instruction]:
    return [_convert_to_rs_instruction(instr) for instr in instrs]


def _convert_to_py_instruction(instr: Any) -> AbstractInstruction:
    if isinstance(instr, quil_rs.Instruction.Nop):
        return Nop()
    if isinstance(instr, quil_rs.Instruction.Halt):
        return Halt()
    if isinstance(instr, quil_rs.Instruction.Wait):
        return Wait()
    if isinstance(instr, quil_rs.Instruction):
        return _convert_to_py_instruction(instr._0)
    if isinstance(instr, quil_rs.Arithmetic):
        return ArithmeticBinaryOp._from_rs_arithmetic(instr)
    if isinstance(instr, quil_rs.BinaryLogic):
        return LogicalBinaryOp._from_rs_binary_logic(instr)
    if isinstance(instr, quil_rs.CalibrationDefinition):
        return DefCalibration._from_rs_calibration(instr)
    if isinstance(instr, quil_rs.Call):
        return Call._from_rs_call(instr)
    if isinstance(instr, quil_rs.Capture):
        return Capture._from_rs_capture(instr)
    if isinstance(instr, quil_rs.CircuitDefinition):
        return DefCircuit._from_rs_circuit_definition(instr)
    if isinstance(instr, quil_rs.Convert):
        return ClassicalConvert._from_rs_convert(instr)
    if isinstance(instr, quil_rs.Comparison):
        return ClassicalComparison._from_rs_comparison(instr)
    if isinstance(instr, quil_rs.Declaration):
        return Declare._from_rs_declaration(instr)
    if isinstance(instr, quil_rs.Delay):
        if len(instr.frame_names) > 0:
            return DelayFrames._from_rs_delay(instr)
        if len(instr.qubits) > 0:
            return DelayQubits._from_rs_delay(instr)
        return Delay._from_rs_delay(instr)
    if isinstance(instr, quil_rs.Exchange):
        return ClassicalExchange._from_rs_exchange(instr)
    if isinstance(instr, quil_rs.Fence):
        if len(instr.qubits) == 0:
            return FenceAll()
        return Fence._from_rs_fence(instr)
    if isinstance(instr, quil_rs.FrameDefinition):
        return DefFrame._from_rs_frame_definition(instr)
    if isinstance(instr, quil_rs.GateDefinition):
        return DefGate._from_rs_gate_definition(instr)
    if isinstance(instr, quil_rs.Gate):
        return Gate._from_rs_gate(instr)
    if isinstance(instr, quil_rs.Include):
        return Include._from_rs_include(instr)
    if isinstance(instr, quil_rs.Jump):
        return Jump._from_rs_jump(instr)
    if isinstance(instr, quil_rs.JumpWhen):
        return JumpWhen._from_rs_jump_when(instr)
    if isinstance(instr, quil_rs.JumpUnless):
        return JumpUnless._from_rs_jump_unless(instr)
    if isinstance(instr, quil_rs.Label):
        return JumpTarget._from_rs_label(instr)
    if isinstance(instr, quil_rs.Load):
        return ClassicalLoad._from_rs_load(instr)
    if isinstance(instr, quil_rs.MeasureCalibrationDefinition):
        return DefMeasureCalibration._from_rs_measure_calibration_definition(instr)
    if isinstance(instr, quil_rs.Measurement):
        return Measurement._from_rs_measurement(instr)
    if isinstance(instr, quil_rs.Move):
        return ClassicalMove._from_rs_move(instr)
    if isinstance(instr, quil_rs.Pragma):
        return Pragma._from_rs_pragma(instr)
    if isinstance(instr, quil_rs.Pulse):
        return Pulse._from_rs_pulse(instr)
    if isinstance(instr, quil_rs.RawCapture):
        return RawCapture._from_rs_raw_capture(instr)
    if isinstance(instr, quil_rs.Reset):
        if instr.qubit is None:
            return Reset._from_rs_reset(instr)
        else:
            return ResetQubit._from_rs_reset(instr)
    if isinstance(instr, quil_rs.SetFrequency):
        return SetFrequency._from_rs_set_frequency(instr)
    if isinstance(instr, quil_rs.SetPhase):
        return SetPhase._from_rs_set_phase(instr)
    if isinstance(instr, quil_rs.SetScale):
        return SetScale._from_rs_set_scale(instr)
    if isinstance(instr, quil_rs.ShiftFrequency):
        return ShiftFrequency._from_rs_shift_frequency(instr)
    if isinstance(instr, quil_rs.ShiftPhase):
        return ShiftPhase._from_rs_shift_phase(instr)
    if isinstance(instr, quil_rs.SwapPhases):
        return SwapPhases._from_rs_swap_phases(instr)
    if isinstance(instr, quil_rs.Store):
        return ClassicalStore._from_rs_store(instr)
    if isinstance(instr, quil_rs.UnaryLogic):
        return UnaryClassicalInstruction._from_rs_unary_logic(instr)
    if isinstance(instr, quil_rs.WaveformDefinition):
        return DefWaveform._from_rs_waveform_definition(instr)
    elif isinstance(instr, AbstractInstruction):
        return instr
    raise ValueError(f"{type(instr)} is not a valid Instruction type")


def _convert_to_py_instructions(instrs: Iterable[quil_rs.Instruction]) -> list[AbstractInstruction]:
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


class Gate(quil_rs.Gate, AbstractInstruction):
    """A quantum gate instruction."""

    def __new__(
        cls,
        name: str,
        params: Sequence[ParameterDesignator],
        qubits: Sequence[Union[Qubit, QubitPlaceholder, FormalArgument, int]],
        modifiers: Union[Sequence[Union[quil_rs.GateModifier, str]], None] = None,
    ) -> Self:
        """Initialize a new gate instruction."""
        if modifiers is None:
            modifiers = []
        return super().__new__(
            cls,
            name,
            _convert_to_rs_expressions(params),
            _convert_to_rs_qubits(qubits),
            [cls._to_rs_gate_modifier(modifier) if isinstance(modifier, str) else modifier for modifier in modifiers],
        )

    def __getnewargs__(self) -> tuple[str, Sequence[ParameterDesignator], list[QubitDesignator], list[str]]:
        return self.name, self.params, self.qubits, self.modifiers

    @classmethod
    def _from_rs_gate(cls, gate: quil_rs.Gate) -> Self:
        return super().__new__(cls, gate.name, gate.parameters, gate.qubits, gate.modifiers)

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Sequence[QubitDesignator]:
        """Get the qubits the gate operates on."""
        if indices:
            return self.get_qubit_indices()
        else:
            return _convert_to_py_qubits(super().qubits)

    @property
    def qubits(self) -> list[QubitDesignator]:
        """Get the qubits the gate operates on."""
        return self.get_qubits(indices=False)  # type: ignore

    @property
    def params(self) -> Sequence[ParameterDesignator]:
        """Get the parameters the gate operates on."""
        return _convert_to_py_expressions(super().parameters)

    @property
    def modifiers(self) -> list[str]:
        """Get all modifiers applied to the gate."""
        return [str(modifier).upper() for modifier in super().modifiers]

    @staticmethod
    def _to_rs_gate_modifier(modifier: str) -> quil_rs.GateModifier:
        modifier = modifier.upper()
        if modifier == "CONTROLLED":
            return quil_rs.GateModifier.CONTROLLED
        if modifier == "DAGGER":
            return quil_rs.GateModifier.DAGGER
        if modifier == "FORKED":
            return quil_rs.GateModifier.FORKED
        raise ValueError(f"{modifier} is not a valid Gate modifier.")

    def get_qubit_indices(self) -> list[int]:
        """Get the qubits the gate operates on, as integer indices."""
        return [qubit._0 for qubit in super().qubits]

    def controlled(
        self,
        control_qubit: Union[
            quil_rs.Qubit,
            QubitDesignator,
            Sequence[Union[QubitDesignator, quil_rs.Qubit]],
        ],
    ) -> "Gate":
        """Return a new `Gate`, derived from this one, with the CONTROLLED modifier added
        with the given control qubit or Sequence of control qubits.
        """
        if isinstance(control_qubit, Sequence):
            res = super()
            for qubit in control_qubit:
                res = res.controlled(_convert_to_rs_qubit(qubit))
            res = self._from_rs_gate(res)
        else:
            res = self._from_rs_gate(super().controlled(_convert_to_rs_qubit(control_qubit)))

        return res

    def forked(
        self,
        fork_qubit: Union[quil_rs.Qubit, QubitDesignator],
        alt_params: Union[Sequence[ParameterDesignator], Sequence[quil_rs_expr.Expression]],
    ) -> "Gate":
        """Return a new `Gate`, derived from this one, with the FORKED modifier added
        with the given fork qubit and additional parameters.
        """
        return self._from_rs_gate(
            super().forked(_convert_to_rs_qubit(fork_qubit), _convert_to_rs_expressions(alt_params))
        )

    def dagger(self) -> "Gate":
        """Return a new `Gate`, derived from this one, with the DAGGER modifier added."""
        return self._from_rs_gate(super().dagger())

    def out(self) -> str:
        """Return the Gate as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


def _strip_modifiers(gate: Gate, limit: Optional[int] = None) -> Gate:
    """Remove modifiers from :py:class:`Gate`.

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
            raise TypeError(f"Unsupported gate modifier {m}")

    stripped = Gate(gate.name, gate.params[:param_index], gate.qubits[qubit_index:], gate.modifiers[limit:])
    return stripped


class Measurement(quil_rs.Measurement, AbstractInstruction):
    """A Quil measurement instruction."""

    def __new__(
        cls,
        qubit: QubitDesignator,
        classical_reg: Optional[MemoryReference],
    ) -> Self:
        """Initialize a new measurement instruction."""
        target = cls._reg_to_target(classical_reg)
        return super().__new__(cls, _convert_to_rs_qubit(qubit), target)

    @classmethod
    def _reg_to_target(cls, classical_reg: Optional[MemoryReference]) -> Optional[quil_rs.MemoryReference]:
        if isinstance(classical_reg, quil_rs.MemoryReference):
            return classical_reg

        if classical_reg is not None:
            return quil_rs.MemoryReference.parse(str(classical_reg))

        return None

    @classmethod
    def _from_rs_measurement(cls, measurement: quil_rs.Measurement) -> "Measurement":
        return super().__new__(cls, measurement.qubit, measurement.target)

    @property  # type: ignore[override]
    def qubit(self) -> QubitDesignator:
        """Get the qubit the instruction measures."""
        return _convert_to_py_qubit(super().qubit)

    def with_qubit(self, qubit: QubitDesignator) -> Self:
        return self.__new__(qubit, self.target)

    @property
    def classical_reg(self) -> Optional[MemoryReference]:
        """Get the MemoryReference that this instruction writes to, if any."""
        target = super().target
        if target is None:
            return None
        return MemoryReference._from_rs_memory_reference(target)

    def with_classical_reg(self, classical_reg: Optional[MemoryReference]) -> Self:
        return self.__new__(self.qubit, classical_reg)

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[set[QubitDesignator], set[int]]:
        """Get the qubit this instruction measures."""
        if indices:
            return self.get_qubit_indices()
        else:
            return {_convert_to_py_qubit(super().qubit)}

    def get_qubit_indices(self) -> set[int]:
        """Get the qubit this instruction measures, as an integer index."""
        return {super().qubit._0}

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class Reset(quil_rs.Reset, AbstractInstruction):
    """The RESET instruction."""

    def __new__(cls, qubit: Optional[Union[Qubit, QubitPlaceholder, FormalArgument, int]] = None) -> Self:
        """Initialize a new reset instruction."""
        rs_qubit: Optional[quil_rs.Qubit] = None
        if qubit is not None:
            rs_qubit = _convert_to_rs_qubit(qubit)
        return super().__new__(cls, rs_qubit)

    @classmethod
    def _from_rs_reset(cls, reset: quil_rs.Reset) -> Self:
        return Reset.__new__(cls, reset.qubit)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Optional[set[QubitDesignator]]:
        """Get the qubit this instruction resets."""
        if super().qubit is None:
            return None
        if indices:
            return self.get_qubit_indices()  # type: ignore
        return {_convert_to_py_qubit(super().qubit)}  # type: ignore

    def get_qubit_indices(self) -> Optional[set[int]]:
        """Get the qubit this instruction resets, as an integer index."""
        if super().qubit is None:
            return None
        return {super().qubit._0}  # type: ignore

    @property  # type: ignore[override]
    def qubit(self) -> Optional[QubitDesignator]:
        """Get the qubit this instruction resets, if any."""
        if super().qubit:
            return _convert_to_py_qubit(super().qubit)  # type: ignore
        return None

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ResetQubit(Reset):
    """A targeted RESET instruction."""

    def __new__(cls, qubit: Union[Qubit, QubitPlaceholder, FormalArgument, int]) -> Self:
        """Initialize a new reset instruction, with a target qubit."""
        if qubit is None:
            raise TypeError("qubit should not be None")
        return super().__new__(cls, qubit)

    @classmethod
    def _from_rs_reset(cls, reset: quil_rs.Reset) -> Self:
        if reset.qubit is not None:
            qubit = _convert_to_py_qubit(reset.qubit)
            return ResetQubit.__new__(cls, qubit)
        raise ValueError("reset.qubit should not be None")


class DefGate(quil_rs.GateDefinition, AbstractInstruction):
    """A DEFGATE directive."""

    def __new__(
        cls,
        name: str,
        matrix: Union[list[list[Expression]], np.ndarray, np.matrix],
        parameters: Optional[list[Parameter]] = None,
    ) -> Self:
        """Initialize a new gate definition.

        :param name: The name of the newly defined gate.
        :param matrix: The matrix defining this gate.
        :param parameters: list of parameters that are used in this gate
        """
        DefGate._validate_matrix(matrix, parameters is not None and len(parameters) > 0)
        specification = DefGate._convert_to_matrix_specification(matrix)
        rs_parameters = [param.name for param in parameters or []]
        return super().__new__(cls, name, rs_parameters, specification)

    def __getnewargs__(self) -> tuple:
        return self.name, self.matrix, self.parameters

    @classmethod
    def _from_rs_gate_definition(cls, gate_definition: quil_rs.GateDefinition) -> Self:
        return super().__new__(
            cls,
            gate_definition.name,
            gate_definition.parameters,
            gate_definition.specification,
        )

    @staticmethod
    def _convert_to_matrix_specification(
        matrix: Union[list[list[Expression]], np.ndarray, np.matrix],
    ) -> quil_rs.GateSpecification:
        to_rs_matrix = np.vectorize(_convert_to_rs_expression, otypes=["O"])
        return quil_rs.GateSpecification.Matrix(to_rs_matrix(np.asarray(matrix)))

    @staticmethod
    def _validate_matrix(
        matrix: Union[list[list[Expression]], np.ndarray, np.matrix], contains_parameters: bool
    ) -> None:
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
            raise ValueError(f"Dimension of matrix must be a power of 2, got {rows}")

        if not contains_parameters:
            np_matrix = np.asarray(matrix, dtype=complex)
            cmp = np_matrix.dot(np_matrix.T.conj())
            is_unitary = np.allclose(np.eye(rows, dtype=complex), cmp)
            if not is_unitary:
                raise ValueError("Matrix must be unitary.")

    def out(self) -> str:
        """Return the Gate as a valid Quil string."""
        return super().to_quil()

    def get_constructor(self) -> Union[Callable[..., Gate], Callable[..., Callable[..., Gate]]]:
        """Return a function that constructs this gate on variable qubit indices.

        For example, `mygate.get_constructor()(1) applies the gate to qubit 1.`
        """
        if self.parameters:
            return lambda *params: lambda *qubits: Gate(
                name=self.name, params=list(params), qubits=list(map(unpack_qubit, qubits))
            )
        else:
            return lambda *qubits: Gate(name=self.name, params=[], qubits=list(map(unpack_qubit, qubits)))

    def num_args(self) -> int:
        """Get the number of qubit arguments the gate takes."""
        rows = len(self.matrix)
        return int(np.log2(rows))

    @property
    def matrix(self) -> np.ndarray:
        """Get the matrix that defines this GateDefinition."""
        to_py_matrix = np.vectorize(_convert_to_py_expression, otypes=["O"])
        return to_py_matrix(np.asarray(super().specification._0))  # type: ignore[no-any-return]

    @property  # type: ignore[override]
    def parameters(self) -> list[Parameter]:
        """Get the parameters this gate definition takes."""
        return [Parameter(name) for name in super().parameters]

    @parameters.setter  # type: ignore[override]
    def parameters(self, parameters: Optional[list[Parameter]]) -> None:
        quil_rs.GateDefinition.parameters.__set__(self, [param.name for param in parameters or []])  # type: ignore[attr-defined] # noqa

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class DefPermutationGate(DefGate):
    """A gate defined by a permutation of numbers."""

    def __new__(cls, name: str, permutation: Union[list[int], np.ndarray]) -> Self:
        """Initialize a new gate definition with a permutation."""
        specification = DefPermutationGate._convert_to_permutation_specification(permutation)
        gate_definition = quil_rs.GateDefinition(name, [], specification)
        return super()._from_rs_gate_definition(gate_definition)

    @staticmethod
    def _convert_to_permutation_specification(permutation: Union[list[int], np.ndarray]) -> quil_rs.GateSpecification:
        return quil_rs.GateSpecification.Permutation([int(x) for x in permutation])

    @property
    def permutation(self) -> list[int]:
        """Get the permutation that defines the gate."""
        return super().specification._0

    def num_args(self) -> int:
        """Get the number of arguments the gate takes."""
        return int(np.log2(len(self.permutation)))

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class DefGateByPaulis(DefGate):
    """A gate definition defined by the exponentiation of a PauliSum."""

    def __new__(
        cls,
        gate_name: str,
        parameters: list[Parameter],
        arguments: list[QubitDesignator],
        body: "PauliSum",
    ) -> Self:
        """Initialize a new gate, defined by a PauliSum."""
        specification = DefGateByPaulis._convert_to_pauli_specification(body, arguments)
        rs_parameters = [param.name for param in parameters]
        gate_definition = quil_rs.GateDefinition(gate_name, rs_parameters, specification)
        return super()._from_rs_gate_definition(gate_definition)

    @staticmethod
    def _convert_to_pauli_specification(
        body: "PauliSum", arguments: list[QubitDesignator]
    ) -> quil_rs.GateSpecification:
        if isinstance(body, Sequence):
            from pyquil.paulis import PauliSum

            body = PauliSum(body)
        return quil_rs.GateSpecification.PauliSum(body._to_rs_pauli_sum(arguments))

    @property
    def arguments(self) -> list[FormalArgument]:
        """Get the arguments the gate takes."""
        return [FormalArgument(arg) for arg in super().specification._0.arguments]

    @property
    def body(self) -> "PauliSum":
        """Get the PauliSum that defines the gate."""
        from pyquil.paulis import PauliSum  # avoids circular import

        return PauliSum._from_rs_pauli_sum(super().specification._0)

    def num_args(self) -> int:
        """Get the number of arguments the gate takes."""
        return len(self.arguments)

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class JumpTarget(quil_rs.Label, AbstractInstruction):
    """Representation of a target that can be jumped to."""

    def __new__(cls, label: Union[Label, LabelPlaceholder]) -> Self:
        """Initialize a new target."""
        return super().__new__(cls, label.target)

    @classmethod
    def _from_rs_label(cls, label: quil_rs.Label) -> "JumpTarget":
        return super().__new__(cls, label.target)

    @property
    def label(self) -> Union[Label, LabelPlaceholder]:
        """Get the target as a label."""
        if isinstance(self.target, quil_rs.Target.Placeholder):
            return LabelPlaceholder._from_rs_target(self.target)
        return Label._from_rs_target(self.target)

    def __repr__(self) -> str:
        return f"<JumpTarget {str(self.label)}>"

    def out(self) -> str:
        """Return the target as valid Quil. Raises an error if the target is an unresolved placeholder."""
        return super().to_quil()


class JumpWhen(quil_rs.JumpWhen, AbstractInstruction):
    """The JUMP-WHEN instruction."""

    def __new__(cls, target: Union[Label, LabelPlaceholder], condition: MemoryReference) -> Self:
        """Initialize a new JumpWhen instruction.

        :param target: The target to jump to if the condition is true.
        :param condition: A memory reference that determines if the jump should be performed. The memory reference must
            refer to an INTEGER or BIT. The jump will be performed if the value in the reference is not 0 when the
            instruction is evaluated.
        """
        return super().__new__(cls, target.target, condition._to_rs_memory_reference())

    @classmethod
    def _from_rs_jump_when(cls, jump_when: quil_rs.JumpWhen) -> Self:
        return super().__new__(cls, jump_when.target, jump_when.condition)

    def out(self) -> str:
        """Return the instruction as valid Quil. Raises an error if the target is an unresolved placeholder."""
        return super().to_quil()

    @property  # type: ignore[override]
    def condition(self) -> MemoryReference:
        """Get the MemoryReference the instruction uses to determine if the jump should be performed or not."""
        return MemoryReference._from_rs_memory_reference(super().condition)

    @condition.setter
    def condition(self, condition: MemoryReference) -> None:
        quil_rs.JumpWhen.condition.__set__(self, condition._to_rs_memory_reference())  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def target(self) -> Union[Label, LabelPlaceholder]:
        """Get the target the instruction will jump to if the condition bit is not 1."""
        if isinstance(super().target, quil_rs.Target.Placeholder):
            return LabelPlaceholder._from_rs_target(super().target)
        return Label._from_rs_target(super().target)

    @target.setter
    def target(self, target: Union[Label, LabelPlaceholder]) -> None:
        quil_rs.JumpWhen.target.__set__(self, target)  # type: ignore[attr-defined]

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class JumpUnless(quil_rs.JumpUnless, AbstractInstruction):
    """The JUMP-UNLESS instruction."""

    def __new__(cls, target: Union[Label, LabelPlaceholder], condition: MemoryReference) -> Self:
        """Initialize a new JumpUnless instruction.

        :param target: The target to jump to if the condition is true.
        :param condition: A memory reference that determines if the jump should be performed. The memory reference must
            refer to an INTEGER or BIT. The jump will be performed if the value in the reference is 0 when the
            instruction is evaluated.
        """
        return super().__new__(cls, target.target, condition._to_rs_memory_reference())

    def __getnewargs__(self) -> tuple[Label | LabelPlaceholder, MemoryReference]:
        return self.target, self.condition

    @classmethod
    def _from_rs_jump_unless(cls, jump_unless: quil_rs.JumpUnless) -> Self:
        return super().__new__(cls, jump_unless.target, jump_unless.condition)

    def out(self) -> str:
        """Return the instruction as valid Quil. Raises an error if the target is an unresolved placeholder."""
        return super().to_quil()

    @property  # type: ignore[override]
    def condition(self) -> MemoryReference:
        """Get the MemoryReference the instruction uses to determine if the jump should be performed or not."""
        return MemoryReference._from_rs_memory_reference(super().condition)

    @condition.setter
    def condition(self, condition: MemoryReference) -> None:
        quil_rs.JumpUnless.condition.__set__(self, condition._to_rs_memory_reference())  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def target(self) -> Union[Label, LabelPlaceholder]:
        """Get the target the instruction will jump to if the condition bit is not 1."""
        if isinstance(super().target, quil_rs.Target.Placeholder):
            return LabelPlaceholder._from_rs_target(super().target)
        return Label._from_rs_target(super().target)

    @target.setter
    def target(self, target: Union[Label, LabelPlaceholder]) -> None:
        quil_rs.JumpUnless.target.__set__(self, target)  # type: ignore[attr-defined]

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class SimpleInstruction(AbstractInstruction):
    """Base class for simple instructions with no arguments."""

    instruction: ClassVar[quil_rs.Instruction]

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return self.instruction.to_quil()

    def __str__(self) -> str:
        return self.instruction.to_quil_or_debug()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SimpleInstruction):
            return self.instruction == other.instruction
        return NotImplemented

    def __hash__(self) -> int:
        return hash(str(self.instruction))


class Halt(SimpleInstruction):
    """The HALT instruction."""

    instruction = quil_rs.Instruction.Halt()


class Wait(SimpleInstruction):
    """The WAIT instruction."""

    instruction = quil_rs.Instruction.Wait()


class Nop(SimpleInstruction):
    """The NOP instruction."""

    instruction = quil_rs.Instruction.Nop()


class UnaryClassicalInstruction(quil_rs.UnaryLogic, AbstractInstruction):
    """Base class for unary classical instructions."""

    op: ClassVar[quil_rs.UnaryOperator]

    def __new__(cls, target: MemoryReference) -> "UnaryClassicalInstruction":
        """Initialize a new unary classical instruction."""
        return super().__new__(cls, cls.op, target._to_rs_memory_reference())

    def __getnewargs__(self) -> tuple[MemoryReference,]:
        return (self.target,)

    @classmethod
    def _from_rs_unary_logic(cls, unary_logic: quil_rs.UnaryLogic) -> Self:
        return super().__new__(cls, unary_logic.operator, unary_logic.operand)

    @property
    def target(self) -> MemoryReference:
        """The MemoryReference that the instruction operates on."""
        return MemoryReference._from_rs_memory_reference(super().operand)

    def out(self) -> str:
        """Return the instruction as a valid Quil string. Raises an error if the instruction contains placeholders."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalNeg(UnaryClassicalInstruction):
    """The NEG instruction."""

    op = quil_rs.UnaryOperator.NEG


class ClassicalNot(UnaryClassicalInstruction):
    """The NOT instruction."""

    op = quil_rs.UnaryOperator.NOT


class LogicalBinaryOp(quil_rs.BinaryLogic, AbstractInstruction):
    """Base class for binary logical classical instructions."""

    op: ClassVar[quil_rs.BinaryOperator]

    def __new__(cls, left: MemoryReference, right: Union[MemoryReference, int]) -> Self:
        """Initialize the operands of the binary logical instruction."""
        destination = left._to_rs_memory_reference()
        source = cls._to_rs_binary_operand(right)
        return super().__new__(cls, cls.op, destination, source)

    def __getnewargs__(self) -> tuple[MemoryReference, MemoryReference | int]:
        return self.left, self.right

    @classmethod
    def _from_rs_binary_logic(cls, binary_logic: quil_rs.BinaryLogic) -> "LogicalBinaryOp":
        return super().__new__(cls, binary_logic.operator, binary_logic.destination, binary_logic.source)

    @staticmethod
    def _to_rs_binary_operand(operand: Union[MemoryReference, int]) -> quil_rs.BinaryOperand:
        if isinstance(operand, MemoryReference):
            return quil_rs.BinaryOperand.MemoryReference(operand._to_rs_memory_reference())
        return quil_rs.BinaryOperand.LiteralInteger(operand)

    @staticmethod
    def _to_py_binary_operand(operand: quil_rs.BinaryOperand) -> Union[MemoryReference, int]:
        match operand:
            case quil_rs.BinaryOperand.LiteralInteger(i):
                return i
            case quil_rs.BinaryOperand.MemoryReference(ref):
                return MemoryReference._from_rs_memory_reference(ref)
        raise ValueError(f"not a valid operand: {operand}")

    @property
    def left(self) -> MemoryReference:
        """The left hand side of the binary expression."""
        return MemoryReference._from_rs_memory_reference(super().destination)

    @property
    def right(self) -> Union[MemoryReference, int]:
        """The right hand side of the binary expression."""
        return self._to_py_binary_operand(super().source)

    def out(self) -> str:
        """Return the instruction as a valid Quil string. Raises an error if the instruction contains placeholders."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalAnd(LogicalBinaryOp):
    """The AND instruction."""

    op = quil_rs.BinaryOperator.AND


class ClassicalInclusiveOr(LogicalBinaryOp):
    """The IOR instruction."""

    op = quil_rs.BinaryOperator.IOR


class ClassicalExclusiveOr(LogicalBinaryOp):
    """The XOR instruction."""

    op = quil_rs.BinaryOperator.XOR


class ArithmeticBinaryOp(quil_rs.Arithmetic, AbstractInstruction):
    """Base class for binary arithmetic classical instructions."""

    op: ClassVar[quil_rs.ArithmeticOperator]

    def __new__(cls, left: MemoryReference, right: Union[MemoryReference, int, float]) -> Self:
        """Initialize the operands of the binary arithmetic instruction."""
        right_operand = _to_rs_arithmetic_operand(right)
        return super().__new__(cls, cls.op, left._to_rs_memory_reference(), right_operand)

    def __getnewargs__(self) -> tuple[MemoryReference, MemoryReference | int | float]:
        return self.left, self.right

    @classmethod
    def _from_rs_arithmetic(cls, arithmetic: quil_rs.Arithmetic) -> "ArithmeticBinaryOp":
        return super().__new__(cls, arithmetic.operator, arithmetic.destination, arithmetic.source)

    @property
    def left(self) -> MemoryReference:
        """The left hand side of the binary expression."""
        return MemoryReference._from_rs_memory_reference(super().destination)

    @property
    def right(self) -> Union[MemoryReference, int, float]:
        """The left hand side of the binary expression."""
        return _to_py_arithmetic_operand(super().source)

    def out(self) -> str:
        """Return the instruction as a valid Quil string. Raises an error if the instruction contains placeholders."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalAdd(ArithmeticBinaryOp):
    """The ADD instruction."""

    op = quil_rs.ArithmeticOperator.ADD


class ClassicalSub(ArithmeticBinaryOp):
    """The SUB instruction."""

    op = quil_rs.ArithmeticOperator.SUBTRACT


class ClassicalMul(ArithmeticBinaryOp):
    """The MUL instruction."""

    op = quil_rs.ArithmeticOperator.MULTIPLY


class ClassicalDiv(ArithmeticBinaryOp):
    """The DIV instruction."""

    op = quil_rs.ArithmeticOperator.DIVIDE


class ClassicalMove(quil_rs.Move, AbstractInstruction):
    """The MOVE instruction."""

    def __new__(cls, left: MemoryReference, right: Union[MemoryReference, int, float]) -> "ClassicalMove":
        """Initialize a new MOVE instruction."""
        return super().__new__(cls, left._to_rs_memory_reference(), _to_rs_arithmetic_operand(right))

    def __getnewargs__(self) -> tuple[MemoryReference, MemoryReference | int | float]:
        return self.left, self.right

    @classmethod
    def _from_rs_move(cls, move: quil_rs.Move) -> Self:
        return super().__new__(cls, move.destination, move.source)

    @property
    def left(self) -> MemoryReference:
        """The left hand side (or "destination") of the move instruction."""
        return MemoryReference._from_rs_memory_reference(super().destination)

    @property
    def right(self) -> Union[MemoryReference, int, float]:
        """The right hand side (or "source") of the move instruction."""
        return _to_py_arithmetic_operand(super().source)

    def out(self) -> str:
        """Return the move instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalExchange(quil_rs.Exchange, AbstractInstruction):
    """The EXCHANGE instruction."""

    def __new__(
        cls,
        left: MemoryReference,
        right: MemoryReference,
    ) -> "ClassicalExchange":
        """Initialize a new EXCHANGE instruction."""
        return super().__new__(cls, left._to_rs_memory_reference(), right._to_rs_memory_reference())

    def __getnewargs__(self) -> tuple[MemoryReference, MemoryReference]:
        return self.left, self.right

    @classmethod
    def _from_rs_exchange(cls, exchange: quil_rs.Exchange) -> Self:
        return super().__new__(cls, exchange.left, exchange.right)

    @property  # type: ignore[override]
    def left(self) -> MemoryReference:
        """The left hand side of the exchange instruction."""
        return MemoryReference._from_rs_memory_reference(super().left)

    @property  # type: ignore[override]
    def right(self) -> MemoryReference:
        """The left hand side of the exchange instruction."""
        return MemoryReference._from_rs_memory_reference(super().right)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalConvert(quil_rs.Convert, AbstractInstruction):
    """The CONVERT instruction."""

    def __new__(cls, left: MemoryReference, right: MemoryReference) -> "ClassicalConvert":
        """Initialize a new CONVERT instruction."""
        return super().__new__(cls, left._to_rs_memory_reference(), right._to_rs_memory_reference())

    def __getnewargs__(self) -> tuple[MemoryReference, MemoryReference]:
        return self.left, self.right

    @classmethod
    def _from_rs_convert(cls, convert: quil_rs.Convert) -> Self:
        return super().__new__(cls, convert.destination, convert.source)

    @property
    def left(self) -> MemoryReference:
        """Return the left hand side (or "destination") of the conversion instruction."""
        return MemoryReference._from_rs_memory_reference(super().destination)

    @property
    def right(self) -> MemoryReference:
        """Return the right hand side (or "source") of the conversion instruction."""
        return MemoryReference._from_rs_memory_reference(super().source)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalLoad(quil_rs.Load, AbstractInstruction):
    """The LOAD instruction."""

    def __new__(cls, target: MemoryReference, left: str, right: MemoryReference) -> "ClassicalLoad":
        """Initialize a new LOAD instruction."""
        return super().__new__(cls, target._to_rs_memory_reference(), left, right._to_rs_memory_reference())

    def __getnewargs__(self) -> tuple[MemoryReference, str, MemoryReference]:
        return self.target, self.left, self.right

    @classmethod
    def _from_rs_load(cls, load: quil_rs.Load) -> Self:
        return super().__new__(cls, load.destination, load.source, load.offset)

    @property
    def target(self) -> MemoryReference:
        """The MemoryReference that the instruction loads into."""
        return MemoryReference._from_rs_memory_reference(super().destination)

    @property
    def left(self) -> str:
        """The left hand side of the LOAD instruction."""
        return super().source

    @property
    def right(self) -> MemoryReference:
        """The right hand side of the LOAD instruction."""
        return MemoryReference._from_rs_memory_reference(super().offset)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


def _to_rs_arithmetic_operand(operand: Union[MemoryReference, int, float]) -> quil_rs.ArithmeticOperand:
    if isinstance(operand, MemoryReference):
        return quil_rs.ArithmeticOperand.MemoryReference(operand._to_rs_memory_reference())
    if isinstance(operand, int):
        return quil_rs.ArithmeticOperand.LiteralInteger(operand)
    if isinstance(operand, float):
        return quil_rs.ArithmeticOperand.LiteralReal(operand)
    raise TypeError(f"{type(operand)} is not a valid ArithmeticOperand")


def _to_py_arithmetic_operand(operand: quil_rs.ArithmeticOperand) -> Union[MemoryReference, int, float]:
    match operand:
        case quil_rs.ArithmeticOperand.MemoryReference(ref):
            return MemoryReference._from_rs_memory_reference(ref)
        case quil_rs.ArithmeticOperand.LiteralInteger(x) | quil_rs.ArithmeticOperand.LiteralReal(x):
            return x
    raise TypeError(f"{type(operand)} is not an ArithmeticOperand")


class ClassicalStore(quil_rs.Store, AbstractInstruction):
    """The STORE instruction."""

    def __new__(cls, target: str, left: MemoryReference, right: Union[MemoryReference, int, float]) -> "ClassicalStore":
        """Initialize a new STORE instruction."""
        rs_right = _to_rs_arithmetic_operand(right)
        return super().__new__(cls, target, left._to_rs_memory_reference(), rs_right)

    def __getnewargs__(self) -> tuple[str, MemoryReference, MemoryReference | int | float]:
        return self.target, self.left, self.right

    @classmethod
    def _from_rs_store(cls, store: quil_rs.Store) -> Self:
        return super().__new__(cls, store.destination, store.offset, store.source)

    @property
    def target(self) -> str:
        """The target of the STORE instruction."""
        return super().destination

    @property
    def left(self) -> MemoryReference:
        """The left hand side of the STORE instruction."""
        return MemoryReference._from_rs_memory_reference(super().offset)

    @property
    def right(self) -> Union[MemoryReference, int, float]:
        """The left hand side of the STORE instruction."""
        return _to_py_arithmetic_operand(super().source)

    def out(self) -> str:
        """Return a valid Quil string representation of the instruction."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalComparison(quil_rs.Comparison, AbstractInstruction):
    """Base class for ternary comparison instructions."""

    op: ClassVar[quil_rs.ComparisonOperator]

    def __new__(
        cls,
        target: MemoryReference,
        left: MemoryReference,
        right: Union[MemoryReference, int, float],
    ) -> "ClassicalComparison":
        """Initialize a new comparison instruction."""
        rs_target, rs_left, rs_right = (
            target._to_rs_memory_reference(),
            left._to_rs_memory_reference(),
            cls._to_comparison_operand(right),
        )
        return super().__new__(cls, cls.op, rs_target, rs_left, rs_right)

    def __getnewargs__(self) -> tuple[MemoryReference, int, float]:
        return self.target, self.left, self.right

    @classmethod
    def _from_rs_comparison(cls, comparison: quil_rs.Comparison) -> Self:
        return super().__new__(cls, comparison.operator, comparison.destination, comparison.lhs, comparison.rhs)

    @staticmethod
    def _to_comparison_operand(operand: Union[MemoryReference, int, float]) -> quil_rs.ComparisonOperand:
        if isinstance(operand, MemoryReference):
            return quil_rs.ComparisonOperand.MemoryReference(operand._to_rs_memory_reference())
        elif isinstance(operand, int):
            return quil_rs.ComparisonOperand.LiteralInteger(operand)
        elif isinstance(operand, float):
            return quil_rs.ComparisonOperand.LiteralReal(operand)
        raise TypeError(f"{type(operand)} is not a valid ComparisonOperand")

    @staticmethod
    def _to_py_operand(operand: quil_rs.ComparisonOperand) -> Union[MemoryReference, int, float]:
        match operand:
            case quil_rs.ComparisonOperand.MemoryReference(ref):
                return MemoryReference._from_rs_memory_reference(ref)
            case quil_rs.ComparisonOperand.LiteralInteger(x) | quil_rs.ComparisonOperand.LiteralReal(x):
                return x
        raise TypeError(f"{type(operand)} is not an ComparisonOperand")

    @property
    def target(self) -> MemoryReference:
        """The target of the comparison."""
        return MemoryReference._from_rs_memory_reference(super().destination)

    @property
    def left(self) -> MemoryReference:
        """The left hand side of the comparison."""
        return MemoryReference._from_rs_memory_reference(super().lhs)

    @property
    def right(self) -> Union[MemoryReference, int, float]:
        """The right hand side of the comparison."""
        return self._to_py_operand(super().rhs)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalEqual(ClassicalComparison):
    """The EQ comparison instruction."""

    op = quil_rs.ComparisonOperator.EQUAL


class ClassicalLessThan(ClassicalComparison):
    """The LT comparison instruction."""

    op = quil_rs.ComparisonOperator.LESS_THAN


class ClassicalLessEqual(ClassicalComparison):
    """The LE comparison instruction."""

    op = quil_rs.ComparisonOperator.LESS_THAN_OR_EQUAL


class ClassicalGreaterThan(ClassicalComparison):
    """The GT comparison instruction."""

    op = quil_rs.ComparisonOperator.GREATER_THAN


class ClassicalGreaterEqual(ClassicalComparison):
    """The GE comparison instruction."""

    op = quil_rs.ComparisonOperator.GREATER_THAN_OR_EQUAL


class Jump(quil_rs.Jump, AbstractInstruction):
    """Representation of an unconditional jump instruction (JUMP)."""

    def __new__(cls, target: Union[Label, LabelPlaceholder]) -> Self:
        """Initialize a new jump instruction."""
        return super().__new__(cls, target.target)

    def __getnewargs__(self) -> tuple:
        return self.target

    @classmethod
    def _from_rs_jump(cls, jump: quil_rs.Jump) -> Self:
        return super().__new__(cls, jump.target)

    @property  # type: ignore[override]
    def target(self) -> Union[Label, LabelPlaceholder]:
        """Get the target of the jump."""
        if isinstance(super().target, quil_rs.Target.Placeholder):
            return LabelPlaceholder._from_rs_target(super().target)
        return Label._from_rs_target(super().target)

    def out(self) -> str:
        """Return the instruction as a valid Quil string. Raises an error if the target is an unresolved placeholder."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class Pragma(quil_rs.Pragma, AbstractInstruction):
    """A PRAGMA instruction.

    This is printed in QUIL as:

        PRAGMA <command> <arg1> <arg2> ... <argn> "<freeform_string>"

    """

    def __new__(
        cls,
        command: str,
        args: Sequence[Union[Qubit, FormalArgument, int, str]] = (),
        freeform_string: str = "",
    ) -> Self:
        """Initialize a new PRAGMA instruction."""
        data = freeform_string or None
        return super().__new__(cls, command, Pragma._to_pragma_arguments(args), data)

    def __getnewargs__(self) -> tuple[str, tuple[QubitDesignator], str]:
        return self.command, self.args, self.freeform_string

    @classmethod
    def _from_rs_pragma(cls, pragma: quil_rs.Pragma) -> "Pragma":
        return super().__new__(cls, pragma.name, pragma.arguments, pragma.data)

    @staticmethod
    def _to_pragma_argument(arg: Sequence[Union[QubitDesignator, str]]) -> list[quil_rs.PragmaArgument]:
        if isinstance(arg, Qubit):
            return quil_rs.PragmaArgument.Integer(arg.index)
        elif isinstance(arg, int):
            return quil_rs.PragmaArgument.Integer(arg)
        elif isinstance(arg, (str, FormalArgument)):
            return quil_rs.PragmaArgument.Identifier(str(arg))
        raise TypeError(f"{type(arg)} isn't a valid PRAGMA argument")

    @staticmethod
    def _to_pragma_arguments(args: Sequence[QubitDesignator | str]) -> list[quil_rs.PragmaArgument]:
        return [Pragma._to_pragma_argument(arg) for arg in args]

    @staticmethod
    def _to_py_argument(arg: quil_rs.PragmaArgument) -> QubitDesignator:
        match arg:
            case quil_rs.PragmaArgument.Integer(i):
                return Qubit(i)
            case quil_rs.PragmaArgument.Identifier(s):
                return FormalArgument(s)
        raise TypeError(f"{type(arg)} isn't a valid PRAGMA argument")

    @staticmethod
    def _to_py_arguments(args: list[quil_rs.PragmaArgument]) -> list[QubitDesignator]:
        return [Pragma._to_py_argument(arg) for arg in args]

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property
    def command(self) -> str:
        """Get the pragma command identifier."""
        return super().name

    @property
    def args(self) -> tuple[QubitDesignator]:
        """Get the arguments of the PRAGMA command."""
        return tuple(Pragma._to_py_arguments(super().arguments))  # type: ignore[return-value]

    @property
    def freeform_string(self) -> str:
        """Get the PRAGMA's freeform string."""
        return super().data or ""


class Declare(quil_rs.Declaration, AbstractInstruction):
    """A DECLARE directive.

    This is printed in Quil as::

        DECLARE <name> <memory-type> (SHARING <other-name> (OFFSET <amount> <type>)* )?
    """

    def __new__(
        cls,
        name: str,
        memory_type: str,
        memory_size: int = 1,
        shared_region: Optional[str] = None,
        offsets: Optional[Sequence[tuple[int, str]]] = None,
    ) -> Self:
        """Initialize a new DECLARE directive."""
        vector = quil_rs.Vector(Declare._memory_type_to_scalar_type(memory_type), memory_size)
        sharing = None
        if shared_region is not None:
            sharing = quil_rs.Sharing(shared_region, Declare._to_rs_offsets(offsets))
        return super().__new__(cls, name, vector, sharing)

    def __getnewargs__(self) -> tuple[str, str, int, str | None, Sequence[tuple[int, str]] | None]:
        return self.name, self.memory_type, self.memory_size, self.shared_region, self.offsets

    @classmethod
    def _from_rs_declaration(cls, declaration: quil_rs.Declaration) -> "Declare":
        return super().__new__(cls, declaration.name, declaration.size, declaration.sharing)

    @staticmethod
    def _memory_type_to_scalar_type(memory_type: str) -> quil_rs.ScalarType:
        memory_type = memory_type.upper()
        if memory_type == "BIT":
            return quil_rs.ScalarType.BIT
        if memory_type == "INTEGER":
            return quil_rs.ScalarType.INTEGER
        if memory_type == "REAL":
            return quil_rs.ScalarType.REAL
        if memory_type == "OCTET":
            return quil_rs.ScalarType.OCTET
        raise ValueError(f"{memory_type} is not a valid scalar type.")

    @staticmethod
    def _to_rs_offsets(offsets: Optional[Sequence[tuple[int, str]]]) -> list[quil_rs.Offset]:
        if offsets is None:
            return []
        return [
            quil_rs.Offset(offset, Declare._memory_type_to_scalar_type(memory_type)) for offset, memory_type in offsets
        ]

    @property
    def memory_type(self) -> str:
        """Get the type of memory being declared."""
        return str(super().size.data_type).upper()

    @property
    def memory_size(self) -> int:
        """Get the number of elements being declared."""
        return super().size.length

    @property
    def shared_region(self) -> Optional[str]:
        """Get the memory region this declaration is sharing with, if any."""
        sharing = super().sharing
        if sharing is None:
            return None
        return sharing.name

    @property
    def offsets(self) -> list[tuple[int, str]]:
        """Get the offsets for this declaration."""
        sharing = super().sharing
        if sharing is None:
            return []
        return [(offset.offset, str(offset.data_type).upper()) for offset in sharing.offsets]

    def asdict(self) -> dict[str, Union[Sequence[tuple[int, str]], Optional[str], int]]:
        """Get the DECLARE directive as a dictionary."""
        return {
            "name": self.name,
            "memory_type": self.memory_type,
            "memory_size": self.memory_size,
            "shared_region": self.shared_region,
            "offsets": self.offsets,
        }

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class Include(quil_rs.Include, AbstractInstruction):
    """An INCLUDE directive."""

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    @classmethod
    def _from_rs_include(cls, include: quil_rs.Include) -> "Include":
        return super().__new__(cls, include.filename)

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class Pulse(quil_rs.Pulse, AbstractInstruction):
    """A PULSE instruction."""

    def __new__(cls, frame: Frame, waveform: Waveform, nonblocking: bool = False) -> Self:
        """Initialize a new Pulse instruction."""
        return super().__new__(cls, not nonblocking, frame, waveform)

    def __getnewargs__(self) -> tuple[Frame, Waveform, bool]:
        return self.frame, self.waveform, self.nonblocking

    @classmethod
    def _from_rs_pulse(cls, pulse: quil_rs.Pulse) -> "Pulse":
        return super().__new__(cls, pulse.blocking, pulse.frame, pulse.waveform)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[set[QubitDesignator], set[int]]:
        """Get the qubits the pulse operates on."""
        if indices:
            return self.get_qubit_indices()
        else:
            return set(_convert_to_py_qubits(super().frame.qubits))

    def get_qubit_indices(self) -> set[int]:
        """Get the qubits the pulse operates on, as integer indices."""
        return {qubit._0 for qubit in super().frame.qubits}

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        """Get the frame the pulse operates on."""
        return Frame._from_rs_frame_identifier(super().frame)

    @frame.setter
    def frame(self, frame: Frame) -> None:
        quil_rs.Pulse.frame.__set__(self, frame)  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def waveform(self) -> Waveform:
        """Get the waveform of the pulse."""
        return _convert_to_py_waveform(super().waveform)

    @waveform.setter
    def waveform(self, waveform: Waveform) -> None:
        quil_rs.Pulse.waveform.__set__(self, waveform)  # type: ignore[attr-defined]

    @property
    def nonblocking(self) -> bool:
        """Return whether the pulse is non-blocking."""
        return not super().blocking

    @nonblocking.setter
    def nonblocking(self, nonblocking: bool) -> None:
        quil_rs.Pulse.blocking.__set__(self, not nonblocking)  # type: ignore[attr-defined]


class SetFrequency(quil_rs.SetFrequency, AbstractInstruction):
    """A SET-FREQUENCY instruction."""

    def __new__(cls, frame: Frame, freq: ParameterDesignator) -> Self:
        """Initialize a new SET-FREQUENCY instruction."""
        return super().__new__(cls, frame, _convert_to_rs_expression(freq))

    def __getnewargs__(self) -> tuple[Frame, ParameterDesignator]:
        return self.frame, self.freq

    @classmethod
    def _from_rs_set_frequency(cls, set_frequency: quil_rs.SetFrequency) -> "SetFrequency":
        return super().__new__(cls, set_frequency.frame, set_frequency.frequency)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        """Get the frame the frequency is set on."""
        return Frame._from_rs_frame_identifier(super().frame)

    @property
    def freq(self) -> ParameterDesignator:
        """Get the frequency that is set."""
        return _convert_to_py_expression(super().frequency)

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[set[QubitDesignator], set[int]]:
        """Get the qubits the frequency is set on."""
        if indices:
            return self.get_qubit_indices()
        return set(self.frame.qubits)

    def get_qubit_indices(self) -> set[int]:
        """Get the qubits the frequency is set on, as integer indices."""
        return {qubit._0 for qubit in super().frame.qubits}


class ShiftFrequency(quil_rs.ShiftFrequency, AbstractInstruction):
    """The SHIFT-FREQUENCY instruction."""

    def __new__(cls, frame: Frame, freq: ParameterDesignator) -> Self:
        """Initialize a new SHIFT-FREQUENCY instruction."""
        return super().__new__(cls, frame, _convert_to_rs_expression(freq))

    def __getnewargs__(self) -> tuple[Frame, ParameterDesignator]:
        return self.frame, self.freq

    @classmethod
    def _from_rs_shift_frequency(cls, shift_frequency: quil_rs.ShiftFrequency) -> "ShiftFrequency":
        return super().__new__(cls, shift_frequency.frame, shift_frequency.frequency)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        """Get the frame the frequency is shifted on."""
        return Frame._from_rs_frame_identifier(super().frame)

    @property
    def freq(self) -> ParameterDesignator:
        """Get the parameter that defines how the frequency is shifted."""
        return _convert_to_py_expression(super().frequency)

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[set[QubitDesignator], set[int]]:
        """Get the qubits the frequency is shifted on."""
        if indices:
            return self.get_qubit_indices()
        return set(self.frame.qubits)

    def get_qubit_indices(self) -> set[int]:
        """Get the qubits the frequency is shifted on, as integer indices."""
        return {qubit._0 for qubit in super().frame.qubits}


class SetPhase(quil_rs.SetPhase, AbstractInstruction):
    """The SET-PHASE instruction."""

    def __new__(cls, frame: Frame, phase: ParameterDesignator) -> Self:
        """Initialize a new SET-PHASE instruction."""
        return super().__new__(cls, frame, _convert_to_rs_expression(phase))

    def __getnewargs__(self) -> tuple[Frame, ParameterDesignator]:
        return self.frame, self.phase

    @classmethod
    def _from_rs_set_phase(cls, set_phase: quil_rs.SetPhase) -> "SetPhase":
        return super().__new__(cls, set_phase.frame, set_phase.phase)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        """Get the frame the phase is set on."""
        return Frame._from_rs_frame_identifier(super().frame)

    @property  # type: ignore[override]
    def phase(self) -> ParameterDesignator:
        """Get the phase this instruction sets."""
        return _convert_to_py_expression(super().phase)

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[set[QubitDesignator], set[int]]:
        """Get the quibts the phase is set on."""
        if indices:
            return self.get_qubit_indices()
        return set(self.frame.qubits)

    def get_qubit_indices(self) -> set[int]:
        """Get the quibts the phase is set on, as integer indices."""
        return {qubit._0 for qubit in super().frame.qubits}


class ShiftPhase(quil_rs.ShiftPhase, AbstractInstruction):
    """The SHIFT-PHASE instruction."""

    def __new__(cls, frame: Frame, phase: ParameterDesignator) -> Self:
        """Initialize a new SHIFT-PHASE instruction."""
        return super().__new__(cls, frame, _convert_to_rs_expression(phase))

    def __getnewargs__(self) -> tuple[Frame, ParameterDesignator]:
        return self.frame, self.phase

    @classmethod
    def _from_rs_shift_phase(cls, shift_phase: quil_rs.ShiftPhase) -> "ShiftPhase":
        return super().__new__(cls, shift_phase.frame, shift_phase.phase)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        """Get the frame the phase is shifted on."""
        return Frame._from_rs_frame_identifier(super().frame)

    @property  # type: ignore[override]
    def phase(self) -> ParameterDesignator:
        """Get the parameter that defines how the phase is shifted."""
        return _convert_to_py_expression(super().phase)

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[set[QubitDesignator], set[int]]:
        """Get the qubits the phase is shifted on."""
        if indices:
            return self.get_qubit_indices()
        return set(self.frame.qubits)

    def get_qubit_indices(self) -> set[int]:
        """Get the qubits the phase is shifted on, as integer indices."""
        return {qubit._0 for qubit in super().frame.qubits}


class SwapPhases(quil_rs.SwapPhases, AbstractInstruction):
    """The SWAP-PHASES instruction."""

    @classmethod
    def _from_rs_swap_phases(cls, swap_phases: quil_rs.SwapPhases) -> "SwapPhases":
        return super().__new__(cls, swap_phases.frame_1, swap_phases.frame_2)

    @property
    def frameA(self) -> Frame:
        """The first frame of the SWAP-PHASES instruction."""
        return Frame._from_rs_frame_identifier(super().frame_1)

    @property
    def frameB(self) -> Frame:
        """The second frame of the SWAP-PHASES instruction."""
        return Frame._from_rs_frame_identifier(super().frame_2)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[set[QubitDesignator], set[int]]:
        """Get the qubits the swap-phases instruction operates on."""
        if indices:
            return self.get_qubit_indices()
        return set(self.frameA.qubits) | set(self.frameB.qubits)

    def get_qubit_indices(self) -> set[int]:
        """Get the qubits the swap-phases instruction operates on, as integer indices."""
        return {qubit._0 for qubit in super().frame_1.qubits + super().frame_2.qubits}


class SetScale(quil_rs.SetScale, AbstractInstruction):
    """The SET-SCALE instruction."""

    def __new__(cls, frame: Frame, scale: ParameterDesignator) -> Self:
        """Initialize a new SET-SCALE instruction."""
        return super().__new__(cls, frame, _convert_to_rs_expression(scale))

    def __getnewargs__(self) -> tuple[Frame, ParameterDesignator]:
        return self.frame, self.scale

    @classmethod
    def _from_rs_set_scale(cls, set_scale: quil_rs.SetScale) -> "SetScale":
        return super().__new__(cls, set_scale.frame, set_scale.scale)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        """Get the frame the scale is set on."""
        return Frame._from_rs_frame_identifier(super().frame)

    @property  # type: ignore[override]
    def scale(self) -> ParameterDesignator:
        """Get the scale that is set."""
        return _convert_to_py_expression(super().scale)

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[set[QubitDesignator], set[int]]:
        """Get the qubits the scale is set on."""
        if indices:
            return self.get_qubit_indices()
        return set(self.frame.qubits)

    def get_qubit_indices(self) -> set[int]:
        """Get the qubits the scale is set on, as integer indices."""
        return {qubit._0 for qubit in super().frame.qubits}


class Call(quil_rs.Call, AbstractInstruction):
    """An instruction that calls an external function declared with a `PRAGMA EXTERN` instruction.

    These calls are generally specific to a particular hardware or virtual machine
    backend. For further detail, see:
    * `Other instructions and Directives <https://github.com/quil-lang/quil/blob/master/rfcs/extern-call.md>`_
        in the Quil specification.
    * `EXTERN / CALL RFC <https://github.com/quil-lang/quil/blob/master/rfcs/extern-call.md>`_
    * `quil#87 <https://github.com/quil-lang/quil/issues/87>`_
    """

    @classmethod
    def _from_rs_call(cls, call: quil_rs.Call) -> "Call":
        return super().__new__(cls, call.name, call.arguments)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class Capture(quil_rs.Capture, AbstractInstruction):
    """The CAPTURE instruction."""

    def __new__(
        cls,
        frame: Frame,
        kernel: Waveform,
        memory_region: MemoryReference,
        nonblocking: bool = False,
    ) -> Self:
        """Initialize a new CAPTURE instruction."""
        rs_memory_reference = _convert_to_rs_expression(memory_region)._0
        return super().__new__(cls, not nonblocking, frame, rs_memory_reference, kernel)

    def __getnewargs__(self) -> tuple[Frame, Waveform, MemoryReference, bool]:
        return self.frame, self.kernel, self.memory_region, self.nonblocking

    @classmethod
    def _from_rs_capture(cls, capture: quil_rs.Capture) -> "Capture":
        return super().__new__(cls, capture.blocking, capture.frame, capture.memory_reference, capture.waveform)

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        """Get the frame this instruction captures."""
        return Frame._from_rs_frame_identifier(super().frame)

    @frame.setter
    def frame(self, frame: Frame) -> None:
        quil_rs.Capture.frame.__set__(self, frame)  # type: ignore[attr-defined]

    @property
    def kernel(self) -> Waveform:
        """Get the kernel waveform."""
        return _convert_to_py_waveform(super().waveform)

    @kernel.setter
    def kernel(self, kernel: Waveform) -> None:
        quil_rs.Capture.waveform.__set__(self, kernel)  # type: ignore[attr-defined]

    @property
    def memory_region(self) -> MemoryReference:
        """Get the memory reference that the capture is written to."""
        return MemoryReference._from_rs_memory_reference(super().memory_reference)

    @memory_region.setter
    def memory_region(self, memory_region: MemoryReference) -> None:
        rs_memory_reference = _convert_to_rs_expression(memory_region)._0
        quil_rs.Capture.memory_reference.__set__(self, rs_memory_reference)  # type: ignore[attr-defined]

    @property
    def nonblocking(self) -> bool:
        """Whether the capture is non-blocking."""
        return not super().blocking

    @nonblocking.setter
    def nonblocking(self, nonblocking: bool) -> None:
        quil_rs.Capture.blocking.__set__(self, not nonblocking)  # type: ignore[attr-defined]

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[set[QubitDesignator], set[int]]:
        """Get the qubits the capture instruction operates on."""
        if indices:
            return self.get_qubit_indices()
        else:
            return set(_convert_to_py_qubits(super().frame.qubits))

    def get_qubit_indices(self) -> set[int]:
        """Get the qubits the capture instruction operates on, as integer indices."""
        return {qubit._0 for qubit in super().frame.qubits}


class RawCapture(quil_rs.RawCapture, AbstractInstruction):
    """The RAW-CAPTURE instruction."""

    def __new__(
        cls,
        frame: Frame,
        duration: float,
        memory_region: MemoryReference,
        nonblocking: bool = False,
    ) -> Self:
        """Initialize a new RAW-CAPTURE instruction."""
        rs_duration = _convert_to_rs_expression(duration)
        rs_memory_reference = _convert_to_rs_expression(memory_region)._0
        return super().__new__(cls, not nonblocking, frame, rs_duration, rs_memory_reference)

    def __getnewargs__(self) -> tuple[Frame, float, MemoryReference, bool]:
        return self.frame, self.duration, self.memory_region, self.nonblocking

    @classmethod
    def _from_rs_raw_capture(cls, raw_capture: quil_rs.RawCapture) -> "RawCapture":
        return super().__new__(
            cls, raw_capture.blocking, raw_capture.frame, raw_capture.duration, raw_capture.memory_reference
        )

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        """Get the frame this instruction operates on."""
        return Frame._from_rs_frame_identifier(super().frame)

    @property  # type: ignore[override]
    def duration(self) -> complex:
        """Get the duration of the capture."""
        match super().duration:
            case quil_rs_expr.Expression.Pi():
                return np.pi
            case quil_rs_expr.Expression.Number(duration):
                return duration
        raise TypeError("self.duration is not a number")

    @property
    def memory_region(self) -> MemoryReference:
        """Get the memory region that the capture is written to."""
        return MemoryReference._from_rs_memory_reference(super().memory_reference)

    @property
    def nonblocking(self) -> bool:
        """Whether the capture is non-blocking."""
        return not super().blocking

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[set[QubitDesignator], set[int]]:
        """Get the qubits the raw-capture instruction operates on."""
        if indices:
            return self.get_qubit_indices()
        else:
            return set(_convert_to_py_qubits(super().frame.qubits))

    def get_qubit_indices(self) -> set[int]:
        """Get the qubits the raw-capture instruction operates on, as integer indices."""
        return {qubit._0 for qubit in super().frame.qubits}


class Delay(quil_rs.Delay, AbstractInstruction):
    """The DELAY instruction."""

    def __new__(cls, frames: list[Frame], qubits: Sequence[Union[int, Qubit, FormalArgument]], duration: float) -> Self:
        """Initialize a new DELAY instruction."""
        frame_names = [frame.name for frame in frames]
        rs_qubits = _convert_to_rs_qubits(Delay._join_frame_qubits(frames, list(qubits)))
        expression = quil_rs_expr.Expression.Number(complex(duration))
        return super().__new__(cls, expression, frame_names, rs_qubits)

    def __getnewargs__(self) -> tuple[list[Frame], Sequence[int | Qubit | FormalArgument], float]:
        return self.frames, self.qubits, self.duration

    @classmethod
    def _from_rs_delay(cls, delay: quil_rs.Delay) -> "Delay":
        return super().__new__(cls, delay.duration, delay.frame_names, delay.qubits)

    @staticmethod
    def _join_frame_qubits(
        frames: Sequence[Frame], qubits: Sequence[Union[int, Qubit, FormalArgument]]
    ) -> list[Union[int, Qubit, FormalArgument]]:
        merged_qubits = set(qubits)
        for frame in frames:
            merged_qubits.update(frame.qubits)  # type: ignore
        return list(merged_qubits)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def qubits(self) -> list[QubitDesignator]:
        """Get the qubits the delay operates on."""
        return _convert_to_py_qubits(super().qubits)

    @property
    def frames(self) -> list[Frame]:
        """Get the frames the delay operates on."""
        return [Frame(self.qubits, name) for name in super().frame_names]

    @property  # type: ignore[override]
    def duration(self) -> float:
        """Get the duration of the delay."""
        return super().duration.to_real()


class DelayFrames(Delay):
    """A DELAY instruction that operates on frames."""

    def __new__(cls, frames: list[Frame], duration: float) -> Self:
        """Initialize a new DELAY instruction that operates on frames."""
        return super().__new__(cls, frames, [], duration)

    def __getnewargs__(self) -> tuple[list[Frame], float]:
        return self.frames, self.duration

    @classmethod
    def _from_rs_delay(cls, delay: quil_rs.Delay) -> "DelayFrames":
        return Delay._from_rs_delay.__func__(cls, delay)  # type: ignore


class DelayQubits(Delay):
    """Initialize a new DELAY instruction that operates on qubits."""

    def __new__(cls, qubits: Sequence[Union[Qubit, FormalArgument]], duration: float) -> Self:
        """Initialize a new DELAY instruction that operates on qubits."""
        return super().__new__(cls, [], qubits, duration)

    def __getnewargs__(self) -> tuple[Sequence[int | Qubit | FormalArgument], float]:
        return self.qubits, self.duration

    @classmethod
    def _from_rs_delay(cls, delay: quil_rs.Delay) -> "DelayQubits":
        return Delay._from_rs_delay.__func__(cls, delay)  # type: ignore


class Fence(quil_rs.Fence, AbstractInstruction):
    """The FENCE instruction."""

    def __new__(cls, qubits: list[Union[Qubit, FormalArgument]]) -> Self:
        """Initialize a new FENCE instruction."""
        return super().__new__(cls, _convert_to_rs_qubits(qubits))

    def __getnewargs__(self) -> tuple[list[Qubit | FormalArgument]]:
        return (self.qubits,)

    @classmethod
    def _from_rs_fence(cls, fence: quil_rs.Fence) -> "Fence":
        return super().__new__(cls, fence.qubits)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def qubits(self) -> list[QubitDesignator]:
        """Get the qubits the fence operates on.

        Note: If qubits is empty, then the instruction fences all qubits.
        """
        return _convert_to_py_qubits(super().qubits)


class FenceAll(Fence):
    """A FENCE instruction that operates on all qubits."""

    def __new__(cls) -> Self:
        """Initialize a new FenceAll instruction."""
        return super().__new__(cls, [])

    def __getnewargs__(self) -> tuple:
        return ()


class DefWaveform(quil_rs.WaveformDefinition, AbstractInstruction):
    """A waveform definition."""

    def __new__(
        cls,
        name: str,
        parameters: list[Parameter],
        entries: list[Union[complex, Expression]],
    ) -> Self:
        """Initialize a new waveform definition."""
        rs_waveform = DefWaveform._build_rs_waveform(parameters, entries)
        return super().__new__(cls, name, rs_waveform)

    def __getnewargs__(self) -> tuple[str, list[Parameter], list[complex | Expression]]:
        return self.name, self.parameters, self.entries

    @classmethod
    def _from_rs_waveform_definition(cls, waveform_definition: quil_rs.WaveformDefinition) -> "DefWaveform":
        return super().__new__(cls, waveform_definition.name, waveform_definition.definition)

    @staticmethod
    def _build_rs_waveform(parameters: list[Parameter], entries: list[Union[complex, Expression]]) -> quil_rs.Waveform:
        rs_parameters = [parameter.name for parameter in parameters]
        return quil_rs.Waveform(_convert_to_rs_expressions(entries), rs_parameters)

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property
    def parameters(self) -> list[Parameter]:
        """Get the parameters of the waveform."""
        return [Parameter(parameter) for parameter in super().definition.parameters]

    @property
    def entries(self) -> Sequence[ParameterDesignator]:
        """Get the entries in the waveform definition."""
        return _convert_to_py_expressions(super().definition.matrix)


class DefCircuit(quil_rs.CircuitDefinition, AbstractInstruction):
    """A circuit definition."""

    def __new__(
        cls,
        name: str,
        parameters: Iterable[Parameter],
        qubits: Iterable[FormalArgument],
        instructions: Iterable[AbstractInstruction],
    ) -> Self:
        """Initialize a new circuit definition."""
        rs_parameters = [parameter.name for parameter in parameters]
        rs_qubits = [qubit.name for qubit in qubits]
        rs_instructions = _convert_to_rs_instructions(instructions)
        return super().__new__(cls, name, rs_parameters, rs_qubits, rs_instructions)

    def __getnewargs__(self) -> tuple[str, list[Parameter], list[FormalArgument], list[AbstractInstruction]]:
        return self.name, self.parameters, self.qubit_variables, self.instructions

    @classmethod
    def _from_rs_circuit_definition(cls, circuit_definition: quil_rs.CircuitDefinition) -> "DefCircuit":
        return super().__new__(
            cls,
            circuit_definition.name,
            circuit_definition.parameters,
            circuit_definition.qubit_variables,
            circuit_definition.instructions,
        )

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def parameters(self) -> list[Parameter]:
        """Get the parameters of the circuit."""
        return [Parameter(parameter) for parameter in super().parameters]

    @parameters.setter  # type: ignore[override]
    def parameters(self, parameters: list[Parameter]) -> None:
        rs_parameters = [parameter.name for parameter in parameters]
        quil_rs.CircuitDefinition.parameters.__set__(self, rs_parameters)  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def qubit_variables(self) -> list[FormalArgument]:
        """Get the qubit parameters of the circuit."""
        return [FormalArgument(qubit) for qubit in super().qubit_variables]

    @qubit_variables.setter  # type: ignore[override]
    def qubit_variables(self, qubits: list[FormalArgument]) -> None:
        rs_qubits = [qubit.name for qubit in qubits]
        quil_rs.CircuitDefinition.qubit_variables.__set__(self, rs_qubits)  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def instructions(self) -> list[AbstractInstruction]:
        """Get the instructions in the circuit."""
        return _convert_to_py_instructions(super().instructions)

    @instructions.setter  # type: ignore[override]
    def instructions(self, instructions: list[AbstractInstruction]) -> None:
        rs_instructions = _convert_to_rs_instructions(instructions)
        quil_rs.CircuitDefinition.instructions.__set__(self, rs_instructions)  # type: ignore[attr-defined]


class DefCalibration(quil_rs.CalibrationDefinition, AbstractInstruction):
    """A calibration definition."""

    def __new__(
        cls,
        name: str,
        parameters: Sequence[ParameterDesignator],
        qubits: Sequence[Union[Qubit, FormalArgument]],
        instrs: Sequence[AbstractInstruction],
        modifiers: Optional[list[quil_rs.GateModifier]] = None,
    ) -> Self:
        """Initialize a new calibration definition."""
        return super().__new__(
            cls,
            quil_rs.CalibrationIdentifier(
                name,
                _convert_to_rs_expressions(parameters),
                _convert_to_rs_qubits(qubits),
                modifiers or [],
            ),
            _convert_to_rs_instructions(instrs),
        )

    def __getnewargs__(
        self,
    ) -> tuple[
        str,
        Sequence[ParameterDesignator],
        list[QubitDesignator],
        list[AbstractInstruction],
        Sequence[quil_rs.GateModifier],
    ]:
        return self.name, self.parameters, self.qubits, self.instrs, self.modifiers

    @classmethod
    def _from_rs_calibration(cls, calibration: quil_rs.CalibrationDefinition) -> "DefCalibration":
        return super().__new__(cls, calibration.identifier, calibration.instructions)

    @property  # type: ignore[override]
    def parameters(self) -> Sequence[ParameterDesignator]:
        """The parameters of the calibration."""
        return _convert_to_py_expressions(super().identifier.parameters)

    @parameters.setter
    def parameters(self, parameters: Sequence[ParameterDesignator]) -> None:
        identifier = super().identifier
        identifier.parameters = _convert_to_rs_expressions(parameters)
        quil_rs.CalibrationDefinition.identifier.__set__(self, identifier)  # type: ignore[attr-defined] # noqa

    @property  # type: ignore[override]
    def qubits(self) -> list[QubitDesignator]:
        """The qubits the calibration operates on."""
        return _convert_to_py_qubits(super().identifier.qubits)

    @qubits.setter
    def qubits(self, qubits: Sequence[QubitDesignator]) -> None:
        identifier = super().identifier
        identifier.qubits = _convert_to_rs_qubits(qubits)
        quil_rs.CalibrationDefinition.identifier.__set__(self, identifier)  # type: ignore[attr-defined]

    @property
    def instrs(self) -> list[AbstractInstruction]:
        """The instructions in the calibration."""
        return _convert_to_py_instructions(super().instructions)

    @instrs.setter
    def instrs(self, instrs: Sequence[AbstractInstruction]) -> None:
        quil_rs.CalibrationDefinition.instructions.__set__(self, _convert_to_rs_instructions(instrs))  # type: ignore[attr-defined] # noqa

    @property  # type: ignore[override]
    def instructions(self) -> list[AbstractInstruction]:
        """The instructions in the calibration."""
        return self.instrs

    @instructions.setter  # type: ignore[override]
    def instructions(self, instructions: list[AbstractInstruction]) -> None:
        self.instrs = instructions

    @property
    def name(self) -> str:
        """Get the name of the calibration."""
        return super().identifier.name

    @name.setter
    def name(self, name: str) -> None:
        identifier = super().identifier
        identifier.name = name
        quil_rs.CalibrationDefinition.identifier.__set__(self, identifier)  # type: ignore

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


MemoryRefStr = MemoryReference | quil_rs.MemoryReference | str


class DefMeasureCalibration(quil_rs.MeasureCalibrationDefinition, AbstractInstruction):
    """A measure calibration definition."""

    def __new__(
        cls,
        qubit: Union[Qubit, FormalArgument],
        memory_reference: Optional[Union[MemoryReference, quil_rs.MemoryReference, str]],
        instrs: list[AbstractInstruction],
    ) -> Self:
        """Initialize a new measure calibration definition."""
        mem_ref = str(memory_reference) if memory_reference else None
        ident = quil_rs.MeasureCalibrationIdentifier(_convert_to_rs_qubit(qubit), mem_ref)
        return super().__new__(cls, ident, _convert_to_rs_instructions(instrs))

    def __getnewargs__(self) -> tuple[Qubit, Optional[MemoryReference], list[quil_rs.Instruction]]:
        ident, instructions = super().__getnewargs__()
        return ident.qubit, ident.target, instructions

    @classmethod
    def _from_rs_measure_calibration_definition(
        cls, calibration: quil_rs.MeasureCalibrationDefinition
    ) -> "DefMeasureCalibration":
        return super().__new__(cls, calibration.identifier, calibration.instructions)

    @property
    def qubit(self) -> QubitDesignator:
        """Get the qubit this calibration matches."""
        return _convert_to_py_qubit(self.identifier.qubit)

    @qubit.setter
    def qubit(self, qubit: QubitDesignator) -> None:
        identifier = self.identifier
        identifier.qubit = _convert_to_rs_qubit(qubit)
        self.identifier = identifier

    @property
    def memory_reference(self) -> Optional[MemoryReference]:
        """Get the memory reference this calibration writes to."""
        target = super().identifier.target
        return MemoryReference._from_parameter_str(target) if target else None

    @memory_reference.setter
    def memory_reference(self, memory_reference: Optional[MemoryReference]) -> None:
        identifier = self.identifier
        identifier.target = memory_reference.out() if memory_reference else None
        self.identifier = identifier

    @property
    def instrs(self) -> list[AbstractInstruction]:
        """Get the instructions in the calibration."""
        return _convert_to_py_instructions(super().instructions)

    @instrs.setter
    def instrs(self, instrs: list[AbstractInstruction]) -> None:
        quil_rs.MeasureCalibrationDefinition.instructions.__set__(self, _convert_to_rs_instructions(instrs))

    @property  # type: ignore[override]
    def instructions(self) -> list[AbstractInstruction]:
        """The instructions in the calibration."""
        return self.instrs

    @instructions.setter  # type: ignore[override]
    def instructions(self, instructions: list[AbstractInstruction]) -> None:
        self.instrs = instructions

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class DefFrame(quil_rs.FrameDefinition, AbstractInstruction):
    """A frame definition."""

    def __new__(
        cls,
        frame: Frame,
        direction: Optional[str] = None,
        initial_frequency: Optional[float] = None,
        hardware_object: Optional[str] = None,
        sample_rate: Optional[float] = None,
        center_frequency: Optional[float] = None,
        enable_raw_capture: Optional[str] = None,
        channel_delay: Optional[float] = None,
    ) -> Self:
        """Get the frame definition."""
        attributes = {
            key: DefFrame._to_attribute_value(value)
            for key, value in zip(
                [
                    "DIRECTION",
                    "INITIAL-FREQUENCY",
                    "HARDWARE-OBJECT",
                    "SAMPLE-RATE",
                    "CENTER-FREQUENCY",
                    "ENABLE-RAW-CAPTURE",
                    "CHANNEL-DELAY",
                ],
                [
                    direction,
                    initial_frequency,
                    hardware_object,
                    sample_rate,
                    center_frequency,
                    enable_raw_capture,
                    channel_delay,
                ],
            )
            if value is not None
        }
        return super().__new__(cls, frame, attributes)

    def __getnewargs__(
        self,
    ) -> tuple[
        Frame,
        str | None,
        float | None,
        str | None,
        float | None,
        float | None,
        str | None,
        float | None,
    ]:
        return (
            self.frame,
            self.get_attribute("DIRECTION"),
            self.get_attribute("INITIAL-FREQUENCY"),
            self.get_attribute("HARDWARE-OBJECT"),
            self.get_attribute("SAMPLE-RATE"),
            self.get_attribute("CENTER-FREQUENCY"),
            self.get_attribute("ENABLE-RAW-CAPTURE"),
            self.get_attribute("CHANNEL-DELAY"),
        )

    @classmethod
    def _from_rs_frame_definition(cls, def_frame: quil_rs.FrameDefinition) -> "DefFrame":
        return super().__new__(cls, def_frame.identifier, def_frame.attributes)

    @classmethod
    def _from_rs_attribute_values(
        cls, frame: quil_rs.FrameIdentifier, attributes: dict[str, quil_rs.AttributeValue]
    ) -> "DefFrame":
        return super().__new__(cls, frame, attributes)

    @staticmethod
    def _to_attribute_value(value: Union[str, float]) -> quil_rs.AttributeValue:
        if isinstance(value, str):
            return quil_rs.AttributeValue.String(value)
        if isinstance(value, (int, float, complex)):
            return quil_rs.AttributeValue.Expression(quil_rs_expr.Expression.Number(complex(value)))
        raise ValueError(f"{type(value)} is not a valid AttributeValue")

    def out(self) -> str:
        """Return the instruction as a valid Quil string."""
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property
    def frame(self) -> Frame:
        """Get the frame identifier."""
        return Frame._from_rs_frame_identifier(super().identifier)

    @frame.setter
    def frame(self, frame: Frame) -> None:
        quil_rs.FrameDefinition.identifier.__set__(self, frame)  # type: ignore[attr-defined]

    def set_attribute(self, name: str, value: Union[str, float]) -> None:
        """Set an attribute on the frame definition."""
        updated = super().attributes
        updated.update({name: DefFrame._to_attribute_value(value)})
        quil_rs.FrameDefinition.attributes.__set__(self, updated)  # type: ignore[attr-defined]

    def get_attribute(self, name: str) -> Optional[Union[str, float]]:
        """Get an attribute's value on the frame definition."""
        match super().attributes.get(name, None):
            case quil_rs.AttributeValue.String(s):
                return s
            case quil_rs.AttributeValue.Expression(quil_rs_expr.Expression.Number(_) as ex):
                return ex.to_real()
        return None

    def __getitem__(self, name: str) -> Union[str, float]:
        if not isinstance(name, str):
            raise TypeError("Frame attribute keys must be strings")
        value = self.get_attribute(name)
        if value is None:
            raise AttributeError(f"Attribute {name} not found")
        return value

    def __setitem__(self, name: str, value: Union[str, float]) -> None:
        if not isinstance(name, str):
            raise TypeError("Frame attribute keys must be strings")
        self.set_attribute(name, value)

    @property
    @deprecated(
        version="4.0",
        reason="Quil now supports generic key/value pairs in DEFFRAMEs. Use get_attribute('DIRECTION') instead.",
    )
    def direction(self) -> Optional[str]:
        """Get the DIRECTION attribute of the frame."""
        return self.get_attribute("DIRECTION")  # type: ignore

    @direction.setter
    @deprecated(
        version="4.0",
        reason="Quil now supports generic key/value pairs in DEFFRAMEs. Use set_attribute('DIRECTION') instead.",
    )
    def direction(self, direction: str) -> None:
        self.set_attribute("DIRECTION", direction)

    @property
    @deprecated(
        version="4.0",
        reason="Quil now supports generic key/value pairs in DEFFRAMEs. Use set_attribute('INITIAL-FREQUENCY') instead.",  # noqa: E501
    )
    def initial_frequency(self) -> Optional[float]:
        """Get the INITIAL-FREQUENCY attribute of the frame."""
        return self.get_attribute("INITIAL-FREQUENCY")  # type: ignore

    @initial_frequency.setter
    @deprecated(
        version="4.0",
        reason="Quil now supports generic key/value pairs in DEFFRAMEs. Use set_attribute('INITIAL-FREQUENCY') instead.",  # noqa: E501
    )
    def initial_frequency(self, initial_frequency: float) -> None:
        self.set_attribute("INITIAL-FREQUENCY", initial_frequency)

    @property
    @deprecated(
        version="4.0",
        reason="Quil now supports generic key/value pairs in DEFFRAMEs. Use get_attribute('HARDWARE-OBJECT') instead.",
    )
    def hardware_object(self) -> Optional[str]:
        """Get the HARDWARE-OBJECT attribute of the frame."""
        return self.get_attribute("HARDWARE-OBJECT")  # type: ignore

    @hardware_object.setter
    @deprecated(
        version="4.0",
        reason="Quil now supports generic key/value pairs in DEFFRAMEs. Use set_attribute('HARDWARE-OBJECT') instead.",
    )
    def hardware_object(self, hardware_object: str) -> None:
        self.set_attribute("HARDWARE-OBJECT", hardware_object)

    @property
    @deprecated(
        version="4.0",
        reason="Quil now supports generic key/value pairs in DEFFRAMEs. Use get_attribute('SAMPLE-RATE') instead.",
    )
    def sample_rate(self) -> Frame:
        """Get the SAMPLE-RATE attribute of the frame."""
        return self.get_attribute("SAMPLE-RATE")  # type: ignore

    @sample_rate.setter
    @deprecated(
        version="4.0",
        reason="Quil now supports generic key/value pairs in DEFFRAMEs. Use set_attribute('SAMPLE-RATE') instead.",
    )
    def sample_rate(self, sample_rate: float) -> None:
        self.set_attribute("SAMPLE-RATE", sample_rate)

    @property
    @deprecated(
        version="4.0",
        reason="Quil now supports generic key/value pairs in DEFFRAMEs. Use get_attribute('CENTER-FREQUENCY') instead.",
    )
    def center_frequency(self) -> float:
        """Get the CENTER-FREQUENCY attribute of the frame."""
        return self.get_attribute("CENTER-FREQUENCY")

    @center_frequency.setter
    @deprecated(
        version="4.0",
        reason="Quil now supports generic key/value pairs in DEFFRAMEs. Use set_attribute('CENTER-FREQUENCY') instead.",
    )
    def center_frequency(self, center_frequency: float) -> None:
        self.set_attribute("CENTER-FREQUENCY", center_frequency)

    @property
    @deprecated(
        version="4.0",
        reason="Quil now supports generic key/value pairs in DEFFRAMEs. Use get_attribute('CHANNEL-DELAY') instead.",
    )
    def channel_delay(self) -> Frame:
        """Get the CHANNEL-DELAY attribute of the frame."""
        return self.get_attribute("CHANNEL-DELAY")  # type: ignore

    @channel_delay.setter
    @deprecated(
        version="4.0",
        reason="Quil now supports generic key/value pairs in DEFFRAMEs. Use set_attribute('CHANNEL-DELAY') instead.",
    )
    def channel_delay(self, channel_delay: float) -> None:
        self.set_attribute("CHANNEL-DELAY", channel_delay)
