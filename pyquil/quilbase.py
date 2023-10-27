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
import json

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
)
from typing_extensions import Self

import numpy as np
from deprecated.sphinx import deprecated

from pyquil.quilatom import (
    Expression,
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
    _convert_to_py_qubit,
    _convert_to_py_qubits,
    _convert_to_rs_expression,
    _convert_to_rs_expressions,
    _convert_to_rs_qubit,
    _convert_to_rs_qubits,
    _convert_to_py_expression,
    _convert_to_py_expressions,
    _convert_to_py_waveform,
    unpack_qubit,
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

    def __init__(self, *args: Any, **_: Any):
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
        if self.__name not in ["AbstractInstruction", "DefGate"] and not self.__is_abstract_instruction:
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


def _convert_to_rs_instruction(instr: Union[AbstractInstruction, quil_rs.Instruction]) -> quil_rs.Instruction:
    if isinstance(instr, quil_rs.Instruction):
        return instr
    if isinstance(instr, quil_rs.Arithmetic):
        return quil_rs.Instruction.from_arithmetic(instr)
    if isinstance(instr, quil_rs.BinaryLogic):
        return quil_rs.Instruction.from_binary_logic(instr)
    if isinstance(instr, quil_rs.Capture):
        return quil_rs.Instruction.from_capture(instr)
    if isinstance(instr, quil_rs.CircuitDefinition):
        return quil_rs.Instruction.from_circuit_definition(instr)
    if isinstance(instr, quil_rs.Calibration):
        return quil_rs.Instruction.from_calibration_definition(instr)
    if isinstance(instr, quil_rs.Convert):
        return quil_rs.Instruction.from_convert(instr)
    if isinstance(instr, quil_rs.Declaration):
        return quil_rs.Instruction.from_declaration(instr)
    if isinstance(instr, quil_rs.Delay):
        return quil_rs.Instruction.from_delay(instr)
    if isinstance(instr, quil_rs.Exchange):
        return quil_rs.Instruction.from_exchange(instr)
    if isinstance(instr, quil_rs.Fence):
        return quil_rs.Instruction.from_fence(instr)
    if isinstance(instr, quil_rs.FrameDefinition):
        return quil_rs.Instruction.from_frame_definition(instr)
    if isinstance(instr, quil_rs.Gate):
        return quil_rs.Instruction.from_gate(instr)
    if isinstance(instr, quil_rs.GateDefinition):
        return quil_rs.Instruction.from_gate_definition(instr)
    if isinstance(instr, Halt):
        return quil_rs.Instruction.new_halt()
    if isinstance(instr, quil_rs.Load):
        return quil_rs.Instruction.from_load(instr)
    if isinstance(instr, quil_rs.MeasureCalibrationDefinition):
        return quil_rs.Instruction.from_measure_calibration_definition(instr)
    if isinstance(instr, quil_rs.Measurement):
        return quil_rs.Instruction.from_measurement(instr)
    if isinstance(instr, Nop):
        return quil_rs.Instruction.new_nop()
    if isinstance(instr, quil_rs.Pragma):
        return quil_rs.Instruction.from_pragma(instr)
    if isinstance(instr, quil_rs.Pulse):
        return quil_rs.Instruction.from_pulse(instr)
    if isinstance(instr, quil_rs.RawCapture):
        return quil_rs.Instruction.from_raw_capture(instr)
    if isinstance(instr, quil_rs.Reset):
        return quil_rs.Instruction.from_reset(instr)
    if isinstance(instr, quil_rs.SetFrequency):
        return quil_rs.Instruction.from_set_frequency(instr)
    if isinstance(instr, quil_rs.SetPhase):
        return quil_rs.Instruction.from_set_phase(instr)
    if isinstance(instr, quil_rs.SetScale):
        return quil_rs.Instruction.from_set_scale(instr)
    if isinstance(instr, quil_rs.ShiftFrequency):
        return quil_rs.Instruction.from_shift_frequency(instr)
    if isinstance(instr, quil_rs.ShiftPhase):
        return quil_rs.Instruction.from_shift_phase(instr)
    if isinstance(instr, quil_rs.SwapPhases):
        return quil_rs.Instruction.from_swap_phases(instr)
    if isinstance(instr, quil_rs.Store):
        return quil_rs.Instruction.from_store(instr)
    if isinstance(instr, Wait):
        return quil_rs.Instruction.new_wait()
    if isinstance(instr, quil_rs.WaveformDefinition):
        return quil_rs.Instruction.from_waveform_definition(instr)
    if isinstance(instr, quil_rs.Label):
        return quil_rs.Instruction.from_label(instr)
    if isinstance(instr, quil_rs.Move):
        return quil_rs.Instruction.from_move(instr)
    if isinstance(instr, quil_rs.Jump):
        return quil_rs.Instruction.from_jump(instr)
    if isinstance(instr, quil_rs.JumpWhen):
        return quil_rs.Instruction.from_jump_when(instr)
    if isinstance(instr, quil_rs.JumpUnless):
        return quil_rs.Instruction.from_jump_unless(instr)
    if isinstance(instr, quil_rs.UnaryLogic):
        return quil_rs.Instruction.from_unary_logic(instr)
    if isinstance(instr, quil_rs.Comparison):
        return quil_rs.Instruction.from_comparison(instr)
    raise ValueError(f"{type(instr)} is not an Instruction")


def _convert_to_rs_instructions(instrs: Iterable[AbstractInstruction]) -> List[quil_rs.Instruction]:
    return [_convert_to_rs_instruction(instr) for instr in instrs]


def _convert_to_py_instruction(instr: Any) -> AbstractInstruction:
    if isinstance(instr, quil_rs.Instruction):
        if instr.is_nop():
            return Nop()
        if instr.is_halt():
            return Halt()
        if instr.is_wait():
            return Wait()
        return _convert_to_py_instruction(instr.inner())
    if isinstance(instr, quil_rs.Capture):
        return Capture._from_rs_capture(instr)
    if isinstance(instr, quil_rs.Calibration):
        return DefCalibration._from_rs_calibration(instr)
    if isinstance(instr, quil_rs.Declaration):
        return Declare._from_rs_declaration(instr)
    if isinstance(instr, quil_rs.Delay):
        return Delay._from_rs_delay(instr)
    if isinstance(instr, quil_rs.Fence):
        if len(instr.qubits) == 0:
            return FenceAll()
        return Fence._from_rs_fence(instr)
    if isinstance(instr, quil_rs.FrameDefinition):
        return DefFrame._from_rs_frame_definition(instr)
    if isinstance(instr, quil_rs.Gate):
        return Gate._from_rs_gate(instr)
    if isinstance(instr, quil_rs.MeasureCalibrationDefinition):
        return DefMeasureCalibration._from_rs_measure_calibration_definition(instr)
    if isinstance(instr, quil_rs.Measurement):
        return Measurement._from_rs_measurement(instr)
    if isinstance(instr, quil_rs.Pragma):
        return Pragma._from_rs_pragma(instr)
    if isinstance(instr, quil_rs.Pulse):
        return Pulse._from_rs_pulse(instr)
    if isinstance(instr, quil_rs.RawCapture):
        return RawCapture._from_rs_raw_capture(instr)
    if isinstance(instr, quil_rs.Reset):
        return Reset._from_rs_reset(instr)
    if isinstance(instr, quil_rs.CircuitDefinition):
        return DefCircuit._from_rs_circuit_definition(instr)
    if isinstance(instr, quil_rs.GateDefinition):
        return DefGate._from_rs_gate_definition(instr)
    if isinstance(instr, quil_rs.WaveformDefinition):
        return DefWaveform._from_rs_waveform_definition(instr)
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
    if isinstance(instr, quil_rs.Label):
        return JumpTarget._from_rs_label(instr)
    if isinstance(instr, quil_rs.Move):
        return ClassicalMove._from_rs_move(instr)
    if isinstance(instr, quil_rs.Jump):
        return Jump._from_rs_jump(instr)
    if isinstance(instr, quil_rs.JumpWhen):
        return JumpWhen._from_rs_jump_when(instr)
    if isinstance(instr, quil_rs.JumpUnless):
        return JumpUnless._from_rs_jump_unless(instr)
    if isinstance(instr, quil_rs.Instruction):
        raise NotImplementedError(f"The {type(instr)} Instruction hasn't been mapped to an AbstractInstruction yet.")
    elif isinstance(instr, AbstractInstruction):
        return instr
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


class Gate(quil_rs.Gate, AbstractInstruction):
    """
    This is the pyQuil object for a quantum gate instruction.
    """

    def __new__(
        cls,
        name: str,
        params: Sequence[ParameterDesignator],
        qubits: Sequence[Union[Qubit, QubitPlaceholder, FormalArgument, int]],
        modifiers: Sequence[quil_rs.GateModifier] = [],
    ) -> Self:
        return super().__new__(
            cls, name, _convert_to_rs_expressions(params), _convert_to_rs_qubits(qubits), list(modifiers)
        )

    @classmethod
    def _from_rs_gate(cls, gate: quil_rs.Gate) -> Self:
        return super().__new__(cls, gate.name, gate.parameters, gate.qubits, gate.modifiers)

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Sequence[QubitDesignator]:
        if indices:
            return self.get_qubit_indices()
        else:
            return _convert_to_py_qubits(super().qubits)

    @property  # type: ignore[override]
    def qubits(self) -> List[QubitDesignator]:
        return self.get_qubits(indices=False)  # type: ignore

    @qubits.setter
    def qubits(self, qubits: Sequence[Union[Qubit, QubitPlaceholder, FormalArgument]]) -> None:
        quil_rs.Gate.qubits.__set__(self, _convert_to_rs_qubits(qubits))  # type: ignore

    @property
    def params(self) -> Sequence[ParameterDesignator]:
        return _convert_to_py_expressions(super().parameters)

    @params.setter
    def params(self, params: Sequence[ParameterDesignator]) -> None:
        quil_rs.Gate.parameters.__set__(self, _convert_to_rs_expressions(params))  # type: ignore

    @property  # type: ignore[override]
    def modifiers(self) -> List[str]:
        return [str(modifier).upper() for modifier in super().modifiers]

    @modifiers.setter
    def modifiers(self, modifiers: Union[List[str], List[quil_rs.GateModifier]]) -> None:
        modifiers = [
            self._to_rs_gate_modifier(modifier) if isinstance(modifier, str) else modifier for modifier in modifiers
        ]
        quil_rs.Gate.modifiers.__set__(self, modifiers)  # type: ignore[attr-defined]

    def _to_rs_gate_modifier(self, modifier: str) -> quil_rs.GateModifier:
        modifier = modifier.upper()
        if modifier == "CONTROLLED":
            return quil_rs.GateModifier.Controlled
        if modifier == "DAGGER":
            return quil_rs.GateModifier.Dagger
        if modifier == "FORKED":
            return quil_rs.GateModifier.Forked
        raise ValueError(f"{modifier} is not a valid Gate modifier.")

    def get_qubit_indices(self) -> List[int]:
        return [qubit.to_fixed() for qubit in super().qubits]

    def controlled(
        self,
        control_qubit: Union[
            quil_rs.Qubit,
            QubitDesignator,
            Sequence[Union[QubitDesignator, quil_rs.Qubit]],
        ],
    ) -> "Gate":
        """
        Add the CONTROLLED modifier to the gate with the given control qubit or Sequence of control
        qubits.
        """
        if isinstance(control_qubit, Sequence):
            for qubit in control_qubit:
                self._update_super(super().controlled(_convert_to_rs_qubit(qubit)))
        else:
            self._update_super(super().controlled(_convert_to_rs_qubit(control_qubit)))

        return self

    def forked(
        self,
        fork_qubit: Union[quil_rs.Qubit, QubitDesignator],
        alt_params: Union[Sequence[ParameterDesignator], Sequence[quil_rs_expr.Expression]],
    ) -> "Gate":
        """
        Add the FORKED modifier to the gate with the given fork qubit and given additional
        parameters.
        """
        forked = super().forked(_convert_to_rs_qubit(fork_qubit), _convert_to_rs_expressions(alt_params))
        self._update_super(forked)
        return self

    def dagger(self) -> "Gate":
        """
        Add the DAGGER modifier to the gate.
        """
        self._update_super(super().dagger())
        return self

    def out(self) -> str:
        return super().to_quil()

    def _update_super(self, gate: quil_rs.Gate) -> None:
        """
        Updates the state of the super class using a new gate.
        The super class does not mutate the value of a gate when adding
        modifiers with methods like `dagger()`, but pyQuil does.
        """
        quil_rs.Gate.name.__set__(self, gate.name)  # type: ignore[attr-defined]
        quil_rs.Gate.parameters.__set__(self, gate.parameters)  # type: ignore[attr-defined]
        quil_rs.Gate.modifiers.__set__(self, gate.modifiers)  # type: ignore[attr-defined]
        quil_rs.Gate.qubits.__set__(self, gate.qubits)  # type: ignore[attr-defined]

    def __str__(self) -> str:
        return super().to_quil_or_debug()


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
    ) -> Self:
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
        return _convert_to_py_qubit(super().qubit)

    @qubit.setter
    def qubit(self, qubit: QubitDesignator) -> None:
        quil_rs.Measurement.qubit.__set__(self, _convert_to_rs_qubit(qubit))  # type: ignore[attr-defined]

    @property
    def classical_reg(self) -> Optional[MemoryReference]:
        target = super().target
        if target is None:
            return None
        return MemoryReference._from_rs_memory_reference(target)

    @classical_reg.setter
    def classical_reg(self, classical_reg: Optional[MemoryReference]) -> None:
        target = self._reg_to_target(classical_reg)
        quil_rs.Measurement.target.__set__(self, target)  # type: ignore[attr-defined]

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[Set[QubitDesignator], Set[int]]:
        if indices:
            return self.get_qubit_indices()
        else:
            return {_convert_to_py_qubit(super().qubit)}

    def get_qubit_indices(self) -> Set[int]:
        return {super().qubit.to_fixed()}

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class Reset(quil_rs.Reset, AbstractInstruction):
    """
    The RESET instruction.
    """

    def __new__(cls, qubit: Optional[Union[Qubit, QubitPlaceholder, FormalArgument]] = None) -> Self:
        rs_qubit: Optional[quil_rs.Qubit] = None
        if qubit is not None:
            rs_qubit = _convert_to_rs_qubit(qubit)
        return super().__new__(cls, rs_qubit)

    @classmethod
    def _from_rs_reset(cls, reset: quil_rs.Reset) -> "Reset":
        return super().__new__(cls, reset.qubit)

    def out(self) -> str:
        return super().to_quil()

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
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

    @property  # type: ignore[override]
    def qubit(self) -> Optional[QubitDesignator]:
        if super().qubit:
            return _convert_to_py_qubit(super().qubit)  # type: ignore
        return None

    @qubit.setter
    def qubit(self, qubit: Optional[QubitDesignator]) -> None:
        rs_qubit: Optional[quil_rs.Qubit] = None
        if qubit is not None:
            rs_qubit = _convert_to_rs_qubit(qubit)
        quil_rs.Reset.qubit.__set__(self, rs_qubit)  # type: ignore[attr-defined]

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ResetQubit(Reset):
    """
    This is the pyQuil object for a Quil targeted reset instruction.
    """

    def __new__(cls, qubit: Union[Qubit, QubitPlaceholder, FormalArgument]) -> Self:
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
    ) -> Self:
        DefGate._validate_matrix(matrix, parameters is not None and len(parameters) > 0)
        specification = DefGate._convert_to_matrix_specification(matrix)
        rs_parameters = [param.name for param in parameters or []]
        return super().__new__(cls, name, rs_parameters, specification)

    @classmethod
    def _from_rs_gate_definition(cls, gate_definition: quil_rs.GateDefinition) -> Self:
        return super().__new__(cls, gate_definition.name, gate_definition.parameters, gate_definition.specification)

    @staticmethod
    def _convert_to_matrix_specification(
        matrix: Union[List[List[Expression]], np.ndarray, np.matrix]
    ) -> quil_rs.GateSpecification:
        to_rs_matrix = np.vectorize(_convert_to_rs_expression, otypes=["O"])
        return quil_rs.GateSpecification.from_matrix(to_rs_matrix(np.asarray(matrix)))

    @staticmethod
    def _validate_matrix(
        matrix: Union[List[List[Expression]], np.ndarray, np.matrix], contains_parameters: bool
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
            raise ValueError("Dimension of matrix must be a power of 2, got {0}".format(rows))

        if not contains_parameters:
            np_matrix = np.asarray(matrix)
            is_unitary = np.allclose(np.eye(rows), np_matrix.dot(np_matrix.T.conj()))
            if not is_unitary:
                raise ValueError("Matrix must be unitary.")

    def out(self) -> str:
        return super().to_quil()

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
        to_py_matrix = np.vectorize(_convert_to_py_expression, otypes=["O"])
        return to_py_matrix(np.asarray(super().specification.to_matrix()))  # type: ignore[no-any-return]

    @matrix.setter
    def matrix(self, matrix: np.ndarray) -> None:
        quil_rs.GateDefinition.specification.__set__(self, DefGate._convert_to_matrix_specification(matrix))  # type: ignore[attr-defined] # noqa

    @property  # type: ignore[override]
    def parameters(self) -> List[Parameter]:
        return [Parameter(name) for name in super().parameters]

    @parameters.setter
    def parameters(self, parameters: Optional[List[Parameter]]) -> None:
        quil_rs.GateDefinition.parameters.__set__(self, [param.name for param in parameters or []])  # type: ignore[attr-defined] # noqa

    def __hash__(self) -> int:
        return hash(self.out())

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class DefPermutationGate(DefGate):
    def __new__(cls, name: str, permutation: Union[List[int], np.ndarray]) -> Self:
        specification = DefPermutationGate._convert_to_permutation_specification(permutation)
        gate_definition = quil_rs.GateDefinition(name, [], specification)
        return super()._from_rs_gate_definition(gate_definition)

    @staticmethod
    def _convert_to_permutation_specification(permutation: Union[List[int], np.ndarray]) -> quil_rs.GateSpecification:
        return quil_rs.GateSpecification.from_permutation([int(x) for x in permutation])

    @property
    def permutation(self) -> List[int]:
        return super().specification.to_permutation()

    @permutation.setter
    def permutation(self, permutation: List[int]) -> None:
        specification = DefPermutationGate._convert_to_permutation_specification(permutation)
        quil_rs.GateDefinition.specification.__set__(self, specification)  # type: ignore[attr-defined]

    def num_args(self) -> int:
        """
        :return: The number of qubit arguments the gate takes.
        """
        return int(np.log2(len(self.permutation)))

    def __str__(self) -> str:
        return super().to_quil_or_debug()


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
    ) -> Self:
        specification = DefGateByPaulis._convert_to_pauli_specification(body, arguments)
        rs_parameters = [param.name for param in parameters]
        gate_definition = quil_rs.GateDefinition(gate_name, rs_parameters, specification)
        return super()._from_rs_gate_definition(gate_definition)

    @staticmethod
    def _convert_to_pauli_specification(
        body: "PauliSum", arguments: List[QubitDesignator]
    ) -> quil_rs.GateSpecification:
        if isinstance(body, Sequence):
            from pyquil.paulis import PauliSum

            body = PauliSum(body)
        return quil_rs.GateSpecification.from_pauli_sum(body._to_rs_pauli_sum(arguments))

    @property
    def arguments(self) -> List[FormalArgument]:
        return [FormalArgument(arg) for arg in super().specification.to_pauli_sum().arguments]

    @arguments.setter
    def arguments(self, arguments: List[QubitDesignator]) -> None:
        pauli_sum = super().specification.to_pauli_sum()
        pauli_sum.arguments = [str(arg) for arg in arguments]
        quil_rs.GateDefinition.specification.__set__(self, quil_rs.GateSpecification.from_pauli_sum(pauli_sum))  # type: ignore[attr-defined] # noqa

    @property
    def body(self) -> "PauliSum":
        from pyquil.paulis import PauliSum  # avoids circular import

        return PauliSum._from_rs_pauli_sum(super().specification.to_pauli_sum())

    @body.setter
    def body(self, body: "PauliSum") -> None:
        specification = quil_rs.GateSpecification.from_pauli_sum(body._to_rs_pauli_sum())
        quil_rs.GateDefinition.specification.__set__(self, specification)  # type: ignore[attr-defined]

    def num_args(self) -> int:
        return len(self.arguments)

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class JumpTarget(quil_rs.Label, AbstractInstruction):
    """
    Representation of a target that can be jumped to.
    """

    def __new__(cls, label: Union[Label, LabelPlaceholder]) -> Self:
        return super().__new__(cls, label.target)

    @classmethod
    def _from_rs_label(cls, label: quil_rs.Label) -> "JumpTarget":
        return super().__new__(cls, label.target)

    @property
    def label(self) -> Union[Label, LabelPlaceholder]:
        if super().target.is_placeholder():
            return LabelPlaceholder._from_rs_target(super().target)
        return Label._from_rs_target(super().target)

    def __repr__(self) -> str:
        return "<JumpTarget {0}>".format(str(self.label))

    def out(self) -> str:
        return super().to_quil()


class JumpWhen(quil_rs.JumpWhen, AbstractInstruction):
    """
    The JUMP-WHEN instruction.
    """

    def __new__(cls, target: Union[Label, LabelPlaceholder], condition: MemoryReference) -> Self:
        return super().__new__(cls, target.target, condition._to_rs_memory_reference())

    @classmethod
    def _from_rs_jump_when(cls, jump_when: quil_rs.JumpWhen) -> Self:
        return super().__new__(cls, jump_when.target, jump_when.condition)

    def out(self) -> str:
        return super().to_quil()

    @property  # type: ignore[override]
    def condition(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().condition)

    @condition.setter
    def condition(self, condition: MemoryReference) -> None:
        quil_rs.JumpWhen.condition.__set__(self, condition._to_rs_memory_reference())  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def target(self) -> Union[Label, LabelPlaceholder]:
        if super().target.is_placeholder():
            return LabelPlaceholder._from_rs_target(super().target)
        return Label._from_rs_target(super().target)

    @target.setter
    def target(self, target: Union[Label, LabelPlaceholder]) -> None:
        quil_rs.JumpWhen.target.__set__(self, target)  # type: ignore[attr-defined]

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class JumpUnless(quil_rs.JumpUnless, AbstractInstruction):
    """
    The JUMP-UNLESS instruction.
    """

    def __new__(cls, target: Union[Label, LabelPlaceholder], condition: MemoryReference) -> Self:
        return super().__new__(cls, target.target, condition._to_rs_memory_reference())

    @classmethod
    def _from_rs_jump_unless(cls, jump_unless: quil_rs.JumpUnless) -> Self:
        return super().__new__(cls, jump_unless.target, jump_unless.condition)

    def out(self) -> str:
        return super().to_quil()

    @property  # type: ignore[override]
    def condition(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().condition)

    @condition.setter
    def condition(self, condition: MemoryReference) -> None:
        quil_rs.JumpUnless.condition.__set__(self, condition._to_rs_memory_reference())  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def target(self) -> Union[Label, LabelPlaceholder]:
        if super().target.is_placeholder():
            return LabelPlaceholder._from_rs_target(super().target)
        return Label._from_rs_target(super().target)

    @target.setter
    def target(self, target: Union[Label, LabelPlaceholder]) -> None:
        quil_rs.JumpUnless.target.__set__(self, target)  # type: ignore[attr-defined]

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class SimpleInstruction(AbstractInstruction):
    """
    Abstract class for simple instructions with no arguments.
    """

    instruction: ClassVar[quil_rs.Instruction]

    def out(self) -> str:
        return self.instruction.to_quil()

    def __str__(self) -> str:
        return self.out()


class Halt(SimpleInstruction):
    """
    The HALT instruction.
    """

    instruction = quil_rs.Instruction.new_halt()


class Wait(SimpleInstruction):
    """
    The WAIT instruction.
    """

    instruction = quil_rs.Instruction.new_wait()


class Nop(SimpleInstruction):
    """
    The NOP instruction.
    """

    instruction = quil_rs.Instruction.new_nop()


class UnaryClassicalInstruction(quil_rs.UnaryLogic, AbstractInstruction):
    """
    The abstract class for unary classical instructions.
    """

    op: ClassVar[quil_rs.UnaryOperator]

    def __new__(cls, target: MemoryReference) -> "UnaryClassicalInstruction":
        return super().__new__(cls, cls.op, target._to_rs_memory_reference())

    @property
    def target(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().operand)

    @target.setter
    def target(self, target: MemoryReference) -> None:
        quil_rs.UnaryLogic.operand.__set__(self, target._to_rs_memory_reference())  # type: ignore

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalNeg(UnaryClassicalInstruction):
    """
    The NEG instruction.
    """

    op = quil_rs.UnaryOperator.Neg


class ClassicalNot(UnaryClassicalInstruction):
    """
    The NOT instruction.
    """

    op = quil_rs.UnaryOperator.Not


class LogicalBinaryOp(quil_rs.BinaryLogic, AbstractInstruction):
    """
    The abstract class for binary logical classical instructions.
    """

    op: ClassVar[quil_rs.BinaryOperator]

    def __new__(cls, left: MemoryReference, right: Union[MemoryReference, int]) -> Self:
        operands = cls._to_rs_binary_operands(left, right)
        return super().__new__(cls, cls.op, operands)

    @staticmethod
    def _to_rs_binary_operand(operand: Union[MemoryReference, int]) -> quil_rs.BinaryOperand:
        if isinstance(operand, MemoryReference):
            return quil_rs.BinaryOperand.from_memory_reference(operand._to_rs_memory_reference())
        return quil_rs.BinaryOperand.from_literal_integer(operand)

    @staticmethod
    def _to_rs_binary_operands(left: MemoryReference, right: Union[MemoryReference, int]) -> quil_rs.BinaryOperands:
        left_operand = left._to_rs_memory_reference()
        right_operand = LogicalBinaryOp._to_rs_binary_operand(right)
        return quil_rs.BinaryOperands(left_operand, right_operand)

    @staticmethod
    def _to_py_binary_operand(operand: quil_rs.BinaryOperand) -> Union[MemoryReference, int]:
        if operand.is_literal_integer():
            return operand.to_literal_integer()
        return MemoryReference._from_rs_memory_reference(operand.to_memory_reference())

    @property
    def left(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().operands.memory_reference)

    @left.setter
    def left(self, left: MemoryReference) -> None:
        operands = super().operands
        operands.memory_reference = left._to_rs_memory_reference()
        quil_rs.BinaryLogic.operands.__set__(self, operands)  # type: ignore[attr-defined]

    @property
    def right(self) -> Union[MemoryReference, int]:
        return self._to_py_binary_operand(super().operands.operand)

    @right.setter
    def right(self, right: Union[MemoryReference, int]) -> None:
        operands = super().operands
        operands.operand = self._to_rs_binary_operand(right)
        quil_rs.BinaryLogic.operands.__set__(self, operands)  # type: ignore[attr-defined]

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalAnd(LogicalBinaryOp):
    """
    The AND instruction.
    """

    op = quil_rs.BinaryOperator.And


class ClassicalInclusiveOr(LogicalBinaryOp):
    """
    The IOR instruction.
    """

    op = quil_rs.BinaryOperator.Ior


class ClassicalExclusiveOr(LogicalBinaryOp):
    """
    The XOR instruction.
    """

    op = quil_rs.BinaryOperator.Xor


class ArithmeticBinaryOp(quil_rs.Arithmetic, AbstractInstruction):
    """
    The abstract class for binary arithmetic classical instructions.
    """

    op: ClassVar[quil_rs.ArithmeticOperator]

    def __new__(cls, left: MemoryReference, right: Union[MemoryReference, int, float]) -> Self:
        left_operand = quil_rs.ArithmeticOperand.from_memory_reference(left._to_rs_memory_reference())
        right_operand = _to_rs_arithmetic_operand(right)
        return super().__new__(cls, cls.op, left_operand, right_operand)

    @property
    def left(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().destination.to_memory_reference())

    @left.setter
    def left(self, left: MemoryReference) -> None:
        quil_rs.Arithmetic.destination.__set__(  # type: ignore[attr-defined]
            self, quil_rs.ArithmeticOperand.from_memory_reference(left._to_rs_memory_reference())
        )

    @property
    def right(self) -> Union[MemoryReference, int, float]:
        return _to_py_arithmetic_operand(super().source)

    @right.setter
    def right(self, right: Union[MemoryReference, int, float]) -> None:
        quil_rs.Arithmetic.source.__set__(self, _to_rs_arithmetic_operand(right))  # type: ignore[attr-defined]

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalAdd(ArithmeticBinaryOp):
    """
    The ADD instruction.
    """

    op = quil_rs.ArithmeticOperator.Add


class ClassicalSub(ArithmeticBinaryOp):
    """
    The SUB instruction.
    """

    op = quil_rs.ArithmeticOperator.Subtract


class ClassicalMul(ArithmeticBinaryOp):
    """
    The MUL instruction.
    """

    op = quil_rs.ArithmeticOperator.Multiply


class ClassicalDiv(ArithmeticBinaryOp):
    """
    The DIV instruction.
    """

    op = quil_rs.ArithmeticOperator.Divide


class ClassicalMove(quil_rs.Move, AbstractInstruction):
    """
    The MOVE instruction.
    """

    def __new__(cls, left: MemoryReference, right: Union[MemoryReference, int, float]) -> "ClassicalMove":
        return super().__new__(cls, left._to_rs_memory_reference(), _to_rs_arithmetic_operand(right))

    @classmethod
    def _from_rs_move(cls, move: quil_rs.Move) -> Self:
        return super().__new__(cls, move.destination, move.source)

    @property
    def left(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().destination)

    @left.setter
    def left(self, left: MemoryReference) -> None:
        quil_rs.Move.destination.__set__(self, left._to_rs_memory_reference())  # type: ignore

    @property
    def right(self) -> Union[MemoryReference, int, float]:
        return _to_py_arithmetic_operand(super().source)

    @right.setter
    def right(self, right: Union[MemoryReference, int, float]) -> None:
        quil_rs.Move.source.__set__(self, _to_rs_arithmetic_operand(right))  # type: ignore

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalExchange(quil_rs.Exchange, AbstractInstruction):
    """
    The EXCHANGE instruction.
    """

    def __new__(
        cls,
        left: MemoryReference,
        right: MemoryReference,
    ) -> "ClassicalExchange":
        return super().__new__(cls, left._to_rs_memory_reference(), right._to_rs_memory_reference())

    @property  # type: ignore[override]
    def left(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().left)

    @left.setter
    def left(self, left: MemoryReference) -> None:
        quil_rs.Exchange.left.__set__(self, left._to_rs_memory_reference())  # type: ignore

    @property  # type: ignore[override]
    def right(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().right)

    @right.setter
    def right(self, right: MemoryReference) -> None:
        quil_rs.Exchange.right.__set__(self, right._to_rs_memory_reference())  # type: ignore

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalConvert(quil_rs.Convert, AbstractInstruction):
    """
    The CONVERT instruction.
    """

    def __new__(cls, left: MemoryReference, right: MemoryReference) -> "ClassicalConvert":
        return super().__new__(cls, left._to_rs_memory_reference(), right._to_rs_memory_reference())

    @property
    def left(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().destination)

    @left.setter
    def left(self, memory_reference: MemoryReference) -> None:
        quil_rs.Convert.destination.__set__(self, memory_reference._to_rs_memory_reference())  # type: ignore

    @property
    def right(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().source)

    @right.setter
    def right(self, memory_reference: MemoryReference) -> None:
        quil_rs.Convert.source.__set__(self, memory_reference._to_rs_memory_reference())  # type: ignore

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalLoad(quil_rs.Load, AbstractInstruction):
    """
    The LOAD instruction.
    """

    def __new__(cls, target: MemoryReference, left: str, right: MemoryReference) -> "ClassicalLoad":
        return super().__new__(cls, target._to_rs_memory_reference(), left, right._to_rs_memory_reference())

    @property
    def target(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().destination)

    @target.setter
    def target(self, target: MemoryReference) -> None:
        quil_rs.Load.destination.__set__(self, target._to_rs_memory_reference())  # type: ignore

    @property
    def left(self) -> str:
        return super().source

    @left.setter
    def left(self, left: str) -> None:
        quil_rs.Load.source.__set__(self, left)  # type: ignore

    @property
    def right(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().offset)

    @right.setter
    def right(self, right: MemoryReference) -> None:
        quil_rs.Load.offset.__set__(self, right._to_rs_memory_reference())  # type: ignore

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


def _to_rs_arithmetic_operand(operand: Union[MemoryReference, int, float]) -> quil_rs.ArithmeticOperand:
    if isinstance(operand, MemoryReference):
        return quil_rs.ArithmeticOperand.from_memory_reference(operand._to_rs_memory_reference())
    if isinstance(operand, int):
        return quil_rs.ArithmeticOperand.from_literal_integer(operand)
    if isinstance(operand, float):
        return quil_rs.ArithmeticOperand.from_literal_real(operand)
    raise TypeError(f"{type(operand)} is not a valid ArithmeticOperand")


def _to_py_arithmetic_operand(operand: quil_rs.ArithmeticOperand) -> Union[MemoryReference, int, float]:
    if not isinstance(operand, quil_rs.ArithmeticOperand):
        raise TypeError(f"{type(operand)} is not an ArithmeticOperand")
    inner = operand.inner()
    if isinstance(inner, quil_rs.MemoryReference):
        return MemoryReference._from_rs_memory_reference(inner)
    return inner


class ClassicalStore(quil_rs.Store, AbstractInstruction):
    """
    The STORE instruction.
    """

    def __new__(cls, target: str, left: MemoryReference, right: Union[MemoryReference, int, float]) -> "ClassicalStore":
        rs_right = _to_rs_arithmetic_operand(right)
        return super().__new__(cls, target, left._to_rs_memory_reference(), rs_right)

    @property
    def target(self) -> str:
        return super().destination

    @target.setter
    def target(self, target: str) -> None:
        quil_rs.Store.destination.__set__(self, target)  # type: ignore

    @property
    def left(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().offset)

    @left.setter
    def left(self, left: MemoryReference) -> None:
        quil_rs.Store.offset.__set__(self, left._to_rs_memory_reference())  # type: ignore

    @property
    def right(self) -> Union[MemoryReference, int, float]:
        return _to_py_arithmetic_operand(super().source)

    @right.setter
    def right(self, right: Union[MemoryReference, int, float]) -> None:
        quil_rs.Store.source.__set__(self, _to_rs_arithmetic_operand(right))  # type: ignore

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalComparison(quil_rs.Comparison, AbstractInstruction):
    """
    Abstract class for ternary comparison instructions.
    """

    op: ClassVar[quil_rs.ComparisonOperator]

    def __new__(
        cls,
        target: MemoryReference,
        left: MemoryReference,
        right: Union[MemoryReference, int, float],
    ) -> "ClassicalComparison":
        operands = (target._to_rs_memory_reference(), left._to_rs_memory_reference(), cls._to_comparison_operand(right))
        return super().__new__(cls, cls.op, operands)

    @staticmethod
    def _to_comparison_operand(operand: Union[MemoryReference, int, float]) -> quil_rs.ComparisonOperand:
        if isinstance(operand, MemoryReference):
            return quil_rs.ComparisonOperand.from_memory_reference(operand._to_rs_memory_reference())
        elif isinstance(operand, int):
            return quil_rs.ComparisonOperand.from_literal_integer(operand)
        elif isinstance(operand, float):
            return quil_rs.ComparisonOperand.from_literal_real(operand)
        raise TypeError(f"{type(operand)} is not a valid ComparisonOperand")

    @staticmethod
    def _to_py_operand(operand: quil_rs.ComparisonOperand) -> Union[MemoryReference, int, float]:
        if not isinstance(operand, quil_rs.ComparisonOperand):
            raise TypeError(f"{type(operand)} is not an ComparisonOperand")
        inner = operand.inner()
        if isinstance(inner, quil_rs.MemoryReference):
            return MemoryReference._from_rs_memory_reference(inner)
        return inner

    @property
    def target(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().operands[0])

    @target.setter
    def target(self, target: MemoryReference) -> None:
        operands = list(super().operands)
        operands[0] = target._to_rs_memory_reference()
        quil_rs.Comparison.operands.__set__(self, tuple(operands))  # type: ignore

    @property
    def left(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().operands[1])

    @left.setter
    def left(self, left: MemoryReference) -> None:
        operands = list(super().operands)
        operands[1] = left._to_rs_memory_reference()
        quil_rs.Comparison.operands.__set__(self, tuple(operands))  # type: ignore

    @property
    def right(self) -> Union[MemoryReference, int, float]:
        return self._to_py_operand(super().operands[2])

    @right.setter
    def right(self, right: MemoryReference) -> None:
        operands = list(super().operands)
        operands[2] = self._to_comparison_operand(right)
        quil_rs.Comparison.operands.__set__(self, tuple(operands))  # type: ignore

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class ClassicalEqual(ClassicalComparison):
    """
    The EQ comparison instruction.
    """

    op = quil_rs.ComparisonOperator.Equal


class ClassicalLessThan(ClassicalComparison):
    """
    The LT comparison instruction.
    """

    op = quil_rs.ComparisonOperator.LessThan


class ClassicalLessEqual(ClassicalComparison):
    """
    The LE comparison instruction.
    """

    op = quil_rs.ComparisonOperator.LessThanOrEqual


class ClassicalGreaterThan(ClassicalComparison):
    """
    The GT comparison instruction.
    """

    op = quil_rs.ComparisonOperator.GreaterThan


class ClassicalGreaterEqual(ClassicalComparison):
    """
    The GE comparison instruction.
    """

    op = quil_rs.ComparisonOperator.GreaterThanOrEqual


class Jump(quil_rs.Jump, AbstractInstruction):
    """
    Representation of an unconditional jump instruction (JUMP).
    """

    def __new__(cls, target: Union[Label, LabelPlaceholder]) -> Self:
        return super().__new__(cls, target.target)

    @classmethod
    def _from_rs_jump(cls, jump: quil_rs.Jump) -> Self:
        return super().__new__(cls, jump.target)

    @property  # type: ignore[override]
    def target(self) -> Union[Label, LabelPlaceholder]:
        if super().target.is_placeholder():
            return LabelPlaceholder._from_rs_target(super().target)
        return Label._from_rs_target(super().target)

    @target.setter
    def target(self, target: Union[Label, LabelPlaceholder]) -> None:
        quil_rs.Jump.target.__set__(self, target.target)  # type: ignore[attr-defined]

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class Pragma(quil_rs.Pragma, AbstractInstruction):
    """
    A PRAGMA instruction.

    This is printed in QUIL as:

        PRAGMA <command> <arg1> <arg2> ... <argn> "<freeform_string>"

    """

    def __new__(
        cls,
        command: str,
        args: Sequence[Union[Qubit, FormalArgument, int, str]] = (),
        freeform_string: str = "",
    ) -> Self:
        data = freeform_string or None
        return super().__new__(cls, command, Pragma._to_pragma_arguments(args), data)

    @classmethod
    def _from_rs_pragma(cls, pragma: quil_rs.Pragma) -> "Pragma":
        return super().__new__(cls, pragma.name, pragma.arguments, pragma.data)

    @staticmethod
    def _to_pragma_arguments(args: Sequence[Union[QubitDesignator, str]]) -> List[quil_rs.PragmaArgument]:
        pragma_arguments = []
        for arg in args:
            if isinstance(arg, Qubit):
                pragma_arguments.append(quil_rs.PragmaArgument.from_integer(arg.index))
            elif isinstance(arg, int):
                pragma_arguments.append(quil_rs.PragmaArgument.from_integer(arg))
            elif isinstance(arg, (str, FormalArgument)):
                pragma_arguments.append(quil_rs.PragmaArgument.from_identifier(str(arg)))
            else:
                raise TypeError(f"{type(arg)} isn't a valid PRAGMA argument")
        return pragma_arguments

    @staticmethod
    def _to_py_arguments(args: List[quil_rs.PragmaArgument]) -> List[QubitDesignator]:
        arguments: List[QubitDesignator] = []
        for arg in args:
            if arg.is_integer():
                arguments.append(Qubit(arg.to_integer()))
            else:
                arguments.append(FormalArgument(arg.to_identifier()))
        return arguments

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property
    def command(self) -> str:
        return super().name

    @command.setter
    def command(self, command: str) -> None:
        quil_rs.Pragma.name.__set__(self, command)  # type: ignore[attr-defined]

    @property
    def args(self) -> Tuple[QubitDesignator]:
        return tuple(Pragma._to_py_arguments(super().arguments))  # type: ignore[return-value]

    @args.setter
    def args(self, args: Sequence[Union[QubitDesignator, str]]) -> None:
        quil_rs.Pragma.arguments.__set__(self, Pragma._to_pragma_arguments(args))  # type: ignore[attr-defined]

    @property
    def freeform_string(self) -> str:
        return super().data or ""

    @freeform_string.setter
    def freeform_string(self, freeform_string: str) -> None:
        quil_rs.Pragma.data.__set__(self, freeform_string)  # type: ignore[attr-defined]


class Declare(quil_rs.Declaration, AbstractInstruction):
    """
    A DECLARE directive.

    This is printed in Quil as::

        DECLARE <name> <memory-type> (SHARING <other-name> (OFFSET <amount> <type>)* )?

    """

    def __new__(
        cls,
        name: str,
        memory_type: str,
        memory_size: int = 1,
        shared_region: Optional[str] = None,
        offsets: Optional[Sequence[Tuple[int, str]]] = None,
    ) -> Self:
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
    def _to_rs_offsets(offsets: Optional[Sequence[Tuple[int, str]]]) -> List[quil_rs.Offset]:
        if offsets is None:
            return []
        return [
            quil_rs.Offset(offset, Declare._memory_type_to_scalar_type(memory_type)) for offset, memory_type in offsets
        ]

    @property
    def memory_type(self) -> str:
        return str(super().size.data_type).upper()

    @memory_type.setter
    def memory_type(self, memory_type: str) -> None:
        vector = super().size
        vector.data_type = Declare._memory_type_to_scalar_type(memory_type)
        quil_rs.Declaration.size.__set__(self, vector)  # type: ignore[attr-defined]

    @property
    def memory_size(self) -> int:
        return super().size.length

    @memory_size.setter
    def memory_size(self, memory_size: int) -> None:
        vector = super().size
        vector.length = memory_size
        quil_rs.Declaration.size.__set__(self, vector)  # type: ignore[attr-defined]

    @property
    def shared_region(self) -> Optional[str]:
        sharing = super().sharing
        if sharing is None:
            return None
        return sharing.name

    @shared_region.setter
    def shared_region(self, shared_region: Optional[str]) -> None:
        sharing = None if not shared_region else quil_rs.Sharing(shared_region, [])
        current_sharing = super().sharing
        if sharing and isinstance(current_sharing, quil_rs.Sharing):
            sharing.offsets = current_sharing.offsets
        quil_rs.Declaration.sharing.__set__(self, sharing)  # type: ignore[attr-defined]

    @property
    def offsets(self) -> List[Tuple[int, str]]:
        sharing = super().sharing
        if sharing is None:
            return []
        return [(offset.offset, str(offset.data_type).upper()) for offset in sharing.offsets]

    @offsets.setter
    def offsets(self, offsets: Optional[List[Tuple[int, str]]]) -> None:
        sharing = super().sharing
        if sharing is None:
            raise ValueError("DECLARE without a shared region cannot use offsets")
        sharing.offsets = Declare._to_rs_offsets(offsets)
        quil_rs.Declaration.sharing.__set__(self, sharing)  # type: ignore[attr-defined]

    def asdict(self) -> Dict[str, Union[Sequence[Tuple[int, str]], Optional[str], int]]:
        return {
            "name": self.name,
            "memory_type": self.memory_type,
            "memory_size": self.memory_size,
            "shared_region": self.shared_region,
            "offsets": self.offsets,
        }

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class Include(quil_rs.Include, AbstractInstruction):
    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class Pulse(quil_rs.Pulse, AbstractInstruction):
    def __new__(cls, frame: Frame, waveform: Waveform, nonblocking: bool = False) -> Self:
        return super().__new__(cls, not nonblocking, frame, waveform)

    @classmethod
    def _from_rs_pulse(cls, pulse: quil_rs.Pulse) -> "Pulse":
        return super().__new__(cls, pulse.blocking, pulse.frame, pulse.waveform)

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[Set[QubitDesignator], Set[int]]:
        if indices:
            return self.get_qubit_indices()
        else:
            return set(_convert_to_py_qubits(super().frame.qubits))

    def get_qubit_indices(self) -> Set[int]:
        return {qubit.to_fixed() for qubit in super().frame.qubits}

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        return Frame._from_rs_frame_identifier(super().frame)

    @frame.setter
    def frame(self, frame: Frame) -> None:
        quil_rs.Pulse.frame.__set__(self, frame)  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def waveform(self) -> Waveform:
        return _convert_to_py_waveform(super().waveform)

    @waveform.setter
    def waveform(self, waveform: Waveform) -> None:
        quil_rs.Pulse.waveform.__set__(self, waveform)  # type: ignore[attr-defined]

    @property
    def nonblocking(self) -> bool:
        return not super().blocking

    @nonblocking.setter
    def nonblocking(self, nonblocking: bool) -> None:
        quil_rs.Pulse.blocking.__set__(self, not nonblocking)  # type: ignore[attr-defined]


class SetFrequency(quil_rs.SetFrequency, AbstractInstruction):
    def __new__(cls, frame: Frame, freq: ParameterDesignator) -> Self:
        return super().__new__(cls, frame, _convert_to_rs_expression(freq))

    @classmethod
    def _from_rs_set_frequency(cls, set_frequency: quil_rs.SetFrequency) -> "SetFrequency":
        return super().__new__(cls, set_frequency.frame, set_frequency.frequency)

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        return Frame._from_rs_frame_identifier(super().frame)

    @frame.setter
    def frame(self, frame: Frame) -> None:
        quil_rs.SetFrequency.frame.__set__(self, frame)  # type: ignore[attr-defined]

    @property
    def freq(self) -> ParameterDesignator:
        return _convert_to_py_expression(super().frequency)

    @freq.setter
    def freq(self, freq: ParameterDesignator) -> None:
        quil_rs.SetFrequency.frequency.__set__(self, _convert_to_rs_expression(freq))  # type: ignore[attr-defined]

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[Set[QubitDesignator], Set[int]]:
        if indices:
            return self.get_qubit_indices()
        return set(self.frame.qubits)

    def get_qubit_indices(self) -> Set[int]:
        return {qubit.to_fixed() for qubit in super().frame.qubits}


class ShiftFrequency(quil_rs.ShiftFrequency, AbstractInstruction):
    def __new__(cls, frame: Frame, freq: ParameterDesignator) -> Self:
        return super().__new__(cls, frame, _convert_to_rs_expression(freq))

    @classmethod
    def _from_rs_shift_frequency(cls, shift_frequency: quil_rs.ShiftFrequency) -> "ShiftFrequency":
        return super().__new__(cls, shift_frequency.frame, shift_frequency.frequency)

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        return Frame._from_rs_frame_identifier(super().frame)

    @frame.setter
    def frame(self, frame: Frame) -> None:
        quil_rs.ShiftFrequency.frame.__set__(self, frame)  # type: ignore[attr-defined]

    @property
    def freq(self) -> ParameterDesignator:
        return _convert_to_py_expression(super().frequency)

    @freq.setter
    def freq(self, freq: ParameterDesignator) -> None:
        quil_rs.ShiftFrequency.frequency.__set__(self, _convert_to_rs_expression(freq))  # type: ignore[attr-defined]

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[Set[QubitDesignator], Set[int]]:
        if indices:
            return self.get_qubit_indices()
        return set(self.frame.qubits)

    def get_qubit_indices(self) -> Set[int]:
        return {qubit.to_fixed() for qubit in super().frame.qubits}


class SetPhase(quil_rs.SetPhase, AbstractInstruction):
    def __new__(cls, frame: Frame, phase: ParameterDesignator) -> Self:
        return super().__new__(cls, frame, _convert_to_rs_expression(phase))

    @classmethod
    def _from_rs_set_phase(cls, set_phase: quil_rs.SetPhase) -> "SetPhase":
        return super().__new__(cls, set_phase.frame, set_phase.phase)

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        return Frame._from_rs_frame_identifier(super().frame)

    @frame.setter
    def frame(self, frame: Frame) -> None:
        quil_rs.SetPhase.frame.__set__(self, frame)  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def phase(self) -> ParameterDesignator:
        return _convert_to_py_expression(super().phase)

    @phase.setter
    def phase(self, phase: ParameterDesignator) -> None:
        quil_rs.SetPhase.phase.__set__(self, _convert_to_rs_expression(phase))  # type: ignore[attr-defined]

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[Set[QubitDesignator], Set[int]]:
        if indices:
            return self.get_qubit_indices()
        return set(self.frame.qubits)

    def get_qubit_indices(self) -> Set[int]:
        return {qubit.to_fixed() for qubit in super().frame.qubits}


class ShiftPhase(quil_rs.ShiftPhase, AbstractInstruction):
    def __new__(cls, frame: Frame, phase: ParameterDesignator) -> Self:
        return super().__new__(cls, frame, _convert_to_rs_expression(phase))

    @classmethod
    def _from_rs_shift_phase(cls, shift_phase: quil_rs.ShiftPhase) -> "ShiftPhase":
        return super().__new__(cls, shift_phase.frame, shift_phase.phase)

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        return Frame._from_rs_frame_identifier(super().frame)

    @frame.setter
    def frame(self, frame: Frame) -> None:
        quil_rs.ShiftPhase.frame.__set__(self, frame)  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def phase(self) -> ParameterDesignator:
        return _convert_to_py_expression(super().phase)

    @phase.setter
    def phase(self, phase: ParameterDesignator) -> None:
        quil_rs.ShiftPhase.phase.__set__(self, _convert_to_rs_expression(phase))  # type: ignore[attr-defined]

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[Set[QubitDesignator], Set[int]]:
        if indices:
            return self.get_qubit_indices()
        return set(self.frame.qubits)

    def get_qubit_indices(self) -> Set[int]:
        return {qubit.to_fixed() for qubit in super().frame.qubits}


class SwapPhases(quil_rs.SwapPhases, AbstractInstruction):
    def __new__(cls, frameA: Frame, frameB: Frame) -> Self:
        return super().__new__(cls, frameA, frameB)

    @classmethod
    def _from_rs_swap_phases(cls, swap_phases: quil_rs.SwapPhases) -> "SwapPhases":
        return super().__new__(cls, swap_phases.frame_1, swap_phases.frame_2)

    @property
    def frameA(self) -> Frame:
        return Frame._from_rs_frame_identifier(super().frame_1)

    @frameA.setter
    def frameA(self, frame: Frame) -> None:
        quil_rs.SwapPhases.frame_1.__set__(self, frame)  # type: ignore[attr-defined]

    @property
    def frameB(self) -> Frame:
        return Frame._from_rs_frame_identifier(super().frame_2)

    @frameB.setter
    def frameB(self, frame: Frame) -> None:
        quil_rs.SwapPhases.frame_2.__set__(self, frame)  # type: ignore[attr-defined]

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[Set[QubitDesignator], Set[int]]:
        if indices:
            return self.get_qubit_indices()
        return set(self.frameA.qubits) | set(self.frameB.qubits)

    def get_qubit_indices(self) -> Set[int]:
        return {qubit.to_fixed() for qubit in super().frame_1.qubits + super().frame_2.qubits}


class SetScale(quil_rs.SetScale, AbstractInstruction):
    def __new__(cls, frame: Frame, scale: ParameterDesignator) -> Self:
        return super().__new__(cls, frame, _convert_to_rs_expression(scale))

    @classmethod
    def _from_rs_set_scale(cls, set_scale: quil_rs.SetScale) -> "SetScale":
        return super().__new__(cls, set_scale.frame, set_scale.scale)

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        return Frame._from_rs_frame_identifier(super().frame)

    @frame.setter
    def frame(self, frame: Frame) -> None:
        quil_rs.SetScale.frame.__set__(self, frame)  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def scale(self) -> ParameterDesignator:
        return _convert_to_py_expression(super().scale)

    @scale.setter
    def scale(self, scale: ParameterDesignator) -> None:
        quil_rs.SetScale.scale.__set__(self, _convert_to_rs_expression(scale))  # type: ignore[attr-defined]

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[Set[QubitDesignator], Set[int]]:
        if indices:
            return self.get_qubit_indices()
        return set(self.frame.qubits)

    def get_qubit_indices(self) -> Set[int]:
        return {qubit.to_fixed() for qubit in super().frame.qubits}


class Capture(quil_rs.Capture, AbstractInstruction):
    def __new__(
        cls,
        frame: Frame,
        kernel: Waveform,
        memory_region: MemoryReference,
        nonblocking: bool = False,
    ) -> Self:
        rs_memory_reference = _convert_to_rs_expression(memory_region).to_address()
        return super().__new__(cls, not nonblocking, frame, rs_memory_reference, kernel)

    @classmethod
    def _from_rs_capture(cls, capture: quil_rs.Capture) -> "Capture":
        return super().__new__(cls, capture.blocking, capture.frame, capture.memory_reference, capture.waveform)

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        return Frame._from_rs_frame_identifier(super().frame)

    @frame.setter
    def frame(self, frame: Frame) -> None:
        quil_rs.Capture.frame.__set__(self, frame)  # type: ignore[attr-defined]

    @property
    def kernel(self) -> Waveform:
        return _convert_to_py_waveform(super().waveform)

    @kernel.setter
    def kernel(self, kernel: Waveform) -> None:
        quil_rs.Capture.waveform.__set__(self, kernel)  # type: ignore[attr-defined]

    @property
    def memory_region(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().memory_reference)

    @memory_region.setter
    def memory_region(self, memory_region: MemoryReference) -> None:
        rs_memory_reference = _convert_to_rs_expression(memory_region).to_address()
        quil_rs.Capture.memory_reference.__set__(self, rs_memory_reference)  # type: ignore[attr-defined]

    @property
    def nonblocking(self) -> bool:
        return not super().blocking

    @nonblocking.setter
    def nonblocking(self, nonblocking: bool) -> None:
        quil_rs.Capture.blocking.__set__(self, not nonblocking)  # type: ignore[attr-defined]

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[Set[QubitDesignator], Set[int]]:
        if indices:
            return self.get_qubit_indices()
        else:
            return set(_convert_to_py_qubits(super().frame.qubits))

    def get_qubit_indices(self) -> Set[int]:
        return {qubit.to_fixed() for qubit in super().frame.qubits}


class RawCapture(quil_rs.RawCapture, AbstractInstruction):
    def __new__(
        cls,
        frame: Frame,
        duration: float,
        memory_region: MemoryReference,
        nonblocking: bool = False,
    ) -> Self:
        rs_duration = _convert_to_rs_expression(duration)
        rs_memory_reference = _convert_to_rs_expression(memory_region).to_address()
        return super().__new__(cls, not nonblocking, frame, rs_duration, rs_memory_reference)

    @classmethod
    def _from_rs_raw_capture(cls, raw_capture: quil_rs.RawCapture) -> "RawCapture":
        return super().__new__(
            cls, raw_capture.blocking, raw_capture.frame, raw_capture.duration, raw_capture.memory_reference
        )

    @property  # type: ignore[override]
    def frame(self) -> Frame:
        return Frame._from_rs_frame_identifier(super().frame)

    @frame.setter
    def frame(self, frame: Frame) -> None:
        quil_rs.RawCapture.frame.__set__(self, frame)  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def duration(self) -> complex:
        return super().duration.to_number()

    @duration.setter
    def duration(self, duration: float) -> None:
        rs_duration = _convert_to_rs_expression(duration)
        quil_rs.RawCapture.duration.__set__(self, rs_duration)  # type: ignore[attr-defined]

    @property
    def memory_region(self) -> MemoryReference:
        return MemoryReference._from_rs_memory_reference(super().memory_reference)

    @memory_region.setter
    def memory_region(self, memory_region: MemoryReference) -> None:
        rs_memory_reference = _convert_to_rs_expression(memory_region).to_address()
        quil_rs.RawCapture.memory_reference.__set__(self, rs_memory_reference)  # type: ignore[attr-defined]

    @property
    def nonblocking(self) -> bool:
        return not super().blocking

    @nonblocking.setter
    def nonblocking(self, nonblocking: bool) -> None:
        quil_rs.RawCapture.blocking.__set__(self, not nonblocking)  # type: ignore[attr-defined]

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed, use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[Set[QubitDesignator], Set[int]]:
        if indices:
            return self.get_qubit_indices()
        else:
            return set(_convert_to_py_qubits(super().frame.qubits))

    def get_qubit_indices(self) -> Set[int]:
        return {qubit.to_fixed() for qubit in super().frame.qubits}


class Delay(quil_rs.Delay, AbstractInstruction):
    def __new__(cls, frames: List[Frame], qubits: Sequence[Union[int, Qubit, FormalArgument]], duration: float) -> Self:
        frame_names = [frame.name for frame in frames]
        rs_qubits = _convert_to_rs_qubits(Delay._join_frame_qubits(frames, list(qubits)))
        expression = quil_rs_expr.Expression.from_number(complex(duration))
        return super().__new__(cls, expression, frame_names, rs_qubits)

    @classmethod
    def _from_rs_delay(cls, delay: quil_rs.Delay) -> "Delay":
        return super().__new__(cls, delay.duration, delay.frame_names, delay.qubits)

    @staticmethod
    def _join_frame_qubits(
        frames: Sequence[Frame], qubits: Sequence[Union[int, Qubit, FormalArgument]]
    ) -> List[Union[int, Qubit, FormalArgument]]:
        merged_qubits = set(qubits)
        for frame in frames:
            merged_qubits.update(frame.qubits)  # type: ignore
        return list(merged_qubits)

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def qubits(self) -> List[QubitDesignator]:
        return _convert_to_py_qubits(super().qubits)

    @qubits.setter
    def qubits(self, qubits: Sequence[Union[int, Qubit, FormalArgument]]) -> None:
        quil_rs.Delay.qubits.__set__(self, _convert_to_rs_qubits(qubits))  # type: ignore

    @property
    def frames(self) -> List[Frame]:
        return [Frame(self.qubits, name) for name in super().frame_names]

    @frames.setter
    def frames(self, frames: List[Frame]) -> None:
        new_qubits = Delay._join_frame_qubits(frames, [])
        frame_names = [frame.name for frame in frames]
        quil_rs.Delay.qubits.__set__(self, _convert_to_rs_qubits(new_qubits))  # type: ignore[attr-defined]
        quil_rs.Delay.frame_names.__set__(self, frame_names)  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def duration(self) -> float:
        return super().duration.to_real()

    @duration.setter
    def duration(self, duration: float) -> None:
        expression = quil_rs_expr.Expression.from_number(complex(duration))
        quil_rs.Delay.duration.__set__(self, expression)  # type: ignore[attr-defined]


class DelayFrames(Delay):
    def __new__(cls, frames: List[Frame], duration: float) -> Self:
        return super().__new__(cls, frames, [], duration)


class DelayQubits(Delay):
    def __new__(cls, qubits: Sequence[Union[Qubit, FormalArgument]], duration: float) -> Self:
        return super().__new__(cls, [], qubits, duration)


class Fence(quil_rs.Fence, AbstractInstruction):
    def __new__(cls, qubits: List[Union[Qubit, FormalArgument]]) -> Self:
        return super().__new__(cls, _convert_to_rs_qubits(qubits))

    @classmethod
    def _from_rs_fence(cls, fence: quil_rs.Fence) -> "Fence":
        return super().__new__(cls, fence.qubits)

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def qubits(self) -> List[QubitDesignator]:
        return _convert_to_py_qubits(super().qubits)

    @qubits.setter
    def qubits(self, qubits: List[Union[Qubit, FormalArgument]]) -> None:
        quil_rs.Fence.qubits.__set__(self, _convert_to_rs_qubits(qubits))  # type: ignore[attr-defined]


class FenceAll(Fence):
    """
    The FENCE instruction.
    """

    def __new__(cls) -> Self:
        return super().__new__(cls, [])


class DefWaveform(quil_rs.WaveformDefinition, AbstractInstruction):
    def __new__(
        cls,
        name: str,
        parameters: List[Parameter],
        entries: List[Union[complex, Expression]],
    ) -> Self:
        rs_waveform = DefWaveform._build_rs_waveform(parameters, entries)
        return super().__new__(cls, name, rs_waveform)

    @classmethod
    def _from_rs_waveform_definition(cls, waveform_definition: quil_rs.WaveformDefinition) -> "DefWaveform":
        return super().__new__(cls, waveform_definition.name, waveform_definition.definition)

    @staticmethod
    def _build_rs_waveform(parameters: List[Parameter], entries: List[Union[complex, Expression]]) -> quil_rs.Waveform:
        rs_parameters = [parameter.name for parameter in parameters]
        return quil_rs.Waveform(_convert_to_rs_expressions(entries), rs_parameters)

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property
    def parameters(self) -> List[Parameter]:
        return [Parameter(parameter) for parameter in super().definition.parameters]

    @parameters.setter
    def parameters(self, parameters: List[Parameter]) -> None:
        waveform = super().definition
        waveform.parameters = [parameter.name for parameter in parameters]
        quil_rs.WaveformDefinition.definition.__set__(self, waveform)  # type: ignore[attr-defined]

    @property
    def entries(self) -> Sequence[ParameterDesignator]:
        return _convert_to_py_expressions(super().definition.matrix)

    @entries.setter
    def entries(self, entries: List[Union[complex, Expression]]) -> None:
        waveform = super().definition
        waveform.matrix = _convert_to_rs_expressions(entries)
        quil_rs.WaveformDefinition.definition.__set__(self, waveform)  # type: ignore[attr-defined]


class DefCircuit(quil_rs.CircuitDefinition, AbstractInstruction):
    def __new__(
        cls,
        name: str,
        parameters: List[Parameter],
        qubits: List[FormalArgument],
        instructions: List[AbstractInstruction],
    ) -> Self:
        rs_parameters = [parameter.name for parameter in parameters]
        rs_qubits = [qubit.name for qubit in qubits]
        rs_instructions = _convert_to_rs_instructions(instructions)
        return super().__new__(cls, name, rs_parameters, rs_qubits, rs_instructions)

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
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property  # type: ignore[override]
    def parameters(self) -> List[Parameter]:
        return [Parameter(parameter) for parameter in super().parameters]

    @parameters.setter
    def parameters(self, parameters: List[Parameter]) -> None:
        rs_parameters = [parameter.name for parameter in parameters]
        quil_rs.CircuitDefinition.parameters.__set__(self, rs_parameters)  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def qubit_variables(self) -> List[FormalArgument]:
        return [FormalArgument(qubit) for qubit in super().qubit_variables]

    @qubit_variables.setter
    def qubit_variables(self, qubits: List[FormalArgument]) -> None:
        rs_qubits = [qubit.name for qubit in qubits]
        quil_rs.CircuitDefinition.qubit_variables.__set__(self, rs_qubits)  # type: ignore[attr-defined]

    @property  # type: ignore[override]
    def instructions(self) -> List[AbstractInstruction]:
        return _convert_to_py_instructions(super().instructions)

    @instructions.setter
    def instructions(self, instructions: List[AbstractInstruction]) -> None:
        rs_instructions = _convert_to_rs_instructions(instructions)
        quil_rs.CircuitDefinition.instructions.__set__(self, rs_instructions)  # type: ignore[attr-defined]


class DefCalibration(quil_rs.Calibration, AbstractInstruction):
    def __new__(
        cls,
        name: str,
        parameters: Sequence[ParameterDesignator],
        qubits: Sequence[Union[Qubit, FormalArgument]],
        instrs: Sequence[AbstractInstruction],
        modifiers: Optional[List[quil_rs.GateModifier]] = None,
    ) -> Self:
        return super().__new__(
            cls,
            name,
            _convert_to_rs_expressions(parameters),
            _convert_to_rs_qubits(qubits),
            _convert_to_rs_instructions(instrs),
            modifiers or [],
        )

    @classmethod
    def _from_rs_calibration(cls, calibration: quil_rs.Calibration) -> "DefCalibration":
        return super().__new__(
            cls,
            calibration.name,
            calibration.parameters,
            calibration.qubits,
            calibration.instructions,
            calibration.modifiers,
        )

    @property  # type: ignore[override]
    def parameters(self) -> Sequence[ParameterDesignator]:
        return _convert_to_py_expressions(super().parameters)

    @parameters.setter
    def parameters(self, parameters: Sequence[ParameterDesignator]) -> None:
        quil_rs.Calibration.parameters.__set__(self, _convert_to_rs_expressions(parameters))  # type: ignore[attr-defined] # noqa

    @property  # type: ignore[override]
    def qubits(self) -> List[QubitDesignator]:
        return _convert_to_py_qubits(super().qubits)

    @qubits.setter
    def qubits(self, qubits: Sequence[QubitDesignator]) -> None:
        quil_rs.Calibration.qubits.__set__(self, _convert_to_rs_qubits(qubits))  # type: ignore[attr-defined]

    @property
    def instrs(self) -> List[AbstractInstruction]:
        return _convert_to_py_instructions(super().instructions)

    @instrs.setter
    def instrs(self, instrs: Sequence[AbstractInstruction]) -> None:
        quil_rs.Calibration.instructions.__set__(self, _convert_to_rs_instructions(instrs))  # type: ignore[attr-defined] # noqa

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class DefMeasureCalibration(quil_rs.MeasureCalibrationDefinition, AbstractInstruction):
    def __new__(
        cls,
        qubit: Optional[Union[Qubit, FormalArgument]],
        memory_reference: MemoryReference,
        instrs: List[AbstractInstruction],
    ) -> Self:
        rs_qubit = None if not qubit else _convert_to_rs_qubit(qubit)
        return super().__new__(
            cls,
            rs_qubit,
            memory_reference.name,
            _convert_to_rs_instructions(instrs),
        )

    @classmethod
    def _from_rs_measure_calibration_definition(
        cls, calibration: quil_rs.MeasureCalibrationDefinition
    ) -> "DefMeasureCalibration":
        return super().__new__(cls, calibration.qubit, calibration.parameter, calibration.instructions)

    @property  # type: ignore[override]
    def qubit(self) -> Optional[QubitDesignator]:
        qubit = super().qubit
        if not qubit:
            return None
        return _convert_to_py_qubit(qubit)

    @qubit.setter
    def qubit(self, qubit: QubitDesignator) -> None:
        quil_rs.MeasureCalibrationDefinition.qubit.__set__(self, _convert_to_rs_qubit(qubit))  # type: ignore[attr-defined] # noqa

    @property
    def memory_reference(self) -> Optional[MemoryReference]:
        return MemoryReference._from_parameter_str(super().parameter)

    @memory_reference.setter
    def memory_reference(self, memory_reference: MemoryReference) -> None:
        quil_rs.MeasureCalibrationDefinition.parameter.__set__(self, memory_reference.name)  # type: ignore[attr-defined] # noqa

    @property
    def instrs(self) -> List[AbstractInstruction]:
        return _convert_to_py_instructions(super().instructions)

    @instrs.setter
    def instrs(self, instrs: List[AbstractInstruction]) -> None:
        quil_rs.MeasureCalibrationDefinition.instructions.__set__(self, _convert_to_rs_instructions(instrs))  # type: ignore[attr-defined] # noqa

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()


class DefFrame(quil_rs.FrameDefinition, AbstractInstruction):
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
        # The quil spec doesn't outline anything for JSON support
        # but it can be used for the hardware_object field.
        # This generates a properly escaped json string
        # then peels off the outer quotation marks, since they
        # are already added to string values on output.
        if hardware_object is not None:
            hardware_object = json.dumps(hardware_object)
            hardware_object = hardware_object[1:-1]
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

    @classmethod
    def _from_rs_frame_definition(cls, def_frame: quil_rs.FrameDefinition) -> "DefFrame":
        return super().__new__(cls, def_frame.identifier, def_frame.attributes)

    @classmethod
    def _from_rs_attribute_values(
        cls, frame: quil_rs.FrameIdentifier, attributes: Dict[str, quil_rs.AttributeValue]
    ) -> "DefFrame":
        return super().__new__(cls, frame, attributes)

    @staticmethod
    def _to_attribute_value(value: Union[str, float]) -> quil_rs.AttributeValue:
        if isinstance(value, str):
            return quil_rs.AttributeValue.from_string(value)
        if isinstance(value, (int, float, complex)):
            return quil_rs.AttributeValue.from_expression(quil_rs_expr.Expression.from_number(complex(value)))
        raise ValueError(f"{type(value)} is not a valid AttributeValue")

    def out(self) -> str:
        return super().to_quil()

    def __str__(self) -> str:
        return super().to_quil_or_debug()

    @property
    def frame(self) -> Frame:
        return Frame._from_rs_frame_identifier(super().identifier)

    @frame.setter
    def frame(self, frame: Frame) -> None:
        quil_rs.FrameDefinition.identifier.__set__(self, frame)  # type: ignore[attr-defined]

    def _set_attribute(self, name: str, value: Union[str, float]) -> None:
        updated = super().attributes
        updated.update({name: DefFrame._to_attribute_value(value)})
        quil_rs.FrameDefinition.attributes.__set__(self, updated)  # type: ignore[attr-defined]

    def _get_attribute(self, name: str) -> Optional[Union[str, float]]:
        value = super().attributes.get(name, None)
        if value is None:
            return None
        if value.is_string():
            return value.to_string()
        return value.to_expression().to_number().real

    @property
    def direction(self) -> Optional[str]:
        return self._get_attribute("DIRECTION")  # type: ignore

    @direction.setter
    def direction(self, direction: str) -> None:
        self._set_attribute("DIRECTION", direction)

    @property
    def initial_frequency(self) -> Optional[float]:
        return self._get_attribute("INITIAL-FREQUENCY")  # type: ignore

    @initial_frequency.setter
    def initial_frequency(self, initial_frequency: float) -> None:
        self._set_attribute("INITIAL-FREQUENCY", initial_frequency)

    @property
    def hardware_object(self) -> Optional[str]:
        return self._get_attribute("HARDWARE-OBJECT")  # type: ignore

    @hardware_object.setter
    def hardware_object(self, hardware_object: str) -> None:
        self._set_attribute("HARDWARE-OBJECT", hardware_object)

    @property
    def sample_rate(self) -> Frame:
        return self._get_attribute("SAMPLE-RATE")  # type: ignore

    @sample_rate.setter
    def sample_rate(self, sample_rate: float) -> None:
        self._set_attribute("SAMPLE-RATE", sample_rate)

    @property
    def center_frequency(self) -> Frame:
        return self._get_attribute("CENTER-FREQUENCY")  # type: ignore

    @center_frequency.setter
    def center_frequency(self, center_frequency: float) -> None:
        self._set_attribute("CENTER-FREQUENCY", center_frequency)
        self._set_attribute("CENTER-FREQUENCY", center_frequency)
