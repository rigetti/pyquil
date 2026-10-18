import copy
import pickle
from collections.abc import Iterable
from math import pi
from numbers import Complex, Number
from typing import Any

import numpy as np
import pytest
import quil
from quil.expression import (
    Address,
    Expression,
    InfixExpression,
    Number,
    Parameter,
    ParameterDesignator,
    quil_cos,
    quil_sin,
)
from quil.instructions import (
    Arithmetic,
    ArithmeticOperator,
    BinaryLogic,
    BinaryOperator,
    CalibrationDefinition,
    Call,
    CallArgument,
    Capture,
    CircuitDefinition,
    Comparison,
    ComparisonOperator,
    Convert,
    Declaration,
    Delay,
    Exchange,
    Fence,
    FixedQubit,
    FormalArgument,
    FrameDefinition,
    FrameIdentifier,
    Gate,
    GateDefinition,
    GateSpecification,
    Halt,
    Include,
    Instruction,
    Label,
    Load,
    MeasureCalibrationDefinition,
    Measurement,
    MemoryReference,
    Move,
    Nop,
    Pragma,
    Pulse,
    Qubit,
    QubitDesignator,
    RawCapture,
    Reset,
    SetFrequency,
    SetPhase,
    SetScale,
    ShiftFrequency,
    ShiftPhase,
    Store,
    SwapPhases,
    UnaryLogic,
    UnaryOperator,
    Wait,
    WaveformDefinition,
    WaveformInvocation,
)
from quil.waveform import Waveform
from syrupy.assertion import SnapshotAssertion

from pyquil.api._compiler import QPUCompiler
from pyquil.gates import X
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import Frame, TemplateWaveform, qubit_index
from pyquil.quilbase import AbstractInstruction, ResetQubit
from pyquil.quiltwaveforms import (
    BoxcarAveragerKernel,
    DragGaussianWaveform,
    ErfSquareWaveform,
    FlatWaveform,
    GaussianWaveform,
    HrmGaussianWaveform,
)


def _unwrap_operand(operand: Any) -> Any:
    """Unwrap an ``ArithmeticOperand``/``BinaryOperand``/``ComparisonOperand``-style enum value.

    These enums wrap bare ``int``/``float``/``MemoryReference`` values in a nested
    ``LiteralInteger``/``LiteralReal``/``MemoryReference`` variant with a ``._0`` field.
    """
    return getattr(operand, "_0", operand)


_OPERAND_WRAPPER_PICKLE_BUG_REASON = (
    "quil-rs's PyO3 bindings don't support copy/deepcopy/pickle for several enum types used as "
    "instruction operands: *Operator enums (e.g. UnaryOperator, BinaryOperator) aren't copyable/"
    "picklable at all, and *Operand-style wrapper enums (e.g. ArithmeticOperand.LiteralInteger) "
    "have a generated __getnewargs__/__new__ that double-wraps the inner value in an extra tuple, "
    "breaking their round-trip too. This is a quil-rs bug, not a pyquil issue."
)


def _unwrap_param(param: Any) -> Any:
    """Unwrap a ``Gate``/``Pulse``-style ``Expression`` parameter for comparison with the original value.

    ``Gate.params`` always returns ``Expression`` instances, wrapping bare ``float``/``MemoryReference``
    values in ``Number``/``Address`` variants. Other ``Expression`` variants
    (e.g. ``InfixExpression``) are returned unchanged, since the pyquil-side arithmetic operators
    already produce the equivalent native ``Expression`` value.
    """
    if isinstance(param, Number):
        value = param.value
        return value.real if value.imag == 0 else value
    if isinstance(param, Address):
        return param.memory_reference
    return param


@pytest.mark.parametrize(
    ("name", "params", "qubits", "declarations"),
    [
        ("X", [], [Qubit(0)], []),
        ("CPHASE", [pi / 2], [Qubit(0), Qubit(1)], []),
        ("RZ", [MemoryReference("theta", 0, 1)], [Qubit(0)], ["theta"]),
        ("RZ", [MemoryReference("alpha", 0) - MemoryReference("beta")], [Qubit(0)], ["alpha", "beta"]),
    ],
    ids=("X-Gate", "CPHASE-Expression", "RZ-MemoryReference", "RZ-MemoryReference-Expression"),
)
class TestGate:
    @pytest.fixture
    def gate(self, name: str, params: list[ParameterDesignator], qubits: list[Qubit], declarations: list[str]) -> Gate:
        return Gate(name, params, qubits)

    @pytest.fixture
    def program(self, gate: Gate, declarations: list[str]) -> Program:
        """Creates a valid quil program using the gate, with declared memory regions."""
        program = Program(gate)
        for name in declarations:
            program.declare(name, "REAL", 1)
        return program

    def test_str(self, gate: Gate, snapshot: SnapshotAssertion):
        assert str(gate) == snapshot

    def test_name(self, gate: Gate, name: str):
        assert gate.name == name

    def test_params(self, gate: Gate, params: list[ParameterDesignator]):
        assert [_unwrap_param(p) for p in gate.params] == params

    def test_qubits(self, gate: Gate, qubits: list[Qubit]):
        assert gate.qubits == qubits

    def test_get_qubits(self, gate: Gate, qubits: list[Qubit]):
        assert gate.get_qubit_indices() == [qubit_index(q) for q in qubits]
        assert gate.qubits == qubits

    def test_controlled_modifier(self, gate: Gate, snapshot: SnapshotAssertion):
        assert str(gate.controlled(Qubit(5))) == snapshot

    def test_dagger_modifier(self, gate: Gate, snapshot: SnapshotAssertion):
        assert str(gate.dagger()) == snapshot

    def test_forked_modifier(self, gate: Gate, params: list[ParameterDesignator], snapshot: SnapshotAssertion):
        alt_params: list[ParameterDesignator] = [n for n in range(len(params))]
        assert str(gate.forked(Qubit(5), alt_params)) == snapshot

    def test_modifiers(self, gate: Gate):
        assert gate.modifiers == []

    def test_repr(self, gate: Gate, snapshot: SnapshotAssertion):
        assert repr(gate) == snapshot

    def test_eq(self, gate: Gate, name: str, params: list[ParameterDesignator], qubits: list[Qubit]):
        assert gate == Gate(name, params, qubits)
        assert not gate != Gate(name, params, qubits)
        not_eq_gate = Gate(f"not-{name}", params, qubits)
        assert not (gate == not_eq_gate)
        assert gate != not_eq_gate

    def test_copy(self, gate: Gate):
        c = copy.copy(gate)
        d = copy.deepcopy(gate)
        assert c == gate
        assert d == gate
        assert isinstance(c, Gate)
        assert isinstance(d, Gate)

    def test_compile(self, program: Program, compiler: QPUCompiler):
        compiler.quil_to_native_quil(program)

    def test_pickle(self, gate: Gate):
        pickled = pickle.dumps(gate)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Gate)
        assert unpickled == gate


# TODO: Restore tests for old constructor to help migration checks.
# TODO(migration-guide): np.diag([Expression(1)] * 4) can't work the way it used to
# if `Expression` is to implement the __array__ interface.

@pytest.mark.parametrize(
    ("name", "parameters", "n_args", "specification"),
    [
        ("NoParamGate", [], 2, np.eye(4)),
        ("ParameterizedGate", [Parameter("X")], 2, [
            [quil_cos(Parameter("X")), 0, 0, 0],
            [0, quil_cos(Parameter("X")), 0, 0],
            [0, 0, quil_cos(Parameter("X")), 0],
            [0, 0, 0, quil_cos(Parameter("X"))],
        ]),
        (
            "MixedTypes", [Parameter("X")], 1, [
                [0, quil_sin(Parameter("X"))],
                [0, 0],
            ],
        ),
        (
            "ParameterlessExpressions", [], 1, np.array([
                    [-quil_cos(np.pi), quil_sin(np.pi)],
                    [quil_sin(np.pi), quil_cos(np.pi)],
                ], dtype=object),
        ),
    ],
    ids=("No-Params", "Params", "MixedTypes", "ParameterlessExpression"),
)
class TestDefGate:
    @pytest.fixture
    def def_gate(
        self, name: str, parameters: list[Parameter] | None, n_args: int, specification: list[list[Any]] | np.ndarray | np.matrix
    ) -> GateDefinition:
        return GateDefinition(name, parameters, specification)

    def test_out(self, def_gate: GateDefinition, snapshot: SnapshotAssertion):
        assert def_gate.to_quil() == snapshot

    def test_str(self, def_gate: GateDefinition, snapshot: SnapshotAssertion):
        assert str(def_gate) == snapshot

    def test_get_constructor(self, def_gate: GateDefinition, snapshot: SnapshotAssertion):
        constructor = def_gate.get_constructor()
        if def_gate.parameters:
            g = constructor(Parameter("theta"))(Qubit(123))  # type: ignore
            assert g.to_quil() == snapshot
        else:
            g = constructor(Qubit(123))
            assert g.to_quil() == snapshot

    def test_num_args(self, def_gate: GateDefinition, n_args: int, specification: list[list[Any]] | np.ndarray | np.matrix):
        assert def_gate.num_args() == n_args

    def test_name(self, def_gate: GateDefinition, name: str):
        assert def_gate.name == name

    def test_matrix(self, def_gate: GateDefinition, specification: list[list[Any]] | np.ndarray | np.matrix):
        assert np.array_equal(def_gate.matrix, GateSpecification.Matrix(specification)._0)

    def test_parameters(self, def_gate: GateDefinition, parameters: list[Parameter] | None):
        assert def_gate.parameters == parameters

    def test_copy(self, def_gate: GateDefinition):
        c = copy.copy(def_gate)
        d = copy.deepcopy(def_gate)
        assert isinstance(c, GateDefinition)
        assert isinstance(d, GateDefinition)
        assert c == d == def_gate

    def test_pickle(self, def_gate: GateDefinition, snapshot: SnapshotAssertion):
        pickled = pickle.dumps(def_gate)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, GateDefinition)
        assert unpickled == snapshot


@pytest.mark.parametrize(
    ("name", "permutation", "expected_num_args"),
    [
        ("PermGate", np.asarray([4, 3, 2, 1]), 2),
    ],
)
class TestDefPermutationGate:
    @pytest.fixture
    def def_permutation_gate(self, name: str, permutation: np.ndarray, expected_num_args: int) -> GateDefinition:
        return GateDefinition(name, None, permutation)

    def test_out(self, def_permutation_gate: GateDefinition, snapshot: SnapshotAssertion):
        assert def_permutation_gate.to_quil() == snapshot

    def test_str(self, def_permutation_gate: GateDefinition, snapshot: SnapshotAssertion):
        assert str(def_permutation_gate) == snapshot

    def test_get_constructor(self, def_permutation_gate: GateDefinition, snapshot: SnapshotAssertion):
        constructor = def_permutation_gate.get_constructor()
        g = constructor(Qubit(123))
        assert g.to_quil() == snapshot

    def test_num_args(
        self, def_permutation_gate: GateDefinition, permutation: list[list[Any]] | np.ndarray | np.matrix,
        expected_num_args: int
    ):
        assert def_permutation_gate.num_args() == expected_num_args

    def test_name(self, def_permutation_gate: GateDefinition, name: str):
        assert def_permutation_gate.name == name

    def test_permutation(
        self, def_permutation_gate: GateDefinition, permutation: list[list[Any]] | np.ndarray | np.matrix
    ):
        assert np.array_equal(def_permutation_gate.permutation, permutation)

    def test_parameters(self, def_permutation_gate: GateDefinition):
        assert not def_permutation_gate.parameters


class TestQuditGateDefinitions:
    """Qudit dimensions for ``DefGate``: matrix sizes may be any prime power, not just ``2**k``.

    A prime-power dimension is always read with the *prime* as the qudit dimension, so 4 means two
    qubits (never one ququart) and 16 means four qubits. Non-prime qudit dimensions cannot be
    inferred from a matrix size and are rejected outright.
    """

    @staticmethod
    def _identity(dimension: int) -> np.ndarray:
        return np.eye(dimension, dtype=complex)

    @pytest.mark.parametrize(
        ("dimension", "expected_num_args"),
        [
            (2, 1),  # one qubit
            (4, 2),  # two qubits, not one ququart
            (8, 3),
            (16, 4),  # four qubits, not two ququarts
            (3, 1),  # one qutrit
            (9, 2),  # two qutrits
            (27, 3),
            (5, 1),
            (25, 2),
        ],
    )
    def test_num_args_for_prime_power_dimensions(self, dimension: int, expected_num_args: int):
        def_gate = GateDefinition(f"QUDIT{dimension}", [], self._identity(dimension))
        assert def_gate.num_args() == expected_num_args

    @pytest.mark.parametrize("dimension", [1, 6, 10, 12, 15, 18, 20, 24])
    def test_rejects_non_prime_power_dimensions(self, dimension: int):
        """6 = 2*3 is ambiguous: a qubit-qutrit pair, or a single six-level system?"""
        with pytest.raises(quil.ValueError, match="prime"):
            GateDefinition(f"BAD{dimension}", [], self._identity(dimension))

    @pytest.mark.parametrize(
        ("levels", "expected_num_args"),
        [(2, 1), (4, 2), (8, 3), (3, 1), (9, 2), (27, 3)],
    )
    def test_permutation_gate_num_args_matches_def_gate(self, levels: int, expected_num_args: int):
        """DefPermutationGate must read its length the way DefGate reads its matrix size."""
        def_gate = GateDefinition(f"PERM{levels}", [], list(reversed(range(levels))))
        assert def_gate.num_args() == expected_num_args

    @pytest.mark.parametrize("levels", [6, 10, 12])
    def test_permutation_gate_rejects_non_prime_power_length(self, levels: int):
        def_gate = GateDefinition(f"PERM{levels}", [], list(reversed(range(levels))))
        with pytest.raises(quil.ValueError, match="prime"):
            def_gate.num_args()

    def test_qutrit_gate_constructor_takes_one_argument(self):
        """A 3x3 definition builds a single-qudit gate."""
        def_gate = GateDefinition("QUTRIT_X", [], np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex))
        gate = def_gate.get_constructor()(Qubit(7))
        assert gate.to_quil() == "QUTRIT_X 7"  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("gate_name", "parameters", "arguments", "body"),
    [
        ("PauliGate", [], [Qubit("p")], PauliSum([])),
        ("PauliGate", [Parameter("theta")], [Qubit("p")], PauliSum([])),
        ("PauliGate", [], [Qubit("p")], PauliSum([PauliTerm("Y", Qubit("p"), 2.0)])),
        (
            "PauliGate",
            [Parameter("theta")],
            [Qubit("p"), Qubit("q")],
            PauliSum(
                [
                    PauliTerm("Z", Qubit("p"), 1.0 * Parameter("theta")),
                    PauliTerm("Y", Qubit("p"), 2.0),
                    PauliTerm("X", Qubit("q"), 3.0),
                    PauliTerm("I", None, 3.0),
                ]
            ),
        ),
    ],
    ids=("Default", "DefaultWithParams", "WithSum", "WithSumAndParams"),
)
class TestDefGateByPaulis:
    @pytest.fixture
    def def_gate_pauli(
        self, gate_name: str, parameters: list[Parameter], arguments: list[QubitDesignator], body: PauliSum
    ) -> GateDefinition:
        specification = GateSpecification.PauliSum(body._to_rs_pauli_sum(arguments))
        return GateDefinition(gate_name, parameters, specification)

    def test_out(self, def_gate_pauli: GateDefinition, snapshot: SnapshotAssertion):
        assert def_gate_pauli.to_quil() == snapshot

    def test_str(self, def_gate_pauli: GateDefinition, snapshot: SnapshotAssertion):
        assert str(def_gate_pauli) == snapshot

    def test_get_constructor(self, def_gate_pauli: GateDefinition, snapshot: SnapshotAssertion):
        constructor = def_gate_pauli.get_constructor()
        if def_gate_pauli.parameters:
            g = constructor(Parameter("theta"))(Qubit(123))  # type: ignore
            assert g.to_quil() == snapshot
        else:
            g = constructor(Qubit(123))
            assert g.to_quil() == snapshot

    def test_num_args(
        self,
        def_gate_pauli: GateDefinition,
        arguments: list[QubitDesignator],
    ):
        assert def_gate_pauli.num_args() == len(arguments)

    def test_name(self, def_gate_pauli: GateDefinition, gate_name: str):
        assert def_gate_pauli.name == gate_name

    def test_parameters(self, def_gate_pauli: GateDefinition, parameters: list[Parameter] | None):
        assert def_gate_pauli.parameters == parameters

    def test_arguments(self, def_gate_pauli: GateDefinition, arguments: list[QubitDesignator]):
        assert def_gate_pauli.arguments == arguments

    def test_body(self, def_gate_pauli: GateDefinition, body: PauliSum):
        if all([isinstance(term.coefficient, Number) for term in body.terms]):
            # PauliTerm equality is only defined for terms with Numbered coefficients
            assert def_gate_pauli.body == body


@pytest.mark.parametrize(
    ("name", "parameters", "qubits", "instrs"),
    [
        ("Calibrate", [], [Qubit(0)], [X(0)]),
        ("Calibrate", [Parameter("X")], [Qubit(0)], [X(0)]),
    ],
    ids=("No-Params", "Params"),
)
class TestDefCalibration:
    @pytest.fixture
    def calibration(
        self,
        name: str,
        parameters: list[ParameterDesignator],
        qubits: list[Qubit],
        instrs: list[AbstractInstruction],
    ) -> CalibrationDefinition:
        return CalibrationDefinition.from_parts(name, parameters, qubits, instrs, None)

    def test_str(self, calibration: CalibrationDefinition, snapshot: SnapshotAssertion):
        assert str(calibration) == snapshot

    def test_out(self, calibration: CalibrationDefinition, snapshot: SnapshotAssertion):
        assert calibration.to_quil() == snapshot

    def test_name(self, calibration: CalibrationDefinition, name: str):
        assert calibration.name == name
        calibration.name = "new_name"
        assert calibration.name == "new_name"

    def test_parameters(self, calibration: CalibrationDefinition, parameters: list[ParameterDesignator]):
        assert calibration.parameters == parameters
        calibration.parameters = [pi / 2]
        assert calibration.parameters == [Number(complex(pi / 2))]

    def test_qubits(self, calibration: CalibrationDefinition, qubits: list[QubitDesignator]):
        assert calibration.qubits == qubits
        calibration.qubits = [Qubit(123)]
        assert calibration.qubits == [Qubit(123)]

    def test_instrs(self, calibration: CalibrationDefinition, instrs: list[AbstractInstruction]):
        assert calibration.instrs == instrs
        calibration.instrs = [Gate("SomeGate", [], [Qubit(0)], [])]
        assert calibration.instrs == [Gate("SomeGate", [], [Qubit(0)], [])]

    def test_copy(self, calibration: CalibrationDefinition):
        assert isinstance(copy.copy(calibration), CalibrationDefinition)
        assert isinstance(copy.deepcopy(calibration), CalibrationDefinition)

    def test_pickle(self, calibration: CalibrationDefinition):
        pickled = pickle.dumps(calibration)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, CalibrationDefinition)
        assert unpickled == calibration


@pytest.mark.parametrize(
    ("qubit", "memory_reference", "instrs"),
    [(Qubit(0), MemoryReference("theta", 0, 1), [X(0)])],
)
class TestDefMeasureCalibration:
    @pytest.fixture
    def measure_calibration(
        self, qubit: Qubit, memory_reference: MemoryReference, instrs: list[AbstractInstruction]
    ) -> MeasureCalibrationDefinition:
        return MeasureCalibrationDefinition(qubit, memory_reference, instrs)

    def test_out(self, measure_calibration: MeasureCalibrationDefinition, snapshot: SnapshotAssertion):
        assert measure_calibration.to_quil() == snapshot

    def test_qubit(self, measure_calibration: MeasureCalibrationDefinition, qubit: Qubit):
        assert measure_calibration.qubit == qubit
        measure_calibration.qubit = Qubit(123)
        assert measure_calibration.qubit == Qubit(123)

    def test_memory_reference(self, measure_calibration: MeasureCalibrationDefinition, memory_reference: MemoryReference):
        assert measure_calibration.memory_reference == memory_reference
        measure_calibration.memory_reference = MemoryReference("new_mem_ref")
        assert measure_calibration.memory_reference == MemoryReference("new_mem_ref")

    def test_instrs(self, measure_calibration: MeasureCalibrationDefinition, instrs: list[AbstractInstruction]):
        assert measure_calibration.instrs == instrs
        measure_calibration.instrs = [Gate("SomeGate", [], [Qubit(0)], []), Fence([0])]
        assert isinstance(measure_calibration.instrs[0], Gate)
        assert not isinstance(measure_calibration.instrs[0], Fence)
        assert isinstance(measure_calibration.instrs[1], Fence)
        assert not isinstance(measure_calibration.instrs[1], Gate)
        assert measure_calibration.instrs == [Gate("SomeGate", [], [Qubit(0)], []), Fence([0])]

    def test_instructions(self, measure_calibration: MeasureCalibrationDefinition, instrs: list[AbstractInstruction]):
        assert measure_calibration.instructions == instrs
        measure_calibration.instrs = [Gate("SomeGate", [], [Qubit(0)], []), Fence([0])]
        assert isinstance(measure_calibration.instrs[0], Gate)
        assert not isinstance(measure_calibration.instrs[0], Fence)
        assert isinstance(measure_calibration.instrs[1], Fence)
        assert not isinstance(measure_calibration.instrs[1], Gate)
        assert measure_calibration.instrs == [Gate("SomeGate", [], [Qubit(0)], []), Fence([0])]

    def test_copy(self, measure_calibration: MeasureCalibrationDefinition):
        assert isinstance(copy.copy(measure_calibration), MeasureCalibrationDefinition)
        assert isinstance(copy.deepcopy(measure_calibration), MeasureCalibrationDefinition)

    def test_pickle(self, measure_calibration: MeasureCalibrationDefinition):
        pickled = pickle.dumps(measure_calibration)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, MeasureCalibrationDefinition)
        assert unpickled == measure_calibration


@pytest.mark.parametrize(
    ("qubit", "classical_reg"),
    [
        (Qubit(0), None),
        (Qubit(1), MemoryReference("theta", 0, 1)),
    ],
    ids=("No-MemoryReference", "MemoryReference"),
)
class TestMeasurement:
    @pytest.fixture
    def measurement(self, qubit: Qubit, classical_reg: MemoryReference | None):
        return Measurement(qubit, classical_reg)

    def test_out(self, measurement: Measurement, snapshot: SnapshotAssertion):
        assert measurement.to_quil() == snapshot

    def test_str(self, measurement: Measurement, snapshot: SnapshotAssertion):
        assert str(measurement) == snapshot

    def test_qubit(self, measurement: Measurement, qubit: Qubit):
        assert measurement.qubit == qubit

    def test_get_qubits(self, measurement: Measurement, qubit: Qubit):
        assert {measurement.qubit} == {qubit}
        match qubit:
            case FixedQubit(index):
                assert measurement.get_qubit_indices() == {index}

    def test_classical_reg(self, measurement: Measurement, classical_reg: MemoryReference):
        assert measurement.target == classical_reg

    def test_copy(self, measurement: Measurement):
        assert isinstance(copy.copy(measurement), Measurement)
        assert isinstance(copy.deepcopy(measurement), Measurement)

    def test_pickle(self, measurement: Measurement):
        pickled = pickle.dumps(measurement)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Measurement)
        assert unpickled == measurement


@pytest.mark.parametrize(
    ("frame", "direction", "initial_frequency", "hardware_object", "sample_rate", "center_frequency", "channel_delay"),
    [
        (Frame([Qubit(0)], "frame"), "direction", 0.0, None, None, None, None),
        (Frame([Qubit(1)], "frame"), "direction", 1.39, "hardware_object", 44.1, 440.0, 0.0),
    ],
    ids=("Frame-Only", "With-Optionals"),
)
class TestDefFrame:
    @pytest.fixture
    def def_frame(
        self,
        frame: Frame,
        direction: str | None,
        initial_frequency: float | None,
        hardware_object: str | None,
        sample_rate: float | None,
        center_frequency: float | None,
        channel_delay: float | None,
    ) -> FrameDefinition:
        args = dict(
            DIRECTION=direction,
            **{"INITIAL-FREQUENCY": initial_frequency},
            **{"HARDWARE-OBJECT": hardware_object},
            **{"SAMPLE-RATE": sample_rate},
            **{"CENTER-FREQUENCY": center_frequency},
            **{"CHANNEL-DELAY": channel_delay},
        )
        return FrameDefinition(frame, {k: v for k, v in args.items() if v is not None})

    def test_out(self, def_frame: FrameDefinition, snapshot: SnapshotAssertion):
        # The ordering of attributes isn't stable and can be printed in different orders across calls.
        # We assert that the first line is definitely the `DEFFRAME` line, and that the following
        # attributes are the same, regardless of their order.
        quil_lines = def_frame.to_quil().splitlines()
        assert quil_lines[0] == snapshot
        assert set(quil_lines[1:]) == snapshot

    def test_str(self, def_frame: FrameDefinition, snapshot: SnapshotAssertion):
        quil_lines = str(def_frame).splitlines()
        assert quil_lines[0] == snapshot
        assert set(quil_lines[1:]) == snapshot

    def test_frame(self, def_frame: FrameDefinition, frame: Frame):
        # Note: `FrameDefinition.frame`/`.identifier` has no setter in the native API (no mutable
        # backing storage reachable from Python); this is a known gap, see MIGRATION-NOTES.md.
        assert def_frame.frame == frame

    def test_direction(self, def_frame: FrameDefinition, direction: str | None):
        assert def_frame.direction == direction
        def_frame.direction = "tx"
        assert def_frame.direction == "tx"

    def test_initial_frequency(self, def_frame: FrameDefinition, initial_frequency: float | None):
        assert def_frame.initial_frequency == initial_frequency
        def_frame.initial_frequency = 3.14
        assert def_frame.initial_frequency == 3.14

    def test_hardware_object(self, def_frame: FrameDefinition, hardware_object: str | None):
        assert def_frame.hardware_object == hardware_object
        def_frame.hardware_object = "bfg"
        assert def_frame.hardware_object == "bfg"

    def test_hardware_object_json(self, def_frame: FrameDefinition, hardware_object: str | None):
        assert def_frame.hardware_object == hardware_object
        def_frame.hardware_object = '{"string": "str", "int": 1, "float": 3.14}'
        assert def_frame.hardware_object == '{"string": "str", "int": 1, "float": 3.14}'

    def test_sample_rate(self, def_frame: FrameDefinition, sample_rate: float | None):
        assert def_frame.sample_rate == sample_rate
        def_frame.sample_rate = 96.0
        assert def_frame.sample_rate == 96.0

    def test_center_frequency(self, def_frame: FrameDefinition, center_frequency: float | None):
        assert def_frame.center_frequency == center_frequency
        def_frame.center_frequency = 432.0
        assert def_frame.center_frequency == 432.0

    def test_channel_delay(self, def_frame: FrameDefinition, channel_delay: float | None):
        assert def_frame.channel_delay == channel_delay
        def_frame.channel_delay = 571.0
        assert def_frame.channel_delay == 571.0

    def test_copy(self, def_frame: FrameDefinition):
        assert isinstance(copy.copy(def_frame), FrameDefinition)
        assert isinstance(copy.deepcopy(def_frame), FrameDefinition)

    def test_pickle(self, def_frame: FrameDefinition):
        pickled = pickle.dumps(def_frame)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, FrameDefinition)
        assert unpickled == def_frame


@pytest.mark.parametrize(
    ("name", "memory_type", "memory_size", "shared_region", "offsets"),
    [
        ("ro", "BIT", 1, None, None),
        ("ro", "OCTET", 5, None, None),
        ("ro", "INTEGER", 5, "theta", None),
        ("ro", "BIT", 5, "theta", [(2, "OCTET")]),
    ],
    ids=("Defaults", "With-Size", "With-Shared", "With-Offsets"),
)
class TestDeclare:
    @pytest.fixture
    def declare(
        self,
        name: str,
        memory_type: str,
        memory_size: int,
        shared_region: str | None,
        offsets: Iterable[tuple[int, str]] | None,
    ) -> Declaration:
        return Declaration(name, memory_type, memory_size, shared_region, offsets)

    def test_out(self, declare: Declaration, snapshot: SnapshotAssertion):
        assert declare.to_quil() == snapshot

    def test_str(self, declare: Declaration, snapshot: SnapshotAssertion):
        assert str(declare) == snapshot

    def test_asdict(self, declare: Declaration, snapshot: SnapshotAssertion):
        assert declare.asdict() == snapshot

    def test_name(self, declare: Declaration, name: str):
        assert declare.name == name

    def test_memory_type(self, declare: Declaration, memory_type: str | None):
        assert declare.memory_type == memory_type

    def test_memory_size(self, declare: Declaration, memory_size: int | None):
        assert declare.memory_size == memory_size

    def test_shared_region(self, declare: Declaration, shared_region: str | None):
        assert declare.shared_region == shared_region

    def test_offsets(self, declare: Declaration, offsets: Iterable[tuple[int, str]] | None):
        expected_offsets = offsets or []
        assert declare.offsets() == expected_offsets

    def test_copy(self, declare: Declaration):
        assert isinstance(copy.copy(declare), Declaration)
        assert isinstance(copy.deepcopy(declare), Declaration)

    def test_pickle(self, declare: Declaration):
        pickled = pickle.dumps(declare)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Declaration)
        assert unpickled == declare


@pytest.mark.parametrize(
    ("command", "args", "freeform_string"),
    [
        ("NO-NOISE", [], None),
        ("DOES-A-THING", [0, "b"], None),
        ("INITIAL_REWIRING", [], "GREEDY"),
        ("READOUT-POVM", [1], "(0.9 0.19999999999999996 0.09999999999999998 0.8)"),
    ],
    ids=("Command-Only", "With-Arg", "With-String", "With-Arg-And-String"),
)
class TestPragma:
    # Note: unlike pre-v5 pyQuil, PRAGMA arguments no longer accept `Qubit`/`FormalArgument` objects
    # directly (only bare `str`/`int`), and `.args`/`.freeform_string` have been superseded by
    # `.arguments`/`.data` (the old names still work but are deprecated). See MIGRATION-NOTES.md.
    @pytest.fixture
    def pragma(self, command: str, args: list[int | str], freeform_string: str | None) -> Pragma:
        return Pragma(command, args, freeform_string)

    def test_out(self, pragma: Pragma, snapshot: SnapshotAssertion):
        assert pragma.to_quil() == snapshot

    def test_str(self, pragma: Pragma, snapshot: SnapshotAssertion):
        assert str(pragma) == snapshot

    def test_command(self, pragma: Pragma, command: str):
        assert pragma.command == command

    def test_args(self, pragma: Pragma, args: list[int | str]):
        assert [arg._0 for arg in pragma.arguments] == args

    def test_freeform_string(self, pragma: Pragma, freeform_string: str | None):
        assert pragma.freeform_string == freeform_string

    def test_copy(self, pragma: Pragma):
        assert isinstance(copy.copy(pragma), Pragma)
        assert isinstance(copy.deepcopy(pragma), Pragma)

    def test_pickle(self, pragma: Pragma):
        pickled = pickle.dumps(pragma)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Pragma)
        assert unpickled == pragma


@pytest.mark.parametrize(
    ("qubit"),
    [
        (Qubit(0)),
        (Qubit("a")),
        (None),
    ],
    ids=("Qubit", "FormalArgument", "None"),
)
class TestReset:
    @pytest.fixture
    def reset_qubit(self, qubit: Qubit) -> Reset:
        return Reset(qubit)

    def test_deprecated(self, qubit: Qubit):
        if qubit is not None:
            with pytest.deprecated_call():
                ResetQubit(qubit)
        else:
            with pytest.raises(TypeError):
                ResetQubit(None)

    def test_out(self, reset_qubit: Reset, snapshot: SnapshotAssertion):
        assert reset_qubit.to_quil() == snapshot

    def test_str(self, reset_qubit: Reset, snapshot: SnapshotAssertion):
        assert str(reset_qubit) == snapshot

    def test_qubit(self, reset_qubit: Reset, qubit: Qubit):
        assert reset_qubit.qubit == qubit

    def test_get_qubits(self, reset_qubit: Reset, qubit: Qubit):
        if qubit is None:
            assert reset_qubit.qubit is None
            assert reset_qubit.get_qubit_indices() is None
        else:
            assert {reset_qubit.qubit} == {qubit}
            if isinstance(qubit, FixedQubit):
                assert reset_qubit.get_qubit_indices() == {qubit.index}

    def test_copy(self, reset_qubit: Reset):
        assert isinstance(copy.copy(reset_qubit), type(reset_qubit))
        assert isinstance(copy.deepcopy(reset_qubit), type(reset_qubit))

    def test_pickle(self, reset_qubit: Reset):
        pickled = pickle.dumps(reset_qubit)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Reset)
        assert unpickled == reset_qubit


@pytest.mark.parametrize(
    ("frames", "duration"),
    [
        ([Frame([Qubit(0)], "frame")], 5.0),
    ],
)
class TestDelayFrames:
    @pytest.fixture
    def delay_frames(self, frames: list[Frame], duration: float) -> Delay:
        return Delay.from_frames(duration, frames)

    def test_out(self, delay_frames: Delay, snapshot: SnapshotAssertion):
        assert delay_frames.to_quil() == snapshot

    def test_frames(self, delay_frames: Delay, frames: list[Frame]):
        assert delay_frames.frames == frames

    def test_duration(self, delay_frames: Delay, duration: float):
        assert _unwrap_param(delay_frames.duration) == duration

    def test_copy(self, delay_frames: Delay):
        assert isinstance(copy.copy(delay_frames), Delay)
        assert isinstance(copy.deepcopy(delay_frames), Delay)

    def test_pickle(self, delay_frames: Delay):
        pickled = pickle.dumps(delay_frames)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Delay)
        assert unpickled == delay_frames


@pytest.mark.parametrize(
    ("qubits", "duration"),
    [
        ([Qubit(0)], 5.0),
        ([Qubit("a")], 2.5),
    ],
    ids=("Qubit", "FormalArgument"),
)
class TestDelayQubits:
    @pytest.fixture
    def delay_qubits(self, qubits: list[Qubit], duration: float) -> Delay:
        return Delay(duration, qubits)

    def test_out(self, delay_qubits: Delay, snapshot: SnapshotAssertion):
        assert delay_qubits.to_quil() == snapshot

    def test_qubits(self, delay_qubits: Delay, qubits: list[Qubit]):
        assert delay_qubits.qubits == qubits

    def test_duration(self, delay_qubits: Delay, duration: float):
        assert _unwrap_param(delay_qubits.duration) == duration

    def test_copy(self, delay_qubits: Delay):
        assert isinstance(copy.copy(delay_qubits), Delay)
        assert isinstance(copy.deepcopy(delay_qubits), Delay)

    def test_pickle(self, delay_qubits: Delay):
        pickled = pickle.dumps(delay_qubits)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Delay)
        assert unpickled == delay_qubits


@pytest.mark.parametrize(
    ("qubits"),
    [
        ([Qubit(0)]),
        ([Qubit("a")]),
    ],
    ids=("Qubit", "FormalArgument"),
)
class TestFence:
    @pytest.fixture
    def fence(self, qubits: list[Qubit]) -> Fence:
        return Fence(qubits)

    def test_out(self, fence: Fence, snapshot: SnapshotAssertion):
        assert fence.to_quil() == snapshot

    def test_qubits(self, fence: Fence, qubits: list[Qubit]):
        assert fence.qubits == qubits

    def test_copy(self, fence: Fence):
        assert isinstance(copy.copy(fence), Fence)
        assert isinstance(copy.deepcopy(fence), Fence)

    def test_pickle(self, fence: Fence):
        pickled = pickle.dumps(fence)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Fence)
        assert unpickled == fence


def test_fence_all():
    fa = Fence([])
    assert fa.to_quil() == "FENCE"
    assert fa.qubits == []


@pytest.mark.parametrize(
    ("name", "parameters", "entries"),
    [
        ("Wavey", [Parameter("x")], [Parameter("x")]),
        (
            "Wavey",
            [Parameter("x"), Parameter("y")],
            [1.0+2.0j, Parameter("x"), (3.0+0.0j)*Parameter("y")],
        ),
    ],
    ids=("With-Param", "With-Params-Complex"),
)
class TestDefWaveform:
    @pytest.fixture
    def def_waveform(self, name: str, parameters: list[Parameter], entries: list[complex | Expression]):
        return WaveformDefinition(name, parameters, entries)

    def test_out(self, def_waveform: WaveformDefinition, snapshot: SnapshotAssertion):
        assert def_waveform.to_quil() == snapshot

    def test_name(self, def_waveform: WaveformDefinition, name: str):
        assert def_waveform.name == name

    def test_parameters(self, def_waveform: WaveformDefinition, parameters: list[Parameter]):
        assert def_waveform.parameters == [p.name for p in parameters]

    def test_entries(self, def_waveform: WaveformDefinition, entries: list[Complex | Expression]):
        expected = [Number(complex(e)) if isinstance(e, complex) else e for e in entries]
        assert def_waveform.entries == expected

    def test_copy(self, def_waveform: WaveformDefinition):
        assert isinstance(copy.copy(def_waveform), WaveformDefinition)
        assert isinstance(copy.deepcopy(def_waveform), WaveformDefinition)

    def test_pickle(self, def_waveform: WaveformDefinition, snapshot: SnapshotAssertion):
        pickled = pickle.dumps(def_waveform)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, WaveformDefinition)
        assert unpickled == snapshot


@pytest.mark.parametrize(
    ("name", "parameters", "qubit_variables", "instructions"),
    [
        ("NiftyCircuit", [], [Qubit("a")], [Measurement(Qubit("a"), None)]),
        (
            "NiftyCircuit",
            [Parameter("theta")],
            [Qubit("a")],
            [
                Declaration("ro", "BIT", 1),
                Measurement(Qubit("a"), MemoryReference("ro")),
            ],
        ),
    ],
    ids=("No-Params", "With-Params"),
)
class TestDefCircuit:
    @pytest.fixture
    def def_circuit(
        self,
        name: str,
        parameters: list[Parameter],
        qubit_variables: list[Qubit],
        instructions: list[Instruction],
    ):
        return CircuitDefinition(
            name,
            parameters,
            qubit_variables,
            instructions,
        )

    def test_out(self, def_circuit: CircuitDefinition, snapshot: SnapshotAssertion):
        assert def_circuit.to_quil() == snapshot

    def test_name(self, def_circuit: CircuitDefinition, name: str):
        assert def_circuit.name == name
        def_circuit.name = "new_name"
        assert def_circuit.name == "new_name"

    def test_parameters(self, def_circuit: CircuitDefinition, parameters: list[Parameter]):
        assert def_circuit.parameters == [p.name for p in parameters]
        def_circuit.parameters = ["z"]
        assert def_circuit.parameters == ["z"]

    def test_qubit_variables(self, def_circuit: CircuitDefinition, qubit_variables: list[Qubit]):
        assert def_circuit.qubit_variables == [q.name for q in qubit_variables]
        def_circuit.qubit_variables = ["qubit"]
        assert def_circuit.qubit_variables == ["qubit"]

    def test_instructions(self, def_circuit: CircuitDefinition, instructions: list[AbstractInstruction]):
        assert def_circuit.instructions == instructions
        def_circuit.instructions = [Gate("new_gate", [], [Qubit(0)], [])]
        assert def_circuit.instructions == [Gate("new_gate", [], [Qubit(0)], [])]

    def test_copy(self, def_circuit: CircuitDefinition):
        assert isinstance(copy.copy(def_circuit), CircuitDefinition)
        assert isinstance(copy.deepcopy(def_circuit), CircuitDefinition)

    def test_pickle(self, def_circuit: CircuitDefinition):
        pickled = pickle.dumps(def_circuit)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, CircuitDefinition)
        assert unpickled == def_circuit


@pytest.mark.parametrize(
    ("frame", "kernel", "memory_region", "nonblocking"),
    [
        (
            Frame([Qubit(123), Qubit("q")], "FRAMEX"),
            WaveformInvocation("WAVEFORMY"),
            MemoryReference("ro"),
            False,
        ),
        (
            Frame([Qubit(123), Qubit("q")], "FRAMEX"),
            WaveformInvocation("WAVEFORMY"),
            MemoryReference("ro"),
            True,
        ),
        (
            Frame([Qubit(123), Qubit("q")], "FRAMEX"),
            FlatWaveform(duration=2.5, iq=complex(1.0, 2.0)),
            MemoryReference("ro"),
            True,
        ),
    ],
    ids=("Blocking", "NonBlocking", "TemplateWaveform"),
)
class TestCapture:
    @pytest.fixture
    def capture(self, frame: Frame, kernel: Waveform, memory_region: MemoryReference, nonblocking: bool):
        return Capture(frame, kernel, memory_region, nonblocking=nonblocking)

    def test_out(self, capture: Capture, snapshot: SnapshotAssertion):
        assert capture.to_quil() == snapshot

    def test_frame(self, capture: Capture, frame: Frame):
        assert capture.frame == frame
        capture.frame = Frame([Qubit(123)], "new-frame")
        assert capture.frame == Frame([Qubit(123)], "new-frame")

    def test_kernel(self, capture: Capture, kernel: Waveform):
        assert capture.waveform == kernel
        capture.waveform = WaveformInvocation("new-waveform")
        assert capture.waveform == WaveformInvocation("new-waveform")

    def test_memory_region(self, capture: Capture, memory_region: MemoryReference):
        assert capture.memory_reference == memory_region
        capture.memory_reference = MemoryReference("new-memory-reference")
        assert capture.memory_reference == MemoryReference("new-memory-reference")

    def test_nonblocking(self, capture: Capture, nonblocking: bool):
        assert capture.nonblocking == nonblocking
        capture.blocking = nonblocking
        assert capture.nonblocking == (not nonblocking)

    def test_copy(self, capture: Capture):
        assert isinstance(copy.copy(capture), Capture)
        assert isinstance(copy.deepcopy(capture), Capture)

    def test_pickle(self, capture: Capture, snapshot: SnapshotAssertion):
        pickled = pickle.dumps(capture)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Capture)
        assert unpickled == snapshot


@pytest.mark.parametrize(
    ("frame", "waveform", "nonblocking"),
    [
        (
            Frame([Qubit(123), Qubit("q")], "FRAMEX"),
            WaveformInvocation("WAVEFORMY"),
            False,
        ),
        (
            Frame([Qubit(123), Qubit("q")], "FRAMEX"),
            WaveformInvocation("WAVEFORMY"),
            True,
        ),
        (
            Frame([Qubit(123), Qubit("q")], "FRAMEX"),
            FlatWaveform(duration=2.5, iq=complex(1.0, 2.0)),
            True,
        ),
        (
            Frame([Qubit(123), Qubit("q")], "FRAMEX"),
            GaussianWaveform(duration=2.5, fwhm=1.0, t0=1.0, phase=0.1),
            True,
        ),
        (
            Frame([Qubit(123), Qubit("q")], "FRAMEX"),
            DragGaussianWaveform(duration=2.5, fwhm=1.0, t0=1.0, anh=0.1, alpha=1.0),
            True,
        ),
        (
            Frame([Qubit(123), Qubit("q")], "FRAMEX"),
            HrmGaussianWaveform(duration=2.5, fwhm=1.0, t0=1.0, anh=0.1, alpha=1.0, second_order_hrm_coeff=0.5),
            True,
        ),
        (
            Frame([Qubit(123), Qubit("q")], "FRAMEX"),
            ErfSquareWaveform(duration=2.5, risetime=1.0, pad_left=1.0, pad_right=0.1, scale=1.0),
            True,
        ),
        (
            Frame([Qubit(123), Qubit("q")], "FRAMEX"),
            BoxcarAveragerKernel(duration=2.5, scale=1.0),
            True,
        ),
    ],
    ids=(
        "Blocking",
        "NonBlocking",
        "FlatWaveform",
        "GaussianWaveform",
        "DragGaussianWaveform",
        "HrmGaussianWaveform",
        "ErfSquareWaveform",
        "BoxcarAveragerKernel",
    ),
)
class TestPulse:
    @pytest.fixture
    def pulse(self, frame: Frame, waveform: Waveform, nonblocking: bool):
        return Pulse(frame, waveform, nonblocking=nonblocking)

    def test_out(self, pulse: Pulse, snapshot: SnapshotAssertion):
        assert pulse.to_quil() == snapshot

    def test_frame(self, pulse: Pulse, frame: Frame):
        assert pulse.frame == frame
        pulse.frame = Frame([Qubit(123)], "new-frame")
        assert pulse.frame == Frame([Qubit(123)], "new-frame")

    def test_waveform(self, pulse: Pulse, waveform: Waveform):
        assert pulse.waveform == waveform
        if isinstance(waveform, TemplateWaveform):
            assert isinstance(pulse.waveform, type(waveform))
            pulse.waveform.samples(0.5)
        pulse.waveform = WaveformInvocation("new-waveform")
        assert pulse.waveform == WaveformInvocation("new-waveform")

    def test_nonblocking(self, pulse: Pulse, nonblocking: bool):
        assert pulse.nonblocking == nonblocking
        pulse.blocking = nonblocking
        assert pulse.nonblocking == (not nonblocking)

    def test_copy(self, pulse: Pulse):
        assert isinstance(copy.copy(pulse), Pulse)
        assert isinstance(copy.deepcopy(pulse), Pulse)

    def test_pickle(self, pulse: Pulse, snapshot: SnapshotAssertion):
        pickled = pickle.dumps(pulse)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Pulse)
        assert unpickled == snapshot


@pytest.mark.parametrize(
    ("frame", "duration", "memory_region", "nonblocking"),
    [
        (
            Frame([Qubit(123), Qubit("q")], "FRAMEX"),
            0.5,
            MemoryReference("ro"),
            False,
        ),
        (
            Frame([Qubit(123), Qubit("q")], "FRAMEX"),
            2.5,
            MemoryReference("ro"),
            True,
        ),
        (
            Frame([Qubit(123), Qubit("q")], "FRAMEX"),
            2.5,
            MemoryReference("ro"),
            True,
        ),
    ],
    ids=("Blocking", "NonBlocking", "FlatWaveform"),
)
class TestRawCapture:
    @pytest.fixture
    def raw_capture(self, frame: Frame, duration: float, memory_region: MemoryReference, nonblocking: bool):
        return RawCapture(frame, duration, memory_region, nonblocking=nonblocking)

    def test_out(self, raw_capture: RawCapture, snapshot: SnapshotAssertion):
        assert raw_capture.to_quil() == snapshot

    def test_frame(self, raw_capture: RawCapture, frame: Frame):
        assert raw_capture.frame == frame

    def test_duration(self, raw_capture: RawCapture, duration: float):
        assert raw_capture.duration.evaluate({}, {}) == duration

    def test_memory_region(self, raw_capture: RawCapture, memory_region: MemoryReference):
        assert raw_capture.memory_reference == memory_region

    def test_nonblocking(self, raw_capture: RawCapture, nonblocking: bool):
        assert raw_capture.nonblocking == nonblocking

    def test_copy(self, raw_capture: RawCapture):
        assert isinstance(copy.copy(raw_capture), RawCapture)
        assert isinstance(copy.deepcopy(raw_capture), RawCapture)

    def test_pickle(self, raw_capture: RawCapture):
        pickled = pickle.dumps(raw_capture)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, RawCapture)
        assert unpickled == raw_capture


@pytest.mark.parametrize(
    ("frame", "expression"),
    [
        (Frame([Qubit(1)], "FRAMEX"), 5.0),
        (Frame([Qubit(2)], "FRAMEX"), MemoryReference("ro")),
    ],
)
class TestFrameMutations:
    @pytest.fixture
    def set_frequency(self, frame: Frame, expression: ParameterDesignator):
        return SetFrequency(frame, expression)

    @pytest.fixture
    def set_phase(self, frame: Frame, expression: ParameterDesignator):
        return SetPhase(frame, expression)

    @pytest.fixture
    def shift_frequency(self, frame: Frame, expression: ParameterDesignator):
        return ShiftFrequency(frame, expression)

    @pytest.fixture
    def shift_phase(self, frame: Frame, expression: ParameterDesignator):
        return ShiftPhase(frame, expression)

    @pytest.fixture
    def set_scale(self, frame: Frame, expression: ParameterDesignator):
        return SetScale(frame, expression)

    @pytest.fixture
    def frame_mutation_instructions(
        self, set_frequency, set_phase, shift_frequency, shift_phase, set_scale
    ) -> tuple[SetFrequency, SetPhase, ShiftFrequency, ShiftPhase, SetScale]:
        return (set_frequency, set_phase, shift_frequency, shift_phase, set_scale)

    def test_out(self, frame_mutation_instructions, snapshot: SnapshotAssertion):
        for instr in frame_mutation_instructions:
            assert instr.to_quil() == snapshot

    def test_frame(self, frame_mutation_instructions, frame: Frame):
        for instr in frame_mutation_instructions:
            assert instr.frame == frame

    def test_get_qubits(self, frame_mutation_instructions, frame: Frame):
        for instr in frame_mutation_instructions:
            assert instr.get_qubit_indices() == set([q.index for q in frame.qubits
                                                     if isinstance(q, FixedQubit)])
            assert set(instr.frame.qubits) == set(frame.qubits)

    def test_expression(self, frame_mutation_instructions, expression: ParameterDesignator):
        expression_names = ["freq", "phase", "freq", "phase", "scale"]
        for instr, expression_name in zip(frame_mutation_instructions, expression_names, strict=False):
            assert _unwrap_param(getattr(instr, expression_name)) == expression

    def test_copy(self, frame_mutation_instructions):
        for instr in frame_mutation_instructions:
            assert isinstance(copy.copy(instr), type(instr))
            assert isinstance(copy.deepcopy(instr), type(instr))


@pytest.mark.parametrize(
    ("frame_a", "frame_b"),
    [(Frame([Qubit(1)], "FRAMEX"), Frame([Qubit(2)], "FRAMEX"))],
)
class TestSwapPhases:
    @pytest.fixture
    def swap_phases(self, frame_a, frame_b):
        return SwapPhases(frame_a, frame_b)

    def test_out(self, swap_phases: SwapPhases, snapshot: SnapshotAssertion):
        assert swap_phases.to_quil() == snapshot

    def test_frames(self, swap_phases: SwapPhases, frame_a: Frame, frame_b: Frame):
        assert swap_phases.frameA == frame_a
        assert swap_phases.frameB == frame_b

    def test_get_qubits(self, swap_phases: SwapPhases, frame_a: Frame, frame_b: Frame):
        expected_qubits = set(frame_a.qubits + frame_b.qubits)
        assert swap_phases.get_qubit_indices() == {q.index for q in expected_qubits if isinstance(q, FixedQubit)}
        assert set(swap_phases.frameA.qubits + swap_phases.frameB.qubits) == expected_qubits

    def test_copy(self, swap_phases: SwapPhases):
        assert isinstance(copy.copy(swap_phases), SwapPhases)
        assert isinstance(copy.deepcopy(swap_phases), SwapPhases)

    def test_pickle(self, swap_phases: SwapPhases):
        pickled = pickle.dumps(swap_phases)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, SwapPhases)
        assert unpickled == swap_phases


@pytest.mark.parametrize(
    ("destination", "source"),
    [
        (MemoryReference("ro"), MemoryReference("bar")),
        (MemoryReference("foo", 5, 10), 5),
        (MemoryReference("bar"), 3.2),
    ],
)
class TestMove:
    @pytest.fixture
    def move(self, destination: MemoryReference, source: MemoryReference | int | float) -> Move:
        return Move(destination, source)

    def test_out(self, move: Move, snapshot: SnapshotAssertion):
        assert move.to_quil() == snapshot

    def test_destination(self, move: Move, destination: MemoryReference):
        assert move.destination == destination

    def test_source(self, move: Move, source: MemoryReference | int | float):
        assert _unwrap_operand(move.source) == source

    def test_copy(self, move: Move):
        assert isinstance(copy.copy(move), Move)
        assert isinstance(copy.deepcopy(move), Move)

    def test_pickle(self, move: Move):
        pickled = pickle.dumps(move)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Move)
        assert unpickled == move


@pytest.mark.parametrize(
    ("left", "right"),
    [(MemoryReference("ro"), MemoryReference("bar")), (MemoryReference("foo", 5, 10), MemoryReference("bar"))],
)
class TestExchange:
    @pytest.fixture
    def exchange(self, left: MemoryReference, right: MemoryReference) -> Exchange:
        return Exchange(left, right)

    def test_out(self, exchange: Exchange, snapshot: SnapshotAssertion):
        assert exchange.to_quil() == snapshot

    def test_left(self, exchange: Exchange, left: MemoryReference):
        assert exchange.left == left

    def test_right(self, exchange: Exchange, right: MemoryReference):
        assert exchange.right == right

    def test_copy(self, exchange: Exchange):
        assert isinstance(copy.copy(exchange), Exchange)
        assert isinstance(copy.deepcopy(exchange), Exchange)

    def test_pickle(self, exchange: Exchange):
        pickled = pickle.dumps(exchange)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Exchange)
        assert unpickled == exchange


@pytest.mark.parametrize(
    ("destination", "source"),
    [(MemoryReference("ro"), MemoryReference("bar")), (MemoryReference("foo", 5, 10), MemoryReference("bar"))],
)
class TestConvert:
    @pytest.fixture
    def convert(self, destination: MemoryReference, source: MemoryReference) -> Convert:
        return Convert(destination, source)

    def test_out(self, convert: Convert, snapshot: SnapshotAssertion):
        assert convert.to_quil() == snapshot

    def test_destination(self, convert: Convert, destination: MemoryReference):
        assert convert.destination == destination

    def test_source(self, convert: Convert, source: MemoryReference):
        assert convert.source == source

    def test_copy(self, convert: Convert):
        assert isinstance(copy.copy(convert), Convert)
        assert isinstance(copy.deepcopy(convert), Convert)

    def test_pickle(self, convert: Convert):
        pickled = pickle.dumps(convert)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Convert)
        assert unpickled == convert


@pytest.mark.parametrize(
    ("destination", "source", "offset"),
    [(MemoryReference("t"), "y", MemoryReference("z")), (MemoryReference("t", 5, 10), "y", MemoryReference("bar", 1))],
)
class TestLoad:
    @pytest.fixture
    def load(self, destination: MemoryReference, source: str, offset: MemoryReference) -> Load:
        return Load(destination, source, offset)

    def test_out(self, load: Load, snapshot: SnapshotAssertion):
        assert load.to_quil() == snapshot

    def test_destination(self, load: Load, destination: MemoryReference):
        assert load.destination == destination

    def test_source(self, load: Load, source: MemoryReference):
        assert load.source == source

    def test_offset(self, load: Load, offset: MemoryReference):
        assert load.offset == offset

    def test_copy(self, load: Load):
        assert isinstance(copy.copy(load), Load)
        assert isinstance(copy.deepcopy(load), Load)

    def test_pickle(self, load: Load):
        pickled = pickle.dumps(load)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Load)
        assert unpickled == load


@pytest.mark.parametrize(
    ("destination", "offset", "source"),
    [
        ("t", MemoryReference("y"), MemoryReference("z")),
        ("t", MemoryReference("y", 5, 10), 2),
        ("t", MemoryReference("y", 5, 10), 3.14),
    ],
)
class TestStore:
    @pytest.fixture
    def store(self, destination: str, offset: MemoryReference, source: MemoryReference) -> Store:
        return Store(destination, offset, source)

    def test_out(self, store: Store, snapshot: SnapshotAssertion):
        assert store.to_quil() == snapshot

    def test_destination(self, store: Store, destination: str):
        assert store.destination == destination

    def test_offset(self, store: Store, offset: MemoryReference):
        assert store.offset == offset

    def test_source(self, store: Store, source: MemoryReference | int | float):
        assert _unwrap_operand(store.source) == source

    def test_copy(self, store: Store):
        assert isinstance(copy.copy(store), Store)
        assert isinstance(copy.deepcopy(store), Store)

    def test_pickle(self, store: Store):
        pickled = pickle.dumps(store)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Store)
        assert unpickled == store


@pytest.mark.parametrize(
    ("op", "target", "left", "right"),
    [
        (ComparisonOperator.EQUAL, MemoryReference("t"), MemoryReference("y"), MemoryReference("z")),
        (ComparisonOperator.LESS_THAN, MemoryReference("t"), MemoryReference("y", 5, 10), 2),
        (ComparisonOperator.LESS_THAN_OR_EQUAL, MemoryReference("t", 2, 4), MemoryReference("y"), 3.14),
        (ComparisonOperator.GREATER_THAN, MemoryReference("t", 2, 4), MemoryReference("y"), 3.14),
        (ComparisonOperator.GREATER_THAN_OR_EQUAL, MemoryReference("t", 2, 4), MemoryReference("y"), 3.14),
    ],
)
class TestComparison:
    @pytest.fixture
    def comparison(
        self, op: ComparisonOperator, target: MemoryReference, left: MemoryReference, right: MemoryReference | int | float
    ) -> Comparison:
        return Comparison(op, target, left, right)

    def test_out(self, comparison: Comparison, snapshot: SnapshotAssertion):
        assert comparison.to_quil() == snapshot

    def test_target(self, comparison: Comparison, target: MemoryReference):
        assert comparison.target == target

    def test_left(self, comparison: Comparison, left: MemoryReference):
        assert comparison.left == left

    def test_right(self, comparison: Comparison, right: MemoryReference | int | float):
        assert _unwrap_operand(comparison.right) == right

    def test_copy(self, comparison: Comparison):
        assert isinstance(copy.copy(comparison), type(comparison))
        assert isinstance(copy.deepcopy(comparison), type(comparison))

    def test_pickle(self, comparison: Comparison):
        pickled = pickle.dumps(comparison)
        unpickled = pickle.loads(pickled)
        assert isinstance(unpickled, Comparison)
        assert isinstance(unpickled, type(comparison))
        assert unpickled == comparison


@pytest.mark.parametrize(
    ("op", "target"),
    [
        ("NEG", MemoryReference("a")),
        ("NOT", MemoryReference("b", 1)),
        ("NEG", MemoryReference("c", 2, 4)),
    ],
)
class TestUnaryLogic:
    @pytest.fixture
    def unary(self, op: str, target: MemoryReference) -> UnaryLogic:
        if op == "NEG":
            return UnaryLogic(UnaryOperator.NEG, target)
        if op == "NOT":
            return UnaryLogic(UnaryOperator.NOT, target)
        raise ValueError(f"Unknown unary classical operator: {op}")

    def test_out(self, unary: UnaryLogic, snapshot: SnapshotAssertion):
        assert unary.to_quil() == snapshot

    def test_target(self, unary: UnaryLogic, target: MemoryReference):
        assert unary.target == target

    def test_copy(self, unary: UnaryLogic):
        assert isinstance(copy.copy(unary), type(unary))
        assert isinstance(copy.deepcopy(unary), type(unary))

    def test_pickle(self, unary: UnaryLogic):
        pickled = pickle.dumps(unary)
        unpickled = pickle.loads(pickled)
        assert unpickled == unary


@pytest.mark.parametrize(
    ("op", "destination", "source"),
    [
        ("ADD", MemoryReference("a"), MemoryReference("b")),
        ("SUB", MemoryReference("b", 1), 1),
        ("MUL", MemoryReference("c", 2, 4), 1.0),
        ("DIV", MemoryReference("c", 2, 4), 4.2),
    ],
)
class TestArithmetic:
    @pytest.fixture
    def arithmetic(self, op: str, destination: MemoryReference, source: MemoryReference | int | float) -> Arithmetic:
        if op == "ADD":
            return Arithmetic(ArithmeticOperator.ADD, destination, source)
        if op == "SUB":
            return Arithmetic(ArithmeticOperator.SUBTRACT, destination, source)
        if op == "MUL":
            return Arithmetic(ArithmeticOperator.MULTIPLY, destination, source)
        if op == "DIV":
            return Arithmetic(ArithmeticOperator.DIVIDE, destination, source)
        raise ValueError(f"Unknown arithmetic operator: {op}")

    def test_out(self, arithmetic: Arithmetic, snapshot: SnapshotAssertion):
        assert arithmetic.to_quil() == snapshot

    def test_destination(self, arithmetic: Arithmetic, destination: MemoryReference):
        assert arithmetic.destination == destination

    def test_source(self, arithmetic: Arithmetic, source: MemoryReference | float | int):
        assert _unwrap_operand(arithmetic.source) == source

    def test_copy(self, arithmetic: Arithmetic):
        assert isinstance(copy.copy(arithmetic), type(arithmetic))
        assert isinstance(copy.deepcopy(arithmetic), type(arithmetic))

    def valid_in_program(self, arithmetic):
        try:
            p = Program(arithmetic)
            p[0] = arithmetic
        except Exception:
            pytest.fail("Arithmetic not valid in Program")

    def test_pickle(self, arithmetic: UnaryLogic, snapshot: SnapshotAssertion):
        pickled = pickle.dumps(arithmetic)
        unpickled = pickle.loads(pickled)
        assert unpickled == snapshot


@pytest.mark.parametrize(
    ("op", "destination", "source"),
    [
        ("AND", MemoryReference("a"), MemoryReference("b")),
        ("IOR", MemoryReference("b", 1), 1),
        ("XOR", MemoryReference("c", 2, 4), 2),
    ],
)
class TestBinaryLogic:
    @pytest.fixture
    def logical(self, op: str, destination: MemoryReference, source: MemoryReference | int) -> BinaryLogic:
        if op == "AND":
            return BinaryLogic(BinaryOperator.AND, destination, source)
        if op == "IOR":
            return BinaryLogic(BinaryOperator.IOR, destination, source)
        if op == "XOR":
            return BinaryLogic(BinaryOperator.XOR, destination, source)
        raise ValueError(f"Unknown logical operator: {op}")

    def test_out(self, logical: BinaryLogic, snapshot: SnapshotAssertion):
        assert logical.to_quil() == snapshot

    def test_destination(self, logical: BinaryLogic, destination: MemoryReference):
        assert logical.destination == destination

    def test_source(self, logical: BinaryLogic, source: MemoryReference | float | int):
        assert _unwrap_operand(logical.source) == source

    def test_copy(self, logical: BinaryLogic):
        assert isinstance(copy.copy(logical), type(logical))
        assert isinstance(copy.deepcopy(logical), type(logical))

    def test_pickle(self, logical: BinaryLogic):
        pickled = pickle.dumps(logical)
        unpickled = pickle.loads(pickled)
        assert unpickled == logical


def test_include():
    include = Include("my-cool-file.quil")
    assert include.to_quil() == 'INCLUDE "my-cool-file.quil"'
    assert include.filename == "my-cool-file.quil"


def test_wait():
    assert Wait.to_quil() == str(Wait) == "WAIT"


def test_halt():
    assert Halt.to_quil() == str(Halt) == "HALT"


def test_nop():
    assert Nop.to_quil() == str(Nop) == "NOP"


def test_pragma_extern():
    """
    Test that pragma extern instructions are copied correctly with
    `Program.copy_everything_except_instructions`.
    """
    program = Program()
    program += Pragma("EXTERN", ["example"])
    program_copy = program.copy_everything_except_instructions()
    assert program_copy == program


class TestCall:
    """Test basic operations of the Call instruction."""

    @pytest.fixture
    def call(self) -> Call:
        return Call("example", [CallArgument.Immediate(complex(1.234))])

    def test_out(self, call: Call):
        assert call.to_quil() == "CALL example 1.234"

    def test_str(self, call: Call):
        assert str(call) == "CALL example 1.234"

    def test_deepcopy(self, call: Call):
        call_deep_copy = copy.deepcopy(call)
        assert call_deep_copy == call
        assert call_deep_copy.arguments[0] is not call.arguments[0]
