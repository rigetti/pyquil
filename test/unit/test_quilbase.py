from math import pi
from typing import Any, List, Optional, Iterable, Tuple, Union
from numbers import Complex, Number

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from pyquil.gates import X
from pyquil.quil import Program
from pyquil.quilbase import (
    AbstractInstruction,
    Declare,
    DefCalibration,
    DefCircuit,
    DefFrame,
    DefGate,
    DefMeasureCalibration,
    DefWaveform,
    DefPermutationGate,
    DefGateByPaulis,
    DefMeasureCalibration,
    DelayFrames,
    DelayQubits,
    Fence,
    FenceAll,
    FormalArgument,
    Gate,
    Measurement,
    MemoryReference,
    ParameterDesignator,
    Parameter,
    Pragma,
    QubitDesignator,
    Reset,
    ResetQubit,
)
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quilatom import BinaryExp, Mul, Frame, Qubit, Expression
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.api._compiler import QPUCompiler


@pytest.mark.parametrize(
    ("name", "params", "qubits"),
    [
        ("X", [], [Qubit(0)]),
        ("CPHASE", [pi / 2], [Qubit(0), Qubit(1)]),
        ("RZ", [MemoryReference("theta", 0, 1)], [Qubit(0)]),
        ("RZ", [MemoryReference("alpha", 0) - MemoryReference("beta")], [Qubit(0)]),
    ],
    ids=("X-Gate", "CPHASE-Expression", "RZ-MemoryReference", "RZ-MemoryReference-Expression"),
)
class TestGate:
    @pytest.fixture
    def gate(self, name: str, params: List[ParameterDesignator], qubits: List[Qubit]) -> Gate:
        return Gate(name, params, qubits)

    @pytest.fixture
    def program(self, params: List[ParameterDesignator], gate: Gate) -> Program:
        """Creates a valid quil program using the gate and declaring memory regions for any of it's parameters"""
        program = Program(gate)
        for param in params:
            if isinstance(param, MemoryReference):
                program.declare(param.name, "REAL", param.declared_size or 1)
            if isinstance(param, BinaryExp):
                if isinstance(param.op1, MemoryReference):
                    program.declare(param.op1.name, "REAL", param.op1.declared_size or 1)
                if isinstance(param.op2, MemoryReference):
                    program.declare(param.op2.name, "REAL", param.op2.declared_size or 1)

        return program

    def test_str(self, gate: Gate, snapshot: SnapshotAssertion):
        assert str(gate) == snapshot

    def test_name(self, gate: Gate, name: str):
        assert gate.name == name

    def test_params(self, gate: Gate, params: List[ParameterDesignator]):
        assert gate.params == params
        gate.params = [pi / 2]
        assert gate.params == [pi / 2]

    def test_qubits(self, gate: Gate, qubits: List[Qubit]):
        assert gate.qubits == qubits
        gate.qubits = [Qubit(123)]
        assert gate.qubits == [Qubit(123)]

    def test_get_qubits(self, gate: Gate, qubits: List[Qubit]):
        assert gate.get_qubit_indices() == {q.index for q in qubits}
        assert gate.get_qubits(indices=False) == set(qubits)

    def test_controlled_modifier(self, gate: Gate, snapshot: SnapshotAssertion):
        assert str(gate.controlled([Qubit(5)])) == snapshot

    def test_dagger_modifier(self, gate: Gate, snapshot: SnapshotAssertion):
        assert str(gate.dagger()) == snapshot

    def test_forked_modifier(self, gate: Gate, params: List[ParameterDesignator], snapshot: SnapshotAssertion):
        alt_params: List[ParameterDesignator] = [n for n in range(len(params))]
        assert str(gate.forked(Qubit(5), alt_params)) == snapshot

    def test_repr(self, gate: Gate, snapshot: SnapshotAssertion):
        assert repr(gate) == snapshot

    def test_compile(self, program: Program, compiler: QPUCompiler):
        try:
            compiler.quil_to_native_quil(program)
        except Exception as e:
            assert False, f"Failed to compile the program: {e}\n{program}"


@pytest.mark.parametrize(
    ("name", "matrix", "parameters"),
    [
        ("NoParamGate", np.eye(4), []),
        ("ParameterizedGate", np.diag([Parameter("X")] * 4), [Parameter("X")]),
    ],
    ids=("No-Params", "Params"),
)
class TestDefGate:
    @pytest.fixture
    def def_gate(
        self, name: str, matrix: Union[List[List[Any]], np.ndarray, np.matrix], parameters: Optional[List[Parameter]]
    ) -> DefGate:
        return DefGate(name, matrix, parameters)

    def test_out(self, def_gate: DefGate, snapshot: SnapshotAssertion):
        assert def_gate.out() == snapshot

    def test_str(self, def_gate: DefGate, snapshot: SnapshotAssertion):
        assert str(def_gate) == snapshot

    def test_get_constructor(self, def_gate: DefGate, snapshot: SnapshotAssertion):
        constructor = def_gate.get_constructor()
        if def_gate.parameters:
            g = constructor(Parameter("theta"))(Qubit(123))
            assert g.out() == snapshot
        else:
            g = constructor(Qubit(123))
            assert g.out() == snapshot

    def test_num_args(self, def_gate: DefGate, matrix: Union[List[List[Any]], np.ndarray, np.matrix]):
        assert def_gate.num_args() == np.log2(len(matrix))

    def test_name(self, def_gate: DefGate, name: str):
        assert def_gate.name == name
        def_gate.name = "new_name"
        assert def_gate.name == "new_name"

    def test_matrix(self, def_gate: DefGate, matrix: Union[List[List[Any]], np.ndarray, np.matrix]):
        assert np.array_equal(def_gate.matrix, matrix)
        new_matrix = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]])
        def_gate.matrix = new_matrix
        assert np.array_equal(def_gate.matrix, new_matrix)

    def test_parameters(self, def_gate: DefGate, parameters: Optional[List[Parameter]]):
        assert def_gate.parameters == parameters
        def_gate.parameters = [Parameter("brand_new_param")]
        assert def_gate.parameters == [Parameter("brand_new_param")]


@pytest.mark.parametrize(
    ("name", "permutation"),
    [
        ("PermGate", np.asarray([4, 3, 2, 1])),
    ],
)
class TestDefPermutationGate:
    @pytest.fixture
    def def_permutation_gate(self, name: str, permutation: np.ndarray) -> DefPermutationGate:
        return DefPermutationGate(name, permutation)

    def test_out(self, def_permutation_gate: DefPermutationGate, snapshot: SnapshotAssertion):
        assert def_permutation_gate.out() == snapshot

    def test_str(self, def_permutation_gate: DefPermutationGate, snapshot: SnapshotAssertion):
        assert str(def_permutation_gate) == snapshot

    def test_get_constructor(self, def_permutation_gate: DefPermutationGate, snapshot: SnapshotAssertion):
        constructor = def_permutation_gate.get_constructor()
        g = constructor(Qubit(123))
        assert g.out() == snapshot

    def test_num_args(
        self, def_permutation_gate: DefPermutationGate, permutation: Union[List[List[Any]], np.ndarray, np.matrix]
    ):
        assert def_permutation_gate.num_args() == np.log2(len(permutation))

    def test_name(self, def_permutation_gate: DefPermutationGate, name: str):
        assert def_permutation_gate.name == name
        def_permutation_gate.name = "new_name"
        assert def_permutation_gate.name == "new_name"

    def test_permutation(
        self, def_permutation_gate: DefPermutationGate, permutation: Union[List[List[Any]], np.ndarray, np.matrix]
    ):
        assert np.array_equal(def_permutation_gate.permutation, permutation)
        new_permutation = [1, 2, 3, 4]
        def_permutation_gate.permutation = new_permutation
        assert np.array_equal(def_permutation_gate.permutation, new_permutation)

    def test_parameters(self, def_permutation_gate: DefPermutationGate):
        assert not def_permutation_gate.parameters


@pytest.mark.parametrize(
    ("gate_name", "parameters", "arguments", "body"),
    [
        ("PauliGate", [], [FormalArgument("p")], PauliSum([])),
        ("PauliGate", [Parameter("theta")], [FormalArgument("p")], PauliSum([])),
        ("PauliGate", [], [FormalArgument("p")], PauliSum([PauliTerm("Y", FormalArgument("p"), 2.0)])),
        (
            "PauliGate",
            [Parameter("theta")],
            [FormalArgument("p"), FormalArgument("q")],
            PauliSum(
                [
                    PauliTerm("Z", FormalArgument("p"), Mul(1.0, Parameter("theta"))),
                    PauliTerm("Y", FormalArgument("p"), 2.0),
                    PauliTerm("X", FormalArgument("q"), 3.0),
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
        self, gate_name: str, parameters: List[Parameter], arguments: List[QubitDesignator], body: PauliSum
    ) -> DefGateByPaulis:
        return DefGateByPaulis(gate_name, parameters, arguments, body)

    def test_out(self, def_gate_pauli: DefGateByPaulis, snapshot: SnapshotAssertion):
        assert def_gate_pauli.out() == snapshot

    def test_str(self, def_gate_pauli: DefGateByPaulis, snapshot: SnapshotAssertion):
        assert str(def_gate_pauli) == snapshot

    def test_get_constructor(self, def_gate_pauli: DefGateByPaulis, snapshot: SnapshotAssertion):
        constructor = def_gate_pauli.get_constructor()
        if def_gate_pauli.parameters:
            g = constructor(Parameter("theta"))(Qubit(123))
            assert g.out() == snapshot
        else:
            g = constructor(Qubit(123))
            assert g.out() == snapshot

    def test_num_args(
        self,
        def_gate_pauli: DefGateByPaulis,
        arguments: List[QubitDesignator],
    ):
        assert def_gate_pauli.num_args() == len(arguments)

    def test_name(self, def_gate_pauli: DefGateByPaulis, gate_name: str):
        assert def_gate_pauli.name == gate_name
        def_gate_pauli.name = "new_name"
        assert def_gate_pauli.name == "new_name"

    def test_parameters(self, def_gate_pauli: DefGate, parameters: Optional[List[Parameter]]):
        assert def_gate_pauli.parameters == parameters
        def_gate_pauli.parameters = [Parameter("brand_new_param")]
        assert def_gate_pauli.parameters == [Parameter("brand_new_param")]

    def test_arguments(self, def_gate_pauli: DefGateByPaulis, arguments: List[QubitDesignator]):
        assert def_gate_pauli.arguments == arguments
        def_gate_pauli.arguments = [FormalArgument("NewArgument")]  # type: ignore
        assert def_gate_pauli.arguments == [FormalArgument("NewArgument")]

    def test_body(self, def_gate_pauli: DefGateByPaulis, body: PauliSum):
        if all([isinstance(term.coefficient, Number) for term in body.terms]):
            # PauliTerm equality is only defined for terms with Numbered coefficients
            assert def_gate_pauli.body == body
        new_body = PauliSum([PauliTerm("X", FormalArgument("a"), 5.0)])
        def_gate_pauli.body = new_body
        assert def_gate_pauli.body == new_body


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
        self, name: str, parameters: List[ParameterDesignator], qubits: List[Qubit], instrs: List[AbstractInstruction]
    ) -> DefCalibration:
        return DefCalibration(name, parameters, qubits, instrs)

    def test_str(self, calibration: DefCalibration, snapshot: SnapshotAssertion):
        assert str(calibration) == snapshot

    def test_out(self, calibration: DefCalibration, snapshot: SnapshotAssertion):
        assert calibration.out() == snapshot

    def test_name(self, calibration: DefCalibration, name: str):
        assert calibration.name == name
        calibration.name = "new_name"
        assert calibration.name == "new_name"

    def test_parameters(self, calibration: DefCalibration, parameters: List[ParameterDesignator]):
        assert calibration.parameters == parameters
        calibration.parameters = [pi / 2]
        assert calibration.parameters == [pi / 2]

    def test_qubits(self, calibration: DefCalibration, qubits: List[Qubit]):
        assert calibration.qubits == qubits
        calibration.qubits = [Qubit(123)]
        assert calibration.qubits == [Qubit(123)]

    def test_instrs(self, calibration: DefCalibration, instrs: List[AbstractInstruction]):
        assert calibration.instrs == instrs
        calibration.instrs = [Gate("SomeGate", [], [Qubit(0)], [])]
        assert calibration.instrs == [Gate("SomeGate", [], [Qubit(0)], [])]


@pytest.mark.parametrize(
    ("qubit", "memory_reference", "instrs"),
    [(Qubit(0), MemoryReference("theta", 0, 1), [X(0)])],
)
class TestDefMeasureCalibration:
    @pytest.fixture
    def measure_calibration(
        self, qubit: Qubit, memory_reference: MemoryReference, instrs: List[AbstractInstruction]
    ) -> DefMeasureCalibration:
        return DefMeasureCalibration(qubit, memory_reference, instrs)

    def test_out(self, measure_calibration: DefMeasureCalibration, snapshot: SnapshotAssertion):
        assert measure_calibration.out() == snapshot

    def test_qubit(self, measure_calibration: DefMeasureCalibration, qubit: Qubit):
        assert measure_calibration.qubit == qubit
        measure_calibration.qubit = Qubit(123)
        assert measure_calibration.qubit == Qubit(123)

    def test_memory_reference(self, measure_calibration: DefMeasureCalibration, memory_reference: MemoryReference):
        assert measure_calibration.memory_reference == memory_reference
        measure_calibration.memory_reference = MemoryReference("new_mem_ref")
        assert measure_calibration.memory_reference == MemoryReference("new_mem_ref")

    def test_instrs(self, measure_calibration: DefMeasureCalibration, instrs: List[AbstractInstruction]):
        assert measure_calibration.instrs == instrs
        measure_calibration.instrs = [Gate("SomeGate", [], [Qubit(0)], [])]
        assert measure_calibration.instrs == [Gate("SomeGate", [], [Qubit(0)], [])]


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
    def measurement(self, qubit: Qubit, classical_reg: Optional[MemoryReference]):
        return Measurement(qubit, classical_reg)

    def test_out(self, measurement: Measurement, snapshot: SnapshotAssertion):
        assert measurement.out() == snapshot

    def test_str(self, measurement: Measurement, snapshot: SnapshotAssertion):
        assert str(measurement) == snapshot

    def test_qubit(self, measurement: Measurement, qubit: Qubit):
        assert measurement.qubit == qubit
        measurement.qubit = Qubit(123)
        assert measurement.qubit == Qubit(123)

    def test_get_qubits(self, measurement: Measurement, qubit: Qubit):
        assert measurement.get_qubits(False) == set([qubit])
        if isinstance(qubit, Qubit):
            assert measurement.get_qubits(True) == set([qubit.index])

    def test_classical_reg(self, measurement: Measurement, classical_reg: MemoryReference):
        assert measurement.classical_reg == classical_reg
        measurement.classical_reg = MemoryReference("new_mem_ref")
        assert measurement.classical_reg == MemoryReference("new_mem_ref")


@pytest.mark.parametrize(
    ("frame", "direction", "initial_frequency", "hardware_object", "sample_rate", "center_frequency"),
    [
        (Frame([Qubit(0)], "frame"), None, None, None, None, None),
        (Frame([Qubit(1)], "frame"), "direction", 1.39, "hardware_object", 44.1, 440.0),
    ],
    ids=("Frame-Only", "With-Optionals"),
)
class TestDefFrame:
    @pytest.fixture
    def def_frame(
        self,
        frame: Frame,
        direction: Optional[str],
        initial_frequency: Optional[float],
        hardware_object: Optional[str],
        sample_rate: Optional[float],
        center_frequency: Optional[float],
    ):
        optional_args = [
            arg
            for arg in [direction, initial_frequency, hardware_object, sample_rate, center_frequency]
            if arg is not None
        ]
        return DefFrame(frame, *optional_args)

    def test_out(self, def_frame: DefFrame, snapshot: SnapshotAssertion):
        # The ordering of attributes isn't stable and can be printed in different orders across calls.
        # We assert that the first line is definitely the `DEFFRAME` line, and that the following
        # attributes are the same, regardless of their order.
        quil_lines = def_frame.out().splitlines()
        assert quil_lines[0] == snapshot
        assert set(quil_lines[1:]) == snapshot

    def test_str(self, def_frame: DefFrame, snapshot: SnapshotAssertion):
        quil_lines = str(def_frame).splitlines()
        assert quil_lines[0] == snapshot
        assert set(quil_lines[1:]) == snapshot

    def test_frame(self, def_frame: DefFrame, frame: Frame):
        assert def_frame.frame == frame
        def_frame.frame = Frame([Qubit(123)], "new_frame")
        assert def_frame.frame == Frame([Qubit(123)], "new_frame")

    def test_direction(self, def_frame: DefFrame, direction: Optional[str]):
        assert def_frame.direction == direction
        def_frame.direction = "tx"
        assert def_frame.direction == "tx"

    def test_initial_frequency(self, def_frame: DefFrame, initial_frequency: Optional[float]):
        assert def_frame.initial_frequency == initial_frequency
        def_frame.initial_frequency = 3.14
        assert def_frame.initial_frequency == 3.14

    def test_hardware_object(self, def_frame: DefFrame, hardware_object: Optional[str]):
        assert def_frame.hardware_object == hardware_object
        def_frame.hardware_object = "bfg"
        assert def_frame.hardware_object == "bfg"

    def test_sample_rate(self, def_frame: DefFrame, sample_rate: Optional[float]):
        assert def_frame.sample_rate == sample_rate
        def_frame.sample_rate = 96.0
        assert def_frame.sample_rate == 96.0

    def test_center_frequency(self, def_frame: DefFrame, center_frequency: Optional[float]):
        assert def_frame.center_frequency == center_frequency
        def_frame.center_frequency = 432.0
        assert def_frame.center_frequency == 432.0


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
        shared_region: Optional[str],
        offsets: Optional[Iterable[Tuple[int, str]]],
    ) -> Declare:
        return Declare(name, memory_type, memory_size, shared_region, offsets)

    def test_out(self, declare: Declare, snapshot: SnapshotAssertion):
        assert declare.out() == snapshot

    def test_str(self, declare: Declare, snapshot: SnapshotAssertion):
        assert str(declare) == snapshot

    def test_asdict(self, declare: Declare, snapshot: SnapshotAssertion):
        assert declare.asdict() == snapshot

    def test_name(self, declare: Declare, name: str):
        assert declare.name == name
        declare.name = "new_name"
        assert declare.name == "new_name"

    def test_memory_type(self, declare: Declare, memory_type: Optional[str]):
        assert declare.memory_type == memory_type
        declare.memory_type = "REAL"
        assert declare.memory_type == "REAL"

    def test_memory_size(self, declare: Declare, memory_size: Optional[int]):
        assert declare.memory_size == memory_size
        declare.memory_size = 100
        assert declare.memory_size == 100

    def test_shared_region(self, declare: Declare, shared_region: Optional[str]):
        assert declare.shared_region == shared_region
        declare.shared_region = "new_shared"
        assert declare.shared_region == "new_shared"

    def test_offsets(self, declare: Declare, offsets: Optional[Iterable[Tuple[int, str]]]):
        expected_offsets = offsets or []
        assert declare.offsets == expected_offsets
        if declare.shared_region is None:
            with pytest.raises(ValueError):
                declare.offsets = [(1, "BIT"), (2, "INTEGER")]
        else:
            declare.offsets = [(1, "BIT"), (2, "INTEGER")]
            assert declare.offsets == [(1, "BIT"), (2, "INTEGER")]


@pytest.mark.parametrize(
    ("command", "args", "freeform_string"),
    [
        ("NO-NOISE", [], ""),
        ("DOES-A-THING", [Qubit(0), FormalArgument("b")], ""),
        ("INITIAL_REWIRING", [], "GREEDY"),
        ("READOUT-POVM", [Qubit(1)], "(0.9 0.19999999999999996 0.09999999999999998 0.8)"),
    ],
    ids=("Command-Only", "With-Arg", "With-String", "With-Arg-And-String"),
)
class TestPragma:
    @pytest.fixture
    def pragma(self, command: str, args: List[Union[QubitDesignator, str]], freeform_string: str) -> Pragma:
        return Pragma(command, args, freeform_string)

    def test_out(self, pragma: Pragma, snapshot: SnapshotAssertion):
        assert pragma.out() == snapshot

    def test_str(self, pragma: Pragma, snapshot: SnapshotAssertion):
        assert str(pragma) == snapshot

    def test_command(self, pragma: Pragma, command: str):
        assert pragma.command == command
        pragma.command = "NEW_COMMAND"
        assert pragma.command == "NEW_COMMAND"

    def test_args(self, pragma: Pragma, args: List[Union[QubitDesignator, str]]):
        assert pragma.args == tuple(args)
        pragma.args = (Qubit(123),)
        assert pragma.args == (Qubit(123),)

    def test_freeform_string(self, pragma: Pragma, freeform_string: str):
        assert pragma.freeform_string == freeform_string
        pragma.freeform_string = "new string"
        assert pragma.freeform_string == "new string"


@pytest.mark.parametrize(
    ("qubit"),
    [
        (Qubit(0)),
        (FormalArgument("a")),
        (None),
    ],
    ids=("Qubit", "FormalArgument", "None"),
)
class TestReset:
    @pytest.fixture
    def reset_qubit(self, qubit: Qubit) -> ResetQubit:
        if qubit is None:
            with pytest.raises(TypeError):
                ResetQubit(qubit)
            return Reset(None)
        return ResetQubit(qubit)

    def test_out(self, reset_qubit: ResetQubit, snapshot: SnapshotAssertion):
        assert reset_qubit.out() == snapshot

    def test_str(self, reset_qubit: ResetQubit, snapshot: SnapshotAssertion):
        assert str(reset_qubit) == snapshot

    def test_qubit(self, reset_qubit: ResetQubit, qubit: Qubit):
        assert reset_qubit.qubit == qubit
        reset_qubit.qubit = FormalArgument("a")
        assert reset_qubit.qubit == FormalArgument("a")

    def test_get_qubits(self, reset_qubit: ResetQubit, qubit: Qubit):
        if qubit is None:
            assert reset_qubit.get_qubits(False) is None
            assert reset_qubit.get_qubit_indices() is None
        else:
            assert reset_qubit.get_qubits(False) == {qubit}
            if isinstance(qubit, Qubit):
                assert reset_qubit.get_qubit_indices() == {qubit.index}


@pytest.mark.parametrize(
    ("frames", "duration"),
    [
        ([Frame([Qubit(0)], "frame")], 5.0),
    ],
)
class TestDelayFrames:
    @pytest.fixture
    def delay_frames(self, frames: List[Frame], duration: float) -> DelayFrames:
        return DelayFrames(frames, duration)

    def test_out(self, delay_frames: DelayFrames, snapshot: SnapshotAssertion):
        assert delay_frames.out() == snapshot

    def test_frames(self, delay_frames: DelayFrames, frames: List[Frame]):
        assert delay_frames.frames == frames
        delay_frames.frames = [Frame([Qubit(123)], "new_frame")]
        assert delay_frames.frames == [Frame([Qubit(123)], "new_frame")]

    def test_duration(self, delay_frames: DelayFrames, duration: float):
        assert delay_frames.duration == duration
        delay_frames.duration = 3.14
        assert delay_frames.duration == 3.14


@pytest.mark.parametrize(
    ("qubits", "duration"),
    [
        ([Qubit(0)], 5.0),
        ([FormalArgument("a")], 2.5),
    ],
    ids=("Qubit", "FormalArgument"),
)
class TestDelayQubits:
    @pytest.fixture
    def delay_qubits(self, qubits: List[Union[Qubit, FormalArgument]], duration: float) -> DelayQubits:
        return DelayQubits(qubits, duration)

    def test_out(self, delay_qubits: DelayQubits, snapshot: SnapshotAssertion):
        assert delay_qubits.out() == snapshot

    def test_qubits(self, delay_qubits: DelayQubits, qubits: List[Qubit]):
        assert delay_qubits.qubits == qubits
        delay_qubits.qubits = [Qubit(123)]  # type: ignore
        assert delay_qubits.qubits == [Qubit(123)]

    def test_duration(self, delay_qubits: DelayQubits, duration: float):
        assert delay_qubits.duration == duration
        delay_qubits.duration = 3.14
        assert delay_qubits.duration == 3.14


@pytest.mark.parametrize(
    ("qubits"),
    [
        ([Qubit(0)]),
        ([FormalArgument("a")]),
    ],
    ids=("Qubit", "FormalArgument"),
)
class TestFence:
    @pytest.fixture
    def fence(self, qubits: List[Union[Qubit, FormalArgument]]) -> Fence:
        return Fence(qubits)

    def test_out(self, fence: Fence, snapshot: SnapshotAssertion):
        assert fence.out() == snapshot

    def test_qubits(self, fence: Fence, qubits: List[Union[Qubit, FormalArgument]]):
        assert fence.qubits == qubits
        fence.qubits = [Qubit(123)]  # type: ignore
        assert fence.qubits == [Qubit(123)]


def test_fence_all():
    fa = FenceAll()
    assert fa.out() == "FENCE"
    assert fa.qubits == []


@pytest.mark.parametrize(
    ("name", "parameters", "entries"),
    [
        ("Wavey", [], []),
        ("Wavey", [Parameter("x")], [Parameter("x")]),
        (
            "Wavey",
            [Parameter("x"), Parameter("y")],
            [complex(1.0, 2.0), Parameter("x"), Mul(complex(3.0, 0.0), Parameter("y"))],
        ),
    ],
    ids=("No-Params-Entries", "With-Param", "With-Params-Complex"),
)
class TestDefWaveform:
    @pytest.fixture
    def def_waveform(self, name: str, parameters: List[Parameter], entries: List[Union[Complex, Expression]]):
        return DefWaveform(name, parameters, entries)

    def test_out(self, def_waveform: DefWaveform, snapshot: SnapshotAssertion):
        assert def_waveform.out() == snapshot

    def test_name(self, def_waveform: DefWaveform, name: str):
        assert def_waveform.name == name
        def_waveform.name = "new_name"
        assert def_waveform.name == "new_name"

    def test_parameters(self, def_waveform: DefWaveform, parameters: List[Parameter]):
        assert def_waveform.parameters == parameters
        def_waveform.parameters = [Parameter("z")]
        assert def_waveform.parameters == [Parameter("z")]

    def test_entries(self, def_waveform: DefWaveform, entries: List[Union[Complex, Expression]]):
        assert def_waveform.entries == entries
        def_waveform.entries = [Parameter("z")]  # type: ignore
        assert def_waveform.entries == [Parameter("z")]


@pytest.mark.parametrize(
    ("name", "parameters", "qubit_variables", "instructions"),
    [
        ("NiftyCircuit", [], [FormalArgument("a")], [Measurement(FormalArgument("a"), None)]),
        (
            "NiftyCircuit",
            [Parameter("theta")],
            [FormalArgument("a")],
            [
                Declare("ro", "BIT", 1),
                Measurement(FormalArgument("a"), MemoryReference("ro")),
                DefGate("ParameterizedGate", np.diag([Parameter("theta")] * 4), [Parameter("theta")]),
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
        parameters: List[Parameter],
        qubit_variables: List[FormalArgument],
        instructions: List[AbstractInstruction],
    ):
        return DefCircuit(name, parameters, qubit_variables, instructions)

    def test_out(self, def_circuit: DefCircuit, snapshot: SnapshotAssertion):
        assert def_circuit.out() == snapshot

    def test_name(self, def_circuit: DefCircuit, name: str):
        assert def_circuit.name == name
        def_circuit.name = "new_name"
        assert def_circuit.name == "new_name"

    def test_parameters(self, def_circuit: DefCircuit, parameters: List[Parameter]):
        assert def_circuit.parameters == parameters
        def_circuit.parameters = [Parameter("z")]
        assert def_circuit.parameters == [Parameter("z")]

    def test_qubit_variables(self, def_circuit: DefCircuit, qubit_variables: List[FormalArgument]):
        assert def_circuit.qubit_variables == qubit_variables
        def_circuit.qubit_variables = [FormalArgument("qubit")]
        assert def_circuit.qubit_variables == [FormalArgument("qubit")]

    def test_instructions(self, def_circuit: DefCircuit, instructions: List[AbstractInstruction]):
        assert def_circuit.instructions == instructions
        def_circuit.instructions = [Gate("new_gate", [], [Qubit(0)], [])]
        assert def_circuit.instructions == [Gate("new_gate", [], [Qubit(0)], [])]
