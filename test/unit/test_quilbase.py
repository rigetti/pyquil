import copy
import pickle
from collections.abc import Iterable
from math import pi
from numbers import Complex, Number
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from pyquil.api._compiler import QPUCompiler
from pyquil.gates import X
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import (
    BinaryExp,
    Expression,
    Frame,
    Mul,
    Qubit,
    TemplateWaveform,
    Waveform,
    WaveformReference,
    quil_cos,
    quil_sin,
)
from pyquil.quilbase import (
    AbstractInstruction,
    ArithmeticBinaryOp,
    Capture,
    ClassicalAdd,
    ClassicalAnd,
    ClassicalComparison,
    ClassicalConvert,
    ClassicalDiv,
    ClassicalEqual,
    ClassicalExchange,
    ClassicalExclusiveOr,
    ClassicalGreaterEqual,
    ClassicalGreaterThan,
    ClassicalInclusiveOr,
    ClassicalLessEqual,
    ClassicalLessThan,
    ClassicalLoad,
    ClassicalMove,
    ClassicalMul,
    ClassicalNeg,
    ClassicalNot,
    ClassicalStore,
    ClassicalSub,
    Declare,
    DefCalibration,
    DefCircuit,
    DefFrame,
    DefGate,
    DefGateByPaulis,
    DefMeasureCalibration,
    DefPermutationGate,
    DefWaveform,
    DelayFrames,
    DelayQubits,
    Fence,
    FenceAll,
    FormalArgument,
    Gate,
    Halt,
    Include,
    LogicalBinaryOp,
    Measurement,
    MemoryReference,
    Nop,
    Parameter,
    ParameterDesignator,
    Pragma,
    Pulse,
    QubitDesignator,
    RawCapture,
    Reset,
    ResetQubit,
    SetFrequency,
    SetPhase,
    SetScale,
    ShiftFrequency,
    ShiftPhase,
    SwapPhases,
    UnaryClassicalInstruction,
    Wait,
    _convert_to_py_instruction,
    _convert_to_rs_instruction,
)
from pyquil.quiltwaveforms import (
    BoxcarAveragerKernel,
    DragGaussianWaveform,
    ErfSquareWaveform,
    FlatWaveform,
    GaussianWaveform,
    HrmGaussianWaveform,
)


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
        assert gate.get_qubit_indices() == [q.index for q in qubits]
        assert gate.get_qubits(indices=False) == qubits

    def test_controlled_modifier(self, gate: Gate, snapshot: SnapshotAssertion):
        assert str(gate.controlled([Qubit(5)])) == snapshot

    def test_dagger_modifier(self, gate: Gate, snapshot: SnapshotAssertion):
        assert str(gate.dagger()) == snapshot

    def test_forked_modifier(self, gate: Gate, params: List[ParameterDesignator], snapshot: SnapshotAssertion):
        alt_params: List[ParameterDesignator] = [n for n in range(len(params))]
        assert str(gate.forked(Qubit(5), alt_params)) == snapshot

    def test_modifiers(self, gate: Gate):
        assert gate.modifiers == []
        gate.modifiers = ["CONTROLLED"]
        assert gate.modifiers == ["CONTROLLED"]

    def test_repr(self, gate: Gate, snapshot: SnapshotAssertion):
        assert repr(gate) == snapshot

    def test_eq(self, gate: Gate, name: str, params: List[ParameterDesignator], qubits: List[Qubit]):
        assert gate == Gate(name, params, qubits)
        assert not gate != Gate(name, params, qubits)
        not_eq_gate = Gate(f"not-{name}", params, qubits)
        assert not (gate == not_eq_gate)
        assert gate != not_eq_gate

    def test_convert(self, gate: Gate):
        rs_gate = _convert_to_rs_instruction(gate)
        assert gate == _convert_to_py_instruction(rs_gate)

    def test_copy(self, gate: Gate):
        assert isinstance(copy.copy(gate), Gate)
        assert isinstance(copy.deepcopy(gate), Gate)

    def test_compile(self, program: Program, compiler: QPUCompiler):
        try:
            compiler.quil_to_native_quil(program)
        except Exception as e:
            raise AssertionError(f"Failed to compile the program: \n{program}") from e

    def test_pickle(self, gate: Gate):
        pickled = pickle.dumps(gate)
        unpickled = pickle.loads(pickled)
        assert unpickled == gate


@pytest.mark.parametrize(
    ("name", "matrix", "parameters"),
    [
        ("NoParamGate", np.eye(4), []),
        ("ParameterizedGate", np.diag([quil_cos(Parameter("X"))] * 4), [Parameter("X")]),
        (
            "MixedTypes",
            np.array(
                [
                    [0, quil_sin(Parameter("X"))],
                    [0, 0],
                ]
            ),
            [Parameter("X")],
        ),
        (
            "ParameterlessExpressions",
            np.array(
                [
                    [-quil_cos(np.pi), quil_sin(np.pi)],
                    [quil_sin(np.pi), quil_cos(np.pi)],
                ]
            ),
            [],
        ),
    ],
    ids=("No-Params", "Params", "MixedTypes", "ParameterlessExpression"),
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
            g = constructor(Parameter("theta"))(Qubit(123))  # type: ignore
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

    def test_copy(self, def_gate: DefGate):
        assert isinstance(copy.copy(def_gate), DefGate)
        assert isinstance(copy.deepcopy(def_gate), DefGate)

    def test_pickle(self, def_gate: DefGate, snapshot: SnapshotAssertion):
        pickled = pickle.dumps(def_gate)
        unpickled = pickle.loads(pickled)
        assert unpickled == snapshot


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
            g = constructor(Parameter("theta"))(Qubit(123))  # type: ignore
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

    def test_qubits(self, calibration: DefCalibration, qubits: List[QubitDesignator]):
        assert calibration.qubits == qubits
        calibration.qubits = [Qubit(123)]
        assert calibration.qubits == [Qubit(123)]

    def test_instrs(self, calibration: DefCalibration, instrs: List[AbstractInstruction]):
        assert calibration.instrs == instrs
        calibration.instrs = [Gate("SomeGate", [], [Qubit(0)], [])]
        assert calibration.instrs == [Gate("SomeGate", [], [Qubit(0)], [])]

    def test_copy(self, calibration: DefCalibration):
        assert isinstance(copy.copy(calibration), DefCalibration)
        assert isinstance(copy.deepcopy(calibration), DefCalibration)

    def test_convert(self, calibration: DefCalibration):
        rs_calibration = _convert_to_rs_instruction(calibration)
        assert calibration == _convert_to_py_instruction(rs_calibration)

    def test_pickle(self, calibration: DefCalibration):
        pickled = pickle.dumps(calibration)
        unpickled = pickle.loads(pickled)
        assert unpickled == calibration


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

    def test_copy(self, measure_calibration: DefMeasureCalibration):
        assert isinstance(copy.copy(measure_calibration), DefMeasureCalibration)
        assert isinstance(copy.deepcopy(measure_calibration), DefMeasureCalibration)

    def test_convert(self, measure_calibration: DefMeasureCalibration):
        rs_measure_calibration = _convert_to_rs_instruction(measure_calibration)
        assert measure_calibration == _convert_to_py_instruction(rs_measure_calibration)

    def test_pickle(self, measure_calibration: DefMeasureCalibration):
        pickled = pickle.dumps(measure_calibration)
        unpickled = pickle.loads(pickled)
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

    def test_copy(self, measurement: Measurement):
        assert isinstance(copy.copy(measurement), Measurement)
        assert isinstance(copy.deepcopy(measurement), Measurement)

    def test_convert(self, measurement: Measurement):
        rs_measurement = _convert_to_rs_instruction(measurement)
        assert measurement == _convert_to_py_instruction(rs_measurement)

    def test_pickle(self, measurement: Measurement):
        pickled = pickle.dumps(measurement)
        unpickled = pickle.loads(pickled)
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
        direction: Optional[str],
        initial_frequency: Optional[float],
        hardware_object: Optional[str],
        sample_rate: Optional[float],
        center_frequency: Optional[float],
        channel_delay: Optional[float],
    ) -> DefFrame:
        args = dict(
            direction=direction,
            initial_frequency=initial_frequency,
            hardware_object=hardware_object,
            sample_rate=sample_rate,
            center_frequency=center_frequency,
            channel_delay=channel_delay,
        )
        return DefFrame(frame, **{k: v for k, v in args.items() if v is not None})

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

    def test_hardware_object_json(self, def_frame: DefFrame, hardware_object: Optional[str]):
        assert def_frame.hardware_object == hardware_object
        def_frame.hardware_object = '{"string": "str", "int": 1, "float": 3.14}'
        assert def_frame.hardware_object == '{"string": "str", "int": 1, "float": 3.14}'

    def test_sample_rate(self, def_frame: DefFrame, sample_rate: Optional[float]):
        assert def_frame.sample_rate == sample_rate
        def_frame.sample_rate = 96.0
        assert def_frame.sample_rate == 96.0

    def test_center_frequency(self, def_frame: DefFrame, center_frequency: Optional[float]):
        assert def_frame.center_frequency == center_frequency
        def_frame.center_frequency = 432.0
        assert def_frame.center_frequency == 432.0

    def test_channel_delay(self, def_frame: DefFrame, channel_delay: Optional[float]):
        assert def_frame.channel_delay == channel_delay
        def_frame.channel_delay = 571.0
        assert def_frame.channel_delay == 571.0

    def test_copy(self, def_frame: DefFrame):
        assert isinstance(copy.copy(def_frame), DefFrame)
        assert isinstance(copy.deepcopy(def_frame), DefFrame)

    def test_convert(self, def_frame: DefFrame):
        rs_def_frame = _convert_to_rs_instruction(def_frame)
        assert def_frame == _convert_to_py_instruction(rs_def_frame)

    def test_pickle(self, def_frame: DefFrame):
        print(def_frame.to_quil())
        pickled = pickle.dumps(def_frame)
        unpickled = pickle.loads(pickled)
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

    def test_copy(self, declare: Declare):
        assert isinstance(copy.copy(declare), Declare)
        assert isinstance(copy.deepcopy(declare), Declare)

    def test_convert(self, declare: Declare):
        rs_declare = _convert_to_rs_instruction(declare)
        assert declare == _convert_to_py_instruction(rs_declare)

    def test_pickle(self, declare: Declare):
        pickled = pickle.dumps(declare)
        unpickled = pickle.loads(pickled)
        assert unpickled == declare


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

    def test_copy(self, pragma: Pragma):
        assert isinstance(copy.copy(pragma), Pragma)
        assert isinstance(copy.deepcopy(pragma), Pragma)

    def test_convert(self, pragma: Pragma):
        rs_pragma = _convert_to_rs_instruction(pragma)
        assert pragma == _convert_to_py_instruction(rs_pragma)

    def test_pickle(self, pragma: Pragma):
        pickled = pickle.dumps(pragma)
        unpickled = pickle.loads(pickled)
        assert unpickled == pragma


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
    def reset_qubit(self, qubit: Qubit) -> Union[Reset, ResetQubit]:
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

    def test_copy(self, reset_qubit: Union[Reset, ResetQubit]):
        assert isinstance(copy.copy(reset_qubit), type(reset_qubit))
        assert isinstance(copy.deepcopy(reset_qubit), type(reset_qubit))

    def test_convert(self, reset_qubit: Reset):
        rs_reset_qubit = _convert_to_rs_instruction(reset_qubit)
        assert reset_qubit == _convert_to_py_instruction(rs_reset_qubit)

    def test_pickle(self, reset_qubit: Reset):
        pickled = pickle.dumps(reset_qubit)
        unpickled = pickle.loads(pickled)
        assert unpickled == reset_qubit


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

    def test_copy(self, delay_frames: DelayFrames):
        assert isinstance(copy.copy(delay_frames), DelayFrames)
        assert isinstance(copy.deepcopy(delay_frames), DelayFrames)

    def test_convert(self, delay_frames: DelayFrames):
        rs_delay_frames = _convert_to_rs_instruction(delay_frames)
        assert delay_frames == _convert_to_py_instruction(rs_delay_frames)

    def test_pickle(self, delay_frames: DelayFrames):
        pickled = pickle.dumps(delay_frames)
        unpickled = pickle.loads(pickled)
        assert unpickled == delay_frames


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

    def test_copy(self, delay_qubits: DelayQubits):
        assert isinstance(copy.copy(delay_qubits), DelayQubits)
        assert isinstance(copy.deepcopy(delay_qubits), DelayQubits)

    def test_convert(self, delay_qubits: DelayQubits):
        rs_delay_qubits = _convert_to_rs_instruction(delay_qubits)
        assert delay_qubits == _convert_to_py_instruction(rs_delay_qubits)

    def test_pickle(self, delay_qubits: DelayQubits):
        pickled = pickle.dumps(delay_qubits)
        unpickled = pickle.loads(pickled)
        assert unpickled == delay_qubits


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

    def test_copy(self, fence: Fence):
        assert isinstance(copy.copy(fence), Fence)
        assert isinstance(copy.deepcopy(fence), Fence)

    def test_convert(self, fence: Fence):
        rs_fence = _convert_to_rs_instruction(fence)
        assert fence == _convert_to_py_instruction(rs_fence)

    def test_pickle(self, fence: Fence):
        pickled = pickle.dumps(fence)
        unpickled = pickle.loads(pickled)
        assert unpickled == fence


def test_fence_all():
    fa = FenceAll()
    assert fa.out() == "FENCE"
    assert fa.qubits == []


@pytest.mark.parametrize(
    ("name", "parameters", "entries"),
    [
        ("Wavey", [Parameter("x")], [Parameter("x")]),
        (
            "Wavey",
            [Parameter("x"), Parameter("y")],
            [complex(1.0, 2.0), Parameter("x"), Mul(complex(3.0, 0.0), Parameter("y"))],
        ),
    ],
    ids=("With-Param", "With-Params-Complex"),
)
class TestDefWaveform:
    @pytest.fixture
    def def_waveform(self, name: str, parameters: List[Parameter], entries: List[Union[complex, Expression]]):
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

    def test_copy(self, def_waveform: DefWaveform):
        assert isinstance(copy.copy(def_waveform), DefWaveform)
        assert isinstance(copy.deepcopy(def_waveform), DefWaveform)

    def test_convert(self, def_waveform: DefWaveform):
        rs_def_waveform = _convert_to_rs_instruction(def_waveform)
        assert def_waveform == _convert_to_py_instruction(rs_def_waveform)

    def test_pickle(self, def_waveform: DefWaveform, snapshot: SnapshotAssertion):
        print(def_waveform.to_quil())
        pickled = pickle.dumps(def_waveform)
        unpickled = pickle.loads(pickled)
        assert unpickled == snapshot


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

    def test_copy(self, def_circuit: DefCircuit):
        assert isinstance(copy.copy(def_circuit), DefCircuit)
        assert isinstance(copy.deepcopy(def_circuit), DefCircuit)

    def test_convert(self, def_circuit: DefCircuit):
        rs_def_circuit = _convert_to_rs_instruction(def_circuit)
        assert def_circuit == _convert_to_py_instruction(rs_def_circuit)

    def test_pickle(self, def_circuit: DefCircuit):
        print(def_circuit.to_quil())
        pickled = pickle.dumps(def_circuit)
        unpickled = pickle.loads(pickled)
        assert unpickled == def_circuit


@pytest.mark.parametrize(
    ("frame", "kernel", "memory_region", "nonblocking"),
    [
        (
            Frame([Qubit(123), FormalArgument("q")], "FRAMEX"),
            WaveformReference("WAVEFORMY"),
            MemoryReference("ro"),
            False,
        ),
        (
            Frame([Qubit(123), FormalArgument("q")], "FRAMEX"),
            WaveformReference("WAVEFORMY"),
            MemoryReference("ro"),
            True,
        ),
        (
            Frame([Qubit(123), FormalArgument("q")], "FRAMEX"),
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
        return Capture(frame, kernel, memory_region, nonblocking)

    def test_out(self, capture: Capture, snapshot: SnapshotAssertion):
        assert capture.out() == snapshot

    def test_frame(self, capture: Capture, frame: Frame):
        assert capture.frame == frame
        capture.frame = Frame([Qubit(123)], "new-frame")
        assert capture.frame == Frame([Qubit(123)], "new-frame")

    def test_kernel(self, capture: Capture, kernel: Waveform):
        assert capture.kernel == kernel
        capture.kernel = WaveformReference("new-waveform")
        assert capture.kernel == WaveformReference("new-waveform")

    def test_memory_region(self, capture: Capture, memory_region: MemoryReference):
        assert capture.memory_region == memory_region
        capture.memory_region = MemoryReference("new-memory-reference")
        assert capture.memory_region == MemoryReference("new-memory-reference")

    def test_nonblocking(self, capture: Capture, nonblocking: bool):
        assert capture.nonblocking == nonblocking
        capture.nonblocking = not nonblocking
        assert capture.nonblocking == (not nonblocking)

    def test_copy(self, capture: Capture):
        assert isinstance(copy.copy(capture), Capture)
        assert isinstance(copy.deepcopy(capture), Capture)

    def test_convert(self, capture: Capture):
        rs_capture = _convert_to_rs_instruction(capture)
        assert capture == _convert_to_py_instruction(rs_capture)

    def test_pickle(self, capture: Capture, snapshot: SnapshotAssertion):
        pickled = pickle.dumps(capture)
        unpickled = pickle.loads(pickled)
        assert unpickled == snapshot


@pytest.mark.parametrize(
    ("frame", "waveform", "nonblocking"),
    [
        (
            Frame([Qubit(123), FormalArgument("q")], "FRAMEX"),
            WaveformReference("WAVEFORMY"),
            False,
        ),
        (
            Frame([Qubit(123), FormalArgument("q")], "FRAMEX"),
            WaveformReference("WAVEFORMY"),
            True,
        ),
        (
            Frame([Qubit(123), FormalArgument("q")], "FRAMEX"),
            FlatWaveform(duration=2.5, iq=complex(1.0, 2.0)),
            True,
        ),
        (
            Frame([Qubit(123), FormalArgument("q")], "FRAMEX"),
            GaussianWaveform(duration=2.5, fwhm=1.0, t0=1.0, phase=0.1),
            True,
        ),
        (
            Frame([Qubit(123), FormalArgument("q")], "FRAMEX"),
            DragGaussianWaveform(duration=2.5, fwhm=1.0, t0=1.0, anh=0.1, alpha=1.0),
            True,
        ),
        (
            Frame([Qubit(123), FormalArgument("q")], "FRAMEX"),
            HrmGaussianWaveform(duration=2.5, fwhm=1.0, t0=1.0, anh=0.1, alpha=1.0, second_order_hrm_coeff=0.5),
            True,
        ),
        (
            Frame([Qubit(123), FormalArgument("q")], "FRAMEX"),
            ErfSquareWaveform(duration=2.5, risetime=1.0, pad_left=1.0, pad_right=0.1, scale=1.0),
            True,
        ),
        (
            Frame([Qubit(123), FormalArgument("q")], "FRAMEX"),
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
        return Pulse(frame, waveform, nonblocking)

    def test_out(self, pulse: Pulse, snapshot: SnapshotAssertion):
        assert pulse.out() == snapshot

    def test_frame(self, pulse: Pulse, frame: Frame):
        assert pulse.frame == frame
        pulse.frame = Frame([Qubit(123)], "new-frame")
        assert pulse.frame == Frame([Qubit(123)], "new-frame")

    def test_waveform(self, pulse: Pulse, waveform: Waveform):
        assert pulse.waveform == waveform
        if isinstance(waveform, TemplateWaveform):
            assert isinstance(pulse.waveform, type(waveform))
            pulse.waveform.samples(0.5)
        pulse.waveform = WaveformReference("new-waveform")
        assert pulse.waveform == WaveformReference("new-waveform")

    def test_nonblocking(self, pulse: Pulse, nonblocking: bool):
        assert pulse.nonblocking == nonblocking
        pulse.nonblocking = not nonblocking
        assert pulse.nonblocking == (not nonblocking)

    def test_copy(self, pulse: Pulse):
        assert isinstance(copy.copy(pulse), Pulse)
        assert isinstance(copy.deepcopy(pulse), Pulse)

    def test_convert(self, pulse: Pulse):
        rs_pulse = _convert_to_rs_instruction(pulse)
        assert pulse == _convert_to_py_instruction(rs_pulse)

    def test_pickle(self, pulse: Pulse, snapshot: SnapshotAssertion):
        pickled = pickle.dumps(pulse)
        unpickled = pickle.loads(pickled)
        assert unpickled == snapshot


@pytest.mark.parametrize(
    ("frame", "duration", "memory_region", "nonblocking"),
    [
        (
            Frame([Qubit(123), FormalArgument("q")], "FRAMEX"),
            0.5,
            MemoryReference("ro"),
            False,
        ),
        (
            Frame([Qubit(123), FormalArgument("q")], "FRAMEX"),
            2.5,
            MemoryReference("ro"),
            True,
        ),
        (
            Frame([Qubit(123), FormalArgument("q")], "FRAMEX"),
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
        return RawCapture(frame, duration, memory_region, nonblocking)

    def test_out(self, raw_capture: RawCapture, snapshot: SnapshotAssertion):
        assert raw_capture.out() == snapshot

    def test_frame(self, raw_capture: RawCapture, frame: Frame):
        assert raw_capture.frame == frame
        raw_capture.frame = Frame([Qubit(123)], "new-frame")
        assert raw_capture.frame == Frame([Qubit(123)], "new-frame")

    def test_duration(self, raw_capture: RawCapture, duration: float):
        assert raw_capture.duration == duration
        raw_capture.duration = 3.14
        assert raw_capture.duration == 3.14

    def test_memory_region(self, raw_capture: RawCapture, memory_region: MemoryReference):
        assert raw_capture.memory_region == memory_region
        raw_capture.memory_region = MemoryReference("new-memory-reference")
        assert raw_capture.memory_region == MemoryReference("new-memory-reference")

    def test_nonblocking(self, raw_capture: RawCapture, nonblocking: bool):
        assert raw_capture.nonblocking == nonblocking
        raw_capture.nonblocking = not nonblocking
        assert raw_capture.nonblocking == (not nonblocking)

    def test_copy(self, raw_capture: RawCapture):
        assert isinstance(copy.copy(raw_capture), RawCapture)
        assert isinstance(copy.deepcopy(raw_capture), RawCapture)

    def test_convert(self, raw_capture: RawCapture):
        rs_raw_capture = _convert_to_rs_instruction(raw_capture)
        assert raw_capture == _convert_to_py_instruction(rs_raw_capture)

    def test_pickle(self, raw_capture: RawCapture):
        pickled = pickle.dumps(raw_capture)
        unpickled = pickle.loads(pickled)
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
    ) -> Tuple[SetFrequency, SetPhase, ShiftFrequency, ShiftPhase, SetScale]:
        return (set_frequency, set_phase, shift_frequency, shift_phase, set_scale)

    def test_out(self, frame_mutation_instructions, snapshot: SnapshotAssertion):
        for instr in frame_mutation_instructions:
            assert instr.out() == snapshot

    def test_frame(self, frame_mutation_instructions, frame: Frame):
        for instr in frame_mutation_instructions:
            assert instr.frame == frame
            instr.frame = Frame([Qubit(123)], "NEW-FRAME")
            assert instr.frame == Frame([Qubit(123)], "NEW-FRAME")

    def test_get_qubits(self, frame_mutation_instructions, frame: Frame):
        for instr in frame_mutation_instructions:
            assert instr.get_qubits() == set([q.index for q in frame.qubits if isinstance(q, Qubit)])
            assert instr.get_qubits(False) == set(frame.qubits)

    def test_expression(self, frame_mutation_instructions, expression: ParameterDesignator):
        expression_names = ["freq", "phase", "freq", "phase", "scale"]
        for instr, expression_name in zip(frame_mutation_instructions, expression_names):
            assert getattr(instr, expression_name) == expression
            setattr(instr, expression_name, 3.14)
            assert getattr(instr, expression_name) == 3.14

    def test_copy(self, frame_mutation_instructions):
        for instr in frame_mutation_instructions:
            assert isinstance(copy.copy(instr), type(instr))
            assert isinstance(copy.deepcopy(instr), type(instr))

    def test_convert(self, frame_mutation_instructions):
        for instr in frame_mutation_instructions:
            rs_instr = _convert_to_rs_instruction(instr)
            assert instr == _convert_to_py_instruction(rs_instr)


@pytest.mark.parametrize(
    ("frame_a", "frame_b"),
    [(Frame([Qubit(1)], "FRAMEX"), Frame([Qubit(2)], "FRAMEX"))],
)
class TestSwapPhases:
    @pytest.fixture
    def swap_phases(self, frame_a, frame_b):
        return SwapPhases(frame_a, frame_b)

    def test_out(self, swap_phases: SwapPhases, snapshot: SnapshotAssertion):
        assert swap_phases.out() == snapshot

    def test_frames(self, swap_phases: SwapPhases, frame_a: Frame, frame_b: Frame):
        assert swap_phases.frameA == frame_a
        assert swap_phases.frameB == frame_b
        swap_phases.frameA = Frame([Qubit(123)], "NEW-FRAME")
        swap_phases.frameB = Frame([Qubit(123)], "NEW-FRAME")
        assert swap_phases.frameA == Frame([Qubit(123)], "NEW-FRAME")
        assert swap_phases.frameB == Frame([Qubit(123)], "NEW-FRAME")

    def test_get_qubits(self, swap_phases: SwapPhases, frame_a: Frame, frame_b: Frame):
        expected_qubits = set(frame_a.qubits + frame_b.qubits)
        assert swap_phases.get_qubits() == set([q.index for q in expected_qubits if isinstance(q, Qubit)])
        assert swap_phases.get_qubits(False) == expected_qubits

    def test_copy(self, swap_phases: SwapPhases):
        assert isinstance(copy.copy(swap_phases), SwapPhases)
        assert isinstance(copy.deepcopy(swap_phases), SwapPhases)

    def test_convert(self, swap_phases: SwapPhases):
        rs_swap_phase = _convert_to_rs_instruction(swap_phases)
        assert swap_phases == _convert_to_py_instruction(rs_swap_phase)

    def test_pickle(self, swap_phases: SwapPhases):
        pickled = pickle.dumps(swap_phases)
        unpickled = pickle.loads(pickled)
        assert unpickled == swap_phases


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (MemoryReference("ro"), MemoryReference("bar")),
        (MemoryReference("foo", 5, 10), 5),
        (MemoryReference("bar"), 3.2),
    ],
)
class TestClassicalMove:
    @pytest.fixture
    def move(self, left: MemoryReference, right: Union[MemoryReference, int, float]) -> ClassicalMove:
        return ClassicalMove(left, right)

    def test_out(self, move: ClassicalMove, snapshot: SnapshotAssertion):
        assert move.out() == snapshot

    def test_left(self, move: ClassicalMove, left: MemoryReference):
        assert move.left == left
        move.left = MemoryReference("new-memory-reference")
        assert move.left == MemoryReference("new-memory-reference")

    def test_right(self, move: ClassicalMove, right: Union[MemoryReference, int, float]):
        assert move.right == right
        move.right = MemoryReference("new-memory-reference")
        assert move.right == MemoryReference("new-memory-reference")

    def test_copy(self, move: ClassicalMove):
        assert isinstance(copy.copy(move), ClassicalMove)
        assert isinstance(copy.deepcopy(move), ClassicalMove)

    def test_convert(self, move: ClassicalMove):
        rs_classical_move = _convert_to_rs_instruction(move)
        assert move == _convert_to_py_instruction(rs_classical_move)

    def test_pickle(self, move: ClassicalMove):
        pickled = pickle.dumps(move)
        unpickled = pickle.loads(pickled)
        assert unpickled == move


@pytest.mark.parametrize(
    ("left", "right"),
    [(MemoryReference("ro"), MemoryReference("bar")), (MemoryReference("foo", 5, 10), MemoryReference("bar"))],
)
class TestClassicalExchange:
    @pytest.fixture
    def exchange(self, left: MemoryReference, right: MemoryReference) -> ClassicalExchange:
        return ClassicalExchange(left, right)

    def test_out(self, exchange: ClassicalExchange, snapshot: SnapshotAssertion):
        assert exchange.out() == snapshot

    def test_left(self, exchange: ClassicalExchange, left: MemoryReference):
        assert exchange.left == left
        exchange.left = MemoryReference("new-memory-reference")
        assert exchange.left == MemoryReference("new-memory-reference")

    def test_right(self, exchange: ClassicalExchange, right: MemoryReference):
        assert exchange.right == right
        exchange.right = MemoryReference("new-memory-reference")
        assert exchange.right == MemoryReference("new-memory-reference")

    def test_copy(self, exchange: ClassicalExchange):
        assert isinstance(copy.copy(exchange), ClassicalExchange)
        assert isinstance(copy.deepcopy(exchange), ClassicalExchange)

    def test_convert(self, exchange: ClassicalExchange):
        rs_classical_exchange = _convert_to_rs_instruction(exchange)
        assert exchange == _convert_to_py_instruction(rs_classical_exchange)

    def test_pickle(self, exchange: ClassicalExchange):
        pickled = pickle.dumps(exchange)
        unpickled = pickle.loads(pickled)
        assert unpickled == exchange


@pytest.mark.parametrize(
    ("left", "right"),
    [(MemoryReference("ro"), MemoryReference("bar")), (MemoryReference("foo", 5, 10), MemoryReference("bar"))],
)
class TestClassicalConvert:
    @pytest.fixture
    def convert(self, left: MemoryReference, right: MemoryReference) -> ClassicalConvert:
        return ClassicalConvert(left, right)

    def test_out(self, convert: ClassicalConvert, snapshot: SnapshotAssertion):
        assert convert.out() == snapshot

    def test_left(self, convert: ClassicalConvert, left: MemoryReference):
        assert convert.left == left
        convert.left = MemoryReference("new-memory-reference")
        assert convert.left == MemoryReference("new-memory-reference")

    def test_right(self, convert: ClassicalConvert, right: MemoryReference):
        assert convert.right == right
        convert.right = MemoryReference("new-memory-reference")
        assert convert.right == MemoryReference("new-memory-reference")

    def test_copy(self, convert: ClassicalConvert):
        assert isinstance(copy.copy(convert), ClassicalConvert)
        assert isinstance(copy.deepcopy(convert), ClassicalConvert)

    def test_convert(self, convert: ClassicalConvert):
        rs_classical_convert = _convert_to_rs_instruction(convert)
        assert convert == _convert_to_py_instruction(rs_classical_convert)

    def test_pickle(self, convert: ClassicalConvert):
        pickled = pickle.dumps(convert)
        unpickled = pickle.loads(pickled)
        assert unpickled == convert


@pytest.mark.parametrize(
    ("target", "left", "right"),
    [(MemoryReference("t"), "y", MemoryReference("z")), (MemoryReference("t", 5, 10), "y", MemoryReference("bar", 1))],
)
class TestClassicalLoad:
    @pytest.fixture
    def load(self, target: MemoryReference, left: str, right: MemoryReference) -> ClassicalLoad:
        return ClassicalLoad(target, left, right)

    def test_out(self, load: ClassicalLoad, snapshot: SnapshotAssertion):
        assert load.out() == snapshot

    def test_target(self, load: ClassicalLoad, target: MemoryReference):
        assert load.target == target
        load.target = MemoryReference("new-memory-reference")
        assert load.target == MemoryReference("new-memory-reference")

    def test_left(self, load: ClassicalLoad, left: MemoryReference):
        assert load.left == left
        load.left = "new-left"
        assert load.left == "new-left"

    def test_right(self, load: ClassicalLoad, right: MemoryReference):
        assert load.right == right
        load.right = MemoryReference("new-memory-reference")
        assert load.right == MemoryReference("new-memory-reference")

    def test_copy(self, load: ClassicalLoad):
        assert isinstance(copy.copy(load), ClassicalLoad)
        assert isinstance(copy.deepcopy(load), ClassicalLoad)

    def test_convert(self, load: ClassicalLoad):
        rs_classical_load = _convert_to_rs_instruction(load)
        assert load == _convert_to_py_instruction(rs_classical_load)

    def test_pickle(self, load: ClassicalLoad):
        pickled = pickle.dumps(load)
        unpickled = pickle.loads(pickled)
        assert unpickled == load


@pytest.mark.parametrize(
    ("target", "left", "right"),
    [
        ("t", MemoryReference("y"), MemoryReference("z")),
        ("t", MemoryReference("y", 5, 10), 2),
        ("t", MemoryReference("y", 5, 10), 3.14),
    ],
)
class TestClassicalStore:
    @pytest.fixture
    def store(self, target: str, left: MemoryReference, right: MemoryReference) -> ClassicalStore:
        return ClassicalStore(target, left, right)

    def test_out(self, store: ClassicalStore, snapshot: SnapshotAssertion):
        assert store.out() == snapshot

    def test_target(self, store: ClassicalStore, target: str):
        assert store.target == target
        store.target = "new-target"
        assert store.target == "new-target"

    def test_left(self, store: ClassicalStore, left: MemoryReference):
        assert store.left == left
        store.left = MemoryReference("new-memory-reference")
        assert store.left == MemoryReference("new-memory-reference")

    def test_right(self, store: ClassicalStore, right: Union[MemoryReference, int, float]):
        assert store.right == right
        store.right = MemoryReference("new-memory-reference")
        assert store.right == MemoryReference("new-memory-reference")

    def test_copy(self, store: ClassicalStore):
        assert isinstance(copy.copy(store), ClassicalStore)
        assert isinstance(copy.deepcopy(store), ClassicalStore)

    def test_convert(self, store: ClassicalStore):
        rs_classical_store = _convert_to_rs_instruction(store)
        assert store == _convert_to_py_instruction(rs_classical_store)

    def test_pickle(self, store: ClassicalStore):
        pickled = pickle.dumps(store)
        unpickled = pickle.loads(pickled)
        assert unpickled == store


@pytest.mark.parametrize(
    ("op", "target", "left", "right"),
    [
        ("EQ", MemoryReference("t"), MemoryReference("y"), MemoryReference("z")),
        ("LT", MemoryReference("t"), MemoryReference("y", 5, 10), 2),
        ("LE", MemoryReference("t", 2, 4), MemoryReference("y"), 3.14),
        ("GT", MemoryReference("t", 2, 4), MemoryReference("y"), 3.14),
        ("GE", MemoryReference("t", 2, 4), MemoryReference("y"), 3.14),
    ],
)
class TestClassicalComparison:
    @pytest.fixture
    def comparison(
        self, op: str, target: MemoryReference, left: MemoryReference, right: Union[MemoryReference, int, float]
    ) -> ClassicalComparison:
        if op == "EQ":
            return ClassicalEqual(target, left, right)
        if op == "LT":
            return ClassicalLessThan(target, left, right)
        if op == "LE":
            return ClassicalLessEqual(target, left, right)
        if op == "GT":
            return ClassicalGreaterThan(target, left, right)
        return ClassicalGreaterEqual(target, left, right)

    def test_out(self, comparison: ClassicalComparison, snapshot: SnapshotAssertion):
        assert comparison.out() == snapshot

    def test_target(self, comparison: ClassicalComparison, target: MemoryReference):
        assert comparison.target == target
        comparison.target = MemoryReference("new-memory-reference")
        assert comparison.target == MemoryReference("new-memory-reference")

    def test_left(self, comparison: ClassicalComparison, left: MemoryReference):
        assert comparison.left == left
        comparison.left = MemoryReference("new-memory-reference")
        assert comparison.left == MemoryReference("new-memory-reference")

    def test_right(self, comparison: ClassicalComparison, right: Union[MemoryReference, int, float]):
        assert comparison.right == right
        comparison.right = MemoryReference("new-memory-reference")
        assert comparison.right == MemoryReference("new-memory-reference")

    def test_copy(self, comparison: ClassicalComparison):
        assert isinstance(copy.copy(comparison), type(comparison))
        assert isinstance(copy.deepcopy(comparison), type(comparison))

    def test_convert(self, comparison: ClassicalComparison):
        rs_classical_comparison = _convert_to_rs_instruction(comparison)
        assert comparison == _convert_to_py_instruction(rs_classical_comparison)

    def test_pickle(self, comparison: ClassicalComparison):
        pickled = pickle.dumps(comparison)
        unpickled = pickle.loads(pickled)
        assert unpickled == comparison


@pytest.mark.parametrize(
    ("op", "target"),
    [
        ("NEG", MemoryReference("a")),
        ("NOT", MemoryReference("b", 1)),
        ("NEG", MemoryReference("c", 2, 4)),
    ],
)
class TestUnaryClassicalInstruction:
    @pytest.fixture
    def unary(self, op: str, target: MemoryReference) -> UnaryClassicalInstruction:
        if op == "NEG":
            return ClassicalNeg(target)
        return ClassicalNot(target)

    def test_out(self, unary: UnaryClassicalInstruction, snapshot: SnapshotAssertion):
        assert unary.out() == snapshot

    def test_target(self, unary: UnaryClassicalInstruction, target: MemoryReference):
        assert unary.target == target
        unary.target = MemoryReference("new-memory-reference")
        assert unary.target == MemoryReference("new-memory-reference")

    def test_copy(self, unary: UnaryClassicalInstruction):
        assert isinstance(copy.copy(unary), type(unary))
        assert isinstance(copy.deepcopy(unary), type(unary))

    def test_convert(self, unary: UnaryClassicalInstruction):
        rs_classical_unary = _convert_to_rs_instruction(unary)
        assert unary == _convert_to_py_instruction(rs_classical_unary)

    def test_pickle(self, unary: UnaryClassicalInstruction):
        pickled = pickle.dumps(unary)
        unpickled = pickle.loads(pickled)
        assert unpickled == unary


@pytest.mark.parametrize(
    ("op", "left", "right"),
    [
        ("ADD", MemoryReference("a"), MemoryReference("b")),
        ("SUB", MemoryReference("b", 1), 1),
        ("MUL", MemoryReference("c", 2, 4), 1.0),
        ("DIV", MemoryReference("c", 2, 4), 4.2),
    ],
)
class TestArithmeticBinaryOp:
    @pytest.fixture
    def arithmetic(
        self, op: str, left: MemoryReference, right: Union[MemoryReference, int, float]
    ) -> ArithmeticBinaryOp:
        if op == "ADD":
            return ClassicalAdd(left, right)
        if op == "SUB":
            return ClassicalSub(left, right)
        if op == "MUL":
            return ClassicalMul(left, right)
        return ClassicalDiv(left, right)

    def test_out(self, arithmetic: ArithmeticBinaryOp, snapshot: SnapshotAssertion):
        assert arithmetic.out() == snapshot

    def test_left(self, arithmetic: ArithmeticBinaryOp, left: MemoryReference):
        assert arithmetic.left == left
        arithmetic.left = MemoryReference("new-memory-reference")
        assert arithmetic.left == MemoryReference("new-memory-reference")

    def test_right(self, arithmetic: ArithmeticBinaryOp, right: Union[MemoryReference, float, int]):
        assert arithmetic.right == right
        arithmetic.right = 3.14
        assert arithmetic.right == 3.14

    def test_copy(self, arithmetic: ArithmeticBinaryOp):
        assert isinstance(copy.copy(arithmetic), type(arithmetic))
        assert isinstance(copy.deepcopy(arithmetic), type(arithmetic))

    def valid_in_program(self, arithmetic):
        try:
            p = Program(arithmetic)
            p[0] == arithmetic
        except Exception:
            pytest.fail("ArithmeticBinaryOp not valid in Program")

    def test_convert(self, arithmetic: UnaryClassicalInstruction):
        rs_classical_arithmetic = _convert_to_rs_instruction(arithmetic)
        assert arithmetic == _convert_to_py_instruction(rs_classical_arithmetic)

    def test_pickle(self, arithmetic: UnaryClassicalInstruction, snapshot: SnapshotAssertion):
        pickled = pickle.dumps(arithmetic)
        unpickled = pickle.loads(pickled)
        assert unpickled == snapshot


@pytest.mark.parametrize(
    ("op", "left", "right"),
    [
        ("AND", MemoryReference("a"), MemoryReference("b")),
        ("IOR", MemoryReference("b", 1), 1),
        ("XOR", MemoryReference("c", 2, 4), 2),
    ],
)
class TestLogicalBinaryOp:
    @pytest.fixture
    def logical(self, op: str, left: MemoryReference, right: Union[MemoryReference, int]) -> LogicalBinaryOp:
        if op == "AND":
            return ClassicalAnd(left, right)
        if op == "IOR":
            return ClassicalInclusiveOr(left, right)
        return ClassicalExclusiveOr(left, right)

    def test_out(self, logical: LogicalBinaryOp, snapshot: SnapshotAssertion):
        assert logical.out() == snapshot

    def test_left(self, logical: LogicalBinaryOp, left: MemoryReference):
        assert logical.left == left
        logical.left = MemoryReference("new-memory-reference")
        assert logical.left == MemoryReference("new-memory-reference")

    def test_right(self, logical: LogicalBinaryOp, right: Union[MemoryReference, float, int]):
        assert logical.right == right
        logical.right = 3
        assert logical.right == 3

    def test_copy(self, logical: LogicalBinaryOp):
        assert isinstance(copy.copy(logical), type(logical))
        assert isinstance(copy.deepcopy(logical), type(logical))

    def test_convert(self, logical: LogicalBinaryOp):
        rs_classical_logical = _convert_to_rs_instruction(logical)
        assert logical == _convert_to_py_instruction(rs_classical_logical)

    def test_pickle(self, logical: LogicalBinaryOp):
        pickled = pickle.dumps(logical)
        unpickled = pickle.loads(pickled)
        assert unpickled == logical


def test_include():
    include = Include("my-cool-file.quil")
    assert include.out() == 'INCLUDE "my-cool-file.quil"'
    include.filename = "my-other-file.quil"
    assert include.filename == "my-other-file.quil"
    rs_include = _convert_to_rs_instruction(include)
    assert include == _convert_to_py_instruction(rs_include)


def test_wait():
    assert Wait().out() == str(Wait()) == "WAIT"
    rs_wait = _convert_to_rs_instruction(Wait())
    assert Wait() == _convert_to_py_instruction(rs_wait)


def test_halt():
    assert Halt().out() == str(Halt()) == "HALT"
    rs_halt = _convert_to_rs_instruction(Halt())
    assert Halt() == _convert_to_py_instruction(rs_halt)


def test_nop():
    assert Nop().out() == str(Nop()) == "NOP"
    rs_nop = _convert_to_rs_instruction(Nop())
    assert Nop() == _convert_to_py_instruction(rs_nop)
