from math import pi
from typing import List, Optional

import pytest
from syrupy.assertion import SnapshotAssertion

from pyquil.gates import X
from pyquil.quil import Program
from pyquil.quilbase import (
    AbstractInstruction,
    DefCalibration,
    DefFrame,
    DefMeasureCalibration,
    Gate,
    Measurement,
    MemoryReference,
    ParameterDesignator,
    Parameter,
)
from pyquil.quilatom import BinaryExp, Frame, Qubit
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
