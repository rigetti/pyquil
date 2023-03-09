from math import pi
from typing import List
from pyquil.quil import Program
from pyquil.quilbase import Gate, MemoryReference, ParameterDesignator
from pyquil.quilatom import BinaryExp, Qubit
from pyquil.api._compiler import QPUCompiler
import pytest


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
    def gate(self, name, params, qubits) -> Gate:
        return Gate(name, params, qubits)

    @pytest.fixture
    def program(self, params, gate) -> Program:
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

    def test_str(self, gate, snapshot):
        assert str(gate) == snapshot

    def test_name(self, gate, name):
        assert gate.name == name

    def test_params(self, gate, params):
        assert gate.params == params
        gate.params = [pi / 2]
        assert gate.params == [pi / 2]

    def test_qubits(self, gate, qubits):
        assert gate.qubits == qubits
        gate.qubits = [Qubit(123)]
        assert gate.qubits == [Qubit(123)]

    def test_get_qubits(self, gate, qubits):
        assert gate.get_qubit_indices() == {q.index for q in qubits}
        assert gate.get_qubits(indices=False) == set(qubits)

    def test_controlled_modifier(self, gate, snapshot):
        assert str(gate.controlled([Qubit(5)])) == snapshot

    def test_dagger_modifier(self, gate, snapshot):
        assert str(gate.dagger()) == snapshot

    def test_forked_modifier(self, gate, params, snapshot):
        alt_params: List[ParameterDesignator] = [n for n in range(len(params))]
        assert str(gate.forked(Qubit(5), alt_params)) == snapshot

    def test_repr(self, gate, snapshot):
        assert repr(gate) == snapshot

    def test_compile(self, program, compiler: QPUCompiler):
        try:
            compiler.quil_to_native_quil(program)
        except Exception as e:
            assert False, f"Failed to compile the program: {e}\n{program}"
