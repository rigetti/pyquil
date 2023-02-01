from math import pi
from typing import List
from pyquil.quilbase import Gate, MemoryReference, ParameterDesignator
from pyquil.quilatom import Qubit
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
    def test_str(self, name, params, qubits, snapshot):
        gate = Gate(name, params, qubits)
        assert str(gate) == snapshot

    def test_repr(self, name, params, qubits, snapshot):
        gate = Gate(name, params, qubits)
        assert repr(gate) == snapshot

    def test_controlled_modifier(self, name, params, qubits, snapshot):
        gate = Gate(name, params, qubits)
        assert str(gate.controlled([Qubit(5)])) == snapshot

    def test_dagger_modifier(self, name, params, qubits, snapshot):
        gate = Gate(name, params, qubits)
        assert str(gate.dagger()) == snapshot

    def test_forked_modifier(self, name, params, qubits, snapshot):
        gate = Gate(name, params, qubits)
        alt_params: List[ParameterDesignator] = [n for n in range(len(params))]
        assert str(gate.forked(Qubit(5), alt_params)) == snapshot

    def test_get_qubits(self, name, params, qubits):
        gate = Gate(name, params, qubits)
        assert gate.get_qubits(indices=True) == {q.index for q in qubits}
        assert gate.get_qubits(indices=False) == set(qubits)
