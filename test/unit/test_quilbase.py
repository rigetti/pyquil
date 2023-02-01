from math import pi
from pyquil.quilbase import Gate
from pyquil.quilatom import Qubit
import pytest


@pytest.mark.parametrize(
    ("name", "params", "qubits"),
    [
        ("X", [], [Qubit(0)]),
        ("CPHASE", [pi / 2], [Qubit(0), Qubit(1)]),
    ],
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

    def test_get_qubits(self, name, params, qubits):
        gate = Gate(name, params, qubits)
        if all(isinstance(q, Qubit) for q in qubits):
            assert gate.get_qubits(indices=True) == {q.index for q in qubits}
        assert gate.get_qubits(indices=False) == set(qubits)
