"""Unit tests for the quax-based state vector simulator."""

import jax.numpy as jnp
import numpy as np
import pytest
import quax as qx

from pyquil.gates import CNOT, RX, RY, RZ, H, X
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, Qubit
from pyquil.quilbase import (
    Declare,
    DefGate,
)
from pyquil.quilbase import (
    Gate as QuilGate,
)
from pyquil.simulation._simulator import PureStateVectorSimulator

_EMPTY_PARAMS = jnp.array([], dtype=float)


def _sv(program, qubits=None, memory_map=None):
    """Compute pure state vector for a gate-only program."""
    sim = PureStateVectorSimulator(program, qubits=qubits)
    if memory_map:
        params = sim.linearize(memory_map)
    else:
        params = _EMPTY_PARAMS
    return sim.compute(params)


class TestSingleQubitGates:
    def test_identity(self):
        p = Program()
        psi = _sv(p, qubits=[0])
        target = qx.StateVector.from_matrix(jnp.array([1.0, 0.0], dtype=complex), dims=(2,))
        assert qx.fidelity(psi, target) > 0.9999

    def test_x_gate(self):
        p = Program(X(0))
        psi = _sv(p, qubits=[0])
        target = qx.StateVector.from_matrix(jnp.array([0.0, 1.0], dtype=complex), dims=(2,))
        assert qx.fidelity(psi, target) > 0.9999

    def test_hadamard(self):
        p = Program(H(0))
        psi = _sv(p, qubits=[0])
        target = qx.StateVector.from_matrix(jnp.array([1.0, 1.0], dtype=complex) / jnp.sqrt(2), dims=(2,))
        assert qx.fidelity(psi, target) > 0.9999

    @pytest.mark.parametrize("angle", [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2])
    def test_rx_gate(self, angle):
        p = Program(RX(angle, 0))
        psi = _sv(p, qubits=[0])
        expected = jnp.asarray(qx.gates.RX(angle).matrix) @ jnp.array([1.0, 0.0], dtype=complex)
        target = qx.StateVector.from_matrix(expected, dims=(2,))
        assert qx.fidelity(psi, target) > 0.9999

    @pytest.mark.parametrize("angle", [0.0, np.pi / 4, np.pi / 2, np.pi])
    def test_ry_gate(self, angle):
        p = Program(RY(angle, 0))
        psi = _sv(p, qubits=[0])
        expected = jnp.asarray(qx.gates.RY(angle).matrix) @ jnp.array([1.0, 0.0], dtype=complex)
        target = qx.StateVector.from_matrix(expected, dims=(2,))
        assert qx.fidelity(psi, target) > 0.9999

    @pytest.mark.parametrize("angle", [0.0, np.pi / 4, np.pi / 2, np.pi])
    def test_rz_gate(self, angle):
        p = Program(RZ(angle, 0))
        psi = _sv(p, qubits=[0])
        expected = jnp.asarray(qx.gates.RZ(angle).matrix) @ jnp.array([1.0, 0.0], dtype=complex)
        target = qx.StateVector.from_matrix(expected, dims=(2,))
        assert qx.fidelity(psi, target) > 0.9999


class TestMultiQubitGates:
    def test_bell_state(self):
        p = Program(H(0), CNOT(0, 1))
        psi = _sv(p, qubits=[0, 1])
        target = qx.StateVector.from_matrix(jnp.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / jnp.sqrt(2), dims=(2, 2))
        assert qx.fidelity(psi, target) > 0.9999

    def test_ghz_state_3q(self):
        p = Program(H(0), CNOT(0, 1), CNOT(1, 2))
        psi = _sv(p, qubits=[0, 1, 2])
        target = qx.StateVector.from_matrix(
            jnp.array([1.0, 0, 0, 0, 0, 0, 0, 1.0], dtype=complex) / jnp.sqrt(2), dims=(2, 2, 2)
        )
        assert qx.fidelity(psi, target) > 0.9999

    def test_qubit_ordering(self):
        """State vector should respect the provided qubit ordering."""
        p = Program(X(5))
        psi = _sv(p, qubits=[5, 6])
        # qubit 5 is index 0, qubit 6 is index 1
        # X on qubit 5 → |10> → state [0, 0, 1, 0]
        target = qx.StateVector.from_matrix(jnp.array([0.0, 0.0, 1.0, 0.0], dtype=complex), dims=(2, 2))
        assert qx.fidelity(psi, target) > 0.9999


class TestParameterizedPrograms:
    def test_parameterized_rx(self):
        p = Program(
            Declare("theta", "REAL"),
            RX(MemoryReference("theta"), 0),
        )
        angle = np.pi / 3
        psi = _sv(p, qubits=[0], memory_map={"theta": [angle]})
        expected = jnp.asarray(qx.gates.RX(angle).matrix) @ jnp.array([1.0, 0.0], dtype=complex)
        target = qx.StateVector.from_matrix(expected, dims=(2,))
        assert qx.fidelity(psi, target) > 0.9999


class TestCustomGates:
    def test_defgate(self):
        """Test that DefGate-defined gates work correctly."""
        cnot_matrix = np.asarray(qx.gates.CNOT.matrix)
        p = Program()
        p += DefGate("MY_CNOT", cnot_matrix)
        p += QuilGate("MY_CNOT", [], [Qubit(0), Qubit(1)])
        # Prepare |1,0> first
        p2 = Program(X(0)) + p
        psi = _sv(p2, qubits=[0, 1])
        # X(0) gives |10>, then CNOT gives |11>
        target = qx.StateVector.from_matrix(jnp.array([0.0, 0.0, 0.0, 1.0], dtype=complex), dims=(2, 2))
        assert qx.fidelity(psi, target) > 0.9999


class TestAutoQubitDetection:
    def test_auto_qubits(self):
        """When qubits=None, should auto-detect from program."""
        p = Program(H(2), CNOT(2, 5))
        psi = _sv(p)
        # Should use qubits [2, 5] in sorted order
        target = qx.StateVector.from_matrix(jnp.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / jnp.sqrt(2), dims=(2, 2))
        assert qx.fidelity(psi, target) > 0.9999
