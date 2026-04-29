"""Unit tests for the quax-based state vector simulator."""

from functools import reduce

import jax
import numpy as np
import pytest
import quax as qx

from pyquil.gates import CNOT, H, RX, RY, RZ, X
from pyquil.quil import Program
from pyquil.quilbase import DefGate
from pyquil.simulation.state_vector import compute_program_state_vector

jax.config.update("jax_enable_x64", True)


def _fidelity(psi, target_vector):
    """Fidelity |<target|psi>|^2 for pure states."""
    psi_np = np.asarray(psi.data).flatten()
    return float(np.abs(np.vdot(target_vector, psi_np)) ** 2)


class TestSingleQubitGates:
    def test_identity(self):
        p = Program()
        psi = compute_program_state_vector(p, qubits=[0])
        assert _fidelity(psi, [1, 0]) == pytest.approx(1.0)

    def test_x_gate(self):
        p = Program(X(0))
        psi = compute_program_state_vector(p, qubits=[0])
        assert _fidelity(psi, [0, 1]) == pytest.approx(1.0)

    def test_hadamard(self):
        p = Program(H(0))
        psi = compute_program_state_vector(p, qubits=[0])
        expected = np.array([1, 1]) / np.sqrt(2)
        assert _fidelity(psi, expected) == pytest.approx(1.0)

    @pytest.mark.parametrize("angle", [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2])
    def test_rx_gate(self, angle):
        p = Program(RX(angle, 0))
        psi = compute_program_state_vector(p, qubits=[0])
        expected = np.asarray(qx.gates.RX(angle).matrix) @ np.array([1, 0])
        assert _fidelity(psi, expected) == pytest.approx(1.0)

    @pytest.mark.parametrize("angle", [0.0, np.pi / 4, np.pi / 2, np.pi])
    def test_ry_gate(self, angle):
        p = Program(RY(angle, 0))
        psi = compute_program_state_vector(p, qubits=[0])
        expected = np.asarray(qx.gates.RY(angle).matrix) @ np.array([1, 0])
        assert _fidelity(psi, expected) == pytest.approx(1.0)

    @pytest.mark.parametrize("angle", [0.0, np.pi / 4, np.pi / 2, np.pi])
    def test_rz_gate(self, angle):
        p = Program(RZ(angle, 0))
        psi = compute_program_state_vector(p, qubits=[0])
        expected = np.asarray(qx.gates.RZ(angle).matrix) @ np.array([1, 0])
        assert _fidelity(psi, expected) == pytest.approx(1.0)


class TestMultiQubitGates:
    def test_bell_state(self):
        p = Program(H(0), CNOT(0, 1))
        psi = compute_program_state_vector(p, qubits=[0, 1])
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        assert _fidelity(psi, expected) == pytest.approx(1.0)

    def test_ghz_state_3q(self):
        p = Program(H(0), CNOT(0, 1), CNOT(1, 2))
        psi = compute_program_state_vector(p, qubits=[0, 1, 2])
        expected = np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)
        assert _fidelity(psi, expected) == pytest.approx(1.0)

    def test_qubit_ordering(self):
        """State vector should respect the provided qubit ordering."""
        p = Program(X(5))
        psi = compute_program_state_vector(p, qubits=[5, 6])
        # qubit 5 is index 0, qubit 6 is index 1
        # X on qubit 5 → |10> → state [0, 0, 1, 0]
        expected = np.array([0, 0, 1, 0])
        assert _fidelity(psi, expected) == pytest.approx(1.0)


class TestParameterizedPrograms:
    def test_parameterized_rx(self):
        from pyquil.quilatom import MemoryReference
        from pyquil.quilbase import Declare

        p = Program(
            Declare("theta", "REAL"),
            RX(MemoryReference("theta"), 0),
        )
        angle = np.pi / 3
        psi = compute_program_state_vector(p, qubits=[0], memory_map={"theta": [angle]})
        expected = np.asarray(qx.gates.RX(angle).matrix) @ np.array([1, 0])
        assert _fidelity(psi, expected) == pytest.approx(1.0)


class TestCustomGates:
    def test_defgate(self):
        """Test that DefGate-defined gates work correctly."""
        cnot_matrix = np.asarray(qx.gates.CNOT.matrix)
        p = Program()
        p += DefGate("MY_CNOT", cnot_matrix)
        from pyquil.quilbase import Gate as QuilGate
        from pyquil.quilatom import Qubit

        p += QuilGate("MY_CNOT", [], [Qubit(0), Qubit(1)])
        # Prepare |1,0> first
        p2 = Program(X(0)) + p
        psi = compute_program_state_vector(p2, qubits=[0, 1])
        # X(0) gives |10>, then CNOT gives |11>
        expected = np.array([0, 0, 0, 1])
        assert _fidelity(psi, expected) == pytest.approx(1.0)


class TestAutoQubitDetection:
    def test_auto_qubits(self):
        """When qubits=None, should auto-detect from program."""
        p = Program(H(2), CNOT(2, 5))
        psi = compute_program_state_vector(p)
        # Should use qubits [2, 5] in sorted order
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        assert _fidelity(psi, expected) == pytest.approx(1.0)
