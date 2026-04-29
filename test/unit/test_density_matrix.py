# Copyright 2024-2026 Rigetti Computing
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

"""Unit tests for the quax-based density matrix simulator."""

from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import quax as qx

from pyquil.noise import Channel, NoiseModel
from pyquil.simulation.density_matrix import compute_program_density_matrix

from pyquil.gates import CCNOT, CNOT, RX, RY, RZ
from pyquil.quil import Program
from pyquil.quilbase import DefGate, Gate


# Gate matrix namespace (replacing deleted pyquil.simulation.matrices)
def _mat(gate):
    return np.asarray(gate.matrix)


class matrices:
    I = _mat(qx.gates.I)
    CNOT = _mat(qx.gates.CNOT)
    CCNOT = _mat(qx.gates.CCNOT)

    @staticmethod
    def RX(angle):
        return _mat(qx.gates.RX(angle))

    @staticmethod
    def RY(angle):
        return _mat(qx.gates.RY(angle))

    @staticmethod
    def RZ(angle):
        return _mat(qx.gates.RZ(angle))

jax.config.update("jax_enable_x64", True)


def _pure_state_fidelity(rho, target_rho):
    """Fidelity for pure target states: F = Tr(rho * |psi><psi|)."""
    rho_np = np.asarray(rho.matrix)
    return float(np.real(np.trace(rho_np @ target_rho)))


def _mixed_state_fidelity(rho_np, target_np):
    """Full Jozsa fidelity for mixed states."""
    w, v = np.linalg.eigh(rho_np)
    w = np.maximum(w, 0)
    sqrt_rho = v @ np.diag(np.sqrt(w)) @ v.conj().T
    M = sqrt_rho @ target_np @ sqrt_rho
    eigvals = np.maximum(np.linalg.eigvalsh(M), 0)
    return float(np.real(np.sum(np.sqrt(eigvals))) ** 2)


def _zero_state_matrix(n_qubits):
    """Construct |0...0><0...0| as a numpy matrix."""
    d = 2**n_qubits
    rho = np.zeros((d, d), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


# ──────────────────────────────────────────────────────────
# Basic unitary tests
# ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_rx(seed):
    """Test that the simulator can simulate a single RX gate."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi)
    program = Program(RX(angle, 0))
    rho = compute_program_density_matrix(program)
    target_rho = matrices.RX(angle) @ _zero_state_matrix(1) @ matrices.RX(angle).conj().T
    assert _pure_state_fidelity(rho, target_rho) > 0.9999


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_tensor_product_state(seed):
    """Test simulation of a 1Q tensor product state."""
    rng = np.random.default_rng(seed)
    num_qubits = 4
    angles = rng.uniform(-np.pi, np.pi, num_qubits)
    program = Program([RX(angle, idx) for idx, angle in enumerate(angles)])
    rho = compute_program_density_matrix(program)
    u = reduce(np.kron, [matrices.RX(angle) for angle in angles])
    target_rho = u @ _zero_state_matrix(num_qubits) @ u.conj().T
    assert _pure_state_fidelity(rho, target_rho) > 0.9999


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_tensor_product_state_layers(seed):
    """Test simulation of a 1Q tensor product state with multiple gates per qubit."""
    rng = np.random.default_rng(seed)
    num_qubits = 4
    angles_0 = rng.uniform(-np.pi, np.pi, num_qubits)
    angles_1 = rng.uniform(-np.pi, np.pi, num_qubits)
    angles_2 = rng.uniform(-np.pi, np.pi, num_qubits)
    program = Program()
    program += [RX(angle, idx) for idx, angle in enumerate(angles_0)]
    program += [RY(angle, idx) for idx, angle in enumerate(angles_1)]
    program += [RZ(angle, idx) for idx, angle in enumerate(angles_2)]
    rho = compute_program_density_matrix(program)
    u = (
        reduce(np.kron, [matrices.RZ(angle) for angle in angles_2])
        @ reduce(np.kron, [matrices.RY(angle) for angle in angles_1])
        @ reduce(np.kron, [matrices.RX(angle) for angle in angles_0])
    )
    target_rho = u @ _zero_state_matrix(num_qubits) @ u.conj().T
    assert _pure_state_fidelity(rho, target_rho) > 0.9999


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_entanglement(seed):
    """Test a 2-qubit state with entanglement (detects wrong-endianness)."""
    rng = np.random.default_rng(seed)
    theta, phi = rng.uniform(-np.pi, np.pi, 2)
    program = Program()
    program += RX(theta, 0)
    program += RX(phi, 1)
    program += CNOT(0, 1)
    rho = compute_program_density_matrix(program)
    u = matrices.CNOT @ np.kron(matrices.RX(theta), matrices.RX(phi))
    target_rho = u @ _zero_state_matrix(2) @ u.conj().T
    assert _pure_state_fidelity(rho, target_rho) > 0.9999


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_multi_qubit_entanglement(seed):
    """Test a 3-qubit state with entanglement (detects wrong-endianness)."""
    rng = np.random.default_rng(seed)
    theta, phi, lam = rng.uniform(-np.pi, np.pi, 3)
    program = Program()
    program += RX(theta, 0)
    program += RY(phi, 1)
    program += RZ(lam, 2)
    program += CNOT(0, 1)
    program += CNOT(1, 2)
    rho = compute_program_density_matrix(program)
    u = (
        np.kron(matrices.I, matrices.CNOT)
        @ np.kron(matrices.CNOT, matrices.I)
        @ reduce(np.kron, [matrices.RX(theta), matrices.RY(phi), matrices.RZ(lam)])
    )
    target_rho = u @ _zero_state_matrix(3) @ u.conj().T
    assert _pure_state_fidelity(rho, target_rho) > 0.9999


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_multi_qubit_gates(seed):
    """Test a 4-qubit state with 3-qubit gates."""
    rng = np.random.default_rng(seed)
    theta, phi, lam, gamma = rng.uniform(-np.pi, np.pi, 4)
    program = Program()
    program += RX(theta, 0)
    program += RY(phi, 1)
    program += RZ(lam, 2)
    program += RX(gamma, 3)
    program += CNOT(0, 1)
    program += CNOT(2, 3)
    program += RX(theta, 0)
    program += RY(phi, 1)
    program += RZ(lam, 2)
    program += RX(gamma, 3)
    program += CCNOT(0, 1, 2)
    program += CCNOT(1, 2, 3)
    rho = compute_program_density_matrix(program)
    u = (
        np.kron(matrices.I, matrices.CCNOT)
        @ np.kron(matrices.CCNOT, matrices.I)
        @ reduce(np.kron, [matrices.RX(theta), matrices.RY(phi), matrices.RZ(lam), matrices.RX(gamma)])
        @ np.kron(matrices.CNOT, matrices.CNOT)
        @ reduce(np.kron, [matrices.RX(theta), matrices.RY(phi), matrices.RZ(lam), matrices.RX(gamma)])
    )
    target_rho = u @ _zero_state_matrix(4) @ u.conj().T
    assert _pure_state_fidelity(rho, target_rho) > 0.9999


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_disjoint_pairs(seed):
    """Test a 4-qubit state with entanglement only between pairs (detects splitting issues)."""
    rng = np.random.default_rng(seed)
    theta, phi, lam, gamma = rng.uniform(-np.pi, np.pi, 4)
    q0, q1, q2, q3 = [int(q) for q in rng.choice(list(range(12)), size=4, replace=False)]
    program = Program()
    program += RX(theta, q0)
    program += RY(phi, q1)
    program += RZ(lam, q2)
    program += RX(gamma, q3)
    program += CNOT(q0, q1)
    program += CNOT(q2, q3)
    rho = compute_program_density_matrix(program, qubits=[q0, q1, q2, q3])
    u = np.kron(matrices.CNOT, matrices.CNOT) @ reduce(
        np.kron, [matrices.RX(theta), matrices.RY(phi), matrices.RZ(lam), matrices.RX(gamma)]
    )
    target_rho = u @ _zero_state_matrix(4) @ u.conj().T
    assert _pure_state_fidelity(rho, target_rho) > 0.9999


def test_defgate():
    """Test a 2-qubit state with a DefGate."""
    rng = np.random.default_rng(5973)
    theta, phi = rng.uniform(-np.pi, np.pi, 2)
    program = Program()
    program += DefGate(name="BLARG", matrix=matrices.CNOT, parameters=[])
    program += RX(theta, 0)
    program += RX(phi, 1)
    program += Gate("BLARG", [], (0, 1))
    rho = compute_program_density_matrix(program)
    u = matrices.CNOT @ np.kron(matrices.RX(theta), matrices.RX(phi))
    target_rho = u @ _zero_state_matrix(2) @ u.conj().T
    assert _pure_state_fidelity(rho, target_rho) > 0.9999


# ──────────────────────────────────────────────────────────
# Noise tests
# ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_1q_depolarizing_noise(seed):
    """Test a 1Q gate with depolarizing noise."""
    rng = np.random.default_rng(seed)
    p = float(np.clip(rng.normal(loc=0.98, scale=0.01), 0.97, 0.99))
    num_qubits = 1
    dim = 2**num_qubits
    angle = rng.uniform(-np.pi, np.pi)
    inst = RX(angle, 0)
    noise_model = NoiseModel(
        channels=frozenset([Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=p)])
    )
    program = Program(inst)
    rho = compute_program_density_matrix(program, noise_model=noise_model)
    rho_np = np.asarray(rho.matrix)

    target_rho = matrices.RX(angle) @ _zero_state_matrix(num_qubits) @ matrices.RX(angle).conj().T
    target_rho = p * target_rho + ((1 - p) / dim) * np.eye(dim)
    assert _mixed_state_fidelity(rho_np, target_rho) > 0.9999


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_multi_qubit_depolarizing_noise_1q(seed):
    """Test depolarizing noise on a 3-qubit state with entanglement (1Q channels only)."""
    rng = np.random.default_rng(seed)
    p = [float(np.clip(rng.normal(loc=0.98, scale=0.01), 0.97, 0.99)) for _ in range(3)]
    theta, phi, lam = rng.uniform(-np.pi, np.pi, 3)
    insts = [RX(theta, 0), RY(phi, 1), RZ(lam, 2)]
    noise_model = NoiseModel(
        channels=frozenset(
            Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=pi) for inst, pi in zip(insts, p)
        )
    )
    program = Program()
    program += RX(theta, 0)
    program += RY(phi, 1)
    program += RZ(lam, 2)
    program += CNOT(0, 1)
    program += CNOT(1, 2)
    rho = compute_program_density_matrix(program, noise_model=noise_model)
    rho_np = np.asarray(rho.matrix)

    # Verify basic properties: trace 1, hermitian, positive semi-definite
    assert np.isclose(np.trace(rho_np), 1.0, atol=1e-10)
    assert np.allclose(rho_np, rho_np.conj().T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(rho_np)
    assert np.all(eigvals > -1e-10)


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_multi_qubit_depolarizing_noise(seed):
    """Test depolarizing noise on a 3-qubit state with entanglement (1Q + 2Q channels)."""
    rng = np.random.default_rng(seed)
    p = [float(np.clip(rng.normal(loc=0.98, scale=0.01), 0.97, 0.99)) for _ in range(5)]
    theta, phi, lam = rng.uniform(-np.pi, np.pi, 3)
    insts = [RX(theta, 0), RY(phi, 1), RZ(lam, 2), CNOT(0, 1), CNOT(1, 2)]
    noise_model = NoiseModel(
        channels=frozenset(
            Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=pi) for inst, pi in zip(insts, p)
        )
    )
    program = Program()
    program += RX(theta, 0)
    program += RY(phi, 1)
    program += RZ(lam, 2)
    program += CNOT(0, 1)
    program += CNOT(1, 2)
    rho = compute_program_density_matrix(program, noise_model=noise_model)
    rho_np = np.asarray(rho.matrix)

    # Verify basic properties
    assert np.isclose(np.trace(rho_np), 1.0, atol=1e-10)
    assert np.allclose(rho_np, rho_np.conj().T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(rho_np)
    assert np.all(eigvals > -1e-10)

    # Noisy state should be less pure than noiseless
    noiseless_rho = np.asarray(compute_program_density_matrix(program).matrix)
    noisy_purity = np.real(np.trace(rho_np @ rho_np))
    noiseless_purity = np.real(np.trace(noiseless_rho @ noiseless_rho))
    assert noisy_purity < noiseless_purity


# ──────────────────────────────────────────────────────────
# Non-zero qubit index tests
# ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_rx_nonzero_index(seed):
    """Test that the simulator can handle a single RX gate with a non-zero index."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi)
    program = Program(RX(angle, 6))
    rho = compute_program_density_matrix(program)
    target_rho = matrices.RX(angle) @ _zero_state_matrix(1) @ matrices.RX(angle).conj().T
    assert _pure_state_fidelity(rho, target_rho) > 0.9999
