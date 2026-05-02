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
from operator import or_
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


def _to_unitary(op):
    """Coerce an Operator (e.g. from RZ which uses scalar multiplication) back to a Unitary."""
    return qx.Unitary.from_matrix(op.matrix, op.dims)


def _mixed_state_fidelity(rho_np, target_np):
    """Full Jozsa fidelity for mixed states."""
    w, v = np.linalg.eigh(rho_np)
    w = np.maximum(w, 0)
    sqrt_rho = v @ np.diag(np.sqrt(w)) @ v.conj().T
    M = sqrt_rho @ target_np @ sqrt_rho
    eigvals = np.maximum(np.linalg.eigvalsh(M), 0)
    return float(np.real(np.sum(np.sqrt(eigvals))) ** 2)


# ──────────────────────────────────────────────────────────
# Basic unitary tests
# ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_rx(seed):
    """Test that the simulator can simulate a single RX gate."""
    key = jax.random.key(seed)
    angle = float(jax.random.uniform(key, minval=-jnp.pi, maxval=jnp.pi))
    program = Program(RX(angle, 0))
    rho = compute_program_density_matrix(program)
    target_rho = qx.gates.RX(angle) @ qx.zero_state_matrix(1)
    assert qx.fidelity(rho, target_rho) > 0.9999


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_tensor_product_state(seed):
    """Test simulation of a 1Q tensor product state."""
    key = jax.random.key(seed)
    num_qubits = 4
    angles = jax.random.uniform(key, shape=(num_qubits,), minval=-jnp.pi, maxval=jnp.pi)
    program = Program([RX(float(angle), idx) for idx, angle in enumerate(angles)])
    rho = compute_program_density_matrix(program)
    u = reduce(or_, [qx.gates.RX(angle) for angle in angles])
    target_rho = u @ qx.zero_state_matrix(num_qubits)
    assert qx.fidelity(rho, target_rho) > 0.9999


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_tensor_product_state_layers(seed):
    """Test simulation of a 1Q tensor product state with multiple gates per qubit."""
    key = jax.random.key(seed)
    num_qubits = 4
    k0, k1, k2 = jax.random.split(key, 3)
    angles_0 = jax.random.uniform(k0, shape=(num_qubits,), minval=-jnp.pi, maxval=jnp.pi)
    angles_1 = jax.random.uniform(k1, shape=(num_qubits,), minval=-jnp.pi, maxval=jnp.pi)
    angles_2 = jax.random.uniform(k2, shape=(num_qubits,), minval=-jnp.pi, maxval=jnp.pi)
    program = Program()
    program += [RX(float(angle), idx) for idx, angle in enumerate(angles_0)]
    program += [RY(float(angle), idx) for idx, angle in enumerate(angles_1)]
    program += [RZ(float(angle), idx) for idx, angle in enumerate(angles_2)]
    rho = compute_program_density_matrix(program)
    u = _to_unitary(
        reduce(or_, [qx.gates.RZ(angle) for angle in angles_2])
        @ reduce(or_, [qx.gates.RY(angle) for angle in angles_1])
        @ reduce(or_, [qx.gates.RX(angle) for angle in angles_0])
    )
    target_rho = u @ qx.zero_state_matrix(num_qubits)
    assert qx.fidelity(rho, target_rho) > 0.9999


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_entanglement(seed):
    """Test a 2-qubit state with entanglement (detects wrong-endianness)."""
    key = jax.random.key(seed)
    theta, phi = jax.random.uniform(key, shape=(2,), minval=-jnp.pi, maxval=jnp.pi)
    program = Program()
    program += RX(float(theta), 0)
    program += RX(float(phi), 1)
    program += CNOT(0, 1)
    rho = compute_program_density_matrix(program)
    u = qx.gates.CNOT @ (qx.gates.RX(theta) | qx.gates.RX(phi))
    target_rho = u @ qx.zero_state_matrix(2)
    assert qx.fidelity(rho, target_rho) > 0.9999


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_multi_qubit_entanglement(seed):
    """Test a 3-qubit state with entanglement (detects wrong-endianness)."""
    key = jax.random.key(seed)
    theta, phi, lam = jax.random.uniform(key, shape=(3,), minval=-jnp.pi, maxval=jnp.pi)
    program = Program()
    program += RX(float(theta), 0)
    program += RY(float(phi), 1)
    program += RZ(float(lam), 2)
    program += CNOT(0, 1)
    program += CNOT(1, 2)
    rho = compute_program_density_matrix(program)
    u = (
        (qx.gates.I | qx.gates.CNOT)
        @ (qx.gates.CNOT | qx.gates.I)
        @ (qx.gates.RX(theta) | qx.gates.RY(phi) | qx.gates.RZ(lam))
    )
    target_rho = u @ qx.zero_state_matrix(3)
    assert qx.fidelity(rho, target_rho) > 0.9999


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_multi_qubit_gates(seed):
    """Test a 4-qubit state with 3-qubit gates."""
    key = jax.random.key(seed)
    theta, phi, lam, gamma = jax.random.uniform(key, shape=(4,), minval=-jnp.pi, maxval=jnp.pi)
    program = Program()
    program += RX(float(theta), 0)
    program += RY(float(phi), 1)
    program += RZ(float(lam), 2)
    program += RX(float(gamma), 3)
    program += CNOT(0, 1)
    program += CNOT(2, 3)
    program += RX(float(theta), 0)
    program += RY(float(phi), 1)
    program += RZ(float(lam), 2)
    program += RX(float(gamma), 3)
    program += CCNOT(0, 1, 2)
    program += CCNOT(1, 2, 3)
    rho = compute_program_density_matrix(program)
    u = (
        (qx.gates.I | qx.gates.CCNOT)
        @ (qx.gates.CCNOT | qx.gates.I)
        @ (qx.gates.RX(theta) | qx.gates.RY(phi) | qx.gates.RZ(lam) | qx.gates.RX(gamma))
        @ (qx.gates.CNOT | qx.gates.CNOT)
        @ (qx.gates.RX(theta) | qx.gates.RY(phi) | qx.gates.RZ(lam) | qx.gates.RX(gamma))
    )
    target_rho = u @ qx.zero_state_matrix(4)
    assert qx.fidelity(rho, target_rho) > 0.9999


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_disjoint_pairs(seed):
    """Test a 4-qubit state with entanglement only between pairs (detects splitting issues)."""
    key = jax.random.key(seed)
    k_angles, k_qubits = jax.random.split(key)
    theta, phi, lam, gamma = jax.random.uniform(k_angles, shape=(4,), minval=-jnp.pi, maxval=jnp.pi)
    q0, q1, q2, q3 = [int(q) for q in jax.random.choice(k_qubits, 12, shape=(4,), replace=False)]
    program = Program()
    program += RX(float(theta), q0)
    program += RY(float(phi), q1)
    program += RZ(float(lam), q2)
    program += RX(float(gamma), q3)
    program += CNOT(q0, q1)
    program += CNOT(q2, q3)
    rho = compute_program_density_matrix(program, qubits=[q0, q1, q2, q3])
    u = (qx.gates.CNOT | qx.gates.CNOT) @ (
        qx.gates.RX(theta) | qx.gates.RY(phi) | qx.gates.RZ(lam) | qx.gates.RX(gamma)
    )
    target_rho = u @ qx.zero_state_matrix(4)
    assert qx.fidelity(rho, target_rho) > 0.9999


def test_defgate():
    """Test a 2-qubit state with a DefGate."""
    key = jax.random.key(5973)
    theta, phi = jax.random.uniform(key, shape=(2,), minval=-jnp.pi, maxval=jnp.pi)
    program = Program()
    program += DefGate(name="BLARG", matrix=matrices.CNOT, parameters=[])
    program += RX(float(theta), 0)
    program += RX(float(phi), 1)
    program += Gate("BLARG", [], (0, 1))
    rho = compute_program_density_matrix(program)
    u = qx.gates.CNOT @ (qx.gates.RX(theta) | qx.gates.RX(phi))
    target_rho = u @ qx.zero_state_matrix(2)
    assert qx.fidelity(rho, target_rho) > 0.9999


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
    pure_rho_np = np.asarray((qx.gates.RX(angle) @ qx.zero_state_matrix(1)).matrix)
    target_rho = qx.DensityMatrix.from_matrix(jnp.asarray(p * pure_rho_np + ((1 - p) / dim) * jnp.eye(dim)), dims=(2,))

    assert qx.fidelity(rho, target_rho) > 0.9999


def test_multi_qubit_depolarizing_noise_1q():
    """Test depolarizing noise on a 3-qubit state with entanglement (1Q channels only)."""
    p = [0.9887177, 0.97, 0.97129439]
    theta, phi, lam = -2.5254901911114866, 1.229585029961344, -1.9248113321783669
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
    target_rho = qx.DensityMatrix.from_matrix(
        jnp.array(
            [
                [
                    6.30173393e-02 + 3.41965066e-18j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    4.34873146e-02 - 3.41965066e-18j,
                    -6.83930132e-18 - 1.28688397e-01j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    -1.36786026e-17 - 1.86481977e-01j,
                ],
                [
                    0.00000000e00 + 0.00000000e00j,
                    9.17646272e-04 + 4.97962896e-20j,
                    6.33253840e-04 - 4.97962896e-20j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    -9.95925792e-20 - 1.87393548e-03j,
                    -1.99185158e-19 - 2.71551438e-03j,
                    0.00000000e00 + 0.00000000e00j,
                ],
                [
                    0.00000000e00 + 0.00000000e00j,
                    6.33253840e-04 + 4.97962896e-20j,
                    4.67908978e-04 + 2.48981448e-20j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    9.95925792e-20 - 1.38464417e-03j,
                    9.95925792e-20 - 1.87393548e-03j,
                    0.00000000e00 + 0.00000000e00j,
                ],
                [
                    4.34873146e-02 + 3.41965066e-18j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    3.21326199e-02 + 1.70982533e-18j,
                    6.83930132e-18 - 9.50873926e-02j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    6.83930132e-18 - 1.28688397e-01j,
                ],
                [
                    -6.83930132e-18 + 1.28688397e-01j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    -6.83930132e-18 + 9.50873926e-02j,
                    3.00725397e-01 + 2.73572053e-17j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    4.06992644e-01 - 2.73572053e-17j,
                ],
                [
                    0.00000000e00 + 0.00000000e00j,
                    -9.95925792e-20 + 1.87393548e-03j,
                    -9.95925792e-20 + 1.38464417e-03j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    4.37910490e-03 + 3.98370317e-19j,
                    5.92654794e-03 - 3.98370317e-19j,
                    0.00000000e00 + 0.00000000e00j,
                ],
                [
                    0.00000000e00 + 0.00000000e00j,
                    1.99185158e-19 + 2.71551438e-03j,
                    9.95925792e-20 + 1.87393548e-03j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    5.92654794e-03 - 3.98370317e-19j,
                    8.58814314e-03 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                ],
                [
                    1.36786026e-17 + 1.86481977e-01j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    6.83930132e-18 + 1.28688397e-01j,
                    4.06992644e-01 - 2.73572053e-17j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    5.89771841e-01 + 0.00000000e00j,
                ],
            ]
        ),
        dims=(2, 2, 2),
    )

    assert qx.fidelity(rho, target_rho) > 0.9999


def test_multi_qubit_depolarizing_noise():
    """Test depolarizing noise on a 3-qubit state with entanglement (1Q + 2Q channels)."""
    p = [0.9887177, 0.97, 0.97129439, 0.9857, 0.97463]
    theta, phi, lam = -2.5254901911114866, 1.229585029961344, -1.9248113321783669
    insts = [RX(theta, 10), RY(phi, 1), RZ(lam, 8), CNOT(10, 1), CNOT(1, 8)]
    qubits = [10, 1, 8]
    noise_model = NoiseModel(
        channels=frozenset(
            Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=pi) for inst, pi in zip(insts, p)
        )
    )
    program = Program(insts)
    rho = compute_program_density_matrix(program, noise_model=noise_model, qubits=qubits)

    target_rho = qx.DensityMatrix.from_matrix(
        jnp.array(
            [
                [
                    6.46234650e-02 + 2.10040276e-19j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    4.17779488e-02 + 0.00000000e00j,
                    6.65722865e-18 - 1.25262516e-01j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    1.31409349e-17 - 1.79151892e-01j,
                ],
                [
                    0.00000000e00 + 0.00000000e00j,
                    1.58045557e-03 + 1.63040223e-20j,
                    6.08362461e-04 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    1.82439069e-19 - 3.43277631e-03j,
                    1.91355769e-19 - 2.60877541e-03j,
                    0.00000000e00 + 0.00000000e00j,
                ],
                [
                    0.00000000e00 + 0.00000000e00j,
                    6.08362461e-04 + 0.00000000e00j,
                    1.14839615e-03 + 4.02234934e-20j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 - 1.33021783e-03j,
                    1.82439069e-19 - 3.43277631e-03j,
                    0.00000000e00 + 0.00000000e00j,
                ],
                [
                    4.17779488e-02 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    3.49527382e-02 + 1.85265714e-18j,
                    1.64569209e-36 - 9.13497728e-02j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    6.65722865e-18 - 1.25262516e-01j,
                ],
                [
                    -6.65722865e-18 + 1.25262516e-01j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    -1.64569209e-36 + 9.13497728e-02j,
                    2.98032644e-01 + 5.46239868e-19j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    3.90994899e-01 + 3.29138418e-36j,
                ],
                [
                    0.00000000e00 + 0.00000000e00j,
                    -1.82439069e-19 + 3.43277631e-03j,
                    0.00000000e00 + 1.33021783e-03j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    9.95061471e-03 + 3.52503614e-19j,
                    5.69359257e-03 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                ],
                [
                    0.00000000e00 + 0.00000000e00j,
                    -1.91355769e-19 + 2.60877541e-03j,
                    -1.82439069e-19 + 3.43277631e-03j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    5.69359257e-03 + 3.82711537e-19j,
                    1.39942079e-02 + 1.11792669e-18j,
                    0.00000000e00 + 0.00000000e00j,
                ],
                [
                    -1.31409349e-17 + 1.79151892e-01j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    -6.65722865e-18 + 1.25262516e-01j,
                    3.90994899e-01 + 2.62818699e-17j,
                    0.00000000e00 + 0.00000000e00j,
                    0.00000000e00 + 0.00000000e00j,
                    5.75717479e-01 + 5.31099796e-17j,
                ],
            ]
        ),
        dims=(2, 2, 2),
    )

    assert qx.fidelity(rho, target_rho) > 0.9999


# ──────────────────────────────────────────────────────────
# Non-zero qubit index tests
# ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [4865, 3845, 3083])
def test_rx_nonzero_index(seed):
    """Test that the simulator can handle a single RX gate with a non-zero index."""
    key = jax.random.key(seed)
    angle = float(jax.random.uniform(key, minval=-jnp.pi, maxval=jnp.pi))
    program = Program(RX(angle, 6))
    rho = compute_program_density_matrix(program)
    target_rho = qx.gates.RX(angle) @ qx.zero_state_matrix(1)
    assert qx.fidelity(rho, target_rho) > 0.9999

# ──────────────────────────────────────────────────────────
# Qudit tests
# ──────────────────────────────────────────────────────────

def test_RX12():
    """TRX12 gate on a qubit: state is auto-promoted to a qutrit, final state matches quax reference."""
    key = jax.random.key(4444)
    angle = float(jax.random.uniform(key, minval=-jnp.pi, maxval=jnp.pi))
    program = Program()
    program += Gate("TRX12", [angle], (5,))
    rho = compute_program_density_matrix(program)
    # Qubit starts as |0⟩ promoted to the qutrit |0⟩; TRX12 acts in the |1⟩-|2⟩ subspace
    # so starting from |0⟩ the state is unchanged (identity on |0⟩).
    target_rho = qx.gates.TRX12(angle) @ qx.zero_state_matrix(dims=(3,))
    assert qx.fidelity(rho, target_rho) > 0.9999

def test_multiqudit():
    """
    Program with a qubit and two qutrits on non-sequential physical qubit indices.

    Uses Gate() for qutrit gates (TRX01, TRX12, TSWAP) and RX for the qubit.
    Verifies automatic per-qudit promotion and the correct final state.
    """
    key = jax.random.key(5555)
    theta, phi, lam = jax.random.uniform(key, shape=(3,), minval=-jnp.pi, maxval=jnp.pi)
    # Physical qubit indices (non-sequential); slot 0=q5 (qubit), slot 1=q3 (qutrit), slot 2=q7 (qutrit)
    q0, q1, q2 = 5, 3, 7
    program = Program()
    program += RX(float(theta), q0)                   # qubit gate
    program += Gate("TRX01", [float(phi)], (q1,))     # qutrit gate in |0⟩-|1⟩ subspace
    program += Gate("TRX12", [float(lam)], (q2,))     # qutrit gate in |1⟩-|2⟩ subspace
    program += Gate("TSWAP", [], (q1, q2))             # 2-qutrit swap
    rho = compute_program_density_matrix(program, qubits=[q0, q1, q2])
    # Build expected state: (I_qubit ⊗ TSWAP) ∘ (RX ⊗ TRX01 ⊗ TRX12) |000⟩
    u = _to_unitary(
        (qx.gates.I | qx.gates.TSWAP)
        @ (qx.gates.RX(theta) | qx.gates.TRX01(phi) | qx.gates.TRX12(lam))
    )
    target_rho = u @ qx.zero_state_matrix(dims=(2, 3, 3))
    assert qx.fidelity(rho, target_rho) > 0.9999

def test_leaky_rx():
    """RX gate with stochastic leakage noise: qubit is promoted to qutrit and |2⟩ gains population."""
    key = jax.random.key(6666)
    angle = float(jax.random.uniform(key, minval=-jnp.pi, maxval=jnp.pi))
    gamma = 0.05  # 5% leakage probability
    rx_inst = RX(angle, 0)
    rx_unitary = qx.gates.RX(angle)
    # Compose leakage (qutrit KrausMap) with RX unitary; promote_hilbert_space handles dim mismatch
    leakage_kraus = qx.stochastic_leakage_operators(gamma)
    process = qx.to_superop(leakage_kraus @ rx_unitary)
    noise_model = NoiseModel(
        channels=frozenset([Channel(inst=rx_inst, process=process, target_unitary=rx_unitary)])
    )
    program = Program(rx_inst)
    rho = compute_program_density_matrix(program, noise_model=noise_model)
    # Target: apply the full leaky-RX superop to |0⟩ of a qutrit
    target_rho = process @ qx.zero_state_matrix(dims=(3,))
    assert qx.fidelity(rho, target_rho) > 0.9999
    # Verify non-trivial leakage: |2⟩⟨2| population must be positive
    p2 = float(jnp.real(rho.matrix[2, 2]))
    assert p2 > 0, f"Expected leaked population in |2⟩, got {p2}"

def test_noisy_program():
    """
    Two-qudit program with stochastic leakage on q0, seepage on q1,
    and a coherent qutrit rotation error (TRX12) applied to both qudits.
    """
    theta, phi = 1.1, -0.8
    epsilon = 0.04   # coherent rotation angle in the |1⟩-|2⟩ subspace
    gamma_leak = 0.03
    gamma_seep = 0.04

    rx0_inst = RX(theta, 0)
    trx01_inst = Gate("TRX01", [phi], (1,))

    # q0: leaky RX — leakage from computational subspace to |2⟩
    rx0_unitary = qx.gates.RX(theta)
    leakage_kraus = qx.stochastic_leakage_operators(gamma_leak)
    process0 = qx.to_superop(leakage_kraus @ rx0_unitary)
    channel0 = Channel(inst=rx0_inst, process=process0, target_unitary=rx0_unitary)

    # q1: TRX01 with seepage — seepage brings |2⟩ population back to |1⟩
    trx01_unitary = qx.gates.TRX01(phi)  # acts on qutrit (3,) space
    seepage_kraus = qx.seepage_operators(gamma_seep)
    process1 = qx.to_superop(seepage_kraus @ trx01_unitary)
    channel1 = Channel(inst=trx01_inst, process=process1, target_unitary=trx01_unitary)

    noise_model = NoiseModel(channels=frozenset([channel0, channel1]))

    # Coherent qutrit rotation error applied after the main gates
    trx12_err = Gate("TRX12", [epsilon], (0,))
    trx12_err_1 = Gate("TRX12", [epsilon], (1,))

    program = Program()
    program += rx0_inst
    program += trx01_inst
    program += trx12_err
    program += trx12_err_1
    rho = compute_program_density_matrix(program, noise_model=noise_model)

    # Build expected state step-by-step mirroring the simulator
    trx12_superop = qx.to_superop(qx.gates.TRX12(epsilon))
    target_rho = qx.zero_state_matrix(dims=(3, 3))
    target_rho = qx.targeted_apply_superop(process0, target_rho, (0,))
    target_rho = qx.targeted_apply_superop(process1, target_rho, (1,))
    target_rho = qx.targeted_apply_superop(trx12_superop, target_rho, (0,))
    target_rho = qx.targeted_apply_superop(trx12_superop, target_rho, (1,))
    assert qx.fidelity(rho, target_rho) > 0.9999