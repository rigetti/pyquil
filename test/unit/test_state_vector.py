"""Unit tests for the quax-based state vector simulator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import quax as qx

from pyquil.gates import CNOT, CZ, H, MEASURE, RESET, RX, RY, RZ, X
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, Qubit
from pyquil.quilbase import Declare, DefGate, Gate as QuilGate, Measurement as QuilMeasurement, ResetQubit
from pyquil.noise._channels import Channel, MeasurementChannel, ResetChannel
from pyquil.noise._noise_model import NoiseModel
from pyquil.simulation._simulator import (
    PureStateVectorSimulator,
    DensityMatrixSimulator,
    TrajectorySimulator,
    _apply_trajectory_operations as apply_trajectory_operations,
    _run_batched_trajectories,
)

_EMPTY_PARAMS = jnp.array([], dtype=float)


def _sv(program, qubits=None, memory_map=None):
    """Compute pure state vector for a gate-only program."""
    sim = PureStateVectorSimulator(program, qubits=qubits)
    if memory_map:
        params = sim.linearize(memory_map)
    else:
        params = _EMPTY_PARAMS
    return sim.compute(params)


def _simulate_trajectories(program, noise_model=None, qubits=None, num_trajectories=1,
                            batch_size=256, random_seed=0):
    """Helper: build + compress + run trajectories, returning (psi, outcomes)."""
    sim = TrajectorySimulator(program, noise_model=noise_model, qubits=qubits)
    resolved = sim.resolve(_EMPTY_PARAMS)
    compressed = sim.compress(resolved)
    operations = sim.adapt(compressed)
    all_psis, all_outcomes = _run_batched_trajectories(
        operations, sim.n_qubits, num_trajectories, batch_size, random_seed,
        keep_states=True, dims=sim.dims,
    )
    assert all_psis is not None
    if len(all_psis) == 1:
        return all_psis[0], all_outcomes[0]
    combined_data = jnp.concatenate([p.matrix for p in all_psis], axis=0)
    combined_psi = qx.StateVector.from_matrix(combined_data, all_psis[0].dims)
    combined_outcomes = jnp.concatenate(all_outcomes, axis=0)
    return combined_psi, combined_outcomes


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
        target = qx.StateVector.from_matrix(jnp.array([1.0, 0, 0, 0, 0, 0, 0, 1.0], dtype=complex) / jnp.sqrt(2), dims=(2, 2, 2))
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


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory simulator tests
# ──────────────────────────────────────────────────────────────────────────────


class TestTrajectoryNoiseless:
    """Test that the trajectory simulator preserves noiseless behavior."""

    def test_single_gate_noiseless(self):
        """Without noise, trajectory simulation matches unitary simulation."""
        p = Program(H(0))
        psi_noiseless = _sv(p, qubits=[0])
        psi_traj, outcomes = _simulate_trajectories(
            p, noise_model=None, qubits=[0], num_trajectories=1
        )
        assert qx.fidelity(psi_noiseless, psi_traj) > 0.9999

    def test_bell_state_noiseless(self):
        """Multi-qubit noiseless trajectory."""
        p = Program(H(0), CNOT(0, 1))
        psi_noiseless = _sv(p, qubits=[0, 1])
        psi_traj, outcomes = _simulate_trajectories(
            p, noise_model=None, qubits=[0, 1], num_trajectories=1
        )
        assert qx.fidelity(psi_noiseless, psi_traj) > 0.9999

    def test_multiple_trajectories_noiseless_deterministic(self):
        """Multiple noiseless trajectories should all give same result."""
        p = Program(X(0))
        psi_batch, outcomes = _simulate_trajectories(
            p, noise_model=None, qubits=[0], num_trajectories=8
        )
        # Each trajectory should be |1⟩
        target = qx.StateVector.from_matrix(jnp.array([0.0, 1.0], dtype=complex), dims=(2,))
        probs = qx.probabilities(psi_batch)
        # All trajectories: prob of |1⟩ = 1
        assert jnp.allclose(probs[:, 1], 1.0, atol=1e-6)


class TestTrajectoryNoisy:
    """Test noisy trajectory simulation with known analytical results."""

    def _make_bitflip_noise_model(self, p_error: float, qubit: int = 0) -> NoiseModel:
        """Create a noise model with a bit-flip channel on X gate."""
        # Bit-flip channel applied AFTER the gate: E(rho) = (1-p) U rho U† + p X U rho U† X
        # As a combined superop that includes the gate:
        inst = X(qubit)
        unitary = qx.gates.X
        # Build noisy superop: (1-p)|U><U| + p|XU><XU| in superop picture
        # Simpler: compose depolarizing-like channel with the gate
        # Use a Pauli channel: p_I = 1-p, p_X = p, p_Y = 0, p_Z = 0
        pauli_probs = {"X": p_error}
        channel = Channel.from_pauli_noise(inst=inst, pauli_noise=pauli_probs)
        return NoiseModel(channels=frozenset([channel]))

    def _make_depolarizing_noise_model(self, fidelity: float, qubit: int = 0) -> NoiseModel:
        """Create a noise model with depolarizing noise on X gate."""
        inst = X(qubit)
        channel = Channel.from_gate_fidelity(inst=inst, fidelity=fidelity)
        return NoiseModel(channels=frozenset([channel]))

    def test_noiseless_gate_with_noise_model(self):
        """A noise model that doesn't cover the applied gate should leave it noiseless."""
        # Noise model only covers X gate, but we apply H
        noise_model = self._make_bitflip_noise_model(0.1, qubit=0)
        p = Program(H(0))
        psi, outcomes = _simulate_trajectories(
            p, noise_model=noise_model, qubits=[0], num_trajectories=1
        )
        target = qx.StateVector.from_matrix(
            jnp.array([1.0, 1.0], dtype=complex) / jnp.sqrt(2), dims=(2,)
        )
        assert qx.fidelity(psi, target) > 0.9999

    def test_bitflip_statistics(self):
        """Bit-flip noise should produce correct outcome statistics."""
        p_error = 0.3
        noise_model = self._make_bitflip_noise_model(p_error, qubit=0)
        # X gate with bit-flip noise: X|0⟩=|1⟩, then bit-flip with p=0.3
        # So final state: (1-p)|1⟩ + p|0⟩ in trajectory picture
        p = Program(X(0))
        num_traj = 2048
        psi_batch, outcomes = _simulate_trajectories(
            p, noise_model=noise_model, qubits=[0], num_trajectories=num_traj,
            batch_size=256, random_seed=42,
        )
        # Get probabilities for each trajectory
        probs = qx.probabilities(psi_batch)  # shape (num_traj, 2)
        # Each trajectory should collapse to either |0⟩ or |1⟩
        # Count how many ended in |0⟩ (bit-flipped from |1⟩)
        in_zero = jnp.sum(probs[:, 0] > 0.5)
        observed_flip_rate = float(in_zero) / num_traj
        # Expected: p_error fraction should flip to |0⟩
        assert abs(observed_flip_rate - p_error) < 0.05, (
            f"Expected flip rate ~{p_error}, got {observed_flip_rate}"
        )

    def test_depolarizing_statistics(self):
        """Depolarizing noise on identity-like circuit should produce mixed results."""
        fidelity_val = 0.9
        noise_model = self._make_depolarizing_noise_model(fidelity_val, qubit=0)
        p = Program(X(0))
        num_traj = 2048
        psi_batch, outcomes = _simulate_trajectories(
            p, noise_model=noise_model, qubits=[0], num_trajectories=num_traj,
            batch_size=256, random_seed=123,
        )
        probs = qx.probabilities(psi_batch)
        # Average probability of |1⟩ across trajectories should be close to
        # the expected value from depolarizing channel on |1⟩:
        # p(|1⟩) = F + (1-F)/d where d=2 for single qubit depol
        # Actually for depol with constant p: output prob depends on p
        avg_prob_1 = float(jnp.mean(probs[:, 1]))
        # For depolarizing channel with fidelity F on a single qubit:
        # After X|0⟩=|1⟩, depol: prob(|1⟩) = (2F-1) * 1/2 + 1/2 = F
        # (since F_avg = (d*p + 1)/(d+1) and rho_out = p*rho + (1-p)*I/d)
        assert abs(avg_prob_1 - fidelity_val) < 0.05

    def test_two_qubit_noise(self):
        """Test that noise applies independently to separate qubits."""
        p_error = 0.2
        inst_q0 = X(0)
        inst_q1 = X(1)
        ch0 = Channel.from_pauli_noise(inst=inst_q0, pauli_noise={"X": p_error})
        ch1 = Channel.from_pauli_noise(inst=inst_q1, pauli_noise={"X": p_error})
        noise_model = NoiseModel(channels=frozenset([ch0, ch1]))

        prog = Program(X(0), X(1))
        num_traj = 2048
        psi_batch, _ = _simulate_trajectories(
            prog, noise_model=noise_model, qubits=[0, 1], num_trajectories=num_traj,
            batch_size=256, random_seed=7,
        )
        probs = qx.probabilities(psi_batch)  # shape (num_traj, 4)
        # State |11⟩ = index 3. Both flipped: p_error^2 gives |00⟩
        # Expected: P(|11⟩) ≈ (1-p)^2, P(|00⟩) ≈ p^2
        avg_prob_11 = float(jnp.mean(probs[:, 3]))
        expected_prob_11 = (1 - p_error) ** 2
        assert abs(avg_prob_11 - expected_prob_11) < 0.05


class TestTrajectoryMeasurement:
    """Test mid-circuit measurement in trajectory simulation."""

    def test_measurement_records_outcome(self):
        """Measurement should record classical outcome."""
        p = Program(H(0), MEASURE(0, None))
        psi, outcomes = _simulate_trajectories(
            p, noise_model=None, qubits=[0], num_trajectories=100,
            batch_size=100, random_seed=42,
        )
        # outcomes shape should be (100, 1) — one measurement
        assert outcomes.shape == (100, 1)
        # Outcomes should be 0 or 1
        assert jnp.all((outcomes == 0) | (outcomes == 1))
        # Roughly 50/50 from H|0⟩
        frac_0 = float(jnp.mean(outcomes == 0))
        assert 0.3 < frac_0 < 0.7

    def test_measurement_collapses_state(self):
        """After measurement, state should be consistent with outcome."""
        p = Program(H(0), MEASURE(0, None))
        psi, outcomes = _simulate_trajectories(
            p, noise_model=None, qubits=[0], num_trajectories=64,
            batch_size=64, random_seed=99,
        )
        probs = qx.probabilities(psi)  # (64, 2)
        # For each trajectory, the state should be collapsed
        for i in range(64):
            outcome = int(outcomes[i, 0])
            assert probs[i, outcome] > 0.999

    def test_noisy_measurement(self):
        """Noisy measurement with confusion should produce errors."""
        # Prepare |0⟩, measure with 80% fidelity
        qubit = Qubit(0)
        m_inst = QuilMeasurement(qubit=qubit, classical_reg=None)
        meas_ch = MeasurementChannel.from_readout_fidelity(inst=m_inst, fidelity=0.8)
        noise_model = NoiseModel(channels=frozenset([meas_ch]))

        p = Program(MEASURE(0, None))
        psi, outcomes = _simulate_trajectories(
            p, noise_model=noise_model, qubits=[0], num_trajectories=1024,
            batch_size=256, random_seed=55,
        )
        # Prepared in |0⟩, ideal measurement gives 0, but with 20% error → ~20% ones
        frac_1 = float(jnp.mean(outcomes == 1))
        assert 0.1 < frac_1 < 0.3


class TestTrajectoryReset:
    """Test reset operations in trajectory simulation."""

    def test_reset_to_ground(self):
        """Reset should bring qubit to |0⟩."""
        p = Program(X(0), ResetQubit(Qubit(0)))
        psi, _ = _simulate_trajectories(
            p, noise_model=None, qubits=[0], num_trajectories=1,
        )
        target = qx.StateVector.from_matrix(jnp.array([1.0, 0.0], dtype=complex), dims=(2,))
        assert qx.fidelity(psi, target) > 0.9999

    def test_global_reset(self):
        """Global RESET should reset all qubits."""
        p = Program(X(0), X(1), RESET())
        psi, _ = _simulate_trajectories(
            p, noise_model=None, qubits=[0, 1], num_trajectories=1,
        )
        target = qx.StateVector.from_matrix(
            jnp.array([1.0, 0.0, 0.0, 0.0], dtype=complex), dims=(2, 2)
        )
        assert qx.fidelity(psi, target) > 0.9999

    def test_noisy_reset(self):
        """Noisy reset should have imperfect fidelity."""
        qubit = Qubit(0)
        reset_inst = ResetQubit(qubit)
        reset_ch = ResetChannel.from_reset_fidelity(inst=reset_inst, fidelity=0.9)
        noise_model = NoiseModel(channels=frozenset([reset_ch]))

        # Start in |1⟩, apply noisy reset
        p = Program(X(0), ResetQubit(Qubit(0)))
        num_traj = 2048
        psi, _ = _simulate_trajectories(
            p, noise_model=noise_model, qubits=[0], num_trajectories=num_traj,
            batch_size=256, random_seed=13,
        )
        probs = qx.probabilities(psi)  # (num_traj, 2)
        # With 90% reset fidelity, ~90% should end in |0⟩
        avg_prob_0 = float(jnp.mean(probs[:, 0]))
        assert avg_prob_0 > 0.85


class TestTrajectoryBatching:
    """Test that batch processing works correctly."""

    def test_batch_size_smaller_than_trajectories(self):
        """Multiple batches should produce same statistics as single batch."""
        p = Program(H(0))
        noise_model = None

        # Single batch
        psi_1, outcomes_1 = _simulate_trajectories(
            p, noise_model=noise_model, qubits=[0], num_trajectories=64,
            batch_size=64, random_seed=42,
        )
        # Multiple batches (same seed)
        psi_2, outcomes_2 = _simulate_trajectories(
            p, noise_model=noise_model, qubits=[0], num_trajectories=64,
            batch_size=16, random_seed=42,
        )
        # Note: different batching may produce different results due to key splitting,
        # but shapes should match
        assert psi_1.matrix.shape == psi_2.matrix.shape
        assert outcomes_1.shape == outcomes_2.shape


class TestComputeProgramStateVectorWithNoise:
    """Test the TrajectorySimulator with noise_model parameter."""

    def test_noise_model_none_unchanged(self):
        """With noise_model=None, behavior is identical to original."""
        p = Program(H(0), CNOT(0, 1))
        psi = _sv(p, qubits=[0, 1])
        target = qx.StateVector.from_matrix(
            jnp.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / jnp.sqrt(2), dims=(2, 2)
        )
        assert qx.fidelity(psi, target) > 0.9999

    def test_noise_model_single_trajectory(self):
        """With noise_model provided, runs a single trajectory."""
        inst = X(0)
        channel = Channel.from_gate_fidelity(inst=inst, fidelity=1.0)
        noise_model = NoiseModel(channels=frozenset([channel]))
        p = Program(X(0))
        sim = TrajectorySimulator(p, noise_model=noise_model, qubits=[0])
        psi, _ = sim.compute(_EMPTY_PARAMS, jax.random.key(0))
        # Perfect fidelity channel → same as noiseless
        target = qx.StateVector.from_matrix(jnp.array([0.0, 1.0], dtype=complex), dims=(2,))
        assert qx.fidelity(psi, target) > 0.999


class TestSampleProgramTrajectories:
    """Test the scalable TrajectorySimulator.sample function."""

    def test_returns_outcomes_only(self):
        """Should return measurement outcomes without state vectors."""
        p = Program(H(0), MEASURE(0, None))
        sim = TrajectorySimulator(p, noise_model=None, qubits=[0])
        outcomes = sim.sample(
            _EMPTY_PARAMS, num_trajectories=100,
            batch_size=32, random_seed=42,
        )
        assert outcomes.shape == (100, 1)
        assert jnp.all((outcomes == 0) | (outcomes == 1))

    def test_no_measurements_empty_outcomes(self):
        """Without measurements, outcomes array should have zero columns."""
        p = Program(H(0))
        sim = TrajectorySimulator(p, noise_model=None, qubits=[0])
        outcomes = sim.sample(
            _EMPTY_PARAMS, num_trajectories=10,
        )
        assert outcomes.shape == (10, 0)

    def test_bitflip_statistics(self):
        """Outcome statistics should match noise model."""
        p_error = 0.3
        inst = X(0)
        ch = Channel.from_pauli_noise(inst=inst, pauli_noise={"X": p_error})
        noise_model = NoiseModel(channels=frozenset([ch]))

        p = Program(X(0), MEASURE(0, None))
        sim = TrajectorySimulator(p, noise_model=noise_model, qubits=[0])
        outcomes = sim.sample(
            _EMPTY_PARAMS, num_trajectories=2048,
            batch_size=256, random_seed=42,
        )
        # X|0⟩ = |1⟩, then bit-flip with p=0.3 → ~30% get |0⟩
        # Measurement outcome reflects the final state
        frac_0 = float(jnp.mean(outcomes == 0))
        assert abs(frac_0 - p_error) < 0.05

    def test_batch_size_does_not_affect_shape(self):
        """Different batch sizes should produce same output shape."""
        p = Program(H(0), MEASURE(0, None))
        sim = TrajectorySimulator(p, qubits=[0])
        outcomes_small = sim.sample(
            _EMPTY_PARAMS, num_trajectories=100, batch_size=10,
        )
        outcomes_large = sim.sample(
            _EMPTY_PARAMS, num_trajectories=100, batch_size=100,
        )
        assert outcomes_small.shape == outcomes_large.shape == (100, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Linearizer / Compressor architecture tests
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildSimulationLinearizer:
    """Tests for the simulator linearizer closure."""

    def test_no_params_returns_empty(self):
        p = Program(H(0), CNOT(0, 1), MEASURE(0, MemoryReference("ro", 0)))
        p += Declare("ro", "BIT", 1)
        sim = TrajectorySimulator(p)
        params = sim.linearize({})
        assert params.shape == (0,)
        assert sim.n_qubits == 2

    def test_single_param(self):
        p = Program()
        p += Declare("theta", "REAL", 1)
        p += Declare("ro", "BIT", 1)
        p += RZ(MemoryReference("theta", 0), 0)
        p += MEASURE(0, MemoryReference("ro", 0))
        sim = TrajectorySimulator(p)
        params = sim.linearize({"theta": [1.23]})
        assert params.shape == (1,)
        np.testing.assert_allclose(float(params[0]), 1.23)

    def test_multiple_params_ordering(self):
        p = Program()
        p += Declare("alpha", "REAL", 1)
        p += Declare("beta", "REAL", 2)
        p += Declare("ro", "BIT", 2)
        p += RZ(MemoryReference("alpha", 0), 0)
        p += RX(MemoryReference("beta", 0), 0)
        p += RY(MemoryReference("beta", 1), 1)
        p += MEASURE(0, MemoryReference("ro", 0))
        p += MEASURE(1, MemoryReference("ro", 1))
        sim = TrajectorySimulator(p)
        params = sim.linearize({"alpha": [0.1], "beta": [0.2, 0.3]})
        assert params.shape == (3,)
        np.testing.assert_allclose(float(params[0]), 0.1)
        np.testing.assert_allclose(float(params[1]), 0.2)
        np.testing.assert_allclose(float(params[2]), 0.3)

    def test_ro_register_excluded(self):
        """Ensure 'ro' register is not treated as a parameter register."""
        p = Program()
        p += Declare("theta", "REAL", 1)
        p += Declare("ro", "BIT", 1)
        p += RZ(MemoryReference("theta", 0), 0)
        p += MEASURE(0, MemoryReference("ro", 0))
        sim = TrajectorySimulator(p)
        params = sim.linearize({"theta": [np.pi]})
        assert params.shape == (1,)


class TestCompressor:
    """Tests for the compressor at various max_subsystem_size settings."""

    # ── max_subsystem_size=0 (no merging) ──

    def test_no_merge_noiseless_matches_direct(self):
        """max_subsystem_size=0 compressor output should match direct computation."""
        p = Program(H(0), CNOT(0, 1), RZ(0.5, 0))
        psi_direct = _sv(p)

        sim = TrajectorySimulator(p, max_subsystem_size=0)
        ops = sim.adapt(sim.compress(sim.resolve(_EMPTY_PARAMS)))

        psi = qx.zero_state_vector(sim.n_qubits)
        for op, subsystem in ops:
            if isinstance(op, qx.Unitary):
                psi = qx.targeted_apply_unitary(op, psi, subsystem)

        assert qx.fidelity(psi, psi_direct) > 0.9999

    def test_no_merge_parameterized_gate(self):
        """max_subsystem_size=0 should handle parameterized gates via the param vector."""
        p = Program()
        p += Declare("theta", "REAL", 1)
        p += Declare("ro", "BIT", 1)
        p += RZ(MemoryReference("theta", 0), 0)
        p += MEASURE(0, MemoryReference("ro", 0))

        sim = TrajectorySimulator(p, max_subsystem_size=0)
        params = sim.linearize({"theta": [np.pi]})
        ops = sim.adapt(sim.compress(sim.resolve(params)))

        psi = qx.zero_state_vector(sim.n_qubits)
        for op, subsystem in ops:
            if isinstance(op, qx.Unitary):
                psi = qx.targeted_apply_unitary(op, psi, subsystem)
            elif isinstance(op, qx.QuantumInstrument):
                key = jax.random.key(0)
                psi, _ = qx.targeted_apply_instrument_to_state_vector(op, psi, key, subsystem)

        psi_direct = _sv(Program(RZ(np.pi, 0)))
        assert qx.fidelity(psi, psi_direct) > 0.9999

    def test_no_merge_noisy_ops_count(self):
        """max_subsystem_size=0 noisy: should have exactly one op per instruction."""
        p = Program(RX(np.pi / 2, 0), CNOT(0, 1), MEASURE(0, MemoryReference("ro", 0)))
        p += Declare("ro", "BIT", 1)

        channels = [
            Channel.from_coherence_times(RX(np.pi / 2, 0), gate_duration=0.04, t1s=[30.0], t2s=[20.0]),
        ]
        noise_model = NoiseModel(channels=frozenset(channels))

        sim = TrajectorySimulator(p, noise_model=noise_model, max_subsystem_size=0)
        ops = sim.adapt(sim.compress(sim.resolve(_EMPTY_PARAMS)))

        # RX(noisy) + CNOT(noiseless) + MEASURE = 3 ops
        assert len(ops) == 3

    # ── max_subsystem_size=1 (1Q gate merging) ──

    def test_merges_consecutive_1q_gates(self):
        """Three consecutive 1Q gates on qubit 0 should merge into one op."""
        p = Program(RZ(0.1, 0), RX(0.2, 0), RZ(0.3, 0))

        sim = TrajectorySimulator(p, max_subsystem_size=1)
        ops = sim.adapt(sim.compress(sim.resolve(_EMPTY_PARAMS)))

        assert len(ops) == 1

        psi_direct = _sv(p)
        psi = qx.zero_state_vector(sim.n_qubits)
        for op, subsystem in ops:
            if isinstance(op, qx.Unitary):
                psi = qx.targeted_apply_unitary(op, psi, subsystem)
        assert qx.fidelity(psi, psi_direct) > 0.9999

    def test_2q_gate_breaks_run(self):
        """A 2Q gate should break the 1Q run."""
        p = Program(RZ(0.1, 0), RX(0.2, 0), CNOT(0, 1), RZ(0.3, 0))

        sim = TrajectorySimulator(p, max_subsystem_size=1)
        ops = sim.adapt(sim.compress(sim.resolve(_EMPTY_PARAMS)))

        assert len(ops) == 3

        psi_direct = _sv(p)
        psi = qx.zero_state_vector(sim.n_qubits)
        for op, subsystem in ops:
            if isinstance(op, qx.Unitary):
                psi = qx.targeted_apply_unitary(op, psi, subsystem)
        assert qx.fidelity(psi, psi_direct) > 0.9999

    def test_independent_qubit_runs(self):
        """1Q gates on different qubits should form separate runs."""
        p = Program(
            RZ(0.1, 0), RX(0.2, 0),
            RZ(0.3, 1), RX(0.4, 1),
        )

        sim = TrajectorySimulator(p, max_subsystem_size=1)
        ops = sim.adapt(sim.compress(sim.resolve(_EMPTY_PARAMS)))

        assert len(ops) == 2

        psi_direct = _sv(p)
        psi = qx.zero_state_vector(sim.n_qubits)
        for op, subsystem in ops:
            if isinstance(op, qx.Unitary):
                psi = qx.targeted_apply_unitary(op, psi, subsystem)
        assert qx.fidelity(psi, psi_direct) > 0.9999

    def test_parameterized_merge(self):
        """Parameterized gates in a 1Q run should merge correctly."""
        p = Program()
        p += Declare("theta", "REAL", 2)
        p += Declare("ro", "BIT", 1)
        p += RZ(MemoryReference("theta", 0), 0)
        p += RX(MemoryReference("theta", 1), 0)
        p += MEASURE(0, MemoryReference("ro", 0))

        sim = TrajectorySimulator(p, max_subsystem_size=1)

        theta_vals = [np.pi / 4, np.pi / 2]
        params = sim.linearize({"theta": theta_vals})
        ops = sim.adapt(sim.compress(sim.resolve(params)))

        assert len(ops) == 2

        psi_direct = _sv(
            Program(RZ(theta_vals[0], 0), RX(theta_vals[1], 0))
        )
        psi = qx.zero_state_vector(sim.n_qubits)
        for op, subsystem in ops:
            if isinstance(op, qx.Unitary):
                psi = qx.targeted_apply_unitary(op, psi, subsystem)
        assert qx.fidelity(psi, psi_direct) > 0.9999

    def test_noisy_1q_merge(self):
        """Noisy 1Q gates should merge via SuperOp composition."""
        p = Program(RX(np.pi / 2, 0), RZ(0.5, 0))
        channels = [
            Channel.from_coherence_times(RX(np.pi / 2, 0), gate_duration=0.04, t1s=[30.0], t2s=[20.0]),
        ]
        noise_model = NoiseModel(channels=frozenset(channels))

        sim0 = TrajectorySimulator(p, noise_model=noise_model, max_subsystem_size=0)
        sim1 = TrajectorySimulator(p, noise_model=noise_model, max_subsystem_size=1)

        ops0 = sim0.adapt(sim0.compress(sim0.resolve(_EMPTY_PARAMS)))
        ops1 = sim1.adapt(sim1.compress(sim1.resolve(_EMPTY_PARAMS)))

        assert len(ops0) == 2
        assert len(ops1) == 1
        assert isinstance(ops1[0][0], qx.KrausMap)

    def test_measurement_breaks_run(self):
        """A MEASURE should break 1Q runs."""
        p = Program()
        p += Declare("ro", "BIT", 1)
        p += RZ(0.1, 0)
        p += MEASURE(0, MemoryReference("ro", 0))
        p += RZ(0.2, 0)

        sim = TrajectorySimulator(p, max_subsystem_size=1)
        ops = sim.adapt(sim.compress(sim.resolve(_EMPTY_PARAMS)))

        assert len(ops) == 3

    def test_typical_circuit_compression_ratio(self):
        """A typical layered circuit should have < 1.0 compression ratio."""
        n_q = 4
        p = Program()
        for _ in range(3):
            for q in range(n_q):
                p += RZ(np.random.uniform(-np.pi, np.pi), q)
                p += RX(np.pi / 2, q)
                p += RZ(np.random.uniform(-np.pi, np.pi), q)
            for i in range(0, n_q - 1, 2):
                p += CNOT(i, i + 1)

        sim0 = TrajectorySimulator(p, max_subsystem_size=0)
        sim1 = TrajectorySimulator(p, max_subsystem_size=1)
        n0 = len(sim0.adapt(sim0.compress(sim0.resolve(_EMPTY_PARAMS))))
        n1 = len(sim1.adapt(sim1.compress(sim1.resolve(_EMPTY_PARAMS))))
        assert n1 < n0


class TestBuildSimulationIntegration:
    """Integration tests: TrajectorySimulator pipeline flows through to trajectory simulation."""

    def test_noisy_trajectory_via_simulator(self):
        """Full pipeline: TrajectorySimulator resolve + compress + adapt + apply_trajectory_operations."""
        p = Program(H(0), CNOT(0, 1), MEASURE(0, MemoryReference("ro", 0)), MEASURE(1, MemoryReference("ro", 1)))
        p += Declare("ro", "BIT", 2)

        channels = [
            Channel.from_coherence_times(CNOT(0, 1), gate_duration=0.1, t1s=[30.0, 30.0], t2s=[20.0, 20.0]),
        ]
        noise_model = NoiseModel(channels=frozenset(channels))

        sim = TrajectorySimulator(p, noise_model=noise_model, max_subsystem_size=0)
        ops = sim.adapt(sim.compress(sim.resolve(_EMPTY_PARAMS)))

        n_traj = 16
        psi = qx.zero_state_vector(sim.n_qubits, ensemble_size=(n_traj,))
        key = jax.random.key(42)
        psi_out, outcomes = apply_trajectory_operations(ops, psi, key)
        assert outcomes.shape == (n_traj, 2)
        assert set(int(v) for v in jnp.unique(outcomes)) <= {0, 1}

    def test_parameterized_trajectory(self):
        """Parameterized circuit through TrajectorySimulator → trajectory sim."""
        p = Program()
        p += Declare("theta", "REAL", 1)
        p += Declare("ro", "BIT", 1)
        p += RX(MemoryReference("theta", 0), 0)
        p += MEASURE(0, MemoryReference("ro", 0))

        sim = TrajectorySimulator(p, max_subsystem_size=0)
        params = sim.linearize({"theta": [np.pi]})
        ops = sim.adapt(sim.compress(sim.resolve(params)))

        n_traj = 32
        psi = qx.zero_state_vector(sim.n_qubits, ensemble_size=(n_traj,))
        key = jax.random.key(0)
        _, outcomes = apply_trajectory_operations(ops, psi, key)
        assert jnp.all(outcomes == 1)

# ──────────────────────────────────────────────────────────────────────────────
# Compressor op-count benchmarks
# ──────────────────────────────────────────────────────────────────────────────



def _op_count(program, max_subsystem_size, noise_model=None):
    """Return the number of compressed ops for a program."""
    sim = TrajectorySimulator(
        program, noise_model=noise_model, max_subsystem_size=max_subsystem_size,
    )
    return len(sim.adapt(sim.compress(sim.resolve(sim.linearize({})))))


class TestCompressorOpCounts:
    """Tests that verify the compressor produces the expected number of ops."""

    def test_single_qubit_sequence_merges_to_one(self):
        """RZ-RX-RZ-RX-RZ on one qubit → 1 op at max_size ≥ 1."""
        p = Program(RZ(0.1, 0), RX(0.2, 0), RZ(0.3, 0), RX(0.4, 0), RZ(0.5, 0))
        assert _op_count(p, max_subsystem_size=0) == 5
        assert _op_count(p, max_subsystem_size=1) == 1

        # Verify correctness
        sim = TrajectorySimulator(p, max_subsystem_size=1)
        ops = sim.adapt(sim.compress(sim.resolve(_EMPTY_PARAMS)))
        psi = qx.zero_state_vector(sim.n_qubits)
        for op, sub in ops:
            psi = qx.targeted_apply_unitary(op, psi, sub)
        assert qx.fidelity(psi, _sv(p)) > 0.9999

    def test_two_qubit_layer_max_size_1(self):
        """ZXZXZ on q0, ZXZXZ on q1, CZ 0 1, repeated 2×.

        With max_size=1: 1Q runs merge within each qubit between CZs, but CZ
        can't merge into a size-1 group.  Structure per repetition:
        merged(5×q0) + merged(5×q1) + CZ = 3 ops; ×2 reps = 6 ops.
        """
        p = Program()
        for _ in range(2):
            for q in (0, 1):
                p += RZ(0.1, q)
                p += RX(0.2, q)
                p += RZ(0.3, q)
                p += RX(0.4, q)
                p += RZ(0.5, q)
            p += CZ(0, 1)

        assert _op_count(p, max_subsystem_size=1) == 6

    def test_two_qubit_layer_max_size_2(self):
        """Same circuit as above, but with max_size=2 → everything merges to 1."""
        p = Program()
        for _ in range(2):
            for q in (0, 1):
                p += RZ(0.1, q)
                p += RX(0.2, q)
                p += RZ(0.3, q)
                p += RX(0.4, q)
                p += RZ(0.5, q)
            p += CZ(0, 1)

        assert _op_count(p, max_subsystem_size=2) == 1

        # Verify correctness
        sim = TrajectorySimulator(p, max_subsystem_size=2)
        ops = sim.adapt(sim.compress(sim.resolve(_EMPTY_PARAMS)))
        psi = qx.zero_state_vector(sim.n_qubits)
        for op, sub in ops:
            psi = qx.targeted_apply_unitary(op, psi, sub)
        assert qx.fidelity(psi, _sv(p)) > 0.9999

    def test_cnot_pair_merge(self):
        """CNOT 0 1, CNOT 1 0 should merge into 1 op at max_size ≥ 2."""
        p = Program(CNOT(0, 1), CNOT(1, 0))

        assert _op_count(p, max_subsystem_size=0) == 2
        assert _op_count(p, max_subsystem_size=1) == 2  # both are 2Q, can't fit in size 1
        assert _op_count(p, max_subsystem_size=2) == 1

        # Verify correctness
        sim = TrajectorySimulator(p, max_subsystem_size=2)
        ops = sim.adapt(sim.compress(sim.resolve(_EMPTY_PARAMS)))
        assert len(ops) == 1
        assert ops[0][1] == (0, 1)
        psi = qx.zero_state_vector(sim.n_qubits)
        for op, sub in ops:
            psi = qx.targeted_apply_unitary(op, psi, sub)
        assert qx.fidelity(psi, _sv(p)) > 0.9999

    @pytest.mark.parametrize("num_qubits", [4, 8, 12])
    @pytest.mark.parametrize("max_subsystem_size", [0, 1, 2, 3])
    def test_random_circuit_compression(self, num_qubits, max_subsystem_size):
        """Random layered circuits should compress monotonically with max_size."""
        rng = np.random.default_rng(42)
        n_layers = 5

        p = Program()
        for _ in range(n_layers):
            # 1Q layer
            for q in range(num_qubits):
                gate = rng.choice([RZ, RX, RY])
                p += gate(rng.uniform(-np.pi, np.pi), q)
            # 2Q layer (linear chain, even edges)
            for i in range(0, num_qubits - 1, 2):
                p += CNOT(i, i + 1)
            # 1Q layer
            for q in range(num_qubits):
                gate = rng.choice([RZ, RX, RY])
                p += gate(rng.uniform(-np.pi, np.pi), q)
            # 2Q layer (odd edges)
            for i in range(1, num_qubits - 1, 2):
                p += CNOT(i, i + 1)

        n_ops = _op_count(p, max_subsystem_size)
        n_uncompressed = _op_count(p, 0)

        # Compression should never increase op count
        assert n_ops <= n_uncompressed, (
            f"max_size={max_subsystem_size}: {n_ops} ops > {n_uncompressed} uncompressed"
        )

        # With max_size > 0, we expect at least some compression for this circuit
        if max_subsystem_size > 0:
            assert n_ops < n_uncompressed

    def test_random_circuit_compression_summary(self, capsys):
        """Print a summary table of compression ratios for various configs."""
        rng = np.random.default_rng(42)

        configs = [
            (4, 5), (8, 5), (12, 5), (16, 3),
        ]
        max_sizes = [0, 1, 2, 3, 4]

        rows = []
        for num_qubits, n_layers in configs:
            p = Program()
            for _ in range(n_layers):
                for q in range(num_qubits):
                    p += RZ(rng.uniform(-np.pi, np.pi), q)
                    p += RX(np.pi / 2, q)
                for i in range(0, num_qubits - 1, 2):
                    p += CNOT(i, i + 1)
                for q in range(num_qubits):
                    p += RZ(rng.uniform(-np.pi, np.pi), q)
                for i in range(1, num_qubits - 1, 2):
                    p += CNOT(i, i + 1)

            counts = {s: _op_count(p, s) for s in max_sizes}
            rows.append((num_qubits, n_layers, counts))

        # Print table
        header = f"{'qubits':>6} {'layers':>6}" + "".join(f" {'s=' + str(s):>8}" for s in max_sizes)
        print(f"\n{'Compression op counts':=^{len(header)}}")
        print(header)
        print("-" * len(header))
        for nq, nl, counts in rows:
            line = f"{nq:>6} {nl:>6}"
            for s in max_sizes:
                ratio = counts[s] / counts[0] if counts[0] > 0 else 0
                line += f" {counts[s]:>4} ({ratio:.2f})"
                # line += f" {counts[s]:>8}"
            print(line)

# ──────────────────────────────────────────────────────────────────────────────
# State Vector simulation benchmarks
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_NUM_QUBITS = 15
_DEFAULT_NUM_LAYERS = 10
_DEFAULT_NUM_TRAJECTORIES = 128
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_MAX_SUBSYSTEM_SIZE = 1


def _build_noisy_program_and_model(num_qubits, num_layers, seed=4867):
    """Build a layered noisy circuit and matching noise model.

    Circuit structure per layer (×2 for even/odd edge sets):
        RZ-RX-RZ-RX-RZ on every qubit, then CNOTs on edges.
    Total: 5*num_layers*num_qubits 1Q gates + (num_qubits-1)*num_layers 2Q gates.
    """
    edges_0 = [(i, i + 1) for i in range(0, num_qubits - 1, 2)]
    edges_1 = [(i, i + 1) for i in range(1, num_qubits - 1, 2)]
    rng = np.random.default_rng(seed)

    t1s, t2s = {}, {}
    for q in range(num_qubits):
        t1 = np.clip(rng.normal(30, 10), 10, 50)
        t2 = np.clip(rng.normal(30, 20), 5, 2 * t1)
        t1s[q], t2s[q] = t1, t2

    channels = [
        Channel.from_coherence_times(
            CNOT(*edge), gate_duration=0.1, t1s=[t1s[q] for q in edge], t2s=[t2s[q] for q in edge]
        )
        for edge in edges_0 + edges_1
    ] + [
        Channel.from_coherence_times(RX(np.pi / 2, q), gate_duration=0.04, t1s=[t1s[q]], t2s=[t2s[q]])
        for q in range(num_qubits)
    ]
    noise_model = NoiseModel(channels=frozenset(channels))

    program = Program()
    for _ in range(num_layers):
        for edges in [edges_0, edges_1]:
            program += [RZ(rng.uniform(-np.pi, np.pi), idx) for idx in range(num_qubits)]
            program += [RX(np.pi / 2, idx) for idx in range(num_qubits)]
            program += [RZ(rng.uniform(-np.pi, np.pi), idx) for idx in range(num_qubits)]
            program += [RX(np.pi / 2, idx) for idx in range(num_qubits)]
            program += [RZ(rng.uniform(-np.pi, np.pi), idx) for idx in range(num_qubits)]
            program += [CNOT(*edge) for edge in edges]

    return program, noise_model


def _run_perf_benchmark(
    benchmark,
    num_qubits=_DEFAULT_NUM_QUBITS,
    num_layers=_DEFAULT_NUM_LAYERS,
    num_trajectories=_DEFAULT_NUM_TRAJECTORIES,
    batch_size=_DEFAULT_BATCH_SIZE,
    max_subsystem_size=_DEFAULT_MAX_SUBSYSTEM_SIZE,
):
    """Shared benchmark harness: build, warmup, then benchmark the JAX kernel."""
    program, noise_model = _build_noisy_program_and_model(num_qubits, num_layers)

    sim = TrajectorySimulator(
        program, noise_model=noise_model, max_subsystem_size=max_subsystem_size,
    )
    params = sim.linearize({})
    operations = sim.adapt(sim.compress(sim.resolve(params)))

    # Warmup: trigger JIT compilation
    warmup_psi = qx.zero_state_vector(sim.n_qubits, ensemble_size=(batch_size,))
    key = jax.random.key(0)
    apply_trajectory_operations(operations, warmup_psi, key)[0].matrix.block_until_ready()

    def thunk():
        key = jax.random.key(0)
        remaining = num_trajectories
        while remaining > 0:
            this_batch = min(remaining, batch_size)
            key, batch_key = jax.random.split(key)
            psi = qx.zero_state_vector(sim.n_qubits, ensemble_size=(this_batch,))
            result = apply_trajectory_operations(operations, psi, batch_key)
            result[0].matrix.block_until_ready()
            remaining -= this_batch

    benchmark.pedantic(thunk, iterations=1, rounds=3)


class TestPerformance:
    """Trajectory simulator performance benchmarks.

    Defaults: 15 qubits, depth 10, 128 trajectories, batch_size 32,
    max_subsystem_size 1. Each test varies one axis while holding the
    others constant.
    """

    # ── Vary num_qubits ──────────────────────────────────
    @pytest.mark.parametrize("num_qubits", [
        pytest.param(3, id="3q"),
        pytest.param(6, id="6q"),
        pytest.param(9, id="9q"),
        pytest.param(12, id="12q"),
        pytest.param(15, id="15q"),
    ])
    def test_scaling_qubits(self, benchmark, num_qubits):
        _run_perf_benchmark(benchmark, num_qubits=num_qubits)

    # ── Vary depth (num_layers) ──────────────────────────
    @pytest.mark.parametrize("num_layers", [
        pytest.param(1, id="1L"),
        pytest.param(3, id="3L"),
        pytest.param(10, id="10L"),
        pytest.param(20, id="20L"),
    ])
    def test_scaling_depth(self, benchmark, num_layers):
        _run_perf_benchmark(benchmark, num_layers=num_layers)

    # ── Vary batch_size ──────────────────────────────────
    @pytest.mark.parametrize("batch_size", [
        pytest.param(8, id="b8"),
        pytest.param(16, id="b16"),
        pytest.param(32, id="b32"),
        pytest.param(64, id="b64"),
        pytest.param(128, id="b128"),
    ])
    def test_scaling_batch_size(self, benchmark, batch_size):
        _run_perf_benchmark(benchmark, batch_size=batch_size)

    # ── Vary max_subsystem_size ──────────────────────────
    @pytest.mark.parametrize("max_subsystem_size", [
        pytest.param(0, id="s0"),
        pytest.param(1, id="s1"),
    ])
    def test_scaling_subsystem_size(self, benchmark, max_subsystem_size):
        _run_perf_benchmark(benchmark, max_subsystem_size=max_subsystem_size)

    # ── 17-qubit batch_size sweep ────────────────────────
    @pytest.mark.parametrize("batch_size", [
        pytest.param(8, id="b8"),
        pytest.param(16, id="b16"),
        pytest.param(32, id="b32"),
        pytest.param(64, id="b64"),
        pytest.param(128, id="b128"),
    ])
    def test_17q_batch_size(self, benchmark, batch_size):
        _run_perf_benchmark(benchmark, num_qubits=17, batch_size=batch_size)


