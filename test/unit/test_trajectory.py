# Copyright 2026 Rigetti Computing
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

"""Unit tests for the quax-based Monte-Carlo trajectory simulator.

Covers noiseless equivalence with the pure-state simulator, noise statistics, measurement and
reset semantics, batching, multi-device sampling, and the compression behaviour seen through
the trajectory simulator's ``resolve``/``compress``/``adapt`` pipeline.  Compression
*correctness* against an exact oracle lives in ``test_trajectory_compression.py``.

.. note::
    The simulators are **big-endian**: ``qubits[0]`` is the most significant subsystem.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import quax as qx

from pyquil.gates import CNOT, CPHASE, CZ, MEASURE, RESET, RX, RY, RZ, H, I, X
from pyquil.noise._channels import Channel, CycleChannel, MeasurementChannel, SuperopChannel, SuperopResetChannel
from pyquil.noise._noise_model import NoiseModel
from pyquil.quil import Program
from pyquil.quilatom import FormalArgument, MemoryReference, Qubit
from pyquil.quilbase import Declare, DefCircuit, ResetQubit
from pyquil.quilbase import Gate as QuilGate
from pyquil.quilbase import Measurement as QuilMeasurement
from pyquil.simulation._simulator import (
    PureStateVectorSimulator,
    TrajectorySimulator,
    _KrausStackLayout,
    _op_to_kraus_matrix,
    _trajectory_keys,
)
from test.unit.simulation_programs import (
    apply_trajectory_operations,
    simulate_state_vector,
    simulate_trajectories as _simulate_trajectories,
)

_sv = simulate_state_vector


# Trajectory simulator tests
# ──────────────────────────────────────────────────────────────────────────────


class TestTrajectoryNoiseless:
    """Test that the trajectory simulator preserves noiseless behavior."""

    def test_single_gate_noiseless(self):
        """Without noise, trajectory simulation matches unitary simulation."""
        p = Program(H(0))
        psi_noiseless = _sv(p, qubits=[0])
        psi_traj, outcomes = _simulate_trajectories(p, noise_model=None, qubits=[0], num_trajectories=1)
        assert qx.fidelity(psi_noiseless, psi_traj) > 0.9999

    def test_bell_state_noiseless(self):
        """Multi-qubit noiseless trajectory."""
        p = Program(H(0), CNOT(0, 1))
        psi_noiseless = _sv(p, qubits=[0, 1])
        psi_traj, outcomes = _simulate_trajectories(p, noise_model=None, qubits=[0, 1], num_trajectories=1)
        assert qx.fidelity(psi_noiseless, psi_traj) > 0.9999

    def test_multiple_trajectories_noiseless_deterministic(self):
        """Multiple noiseless trajectories should all give same result."""
        p = Program(X(0))
        psi_batch, outcomes = _simulate_trajectories(p, noise_model=None, qubits=[0], num_trajectories=8)
        # Each trajectory should be |1⟩
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
        # Build noisy superop: (1-p)|U><U| + p|XU><XU| in superop picture
        # Simpler: compose depolarizing-like channel with the gate
        # Use a Pauli channel: p_I = 1-p, p_X = p, p_Y = 0, p_Z = 0
        pauli_probs = {"X": p_error}
        channel = SuperopChannel.from_pauli_noise(inst=inst, pauli_noise=pauli_probs)
        return NoiseModel.from_channels([channel])

    def _make_depolarizing_noise_model(self, fidelity: float, qubit: int = 0) -> NoiseModel:
        """Create a noise model with depolarizing noise on X gate."""
        inst = X(qubit)
        channel = Channel.from_gate_fidelity(inst=inst, fidelity=fidelity)
        return NoiseModel.from_channels([channel])

    def test_noiseless_gate_with_noise_model(self):
        """A noise model that doesn't cover the applied gate should leave it noiseless."""
        # Noise model only covers X gate, but we apply H
        noise_model = self._make_bitflip_noise_model(0.1, qubit=0)
        p = Program(H(0))
        psi, outcomes = _simulate_trajectories(p, noise_model=noise_model, qubits=[0], num_trajectories=1)
        target = qx.StateVector.from_matrix(jnp.array([1.0, 1.0], dtype=complex) / jnp.sqrt(2), dims=(2,))
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
            p,
            noise_model=noise_model,
            qubits=[0],
            num_trajectories=num_traj,
            seed=42,
        )
        # Get probabilities for each trajectory
        probs = qx.probabilities(psi_batch)  # shape (num_traj, 2)
        # Each trajectory should collapse to either |0⟩ or |1⟩
        # Count how many ended in |0⟩ (bit-flipped from |1⟩)
        in_zero = jnp.sum(probs[:, 0] > 0.5)
        observed_flip_rate = float(in_zero) / num_traj
        # Expected: p_error fraction should flip to |0⟩
        assert abs(observed_flip_rate - p_error) < 0.05, f"Expected flip rate ~{p_error}, got {observed_flip_rate}"

    def test_depolarizing_statistics(self):
        """Depolarizing noise on identity-like circuit should produce mixed results."""
        fidelity_val = 0.9
        noise_model = self._make_depolarizing_noise_model(fidelity_val, qubit=0)
        p = Program(X(0))
        num_traj = 2048
        psi_batch, outcomes = _simulate_trajectories(
            p,
            noise_model=noise_model,
            qubits=[0],
            num_trajectories=num_traj,
            seed=123,
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
        ch0 = SuperopChannel.from_pauli_noise(inst=inst_q0, pauli_noise={"X": p_error})
        ch1 = SuperopChannel.from_pauli_noise(inst=inst_q1, pauli_noise={"X": p_error})
        noise_model = NoiseModel.from_channels([ch0, ch1])

        prog = Program(X(0), X(1))
        num_traj = 2048
        psi_batch, _ = _simulate_trajectories(
            prog,
            noise_model=noise_model,
            qubits=[0, 1],
            num_trajectories=num_traj,
            seed=7,
        )
        probs = qx.probabilities(psi_batch)  # shape (num_traj, 4)
        # State |11⟩ = index 3. Both flipped: p_error^2 gives |00⟩
        # Expected: P(|11⟩) ≈ (1-p)^2, P(|00⟩) ≈ p^2
        avg_prob_11 = float(jnp.mean(probs[:, 3]))
        expected_prob_11 = (1 - p_error) ** 2
        assert abs(avg_prob_11 - expected_prob_11) < 0.05


class TestCycleChannelGateApplication:
    """A ``CycleChannel`` gate constituent is applied as its ``process``.

    ``Channel.process`` carries the gate (composed with any noise); the resolver
    expands a cycle by applying each constituent's ``process`` verbatim.  This
    guards, end to end, that a cycle gate whose unitary lives in its ``process``
    actually acts on the state — the surface-code kraus/stim mismatch was caused by
    a noise model that put an identity in ``process`` for a real gate, dropping it,
    which is a noise-model error rather than a simulator one.
    """

    def test_trajectory_applies_cycle_gate(self):
        """The cycle's ``X`` (carried in the channel's process) flips the qubit."""
        q = FormalArgument("q")
        dc = DefCircuit("CX", [], [q], [X(q)])
        cyc = QuilGate("CX", [], [Qubit(0)])
        program = Program(dc, cyc, Declare("ro", "BIT", 1), QuilMeasurement(Qubit(0), MemoryReference("ro", 0)))
        gate_channel = SuperopChannel(inst=X(0), process=qx.to_superop(qx.gates.X), ideal_unitary=qx.gates.X)
        nm = NoiseModel.from_channels([CycleChannel(inst=cyc, defcircuit=dc, channels=(gate_channel,))])
        sim = TrajectorySimulator(program, noise_model=nm, qubits=[0])
        outcomes = np.asarray(sim.sample(sim.linearize({}), num_trajectories=16, batch_size=16, key=jax.random.key(0)))
        assert outcomes.shape == (16, 1)
        assert np.all(outcomes == 1)


class TestTrajectoryMeasurement:
    """Test mid-circuit measurement in trajectory simulation."""

    def test_measurement_records_outcome(self):
        """Measurement should record classical outcome."""
        p = Program(H(0), MEASURE(0, None))
        psi, outcomes = _simulate_trajectories(
            p,
            noise_model=None,
            qubits=[0],
            num_trajectories=100,
            seed=42,
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
            p,
            noise_model=None,
            qubits=[0],
            num_trajectories=64,
            seed=99,
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
        noise_model = NoiseModel.from_channels([meas_ch])

        p = Program(MEASURE(0, None))
        psi, outcomes = _simulate_trajectories(
            p,
            noise_model=noise_model,
            qubits=[0],
            num_trajectories=1024,
            seed=55,
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
            p,
            noise_model=None,
            qubits=[0],
            num_trajectories=1,
        )
        target = qx.StateVector.from_matrix(jnp.array([1.0, 0.0], dtype=complex), dims=(2,))
        assert qx.fidelity(psi, target) > 0.9999

    def test_global_reset(self):
        """Global RESET should reset all qubits."""
        p = Program(X(0), X(1), RESET())
        psi, _ = _simulate_trajectories(
            p,
            noise_model=None,
            qubits=[0, 1],
            num_trajectories=1,
        )
        target = qx.StateVector.from_matrix(jnp.array([1.0, 0.0, 0.0, 0.0], dtype=complex), dims=(2, 2))
        assert qx.fidelity(psi, target) > 0.9999

    def test_noisy_reset(self):
        """Noisy reset should have imperfect fidelity."""
        qubit = Qubit(0)
        reset_inst = ResetQubit(qubit)
        reset_ch = SuperopResetChannel.from_reset_fidelity(inst=reset_inst, fidelity=0.9)
        noise_model = NoiseModel.from_channels([reset_ch])

        # Start in |1⟩, apply noisy reset
        p = Program(X(0), ResetQubit(Qubit(0)))
        num_traj = 2048
        psi, _ = _simulate_trajectories(
            p,
            noise_model=noise_model,
            qubits=[0],
            num_trajectories=num_traj,
            seed=13,
        )
        probs = qx.probabilities(psi)  # (num_traj, 2)
        # With 90% reset fidelity, ~90% should end in |0⟩
        avg_prob_0 = float(jnp.mean(probs[:, 0]))
        assert avg_prob_0 > 0.85


class TestTrajectoryBatching:
    """Test that batch processing works correctly."""

    def test_compute_is_reproducible_at_a_fixed_key(self):
        """The same keys give the same trajectories, states included."""
        p = Program(H(0), MEASURE(0, None))
        sim = TrajectorySimulator(p, qubits=[0])
        keys = jax.random.split(jax.random.key(42), 64)

        psi_1, outcomes_1 = sim.compute(None, keys)
        psi_2, outcomes_2 = sim.compute(None, keys)

        assert np.array_equal(outcomes_1, outcomes_2)
        assert jnp.array_equal(psi_1.matrix, psi_2.matrix)

    def test_trajectory_keys_are_split_invariant(self):
        """Keys come from the global trajectory index, so slicing the range composes.

        This is the property that makes ``sample`` independent of ``batch_size`` and of the
        device count: trajectory *g* gets ``fold_in(key, g)`` however the work is divided.
        """
        key = jax.random.key(11)
        whole = jax.random.key_data(_trajectory_keys(key, 0, 8))
        halves = jnp.concatenate(
            [jax.random.key_data(_trajectory_keys(key, 0, 4)), jax.random.key_data(_trajectory_keys(key, 4, 4))]
        )

        assert jnp.array_equal(whole, halves)


class TestComputeProgramStateVectorWithNoise:
    """Test the TrajectorySimulator with noise_model parameter."""

    def test_noise_model_none_unchanged(self):
        """With noise_model=None, behavior is identical to original."""
        p = Program(H(0), CNOT(0, 1))
        psi = _sv(p, qubits=[0, 1])
        target = qx.StateVector.from_matrix(jnp.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / jnp.sqrt(2), dims=(2, 2))
        assert qx.fidelity(psi, target) > 0.9999

    def test_noise_model_single_trajectory(self):
        """With noise_model provided, runs a single trajectory."""
        inst = X(0)
        channel = Channel.from_gate_fidelity(inst=inst, fidelity=1.0)
        noise_model = NoiseModel.from_channels([channel])
        p = Program(X(0))
        sim = TrajectorySimulator(p, noise_model=noise_model, qubits=[0])
        psi, _ = sim.compute(None, jax.random.key(0))
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
            None,
            num_trajectories=100,
            key=jax.random.key(42),
        )
        assert outcomes.shape == (100, 1)
        assert jnp.all((outcomes == 0) | (outcomes == 1))

    def test_no_measurements_empty_outcomes(self):
        """Without measurements, outcomes array should have zero columns."""
        p = Program(H(0))
        sim = TrajectorySimulator(p, noise_model=None, qubits=[0])
        outcomes = sim.sample(
            None,
            num_trajectories=10,
        )
        assert outcomes.shape == (10, 0)

    def test_bitflip_statistics(self):
        """Outcome statistics should match noise model."""
        p_error = 0.3
        inst = X(0)
        ch = SuperopChannel.from_pauli_noise(inst=inst, pauli_noise={"X": p_error})
        noise_model = NoiseModel.from_channels([ch])

        p = Program(X(0), MEASURE(0, None))
        sim = TrajectorySimulator(p, noise_model=noise_model, qubits=[0])
        outcomes = sim.sample(
            None,
            num_trajectories=2048,
            key=jax.random.key(42),
        )
        # X|0⟩ = |1⟩, then bit-flip with p=0.3 → ~30% get |0⟩
        # Measurement outcome reflects the final state
        frac_0 = float(jnp.mean(outcomes == 0))
        assert abs(frac_0 - p_error) < 0.05

    def test_batch_size_does_not_affect_results(self):
        """``batch_size`` is a memory knob: at the same key the shots are *identical*.

        Each trajectory's key comes from its global index, not from a per-batch split, so how
        the run was divided into calls cannot reach the outcomes.
        """
        p = Program(H(0), MEASURE(0, None))
        sim = TrajectorySimulator(p, qubits=[0])
        key = jax.random.key(3)

        outcomes_small = sim.sample(None, num_trajectories=100, key=key, batch_size=10)
        outcomes_large = sim.sample(None, num_trajectories=100, key=key, batch_size=100)

        assert outcomes_small.shape == outcomes_large.shape == (100, 1)
        assert np.array_equal(outcomes_small, outcomes_large)

    def test_sample_matches_a_batched_compute(self):
        """``sample`` is ``compute`` over the same per-trajectory keys, with the states dropped."""
        p = Program(H(0), MEASURE(0, None))
        sim = TrajectorySimulator(p, qubits=[0])
        key = jax.random.key(5)

        sampled = sim.sample(None, num_trajectories=32, key=key, batch_size=8)
        _, computed = sim.compute(None, _trajectory_keys(key, 0, 32))

        assert np.array_equal(sampled, computed)

    def test_default_key_is_fresh(self):
        """Omitting the key draws new entropy, so repeated calls accumulate samples."""
        p = Program(H(0), MEASURE(0, None))
        sim = TrajectorySimulator(p, qubits=[0])

        assert not np.array_equal(sim.sample(None, num_trajectories=200), sim.sample(None, num_trajectories=200))


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
        assert len(sim.qubits) == 2

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

    def test_no_merge_pure_state_compute_matches_compressed(self):
        """PureStateVectorSimulator should support disabling compression."""
        p = Program(H(0), CNOT(0, 1), RZ(0.5, 0))
        compressed = PureStateVectorSimulator(p)
        uncompressed = PureStateVectorSimulator(p, max_subsystem_size=0)

        psi_compressed = compressed.compute()
        psi_uncompressed = uncompressed.compute()

        assert qx.fidelity(psi_uncompressed, psi_compressed) > 0.9999

    def test_parametric_non_contiguous_gate_embeds_in_larger_group(self):
        """Parametric multi-qubit gates should embed correctly into larger groups."""
        p = Program()
        p += Declare("theta", "REAL", 1)
        p += H(0)
        p += H(1)
        p += H(2)
        p += CPHASE(MemoryReference("theta", 0), 0, 2)
        p += CPHASE(0.37, 0, 1)

        compressed = PureStateVectorSimulator(p, qubits=[0, 1, 2], max_subsystem_size=3)
        uncompressed = PureStateVectorSimulator(p, qubits=[0, 1, 2], max_subsystem_size=0)
        params = compressed.linearize({"theta": [0.91]})

        assert compressed.bases == ((0, 1, 2),)
        assert qx.fidelity(compressed.compute(params), uncompressed.compute(params)) > 0.9999

    def test_no_merge_noiseless_matches_direct(self):
        """max_subsystem_size=0 compressor output should match direct computation."""
        p = Program(H(0), CNOT(0, 1), RZ(0.5, 0))
        psi_direct = _sv(p)

        sim = TrajectorySimulator(p, max_subsystem_size=0)
        ops = sim.adapt(sim.compress(sim.resolve()))

        psi = qx.zero_state_vector(dims=sim.dims)
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

        psi = qx.zero_state_vector(dims=sim.dims)
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
        noise_model = NoiseModel.from_channels(channels)

        sim = TrajectorySimulator(p, noise_model=noise_model, max_subsystem_size=0)
        ops = sim.adapt(sim.compress(sim.resolve()))

        # RX(noisy) + CNOT(noiseless) + MEASURE = 3 ops
        assert len(ops) == 3

    # ── max_subsystem_size=1 (1Q gate merging) ──

    def test_merges_consecutive_1q_gates(self):
        """Three consecutive 1Q gates on qubit 0 should merge into one op."""
        p = Program(RZ(0.1, 0), RX(0.2, 0), RZ(0.3, 0))

        sim = TrajectorySimulator(p, max_subsystem_size=1)
        ops = sim.adapt(sim.compress(sim.resolve()))

        assert len(ops) == 1

        psi_direct = _sv(p)
        psi = qx.zero_state_vector(dims=sim.dims)
        for op, subsystem in ops:
            if isinstance(op, qx.Unitary):
                psi = qx.targeted_apply_unitary(op, psi, subsystem)
        assert qx.fidelity(psi, psi_direct) > 0.9999

    def test_2q_gate_breaks_run(self):
        """A 2Q gate should break the 1Q run."""
        p = Program(RZ(0.1, 0), RX(0.2, 0), CNOT(0, 1), RZ(0.3, 0))

        sim = TrajectorySimulator(p, max_subsystem_size=1)
        ops = sim.adapt(sim.compress(sim.resolve()))

        assert len(ops) == 3

        psi_direct = _sv(p)
        psi = qx.zero_state_vector(dims=sim.dims)
        for op, subsystem in ops:
            if isinstance(op, qx.Unitary):
                psi = qx.targeted_apply_unitary(op, psi, subsystem)
        assert qx.fidelity(psi, psi_direct) > 0.9999

    def test_independent_qubit_runs(self):
        """1Q gates on different qubits should form separate runs."""
        p = Program(
            RZ(0.1, 0),
            RX(0.2, 0),
            RZ(0.3, 1),
            RX(0.4, 1),
        )

        sim = TrajectorySimulator(p, max_subsystem_size=1)
        ops = sim.adapt(sim.compress(sim.resolve()))

        assert len(ops) == 2

        psi_direct = _sv(p)
        psi = qx.zero_state_vector(dims=sim.dims)
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

        psi_direct = _sv(Program(RZ(theta_vals[0], 0), RX(theta_vals[1], 0)))
        psi = qx.zero_state_vector(dims=sim.dims)
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
        noise_model = NoiseModel.from_channels(channels)

        sim0 = TrajectorySimulator(p, noise_model=noise_model, max_subsystem_size=0)
        sim1 = TrajectorySimulator(p, noise_model=noise_model, max_subsystem_size=1)

        ops0 = sim0.adapt(sim0.compress(sim0.resolve()))
        ops1 = sim1.adapt(sim1.compress(sim1.resolve()))

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
        ops = sim.adapt(sim.compress(sim.resolve()))

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
        n0 = len(sim0.adapt(sim0.compress(sim0.resolve())))
        n1 = len(sim1.adapt(sim1.compress(sim1.resolve())))
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
        noise_model = NoiseModel.from_channels(channels)

        sim = TrajectorySimulator(p, noise_model=noise_model, max_subsystem_size=0)
        ops = sim.adapt(sim.compress(sim.resolve()))

        n_traj = 16
        psi = qx.zero_state_vector(dims=sim.dims, ensemble_size=(n_traj,))
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
        psi = qx.zero_state_vector(dims=sim.dims, ensemble_size=(n_traj,))
        key = jax.random.key(0)
        _, outcomes = apply_trajectory_operations(ops, psi, key)
        assert jnp.all(outcomes == 1)


# ──────────────────────────────────────────────────────────────────────────────
# Compressor op-count tests
# ──────────────────────────────────────────────────────────────────────────────


def _op_count(program, max_subsystem_size, noise_model=None):
    """Return the number of compressed ops for a program."""
    sim = TrajectorySimulator(
        program,
        noise_model=noise_model,
        max_subsystem_size=max_subsystem_size,
    )
    return len(sim.adapt(sim.compress(sim.resolve(sim.linearize({})))))


class TestCompressorOpCounts:
    """Tests that verify the compressor produces the expected number of ops."""

    def test_cycle_channel_expands_and_compresses(self):
        formal_qubit = FormalArgument("q")
        defcircuit = DefCircuit(
            "SINGLE_QUBIT_CYCLE",
            [],
            [formal_qubit],
            [RX(0.1, formal_qubit), RZ(0.2, formal_qubit), RX(0.3, formal_qubit)],
        )
        cycle_inst = QuilGate("SINGLE_QUBIT_CYCLE", [], [0])
        channels = tuple(
            Channel.from_depolarizing_constant(inst, depolarizing_constant=0.99)
            for inst in (RX(0.1, 0), RZ(0.2, 0), RX(0.3, 0))
        )
        noise_model = NoiseModel.from_channels(
            [CycleChannel(inst=cycle_inst, defcircuit=defcircuit, channels=channels)]
        )
        program = Program(defcircuit, cycle_inst)

        sim = TrajectorySimulator(program, noise_model=noise_model, max_subsystem_size=1)
        resolved = sim.resolve()
        compressed = sim.compress(resolved)

        assert len(resolved) == 3
        assert all(isinstance(op, qx.SuperOp) for op, _ in resolved)
        assert len(compressed) == 1

    def test_expanded_cycle_without_cycle_channel_uses_gate_channels(self):
        formal_qubit = FormalArgument("q")
        defcircuit = DefCircuit(
            "INDIVIDUAL_NOISE_CYCLE",
            [],
            [formal_qubit],
            [RX(0.1, formal_qubit), RZ(0.2, formal_qubit)],
        )
        cycle_inst = QuilGate("INDIVIDUAL_NOISE_CYCLE", [], [0])
        noise_model = NoiseModel.from_channels([Channel.from_depolarizing_constant(RX(0.1, 0), 0.99)])
        program = Program(defcircuit, cycle_inst)

        sim = TrajectorySimulator(program, noise_model=noise_model, max_subsystem_size=0)
        resolved = sim.resolve()

        assert len(resolved) == 2
        assert isinstance(resolved[0][0], qx.SuperOp)
        assert isinstance(resolved[1][0], qx.Unitary)

    def test_single_qubit_sequence_merges_to_one(self):
        """RZ-RX-RZ-RX-RZ on one qubit → 1 op at max_size ≥ 1."""
        p = Program(RZ(0.1, 0), RX(0.2, 0), RZ(0.3, 0), RX(0.4, 0), RZ(0.5, 0))
        assert _op_count(p, max_subsystem_size=0) == 5
        assert _op_count(p, max_subsystem_size=1) == 1

        # Verify correctness
        sim = TrajectorySimulator(p, max_subsystem_size=1)
        ops = sim.adapt(sim.compress(sim.resolve()))
        psi = qx.zero_state_vector(dims=sim.dims)
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
        ops = sim.adapt(sim.compress(sim.resolve()))
        psi = qx.zero_state_vector(dims=sim.dims)
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
        ops = sim.adapt(sim.compress(sim.resolve()))
        assert len(ops) == 1
        assert ops[0][1] == (0, 1)
        psi = qx.zero_state_vector(dims=sim.dims)
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
        assert n_ops <= n_uncompressed, f"max_size={max_subsystem_size}: {n_ops} ops > {n_uncompressed} uncompressed"

        # With max_size > 0, we expect at least some compression for this circuit
        if max_subsystem_size > 0:
            assert n_ops < n_uncompressed

    def test_random_circuit_compression_summary(self, capsys):
        """Print a summary table of compression ratios for various configs."""
        rng = np.random.default_rng(42)

        configs = [
            (4, 5),
            (8, 5),
            (12, 5),
            (16, 3),
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
# Multi-device / sharding tests
# ──────────────────────────────────────────────────────────────────────────────


class TestMultiDeviceTrajectory:
    """Tests that exercise the multi-device code paths.

    On a single-CPU host these still validate the padding/unpadding logic
    and the ``devices`` parameter plumbing.  On a multi-device host they
    exercise real data-parallel ``jax.pmap`` execution (one replica per device).
    """

    def test_devices_parameter_accepted(self):
        """TrajectorySimulator should accept a ``devices`` keyword."""
        p = Program(H(0), MEASURE(0, None))
        sim = TrajectorySimulator(p, qubits=[0], devices=jax.devices())
        outcomes = sim.sample(None, num_trajectories=10)
        assert outcomes.shape == (10, 1)

    def test_sample_results_match_single_device(self):
        """Outcomes shape and value range must be the same regardless of device list."""
        p = Program(H(0), MEASURE(0, None))
        sim_default = TrajectorySimulator(p, qubits=[0])
        sim_explicit = TrajectorySimulator(p, qubits=[0], devices=jax.devices())

        out_default = sim_default.sample(None, num_trajectories=64, batch_size=16, key=jax.random.key(99))
        out_explicit = sim_explicit.sample(None, num_trajectories=64, batch_size=16, key=jax.random.key(99))

        assert out_default.shape == out_explicit.shape
        assert jnp.all((out_default == 0) | (out_default == 1))
        assert jnp.all((out_explicit == 0) | (out_explicit == 1))

    def test_padding_stripped_correctly(self):
        """When num_trajectories is not a multiple of n_devices, padding must be removed."""
        p = Program(H(0), MEASURE(0, None))
        sim = TrajectorySimulator(p, qubits=[0], devices=jax.devices())
        # 7 is unlikely to be a multiple of any device count > 1
        outcomes = sim.sample(None, num_trajectories=7, batch_size=7)
        assert outcomes.shape == (7, 1)

    def test_sample_spans_several_batches(self):
        """A run wider than one batch still returns exactly the trajectories asked for."""
        p = Program(H(0), MEASURE(0, None))
        sim = TrajectorySimulator(p, qubits=[0], devices=jax.devices())

        outcomes = sim.sample(None, num_trajectories=20, key=jax.random.key(42), batch_size=6)

        assert outcomes.shape == (20, 1)

    def test_noisy_sample_with_devices(self):
        """Multi-device path should work with noise models."""
        p_error = 0.3
        ch = SuperopChannel.from_pauli_noise(inst=X(0), pauli_noise={"X": p_error})
        noise_model = NoiseModel.from_channels([ch])
        p = Program(X(0), MEASURE(0, None))
        sim = TrajectorySimulator(p, noise_model=noise_model, qubits=[0], devices=jax.devices())
        outcomes = sim.sample(None, num_trajectories=1024, batch_size=256, key=jax.random.key(7))
        assert outcomes.shape == (1024, 1)
        frac_0 = float(jnp.mean(outcomes == 0))
        assert abs(frac_0 - p_error) < 0.05


# ──────────────────────────────────────────────────────────────────────────────
# Kraus stack layout
# ──────────────────────────────────────────────────────────────────────────────


class TestKrausStackLayout:
    """The layout is derived from the merge plan, never from a resolved circuit.

    Deriving it structurally is what lets the kernel be compiled once at construction and what
    keeps a parametric program from failing at some angles and not others; these tests pin the
    derivation against the same layout measured off real matrices.
    """

    @pytest.mark.parametrize("max_subsystem_size", [0, 1, 2, 3])
    @pytest.mark.parametrize("noisy", [False, True])
    def test_structural_derivation_matches_measured_matrices(self, max_subsystem_size, noisy):
        """``from_plan`` and ``from_operations`` must agree, field for field."""
        p = Program(
            Declare("theta", "REAL", 1),
            H(0),
            RX(MemoryReference("theta", 0), 1),
            CNOT(0, 1),
            X(2),
            MEASURE(0, None),
            MEASURE(2, None),
        )
        noise_model = (
            NoiseModel.from_channels([Channel.from_gate_fidelity(inst=X(2), fidelity=0.99)]) if noisy else None
        )
        sim = TrajectorySimulator(p, noise_model=noise_model, max_subsystem_size=max_subsystem_size)

        measured = _KrausStackLayout.from_operations(
            sim.operations(jnp.zeros(sim.num_parameters)),
            sim.dims,
            program_index=tuple(nodes[0] for nodes, _ in sim.plan.groups),
        )

        assert sim._layout == measured

    @pytest.mark.parametrize("max_subsystem_size", [0, 1, 2])
    def test_structural_derivation_handles_a_promoted_channel(self, max_subsystem_size):
        """A qubit-dimension channel on a slot a qutrit gate promoted to d=3.

        ``MergePlan.apply`` passes a lone operation through untouched, so a singleton channel
        is decomposed at *its own* dimension and only then padded up by the promotion.  Sizing
        its slot by the register dimension instead would waste rows on every leakage model.
        """
        noise_model = NoiseModel.from_channels([Channel.from_gate_fidelity(inst=X(0), fidelity=0.99)])
        p = Program(X(0), QuilGate("TX", [], [Qubit(0)]))
        sim = TrajectorySimulator(p, qubits=[0], noise_model=noise_model, max_subsystem_size=max_subsystem_size)

        measured = _KrausStackLayout.from_operations(
            sim.operations(None),
            sim.dims,
            program_index=tuple(nodes[0] for nodes, _ in sim.plan.groups),
        )

        assert sim.dims == (3,)
        assert sim._layout == measured
        if max_subsystem_size == 0:
            # The channel's own d^2 = 4, not the register's d^2 = 9.
            assert sim._layout.base_max_k == (4,)

    def test_kraus_counts_do_not_depend_on_the_parameters(self):
        """A merged group holding two channels around a parametric gate used to break this.

        The composite's Choi rank varies with the angle, so a layout probed at ``theta = 0``
        under-counted and every other angle raised at ``stack()``.
        """
        noise_model = NoiseModel.from_channels([SuperopChannel.from_pauli_noise(inst=X(0), pauli_noise={"X": 0.2})])
        p = Program(Declare("theta", "REAL", 1), X(0), RZ(MemoryReference("theta", 0), 0), X(0))
        sim = TrajectorySimulator(p, noise_model=noise_model, max_subsystem_size=2)

        # One group, so the layout's budget has to cover the *widest* angle, not theta = 0.
        assert sim.plan.groups == (((0, 1, 2), (0,)),)
        for theta in (0.0, 0.3, 0.7, 1.0):
            psi, _ = sim.compute(jnp.array([theta]), jax.random.key(0))
            assert jnp.allclose(jnp.linalg.norm(psi.matrix), 1.0)

    def test_unitary_groups_need_one_kraus_operator(self):
        """A unitary is rank one: no Choi round-trip, no zero rows to carry."""
        matrix, divisor, is_measurement = _op_to_kraus_matrix(qx.gates.H)

        assert matrix.shape == (1, 2, 2)
        assert (divisor, is_measurement) == (1, False)
        assert jnp.allclose(matrix[0], qx.gates.H.matrix)

    def test_base_budgets_are_per_subsystem(self):
        """Each base carries only what it needs, rather than the widest base in the circuit."""
        p = Program(H(0), CNOT(0, 1), MEASURE(0, None))
        sim = TrajectorySimulator(p, qubits=[0, 1], max_subsystem_size=1)
        budget = dict(zip(sim._layout.bases, sim._layout.base_max_k, strict=True))

        # The measured qubit needs num_outcomes * d^2 = 2 * 4; the bare CNOT base needs one.
        assert budget[(0,)] == 8
        assert budget[(0, 1)] == 1

    def test_program_index_orders_the_outcome_columns(self):
        """``program_index`` re-sorts measurement columns, in isolation from any planner."""
        instrument = qx.gates.MEASURE(dim=2)
        operations = [(instrument, (0,)), (qx.gates.H, (0,)), (instrument, (0,))]

        layout = _KrausStackLayout.from_operations(operations, (2,), program_index=(2, 1, 0))

        assert layout.measure_positions == (2, 0)

    def test_rejects_a_program_index_of_the_wrong_length(self):
        with pytest.raises(ValueError, match="program_index has 2 entries for 1 operation"):
            _KrausStackLayout.from_operations([(qx.gates.H, (0,))], (2,), program_index=(0, 1))

    def test_stack_rejects_the_wrong_operation_count(self):
        layout = _KrausStackLayout.from_operations([(qx.gates.H, (0,))], (2,))

        with pytest.raises(ValueError, match="covers 1 operation.* but 2 were given"):
            layout.stack([(qx.gates.H, (0,)), (qx.gates.H, (0,))])

    def test_stack_rejects_a_mismatched_subsystem(self):
        layout = _KrausStackLayout.from_operations([(qx.gates.H, (0,))], (2, 2))

        with pytest.raises(ValueError, match=r"acts on \(1,\); the layout expects \(0,\)"):
            layout.stack([(qx.gates.H, (1,))])

    def test_stack_rejects_too_many_kraus_operators(self):
        """A layout sized for a unitary cannot hold a channel in that slot."""
        layout = _KrausStackLayout.from_operations([(qx.gates.H, (0,))], (2,))

        with pytest.raises(ValueError, match="but the layout allows 1"):
            layout.stack([(qx.to_kraus(qx.channels.depolarizing(0.1, (2,))), (0,))])

    def test_rejects_an_operator_it_cannot_sample(self):
        with pytest.raises(TypeError, match="Unsupported operator type"):
            _op_to_kraus_matrix(qx.to_choi(qx.gates.H))


# ──────────────────────────────────────────────────────────────────────────────
# Outcome column order
# ──────────────────────────────────────────────────────────────────────────────


def test_outcome_columns_follow_program_order_when_the_plan_reorders_them():
    """Measurement columns are labelled by program index, not by emission position.

    Merging can legitimately swap two operations that share no qudit, so a ``MEASURE`` blocked
    behind a wide merge is emitted *after* one that came later in the program.  Here ops 0, 1
    and 4 fuse into a three-qudit group that sits between the two measurements, which pushes
    ``MEASURE 0`` out in front of ``MEASURE 1``.
    """
    p = Program(
        X(1),
        CZ(1, 2),
        QuilMeasurement(qubit=Qubit(1), classical_reg=None),
        QuilMeasurement(qubit=Qubit(0), classical_reg=None),
        CZ(0, 2),
    )
    sim = TrajectorySimulator(p, qubits=[0, 1, 2], max_subsystem_size=3)

    # The planner emits MEASURE 0 (program index 3) before MEASURE 1 (program index 2).
    assert [nodes[0] for nodes, _ in sim.plan.groups] == [3, 0, 2]
    assert sim._layout.measure_positions == (2, 0)

    _, outcomes = sim.compute(None, jax.random.split(jax.random.key(0), 4))

    # Program order is [MEASURE 1, MEASURE 0] and the state is |0> X|0> = |0>|1>.
    # Emission order would give [0, 1] instead.
    assert np.all(np.asarray(outcomes) == [1, 0])


# ──────────────────────────────────────────────────────────────────────────────
# Tracing and precision
# ──────────────────────────────────────────────────────────────────────────────


class TestTrajectoryInitialState:
    """Seeding a trajectory from a caller-supplied pure state.

    This is what lets a shared circuit prefix be evolved once and only the varying tail be run
    per trajectory.  Both halves must span the same register, so an idle qubit is padded with
    ``I``.
    """

    def test_prefix_then_tail_reproduces_the_whole_circuit_exactly(self):
        """Same keys and the same tail give the same trajectories, not merely the same statistics."""
        qubits = [0, 1, 2]
        prefix = Program(H(0), CNOT(0, 1), RX(0.3, 1), I(2))
        tail = Program(Declare("ro", "BIT", 3), RX(1.1, 0), CNOT(1, 2))
        tail += [MEASURE(q, ("ro", i)) for i, q in enumerate(qubits)]

        keys = jax.random.split(jax.random.key(3), 64)
        _, whole = TrajectorySimulator(prefix + tail, qubits=qubits).compute(None, keys)

        psi = PureStateVectorSimulator(prefix, qubits=qubits).compute()
        simulator = TrajectorySimulator(tail, qubits=qubits)
        _, split = jax.vmap(lambda key: simulator.compute(None, key, psi))(keys)
        np.testing.assert_array_equal(np.asarray(split), np.asarray(whole))

    def test_a_noisy_tail_from_a_prepared_state_matches_the_whole_circuit(self):
        qubits = [0, 1]
        prefix = Program(H(0), I(1))
        tail = Program(Declare("ro", "BIT", 2), CNOT(0, 1))
        tail += [MEASURE(q, ("ro", i)) for i, q in enumerate(qubits)]
        noise_model = NoiseModel.from_channels([Channel.from_depolarizing_constant(CNOT(0, 1), 0.9)])

        whole = TrajectorySimulator(prefix + tail, qubits=qubits, noise_model=noise_model)
        outcomes = np.asarray(whole.sample(num_trajectories=8000, key=jax.random.key(11)))

        psi = PureStateVectorSimulator(prefix, qubits=qubits).compute()
        simulator = TrajectorySimulator(tail, qubits=qubits, noise_model=noise_model)
        keys = jax.random.split(jax.random.key(11), 8000)
        _, split = jax.vmap(lambda key: simulator.compute(None, key, psi))(keys)
        split = np.asarray(split)

        def frequencies(samples):
            return np.bincount(samples[:, 0] * 2 + samples[:, 1], minlength=4) / len(samples)

        np.testing.assert_allclose(frequencies(split), frequencies(outcomes), atol=0.02)

    def test_a_density_matrix_is_rejected(self):
        """A trajectory carries a pure state; a mixed one has to be unravelled by the caller."""
        simulator = TrajectorySimulator(Program(X(0)), qubits=[0])
        with pytest.raises(TypeError, match="must be a StateVector"):
            simulator.compute(None, jax.random.key(0), qx.zero_state_matrix(dims=(2,)))

    def test_a_mismatched_register_is_rejected(self):
        simulator = TrajectorySimulator(Program(X(0)), qubits=[0])
        with pytest.raises(ValueError, match=r"dims \(2, 2\) but this simulator's register is \(2,\)"):
            simulator.compute(None, jax.random.key(0), qx.zero_state_vector(dims=(2, 2)))

    def test_the_initial_state_is_cast_to_the_evolution_dtype(self):
        simulator = TrajectorySimulator(Program(X(0)), qubits=[0], evolution_dtype=jnp.complex64)
        state, _ = simulator.compute(None, jax.random.key(0), qx.zero_state_vector(dims=(2,)))
        assert state.matrix.dtype == jnp.complex64


class TestTracingAndPrecision:
    def test_compute_can_be_jitted(self):
        """No data-dependent shape survives in the pipeline, so a whole sweep can be compiled."""
        noise_model = NoiseModel.from_channels([SuperopChannel.from_pauli_noise(inst=X(0), pauli_noise={"X": 0.1})])
        p = Program(Declare("theta", "REAL", 1), X(0), RX(MemoryReference("theta", 0), 0), MEASURE(0, None))
        sim = TrajectorySimulator(p, noise_model=noise_model)
        keys = jax.random.split(jax.random.key(0), 16)
        params = jnp.array([0.4])

        eager_psi, eager_outcomes = sim.compute(params, keys)
        jitted_psi, jitted_outcomes = jax.jit(lambda prm: sim.compute(prm, keys))(params)

        assert np.array_equal(eager_outcomes, jitted_outcomes)
        assert jnp.allclose(eager_psi.matrix, jitted_psi.matrix)

    def test_evolution_dtype_runs_in_single_precision(self):
        """The kernel evolves at the requested dtype while preprocessing stays at 64-bit."""
        p = Program(H(0), CNOT(0, 1), MEASURE(0, None))
        single = TrajectorySimulator(p, qubits=[0, 1], evolution_dtype=jnp.complex64)
        double = TrajectorySimulator(p, qubits=[0, 1])

        psi, outcomes = single.compute(None, jax.random.split(jax.random.key(0), 2048))

        assert psi.matrix.dtype == jnp.complex64
        assert double.evolution_dtype == jnp.complex128
        assert jnp.allclose(jnp.linalg.norm(psi.matrix, axis=-1), 1.0, atol=1e-5)
        # Sampling is a different draw at a different precision, so only the statistics agree.
        reference = double.compute(None, jax.random.split(jax.random.key(0), 2048))[1]
        assert abs(float(outcomes.mean()) - float(reference.mean())) < 0.05

    def test_rejects_a_real_evolution_dtype(self):
        with pytest.raises(ValueError, match="must be a complex dtype"):
            TrajectorySimulator(Program(H(0)), evolution_dtype=jnp.float64)
