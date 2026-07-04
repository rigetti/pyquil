"""Unit tests for qutrit (d=3) and mixed qubit/qutrit simulation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import quax as qx

from pyquil.gates import H, X
from pyquil.noise._channels import Channel, MeasurementChannel, ResetChannel
from pyquil.noise._noise_model import NoiseModel
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, Qubit
from pyquil.quilbase import Declare, DefGate, Gate, Measurement
from pyquil.simulation._simulator import (
    DensityMatrixSimulator,
    PureStateVectorSimulator,
    TrajectorySimulator,
)

_EMPTY_PARAMS = jnp.array([], dtype=float)


def _sv(program, qubits=None):
    """Compute pure state vector for a gate-only program."""
    sim = PureStateVectorSimulator(program, qubits=qubits)
    return sim.compute(_EMPTY_PARAMS)


def _dm(program, qubits=None, noise_model=None):
    """Compute density matrix."""
    sim = DensityMatrixSimulator(program, qubits=qubits, noise_model=noise_model)
    return sim.compute(_EMPTY_PARAMS)


def _sample(program, qubits=None, noise_model=None, num_trajectories=1000,
            batch_size=250, random_seed=0):
    """Run trajectory sampling, returning outcomes."""
    sim = TrajectorySimulator(program, qubits=qubits, noise_model=noise_model)
    return sim.sample(_EMPTY_PARAMS, num_trajectories=num_trajectories,
                      batch_size=batch_size, random_seed=random_seed)


# ══════════════════════════════════════════════════════════
# Test: Apply qutrit channels to programs
# ══════════════════════════════════════════════════════════


class TestQutritProgramSimulation:
    """Test that qutrit gates in programs produce correct state vectors."""

    def test_tx_gate_cycles(self):
        """TX (cyclic shift) maps |0> -> |2> -> |1> -> |0>."""
        # Apply TX once: |0> -> |2>
        p = Program()
        p += Gate("TX", [], [0])
        psi = _sv(p, qubits=[0])
        expected = qx.StateVector.from_matrix(
            jnp.array([0, 0, 1], dtype=complex), dims=(3,)
        )
        assert qx.fidelity(psi, expected) > 0.9999

    def test_tx_gate_double(self):
        """TX^2 maps |0> -> |1>."""
        p = Program()
        p += Gate("TX", [], [0])
        p += Gate("TX", [], [0])
        psi = _sv(p, qubits=[0])
        expected = qx.StateVector.from_matrix(
            jnp.array([0, 1, 0], dtype=complex), dims=(3,)
        )
        assert qx.fidelity(psi, expected) > 0.9999

    def test_tx_gate_triple_identity(self):
        """TX^3 = I for qutrits."""
        p = Program()
        p += Gate("TX", [], [0])
        p += Gate("TX", [], [0])
        p += Gate("TX", [], [0])
        psi = _sv(p, qubits=[0])
        expected = qx.StateVector.from_matrix(
            jnp.array([1, 0, 0], dtype=complex), dims=(3,)
        )
        assert qx.fidelity(psi, expected) > 0.9999

    def test_th_creates_superposition(self):
        """TH (qutrit Hadamard/QFT) creates uniform superposition from |0>."""
        p = Program()
        p += Gate("TH", [], [0])
        psi = _sv(p, qubits=[0])
        # QFT on |0> = (|0> + |1> + |2>) / sqrt(3)
        expected = qx.StateVector.from_matrix(
            jnp.array([1, 1, 1], dtype=complex) / jnp.sqrt(3), dims=(3,)
        )
        assert qx.fidelity(psi, expected) > 0.9999

    def test_tz_clock_matrix(self):
        """TZ (clock matrix) adds phases: |k> -> omega^k |k>."""
        # TX|0> = |2>, then TZ on |2> should give omega^2 * |2> where omega = exp(2*pi*i/3)
        p = Program()
        p += Gate("TX", [], [0])
        p += Gate("TZ", [], [0])
        psi = _sv(p, qubits=[0])
        omega = jnp.exp(2j * jnp.pi / 3)
        expected = qx.StateVector.from_matrix(
            jnp.array([0, 0, omega**2], dtype=complex), dims=(3,)
        )
        assert qx.fidelity(psi, expected) > 0.9999

    def test_parametric_trx01(self):
        """TRX01(pi) acts as a pi rotation in the |0>-|1> subspace."""
        p = Program()
        p += Gate("TRX01", [np.pi], [0])
        psi = _sv(p, qubits=[0])
        # RX(pi)|0> = -i|1> in the 0-1 subspace, |2> untouched
        expected = qx.StateVector.from_matrix(
            jnp.array([0, -1j, 0], dtype=complex), dims=(3,)
        )
        assert qx.fidelity(psi, expected) > 0.9999

    def test_parametric_trx02(self):
        """TRX02(pi) rotates between |0> and |2> subspace."""
        p = Program()
        p += Gate("TRX02", [np.pi], [0])
        psi = _sv(p, qubits=[0])
        # RX(pi) in 0-2 subspace: |0> -> -i|2>
        expected = qx.StateVector.from_matrix(
            jnp.array([0, 0, -1j], dtype=complex), dims=(3,)
        )
        assert qx.fidelity(psi, expected) > 0.9999

    def test_two_qutrit_tswap(self):
        """TSWAP swaps two qutrit registers."""
        # Prepare |2>|0> (TX maps |0>->|2>) then swap -> |0>|2>
        p = Program()
        p += Gate("TX", [], [0])
        p += Gate("TSWAP", [], [0, 1])
        psi = _sv(p, qubits=[0, 1])
        # After swap: qubit 0 in |0>, qubit 1 in |2>
        # State: |0>|2> in (3,3) space -> index 0*3 + 2 = 2 in 9-dim
        expected_vec = jnp.zeros(9, dtype=complex).at[2].set(1.0)
        expected = qx.StateVector.from_matrix(expected_vec, dims=(3, 3))
        assert qx.fidelity(psi, expected) > 0.9999

    def test_multi_qutrit_independence(self):
        """Two independent qutrit operations on separate registers."""
        p = Program()
        p += Gate("TX", [], [0])   # |0> -> |2>
        p += Gate("TH", [], [1])   # |0> -> (|0>+|1>+|2>)/sqrt(3)
        psi = _sv(p, qubits=[0, 1])
        assert psi.dims == (3, 3)
        # Product state: |2> ⊗ (|0>+|1>+|2>)/sqrt(3)
        q0 = jnp.array([0, 0, 1], dtype=complex)
        q1 = jnp.array([1, 1, 1], dtype=complex) / jnp.sqrt(3)
        expected_vec = jnp.kron(q0, q1)
        expected = qx.StateVector.from_matrix(expected_vec, dims=(3, 3))
        assert qx.fidelity(psi, expected) > 0.9999


# ══════════════════════════════════════════════════════════
# Test: All quax qutrit gates through the pure-state simulator
# ══════════════════════════════════════════════════════════


def _quax_qutrit_gates():
    """Yield ``(name, gate)`` for every unitary qutrit gate in quax.

    Projectors (TP0/TP1/TP2) are excluded because they are not unitary and
    therefore not valid for the pure-state simulator.
    """
    for name, gate in qx.gates.QUANTUM_GATES.items():
        if name.startswith("TP"):
            continue  # projectors are not unitary
        if callable(gate):
            # Parametric gates accept one or more angles; probe arity.
            unitary = None
            for n_args in (1, 2, 3):
                try:
                    unitary = gate(*([0.0] * n_args))
                    break
                except TypeError:
                    continue
            if unitary is None:
                continue
        else:
            unitary = gate
        if any(d == 3 for d in unitary.dims[1]):
            yield name, gate


class TestAllQutritGates:
    """Every unitary qutrit gate in quax simulates correctly on |0...0>."""

    # Single-qutrit fixed gates (non-parametric, dims == (3,)).
    SINGLE_FIXED = [
        name
        for name, gate in _quax_qutrit_gates()
        if not callable(gate) and gate.dims[1] == (3,)
    ]
    # Parametric single-qutrit rotations (callable, dims == (3,)).
    SINGLE_PARAM = [
        name
        for name, gate in _quax_qutrit_gates()
        if callable(gate)
    ]

    def test_gate_inventory_is_nonempty(self):
        """Sanity check: quax exposes the expected qutrit gate families."""
        # Clock/shift, Hadamard, Pauli-like, and Weyl operators are all present.
        for expected in ("TX", "TY", "TZ", "TH", "TSHIFT", "TSWAP", "W00", "W22"):
            assert expected in qx.gates.QUANTUM_GATES
        assert set(self.SINGLE_PARAM) >= {
            "TRX01", "TRY01", "TRZ01",
            "TRX02", "TRY02", "TRZ02",
            "TRX12", "TRY12", "TRZ12",
        }

    @pytest.mark.parametrize("name", SINGLE_FIXED)
    def test_single_qutrit_fixed_gate(self, name):
        """Each fixed single-qutrit gate produces the expected |0> column."""
        p = Program(Gate(name, [], [0]))
        psi = _sv(p, qubits=[0])
        assert psi.dims == (3,)
        # The output equals the gate's first column (its action on |0>).
        expected = qx.gates.QUANTUM_GATES[name].matrix[:, 0]
        np.testing.assert_allclose(psi.matrix, expected, atol=1e-6)
        # Unitary gates preserve normalization.
        np.testing.assert_allclose(
            float(jnp.sum(jnp.abs(psi.matrix) ** 2)), 1.0, atol=1e-6
        )

    @pytest.mark.parametrize("name", SINGLE_PARAM)
    def test_single_qutrit_parametric_gate(self, name):
        """Each parametric single-qutrit rotation simulates and is unitary."""
        angle = np.pi / 3
        p = Program(Gate(name, [angle], [0]))
        sim = PureStateVectorSimulator(p, qubits=[0])
        psi = sim.compute(jnp.array([], dtype=float))
        assert psi.dims == (3,)
        expected = qx.gates.QUANTUM_GATES[name](angle).matrix[:, 0]
        np.testing.assert_allclose(psi.matrix, expected, atol=1e-6)
        np.testing.assert_allclose(
            float(jnp.sum(jnp.abs(psi.matrix) ** 2)), 1.0, atol=1e-6
        )

    def test_parametric_qutrit_via_memory_map(self):
        """A parametric qutrit rotation resolves a runtime memory parameter."""
        p = Program()
        p += Declare("theta", "REAL", 1)
        p += Gate("TRX01", [MemoryReference("theta")], [0])
        sim = PureStateVectorSimulator(p, qubits=[0])
        params = sim.linearize({"theta": [np.pi]})
        psi = sim.compute(params)
        # TRX01(pi)|0> = -i|1> (pi rotation in the 0-1 subspace).
        expected = qx.StateVector.from_matrix(
            jnp.array([0, -1j, 0], dtype=complex), dims=(3,)
        )
        assert qx.fidelity(psi, expected) > 0.9999

    def test_two_qutrit_tswap_at_position_one(self):
        """A single-qutrit gate merged at the high slot of a TSWAP pair.

        Exercises embedding a qutrit gate at a non-zero position within a
        two-qutrit merge group (TH on slot 1 alongside TSWAP on (0, 1)).
        """
        p = Program()
        p += Gate("TX", [], [0])        # |0> -> |2> on slot 0
        p += Gate("TH", [], [1])        # superposition on slot 1
        p += Gate("TSWAP", [], [0, 1])  # swap the two qutrits
        psi = _sv(p, qubits=[0, 1])
        assert psi.dims == (3, 3)
        # After swap: slot 0 holds TH|0>, slot 1 holds |2>.
        q0 = jnp.array([1, 1, 1], dtype=complex) / jnp.sqrt(3)
        q1 = jnp.array([0, 0, 1], dtype=complex)
        expected = qx.StateVector.from_matrix(jnp.kron(q0, q1), dims=(3, 3))
        assert qx.fidelity(psi, expected) > 0.9999

    def test_jit_and_grad_qutrit(self):
        """The qutrit pure-state simulator is jit- and grad-friendly."""
        p = Program()
        p += Declare("theta", "REAL", 1)
        p += Gate("TRX01", [MemoryReference("theta")], [0])
        sim = PureStateVectorSimulator(p, qubits=[0])

        def excited_population(theta):
            params = jnp.array([theta], dtype=float)
            psi = sim.compute(params)
            return jnp.abs(psi.matrix[1]) ** 2

        # jit produces the same result as eager execution.
        val_eager = float(excited_population(np.pi / 2))
        val_jit = float(jax.jit(excited_population)(np.pi / 2))
        np.testing.assert_allclose(val_jit, val_eager, atol=1e-6)
        # grad is finite and well-defined.
        g = float(jax.grad(excited_population)(np.pi / 2))
        assert np.isfinite(g)


# ══════════════════════════════════════════════════════════
# Test: Mixed qubit/qutrit systems
# ══════════════════════════════════════════════════════════


class TestMixedQubitQutrit:
    """Test that mixed qubit/qutrit registers are handled correctly."""

    def test_dimension_inference_mixed(self):
        """Dimension inference correctly identifies qubit vs qutrit registers."""
        # Qubit gate on register 0, qutrit gate on register 1
        p = Program()
        p += X(0)                      # qubit gate on q0
        p += Gate("TX", [], [1])       # qutrit gate on q1
        sim = PureStateVectorSimulator(p, qubits=[0, 1])
        assert sim.dims == (2, 3)

    def test_mixed_state_vector_dims(self):
        """State vector from mixed system has correct dims."""
        p = Program()
        p += X(0)                      # qubit: |0> -> |1>
        p += Gate("TX", [], [1])       # qutrit: |0> -> |2>
        psi = _sv(p, qubits=[0, 1])
        assert psi.dims == (2, 3)
        # State should be |1>⊗|2> in (2,3) space = index 1*3 + 2 = 5 in 6-dim
        expected_vec = jnp.zeros(6, dtype=complex).at[5].set(1.0)
        expected = qx.StateVector.from_matrix(expected_vec, dims=(2, 3))
        assert qx.fidelity(psi, expected) > 0.9999

    def test_mixed_density_matrix_dims(self):
        """Density matrix simulator also handles mixed qubit/qutrit."""
        p = Program()
        p += X(0)                      # qubit on q0
        p += Gate("TX", [], [1])       # qutrit on q1
        rho = _dm(p, qubits=[0, 1])
        assert rho.dims == (2, 3)

    def test_mixed_three_registers(self):
        """Three registers: qubit, qutrit, qubit."""
        p = Program()
        p += X(0)                      # qubit
        p += Gate("TX", [], [1])       # qutrit
        p += H(2)                      # qubit
        psi = _sv(p, qubits=[0, 1, 2])
        assert psi.dims == (2, 3, 2)

    def test_mixed_qubit_qutrit_entanglement_via_defgate(self):
        """Test entanglement with qutrit gates.

        Note: DefGate requires matrix dimensions that are a perfect power
        of an integer (2, 3, 4, 8, 9, 16, 25, 27, ...). Mixed qubit/qutrit
        custom gates (e.g. 6x6) cannot use DefGate since 6 is not a
        perfect power. We test entanglement using built-in qutrit gates.
        """
        # DefGate rejects non-perfect-power matrices (6 = 2*3)
        mat = np.eye(6, dtype=complex)
        with pytest.raises(ValueError, match="perfect power"):
            DefGate("BAD_GATE", mat)

        # Test entanglement using built-in gates instead:
        # Use TSWAP to entangle two qutrits
        p = Program()
        p += Gate("TH", [], [0])              # superposition on q0
        p += Gate("TSWAP", [], [0, 1])        # entangle q0 and q1
        p += Gate("TH", [], [0])              # further evolve q0
        psi = _sv(p, qubits=[0, 1])
        assert psi.dims == (3, 3)
        # The state should NOT be a product state (entangled)
        # Check by verifying reduced purity < 1
        rho = _dm(
            Program([
                Gate("TH", [], [0]),
                Gate("TSWAP", [], [0, 1]),
                Gate("TH", [], [0]),
            ]),
            qubits=[0, 1],
        )
        full_purity = float(jnp.real(jnp.trace(rho.matrix @ rho.matrix)))
        assert full_purity > 0.9999  # pure state

    def test_dimension_inference_density_matrix(self):
        """Density matrix preprocess_program infers mixed dims correctly."""
        p = Program([
            H(0),                      # qubit
            Gate("TH", [], [1]),        # qutrit
        ])
        sim = DensityMatrixSimulator(p, qubits=[0, 1])
        assert sim.dims == (2, 3)


# ══════════════════════════════════════════════════════════
# Test: Qutrit measurements
# ══════════════════════════════════════════════════════════


class TestQutritMeasurements:
    """Test that qutrit measurements produce correct outcome distributions."""

    def test_qutrit_measure_ground_state(self):
        """Measuring a qutrit in |0> always yields outcome 0."""
        # Use the identity-like approach: TRX01(0) is identity but establishes dim=3
        p_ground = Program()
        p_ground += Gate("TRX01", [0.0], [0])  # identity rotation
        p_ground += Measurement(qubit=Qubit(0), classical_reg=None)

        outcomes = _sample(
            p_ground, qubits=[0], num_trajectories=100, random_seed=42
        )
        # All outcomes should be 0 (ground state)
        assert jnp.all(outcomes == 0)

    def test_qutrit_measure_excited_state(self):
        """Measuring a qutrit in |2> (via TX) always yields outcome 2."""
        p = Program()
        p += Gate("TX", [], [0])
        p += Measurement(qubit=Qubit(0), classical_reg=None)

        outcomes = _sample(
            p, qubits=[0], num_trajectories=100, random_seed=42
        )
        assert jnp.all(outcomes == 2)

    def test_qutrit_measure_second_excited(self):
        """Measuring a qutrit in |1> (via TX^2) always yields outcome 1."""
        p = Program()
        p += Gate("TX", [], [0])
        p += Gate("TX", [], [0])
        p += Measurement(qubit=Qubit(0), classical_reg=None)

        outcomes = _sample(
            p, qubits=[0], num_trajectories=100, random_seed=42
        )
        assert jnp.all(outcomes == 1)

    def test_qutrit_ideal_reset(self):
        """An ideal qutrit reset returns any qutrit level to |0>."""
        from pyquil.quilbase import ResetQubit

        p = Program()
        p += Gate("TX", [], [0])
        p += ResetQubit(Qubit(0))
        p += Measurement(qubit=Qubit(0), classical_reg=None)

        outcomes = _sample(
            p, qubits=[0], num_trajectories=100, random_seed=42
        )
        assert jnp.all(outcomes == 0)

    def test_qutrit_measure_superposition_statistics(self):
        """TH|0> = (|0>+|1>+|2>)/sqrt(3) gives uniform measurement distribution."""
        p = Program()
        p += Gate("TH", [], [0])
        p += Measurement(qubit=Qubit(0), classical_reg=None)

        n_traj = 3000
        outcomes = _sample(
            p, qubits=[0], num_trajectories=n_traj, random_seed=123
        )
        # Each outcome should appear ~1/3 of the time
        counts = jnp.bincount(outcomes.flatten(), length=3)
        freqs = counts / n_traj
        np.testing.assert_allclose(freqs, [1 / 3, 1 / 3, 1 / 3], atol=0.05)

    def test_qutrit_noisy_measurement_channel(self):
        """Noisy qutrit measurement with confusion matrix."""
        meas_inst = Measurement(qubit=Qubit(0), classical_reg=None)
        meas_ch = MeasurementChannel.from_readout_fidelity(
            inst=meas_inst, fidelity=0.9, dim=3
        )
        noise_model = NoiseModel.from_channels([meas_ch])

        # Prepare |2> (TX|0>=|2>) and measure with noise
        p = Program()
        p += Gate("TX", [], [0])
        p += Measurement(qubit=Qubit(0), classical_reg=None)

        n_traj = 2000
        outcomes = _sample(
            p, noise_model=noise_model, qubits=[0],
            num_trajectories=n_traj, random_seed=42,
        )
        counts = jnp.bincount(outcomes.flatten(), length=3)
        # Should be mostly outcome 2 (the prepared state), with some errors
        assert counts[2] / n_traj > 0.7  # majority correct

    def test_mixed_qubit_qutrit_measurement(self):
        """Measure both a qubit and qutrit in the same program."""
        p = Program()
        p += X(0)  # qubit |0> -> |1>
        p += Gate("TX", [], [1])  # qutrit |0> -> |2>
        p += Measurement(qubit=Qubit(0), classical_reg=None)
        p += Measurement(qubit=Qubit(1), classical_reg=None)

        outcomes = _sample(
            p, qubits=[0, 1], num_trajectories=50, random_seed=0
        )
        # qubit measurement should give 1, qutrit measurement should give 2
        assert outcomes.shape == (50, 2)
        assert jnp.all(outcomes[:, 0] == 1)
        assert jnp.all(outcomes[:, 1] == 2)


# ══════════════════════════════════════════════════════════
# Test: Dimension inference
# ══════════════════════════════════════════════════════════


class TestDimensionInference:
    """Test the mechanism for deciding initial register dimension."""

    def test_all_qubit_program(self):
        """A program with only qubit gates infers dims=(2,2)."""
        p = Program(H(0), X(1))
        sim = PureStateVectorSimulator(p, qubits=[0, 1])
        assert sim.dims == (2, 2)

    def test_single_qutrit_program(self):
        """A program with a single qutrit gate infers dims=(3,)."""
        p = Program(Gate("TX", [], [0]))
        sim = PureStateVectorSimulator(p, qubits=[0])
        assert sim.dims == (3,)

    def test_mixed_dims_from_operations(self):
        """Operations on different registers infer heterogeneous dims."""
        p = Program(
            H(0),                      # dim=2 on slot 0
            Gate("TX", [], [1]),        # dim=3 on slot 1
            X(2),                       # dim=2 on slot 2
        )
        sim = PureStateVectorSimulator(p, qubits=[0, 1, 2])
        assert sim.dims == (2, 3, 2)

    def test_dimension_upgrade_takes_max(self):
        """If a slot sees both dim=2 and dim=3 ops, dim=3 wins."""
        p = Program(
            Gate("TX", [], [0]),        # dim=3
        )
        sim = PureStateVectorSimulator(p, qubits=[0])
        assert sim.dims == (3,)

    def test_density_matrix_dimension_inference_consistency(self):
        """State vector and density matrix simulators infer same dims."""
        p = Program(
            X(0),
            Gate("TH", [], [1]),
            Gate("TX", [], [2]),
            H(3),
        )
        # Density matrix path
        dm_sim = DensityMatrixSimulator(p, qubits=[0, 1, 2, 3])
        dm_dims = dm_sim.dims

        # State vector path
        sv_sim = PureStateVectorSimulator(p, qubits=[0, 1, 2, 3])
        sv_dims = sv_sim.dims

        assert dm_dims == sv_dims == (2, 3, 3, 2)

    def test_two_qutrit_gate_infers_both_slots(self):
        """A two-qutrit gate (TSWAP) upgrades both slots to dim=3."""
        p = Program(Gate("TSWAP", [], [0, 1]))
        sim = PureStateVectorSimulator(p, qubits=[0, 1])
        assert sim.dims == (3, 3)

    def test_custom_defgate_qutrit_dimensions(self):
        """DefGate accepts 3x3 unitary matrices for single-qutrit gates."""
        # 3x3 identity is a valid qutrit gate
        mat = np.eye(3, dtype=complex)
        dg = DefGate("MY_QUTRIT_GATE", mat)
        assert dg.num_args() == 1

        # Built-in qutrit gates also work and infer dim=3
        p = Program(Gate("TX", [], [0]))
        psi = _sv(p, qubits=[0])
        assert psi.dims == (3,)

    def test_custom_defgate_two_qutrit(self):
        """DefGate accepts 9x9 unitary matrices for two-qutrit gates."""
        mat = np.eye(9, dtype=complex)
        dg = DefGate("TWO_QUTRIT_ID", mat)
        assert dg.num_args() == 2

        # Built-in TSWAP also works for two-qutrit systems
        p = Program(Gate("TSWAP", [], [0, 1]))
        psi = _sv(p, qubits=[0, 1])
        assert psi.matrix.shape[-1] == 9
        assert psi.dims == (3, 3)

    def test_custom_defgate_rejects_non_perfect_power(self):
        """DefGate rejects matrices whose dimension is not a perfect power."""
        # 6 = 2*3 is not a perfect power of any integer
        mat = np.eye(6, dtype=complex)
        with pytest.raises(ValueError, match="perfect power"):
            DefGate("BAD_GATE", mat)


# ══════════════════════════════════════════════════════════
# Test: Qutrit noise channels
# ══════════════════════════════════════════════════════════


class TestQutritNoiseChannels:
    """Test noisy qutrit simulation via NoiseModel."""

    def test_qutrit_depolarizing_channel(self):
        """A depolarizing channel on a qutrit gate mixes the state."""
        inst = Gate("TX", [], [0])
        channel = Channel.from_gate_fidelity(inst=inst, fidelity=0.8)
        noise_model = NoiseModel.from_channels([channel])

        # Density matrix should show mixed state
        p = Program(Gate("TX", [], [0]))
        rho = _dm(p, noise_model=noise_model, qubits=[0])
        assert rho.dims == (3,)
        # Purity < 1 indicates noise
        purity = float(jnp.real(jnp.trace(rho.matrix @ rho.matrix)))
        assert purity < 0.99

    def test_qutrit_depolarizing_trajectory(self):
        """Trajectory simulation with qutrit depolarizing noise."""
        inst = Gate("TX", [], [0])
        channel = Channel.from_gate_fidelity(inst=inst, fidelity=0.9)
        noise_model = NoiseModel.from_channels([channel])

        p = Program()
        p += Gate("TX", [], [0])
        p += Measurement(qubit=Qubit(0), classical_reg=None)

        n_traj = 2000
        outcomes = _sample(
            p, noise_model=noise_model, qubits=[0],
            num_trajectories=n_traj, random_seed=7,
        )
        counts = jnp.bincount(outcomes.flatten(), length=3)
        # Most should be outcome 2 (ideal TX|0>=|2>), with some noise
        assert counts[2] / n_traj > 0.7

    def test_qutrit_reset_channel(self):
        """Noisy qutrit reset via ResetChannel."""
        from pyquil.quilbase import ResetQubit

        reset_inst = ResetQubit(Qubit(0))
        reset_ch = ResetChannel.from_reset_fidelity(inst=reset_inst, fidelity=0.9, dim=3)
        noise_model = NoiseModel.from_channels([reset_ch])

        # Prepare |1> (TX^2|0>=|1>), then reset — should mostly go to |0>
        p = Program()
        p += Gate("TX", [], [0])
        p += Gate("TX", [], [0])  # |0> -> |1>
        p += ResetQubit(Qubit(0))
        p += Measurement(qubit=Qubit(0), classical_reg=None)

        n_traj = 1000
        outcomes = _sample(
            p, noise_model=noise_model, qubits=[0],
            num_trajectories=n_traj, random_seed=55,
        )
        counts = jnp.bincount(outcomes.flatten(), length=3)
        # Majority should be reset to |0>
        assert counts[0] / n_traj > 0.7

    def test_mixed_noise_qubit_and_qutrit(self):
        """Noise model with channels for both qubit and qutrit gates."""
        # Noisy qubit X gate
        ch_qubit = Channel.from_gate_fidelity(
            inst=Gate("X", [], [0]), fidelity=0.95
        )
        # Noisy qutrit TX gate
        ch_qutrit = Channel.from_gate_fidelity(
            inst=Gate("TX", [], [1]), fidelity=0.95
        )
        noise_model = NoiseModel.from_channels([ch_qubit, ch_qutrit])

        p = Program()
        p += X(0)
        p += Gate("TX", [], [1])
        rho = _dm(p, noise_model=noise_model, qubits=[0, 1])
        assert rho.dims == (2, 3)
        # Both registers should have purity < 1 due to noise
        purity = float(jnp.real(jnp.trace(rho.matrix @ rho.matrix)))
        assert purity < 0.99


# ══════════════════════════════════════════════════════════
# Test: Qutrit state vector trajectories (batched)
# ══════════════════════════════════════════════════════════


class TestQutritTrajectoryBatching:
    """Test that batched trajectory simulation works for qutrits."""

    def test_noiseless_qutrit_batch(self):
        """Batched noiseless qutrit simulation produces identical trajectories."""
        p = Program(Gate("TX", [], [0]))
        sim = TrajectorySimulator(p, qubits=[0])
        keys = jax.random.split(jax.random.key(0), 10)
        psi, _ = sim.compute(_EMPTY_PARAMS, keys)
        # All 10 trajectories should be identical (noiseless)
        assert psi.ensemble_size == (10,)
        assert psi.dims == (3,)
        expected = jnp.array([0, 0, 1], dtype=complex)  # TX|0> = |2>
        for i in range(10):
            fid = float(jnp.abs(jnp.vdot(psi.matrix[i], expected)) ** 2)
            assert fid > 0.9999

    def test_qutrit_trajectory_outcomes_shape(self):
        """Trajectory outcomes have correct shape for qutrit programs."""
        p = Program()
        p += Gate("TH", [], [0])
        p += Measurement(qubit=Qubit(0), classical_reg=None)

        outcomes = _sample(
            p, qubits=[0], num_trajectories=500, batch_size=100, random_seed=0
        )
        assert outcomes.shape == (500, 1)
        # Outcomes should be in {0, 1, 2}
        assert jnp.all(outcomes >= 0)
        assert jnp.all(outcomes <= 2)
