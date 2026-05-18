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

"""Unit tests for the quax-based noise model (Channel, MeasurementChannel, ResetChannel, NoiseModel)."""

import jax.numpy as jnp
import numpy as np
import pytest
import quax as qx

from pyquil.gates import CNOT, MEASURE, RESET, RX, RY, X
from pyquil.noise._channels import Channel, MeasurementChannel, ResetChannel, get_instruction_unitary
from pyquil.noise._noise_model import NoiseModel
from pyquil.quil import Program
from pyquil.quilbase import Gate, Measurement, ResetQubit
from pyquil.simulation._simulator import DensityMatrixSimulator

_EMPTY_PARAMS = jnp.array([], dtype=float)


def _dm(program, noise_model=None, qubits=None):
    """Compute density matrix."""
    sim = DensityMatrixSimulator(program, qubits=qubits, noise_model=noise_model)
    return sim.compute(_EMPTY_PARAMS)


# ──────────────────────────────────────────────────────────
# Channel tests
# ──────────────────────────────────────────────────────────


class TestChannel:
    def test_from_depolarizing_constant(self):
        """Channel.from_depolarizing_constant produces valid superoperator."""
        inst = RX(np.pi / 4, 0)
        ch = Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=0.98)
        assert isinstance(ch.process, qx.SuperOp)
        # Process fidelity should be close to the depolarizing constant
        assert ch.fidelity < 1.0
        assert ch.fidelity > 0.95

    def test_from_gate_fidelity(self):
        """Channel.from_gate_fidelity produces correct fidelity."""
        inst = RX(np.pi / 2, 0)
        ch = Channel.from_gate_fidelity(inst=inst, fidelity=0.99)
        assert abs(ch.fidelity - 0.99) < 0.001

    def test_from_pauli_fidelity(self):
        """Channel.from_pauli_fidelity produces a valid channel."""
        inst = X(0)
        ch = Channel.from_pauli_fidelity(inst=inst, pauli_fidelity=0.97)
        assert isinstance(ch.process, qx.SuperOp)
        assert ch.pauli_fidelity == pytest.approx(0.97, abs=0.001)

    def test_from_pauli_noise(self):
        """Channel.from_pauli_noise produces a valid Pauli noise channel."""
        inst = RX(0.5, 0)
        ch = Channel.from_pauli_noise(inst=inst, pauli_noise={"X": 0.01, "Z": 0.02})
        assert isinstance(ch.process, qx.SuperOp)
        assert ch.fidelity < 1.0

    def test_from_coherence_times(self):
        """Channel.from_coherence_times produces a valid decoherence channel."""
        inst = RX(np.pi / 4, 0)
        ch = Channel.from_coherence_times(inst=inst, gate_duration=40e-9, t1s=[30e-6], t2s=[20e-6])
        assert isinstance(ch.process, qx.SuperOp)
        assert ch.fidelity < 1.0
        assert ch.fidelity > 0.99  # short gate relative to T1/T2

    def test_qubits(self):
        """Channel.qubits reflects the instruction's qubits."""
        ch = Channel.from_depolarizing_constant(inst=RX(0.1, 3), depolarizing_constant=0.99)
        assert ch.qubits == [3]

    def test_num_qubits(self):
        """Channel.num_qubits is correct for 2Q gates."""
        ch = Channel.from_depolarizing_constant(inst=CNOT(0, 1), depolarizing_constant=0.95)
        assert ch.num_qubits == 2

    def test_fidelity_properties(self):
        """Fidelity, infidelity, pauli_fidelity, pauli_infidelity are consistent."""
        ch = Channel.from_gate_fidelity(inst=RX(0.3, 0), fidelity=0.98)
        assert ch.fidelity + ch.infidelity == pytest.approx(1.0)
        assert ch.pauli_fidelity + ch.pauli_infidelity == pytest.approx(1.0)
        assert ch.stochastic_fidelity + ch.stochastic_infidelity == pytest.approx(1.0)
        assert ch.coherent_fidelity + ch.coherent_infidelity == pytest.approx(1.0)

    def test_noise_process(self):
        """noise_process factors out the ideal gate unitary."""
        ch = Channel.from_depolarizing_constant(inst=RX(np.pi / 4, 0), depolarizing_constant=0.99)
        noise = ch.noise_process
        assert isinstance(noise, qx.SuperOp)

    def test_is_pauli(self):
        """A depolarizing channel on a Clifford gate should be a Pauli channel."""
        ch = Channel.from_depolarizing_constant(inst=X(0), depolarizing_constant=0.98)
        assert ch.is_pauli()

    def test_pauli_twirl(self):
        """Pauli twirl of a channel on a Clifford gate should be a Pauli channel."""
        ch = Channel.from_random_coherent_error(
            inst=X(0), process_fidelity=0.97, rng=np.random.default_rng(42)
        )
        twirled = ch.pauli_twirl()
        assert twirled.is_pauli()

    def test_unitarity(self):
        """A depolarizing channel should have unitarity < 1."""
        ch = Channel.from_depolarizing_constant(inst=RX(0.5, 0), depolarizing_constant=0.95)
        assert 0 < ch.unitarity < 1.0

    def test_pauli_vector_sums_to_one(self):
        """Pauli error probability vector should sum to 1."""
        ch = Channel.from_depolarizing_constant(inst=X(0), depolarizing_constant=0.97)
        pv = ch.pauli_vector
        assert float(jnp.sum(pv)) == pytest.approx(1.0, abs=1e-8)

    def test_perfect_channel(self):
        """A depolarizing constant of 1.0 should give fidelity 1.0."""
        ch = Channel.from_depolarizing_constant(inst=RX(np.pi, 0), depolarizing_constant=1.0)
        assert ch.fidelity == pytest.approx(1.0, abs=1e-10)


# ──────────────────────────────────────────────────────────
# MeasurementChannel tests
# ──────────────────────────────────────────────────────────


class TestMeasurementChannel:
    def test_from_readout_fidelity(self):
        """MeasurementChannel.from_readout_fidelity produces a valid quantum instrument."""
        inst = Measurement(Gate("MEASURE", [], [0]).qubits[0], None)
        # Use the pyquil MEASURE gate to get qubit
        prog = Program(MEASURE(0, None))
        meas_inst = [i for i in prog if isinstance(i, Measurement)][0]
        ch = MeasurementChannel.from_readout_fidelity(inst=meas_inst, fidelity=0.95)
        assert isinstance(ch.process, qx.QuantumInstrument)

    def test_from_readout_fidelity_with_asymmetry(self):
        """MeasurementChannel with asymmetry produces asymmetric confusion."""
        prog = Program(MEASURE(0, None))
        meas_inst = [i for i in prog if isinstance(i, Measurement)][0]
        ch = MeasurementChannel.from_readout_fidelity(inst=meas_inst, fidelity=0.95, asymmetry=0.5)
        assert isinstance(ch.process, qx.QuantumInstrument)

    def test_qubits(self):
        """MeasurementChannel.qubits returns the correct qubit."""
        prog = Program(MEASURE(5, None))
        meas_inst = [i for i in prog if isinstance(i, Measurement)][0]
        ch = MeasurementChannel.from_readout_fidelity(inst=meas_inst, fidelity=0.99)
        assert ch.qubits == [5]


# ──────────────────────────────────────────────────────────
# NoiseModel tests
# ──────────────────────────────────────────────────────────


class TestNoiseModel:
    def test_empty_model(self):
        """An empty NoiseModel has no channels."""
        nm = NoiseModel(channels=frozenset())
        assert nm.get_channel(RX(0.5, 0)) is None

    def test_get_channel_gate(self):
        """NoiseModel.get_channel returns the correct Channel for a gate."""
        inst = RX(np.pi / 4, 0)
        ch = Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=0.98)
        nm = NoiseModel(channels=frozenset([ch]))
        retrieved = nm.get_channel(inst)
        assert retrieved is ch

    def test_get_channel_returns_none_for_missing(self):
        """get_channel returns None for instructions not in the model."""
        inst = RX(np.pi / 4, 0)
        ch = Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=0.98)
        nm = NoiseModel(channels=frozenset([ch]))
        other_inst = RY(np.pi / 2, 1)
        assert nm.get_channel(other_inst) is None

    def test_multiple_channels(self):
        """NoiseModel with multiple channels retrieves each correctly."""
        inst1 = RX(0.5, 0)
        inst2 = RY(0.3, 1)
        inst3 = CNOT(0, 1)
        ch1 = Channel.from_depolarizing_constant(inst=inst1, depolarizing_constant=0.99)
        ch2 = Channel.from_depolarizing_constant(inst=inst2, depolarizing_constant=0.97)
        ch3 = Channel.from_depolarizing_constant(inst=inst3, depolarizing_constant=0.95)
        nm = NoiseModel(channels=frozenset([ch1, ch2, ch3]))
        assert nm.get_channel(inst1) is ch1
        assert nm.get_channel(inst2) is ch2
        assert nm.get_channel(inst3) is ch3


# ──────────────────────────────────────────────────────────
# get_instruction_unitary tests
# ──────────────────────────────────────────────────────────


class TestGetInstructionUnitary:
    def test_standard_gate(self):
        """get_instruction_unitary resolves standard gates."""
        u = get_instruction_unitary(RX(np.pi / 2, 0))
        assert isinstance(u, qx.Unitary)
        assert u.matrix.shape == (2, 2)

    def test_two_qubit_gate(self):
        """get_instruction_unitary resolves 2Q gates."""
        u = get_instruction_unitary(CNOT(0, 1))
        assert isinstance(u, qx.Unitary)
        assert u.matrix.shape == (4, 4)

    def test_custom_gate(self):
        """get_instruction_unitary resolves custom gates from custom_gates dict."""
        custom_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        inst = Gate("MY_GATE", [], [0])
        u = get_instruction_unitary(inst, custom_gates={"MY_GATE": qx.Unitary.from_matrix(custom_matrix, ((2,), (2,)))})
        assert isinstance(u, qx.Unitary)
        assert np.allclose(np.asarray(u.matrix), custom_matrix)


# ──────────────────────────────────────────────────────────
# ResetChannel tests
# ──────────────────────────────────────────────────────────


class TestResetChannel:
    def test_from_reset_fidelity_perfect(self):
        """A perfect reset (fidelity=1.0) should produce a valid superoperator."""
        inst = RESET(0)
        ch = ResetChannel.from_reset_fidelity(inst=inst, fidelity=1.0)
        assert isinstance(ch.process, qx.SuperOp)

    def test_from_reset_fidelity_noisy(self):
        """A noisy reset should have lower fidelity than a perfect one."""
        inst = RESET(0)
        ch_perfect = ResetChannel.from_reset_fidelity(inst=inst, fidelity=1.0)
        ch_noisy = ResetChannel.from_reset_fidelity(inst=inst, fidelity=0.95)
        assert isinstance(ch_noisy.process, qx.SuperOp)
        assert ch_noisy.fidelity < ch_perfect.fidelity

    def test_qubits(self):
        """ResetChannel.qubits returns the correct qubit."""
        inst = RESET(3)
        ch = ResetChannel.from_reset_fidelity(inst=inst, fidelity=0.99)
        assert ch.qubits == [3]

    def test_ideal_reset_maps_excited_to_ground(self):
        """An ideal reset on an excited qubit should produce |0><0|."""
        inst = RESET(0)
        ch = ResetChannel.from_reset_fidelity(inst=inst, fidelity=1.0)
        noise_model = NoiseModel(channels=frozenset([ch]))
        # Prepare |1> then reset
        program = Program(X(0), RESET(0))
        rho = _dm(program, noise_model=noise_model)
        target_rho = qx.zero_state_matrix(1)
        assert qx.fidelity(rho, target_rho) > 0.9999

    def test_ideal_reset_maps_superposition_to_ground(self):
        """An ideal reset on a superposition state should produce |0><0|."""
        inst = RESET(0)
        ch = ResetChannel.from_reset_fidelity(inst=inst, fidelity=1.0)
        noise_model = NoiseModel(channels=frozenset([ch]))
        # Prepare |+> then reset
        program = Program(RX(np.pi / 2, 0), RESET(0))
        rho = _dm(program, noise_model=noise_model)
        target_rho = qx.zero_state_matrix(1)
        assert qx.fidelity(rho, target_rho) > 0.9999

    def test_noisy_reset_reduces_fidelity(self):
        """A noisy reset should produce a state with fidelity < 1 relative to |0><0|."""
        inst = RESET(0)
        ch = ResetChannel.from_reset_fidelity(inst=inst, fidelity=0.90)
        noise_model = NoiseModel(channels=frozenset([ch]))
        program = Program(X(0), RESET(0))
        rho = _dm(program, noise_model=noise_model)
        target_rho = qx.zero_state_matrix(1)
        fid = float(qx.fidelity(rho, target_rho))
        # Should be less than perfect but still high
        assert 0.85 < fid < 1.0

    def test_reset_in_multi_qubit_circuit(self):
        """Reset on one qubit should not affect the other qubit."""
        inst = RESET(0)
        ch = ResetChannel.from_reset_fidelity(inst=inst, fidelity=1.0)
        noise_model = NoiseModel(channels=frozenset([ch]))
        # Prepare |11> then reset qubit 0
        program = Program(X(0), X(1), RESET(0))
        rho = _dm(program, noise_model=noise_model)
        # Expected state: |0> on qubit 0, |1> on qubit 1 → |01>
        target_rho = (qx.gates.I | qx.gates.X) @ qx.zero_state_matrix(2)
        assert qx.fidelity(rho, target_rho) > 0.9999

    def test_global_reset(self):
        """A global RESET (no qubit specified) resets all qubits to |0>."""
        program = Program(X(0), X(1), RESET())
        rho = _dm(program)
        target_rho = qx.zero_state_matrix(2)
        assert qx.fidelity(rho, target_rho) > 0.9999
