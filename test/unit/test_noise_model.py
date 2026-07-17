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

from pyquil.external.rpcq import CompilerISA
from pyquil.gates import CNOT, MEASURE, RESET, RX, RY, RZ, X
from pyquil.noise._channels import (
    Channel,
    CycleChannel,
    MeasurementChannel,
    ResetChannel,
    _build_cycle_channel,
    _resolve_params,
    get_custom_gates_from_program,
    get_instruction_unitary,
)
from pyquil.noise._noise_model import NoiseModel
from pyquil.quil import Program
from pyquil.quilatom import FormalArgument, Qubit
from pyquil.quilbase import DefCircuit, Gate, Measurement, ResetQubit

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
        ch = Channel.from_random_coherent_error(inst=X(0), process_fidelity=0.97, rng=np.random.default_rng(42))
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

    def test_from_pauli_noise_rejects_invalid_probabilities(self):
        """Pauli error rates must be probabilities with total error no greater than 1."""
        with pytest.raises(ValueError, match="negative"):
            Channel.from_pauli_noise(inst=RX(0.5, 0), pauli_noise={"X": -0.1})

        with pytest.raises(ValueError, match="at most 1.0"):
            Channel.from_pauli_noise(inst=RX(0.5, 0), pauli_noise={"X": 0.6, "Z": 0.5})

    def test_from_pauli_noise_two_qubit(self):
        """from_pauli_noise builds the correct 16-term Pauli channel for a 2Q gate (regression)."""
        pauli_noise = {"IX": 0.01, "XI": 0.005, "ZZ": 0.02}
        ch = Channel.from_pauli_noise(inst=CNOT(0, 1), pauli_noise=pauli_noise)
        pv = np.asarray(ch.pauli_vector)
        assert pv.size == 16
        assert float(jnp.sum(ch.pauli_vector)) == pytest.approx(1.0, abs=1e-3)
        terms = [a + b for a in "IXYZ" for b in "IXYZ"]
        rates = dict(zip(terms, pv, strict=True))
        for term, rate in pauli_noise.items():
            assert rates[term] == pytest.approx(rate, abs=1e-3)
        assert rates["II"] == pytest.approx(1.0 - sum(pauli_noise.values()), abs=1e-3)

    def test_json_roundtrip_preserves_qutrit_dims(self):
        """Channel JSON includes explicit dims for non-qubit operators."""
        qutrit_x = jnp.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=complex,
        )
        target_unitary = qx.Unitary.from_matrix(qutrit_x, ((3,), (3,)))
        channel = Channel(
            inst=Gate("TX", [], [0]), process=qx.to_superop(target_unitary), target_unitary=target_unitary
        )

        restored = Channel.from_json(channel.to_json())

        assert restored.inst == channel.inst
        assert restored.process.dims == ((3,), (3,))
        assert restored.target_unitary.dims == ((3,), (3,))
        assert jnp.allclose(restored.process.matrix, channel.process.matrix)


# ──────────────────────────────────────────────────────────
# MeasurementChannel tests
# ──────────────────────────────────────────────────────────


class TestMeasurementChannel:
    def test_from_readout_fidelity(self):
        """MeasurementChannel.from_readout_fidelity produces a valid quantum instrument."""
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

    @pytest.mark.parametrize("asymmetry", [0.0, 0.5])
    def test_pow_scales_readout_noise(self, asymmetry):
        """MeasurementChannel ** power scales readout noise via the stochastic generator."""
        prog = Program(MEASURE(0, None))
        meas_inst = [i for i in prog if isinstance(i, Measurement)][0]
        ch = MeasurementChannel.from_readout_fidelity(inst=meas_inst, fidelity=0.95, asymmetry=asymmetry)

        def bitflip(channel):
            cm = np.asarray(channel.confusion_matrix)
            return 1.0 - 0.5 * (float(cm[0, 0]) + float(cm[1, 1]))

        assert bitflip(ch**0.0) == pytest.approx(0.0, abs=1e-3)
        assert bitflip(ch**1.0) == pytest.approx(bitflip(ch), abs=1e-3)
        assert bitflip(ch**2.0) > bitflip(ch)
        # The generator construction keeps the result exactly column-stochastic and non-negative.
        powered = np.asarray((ch**1.5).confusion_matrix)
        assert np.all(powered >= -1e-9)
        assert np.allclose(powered.sum(axis=0), 1.0, atol=1e-6)

    def test_pow_rejects_non_embeddable_measurement(self):
        """A confusion matrix with no real generator cannot be fractionally powered."""
        prog = Program(MEASURE(0, None))
        meas_inst = [i for i in prog if isinstance(i, Measurement)][0]
        # Near-complete bit flip: eigenvalue ~ -0.8, so the matrix is not embeddable.
        confusion = jnp.array([[0.1, 0.9], [0.9, 0.1]])
        ch = MeasurementChannel.from_confusion_and_transition(meas_inst, confusion, jnp.eye(2))
        with pytest.raises(ValueError, match="not embeddable|not a valid"):
            _ = ch**0.5

    def test_from_binary_discriminator_qubit_is_faithful_readout(self):
        """Regression: dim=2/threshold=1 is a real qubit readout, not an always-0 collapse.

        The previous implementation mapped both |0> and |1> to outcome 0 for a qubit,
        silently erasing all measurement information.
        """
        prog = Program(MEASURE(0, None))
        meas_inst = [i for i in prog if isinstance(i, Measurement)][0]
        ch = MeasurementChannel.from_binary_discriminator(inst=meas_inst, dim=2, threshold=1)
        cm = np.asarray(ch.confusion_matrix)
        assert cm.shape == (2, 2)  # two reachable outcomes
        assert np.allclose(cm, np.eye(2))  # |0> -> 0, |1> -> 1

    def test_from_binary_discriminator_qutrit_split(self):
        """dim=3 splits into exactly two outcomes at the threshold."""
        prog = Program(MEASURE(0, None))
        meas_inst = [i for i in prog if isinstance(i, Measurement)][0]
        # threshold=2: {0,1} -> 0, {2} -> 1 (flag leakage only)
        cm2 = np.asarray(MeasurementChannel.from_binary_discriminator(inst=meas_inst, dim=3, threshold=2).confusion_matrix)
        assert np.allclose(cm2, [[1, 1, 0], [0, 0, 1]])
        # threshold=1: {0} -> 0, {1,2} -> 1 (ground vs excited-or-leaked)
        cm1 = np.asarray(MeasurementChannel.from_binary_discriminator(inst=meas_inst, dim=3, threshold=1).confusion_matrix)
        assert np.allclose(cm1, [[1, 0, 0], [0, 1, 1]])

    def test_from_binary_discriminator_fidelity_degrades(self):
        """Sub-unit fidelity stays column-stochastic and keeps both outcomes reachable."""
        prog = Program(MEASURE(0, None))
        meas_inst = [i for i in prog if isinstance(i, Measurement)][0]
        cm = np.asarray(
            MeasurementChannel.from_binary_discriminator(inst=meas_inst, dim=2, threshold=1, fidelity=0.9).confusion_matrix
        )
        assert cm.shape == (2, 2)
        assert np.allclose(cm.sum(axis=0), 1.0)
        assert cm[1, 1] > cm[0, 1]  # |1> still most likely reads as outcome 1

    def test_from_binary_discriminator_rejects_bad_threshold(self):
        """threshold must satisfy 1 <= threshold < dim."""
        prog = Program(MEASURE(0, None))
        meas_inst = [i for i in prog if isinstance(i, Measurement)][0]
        with pytest.raises(ValueError, match="threshold"):
            MeasurementChannel.from_binary_discriminator(inst=meas_inst, dim=2, threshold=2)

    def test_json_roundtrip_preserves_qutrit_dims(self):
        """MeasurementChannel JSON includes explicit dims for non-qubit instruments."""
        prog = Program(MEASURE(0, None))
        meas_inst = [i for i in prog if isinstance(i, Measurement)][0]
        channel = MeasurementChannel.from_readout_fidelity(inst=meas_inst, fidelity=0.95, dim=3)

        restored = MeasurementChannel.from_json(channel.to_json())

        assert restored.inst == channel.inst
        assert restored.process.dims == channel.process.dims
        assert restored.process.measured_qudits == channel.process.measured_qudits
        assert jnp.allclose(restored.process.matrix, channel.process.matrix)


# ──────────────────────────────────────────────────────────
# NoiseModel tests
# ──────────────────────────────────────────────────────────


class TestNoiseModel:
    def test_empty_model(self):
        """An empty NoiseModel has no channels."""
        nm = NoiseModel()
        assert nm.get_channel(RX(0.5, 0)) is None

    def test_constructor_accepts_instruction_mapping(self):
        """NoiseModel stores channels keyed by instruction."""
        inst = RX(np.pi / 4, 0)
        ch = Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=0.98)
        nm = NoiseModel(channels={inst: ch})
        assert nm.channels[inst] is ch
        assert nm.get_channel(inst) is ch

    def test_constructor_rejects_channel_iterable(self):
        """Sequence construction should go through from_channels."""
        inst = RX(np.pi / 4, 0)
        ch = Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=0.98)
        with pytest.raises(TypeError, match="from_channels"):
            NoiseModel(channels=[ch])  # type: ignore[arg-type]

    def test_pickle_roundtrip(self):
        """NoiseModel survives pickling (its MappingProxyType channels would otherwise block it).

        This is what lets a model be shipped to multiprocessing workers.
        """
        import pickle

        gate = RX(np.pi / 4, 0)
        prog = Program(MEASURE(0, None))
        meas_inst = [i for i in prog if isinstance(i, Measurement)][0]
        nm = NoiseModel.from_channels(
            [
                Channel.from_depolarizing_constant(inst=gate, depolarizing_constant=0.98),
                MeasurementChannel.from_readout_fidelity(inst=meas_inst, fidelity=0.95),
            ]
        )
        restored = pickle.loads(pickle.dumps(nm))
        assert set(restored.channels) == set(nm.channels)
        assert isinstance(restored.get_channel(gate), Channel)
        # Channels survive the round-trip intact (exercises value-based __eq__).
        assert restored == nm

    def test_constructor_rejects_mismatched_mapping_key(self):
        """Mapping keys must match the instruction stored on each channel."""
        inst = RX(np.pi / 4, 0)
        ch = Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=0.98)
        with pytest.raises(ValueError, match="does not match"):
            NoiseModel(channels={RY(np.pi / 2, 0): ch})

    def test_from_channels_rejects_duplicates(self):
        """Duplicate instruction channels are ambiguous and rejected."""
        inst = RX(np.pi / 4, 0)
        ch1 = Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=0.98)
        ch2 = Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=0.97)
        with pytest.raises(ValueError, match="Duplicate noise channel"):
            NoiseModel.from_channels([ch1, ch2])

    def test_get_channel_gate(self):
        """NoiseModel.get_channel returns the correct Channel for a gate."""
        inst = RX(np.pi / 4, 0)
        ch = Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=0.98)
        nm = NoiseModel.from_channels([ch])
        retrieved = nm.get_channel(inst)
        assert retrieved is ch

    def test_get_channel_returns_none_for_missing(self):
        """get_channel returns None for instructions not in the model."""
        inst = RX(np.pi / 4, 0)
        ch = Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=0.98)
        nm = NoiseModel.from_channels([ch])
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
        nm = NoiseModel.from_channels([ch1, ch2, ch3])
        assert nm.get_channel(inst1) is ch1
        assert nm.get_channel(inst2) is ch2
        assert nm.get_channel(inst3) is ch3

    def test_json_roundtrip(self):
        """NoiseModel JSON keeps the existing channel-list wire format."""
        ch = Channel.from_depolarizing_constant(inst=RX(np.pi / 4, 0), depolarizing_constant=0.98)
        meas_ch = MeasurementChannel.from_readout_fidelity(inst=MEASURE(1, None), fidelity=0.95)
        nm = NoiseModel.from_channels([ch, meas_ch])

        restored = NoiseModel.from_json(nm.to_json())

        assert restored == nm
        assert set(restored.channels) == {ch.inst, meas_ch.inst}

    def test_add_combines_disjoint_channels(self):
        """NoiseModel addition preserves disjoint channels from both operands."""
        ch1 = Channel.from_depolarizing_constant(inst=RX(np.pi / 4, 0), depolarizing_constant=0.98)
        ch2 = Channel.from_depolarizing_constant(inst=RY(np.pi / 4, 1), depolarizing_constant=0.97)

        combined = NoiseModel.from_channels([ch1]) + NoiseModel.from_channels([ch2])

        assert combined.get_channel(ch1.inst) == ch1
        assert combined.get_channel(ch2.inst) == ch2

    def test_add_rejects_overlapping_channels(self):
        """Addition is a disjoint union; a shared instruction is a conflict, not a composition."""
        inst = RX(np.pi / 4, 0)
        ch1 = Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=0.98)
        ch2 = Channel.from_depolarizing_constant(inst=inst, depolarizing_constant=0.97)

        with pytest.raises(ValueError, match="same instruction"):
            _ = NoiseModel.from_channels([ch1]) + NoiseModel.from_channels([ch2])

    def test_with_channels_returns_extended_model(self):
        """with_channels returns a new model and rejects duplicate instructions."""
        ch1 = Channel.from_depolarizing_constant(inst=RX(np.pi / 4, 0), depolarizing_constant=0.98)
        ch2 = Channel.from_depolarizing_constant(inst=RY(np.pi / 4, 1), depolarizing_constant=0.97)
        nm = NoiseModel.from_channels([ch1])

        extended = nm.with_channels([ch2])

        assert nm.get_channel(ch2.inst) is None
        assert extended.get_channel(ch1.inst) is ch1
        assert extended.get_channel(ch2.inst) is ch2
        with pytest.raises(ValueError, match="Duplicate noise channel"):
            nm.with_channels([ch1])

    def test_noise_model_is_unhashable(self):
        """NoiseModel is unhashable (consistent with value-based equality)."""
        nm = NoiseModel.from_channels([Channel.from_depolarizing_constant(RX(0.5, 0), 0.98)])
        with pytest.raises(TypeError):
            hash(nm)


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

    def test_get_custom_gates_from_program_qutrit(self):
        """get_custom_gates_from_program infers qudit (base, exponent) dims, not just qubit dims."""
        qutrit_x = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
        program = Program()
        program.defgate("QUTRIT_X", qutrit_x)

        custom_gates = get_custom_gates_from_program(program)
        unitary = custom_gates["QUTRIT_X"]
        assert isinstance(unitary, qx.Unitary)
        # A single qutrit: dims are ((3,), (3,)), not the qubit-only ((2, 2), (2, 2)).
        assert unitary.dims == ((3,), (3,))
        assert np.allclose(np.asarray(unitary.matrix), qutrit_x)

    def test_resolve_params_real(self):
        """_resolve_params returns concrete floats for real parameters."""
        assert _resolve_params([1.5, 2]) == [1.5, 2.0]

    def test_resolve_params_warns_on_imaginary(self, caplog):
        """A non-negligible imaginary part is dropped with a warning, keeping the real part."""
        with caplog.at_level("WARNING"):
            resolved = _resolve_params([1.5 + 0.3j])
        assert resolved == [1.5]
        assert "imaginary part" in caplog.text


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

    def test_global_reset_channel_rejected(self):
        """ResetChannel is intentionally scoped to targeted resets."""
        with pytest.raises(TypeError, match="targeted"):
            ResetChannel.from_reset_fidelity(inst=RESET(), fidelity=1.0)

    def test_json_roundtrip_preserves_qutrit_dims(self):
        """ResetChannel JSON includes explicit dims for non-qubit resets."""
        channel = ResetChannel.from_reset_fidelity(inst=ResetQubit(0), fidelity=0.9, dim=3)

        restored = ResetChannel.from_json(channel.to_json())

        assert restored.inst == channel.inst
        assert restored.process.dims == ((3,), (3,))
        assert jnp.allclose(restored.process.matrix, channel.process.matrix)


# ──────────────────────────────────────────────────────────
# Channel equality / hashing semantics
# ──────────────────────────────────────────────────────────


class TestChannelEqualityAndHashing:
    def test_channels_are_unhashable(self):
        """Channels hold jax arrays and are intentionally unhashable."""
        ch = Channel.from_depolarizing_constant(RX(0.5, 0), 0.98)
        with pytest.raises(TypeError):
            hash(ch)

    def test_channel_equality_is_exact(self):
        """Channel equality is exact: identical builds are equal, different ones are not."""
        ch1 = Channel.from_depolarizing_constant(RX(0.5, 0), 0.98)
        ch2 = Channel.from_depolarizing_constant(RX(0.5, 0), 0.98)
        ch3 = Channel.from_depolarizing_constant(RX(0.5, 0), 0.97)
        assert ch1 == ch2
        assert ch1 != ch3
        assert ch1 != "not a channel"

    def test_channel_inequality_on_different_instruction(self):
        """Channels on different instructions are never equal."""
        ch_a = Channel.from_depolarizing_constant(RX(0.5, 0), 0.98)
        ch_b = Channel.from_depolarizing_constant(RX(0.5, 1), 0.98)
        assert ch_a != ch_b


# ──────────────────────────────────────────────────────────
# Channel construction / analysis coverage
# ──────────────────────────────────────────────────────────


class TestChannelAnalysis:
    def test_from_mixture(self):
        """from_mixture builds a noisy channel from unitary errors with probabilities."""
        z = qx.Unitary.from_matrix(jnp.array([[1, 0], [0, -1]], dtype=complex), ((2,), (2,)))
        ch = Channel.from_mixture(X(0), constituents=[z], probabilities=[0.1])
        assert isinstance(ch.process, qx.SuperOp)
        assert ch.pauli_infidelity > 0.0

    def test_coherent_and_stochastic_decomposition(self):
        """to_coherent_channel / to_stochastic_channel split the noise into components."""
        ch = Channel.from_random_coherent_error(X(0), process_fidelity=0.95, rng=np.random.default_rng(0))
        coherent = ch.to_coherent_channel()
        stochastic = ch.to_stochastic_channel()
        assert isinstance(coherent, Channel)
        assert isinstance(stochastic, Channel)
        # Coherent + stochastic infidelity decomposition is non-negative and finite.
        assert np.isfinite(ch.coherent_infidelity)
        assert np.isfinite(ch.stochastic_infidelity)

    def test_pauli_twirl_is_pauli(self):
        """Twirling a coherent-error channel yields a stochastic Pauli channel."""
        ch = Channel.from_random_coherent_error(X(0), process_fidelity=0.95, rng=np.random.default_rng(1))
        twirled = ch.pauli_twirl()
        assert twirled.is_pauli()


# ──────────────────────────────────────────────────────────
# NoiseModel.from_isa
# ──────────────────────────────────────────────────────────


class TestFromIsa:
    @staticmethod
    def _isa() -> CompilerISA:
        return CompilerISA.parse_obj(
            {
                "1Q": {
                    "0": {
                        "id": 0,
                        "gates": [
                            {
                                "operator_type": "gate",
                                "operator": "RX",
                                "parameters": [1.5707963267948966],
                                "arguments": ["_"],
                                "fidelity": 0.99,
                            },
                            # A fidelity-less measurement entry must not mask the real one below.
                            {"operator_type": "measure", "qubit": "0", "fidelity": None},
                            {"operator_type": "measure", "qubit": "0", "fidelity": 0.95},
                        ],
                    },
                    "1": {"id": 1, "gates": []},
                },
                "2Q": {
                    "0-1": {
                        "ids": [0, 1],
                        "gates": [
                            {
                                "operator_type": "gate",
                                "operator": "CZ",
                                "parameters": [],
                                "arguments": ["_", "_"],
                                "fidelity": 0.9,
                            }
                        ],
                    }
                },
            }
        )

    def test_builds_gate_and_edge_channels(self):
        nm = NoiseModel.from_isa(self._isa())
        assert isinstance(nm.get_channel(Gate("RX", [1.5707963267948966], [0])), Channel)
        assert isinstance(nm.get_channel(Gate("CZ", [], [0, 1])), Channel)

    def test_measurement_dedup_prefers_real_fidelity(self):
        """A None-fidelity measure entry must not block a later usable one (dedup ordering)."""
        nm = NoiseModel.from_isa(self._isa())
        channel = nm.get_channel(Measurement(qubit=Qubit(0), classical_reg=None))
        assert isinstance(channel, MeasurementChannel)


class TestCycleChannel:
    def test_complete_cycle_constructs(self):
        """A CycleChannel whose channels cover every DefCircuit body instruction is valid."""
        channels = tuple(
            Channel.from_depolarizing_constant(inst, depolarizing_constant=0.99) for inst in (RX(0.1, 0), RZ(0.2, 1))
        )
        cycle = _build_cycle_channel(list(channels))
        assert cycle.channels == channels

    def test_incomplete_cycle_rejected(self):
        """A body instruction with no corresponding channel is a footgun and must raise."""
        q0, q1 = FormalArgument("q0"), FormalArgument("q1")
        # DefCircuit body has two gates, but only one channel is supplied for q0.
        defcircuit = DefCircuit("CYCLE", [], [q0, q1], [RX(0.1, q0), RZ(0.2, q1)])
        cycle_inst = Gate("CYCLE", [], [Qubit(0), Qubit(1)])
        channels = (Channel.from_depolarizing_constant(RX(0.1, 0), depolarizing_constant=0.99),)

        with pytest.raises(ValueError, match="incomplete"):
            _ = CycleChannel(inst=cycle_inst, defcircuit=defcircuit, channels=channels)
