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

import pickle
from dataclasses import replace

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
    SuperopChannel,
    SuperopResetChannel,
    _build_cycle_channel,
    _ChannelBase,
    _evaluate_parameter_designators,
    _operator_dims_from_dimension,
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
        expected_fidelity = qx.depolarizing_constant_to_average_fidelity(0.98, dims=(2,))
        np.testing.assert_allclose(ch.average_gate_fidelity, expected_fidelity, rtol=1e-4)

    def test_from_gate_fidelity(self):
        """Channel.from_gate_fidelity produces correct fidelity."""
        inst = RX(np.pi / 2, 0)
        ch = Channel.from_gate_fidelity(inst=inst, fidelity=0.99)
        assert abs(ch.average_gate_fidelity - 0.99) < 0.001

    def test_from_pauli_fidelity(self):
        """Channel.from_pauli_fidelity produces a valid channel."""
        inst = X(0)
        ch = Channel.from_pauli_fidelity(inst=inst, pauli_fidelity=0.97)
        assert isinstance(ch.process, qx.SuperOp)
        assert ch.process_fidelity == pytest.approx(0.97, abs=0.001)

    def test_from_pauli_generators(self):
        """Channel.from_pauli_generators produces a valid Pauli-dissipation channel."""
        inst = RX(0.5, 0)
        ch = Channel.from_pauli_generators(inst=inst, pauli_generators={"X": 0.01, "Z": 0.02})
        assert isinstance(ch.process, qx.SuperOp)
        assert ch.average_gate_fidelity < 1.0

    def test_from_coherence_times(self):
        """Channel.from_coherence_times produces a valid decoherence channel."""
        inst = RX(np.pi / 4, 0)
        ch = Channel.from_coherence_times(inst=inst, gate_duration=40e-9, t1s=[30e-6], t2s=[20e-6])
        assert isinstance(ch.process, qx.SuperOp)
        assert ch.average_gate_fidelity < 1.0
        assert ch.average_gate_fidelity > 0.99  # short gate relative to T1/T2

    def test_qubits(self):
        """SuperopChannel.qubits reflects the instruction's qubits."""
        ch = Channel.from_depolarizing_constant(inst=RX(0.1, 3), depolarizing_constant=0.99)
        assert ch.qubits == [3]

    def test_num_qubits(self):
        """SuperopChannel.num_qubits is correct for 2Q gates."""
        ch = Channel.from_depolarizing_constant(inst=CNOT(0, 1), depolarizing_constant=0.95)
        assert ch.num_qubits == 2

    def test_fidelity_properties(self):
        """Fidelity, infidelity, pauli_fidelity, pauli_infidelity are consistent."""
        ch = Channel.from_gate_fidelity(inst=RX(0.3, 0), fidelity=0.98)
        assert ch.average_gate_fidelity + ch.average_gate_infidelity == pytest.approx(1.0)
        assert ch.process_fidelity + ch.process_infidelity == pytest.approx(1.0)
        assert ch.stochastic_fidelity + ch.stochastic_infidelity == pytest.approx(1.0)
        assert ch.coherent_fidelity + ch.coherent_infidelity == pytest.approx(1.0)

    def test_error_process(self):
        """error_process factors out the ideal gate unitary."""
        ch = Channel.from_depolarizing_constant(inst=RX(np.pi / 4, 0), depolarizing_constant=0.99)
        noise = ch.error_process
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
        pv = ch.to_pauli_vector()
        assert float(jnp.sum(pv)) == pytest.approx(1.0, abs=1e-8)

    def test_perfect_channel(self):
        """A depolarizing constant of 1.0 should give fidelity 1.0."""
        ch = Channel.from_depolarizing_constant(inst=RX(np.pi, 0), depolarizing_constant=1.0)
        assert ch.average_gate_fidelity == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("depolarizing_constant", [-0.5, 1.5])
    def test_from_depolarizing_constant_rejects_out_of_range(self, depolarizing_constant):
        """An out-of-range shrink factor is a caller error, not something to clip silently.

        Regression guard: values were clipped into [tiny, 1], so ``p = -0.5`` quietly produced a
        channel with fidelity 0.25 and ``p = 1.5`` a noiseless one.
        """
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            Channel.from_depolarizing_constant(X(0), depolarizing_constant=depolarizing_constant)

    @pytest.mark.parametrize("depolarizing_constant", [1.0, 0.99, 0.5, 0.0])
    def test_from_depolarizing_constant_shrinks_by_exactly_p(self, depolarizing_constant):
        """The error PTM's traceless diagonal must equal the requested shrink factor."""
        ch = Channel.from_depolarizing_constant(X(0), depolarizing_constant=depolarizing_constant)
        ptm = np.asarray(qx.to_pauli_liouville(ch.error_process).matrix).real
        np.testing.assert_allclose(np.diag(ptm), [1.0] + [depolarizing_constant] * 3, atol=1e-9)

    def test_gate_time_scales_the_noise(self):
        """Rates are per unit time, so a longer gate at fixed rates accumulates more noise."""
        short = Channel.from_pauli_generators(RX(np.pi / 2, 0), {"Z": 0.01}, gate_time=1.0)
        long = Channel.from_pauli_generators(RX(np.pi / 2, 0), {"Z": 0.01}, gate_time=10.0)
        assert long.process_fidelity < short.process_fidelity
        # The gate itself is unchanged: the Hamiltonian is rescaled to reproduce it either way.
        for ch in (short, long):
            ideal = replace(ch, lindbladian=ch.target_lindbladian)
            assert ideal.process_infidelity == pytest.approx(0.0, abs=1e-6)

    def test_from_pauli_generators_rejects_negative_rate(self):
        """Pauli generator rates must be non-negative (they are Lindbladian rates, not probs)."""
        with pytest.raises(ValueError, match="negative"):
            Channel.from_pauli_generators(inst=RX(0.5, 0), pauli_generators={"X": -0.1})

    def test_from_pauli_generators_rejects_wrong_length(self):
        """A Pauli term must have one character per qubit."""
        with pytest.raises(ValueError, match="length"):
            Channel.from_pauli_generators(inst=RX(0.5, 0), pauli_generators={"XX": 0.1})

    def test_from_pauli_generators_two_qubit(self):
        """from_pauli_generators builds a valid 16-term Pauli-error vector for a 2Q gate.

        The rates are Lindbladian generator rates, so the exponentiated channel does not reproduce
        the input rates exactly; we check the vector is a normalized distribution whose weight sits
        on the identity and the specified error terms.
        """
        pauli_generators = {"IX": 0.01, "XI": 0.005, "ZZ": 0.02}
        ch = Channel.from_pauli_generators(inst=CNOT(0, 1), pauli_generators=pauli_generators)
        pv = np.asarray(ch.to_pauli_vector())
        assert pv.size == 16
        assert float(jnp.sum(ch.to_pauli_vector())) == pytest.approx(1.0, abs=1e-3)
        terms = [a + b for a in "IXYZ" for b in "IXYZ"]
        rates = dict(zip(terms, pv, strict=True))
        # Identity keeps the dominant weight; each specified error term is populated.
        assert rates["II"] > 0.9
        for term in pauli_generators:
            assert rates[term] > 0.0
        # An unspecified error term carries negligible weight (only tiny higher-order cross terms).
        assert rates["YY"] < 1e-3

    def test_superop_from_pauli_noise_reproduces_exact_probabilities(self):
        """SuperopChannel.from_pauli_noise is the one-shot post-gate model: exact error probabilities."""
        pauli_noise = {"IX": 0.01, "XI": 0.005, "ZZ": 0.02}
        ch = SuperopChannel.from_pauli_noise(inst=CNOT(0, 1), pauli_noise=pauli_noise)
        assert isinstance(ch, SuperopChannel)
        pv = np.asarray(ch.to_pauli_vector())
        assert pv.size == 16
        assert float(jnp.sum(pv)) == pytest.approx(1.0, abs=1e-6)
        terms = [a + b for a in "IXYZ" for b in "IXYZ"]
        rates = dict(zip(terms, pv, strict=True))
        for term, rate in pauli_noise.items():
            assert rates[term] == pytest.approx(rate, abs=1e-3)
        assert rates["II"] == pytest.approx(1.0 - sum(pauli_noise.values()), abs=1e-3)

    def test_superop_from_pauli_noise_rejects_excessive_probability(self):
        """The one-shot model requires probabilities summing to at most 1."""
        with pytest.raises(ValueError, match="at most 1.0"):
            SuperopChannel.from_pauli_noise(inst=RX(0.5, 0), pauli_noise={"X": 0.6, "Z": 0.5})

    def test_json_roundtrip_preserves_qutrit_dims(self):
        """SuperopChannel JSON includes explicit dims for non-qubit operators."""
        qutrit_x = jnp.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=complex,
        )
        target_unitary = qx.Unitary.from_matrix(qutrit_x, ((3,), (3,)))
        channel = SuperopChannel(
            inst=Gate("TX", [], [0]), process=qx.to_superop(target_unitary), ideal_unitary=target_unitary
        )

        restored = SuperopChannel.from_json(channel.to_json())

        assert restored.inst == channel.inst
        assert restored.process.dims == ((3,), (3,))
        assert restored.ideal_unitary.dims == ((3,), (3,))
        assert jnp.allclose(restored.process.matrix, channel.process.matrix)


# ──────────────────────────────────────────────────────────
# MeasurementChannel tests
# ──────────────────────────────────────────────────────────


class TestMeasurementChannel:
    def test_from_readout_fidelity(self):
        """MeasurementChannel.from_readout_fidelity produces a valid quantum instrument."""
        meas_inst = MEASURE(0, None)
        ch = MeasurementChannel.from_readout_fidelity(inst=meas_inst, fidelity=0.95)
        assert isinstance(ch.process, qx.QuantumInstrument)

    def test_from_readout_fidelity_with_asymmetry(self):
        """MeasurementChannel with asymmetry produces asymmetric confusion."""
        meas_inst = MEASURE(0, None)
        ch = MeasurementChannel.from_readout_fidelity(inst=meas_inst, fidelity=0.95, asymmetry=0.5)
        assert isinstance(ch.process, qx.QuantumInstrument)

    def test_qubits(self):
        """MeasurementChannel.qubits returns the correct qubit."""
        meas_inst = MEASURE(5, None)
        ch = MeasurementChannel.from_readout_fidelity(inst=meas_inst, fidelity=0.99)
        assert ch.qubits == [5]

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_from_readout_fidelity_average_diagonal_is_the_fidelity(self, dim):
        """``fidelity`` is the average over levels of P(outcome j | prepared j)."""
        ch = MeasurementChannel.from_readout_fidelity(inst=MEASURE(0, None), fidelity=0.9, dim=dim)
        confusion = np.asarray(ch.confusion_matrix)
        assert np.diag(confusion).mean() == pytest.approx(0.9, abs=1e-9)
        np.testing.assert_allclose(confusion.sum(axis=0), np.ones(dim), atol=1e-9)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"fidelity": 0.9, "dim": 1}, "at least 2"),
            ({"fidelity": 1.5}, r"fidelity must lie in \[0, 1\]"),
            ({"fidelity": -0.2}, r"fidelity must lie in \[0, 1\]"),
            ({"fidelity": 0.9, "asymmetry": 3.0}, r"asymmetry must lie in \[-1, 1\]"),
            ({"fidelity": 0.4, "asymmetry": 1.0}, "cannot be realized"),
        ],
        ids=["dim", "fidelity-high", "fidelity-low", "asymmetry", "joint-constraint"],
    )
    def test_from_readout_fidelity_validates_its_arguments(self, kwargs, match):
        """Bad arguments get a message naming the parameter, not a downstream quax complaint."""
        with pytest.raises(ValueError, match=match):
            MeasurementChannel.from_readout_fidelity(inst=MEASURE(0, None), **kwargs)

    def test_from_binary_discriminator_qubit_is_faithful_readout(self):
        """dim=2/threshold=1 must be a faithful qubit readout: |0> -> 0 and |1> -> 1.

        The failure mode this guards against is a discriminator that maps every level to outcome
        0, which erases all measurement information while still looking like a valid instrument.
        """
        meas_inst = MEASURE(0, None)
        ch = MeasurementChannel.from_binary_discriminator(inst=meas_inst, dim=2, threshold=1)
        cm = np.asarray(ch.confusion_matrix)
        assert cm.shape == (2, 2)  # two reachable outcomes
        assert np.allclose(cm, np.eye(2))  # |0> -> 0, |1> -> 1

    def test_from_binary_discriminator_qutrit_split(self):
        """dim=3 splits into exactly two outcomes at the threshold."""
        meas_inst = MEASURE(0, None)
        # threshold=2: {0,1} -> 0, {2} -> 1 (flag leakage only)
        cm2 = np.asarray(
            MeasurementChannel.from_binary_discriminator(inst=meas_inst, dim=3, threshold=2).confusion_matrix
        )
        assert np.allclose(cm2, [[1, 1, 0], [0, 0, 1]])
        # threshold=1: {0} -> 0, {1,2} -> 1 (ground vs excited-or-leaked)
        cm1 = np.asarray(
            MeasurementChannel.from_binary_discriminator(inst=meas_inst, dim=3, threshold=1).confusion_matrix
        )
        assert np.allclose(cm1, [[1, 0, 0], [0, 1, 1]])

    def test_from_binary_discriminator_fidelity_degrades(self):
        """Sub-unit fidelity stays column-stochastic and keeps both outcomes reachable."""
        meas_inst = MEASURE(0, None)
        cm = np.asarray(
            MeasurementChannel.from_binary_discriminator(
                inst=meas_inst, dim=2, threshold=1, fidelity=0.9
            ).confusion_matrix
        )
        assert cm.shape == (2, 2)
        assert np.allclose(cm.sum(axis=0), 1.0)
        assert cm[1, 1] > cm[0, 1]  # |1> still most likely reads as outcome 1

    def test_from_binary_discriminator_rejects_bad_threshold(self):
        """threshold must satisfy 1 <= threshold < dim."""
        meas_inst = MEASURE(0, None)
        with pytest.raises(ValueError, match="threshold"):
            MeasurementChannel.from_binary_discriminator(inst=meas_inst, dim=2, threshold=2)

    def test_json_roundtrip_preserves_qutrit_dims(self):
        """MeasurementChannel JSON includes explicit dims for non-qubit instruments."""
        meas_inst = MEASURE(0, None)
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

        gate = RX(np.pi / 4, 0)
        meas_inst = MEASURE(0, None)
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
        """NoiseModel.get_channel returns the correct SuperopChannel for a gate."""
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

    def test_evaluate_parameter_designators_real(self):
        """_evaluate_parameter_designators returns concrete floats for real parameters."""
        assert _evaluate_parameter_designators([1.5, 2]) == [1.5, 2.0]

    def test_evaluate_parameter_designators_rejects_imaginary(self):
        """A non-negligible imaginary part is a caller error, not something to truncate silently."""
        with pytest.raises(ValueError, match="imaginary part"):
            _evaluate_parameter_designators([1.5 + 0.3j])

    def test_evaluate_parameter_designators_tolerates_numerical_imaginary(self):
        """A round-off-scale imaginary part is dropped without complaint."""
        assert _evaluate_parameter_designators([1.5 + 1e-18j]) == [1.5]


# ──────────────────────────────────────────────────────────
# SuperopResetChannel tests
# ──────────────────────────────────────────────────────────


class TestSuperopResetChannel:
    def test_from_reset_fidelity_perfect(self):
        """A perfect reset (fidelity=1.0) should produce a valid superoperator."""
        inst = RESET(0)
        ch = SuperopResetChannel.from_reset_fidelity(inst=inst, fidelity=1.0)
        assert isinstance(ch.process, qx.SuperOp)

    def test_from_reset_fidelity_noisy(self):
        """A noisy reset should have lower fidelity than a perfect one."""
        inst = RESET(0)
        ch_perfect = SuperopResetChannel.from_reset_fidelity(inst=inst, fidelity=1.0)
        ch_noisy = SuperopResetChannel.from_reset_fidelity(inst=inst, fidelity=0.95)
        assert isinstance(ch_noisy.process, qx.SuperOp)
        assert ch_noisy.process_fidelity < ch_perfect.process_fidelity

    @pytest.mark.parametrize("dim", [2, 3, 4])
    @pytest.mark.parametrize("fidelity", [1.0, 0.99, 0.9, 0.75, 0.5001])
    def test_from_reset_fidelity_is_the_resulting_fidelity(self, dim, fidelity):
        """The requested fidelity must be the fidelity the channel reports back.

        Regression guard: the argument used to be applied directly as a depolarizing shrink
        factor, so ``from_reset_fidelity(0.9).process_fidelity`` came out as 0.95. The ideal reset
        is rank one, giving F = (1 + (d - 1) p) / d rather than F = p.
        """
        ch = SuperopResetChannel.from_reset_fidelity(inst=ResetQubit(0), fidelity=fidelity, dim=dim)
        assert ch.process_fidelity == pytest.approx(fidelity, abs=1e-9)

    def test_from_reset_fidelity_rejects_out_of_range(self):
        """A fully depolarized reset already has process fidelity 1/d; below that is unreachable."""
        with pytest.raises(ValueError, match="must lie in"):
            SuperopResetChannel.from_reset_fidelity(inst=ResetQubit(0), fidelity=0.4)
        with pytest.raises(ValueError, match="must lie in"):
            SuperopResetChannel.from_reset_fidelity(inst=ResetQubit(0), fidelity=1.5)

    def test_qubits(self):
        """SuperopResetChannel.qubits returns the correct qubit."""
        inst = RESET(3)
        ch = SuperopResetChannel.from_reset_fidelity(inst=inst, fidelity=0.99)
        assert ch.qubits == [3]

    def test_global_reset_channel_rejected(self):
        """SuperopResetChannel is intentionally scoped to targeted resets."""
        with pytest.raises(TypeError, match="targeted"):
            SuperopResetChannel.from_reset_fidelity(inst=RESET(), fidelity=1.0)

    def test_json_roundtrip_preserves_qutrit_dims(self):
        """SuperopResetChannel JSON includes explicit dims for non-qubit resets."""
        channel = SuperopResetChannel.from_reset_fidelity(inst=ResetQubit(0), fidelity=0.9, dim=3)

        restored = SuperopResetChannel.from_json(channel.to_json())

        assert restored.inst == channel.inst
        assert restored.process.dims == ((3,), (3,))
        assert jnp.allclose(restored.process.matrix, channel.process.matrix)


# ──────────────────────────────────────────────────────────
# SuperopChannel equality / hashing semantics
# ──────────────────────────────────────────────────────────


class TestChannelEqualityAndHashing:
    def test_channels_are_unhashable(self):
        """Channels hold jax arrays and are intentionally unhashable."""
        ch = Channel.from_depolarizing_constant(RX(0.5, 0), 0.98)
        with pytest.raises(TypeError):
            hash(ch)

    def test_channel_equality_is_elementwise_close(self):
        """Channel equality is element-wise ``jnp.allclose`` on the process/unitary matrices.

        Identical builds are equal, distinct noise is not, and a JSON round-trip stays equal.
        """
        ch1 = Channel.from_depolarizing_constant(RX(0.5, 0), 0.98)
        ch2 = Channel.from_depolarizing_constant(RX(0.5, 0), 0.98)
        ch3 = Channel.from_depolarizing_constant(RX(0.5, 0), 0.97)
        assert ch1 == ch2
        assert ch1 != ch3
        assert ch1 != "not a channel"
        assert ch1 == Channel.from_json(ch1.to_json())

    def test_channel_inequality_on_different_instruction(self):
        """Channels on different instructions are never equal."""
        ch_a = Channel.from_depolarizing_constant(RX(0.5, 0), 0.98)
        ch_b = Channel.from_depolarizing_constant(RX(0.5, 1), 0.98)
        assert ch_a != ch_b


# ──────────────────────────────────────────────────────────
# SuperopChannel construction / analysis coverage
# ──────────────────────────────────────────────────────────


class TestChannelAnalysis:
    def test_from_mixture(self):
        """from_mixture builds a noisy channel from unitary errors with probabilities."""
        z = qx.Unitary.from_matrix(jnp.array([[1, 0], [0, -1]], dtype=complex), ((2,), (2,)))
        ch = Channel.from_mixture(X(0), constituents=[z], rates=[0.1])
        assert isinstance(ch.process, qx.SuperOp)
        assert ch.process_infidelity > 0.0

    def test_coherent_and_stochastic_decomposition(self):
        """to_coherent_channel / to_stochastic_channel split the noise into components."""
        ch = Channel.from_random_coherent_error(X(0), process_fidelity=0.95, rng=np.random.default_rng(0))
        coherent = ch.to_coherent_channel()
        stochastic = ch.to_stochastic_channel()
        assert isinstance(coherent, SuperopChannel)
        assert isinstance(stochastic, SuperopChannel)
        # Coherent + stochastic infidelity decomposition is non-negative and finite.
        assert np.isfinite(ch.coherent_infidelity)
        assert np.isfinite(ch.stochastic_infidelity)

    def test_stochastic_channel_is_cptp_and_matches_stochastic_fidelity(self):
        """The stochastic component must be a physical channel carrying the stochastic fidelity.

        Regression guard: the decomposition previously built its superoperators with hand-rolled
        Kronecker products in the wrong convention, yielding a non-CP, non-trace-preserving
        channel whose fidelity bore no relation to ``stochastic_fidelity``.
        """
        ch = Channel.from_random_coherent_error(
            RX(np.pi / 2, 0), process_fidelity=0.99, rng=np.random.default_rng(7)
        ) + Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.995)
        stochastic = ch.to_stochastic_channel()

        # Completely positive: the Choi matrix is positive semi-definite.
        choi_eigenvalues = np.linalg.eigvalsh(np.asarray(qx.to_choi(stochastic.process).matrix))
        assert choi_eigenvalues.min() > -1e-9

        # Trace preserving: the first row of the Pauli transfer matrix is (1, 0, ..., 0).
        ptm = np.asarray(qx.to_pauli_liouville(stochastic.process).matrix)
        np.testing.assert_allclose(ptm[0].real, [1.0, 0.0, 0.0, 0.0], atol=1e-9)

        # The stochastic part carries exactly the channel's stochastic fidelity.
        assert stochastic.process_fidelity == pytest.approx(ch.stochastic_fidelity, abs=1e-5)

    def test_coherent_channel_matches_coherent_fidelity_and_is_unitary(self):
        """The coherent component is a unitary channel carrying the coherent fidelity."""
        ch = Channel.from_random_coherent_error(
            RX(np.pi / 2, 0), process_fidelity=0.99, rng=np.random.default_rng(7)
        ) + Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.995)
        coherent = ch.to_coherent_channel()
        assert coherent.unitarity == pytest.approx(1.0, abs=1e-6)
        assert coherent.process_fidelity == pytest.approx(ch.coherent_fidelity, abs=1e-3)

    def test_pauli_twirl_is_pauli(self):
        """Twirling a coherent-error channel yields a stochastic Pauli channel."""
        ch = Channel.from_random_coherent_error(X(0), process_fidelity=0.95, rng=np.random.default_rng(1))
        twirled = ch.pauli_twirl()
        assert twirled.is_pauli()

    @pytest.mark.parametrize("gate", [X(0), RX(np.pi / 2, 0), CNOT(0, 1)], ids=lambda g: g.out())
    def test_pauli_twirl_preserves_the_gate(self, gate):
        """Twirling must randomize the error and leave the gate intact.

        Regression guard: the twirl used to be applied to the full process, gate included, which
        for any gate whose transfer matrix is not already diagonal (e.g. ``RX(pi/2)``) deleted the
        gate itself and roughly halved the process fidelity.
        """
        ch = Channel.from_random_coherent_error(
            gate, process_fidelity=0.97, rng=np.random.default_rng(3)
        ) + Channel.from_depolarizing_constant(gate, 0.99)
        twirled = ch.pauli_twirl()

        # The error is now a stochastic Pauli channel...
        assert twirled.is_pauli()
        # ...the gate survives...
        assert jnp.allclose(twirled.ideal_unitary.matrix, ch.ideal_unitary.matrix)
        # ...and twirling preserves process fidelity (to PTM round-trip precision). The bug this
        # guards against moved the fidelity by ~0.5, so this bound is far tighter than needed.
        assert twirled.process_fidelity == pytest.approx(ch.process_fidelity, abs=1e-7)

    def test_pauli_twirl_matches_explicit_clifford_twirl(self):
        """For a Clifford gate, twirling the error equals the textbook twirl of the whole channel.

        The textbook Pauli twirl of a noisy gate is
        ``(1/4^n) sum_k S(P'_k)^dag @ E @ S(P_k)`` with ``P'_k = U P_k U^dag``. Because
        ``U P_k = P'_k U``, the gate factors out of the average, and for Clifford ``U`` the
        conjugated set ``{P'_k}`` is again the Pauli group -- so the two agree exactly.
        """
        gate = RX(np.pi / 2, 0)
        ch = Channel.from_random_coherent_error(
            gate, process_fidelity=0.97, rng=np.random.default_rng(4)
        ) + Channel.from_depolarizing_constant(gate, 0.99)

        unitary = np.asarray(ch.ideal_unitary.matrix)
        process = np.asarray(ch.process.matrix)
        paulis = np.asarray(qx.ensembles.PAULIS.matrix)

        def superop(matrix: np.ndarray) -> np.ndarray:
            # quax's SuperOp convention is kron(conj(U), U).
            return np.kron(matrix.conj(), matrix)

        explicit = np.zeros_like(process)
        for pauli in paulis:
            conjugated = unitary @ pauli @ unitary.conj().T
            explicit += superop(conjugated).conj().T @ process @ superop(pauli)
        explicit /= len(paulis)

        assert jnp.allclose(ch.pauli_twirl().process.matrix, jnp.asarray(explicit), atol=1e-12)


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
        nm = NoiseModel.from_compiler_isa(self._isa())
        assert isinstance(nm.get_channel(Gate("RX", [1.5707963267948966], [0])), Channel)
        assert isinstance(nm.get_channel(Gate("CZ", [], [0, 1])), Channel)

    def test_measurement_dedup_prefers_real_fidelity(self):
        """A None-fidelity measure entry must not block a later usable one (dedup ordering)."""
        nm = NoiseModel.from_compiler_isa(self._isa())
        channel = nm.get_channel(Measurement(qubit=Qubit(0), classical_reg=None))
        assert isinstance(channel, MeasurementChannel)

    def test_perfect_fidelities_get_no_channel(self):
        """A fidelity of 1.0 means ideal, for readout as well as for gates."""
        isa = CompilerISA.parse_obj(
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
                                "fidelity": 1.0,
                            },
                            {"operator_type": "measure", "qubit": "0", "fidelity": 1.0},
                        ],
                    }
                },
                "2Q": {},
            }
        )
        nm = NoiseModel.from_compiler_isa(isa)
        assert nm.get_channel(Gate("RX", [1.5707963267948966], [0])) is None
        assert nm.get_channel(Measurement(qubit=Qubit(0), classical_reg=None)) is None

    def test_from_isa_accepts_a_qcs_instruction_set_architecture(self, qcs_aspen8_isa):
        """The preferred entry point takes the qcs_sdk ISA, not the rpcq CompilerISA."""
        nm = NoiseModel.from_isa(qcs_aspen8_isa)
        assert len(nm.channels) > 0
        assert all(isinstance(ch, (Channel, MeasurementChannel)) for ch in nm.channels.values())

    def test_from_isa_matches_from_compiler_isa(self, qcs_aspen8_isa, aspen8_compiler_isa):
        """Both ISA entry points describe the same device, so they must agree."""
        assert NoiseModel.from_isa(qcs_aspen8_isa) == NoiseModel.from_compiler_isa(aspen8_compiler_isa)


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

    def test_cycle_can_contain_reset_channel(self):
        """A cycle may mix a gate channel with a reset channel on disjoint qubits."""
        gate = Channel.from_depolarizing_constant(RX(0.1, 0), depolarizing_constant=0.99)
        reset = ResetChannel.from_amplitude_damping(ResetQubit(1), gamma=0.5)
        cycle = gate | reset
        assert isinstance(cycle, CycleChannel)
        assert cycle.channels == (gate, reset)
        assert set(cycle.qubits) == {0, 1}

    def test_reset_channel_or_from_reset_side(self):
        """`reset | other` builds a cycle from the reset channel's side too."""
        reset = SuperopResetChannel.from_reset_fidelity(ResetQubit(0), fidelity=0.95)
        measure = MeasurementChannel.from_readout_fidelity(MEASURE(1, None), fidelity=0.98)
        cycle = reset | measure
        assert isinstance(cycle, CycleChannel)
        assert cycle.channels == (reset, measure)

    def test_cycle_with_reset_json_roundtrip(self):
        """A cycle containing a reset channel survives a JSON round-trip."""
        gate = Channel.from_depolarizing_constant(RX(0.1, 0), depolarizing_constant=0.99)
        reset = ResetChannel.from_amplitude_damping(ResetQubit(1), gamma=0.5)
        cycle = gate | reset
        assert CycleChannel.from_json(cycle.to_json()) == cycle

    def test_json_roundtrip_preserves_externally_built_defcircuit(self):
        """A cycle built elsewhere keeps its own name, formal arguments, and body ordering.

        Regression guard: serialization used to write only the constituent channels, so
        deserialization rebuilt every cycle as a generic ``CYCLE`` on sorted qubits, silently
        discarding a DEFCIRCUIT constructed by a caller.
        """
        qa, qb = FormalArgument("a"), FormalArgument("b")
        defcircuit = DefCircuit("SYNDROME_ROUND", [], [qa, qb], [RZ(0.2, qb), RX(0.1, qa)])
        cycle_inst = Gate("SYNDROME_ROUND", [], [Qubit(5), Qubit(3)])
        channels = (
            Channel.from_depolarizing_constant(RZ(0.2, 3), depolarizing_constant=0.99),
            Channel.from_depolarizing_constant(RX(0.1, 5), depolarizing_constant=0.98),
        )
        cycle = CycleChannel(inst=cycle_inst, defcircuit=defcircuit, channels=channels)

        restored = CycleChannel.from_json(cycle.to_json())

        assert restored.inst == cycle_inst
        assert restored.defcircuit == defcircuit
        assert restored.defcircuit.name == "SYNDROME_ROUND"
        assert restored.qubits == [5, 3]
        assert restored == cycle

    def test_or_chains_into_a_flat_three_operation_cycle(self):
        """``a | b | c`` must build one three-operation cycle.

        Regression guard: ``a | b`` returns a CycleChannel, which had no ``__or__``, so the second
        ``|`` raised TypeError and wider cycles were reachable only through a private helper.
        """
        a = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        b = Channel.from_depolarizing_constant(RX(np.pi / 2, 1), 0.99)
        c = MeasurementChannel.from_readout_fidelity(MEASURE(2, None), fidelity=0.98)

        cycle = a | b | c
        assert isinstance(cycle, CycleChannel)
        assert cycle.channels == (a, b, c)
        assert sorted(cycle.qubits) == [0, 1, 2]

        # Cycles compose from either side, and stay flat.
        from_cycle_on_the_right = a | (b | c)
        assert from_cycle_on_the_right.channels == (a, b, c)

    def test_or_rejects_overlapping_qubits_when_chaining(self):
        a = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        b = Channel.from_depolarizing_constant(RX(np.pi / 2, 1), 0.99)
        c = Channel.from_depolarizing_constant(RX(np.pi / 2, 1), 0.98)
        with pytest.raises(ValueError, match="overlapping qubits"):
            _ = a | b | c

    def test_process_fidelity_is_the_exact_product(self):
        """Process fidelity is multiplicative over the disjoint constituents."""
        a = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        b = Channel.from_depolarizing_constant(RX(np.pi / 2, 1), 0.99)
        cycle = a | b
        assert cycle.process_fidelity == pytest.approx(a.process_fidelity * b.process_fidelity)
        assert cycle.process_infidelity == pytest.approx(1.0 - cycle.process_fidelity)

    def test_average_gate_fidelity_converts_once_over_the_full_dimension(self):
        """Average gate fidelity is not multiplicative, so it must be converted once at the end.

        Regression guard: the cycle used to multiply its constituents' average gate fidelities,
        which overstates the result because the process-to-average conversion depends on the total
        Hilbert-space dimension.
        """
        a = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        b = Channel.from_depolarizing_constant(RX(np.pi / 2, 1), 0.99)
        cycle = a | b

        expected = float(qx.process_fidelity_to_average_fidelity(cycle.process_fidelity, dims=(2, 2)))
        assert cycle.average_gate_fidelity == pytest.approx(expected)
        assert cycle.average_gate_infidelity == pytest.approx(1.0 - expected)

        # The naive product of averages is a different (larger) number.
        naive_product = a.average_gate_fidelity * b.average_gate_fidelity
        assert naive_product > cycle.average_gate_fidelity
        assert not np.isclose(naive_product, cycle.average_gate_fidelity, atol=1e-6)

    def test_fidelities_ignore_measurement_only_cycles(self):
        """Documented (and surprising) behavior: readout noise carries no gate fidelity."""
        m0 = MeasurementChannel.from_readout_fidelity(MEASURE(0, None), fidelity=0.9)
        m1 = MeasurementChannel.from_readout_fidelity(MEASURE(1, None), fidelity=0.9)
        cycle = m0 | m1
        assert cycle.process_fidelity == 1.0
        assert cycle.average_gate_fidelity == 1.0

    def test_expanded_instructions_is_an_immutable_tuple(self):
        """A cached property on a frozen dataclass must not hand out a mutable list."""
        a = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        b = Channel.from_depolarizing_constant(RX(np.pi / 2, 1), 0.99)
        assert isinstance((a | b).expanded_instructions, tuple)

    def test_arity_mismatch_reports_clearly(self):
        """A cycle gate whose arity disagrees with its DEFCIRCUIT gets a real error message."""
        q0, q1 = FormalArgument("q0"), FormalArgument("q1")
        defcircuit = DefCircuit("CYCLE", [], [q0, q1], [RX(0.1, q0), RZ(0.2, q1)])
        cycle_inst = Gate("CYCLE", [], [Qubit(0)])  # one qubit, two formal arguments
        channels = (Channel.from_depolarizing_constant(RX(0.1, 0), depolarizing_constant=0.99),)
        with pytest.raises(ValueError, match="formal argument"):
            _ = CycleChannel(inst=cycle_inst, defcircuit=defcircuit, channels=channels)


# ──────────────────────────────────────────────────────────
# Channel tests
# ──────────────────────────────────────────────────────────


class TestChannelGeneratorOps:
    def test_process_is_superop_and_is_gate_channel(self):
        """process is derived by evolving the generator; the channel is a _ChannelBase."""
        ch = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        assert isinstance(ch, _ChannelBase)
        assert isinstance(ch.process, qx.SuperOp)

    def test_matches_superop_channel_fidelity(self):
        """A depolarizing Channel matches the equivalent superoperator SuperopChannel."""
        lind = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        superop = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        assert lind.process_fidelity == pytest.approx(superop.process_fidelity, abs=1e-6)

    def test_pow_scales_noise_and_keeps_gate(self):
        """** scales the noise while preserving the ideal gate; result stays a Channel."""
        ch = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        assert (ch**0.0).process_infidelity == pytest.approx(0.0, abs=1e-6)
        assert (ch**1.0).process_infidelity == pytest.approx(ch.process_infidelity, abs=1e-6)
        assert (ch**2.0).process_infidelity == pytest.approx(2 * ch.process_infidelity, rel=1e-2)
        assert isinstance(ch**2.0, Channel)
        assert jnp.allclose((ch**2.0).ideal_unitary.matrix, ch.ideal_unitary.matrix)

    def test_add_combines_noise_on_same_gate(self):
        """Adding two channels on the same gate keeps the gate and combines the noise."""
        a = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        b = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.98)
        combined = a + b
        assert isinstance(combined, Channel)
        assert combined.process_infidelity > a.process_infidelity
        assert jnp.allclose(combined.ideal_unitary.matrix, a.ideal_unitary.matrix)

    def test_add_rejects_different_gates(self):
        a = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        b = Channel.from_depolarizing_constant(RY(np.pi / 2, 1), 0.99)
        with pytest.raises(ValueError, match="different gates"):
            _ = a + b

    def test_superop_ops_downgrade_to_channel(self):
        """Operations that leave the Lindbladian manifold return a plain SuperopChannel."""
        ch = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        assert isinstance(ch.pauli_twirl(), SuperopChannel)
        assert isinstance(ch.to_coherent_channel(), SuperopChannel)
        assert isinstance(ch.to_stochastic_channel(), SuperopChannel)
        # Composing with a non-Lindbladian channel falls back to a superoperator SuperopChannel.
        superop = ch.pauli_twirl()
        assert isinstance(ch @ superop, SuperopChannel) and not isinstance(ch @ superop, Channel)

    def test_matmul_is_exact_superoperator_composition(self):
        """``@`` composes the processes exactly, so it returns a SuperopChannel, not a Channel."""
        a = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        b = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.98)
        composed = a @ b

        # Not a Channel: composing two Lindbladian evolutions is not itself one.
        assert isinstance(composed, SuperopChannel)
        assert not isinstance(composed, Channel)

        expected = qx.to_superop(a.process @ qx.to_superop(a.ideal_unitary.h) @ b.process)
        assert jnp.allclose(composed.process.matrix, expected.matrix)
        assert composed.process_infidelity > a.process_infidelity
        assert jnp.allclose(composed.ideal_unitary.matrix, a.ideal_unitary.matrix)

    def test_matmul_differs_from_add_for_noncommuting_generators(self):
        """``@`` (exact composition) and ``+`` (generator sum) agree only when generators commute.

        Regression guard: ``@`` used to be implemented as ``+``, which silently substituted a
        first-order approximation for the composition.
        """
        gate = RX(np.pi / 2, 0)
        a = Channel.from_pauli_generators(gate, {"Y": 0.9})
        b = Channel.from_pauli_generators(gate, {"Z": 0.7})

        composed = a @ b
        summed = a + b
        assert not jnp.allclose(composed.process.matrix, summed.process.matrix)

        # ...and they do agree when the noise commutes with the gate and with itself.
        commuting_a = Channel.from_pauli_generators(gate, {"X": 0.9})
        commuting_b = Channel.from_pauli_generators(gate, {"X": 0.7})
        assert jnp.allclose(
            (commuting_a @ commuting_b).process.matrix,
            (commuting_a + commuting_b).process.matrix,
        )

    def test_matmul_rejects_different_gates(self):
        a = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        b = Channel.from_depolarizing_constant(RY(np.pi / 2, 1), 0.99)
        with pytest.raises(ValueError, match="different gates"):
            _ = a @ b

    def test_add_rejects_different_gate_times(self):
        """Jump operators are per-unit-time rates, so mismatched gate times cannot be summed."""
        a = Channel.from_pauli_generators(RX(np.pi / 2, 0), {"X": 0.01}, gate_time=1.0)
        b = Channel.from_pauli_generators(RX(np.pi / 2, 0), {"X": 0.01}, gate_time=100.0)
        with pytest.raises(ValueError, match="different gate times"):
            _ = a + b

    def test_pow_equals_repeated_addition(self):
        """``ch ** 2`` scales the noise, matching ``ch + ch``."""
        ch = Channel.from_pauli_generators(RX(np.pi / 2, 0), {"X": 0.02, "Z": 0.01})
        assert jnp.allclose((ch**2.0).process.matrix, (ch + ch).process.matrix)
        # power 0 recovers the ideal gate
        assert (ch**0.0).process_infidelity == pytest.approx(0.0, abs=1e-6)

    def test_noise_and_target_lindbladians_decompose_the_generator(self):
        """`lindbladian` splits into `noise_lindbladian + target_lindbladian`."""
        ch = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        recombined = ch.noise_lindbladian + ch.target_lindbladian
        assert jnp.allclose(recombined.matrix, ch.lindbladian.matrix)
        # The target alone (zero dissipation) reproduces the ideal gate.
        target_only = replace(ch, lindbladian=ch.target_lindbladian)
        assert target_only.process_infidelity == pytest.approx(0.0, abs=1e-6)

    def test_coherence_times(self):
        """A short gate relative to T1/T2 gives high fidelity."""
        ch = Channel.from_coherence_times(RX(np.pi / 2, 0), gate_duration=40e-9, t1s=[30e-6], t2s=[20e-6])
        assert isinstance(ch.process, qx.SuperOp)
        assert ch.average_gate_fidelity > 0.99

    def test_json_roundtrip(self):
        ch = Channel.from_depolarizing_constant(RX(0.3, 0), 0.97)
        assert Channel.from_json(ch.to_json()) == ch


class TestCoherenceTimes:
    """T1/T2 -> Tphi conversion, which uses 1/T2 = 1/(2*T1) + 1/Tphi."""

    @pytest.mark.parametrize("t2_over_t1", [0.5, 1.0, 1.5, 2.0])
    def test_physical_t2_range_gives_a_cptp_channel(self, t2_over_t1):
        """Every physical T2 (up to and including 2*T1) must produce a valid channel.

        Regression guard: the Tphi conversion used to omit the factor of two on T1, so any
        T1 < T2 <= 2*T1 -- which is the entire realistic range -- produced a negative pure-dephasing
        time, NaN fidelities, and a LinAlgError from the unitarity computation.
        """
        t1 = 20e-6
        ch = Channel.from_coherence_times(RX(np.pi / 2, 0), 40e-9, t1s=[t1], t2s=[t2_over_t1 * t1])

        assert np.isfinite(ch.process_fidelity)
        assert ch.process_fidelity > 0.99  # 40 ns gate against a 20 us T1
        assert np.isfinite(ch.unitarity)
        choi_eigenvalues = np.linalg.eigvalsh(np.asarray(qx.to_choi(ch.process).matrix))
        assert choi_eigenvalues.min() > -1e-9

    def test_default_t2_is_twice_t1(self):
        """Omitting T2 means no pure dephasing, i.e. the same channel as T2 == 2*T1."""
        t1 = 20e-6
        default = Channel.from_coherence_times(RX(np.pi / 2, 0), 40e-9, t1s=[t1])
        explicit = Channel.from_coherence_times(RX(np.pi / 2, 0), 40e-9, t1s=[t1], t2s=[2 * t1])
        assert jnp.allclose(default.process.matrix, explicit.process.matrix)
        assert np.isfinite(default.process_fidelity)

    def test_more_dephasing_lowers_fidelity(self):
        """Shorter T2 at fixed T1 means more pure dephasing and a worse gate."""
        t1 = 20e-6
        dephasing = Channel.from_coherence_times(RX(np.pi / 2, 0), 40e-9, t1s=[t1], t2s=[0.5 * t1])
        relaxation_only = Channel.from_coherence_times(RX(np.pi / 2, 0), 40e-9, t1s=[t1], t2s=[2 * t1])
        assert dephasing.process_fidelity < relaxation_only.process_fidelity

    def test_rejects_t2_above_twice_t1(self):
        """T2 > 2*T1 implies a negative dephasing rate and is unphysical."""
        with pytest.raises(ValueError, match="T2 must not exceed"):
            Channel.from_coherence_times(RX(np.pi / 2, 0), 40e-9, t1s=[20e-6], t2s=[41e-6])

    def test_rejects_non_positive_times(self):
        with pytest.raises(ValueError, match="must be positive"):
            Channel.from_coherence_times(RX(np.pi / 2, 0), 40e-9, t1s=[20e-6], t2s=[0.0])

    def test_reset_channel_shares_the_conversion(self):
        """ResetChannel.from_coherence_times has the same default and the same guard."""
        reset = ResetChannel.from_coherence_times(ResetQubit(0), duration=1.0, t1=2.0)
        assert np.isfinite(reset.process_fidelity)
        with pytest.raises(ValueError, match="T2 must not exceed"):
            ResetChannel.from_coherence_times(ResetQubit(0), duration=1.0, t1=2.0, t2=5.0)


# ──────────────────────────────────────────────────────────
# ResetChannel tests
# ──────────────────────────────────────────────────────────


class TestResetChannel:
    def test_stronger_damping_gives_better_reset(self):
        weak = ResetChannel.from_amplitude_damping(ResetQubit(0), gamma=0.5)
        strong = ResetChannel.from_amplitude_damping(ResetQubit(0), gamma=5.0)
        assert isinstance(weak.process, qx.SuperOp)
        assert strong.process_fidelity > weak.process_fidelity

    def test_pow_scales_relaxation(self):
        ch = ResetChannel.from_amplitude_damping(ResetQubit(0), gamma=0.5)
        assert isinstance(ch**2.0, ResetChannel)
        assert (ch**2.0).process_fidelity > ch.process_fidelity

    def test_rejects_global_reset(self):
        from pyquil.quilbase import Reset

        with pytest.raises(TypeError, match="targeted"):
            ResetChannel.from_lindbladian(Reset(), qx.lindbladians.amplitude_damping(1.0, (2,)))

    def test_json_roundtrip(self):
        ch = ResetChannel.from_coherence_times(ResetQubit(0), duration=1.0, t1=2.0, t2=1.5)
        assert ResetChannel.from_json(ch.to_json()) == ch


# ──────────────────────────────────────────────────────────
# Qudit support boundaries
# ──────────────────────────────────────────────────────────


class TestQuditSupport:
    """The Pauli basis is a qubit-only construct; qudits must get a real error, not a shape crash."""

    QUTRIT_X = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)

    @pytest.fixture
    def qutrit_gate(self):
        gate = Gate("QUTRIT_X", [], [0])
        custom_gates = {
            "QUTRIT_X": qx.Unitary.from_matrix(jnp.asarray(self.QUTRIT_X), _operator_dims_from_dimension(3))
        }
        return gate, custom_gates

    def test_lindbladian_constructors_support_qutrits(self, qutrit_gate):
        """The generator-based path is dimension-agnostic and must keep working."""
        gate, custom_gates = qutrit_gate
        ch = Channel.from_depolarizing_constant(gate, 0.99, custom_gates=custom_gates)
        assert ch.dims == (3,)

        # The contract is the shrink factor, not the fidelity: for d = 3 a shrink of 0.99 gives
        # a process fidelity of (1 + (d^2 - 1) * p) / d^2 = 0.991111...
        expected = float(qx.process_fidelity_to_depolarizing_constant(ch.process_fidelity, (3,)))
        assert expected == pytest.approx(0.99, abs=1e-6)
        assert Channel.from_json(ch.to_json()) == ch

    def test_pauli_constructors_reject_qutrits_with_a_clear_message(self, qutrit_gate):
        """Regression guard: these used to fail with raw JAX reshape errors."""
        gate, custom_gates = qutrit_gate
        with pytest.raises(ValueError, match="qubits only"):
            Channel.from_pauli_generators(gate, {"X": 0.01}, custom_gates=custom_gates)
        with pytest.raises(ValueError, match="qubits only"):
            SuperopChannel.from_pauli_noise(gate, {"X": 0.01}, custom_gates=custom_gates)
        with pytest.raises(ValueError, match="qubits only"):
            Channel.from_random_coherent_error(gate, 0.99, custom_gates=custom_gates)

    def test_pauli_analyses_reject_qutrits_with_a_clear_message(self, qutrit_gate):
        """``is_pauli`` in particular used to return a meaningless answer rather than raising."""
        gate, custom_gates = qutrit_gate
        ch = Channel.from_depolarizing_constant(gate, 0.99, custom_gates=custom_gates)
        with pytest.raises(ValueError, match="qubits only"):
            ch.is_pauli()
        with pytest.raises(ValueError, match="qubits only"):
            ch.to_pauli_vector()

    def test_error_message_names_the_operation_and_the_dims(self, qutrit_gate):
        gate, custom_gates = qutrit_gate
        with pytest.raises(ValueError, match=r"from_pauli_generators.*\(3,\)"):
            Channel.from_pauli_generators(gate, {"X": 0.01}, custom_gates=custom_gates)


# ──────────────────────────────────────────────────────────
# Random coherent error
# ──────────────────────────────────────────────────────────


class TestRandomCoherentError:
    @pytest.mark.parametrize("gate", [X(0), RX(np.pi / 2, 0), CNOT(0, 1)], ids=lambda g: g.out())
    @pytest.mark.parametrize("requested", [0.999, 0.99, 0.9, 0.5])
    def test_achieves_the_requested_process_fidelity(self, gate, requested):
        """The angle is solved numerically, so multi-qubit gates hit the target exactly too.

        Regression guard: the closed form assumed the drawn generator squares to a multiple of the
        identity, which only holds for a single qudit; a CZ at 0.9 came out at 0.9009. The bound
        below is ~1000x tighter than that error; the residual is round-off from rebuilding the
        channel through its Hamiltonian.
        """
        ch = Channel.from_random_coherent_error(gate, requested, rng=np.random.default_rng(3))
        assert ch.process_fidelity == pytest.approx(requested, abs=1e-6)

    def test_error_is_purely_coherent(self):
        ch = Channel.from_random_coherent_error(CNOT(0, 1), 0.97, rng=np.random.default_rng(5))
        assert ch.unitarity == pytest.approx(1.0, abs=1e-6)
        assert ch.stochastic_infidelity == pytest.approx(0.0, abs=1e-6)

    def test_perfect_fidelity_gives_the_ideal_gate(self):
        ch = Channel.from_random_coherent_error(X(0), 1.0, rng=np.random.default_rng(1))
        assert ch.process_infidelity == pytest.approx(0.0, abs=1e-9)

    def test_is_reproducible_for_a_given_seed(self):
        a = Channel.from_random_coherent_error(X(0), 0.97, rng=np.random.default_rng(11))
        b = Channel.from_random_coherent_error(X(0), 0.97, rng=np.random.default_rng(11))
        assert jnp.allclose(a.process.matrix, b.process.matrix)

    @pytest.mark.parametrize("requested", [0.0, -0.1, 1.5])
    def test_rejects_out_of_range_fidelity(self, requested):
        with pytest.raises(ValueError, match=r"must lie in \(0, 1\]"):
            Channel.from_random_coherent_error(X(0), requested, rng=np.random.default_rng(1))


# ──────────────────────────────────────────────────────────
# Pauli noise / twirl bookkeeping
# ──────────────────────────────────────────────────────────


class TestPauliNoiseIdentityHandling:
    def test_rejects_explicit_identity_term(self):
        """The identity probability is implicit; supplying it broke trace preservation.

        Regression guard: an explicit ``"I"`` key won the assignment over the ``1 - sum(others)``
        normalization, producing a channel whose Choi trace was 1.2 instead of 2.
        """
        with pytest.raises(ValueError, match="is the identity"):
            SuperopChannel.from_pauli_noise(X(0), {"I": 0.5, "X": 0.1})
        with pytest.raises(ValueError, match="is the identity"):
            SuperopChannel.from_pauli_noise(CNOT(0, 1), {"II": 0.5})

    @pytest.mark.parametrize(
        "pauli_noise",
        [{"X": 0.02}, {"X": 0.02, "Y": 0.03, "Z": 0.05}, {"Z": 0.0}],
        ids=["one-term", "three-terms", "zero-rate"],
    )
    def test_is_trace_preserving(self, pauli_noise):
        ch = SuperopChannel.from_pauli_noise(X(0), pauli_noise)
        ptm = np.asarray(qx.to_pauli_liouville(ch.process).matrix).real
        np.testing.assert_allclose(ptm[0], [1.0, 0.0, 0.0, 0.0], atol=1e-9)
        choi_trace = np.trace(np.asarray(qx.to_choi(ch.process).matrix)).real
        assert choi_trace == pytest.approx(2.0, abs=1e-9)

    def test_pauli_vector_recovers_the_input_probabilities(self):
        pauli_noise = {"X": 0.02, "Y": 0.03, "Z": 0.05}
        ch = SuperopChannel.from_pauli_noise(X(0), pauli_noise)
        vector = np.asarray(ch.to_pauli_vector())
        np.testing.assert_allclose(vector, [0.90, 0.02, 0.03, 0.05], atol=1e-9)
