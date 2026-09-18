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

"""Unit tests for the quax-based density-matrix simulator.

Structured as a superset of ``test_state_vector.py``: every gate-only behaviour checked there
is re-checked here (a density matrix must reproduce ``|psi><psi|``), and the noisy, measurement,
reset and qudit behaviour that only exists for density matrices is added on top.

Three independent yardsticks are used, in decreasing order of preference:

1. **Analytic values.** Preferred where a closed form exists (depolarizing shrink factors,
   ``exp(-t/T1)`` populations, over-rotation angles) because the expected value documents the
   physics rather than restating the implementation.
2. **pyQuil's own reference simulators.** ``ReferenceWavefunctionSimulator`` /
   ``ReferenceDensitySimulator`` are separate implementations already in the repo. They are
   *little-endian*, so amplitudes have to be reversed -- see ``_reverse_endianness``.
3. **Frozen end-to-end goldens.** Two Lindblad results are hard-coded, having been produced with
   ``qutip``'s ``mesolve``. These are *not* independent validation of the physics: quax's
   ``evolve`` is already tested against qutip upstream, so agreement there is expected. What they
   pin is the pyQuil-side chain around it -- the T1/T2 to jump-operator conversion, the
   gate-Hamiltonian scaling, the subsystem embedding, and the simulator's application of the
   result -- as a single frozen number per case. qutip is deliberately **not** a pyQuil
   dependency; the values live here rather than being computed at test time, and each carries the
   snippet that produced it so it can be regenerated when needed.

.. note::
    These simulators are **big-endian**: ``qubits[0]`` is the most significant subsystem, so
    ``X 0`` on a two-qubit register gives ``|10>``. The rest of pyQuil is little-endian.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import quax as qx

from pyquil.gates import CNOT, MEASURE, RESET, RX, RY, H, I, X
from pyquil.noise._channels import (
    Channel,
    MeasurementChannel,
    ResetChannel,
    SuperopChannel,
    SuperopResetChannel,
)
from pyquil.noise._noise_model import NoiseModel
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, quil_cis
from pyquil.quilbase import Declare, Gate, ResetQubit
from pyquil.simulation._reference import ReferenceDensitySimulator, ReferenceWavefunctionSimulator
from pyquil.simulation._simulator import DensityMatrixSimulator
from test.unit.simulation_programs import (
    COMPARABLE,
    DM_ONLY_EXPECTED,
    DM_ONLY_PROGRAMS,
    PROGRAMS,
    assert_physical,
    assert_pure,
    relabel_contiguous,
    reverse_endianness,
    simulate_density_matrix,
    simulate_state_vector,
)

# quax's qutrit X: the cyclic shift |0> -> |2> -> |1> -> |0>.
_QUTRIT_X = np.asarray(qx.gates.QUANTUM_GATES["TX"].matrix)


def _dm(program, qubits=None, noise_model=None, memory_map=None, **kwargs):
    """Final density matrix as a plain numpy array."""
    return np.asarray(simulate_density_matrix(program, qubits, noise_model, memory_map, **kwargs).matrix)


def _sv(program, qubits=None, memory_map=None, **kwargs):
    """Final state vector as a flat numpy array."""
    return np.asarray(simulate_state_vector(program, qubits, memory_map, **kwargs).matrix).reshape(-1)


ALL_PROGRAMS = PROGRAMS | DM_ONLY_PROGRAMS


class TestNoiselessMatchesStateVector:
    """A noiseless density matrix must equal the outer product of the pure state."""

    @pytest.mark.parametrize("name", sorted(PROGRAMS))
    def test_matches_state_vector(self, name):
        program = PROGRAMS[name]
        assert_pure(_dm(program), _sv(program))

    @pytest.mark.parametrize("name", sorted(ALL_PROGRAMS))
    @pytest.mark.parametrize("max_subsystem_size", [1, 2, 3, 4])
    def test_independent_of_compressor_budget(self, name, max_subsystem_size):
        """``max_subsystem_size`` is a performance knob and must not change the result."""
        program = ALL_PROGRAMS[name]
        reference = _dm(program)
        got = _dm(program, max_subsystem_size=max_subsystem_size)
        np.testing.assert_allclose(got, reference, atol=1e-10)

    @pytest.mark.parametrize("name", sorted(ALL_PROGRAMS))
    def test_is_physical(self, name):
        assert_physical(_dm(ALL_PROGRAMS[name]))

    @pytest.mark.parametrize("name", sorted(DM_ONLY_PROGRAMS))
    def test_mid_circuit_measure_and_reset_match_analytic_state(self, name):
        """Mid-circuit MEASURE and RESET produce the outcome-averaged state."""
        np.testing.assert_allclose(_dm(DM_ONLY_PROGRAMS[name]), DM_ONLY_EXPECTED[name], atol=1e-10)


class TestAgainstPyquilReferenceSimulators:
    """Cross-check against pyQuil's independent (little-endian) reference implementations.

    The reference simulators take a contiguous register, so a sparse program is relabelled to
    ``0..n-1`` for them; the quax simulators number subsystems by sorted qubit, so the two agree
    index for index.
    """

    @pytest.mark.parametrize("name", COMPARABLE)
    def test_matches_reference_density_simulator(self, name):
        program = PROGRAMS[name]
        n = len(program.get_qubit_indices())
        got = reverse_endianness(_dm(program), n)
        expected = np.asarray(ReferenceDensitySimulator(n_qubits=n).do_program(relabel_contiguous(program)).density)
        np.testing.assert_allclose(got, expected, atol=1e-9)

    @pytest.mark.parametrize("name", COMPARABLE)
    def test_matches_reference_wavefunction_simulator(self, name):
        program = PROGRAMS[name]
        n = len(program.get_qubit_indices())
        got = reverse_endianness(_sv(program), n)
        reference = ReferenceWavefunctionSimulator(n_qubits=n).do_program(relabel_contiguous(program))
        expected = np.asarray(reference.wf).reshape(-1)
        # Compare up to global phase.
        overlap = abs(np.vdot(expected, got)) / (np.linalg.norm(expected) * np.linalg.norm(got))
        assert overlap == pytest.approx(1.0, abs=1e-9)


class TestBigEndianOrdering:
    """The register order is big-endian: ``qubits[0]`` is the most significant subsystem."""

    def test_x_on_first_qubit_sets_high_bit(self):
        rho = _dm(Program(X(0), I(1)), qubits=[0, 1])
        assert int(np.argmax(np.real(np.diag(rho)))) == 0b10

    def test_x_on_second_qubit_sets_low_bit(self):
        rho = _dm(Program(I(0), X(1)), qubits=[0, 1])
        assert int(np.argmax(np.real(np.diag(rho)))) == 0b01

    def test_explicit_qubit_order_is_respected(self):
        """Reversing ``qubits`` reverses which subsystem the amplitude lands in."""
        assert int(np.argmax(np.real(np.diag(_dm(Program(X(5), I(6)), qubits=[5, 6]))))) == 0b10
        assert int(np.argmax(np.real(np.diag(_dm(Program(X(5), I(6)), qubits=[6, 5]))))) == 0b01

    def test_opposite_of_pyquil_reference(self):
        """Documents the deliberate divergence from the rest of pyQuil."""
        program = Program(X(0), I(1))
        ours = int(np.argmax(np.abs(_sv(program, qubits=[0, 1]))))
        theirs = int(np.argmax(np.abs(np.asarray(ReferenceWavefunctionSimulator(n_qubits=2).do_program(program).wf))))
        assert (ours, theirs) == (0b10, 0b01)


class TestSingleQubitGateNoise:
    """Noise attached to 1Q gates, with analytic expectations."""

    @pytest.mark.parametrize("angle", [np.pi / 2, np.pi])
    @pytest.mark.parametrize("shrink", [1.0, 0.99, 0.9, 0.5])
    def test_depolarizing_shrinks_bloch_vector_by_p(self, angle, shrink):
        """A depolarizing channel of constant p scales the Bloch vector by exactly p."""
        gate = RX(angle, 0)
        noisy = _dm(
            Program(gate),
            qubits=[0],
            noise_model=NoiseModel.from_channels([Channel.from_depolarizing_constant(gate, shrink)]),
        )
        ideal = _dm(Program(gate), qubits=[0])
        paulis = {"X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
        for name, pauli in paulis.items():
            got = np.trace(noisy @ pauli).real
            expected = shrink * np.trace(ideal @ pauli).real
            assert got == pytest.approx(expected, abs=1e-9), f"<{name}> for RX({angle}) at p={shrink}"
        assert_physical(noisy)

    @pytest.mark.parametrize("angle", [np.pi / 2, np.pi])
    def test_pauli_noise_reproduces_its_error_probabilities(self, angle):
        """A one-shot Pauli channel applies exactly the requested error probabilities."""
        gate = RX(angle, 0)
        pauli_noise = {"X": 0.03, "Y": 0.05, "Z": 0.07}
        channel = SuperopChannel.from_pauli_noise(gate, pauli_noise)
        rho = _dm(Program(gate), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        assert_physical(rho)

        # Rebuild the expected state by hand: apply each Pauli to the ideal state with its
        # probability. This is the definition of the channel, independent of the implementation.
        ideal = _dm(Program(gate), qubits=[0])
        mats = {
            "I": np.eye(2),
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.diag([1, -1]).astype(complex),
        }
        probs = {"I": 1.0 - sum(pauli_noise.values()), **pauli_noise}
        expected = sum(prob * mats[term] @ ideal @ mats[term].conj().T for term, prob in probs.items())
        np.testing.assert_allclose(rho, expected, atol=1e-10)

    @pytest.mark.parametrize("angle", [np.pi / 2, np.pi])
    def test_coherent_over_rotation_is_a_pure_rotation_error(self, angle):
        """A coherent error keeps the state pure and rotates it by the extra angle."""
        over = 0.05
        gate = RX(angle, 0)
        # An over-rotation: the noisy process is RX(angle + over) where the ideal is RX(angle).
        channel = SuperopChannel(
            inst=gate,
            process=qx.to_superop(qx.gates.RX(angle + over)),
            ideal_unitary=qx.gates.RX(angle),
        )
        rho = _dm(Program(gate), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        assert_physical(rho)
        assert np.trace(rho @ rho).real == pytest.approx(1.0, abs=1e-9), "coherent error must stay pure"
        np.testing.assert_allclose(rho, _dm(Program(RX(angle + over, 0)), qubits=[0]), atol=1e-10)
        # A purely coherent error has unit unitarity and no stochastic component.
        assert channel.unitarity == pytest.approx(1.0, abs=1e-8)
        assert channel.stochastic_infidelity == pytest.approx(0.0, abs=1e-8)

    def test_random_coherent_error_is_unitary_and_hits_requested_fidelity(self):
        gate = RX(np.pi / 2, 0)
        channel = Channel.from_random_coherent_error(gate, 0.97, rng=np.random.default_rng(4))
        rho = _dm(Program(gate), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        assert_physical(rho)
        assert np.trace(rho @ rho).real == pytest.approx(1.0, abs=1e-8)
        assert channel.process_fidelity == pytest.approx(0.97, abs=1e-6)

    def test_superop_and_lindbladian_flavors_are_both_applied(self):
        """Regression: the resolver used to match only the Lindbladian class, dropping the other."""
        gate = X(0)
        ideal = _dm(Program(gate), qubits=[0])
        for channel in (
            Channel.from_depolarizing_constant(gate, 0.8),
            SuperopChannel.from_pauli_noise(gate, {"Y": 0.2}),
        ):
            rho = _dm(Program(gate), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
            assert not np.allclose(rho, ideal, atol=1e-6), f"{type(channel).__name__} noise was dropped"


class TestTwoQubitGateNoise:
    def test_depolarizing_on_cnot_shrinks_all_two_qubit_paulis(self):
        gate = CNOT(0, 1)
        shrink = 0.9
        program = Program(H(0), gate)
        noisy = _dm(
            program,
            qubits=[0, 1],
            noise_model=NoiseModel.from_channels([Channel.from_depolarizing_constant(gate, shrink)]),
        )
        ideal = _dm(program, qubits=[0, 1])
        assert_physical(noisy)
        single = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
        for a in single:
            for b in single:
                op = np.kron(a, b)
                if np.allclose(op, np.eye(4)):
                    continue
                assert np.trace(noisy @ op).real == pytest.approx(shrink * np.trace(ideal @ op).real, abs=1e-9)

    def test_noise_only_touches_the_gate_it_is_attached_to(self):
        """A channel on CNOT 0 1 must leave an untouched spectator qubit pure."""
        program = Program(H(0), CNOT(0, 1), X(2))
        rho = _dm(
            program,
            qubits=[0, 1, 2],
            noise_model=NoiseModel.from_channels([Channel.from_depolarizing_constant(CNOT(0, 1), 0.7)]),
        )
        assert_physical(rho)
        # Trace out qubits 0 and 1 (the two most significant subsystems).
        reduced = rho.reshape(2, 2, 2, 2, 2, 2).trace(axis1=0, axis2=3).trace(axis1=0, axis2=2)
        np.testing.assert_allclose(reduced, np.diag([0.0, 1.0]), atol=1e-9)

    @pytest.mark.parametrize("control,target", [(0, 1), (1, 0)])
    def test_noise_follows_operand_order(self, control, target):
        """A channel keyed on CNOT c t applies to that operand order, not the sorted one."""
        gate = CNOT(control, target)
        program = Program(X(control), gate)
        rho = _dm(
            program,
            qubits=[0, 1],
            noise_model=NoiseModel.from_channels([Channel.from_depolarizing_constant(gate, 0.85)]),
        )
        assert_physical(rho)
        # Ideal result: control stays 1, target flips to 1 -> |11>, whatever the operand order.
        assert np.real(np.diag(rho))[0b11] == max(np.real(np.diag(rho)))


class TestDecoherence:
    """State-dependent (non-unital) noise: T1 relaxation and T2 dephasing."""

    def test_amplitude_damping_matches_exponential_decay(self):
        """Excited population after damping for t must be exp(-t/T1)."""
        t1, duration = 20e-6, 5e-6
        channel = ResetChannel.from_amplitude_damping(ResetQubit(0), gamma=1.0 / t1, gate_time=duration)
        rho = _dm(Program(X(0), RESET(0)), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        assert_physical(rho)
        assert rho[1, 1].real == pytest.approx(np.exp(-duration / t1), abs=1e-10)

    def test_decoherence_is_state_dependent(self):
        """T1 relaxation must decay |1> but leave |0> alone -- the hallmark of a non-unital channel."""
        t1, t2, duration = 2e-6, 1.5e-6, 1e-6
        channel = Channel.from_coherence_times(I(0), gate_duration=duration, t1s=[t1], t2s=[t2])
        noise_model = NoiseModel.from_channels([channel])
        from_ground = _dm(Program(I(0)), qubits=[0], noise_model=noise_model)
        from_excited = _dm(Program(X(0), I(0)), qubits=[0], noise_model=noise_model)
        np.testing.assert_allclose(from_ground, np.diag([1.0, 0.0]), atol=1e-9)
        assert from_excited[1, 1].real == pytest.approx(np.exp(-duration / t1), abs=1e-9)

    def test_dephasing_decays_coherence_at_t2(self):
        """Starting from |+>, the off-diagonal must decay as exp(-t/T2)."""
        t1, t2, duration = 100.0, 1.0, 0.3
        channel = Channel.from_coherence_times(I(0), gate_duration=duration, t1s=[t1], t2s=[t2])
        rho = _dm(Program(H(0), I(0)), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        assert_physical(rho)
        assert abs(rho[0, 1]) == pytest.approx(0.5 * np.exp(-duration / t2), abs=1e-6)

    def test_t1_t2_during_rx_pi_matches_frozen_lindblad_golden(self):
        """End-to-end golden for 1Q T1/T2 during a gate.

        Produced with qutip's ``mesolve``, but see the module docstring: quax's ``evolve`` is
        already tested against qutip, so this pins the pyQuil-side chain (T1/T2 conversion,
        gate-Hamiltonian scaling, application) rather than the integrator.

        Regenerate with::

            ch = Channel.from_coherence_times(RX(np.pi, 0), 40e-9, t1s=[20e-6], t2s=[15e-6])
            H = qutip.Qobj(np.asarray(ch.lindbladian.hamiltonian.matrix))
            c_ops = [
                qutip.Qobj(j) for j in np.asarray(ch.lindbladian.jump_operators.matrix) if np.linalg.norm(j) > 1e-15
            ]
            qutip.mesolve(H, qutip.Qobj(np.diag([1.0, 0.0]).astype(complex)), [0, 40e-9], c_ops=c_ops)
        """
        channel = Channel.from_coherence_times(RX(np.pi, 0), gate_duration=40e-9, t1s=[20e-6], t2s=[15e-6])
        rho = _dm(Program(RX(np.pi, 0)), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        expected = np.array([[0.00116584637 + 0j, 0.000635886393j], [-0.000635886393j, 0.99883415363 + 0j]])
        assert_physical(rho)
        np.testing.assert_allclose(rho, expected, atol=1e-9)

    def test_t1_t2_during_cnot_matches_frozen_lindblad_golden(self):
        """End-to-end golden for a 2Q Lindbladian, which also pins the subsystem embedding.

        Prepared state is ``H 0; X 1``; the noisy CNOT then runs for 1 us with
        ``T1=[2, 3] us`` and ``T2=[1.5, 2] us``. Regenerate as in
        :meth:`test_t1_t2_during_rx_pi_matches_frozen_lindblad_golden`, starting from the
        prepared state. The two-qubit case is the one that would catch a transposed or
        misordered embedding, which the 1Q case cannot.
        """
        channel = Channel.from_coherence_times(CNOT(0, 1), gate_duration=1e-6, t1s=[2e-6, 3e-6], t2s=[1.5e-6, 2.0e-6])
        rho = _dm(Program(H(0), X(1), CNOT(0, 1)), qubits=[0, 1], noise_model=NoiseModel.from_channels([channel]))
        expected = np.array(
            [
                [0.252534837094 + 0j, -0.032556198963j, -0.026860937255j, 0.032994505303 + 0j],
                [0.032556198963j, 0.444199833049 + 0j, 0.169470952631 + 0j, -0.009365062981j],
                [0.026860937255j, 0.169470952631 + 0j, 0.255781666439 + 0j, 0.02613967993j],
                [0.032994505303 + 0j, 0.009365062981j, -0.02613967993j, 0.047483663418 + 0j],
            ]
        )
        assert_physical(rho)
        np.testing.assert_allclose(rho, expected, atol=1e-9)
        assert np.trace(rho @ rho).real < 0.5, "state should be strongly mixed"


class TestMeasurementAndReset:
    def test_measure_dephases_in_the_computational_basis(self):
        """MEASURE is applied as a dephasing channel: the reduced state loses coherence."""
        rho = _dm(Program(H(0), MEASURE(0, None)), qubits=[0])
        np.testing.assert_allclose(rho, np.diag([0.5, 0.5]), atol=1e-10)

    def test_measure_preserves_populations(self):
        rho = _dm(Program(RY(0.9, 0), MEASURE(0, None)), qubits=[0])
        ideal = _dm(Program(RY(0.9, 0)), qubits=[0])
        np.testing.assert_allclose(np.diag(rho), np.diag(ideal), atol=1e-10)
        assert abs(rho[0, 1]) == pytest.approx(0.0, abs=1e-10)

    def test_readout_noise_channel_is_applied(self):
        channel = MeasurementChannel.from_readout_fidelity(MEASURE(0, None), fidelity=0.9)
        rho = _dm(Program(H(0), MEASURE(0, None)), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        assert_physical(rho)

    def test_ideal_targeted_reset_returns_ground_state(self):
        for prep in (Program(X(0)), Program(H(0)), Program(RY(0.7, 0))):
            rho = _dm(prep + Program(RESET(0)), qubits=[0])
            np.testing.assert_allclose(rho, np.diag([1.0, 0.0]), atol=1e-10)

    def test_reset_leaves_other_qubits_alone(self):
        rho = _dm(Program(X(0), X(1), RESET(0)), qubits=[0, 1])
        np.testing.assert_allclose(rho, np.diag([0.0, 1.0, 0.0, 0.0]), atol=1e-10)

    def test_global_reset_returns_all_qubits_to_ground(self):
        rho = _dm(Program(X(0), X(1), RESET()), qubits=[0, 1])
        np.testing.assert_allclose(rho, np.diag([1.0, 0.0, 0.0, 0.0]), atol=1e-10)

    @pytest.mark.parametrize("fidelity", [1.0, 0.9, 0.75])
    def test_noisy_reset_fidelity_is_honoured(self, fidelity):
        channel = SuperopResetChannel.from_reset_fidelity(ResetQubit(0), fidelity=fidelity)
        rho = _dm(Program(X(0), RESET(0)), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        assert_physical(rho)
        # depolarizing_p @ RESET on any input gives p|0><0| + (1-p)I/2, with p = 2F - 1.
        p = 2 * fidelity - 1
        np.testing.assert_allclose(rho, p * np.diag([1.0, 0.0]) + (1 - p) * np.eye(2) / 2, atol=1e-9)

    @pytest.mark.parametrize("reset", [RESET(0), RESET()])
    def test_reset_noise_applies_to_targeted_and_global_forms(self, reset):
        """Regression: global RESET used to ignore the noise model entirely."""
        channel = SuperopResetChannel.from_reset_fidelity(ResetQubit(0), fidelity=0.5)
        rho = _dm(Program(X(0)) + Program(reset), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        np.testing.assert_allclose(rho, np.eye(2) / 2, atol=1e-9)


class TestOutcomeProbabilities:
    """``outcome_probabilities`` reads the terminal measurements, readout error included."""

    @staticmethod
    def _measured(*instructions, qubits=(0,)):
        program = Program()
        program += Declare("ro", "BIT", len(qubits))
        for inst in instructions:
            program += inst
        for i, q in enumerate(qubits):
            program += MEASURE(q, ("ro", i))
        return program

    def test_ideal_single_qubit_measurement(self):
        sim = DensityMatrixSimulator(self._measured(H(0)))
        np.testing.assert_allclose(sim.outcome_probabilities(), [0.5, 0.5], atol=1e-10)
        assert sim.measured_qubits == (0,)

    def test_joint_distribution_of_a_bell_pair(self):
        sim = DensityMatrixSimulator(self._measured(H(0), CNOT(0, 1), qubits=(0, 1)))
        probs = sim.outcome_probabilities()
        assert probs.shape == (2, 2)
        np.testing.assert_allclose(probs, [[0.5, 0.0], [0.0, 0.5]], atol=1e-10)
        assert sim.measured_qubits == (0, 1)

    def test_axes_follow_measurement_order_not_qubit_order(self):
        program = Program()
        program += Declare("ro", "BIT", 2)
        program += X(1)
        program += MEASURE(1, ("ro", 0))
        program += MEASURE(0, ("ro", 1))
        sim = DensityMatrixSimulator(program)
        assert sim.measured_qubits == (1, 0)
        np.testing.assert_allclose(sim.outcome_probabilities(), [[0.0, 0.0], [1.0, 0.0]], atol=1e-10)

    def test_unmeasured_qubits_are_traced_out(self):
        sim = DensityMatrixSimulator(self._measured(H(0), CNOT(0, 1), RY(0.4, 2), qubits=(1,)))
        np.testing.assert_allclose(sim.outcome_probabilities(), [0.5, 0.5], atol=1e-10)

    def test_readout_error_is_included(self):
        channel = MeasurementChannel.from_readout_fidelity(MEASURE(0, None), fidelity=0.8, asymmetry=0.5)
        sim = DensityMatrixSimulator(self._measured(RX(0.7, 0)), noise_model=NoiseModel.from_channels([channel]))
        populations = np.array([np.cos(0.35) ** 2, np.sin(0.35) ** 2])
        expected = np.asarray(channel.process.confusion_matrix) @ populations
        np.testing.assert_allclose(sim.outcome_probabilities(), expected, atol=1e-10)
        # The state itself is unaffected by classification error: the populations are exact.
        np.testing.assert_allclose(np.real(np.diag(sim.compute().matrix)), populations, atol=1e-10)

    def test_mid_circuit_measurement_is_not_read_out(self):
        program = Program()
        program += Declare("ro", "BIT", 1)
        program += H(0)
        program += MEASURE(0, ("ro", 0))
        program += X(0)
        program += MEASURE(0, ("ro", 0))
        sim = DensityMatrixSimulator(program)
        assert sim.measured_qubits == (0,)
        np.testing.assert_allclose(sim.outcome_probabilities(), [0.5, 0.5], atol=1e-10)

    def test_compute_is_unchanged_by_holding_measurements_back(self):
        channel = MeasurementChannel.from_readout_fidelity(MEASURE(0, None), fidelity=0.8, asymmetry=0.5)
        noise_model = NoiseModel.from_channels([channel])
        program = self._measured(RX(0.7, 0), CNOT(0, 1), qubits=(0,))
        rho = _dm(program, noise_model=noise_model)
        # Terminal MEASURE held back and re-applied == collapsed inline (via a total channel).
        collapsed = program.copy()
        collapsed += I(0)  # a later operation on qubit 0 makes the measurement non-terminal
        np.testing.assert_allclose(rho, _dm(collapsed, noise_model=noise_model), atol=1e-12)

    def test_jit_and_grad(self):
        program = Program()
        program += Declare("ro", "BIT", 1)
        program += Declare("theta", "REAL", 1)
        program += RX(MemoryReference("theta", 0), 0)
        program += MEASURE(0, ("ro", 0))
        sim = DensityMatrixSimulator(program)
        theta = 0.3
        params = jnp.array([theta])
        np.testing.assert_allclose(jax.jit(sim.outcome_probabilities)(params), sim.outcome_probabilities(params))
        grad = jax.grad(lambda p: sim.outcome_probabilities(p)[1])(params)
        np.testing.assert_allclose(grad, [np.sin(theta) / 2], atol=1e-10)

    def test_program_without_measurement_reports_clearly(self):
        sim = DensityMatrixSimulator(Program(H(0)))
        with pytest.raises(ValueError, match="no terminal measurement"):
            sim.outcome_probabilities()


class TestInitialState:
    """``compute``/``outcome_probabilities`` starting from a caller-supplied state.

    The point of the argument is to let a circuit be split in two so that a shared prefix is
    evolved once and only the varying tail is re-run.  What these check is the property that
    makes the split sound: prefix-then-tail must equal the whole circuit exactly.

    Both halves have to span the same register -- ``qubits`` must list exactly the qubits the
    program acts on -- so a half that leaves a qubit idle pads it with ``I``.
    """

    PREFIX = Program(H(0), CNOT(0, 1), RX(0.3, 1), I(2))
    TAIL = Program(RX(1.1, 0), CNOT(1, 2))

    def test_prefix_then_tail_equals_the_whole_circuit(self):
        qubits = [0, 1, 2]
        whole = DensityMatrixSimulator(self.PREFIX + self.TAIL, qubits=qubits).compute()
        prefix = DensityMatrixSimulator(self.PREFIX, qubits=qubits).compute()
        split = DensityMatrixSimulator(self.TAIL, qubits=qubits).compute(None, prefix)
        np.testing.assert_allclose(np.asarray(split.matrix), np.asarray(whole.matrix), atol=1e-12)

    def test_outcome_probabilities_from_a_prepared_state(self):
        qubits = [0, 1, 2]
        measured = Program(Declare("ro", "BIT", 3)) + self.TAIL
        measured += [MEASURE(q, ("ro", i)) for i, q in enumerate(qubits)]
        whole = DensityMatrixSimulator(self.PREFIX + measured, qubits=qubits).outcome_probabilities()
        prefix = DensityMatrixSimulator(self.PREFIX, qubits=qubits).compute()
        split = DensityMatrixSimulator(measured, qubits=qubits).outcome_probabilities(None, prefix)
        np.testing.assert_allclose(np.asarray(split), np.asarray(whole), atol=1e-12)

    def test_a_noisy_prefix_carries_its_mixedness_into_the_tail(self):
        """The split has to work for a *mixed* prefix, which is the case it exists for."""
        qubits = [0, 1]
        noise_model = NoiseModel.from_channels([Channel.from_depolarizing_constant(H(0), 0.8)])
        prefix, tail = Program(H(0), I(1)), Program(CNOT(0, 1))
        whole = DensityMatrixSimulator(prefix + tail, qubits=qubits, noise_model=noise_model).compute()
        prepared = DensityMatrixSimulator(prefix, qubits=qubits, noise_model=noise_model).compute()
        assert np.trace(np.asarray(prepared.matrix) @ np.asarray(prepared.matrix)).real < 0.99, "prefix is mixed"
        split = DensityMatrixSimulator(tail, qubits=qubits, noise_model=noise_model).compute(None, prepared)
        np.testing.assert_allclose(np.asarray(split.matrix), np.asarray(whole.matrix), atol=1e-12)

    def test_a_state_vector_is_rejected_by_the_density_matrix_backend(self):
        """The dangerous case: without the check this broadcasts into a wrong-but-plausible matrix."""
        sim = DensityMatrixSimulator(Program(X(0)), qubits=[0])
        with pytest.raises(TypeError, match="must be a DensityMatrix"):
            sim.compute(None, qx.zero_state_vector(dims=(2,)))

    def test_a_mismatched_register_is_rejected(self):
        sim = DensityMatrixSimulator(Program(X(0)), qubits=[0])
        with pytest.raises(ValueError, match=r"dims \(2, 2\) but this simulator's register is \(2,\)"):
            sim.compute(None, qx.zero_state_matrix(dims=(2, 2)))


class TestPrecisionWarnings:
    def test_warns_about_reduced_matmul_precision_on_accelerators(self, monkeypatch):
        monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
        with pytest.warns(UserWarning, match="matmul precision"):
            DensityMatrixSimulator(Program(H(0)))
        with jax.default_matmul_precision("highest"), warnings.catch_warnings():
            warnings.simplefilter("error")
            DensityMatrixSimulator(Program(H(0)))

    def test_no_matmul_warning_on_cpu(self, monkeypatch):
        monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            DensityMatrixSimulator(Program(H(0)))


class TestQutritsAndLeakage:
    @pytest.fixture
    def qutrit_program(self):
        program = Program()
        program.defgate("QUTRIT_X", _QUTRIT_X)
        return program

    def test_qutrit_gate_cycles_the_levels(self, qutrit_program):
        program = qutrit_program.copy()
        program += Gate("QUTRIT_X", [], [0])
        rho = _dm(program, qubits=[0])
        assert rho.shape == (3, 3)
        np.testing.assert_allclose(rho, np.diag([0.0, 0.0, 1.0]), atol=1e-10)

    def test_qutrit_matches_state_vector(self, qutrit_program):
        program = qutrit_program.copy()
        program += Gate("QUTRIT_X", [], [0])
        program += Gate("QUTRIT_X", [], [0])
        assert_pure(_dm(program, qubits=[0]), _sv(program, qubits=[0]))

    def test_leakage_populates_the_second_excited_level(self, qutrit_program):
        """A gate that leaks |1> -> |2> must show up as population in level 2."""
        leak = 0.2
        # A rotation in the 1-2 subspace by angle phi moves sin^2(phi / 2) of the population.
        phi = 2 * np.arcsin(np.sqrt(leak))
        program = qutrit_program.copy()
        program.defgate("LEAK", np.asarray(qx.gates.TRY12(phi).matrix))
        program += Gate("QUTRIT_X", [], [0])  # |0> -> |2>
        program += Gate("QUTRIT_X", [], [0])  # |2> -> |1>
        program += Gate("LEAK", [], [0])  # partially leak |1> -> |2>
        rho = _dm(program, qubits=[0])
        assert_physical(rho)
        np.testing.assert_allclose(np.real(np.diag(rho)), [0.0, 1 - leak, leak], atol=1e-10)

    def test_qutrit_depolarizing_shrinks_toward_maximally_mixed(self, qutrit_program):
        """A depolarizing channel on a qutrit shrinks toward I/3, not I/2."""
        program = qutrit_program.copy()
        program += Gate("QUTRIT_X", [], [0])
        shrink = 0.4
        custom_gates = {"QUTRIT_X": qx.Unitary.from_matrix(jnp.asarray(_QUTRIT_X), ((3,), (3,)))}
        channel = Channel.from_depolarizing_constant(Gate("QUTRIT_X", [], [0]), shrink, custom_gates=custom_gates)
        rho = _dm(program, qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        assert_physical(rho)
        ideal = np.diag([0.0, 0.0, 1.0]).astype(complex)
        expected = shrink * ideal + (1 - shrink) * np.eye(3) / 3
        np.testing.assert_allclose(rho, expected, atol=1e-9)

    def test_qutrit_reset_returns_to_ground(self, qutrit_program):
        program = qutrit_program.copy()
        program += Gate("QUTRIT_X", [], [0])
        program += RESET(0)
        rho = _dm(program, qubits=[0])
        np.testing.assert_allclose(rho, np.diag([1.0, 0.0, 0.0]), atol=1e-10)

    def test_mixed_qubit_qutrit_register(self, qutrit_program):
        program = qutrit_program.copy()
        program += Gate("QUTRIT_X", [], [0])
        program += X(1)
        sim = DensityMatrixSimulator(program, qubits=[0, 1])
        assert sim.dims == (3, 2)
        rho = np.asarray(sim.compute().matrix)
        assert rho.shape == (6, 6)
        assert_pure(rho, _sv(program, qubits=[0, 1]))


class TestParametricPrograms:
    def test_parametric_matches_literal(self):
        program = Program(Declare("theta", "REAL", 1), RX(MemoryReference("theta", 0), 0))
        rho = _dm(program, qubits=[0], memory_map={"theta": [0.6]})
        np.testing.assert_allclose(rho, _dm(Program(RX(0.6, 0)), qubits=[0]), atol=1e-10)

    def test_noise_on_a_literal_gate_alongside_a_parametric_one(self):
        """Noise attaches to literal-angle gates; parametric gates intentionally carry none.

        Noise on a runtime-parametric gate is deliberately unsupported: the only truly
        continuous gate on hardware is the virtual ``RZ``, which has no pulse and so no noise,
        while every other rotation is calibrated at fixed angles. See
        ``test_simulator_jit_grad.py::test_noise_model_does_not_apply_to_parametric_gates``.
        Here the channel sits on the literal ``X 0``, which does apply.
        """
        program = Program(Declare("theta", "REAL", 1), RX(MemoryReference("theta", 0), 0), X(0))
        noise_model = NoiseModel.from_channels([Channel.from_depolarizing_constant(X(0), 0.9)])
        sim = DensityMatrixSimulator(program, qubits=[0], noise_model=noise_model)
        rho = np.asarray(sim.compute(sim.linearize({"theta": [0.6]})).matrix)
        assert_physical(rho)
        clean = _dm(program, qubits=[0], memory_map={"theta": [0.6]})
        assert not np.allclose(rho, clean, atol=1e-6), "noise on the literal gate must be applied"


class TestCustomGates:
    def test_defgate_matches_state_vector(self):
        matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        program = Program()
        program.defgate("MYCNOT", matrix)
        program += Program(X(0), Gate("MYCNOT", [], [0, 1]))
        assert_pure(_dm(program, qubits=[0, 1]), _sv(program, qubits=[0, 1]))

    def test_defgate_with_noise(self):
        matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        program = Program()
        program.defgate("MYX", matrix)
        inst = Gate("MYX", [], [0])
        program += inst
        custom_gates = {"MYX": qx.Unitary.from_matrix(jnp.asarray(matrix), ((2,), (2,)))}
        channel = Channel.from_depolarizing_constant(inst, 0.8, custom_gates=custom_gates)
        rho = _dm(program, qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        assert_physical(rho)
        expected = 0.8 * np.diag([0.0, 1.0]) + 0.2 * np.eye(2) / 2
        np.testing.assert_allclose(rho, expected, atol=1e-9)


class TestErrorHandling:
    def test_duplicate_qubits_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            DensityMatrixSimulator(Program(X(0)), qubits=[0, 0])

    def test_qubit_outside_register_reports_clearly(self):
        with pytest.raises(ValueError, match=r"must be exactly the qubits the program acts on.*missing: \[3\]"):
            _dm(Program(X(0), X(3)), qubits=[0])

    def test_spectator_qubit_reports_clearly(self):
        """``qubits`` fixes the register *order*; a qubit the program never touches is an error."""
        with pytest.raises(ValueError, match=r"not in the program: \[1\]"):
            _dm(Program(X(0)), qubits=[0, 1])

    def test_qubits_may_reorder_the_register(self):
        program = Program(X(0), I(1))
        np.testing.assert_allclose(
            np.diag(_dm(program, qubits=[1, 0])), np.diag(_dm(program, qubits=[0, 1]))[[0, 2, 1, 3]], atol=1e-12
        )

    def test_wrong_parameter_count_reports_clearly(self):
        program = Program(Declare("theta", "REAL", 1), RX(MemoryReference("theta", 0), 0))
        sim = DensityMatrixSimulator(program, qubits=[0])
        with pytest.raises(ValueError, match="Expected 1 parameter"):
            sim.compute(jnp.array([]))
        with pytest.raises(ValueError, match="Expected 1 parameter"):
            sim.compute(jnp.array([0.1, 0.2]))
        with pytest.raises(ValueError, match="cannot be omitted"):
            sim.compute()

    @pytest.mark.parametrize(
        ("program", "modifier"),
        [
            (Program(RX(0.7, 0).dagger()), "DAGGER"),
            (Program(X(1).controlled(0)), "CONTROLLED"),
            (Program(Declare("theta", "REAL", 1), RX(MemoryReference("theta", 0), 0).dagger()), "DAGGER"),
        ],
        ids=["dagger", "controlled", "dagger_parametric"],
    )
    def test_unsupported_modifier_reports_clearly(self, program, modifier):
        """Modifiers are rejected, never silently dropped.

        Silently dropping one is the dangerous failure: ``DAGGER RX(0.7) 0`` would simulate
        ``RX(+0.7)`` and return a plausible wrong state.
        """
        with pytest.raises(ValueError, match=f"modifiers are not supported.*{modifier}"):
            _dm(program)

    def test_complex_valued_parameter_for_builtin_gate_reports_clearly(self):
        theta = MemoryReference("theta", 0)
        program = Program(Declare("theta", "REAL", 1), RX(quil_cis(theta), 0))
        with pytest.raises(ValueError, match="complex-valued"):
            DensityMatrixSimulator(program, qubits=[0])

    def test_feed_forward_parameter_reports_clearly(self):
        program = Program("DECLARE ro BIT[1]", "MEASURE 0 ro[0]", "RX(ro[0]) 0")
        with pytest.raises(ValueError, match="Feed-forward"):
            DensityMatrixSimulator(program, qubits=[0])
