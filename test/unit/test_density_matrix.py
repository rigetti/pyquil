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

import jax.numpy as jnp
import numpy as np
import pytest
import quax as qx

from pyquil.gates import CCNOT, CNOT, CSWAP, CZ, MEASURE, RESET, RX, RY, RZ, SWAP, H, I, S, T, X, Y, Z
from pyquil.noise._channels import (
    Channel,
    MeasurementChannel,
    ResetChannel,
    SuperopChannel,
    SuperopResetChannel,
)
from pyquil.noise._noise_model import NoiseModel
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Declare, Gate, ResetQubit
from pyquil.simulation._reference import ReferenceDensitySimulator, ReferenceWavefunctionSimulator
from pyquil.simulation._simulator import DensityMatrixSimulator, PureStateVectorSimulator

_EMPTY_PARAMS = jnp.array([], dtype=float)

# Qutrit X (cyclic shift |0>->|1>->|2>->|0>) and a partial-leakage rotation on the 1-2 subspace.
_QUTRIT_X = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)


def _dm(program, qubits=None, noise_model=None, memory_map=None, **kwargs):
    """Final density matrix as a plain numpy array."""
    sim = DensityMatrixSimulator(program, qubits=qubits, noise_model=noise_model, **kwargs)
    params = sim.linearize(memory_map) if memory_map else _EMPTY_PARAMS
    return np.asarray(sim.compute(params).matrix)


def _sv(program, qubits=None, memory_map=None, **kwargs):
    """Final state vector as a flat numpy array."""
    sim = PureStateVectorSimulator(program, qubits=qubits, **kwargs)
    params = sim.linearize(memory_map) if memory_map else _EMPTY_PARAMS
    return np.asarray(sim.compute(params).matrix).reshape(-1)


def _reverse_endianness(array, n_qubits):
    """Reverse qubit significance, to compare against pyQuil's little-endian simulators.

    Works for a state vector (1-D) or a density matrix (2-D) over ``n_qubits`` qubits.
    """
    axes = tuple(reversed(range(n_qubits)))
    if array.ndim == 1:
        return array.reshape((2,) * n_qubits).transpose(axes).reshape(-1)
    tensor = array.reshape((2,) * (2 * n_qubits))
    perm = axes + tuple(n + n_qubits for n in axes)
    return tensor.transpose(perm).reshape(2**n_qubits, 2**n_qubits)


def _assert_pure(rho, psi, atol=1e-10):
    """Assert ``rho == |psi><psi|``."""
    np.testing.assert_allclose(rho, np.outer(psi, psi.conj()), atol=atol)


def _assert_physical(rho, atol=1e-9):
    """Assert *rho* is a valid density matrix: unit trace, Hermitian, positive semi-definite."""
    assert np.trace(rho).real == pytest.approx(1.0, abs=atol)
    np.testing.assert_allclose(rho, rho.conj().T, atol=atol)
    assert np.linalg.eigvalsh(rho).min() > -atol


# Programs reused across the noiseless equivalence tests. Deliberately includes gates whose
# operands are *not* in ascending order -- the compressor advertises a sorted subsystem for
# merged groups but an unmerged operation keeps its own operand order, and getting that wrong
# silently permutes qudits in the density-matrix path only.
_PROGRAMS = {
    "empty": Program(),
    "single_x": Program(X(0)),
    "hadamard": Program(H(0)),
    "bell": Program(H(0), CNOT(0, 1)),
    "ghz3": Program(H(0), CNOT(0, 1), CNOT(1, 2)),
    "clifford_chain": Program(H(0), S(0), T(0), H(0), Z(0), Y(0)),
    "rotations": Program(RX(0.3, 0), RY(0.7, 0), RZ(1.1, 0)),
    "two_qubit_mixed": Program(RX(0.4, 0), RY(0.9, 1), CNOT(0, 1), RZ(1.1, 0), CZ(0, 1)),
    "ccnot_sorted": Program(H(0), H(1), CCNOT(0, 1, 2)),
    "ccnot_reversed": Program(X(1), X(2), CCNOT(2, 1, 0)),
    "cnot_reversed": Program(X(1), CNOT(1, 0)),
    "cswap_reversed": Program(X(0), X(2), CSWAP(2, 1, 0)),
    "swap_reversed": Program(X(1), SWAP(2, 0)),
    "deep": Program(RX(0.4, 0), RY(0.9, 1), CNOT(0, 1), RZ(1.1, 0), CZ(1, 2), RX(0.2, 2), SWAP(0, 2)),
    "idle_qubit": Program(X(0), I(1)),
}


class TestNoiselessMatchesStateVector:
    """A noiseless density matrix must equal the outer product of the pure state."""

    @pytest.mark.parametrize("name", sorted(_PROGRAMS))
    def test_matches_state_vector(self, name):
        program = _PROGRAMS[name]
        qubits = sorted(program.get_qubit_indices()) or [0]
        _assert_pure(_dm(program, qubits=qubits), _sv(program, qubits=qubits))

    @pytest.mark.parametrize("name", sorted(_PROGRAMS))
    @pytest.mark.parametrize("max_subsystem_size", [1, 2, 3, 4])
    def test_independent_of_compressor_budget(self, name, max_subsystem_size):
        """``max_subsystem_size`` is a performance knob and must not change the result."""
        program = _PROGRAMS[name]
        qubits = sorted(program.get_qubit_indices()) or [0]
        reference = _dm(program, qubits=qubits)
        got = _dm(program, qubits=qubits, max_subsystem_size=max_subsystem_size)
        np.testing.assert_allclose(got, reference, atol=1e-10)

    @pytest.mark.parametrize("name", sorted(_PROGRAMS))
    def test_is_physical(self, name):
        program = _PROGRAMS[name]
        qubits = sorted(program.get_qubit_indices()) or [0]
        _assert_physical(_dm(program, qubits=qubits))


class TestAgainstPyquilReferenceSimulators:
    """Cross-check against pyQuil's independent (little-endian) reference implementations."""

    _COMPARABLE = [
        "single_x",
        "hadamard",
        "bell",
        "ghz3",
        "clifford_chain",
        "rotations",
        "two_qubit_mixed",
        "ccnot_sorted",
        "ccnot_reversed",
        "cnot_reversed",
        "cswap_reversed",
        "deep",
    ]

    @pytest.mark.parametrize("name", _COMPARABLE)
    def test_matches_reference_density_simulator(self, name):
        program = _PROGRAMS[name]
        n = len(sorted(program.get_qubit_indices()))
        got = _reverse_endianness(_dm(program, qubits=sorted(program.get_qubit_indices())), n)
        expected = np.asarray(ReferenceDensitySimulator(n_qubits=n).do_program(program).density)
        np.testing.assert_allclose(got, expected, atol=1e-9)

    @pytest.mark.parametrize("name", _COMPARABLE)
    def test_matches_reference_wavefunction_simulator(self, name):
        program = _PROGRAMS[name]
        n = len(sorted(program.get_qubit_indices()))
        got = _reverse_endianness(_sv(program, qubits=sorted(program.get_qubit_indices())), n)
        expected = np.asarray(ReferenceWavefunctionSimulator(n_qubits=n).do_program(program).wf).reshape(-1)
        # Compare up to global phase.
        overlap = abs(np.vdot(expected, got)) / (np.linalg.norm(expected) * np.linalg.norm(got))
        assert overlap == pytest.approx(1.0, abs=1e-9)



class TestBigEndianOrdering:
    """The register order is big-endian: ``qubits[0]`` is the most significant subsystem."""

    def test_x_on_first_qubit_sets_high_bit(self):
        rho = _dm(Program(X(0)), qubits=[0, 1])
        assert int(np.argmax(np.real(np.diag(rho)))) == 0b10

    def test_x_on_second_qubit_sets_low_bit(self):
        rho = _dm(Program(X(1)), qubits=[0, 1])
        assert int(np.argmax(np.real(np.diag(rho)))) == 0b01

    def test_explicit_qubit_order_is_respected(self):
        """Reversing ``qubits`` reverses which subsystem the amplitude lands in."""
        assert int(np.argmax(np.real(np.diag(_dm(Program(X(5)), qubits=[5, 6]))))) == 0b10
        assert int(np.argmax(np.real(np.diag(_dm(Program(X(5)), qubits=[6, 5]))))) == 0b01

    def test_opposite_of_pyquil_reference(self):
        """Documents the deliberate divergence from the rest of pyQuil."""
        program = Program(X(0))
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
        _assert_physical(noisy)

    @pytest.mark.parametrize("angle", [np.pi / 2, np.pi])
    def test_pauli_noise_reproduces_its_error_probabilities(self, angle):
        """A one-shot Pauli channel applies exactly the requested error probabilities."""
        gate = RX(angle, 0)
        pauli_noise = {"X": 0.03, "Y": 0.05, "Z": 0.07}
        channel = SuperopChannel.from_pauli_noise(gate, pauli_noise)
        rho = _dm(Program(gate), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        _assert_physical(rho)

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
        _assert_physical(rho)
        assert np.trace(rho @ rho).real == pytest.approx(1.0, abs=1e-9), "coherent error must stay pure"
        np.testing.assert_allclose(rho, _dm(Program(RX(angle + over, 0)), qubits=[0]), atol=1e-10)
        # A purely coherent error has unit unitarity and no stochastic component.
        assert channel.unitarity == pytest.approx(1.0, abs=1e-8)
        assert channel.stochastic_infidelity == pytest.approx(0.0, abs=1e-8)

    def test_random_coherent_error_is_unitary_and_hits_requested_fidelity(self):
        gate = RX(np.pi / 2, 0)
        channel = Channel.from_random_coherent_error(gate, 0.97, rng=np.random.default_rng(4))
        rho = _dm(Program(gate), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        _assert_physical(rho)
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
        _assert_physical(noisy)
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
        _assert_physical(rho)
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
        _assert_physical(rho)
        # Ideal result: control stays 1, target flips to 1 -> |11>, whatever the operand order.
        assert np.real(np.diag(rho))[0b11] == max(np.real(np.diag(rho)))


class TestDecoherence:
    """State-dependent (non-unital) noise: T1 relaxation and T2 dephasing."""

    def test_amplitude_damping_matches_exponential_decay(self):
        """Excited population after damping for t must be exp(-t/T1)."""
        t1, duration = 20e-6, 5e-6
        channel = ResetChannel.from_amplitude_damping(ResetQubit(0), gamma=1.0 / t1, gate_time=duration)
        rho = _dm(Program(X(0), RESET(0)), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        _assert_physical(rho)
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
        _assert_physical(rho)
        assert abs(rho[0, 1]) == pytest.approx(0.5 * np.exp(-duration / t2), abs=1e-6)

    def test_t1_t2_during_rx_pi_matches_frozen_lindblad_golden(self):
        """End-to-end golden for 1Q T1/T2 during a gate.

        Produced with qutip's ``mesolve``, but see the module docstring: quax's ``evolve`` is
        already tested against qutip, so this pins the pyQuil-side chain (T1/T2 conversion,
        gate-Hamiltonian scaling, application) rather than the integrator.

        Regenerate with::

            ch = Channel.from_coherence_times(RX(np.pi, 0), 40e-9, t1s=[20e-6], t2s=[15e-6])
            H = qutip.Qobj(np.asarray(ch.lindbladian.hamiltonian.matrix))
            c_ops = [qutip.Qobj(j) for j in np.asarray(ch.lindbladian.jump_operators.matrix) if np.linalg.norm(j) > 1e-15]
            qutip.mesolve(H, qutip.Qobj(np.diag([1.0, 0.0]).astype(complex)), [0, 40e-9], c_ops=c_ops)
        """
        channel = Channel.from_coherence_times(RX(np.pi, 0), gate_duration=40e-9, t1s=[20e-6], t2s=[15e-6])
        rho = _dm(Program(RX(np.pi, 0)), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        expected = np.array([[0.00116584637 + 0j, 0.000635886393j], [-0.000635886393j, 0.99883415363 + 0j]])
        _assert_physical(rho)
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
        _assert_physical(rho)
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
        _assert_physical(rho)

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
        _assert_physical(rho)
        # depolarizing_p @ RESET on any input gives p|0><0| + (1-p)I/2, with p = 2F - 1.
        p = 2 * fidelity - 1
        np.testing.assert_allclose(rho, p * np.diag([1.0, 0.0]) + (1 - p) * np.eye(2) / 2, atol=1e-9)

    @pytest.mark.parametrize("reset", [RESET(0), RESET()])
    def test_reset_noise_applies_to_targeted_and_global_forms(self, reset):
        """Regression: global RESET used to ignore the noise model entirely."""
        channel = SuperopResetChannel.from_reset_fidelity(ResetQubit(0), fidelity=0.5)
        rho = _dm(Program(X(0)) + Program(reset), qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        np.testing.assert_allclose(rho, np.eye(2) / 2, atol=1e-9)


class TestQutritsAndLeakage:
    @pytest.fixture
    def qutrit_program(self):
        program = Program()
        program.defgate("QUTRIT_X", _QUTRIT_X)
        return program

    def test_qutrit_gate_cycles_the_levels(self, qutrit_program):
        program = qutrit_program + Program(Gate("QUTRIT_X", [], [0]))
        rho = _dm(program, qubits=[0])
        assert rho.shape == (3, 3)
        np.testing.assert_allclose(rho, np.diag([0.0, 1.0, 0.0]), atol=1e-10)

    def test_qutrit_matches_state_vector(self, qutrit_program):
        program = qutrit_program + Program(Gate("QUTRIT_X", [], [0]), Gate("QUTRIT_X", [], [0]))
        _assert_pure(_dm(program, qubits=[0]), _sv(program, qubits=[0]))

    def test_leakage_populates_the_second_excited_level(self, qutrit_program):
        """A gate that leaks |1> -> |2> must show up as population in level 2."""
        leak = 0.2
        c, s = np.sqrt(1 - leak), np.sqrt(leak)
        leaky = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=complex)
        program = qutrit_program
        program.defgate("LEAK", leaky)
        program += Gate("QUTRIT_X", [], [0])  # |0> -> |1>
        program += Gate("LEAK", [], [0])  # partially leak |1> -> |2>
        rho = _dm(program, qubits=[0])
        _assert_physical(rho)
        np.testing.assert_allclose(np.real(np.diag(rho)), [0.0, 1 - leak, leak], atol=1e-10)

    def test_qutrit_depolarizing_shrinks_toward_maximally_mixed(self, qutrit_program):
        """A depolarizing channel on a qutrit shrinks toward I/3, not I/2."""
        program = qutrit_program + Program(Gate("QUTRIT_X", [], [0]))
        shrink = 0.4
        custom_gates = {"QUTRIT_X": qx.Unitary.from_matrix(jnp.asarray(_QUTRIT_X), ((3,), (3,)))}
        channel = Channel.from_depolarizing_constant(Gate("QUTRIT_X", [], [0]), shrink, custom_gates=custom_gates)
        rho = _dm(program, qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        _assert_physical(rho)
        ideal = np.diag([0.0, 1.0, 0.0]).astype(complex)
        expected = shrink * ideal + (1 - shrink) * np.eye(3) / 3
        np.testing.assert_allclose(rho, expected, atol=1e-9)

    def test_qutrit_reset_returns_to_ground(self, qutrit_program):
        program = qutrit_program + Program(Gate("QUTRIT_X", [], [0]), RESET(0))
        rho = _dm(program, qubits=[0])
        np.testing.assert_allclose(rho, np.diag([1.0, 0.0, 0.0]), atol=1e-10)

    def test_mixed_qubit_qutrit_register(self, qutrit_program):
        program = qutrit_program + Program(Gate("QUTRIT_X", [], [0]), X(1))
        sim = DensityMatrixSimulator(program, qubits=[0, 1])
        assert sim.dims == (3, 2)
        rho = np.asarray(sim.compute(_EMPTY_PARAMS).matrix)
        assert rho.shape == (6, 6)
        _assert_pure(rho, _sv(program, qubits=[0, 1]))


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
        _assert_physical(rho)
        clean = _dm(program, qubits=[0], memory_map={"theta": [0.6]})
        assert not np.allclose(rho, clean, atol=1e-6), "noise on the literal gate must be applied"


class TestCustomGates:
    def test_defgate_matches_state_vector(self):
        matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        program = Program()
        program.defgate("MYCNOT", matrix)
        program += Program(X(0), Gate("MYCNOT", [], [0, 1]))
        _assert_pure(_dm(program, qubits=[0, 1]), _sv(program, qubits=[0, 1]))

    def test_defgate_with_noise(self):
        matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        program = Program()
        program.defgate("MYX", matrix)
        inst = Gate("MYX", [], [0])
        program += inst
        custom_gates = {"MYX": qx.Unitary.from_matrix(jnp.asarray(matrix), ((2,), (2,)))}
        channel = Channel.from_depolarizing_constant(inst, 0.8, custom_gates=custom_gates)
        rho = _dm(program, qubits=[0], noise_model=NoiseModel.from_channels([channel]))
        _assert_physical(rho)
        expected = 0.8 * np.diag([0.0, 1.0]) + 0.2 * np.eye(2) / 2
        np.testing.assert_allclose(rho, expected, atol=1e-9)


class TestErrorHandling:
    def test_duplicate_qubits_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            DensityMatrixSimulator(Program(X(0)), qubits=[0, 0])

    def test_qubit_outside_register_reports_clearly(self):
        with pytest.raises(ValueError, match="simulated register"):
            _dm(Program(X(0), X(3)), qubits=[0])

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
        ],
        ids=["dagger", "controlled"],
    )
    def test_unsupported_modifier_reports_clearly(self, program, modifier):
        """Modifiers are rejected, never silently dropped.

        Silently dropping one is the dangerous failure: ``DAGGER RX(0.7) 0`` would simulate
        ``RX(+0.7)`` and return a plausible wrong state.
        """
        with pytest.raises(ValueError, match=f"modifiers are not supported.*{modifier}"):
            _dm(program, qubits=[0, 1])

    def test_expression_valued_parameter_reports_clearly(self):
        program = Program(Declare("theta", "REAL", 1), RX(MemoryReference("theta", 0) / 2, 0))
        with pytest.raises(ValueError, match="expression over memory"):
            DensityMatrixSimulator(program, qubits=[0])

    def test_feed_forward_parameter_reports_clearly(self):
        program = Program("DECLARE ro BIT[1]", "MEASURE 0 ro[0]", "RX(ro[0]) 0")
        with pytest.raises(ValueError, match="Feed-forward"):
            DensityMatrixSimulator(program, qubits=[0])
