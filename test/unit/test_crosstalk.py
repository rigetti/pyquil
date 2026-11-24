"""Tests for crosstalk cycle channels: the representation and the accuracy of its lowering.

Accuracy is measured against :meth:`CrosstalkCycleChannel.exact_process`, the dense evolution of the
joint generator, as the spectral norm of the difference of the two superoperators.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import quax as qx
from scipy.linalg import expm

from pyquil.gates import CZ, MEASURE, RX, I
from pyquil.noise._channels import Channel, CycleChannel, MeasurementChannel, SuperopChannel
from pyquil.noise._crosstalk import CrosstalkCycleChannel, _pauli_terms, xy_crosstalk, zz_crosstalk
from pyquil.noise._noise_model import NoiseModel, estimate_program_fidelity
from pyquil.paulis import PauliSum, PauliTerm, sX, sY, sZ
from pyquil.quil import Program
from pyquil.quilatom import Parameter, Qubit
from pyquil.quilbase import Gate
from pyquil.simulation._resolver import expand_program
from pyquil.simulation._simulator import DensityMatrixSimulator, TrajectorySimulator

CZ_TIME = 200e-9
RX_TIME = 40e-9
T1 = T2 = 30e-6
NO_CROSSTALK = PauliSum([])

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.diag([1.0, -1.0]).astype(complex)


def _channel(gate, duration, coherence_time=None, dims=None):
    """A gate channel over ``duration``: ideal, or with T1 = T2 = ``coherence_time``."""
    n = len(gate.qubits)
    if coherence_time is None:
        return Channel.from_lindbladian(gate, qx.lindbladians.depolarizing(0.0, dims or (2,) * n), duration)
    return Channel.from_coherence_times(gate, duration, [coherence_time] * n, [coherence_time] * n)


def _cycle(gates, duration, coherence_time=None):
    channels = [_channel(gate, duration, coherence_time) for gate in gates]
    cycle = channels[0]
    for channel in channels[1:]:
        cycle = cycle | channel
    if isinstance(cycle, CycleChannel):
        return cycle
    return cycle | _channel(I(max(cycle.qubits) + 1), duration, coherence_time)


def _matrix(superop):
    return np.asarray(superop.matrix)


def _compose(channel, operations):
    """Compose *operations* into one superoperator on the cycle's qubits."""
    qubits = channel.qubits
    dims = tuple(channel._qubit_dims[q] for q in qubits)
    total = None
    for op, op_qubits in operations:
        embedded = qx.embed(qx.to_superop(op), dims, tuple(qubits.index(q) for q in op_qubits))
        total = embedded if total is None else embedded @ total
    return total


def _lowered(channel):
    """The channel a simulator applies: :meth:`CrosstalkCycleChannel.operations`, composed."""
    return _compose(channel, channel.operations())


def _bare(channel):
    """The cycle without its crosstalk."""
    return _compose(channel, channel.operations()[-len(channel.channels) :])


def _error(channel):
    """Spectral-norm distance between the lowered and the exact process."""
    return float(np.linalg.norm(_matrix(_lowered(channel)) - _matrix(channel.exact_process()), 2))


def _effect(channel):
    """Size of the crosstalk itself: the error made by ignoring it altogether."""
    return float(np.linalg.norm(_matrix(_bare(channel)) - _matrix(channel.exact_process()), 2))


def _slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def _qubits(channel):
    return [qubits for _, qubits in channel.operations()]


RX_GATES = (RX(np.pi / 2, 0), RX(np.pi / 2, 1), I(2))


def _rx_channel(rate, steps=1, coherence_time=None):
    cycle = _cycle(RX_GATES, RX_TIME, coherence_time)
    return CrosstalkCycleChannel(cycle, zz_crosstalk(0, 1, rate) + xy_crosstalk(1, 2, rate), RX_TIME, steps)


def _cz_channel(crosstalk, steps=1, coherence_time=None):
    return CrosstalkCycleChannel(_cycle((CZ(0, 1), CZ(2, 3)), CZ_TIME, coherence_time), crosstalk, CZ_TIME, steps)


def _crosstalk_infidelity(channel, ideal):
    """Exact process infidelity against the ideal gates, of a channel whose gates are noiseless."""
    return 1 - float(qx.process_fidelity(channel.exact_process(), ideal))


# ══════════════════════════════════════════════════════════
# Building blocks
# ══════════════════════════════════════════════════════════


class TestCrosstalkHamiltonians:
    def test_zz_shifts_only_the_doubly_excited_state(self):
        ((hamiltonian, qubits),) = _pauli_terms(zz_crosstalk(3, 5, 1e5))
        diagonal = np.diag(np.asarray(hamiltonian.matrix))
        assert qubits == (3, 5)
        np.testing.assert_allclose(diagonal - diagonal[0], [0, 0, 0, 2 * np.pi * 1e5], atol=1e-9)

    def test_xy_exchanges_single_excitations(self):
        ((hamiltonian, _),) = _pauli_terms(xy_crosstalk(0, 1, 1e5))
        expected = np.zeros((4, 4))
        expected[1, 2] = expected[2, 1] = 2 * np.pi * 1e5
        np.testing.assert_allclose(np.asarray(hamiltonian.matrix), expected, atol=1e-9)

    def test_matrices_are_big_endian_in_qubit_order(self):
        ((hamiltonian, qubits),) = _pauli_terms(PauliSum([sX(5) * sZ(3)]))
        assert qubits == (3, 5)
        np.testing.assert_allclose(np.asarray(hamiltonian.matrix), np.kron(PAULI_Z, PAULI_X))

    def test_terms_group_by_their_widest_support(self):
        crosstalk = zz_crosstalk(0, 1, 1.0) + xy_crosstalk(1, 2, 1.0) + 0.5 * sY(2) + 2.0
        groups = _pauli_terms(crosstalk)
        assert [qubits for _, qubits in groups] == [(0, 1), (1, 2)]
        y_on_2 = 0.5 * np.kron(np.eye(2), PAULI_Y)
        xy = np.pi * (np.kron(PAULI_X, PAULI_X) + np.kron(PAULI_Y, PAULI_Y))
        np.testing.assert_allclose(np.asarray(groups[1][0].matrix), xy + y_on_2, atol=1e-12)


class TestValidation:
    def test_crosstalk_outside_cycle(self):
        with pytest.raises(ValueError, match="outside the cycle"):
            _cz_channel(zz_crosstalk(1, 9, 1e5))

    def test_coefficients_must_be_real_numbers(self):
        with pytest.raises(ValueError, match="real numbers"):
            _cz_channel(1j * sZ(1) * sZ(2))
        with pytest.raises(ValueError, match="real numbers"):
            _cz_channel(PauliSum([PauliTerm("Z", 1, Parameter("theta"))]))

    def test_crosstalk_must_be_a_pauli_sum(self):
        with pytest.raises(TypeError, match="PauliSum"):
            _cz_channel(np.eye(4))

    def test_a_single_term_is_accepted(self):
        channel = _cz_channel(2.0 * sZ(1) * sZ(2))
        assert isinstance(channel.crosstalk, PauliSum)

    def test_duration_and_steps(self):
        cycle = _cycle((CZ(0, 1), CZ(2, 3)), CZ_TIME)
        with pytest.raises(ValueError, match="duration"):
            CrosstalkCycleChannel(cycle, NO_CROSSTALK, 0.0)
        with pytest.raises(ValueError, match="steps"):
            CrosstalkCycleChannel(cycle, NO_CROSSTALK, CZ_TIME, steps=0)

    def test_exact_process_is_size_limited(self):
        with pytest.raises(ValueError, match="max_dimension"):
            _cz_channel(zz_crosstalk(1, 2, 1e5)).exact_process(max_dimension=8)

    def test_exact_process_needs_generators(self):
        cz = _channel(CZ(0, 1), CZ_TIME)
        raw = SuperopChannel(inst=I(2), process=qx.to_superop(qx.gates.I), ideal_unitary=qx.gates.I)
        channel = CrosstalkCycleChannel(cz | raw, zz_crosstalk(1, 2, 1e5), CZ_TIME)
        with pytest.raises(TypeError, match="needs a generator"):
            channel.exact_process()
        # The lowering itself only needs the ideal unitary, so it still works.
        assert _qubits(channel) == [(1, 2), (0, 1), (2,)]


# ══════════════════════════════════════════════════════════
# Accuracy of the lowering
# ══════════════════════════════════════════════════════════


class TestCommutingCrosstalk:
    """ZZ commutes with CZ, so its toggling-frame average is the term itself."""

    @pytest.mark.parametrize("rate", [1e4, 1e5, 1e6])
    def test_zz_during_cz_is_exact(self, rate):
        assert _error(_cz_channel(zz_crosstalk(1, 2, rate))) < 1e-10

    def test_zz_during_cz_stays_two_qubit(self):
        assert _qubits(_cz_channel(zz_crosstalk(1, 2, 1e5))) == [(1, 2), (0, 1), (2, 3)]

    def test_commuting_terms_use_a_single_sweep(self):
        crosstalk = zz_crosstalk(1, 2, 1e5) + zz_crosstalk(0, 3, 5e4) + zz_crosstalk(0, 1, 2e4)
        channel = _cz_channel(crosstalk)
        assert _qubits(channel) == [(1, 2), (0, 3), (0, 1), (0, 1), (2, 3)]
        assert _error(channel) < 1e-10

    def test_the_cz_frame_does_not_depend_on_round_off(self):
        """CZ's eigenvalue -1 sits on the logarithm's branch cut; its sign must not flip the frame."""

        def channel(imaginary):
            unitary = qx.Unitary.from_matrix(jnp.diag(jnp.array([1, 1, 1, complex(-1, imaginary)])), ((2, 2), (2, 2)))
            noise = qx.lindbladians.depolarizing(0.0, (2, 2))
            cz = Channel._from_noise_lindbladian(CZ(0, 1), unitary, noise, CZ_TIME)
            return CrosstalkCycleChannel(cz | _channel(I(2), CZ_TIME), xy_crosstalk(1, 2, 1e5), CZ_TIME)

        lowered = [_matrix(_lowered(channel(imaginary))) for imaginary in (1e-17, -1e-17, 0.0, -0.0)]
        for other in lowered[1:]:
            np.testing.assert_allclose(other, lowered[0], atol=1e-12)

    def test_error_with_decoherence_is_bilinear(self):
        """With dissipation the only error left is the ideal-frame one, O(epsilon * gamma * T)."""
        errors = {
            (rate, t1): _error(_cz_channel(zz_crosstalk(1, 2, rate), coherence_time=t1))
            for rate in (1e4, 1e5)
            for t1 in (30e-6, 300e-6)
        }
        epsilon_gamma = {key: 2 * np.pi * key[0] * CZ_TIME * CZ_TIME / key[1] for key in errors}
        ratios = [errors[key] / epsilon_gamma[key] for key in errors]
        assert max(ratios) < 1.0
        assert max(ratios) / min(ratios) < 1.5


class TestNonCommutingCrosstalk:
    def test_error_is_second_order_in_the_coupling(self):
        rates = np.array([2e4, 4e4, 8e4, 1.6e5])
        errors = [_error(_rx_channel(rate)) for rate in rates]
        assert 1.9 < _slope(rates, errors) < 2.1

    def test_error_falls_with_steps(self):
        errors = [_error(_rx_channel(1e5, steps)) for steps in (1, 2, 4)]
        assert errors[0] / errors[1] > 3.0
        assert errors[1] / errors[2] > 3.0

    def test_toggling_frame_beats_naive_splitting(self):
        """Applying the bare crosstalk before the gates is wrong at first order in the coupling."""
        rates = np.array([2e4, 4e4, 8e4, 1.6e5])
        naive_errors = []
        for rate in rates:
            channel = _rx_channel(rate)
            naive = _bare(channel)
            for hamiltonian, qubits in _pauli_terms(channel.crosstalk):
                unitary = qx.evolve(hamiltonian, RX_TIME)
                naive = naive @ qx.embed(qx.to_superop(unitary), (2, 2, 2), qubits)
            naive_errors.append(float(np.linalg.norm(_matrix(naive) - _matrix(channel.exact_process()), 2)))
        assert 0.9 < _slope(rates, naive_errors) < 1.1
        assert naive_errors[-2] > 20 * _error(_rx_channel(rates[-2]))

    def test_non_commuting_terms_are_strang_split(self):
        """Once toggled, ZZ(0, 1) and XY(1, 2) do not commute: half, full, half, then the gates."""
        assert _qubits(_rx_channel(1e5)) == [(0, 1), (1, 2), (0, 1), (0,), (1,), (2,)]

    def test_splitting_error_is_third_order(self):
        """Against exponentiating the slice's summed generator, Strang splitting adds O(epsilon^3)."""
        rates = np.array([4e5, 8e5, 1.6e6])
        extra = []
        for rate in rates:
            channel = _rx_channel(rate)
            (terms,) = channel._averaged_terms(1)
            summed = sum(qx.embed(h, (2, 2, 2), qubits).matrix for h, qubits in terms)
            fused = qx.Unitary.from_matrix(
                jnp.asarray(expm(-1j * RX_TIME * np.asarray(summed))), ((2, 2, 2), (2, 2, 2))
            )
            reference = _compose(channel, [(fused, (0, 1, 2)), *channel.operations()[-3:]])
            extra.append(float(np.linalg.norm(_matrix(_lowered(channel)) - _matrix(reference), 2)))
        assert 2.8 < _slope(rates, extra) < 3.2

    def test_xy_during_cz_converges(self):
        """XY does not commute with CZ: the averaged term spreads over both gates, and steps help."""
        assert sorted(_qubits(_cz_channel(xy_crosstalk(1, 2, 1e5)))[0]) == [0, 1, 2, 3]
        errors = [_error(_cz_channel(xy_crosstalk(1, 2, 1e5), steps)) for steps in (1, 2, 4, 8)]
        assert errors[0] / errors[1] > 2.5
        assert errors[2] / errors[3] > 3.5
        assert errors[3] < errors[0] / 30


class TestExperimentalRegime:
    """10-100 kHz couplings on 40 ns and 200 ns cycles with T1 = T2 = 30 us."""

    @pytest.mark.parametrize("rate", [1e4, 1e5])
    @pytest.mark.parametrize(
        ("build", "bound"),
        [
            (lambda rate: _rx_channel(rate, coherence_time=T1), 0.01),
            (lambda rate: _cz_channel(zz_crosstalk(1, 2, rate), coherence_time=T1), 0.01),
            (lambda rate: _cz_channel(xy_crosstalk(1, 2, rate), coherence_time=T1), 0.1),
        ],
        ids=["rx-zz-xy", "cz-zz", "cz-xy"],
    )
    def test_single_step_error_is_a_small_fraction_of_the_effect(self, build, bound, rate):
        channel = build(rate)
        assert _error(channel) < bound * _effect(channel)

    def test_mixed_five_qubit_line(self):
        cycle = _cycle((CZ(0, 1), CZ(2, 3), RX(np.pi / 2, 4)), CZ_TIME, T1)
        crosstalk = sum(
            (zz_crosstalk(a, b, 5e4) + xy_crosstalk(a, b, 2e4) for a, b in ((0, 1), (1, 2), (2, 3), (3, 4))),
            NO_CROSSTALK,
        )
        errors = [_error(CrosstalkCycleChannel(cycle, crosstalk, CZ_TIME, steps)) for steps in (1, 4)]
        effect = _effect(CrosstalkCycleChannel(cycle, crosstalk, CZ_TIME))
        assert errors[0] < 0.1 * effect
        assert errors[1] < 0.02 * effect

    def test_qutrit_constituent(self):
        """Crosstalk on a qutrit acts on its qubit subspace and leaves the leakage level alone."""
        cz = Channel.from_lindbladian(CZ(0, 1), qx.lindbladians.depolarizing(1e3, (3, 3)), CZ_TIME)
        channel = CrosstalkCycleChannel(cz | _channel(I(2), CZ_TIME), zz_crosstalk(1, 2, 1e5), CZ_TIME)
        unitary, qubits = channel.operations()[0]
        assert qubits == (1, 2)
        leaked = np.asarray(unitary.matrix).reshape(3, 2, 3, 2)[2, :, 2, :]
        np.testing.assert_allclose(leaked, np.eye(2), atol=1e-12)
        assert _error(channel) < 1e-3 * _effect(channel)


# ══════════════════════════════════════════════════════════
# Channel behaviour
# ══════════════════════════════════════════════════════════


class TestProcessFidelity:
    @pytest.mark.parametrize("rate", [1e5, 1e6])
    def test_does_not_depend_on_steps(self, rate):
        """Coherent errors add in amplitude: slicing the cycle must not dilute the estimate."""
        fidelities = [_cz_channel(zz_crosstalk(1, 2, rate), steps).process_fidelity for steps in (1, 2, 4, 8)]
        np.testing.assert_allclose(fidelities, fidelities[0], rtol=1e-12)

    @pytest.mark.parametrize(
        ("build", "ideal"),
        [
            (lambda: _cz_channel(zz_crosstalk(1, 2, 1e5)), qx.gates.CZ | qx.gates.CZ),
            (lambda: _cz_channel(xy_crosstalk(1, 2, 1e5)), qx.gates.CZ | qx.gates.CZ),
            (lambda: _rx_channel(1e5), qx.gates.RX(np.pi / 2) | qx.gates.RX(np.pi / 2) | qx.gates.I),
            (lambda: _rx_channel(4e5, steps=4), qx.gates.RX(np.pi / 2) | qx.gates.RX(np.pi / 2) | qx.gates.I),
        ],
        ids=["cz-zz", "cz-xy", "rx-strang", "rx-strang-4-steps"],
    )
    def test_matches_the_exact_crosstalk_infidelity(self, build, ideal):
        channel = build()
        exact = _crosstalk_infidelity(channel, ideal)
        assert channel.process_infidelity == pytest.approx(exact, rel=0.05)

    def test_combines_with_the_gate_noise(self):
        channel = _cz_channel(zz_crosstalk(1, 2, 1e5), coherence_time=T1)
        exact = float(qx.process_fidelity(channel.exact_process(), qx.gates.CZ | qx.gates.CZ))
        assert channel.process_fidelity < channel.cycle.process_fidelity
        # The product of the two fidelities is correct to first order in each infidelity.
        assert abs(channel.process_fidelity - exact) < 1e-4
        assert channel.process_infidelity == pytest.approx(1 - channel.process_fidelity)
        assert channel.average_gate_fidelity > channel.process_fidelity
        assert channel.average_gate_infidelity == pytest.approx(1 - channel.average_gate_fidelity)


class TestChannel:
    def test_delegates_to_cycle(self):
        channel = _cz_channel(zz_crosstalk(1, 2, 1e5))
        assert channel.inst == channel.cycle.inst
        assert channel.defcircuit == channel.cycle.defcircuit
        assert channel.channels == channel.cycle.channels
        assert channel.qubits == [0, 1, 2, 3]

    def test_no_crosstalk_reduces_to_the_cycle(self):
        channel = _cz_channel(NO_CROSSTALK, coherence_time=T1)
        assert len(channel.operations()) == 2
        assert channel.process_fidelity == channel.cycle.process_fidelity
        assert _error(channel) < 1e-10

    def test_json_round_trip(self):
        channel = _rx_channel(1e5, steps=2, coherence_time=T1)
        restored = CrosstalkCycleChannel.from_json(channel.to_json())
        assert restored == channel
        np.testing.assert_allclose(_matrix(_lowered(restored)), _matrix(_lowered(channel)), atol=1e-12)
        empty = _cz_channel(NO_CROSSTALK)
        assert CrosstalkCycleChannel.from_json(empty.to_json()) == empty

    def test_equality(self):
        channel = _cz_channel(zz_crosstalk(1, 2, 1e5))
        assert channel == _cz_channel(zz_crosstalk(1, 2, 1e5))
        assert channel != _cz_channel(zz_crosstalk(1, 2, 2e5))
        assert channel != _cz_channel(zz_crosstalk(1, 2, 1e5), steps=2)
        assert channel != channel.cycle

    def test_measurement_constituents_get_an_identity_frame(self):
        gate = _channel(RX(np.pi / 2, 0), RX_TIME)
        measurement = MeasurementChannel.from_readout_fidelity(MEASURE(1, None), fidelity=0.98)
        channel = CrosstalkCycleChannel(gate | measurement, zz_crosstalk(0, 1, 1e5), RX_TIME)
        operations = channel.operations()
        assert [qubits for _, qubits in operations] == [(0, 1), (0,), (1,)]
        assert isinstance(operations[0][0], qx.Unitary)
        assert isinstance(operations[-1][0], qx.QuantumInstrument)


# ══════════════════════════════════════════════════════════
# Noise model and simulators
# ══════════════════════════════════════════════════════════


def _program(channel):
    return Program(channel.defcircuit, channel.inst)


def _evolved(superop, dims):
    """The superoperator applied to the all-zero state."""
    dimension = int(np.prod(dims))
    matrix = np.zeros((dimension, dimension), dtype=complex)
    matrix[0, 0] = 1.0
    rho = qx.DensityMatrix.from_matrix(jax.numpy.asarray(matrix), tuple(dims))
    return np.asarray(qx.apply_superop_to_density_matrix(superop, rho).matrix)


class TestNoiseModel:
    def test_lookup_and_json(self):
        channel = _rx_channel(1e5, coherence_time=T1)
        model = NoiseModel.from_channels([channel])
        assert model.get_channel(channel.inst) is channel
        restored = NoiseModel.from_json(model.to_json())
        assert restored.get_channel(channel.inst) == channel

    def test_estimate_program_fidelity(self):
        channel = _rx_channel(1e5, coherence_time=T1)
        model = NoiseModel.from_channels([channel])
        program = _program(channel) + channel.inst
        assert estimate_program_fidelity(program, model) == pytest.approx(channel.process_fidelity**2)


class TestResolver:
    def test_crosstalk_unitaries_precede_constituents(self):
        channel = _rx_channel(1e5, coherence_time=T1)
        ops, qubit_tuples, _ = expand_program(_program(channel), NoiseModel.from_channels([channel]))
        assert qubit_tuples == ((0, 1), (1, 2), (0, 1), (0,), (1,), (2,))
        assert [type(op) for op in ops[:3]] == [qx.Unitary] * 3
        assert all(isinstance(op, qx.SuperOp) for op in ops[3:])

    def test_steps_repeat_the_crosstalk_layer(self):
        channel = _rx_channel(1e5, steps=3, coherence_time=T1)
        _, qubit_tuples, _ = expand_program(_program(channel), NoiseModel.from_channels([channel]))
        assert len(qubit_tuples) == 3 * 3 + 3

    def test_rejected_for_a_plain_gate(self):
        channel = _rx_channel(1e5)
        gate = Gate("RX", [np.pi / 2], [Qubit(0)])

        class Model:
            def get_channel(self, inst):
                return channel if inst == gate else None

        with pytest.raises(ValueError, match="CrosstalkCycleChannel for RX was not expanded"):
            expand_program(Program(gate), Model())


class TestSimulators:
    @pytest.mark.parametrize("max_subsystem_size", [0, 1, 2, 3])
    def test_density_matrix_applies_the_lowered_process(self, max_subsystem_size):
        channel = _rx_channel(1e5, steps=2, coherence_time=T1)
        sim = DensityMatrixSimulator(
            _program(channel), noise_model=NoiseModel.from_channels([channel]), max_subsystem_size=max_subsystem_size
        )
        rho = np.asarray(sim.compute().matrix)
        np.testing.assert_allclose(rho, _evolved(_lowered(channel), sim.dims), atol=1e-10)

    def test_density_matrix_close_to_exact_evolution(self):
        channel = _cz_channel(xy_crosstalk(1, 2, 1e5), steps=4, coherence_time=T1)
        # Start from a state the exchange acts on, prepared by gates of the program rather than
        # of the noise model.
        program = Program(RX(np.pi / 2, 0), RX(np.pi, 1), RX(np.pi / 2, 3)) + _program(channel)
        sim = DensityMatrixSimulator(program, noise_model=NoiseModel.from_channels([channel]))
        rho = np.asarray(sim.compute().matrix)

        ideal = Program(RX(np.pi / 2, 0), RX(np.pi, 1), RX(np.pi / 2, 3), I(2))
        prepared = DensityMatrixSimulator(ideal).compute()
        exact = np.asarray(qx.apply_superop_to_density_matrix(channel.exact_process(), prepared).matrix)
        without = np.asarray(qx.apply_superop_to_density_matrix(_bare(channel), prepared).matrix)

        assert np.linalg.norm(rho - exact) < 0.02 * np.linalg.norm(without - exact)

    def test_trajectories_average_to_the_density_matrix(self):
        channel = _rx_channel(1e6, coherence_time=1e-6)
        model = NoiseModel.from_channels([channel])
        program = _program(channel)
        rho = np.asarray(DensityMatrixSimulator(program, noise_model=model).compute().matrix)

        sim = TrajectorySimulator(program, noise_model=model)
        states, _ = sim.compute(None, jax.random.split(jax.random.key(7), 4096))
        vectors = np.asarray(states.matrix).reshape(4096, -1)
        estimate = np.einsum("ti,tj->ij", vectors, vectors.conj()) / 4096
        assert np.linalg.norm(estimate - rho) < 0.05

    def test_crosstalk_unitaries_take_the_unitary_branch(self):
        """A crosstalk unitary is applied directly: one operator, no sampling, its own branch."""
        channel = _cz_channel(zz_crosstalk(0, 1, 1e5), coherence_time=T1)
        sim = TrajectorySimulator(
            _program(channel), noise_model=NoiseModel.from_channels([channel]), max_subsystem_size=0
        )
        layout = sim._layout
        branches = list(zip(layout.bases, layout.base_max_k, layout.base_unitary, strict=True))
        # The crosstalk sits on the qubits of a noisy CZ, yet does not inherit its sixteen operators.
        assert ((0, 1), 1, True) in branches
        assert ((0, 1), 16, False) in branches
