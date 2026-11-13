"""Tests for crosstalk cycle channels: the representation and the accuracy of its lowering.

Accuracy is measured against :meth:`CrosstalkCycleChannel.exact_process`, the dense evolution of the
joint generator, as the spectral norm of the difference of the two superoperators.
"""

import jax
import numpy as np
import pytest
import quax as qx
from scipy.linalg import expm

from pyquil.gates import CZ, MEASURE, RX, I
from pyquil.noise._channels import Channel, CycleChannel, MeasurementChannel, SuperopChannel
from pyquil.noise._crosstalk import (
    Coupling,
    CrosstalkCycleChannel,
    _fuse_terms,
    _prune_support,
    _slice_average,
    _Term,
)
from pyquil.noise._noise_model import NoiseModel, estimate_program_fidelity
from pyquil.quil import Program
from pyquil.quilatom import Qubit
from pyquil.quilbase import Gate
from pyquil.simulation._resolver import expand_program
from pyquil.simulation._simulator import DensityMatrixSimulator, TrajectorySimulator

CZ_TIME = 200e-9
RX_TIME = 40e-9
T1 = T2 = 30e-6


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


def _error(channel):
    """Spectral-norm distance between the lowered and the exact process."""
    return float(np.linalg.norm(_matrix(channel.lowered_process()) - _matrix(channel.exact_process()), 2))


def _effect(channel):
    """Size of the crosstalk itself: the error made by ignoring the couplings altogether."""
    bare = CrosstalkCycleChannel(channel.cycle, (), channel.duration)
    return float(np.linalg.norm(_matrix(bare.lowered_process()) - _matrix(channel.exact_process()), 2))


def _slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


RX_GATES = (RX(np.pi / 2, 0), RX(np.pi / 2, 1), I(2))


def _rx_channel(rate_hz, steps=1, coherence_time=None, max_fused_qubits=4):
    cycle = _cycle(RX_GATES, RX_TIME, coherence_time)
    couplings = (Coupling.zz(0, 1, rate_hz), Coupling.xy(1, 2, rate_hz))
    return CrosstalkCycleChannel(cycle, couplings, RX_TIME, steps, max_fused_qubits)


def _cz_channel(coupling, steps=1, coherence_time=None):
    return CrosstalkCycleChannel(_cycle((CZ(0, 1), CZ(2, 3)), CZ_TIME, coherence_time), (coupling,), CZ_TIME, steps)


# ══════════════════════════════════════════════════════════
# Building blocks
# ══════════════════════════════════════════════════════════


class TestCoupling:
    def test_zz_shifts_only_the_doubly_excited_state(self):
        coupling = Coupling.zz(3, 5, 1e5)
        np.testing.assert_allclose(coupling.hamiltonian.matrix, np.diag([0, 0, 0, 2 * np.pi * 1e5]))
        assert coupling.qubits == (3, 5)

    def test_xy_exchanges_single_excitations(self):
        matrix = np.asarray(Coupling.xy(0, 1, 1e5).hamiltonian.matrix)
        x, y = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])
        np.testing.assert_allclose(matrix, np.pi * 1e5 * (np.kron(x, x) + np.kron(y, y)), atol=1e-9)

    def test_qutrit_couplings(self):
        zz = np.asarray(Coupling.zz(0, 1, 1.0, dims=(3, 2)).hamiltonian.matrix)
        np.testing.assert_allclose(np.diag(zz) / (2 * np.pi), [0, 0, 0, 1, 0, 2])
        xy = np.asarray(Coupling.xy(0, 1, 1.0, dims=(3, 3)).hamiltonian.matrix)
        assert xy[1, 3] == xy[3, 1] == 2 * np.pi  # |01><10| + h.c. with three levels each
        assert np.count_nonzero(xy) == 2

    def test_rejects_non_hermitian_and_mismatched(self):
        with pytest.raises(ValueError, match="Hermitian"):
            Coupling.from_matrix((0, 1), np.triu(np.ones((4, 4)), 1))
        with pytest.raises(ValueError, match="distinct"):
            Coupling.zz(1, 1, 1.0)
        with pytest.raises(ValueError, match="needs a Hamiltonian on 3"):
            Coupling((0, 1, 2), Coupling.zz(0, 1, 1.0).hamiltonian)

    def test_json_round_trip(self):
        coupling = Coupling.xy(4, 2, 3e4)
        assert Coupling.from_json(coupling.to_json()) == coupling


class TestHelpers:
    def test_slice_average_matches_quadrature(self):
        rng = np.random.default_rng(0)
        frame = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        frame = frame + frame.conj().T
        coupling = np.diag([0.0, 0.0, 0.0, 1.0])
        start, stop = 0.3, 1.1
        times = np.linspace(start, stop, 4001)
        toggled = np.array([expm(1j * frame * s) @ coupling @ expm(-1j * frame * s) for s in times])
        expected = np.trapezoid(toggled, times, axis=0) / (stop - start)
        np.testing.assert_allclose(_slice_average(coupling, frame, start, stop), expected, atol=1e-6)

    def test_prune_support_drops_identity_qudits(self):
        zz = np.diag([1.0, -1.0, -1.0, 1.0])
        matrix = np.kron(np.eye(2), np.kron(zz, np.eye(3)))  # acts on the middle two of four qudits
        reduced, qubits, dims = _prune_support(matrix, (7, 8, 9, 10), (2, 2, 2, 3))
        assert qubits == (8, 9)
        assert dims == (2, 2)
        np.testing.assert_allclose(reduced, zz)

    def test_prune_support_of_identity_is_scalar(self):
        reduced, qubits, dims = _prune_support(2.5 * np.eye(4), (0, 1), (2, 2))
        assert qubits == () and dims == ()
        np.testing.assert_allclose(reduced, [[2.5]])


class TestValidation:
    def test_coupling_outside_cycle(self):
        with pytest.raises(ValueError, match="outside the cycle"):
            CrosstalkCycleChannel(_cycle((CZ(0, 1), CZ(2, 3)), CZ_TIME), (Coupling.zz(1, 9, 1e5),), CZ_TIME)

    def test_coupling_dims_must_match(self):
        with pytest.raises(ValueError, match="has dims"):
            CrosstalkCycleChannel(
                _cycle((CZ(0, 1), CZ(2, 3)), CZ_TIME), (Coupling.zz(1, 2, 1e5, dims=(3, 2)),), CZ_TIME
            )

    def test_duration_and_steps(self):
        cycle = _cycle((CZ(0, 1), CZ(2, 3)), CZ_TIME)
        with pytest.raises(ValueError, match="duration"):
            CrosstalkCycleChannel(cycle, (), 0.0)
        with pytest.raises(ValueError, match="steps"):
            CrosstalkCycleChannel(cycle, (), CZ_TIME, steps=0)
        with pytest.raises(ValueError, match="max_fused_qubits"):
            CrosstalkCycleChannel(cycle, (), CZ_TIME, max_fused_qubits=0)

    def test_dense_processes_are_size_limited(self):
        channel = _cz_channel(Coupling.zz(1, 2, 1e5))
        with pytest.raises(ValueError, match="max_dimension"):
            channel.exact_process(max_dimension=8)
        with pytest.raises(ValueError, match="max_dimension"):
            channel.lowered_process(max_dimension=8)

    def test_exact_process_needs_generators(self):
        cz = _channel(CZ(0, 1), CZ_TIME)
        raw = SuperopChannel(inst=I(2), process=qx.to_superop(qx.gates.I), ideal_unitary=qx.gates.I)
        channel = CrosstalkCycleChannel(cz | raw, (Coupling.zz(1, 2, 1e5),), CZ_TIME)
        with pytest.raises(TypeError, match="needs a generator"):
            channel.exact_process()
        # The lowering itself only needs the ideal unitary, so it still works.
        assert [qubits for _, qubits in channel.operations()] == [(1, 2), (0, 1), (2,)]


# ══════════════════════════════════════════════════════════
# Accuracy of the lowering
# ══════════════════════════════════════════════════════════


class TestCommutingCrosstalk:
    """ZZ commutes with CZ, so its toggling-frame average is the coupling itself."""

    @pytest.mark.parametrize("rate_hz", [1e4, 1e5, 1e6])
    def test_zz_during_cz_is_exact(self, rate_hz):
        channel = _cz_channel(Coupling.zz(1, 2, rate_hz))
        assert _error(channel) < 1e-10

    def test_zz_during_cz_stays_two_qubit(self):
        channel = _cz_channel(Coupling.zz(1, 2, 1e5))
        assert [qubits for _, qubits in channel.operations()] == [(1, 2), (0, 1), (2, 3)]

    def test_commuting_terms_use_a_single_sweep(self):
        cycle = _cycle((CZ(0, 1), CZ(2, 3)), CZ_TIME)
        couplings = (Coupling.zz(1, 2, 1e5), Coupling.zz(0, 3, 5e4), Coupling.zz(0, 1, 2e4))
        channel = CrosstalkCycleChannel(cycle, couplings, CZ_TIME, max_fused_qubits=2)
        assert len(channel.operations()) == len(couplings) + 2
        assert _error(channel) < 1e-10

    def test_commuting_terms_fuse_into_one_unitary(self):
        cycle = _cycle((CZ(0, 1), CZ(2, 3)), CZ_TIME)
        couplings = (Coupling.zz(1, 2, 1e5), Coupling.zz(0, 3, 5e4), Coupling.zz(0, 1, 2e4))
        channel = CrosstalkCycleChannel(cycle, couplings, CZ_TIME)
        assert [sorted(qubits) for _, qubits in channel.operations()] == [[0, 1, 2, 3], [0, 1], [2, 3]]
        assert _error(channel) < 1e-10

    def test_error_with_decoherence_is_bilinear(self):
        """With dissipation the only error left is the ideal-frame one, O(epsilon * gamma * T)."""
        errors = {
            (rate, t1): _error(_cz_channel(Coupling.zz(1, 2, rate), coherence_time=t1))
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
        """Applying the bare coupling before the gates is wrong at first order in the coupling."""
        rates = np.array([2e4, 4e4, 8e4, 1.6e5])
        naive_errors = []
        for rate in rates:
            channel = _rx_channel(rate)
            dims = (2, 2, 2)
            naive = CrosstalkCycleChannel(channel.cycle, (), RX_TIME).lowered_process()
            for coupling in channel.couplings:
                unitary = qx.Unitary.from_matrix(
                    jax.numpy.asarray(expm(-1j * RX_TIME * np.asarray(coupling.hamiltonian.matrix))),
                    ((2, 2), (2, 2)),
                )
                naive = naive @ qx.embed(qx.to_superop(unitary), dims, coupling.qubits)
            naive_errors.append(float(np.linalg.norm(_matrix(naive) - _matrix(channel.exact_process()), 2)))
        assert 0.9 < _slope(rates, naive_errors) < 1.1
        assert naive_errors[-2] > 20 * _error(_rx_channel(rates[-2]))

    def test_xy_during_cz_converges(self):
        """XY does not commute with CZ: the averaged term spreads over both gates, and steps help."""
        channel = _cz_channel(Coupling.xy(1, 2, 1e5))
        assert sorted(channel.operations()[0][1]) == [0, 1, 2, 3]
        errors = [_error(_cz_channel(Coupling.xy(1, 2, 1e5), steps)) for steps in (1, 2, 4, 8)]
        assert errors[0] / errors[1] > 2.5
        assert errors[2] / errors[3] > 3.5
        assert errors[3] < errors[0] / 30


class TestFusion:
    def test_lattice_couplers_fuse_into_stars(self):
        """Five couplers around one qubit and a detached pair: one star per qubit budget."""
        zz = np.diag([0.0, 0.0, 0.0, 1.0])
        terms = [_Term(zz, (0, q), (2, 2)) for q in (1, 2, 3, 4)] + [_Term(zz, (5, 6), (2, 2))]
        fused = _fuse_terms(terms, 4)
        assert [term.qubits for term in fused] == [(0, 1, 2, 3), (0, 4), (5, 6)]
        assert [term.qubits for term in _fuse_terms(terms, 5)] == [(0, 1, 2, 3, 4), (5, 6)]
        assert len(_fuse_terms(terms, 2)) == len(terms)

    def test_fused_matrix_is_the_sum_of_its_terms(self):
        zz = np.diag([0.0, 0.0, 0.0, 1.0])
        (fused,) = _fuse_terms([_Term(zz, (0, 1), (2, 2)), _Term(2 * zz, (1, 2), (2, 2))], 3)
        expected = np.kron(zz, np.eye(2)) + 2 * np.kron(np.eye(2), zz)
        np.testing.assert_allclose(fused.matrix, expected)

    def test_wide_terms_are_kept(self):
        wide = _Term(np.eye(8), (0, 1, 2), (2, 2, 2))
        (kept,) = _fuse_terms([wide], 2)
        assert kept.qubits == wide.qubits
        np.testing.assert_allclose(kept.matrix, wide.matrix)

    def test_unfused_terms_are_trotterized_at_third_order(self):
        """With fusion off the two couplings are Strang-split, adding an O(epsilon^3) error."""
        rates = np.array([4e5, 8e5, 1.6e6])
        extra = [
            float(
                np.linalg.norm(
                    _matrix(_rx_channel(rate, max_fused_qubits=2).lowered_process())
                    - _matrix(_rx_channel(rate).lowered_process()),
                    2,
                )
            )
            for rate in rates
        ]
        assert 2.8 < _slope(rates, extra) < 3.2
        split = _rx_channel(1e5, max_fused_qubits=2)
        assert [qubits for _, qubits in split.operations()][:3] == [(0, 1), (1, 2), (0, 1)]
        assert _error(split) < 0.01 * _effect(split)


class TestExperimentalRegime:
    """10-100 kHz couplings on 40 ns and 200 ns cycles with T1 = T2 = 30 us."""

    @pytest.mark.parametrize("rate_hz", [1e4, 1e5])
    @pytest.mark.parametrize(
        ("build", "bound"),
        [
            (lambda rate: _rx_channel(rate, coherence_time=T1), 0.01),
            (lambda rate: _cz_channel(Coupling.zz(1, 2, rate), coherence_time=T1), 0.01),
            (lambda rate: _cz_channel(Coupling.xy(1, 2, rate), coherence_time=T1), 0.1),
        ],
        ids=["rx-zz-xy", "cz-zz", "cz-xy"],
    )
    def test_single_step_error_is_a_small_fraction_of_the_effect(self, build, bound, rate_hz):
        channel = build(rate_hz)
        assert _error(channel) < bound * _effect(channel)

    def test_process_fidelity_matches_exact(self):
        for channel in (_rx_channel(1e5, coherence_time=T1), _cz_channel(Coupling.xy(1, 2, 1e5), coherence_time=T1)):
            infidelity = 1 - float(qx.process_fidelity(channel.lowered_process(), channel.exact_process()))
            assert infidelity < 1e-4

    def test_mixed_five_qubit_line(self):
        cycle = _cycle((CZ(0, 1), CZ(2, 3), RX(np.pi / 2, 4)), CZ_TIME, T1)
        couplings = tuple(
            coupling
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 4))
            for coupling in (Coupling.zz(a, b, 5e4), Coupling.xy(a, b, 2e4))
        )
        errors = [_error(CrosstalkCycleChannel(cycle, couplings, CZ_TIME, steps)) for steps in (1, 4)]
        effect = _effect(CrosstalkCycleChannel(cycle, couplings, CZ_TIME))
        assert errors[0] < 0.1 * effect
        assert errors[1] < 0.02 * effect

    def test_qutrit_constituent(self):
        dims = (3, 3)
        gate = CZ(0, 1)
        noise = qx.lindbladians.depolarizing(1e3, dims)
        cz = Channel.from_lindbladian(gate, noise, CZ_TIME)
        idle = _channel(I(2), CZ_TIME)
        channel = CrosstalkCycleChannel(cz | idle, (Coupling.zz(1, 2, 1e5, dims=(3, 2)),), CZ_TIME)
        assert channel.operations()[0][1] == (1, 2)
        assert _error(channel) < 1e-3 * _effect(channel)


# ══════════════════════════════════════════════════════════
# Channel behaviour
# ══════════════════════════════════════════════════════════


class TestChannel:
    def test_delegates_to_cycle(self):
        channel = _cz_channel(Coupling.zz(1, 2, 1e5))
        assert channel.inst == channel.cycle.inst
        assert channel.defcircuit == channel.cycle.defcircuit
        assert channel.channels == channel.cycle.channels
        assert channel.qubits == [0, 1, 2, 3]

    def test_no_couplings_reduces_to_the_cycle(self):
        channel = CrosstalkCycleChannel(_cycle((CZ(0, 1), CZ(2, 3)), CZ_TIME, T1), (), CZ_TIME)
        assert len(channel.operations()) == 2
        assert channel.process_fidelity == channel.cycle.process_fidelity
        assert _error(channel) < 1e-10

    def test_process_fidelity_estimate(self):
        channel = _cz_channel(Coupling.zz(1, 2, 1e5), coherence_time=T1)
        exact = float(qx.process_fidelity(channel.exact_process(), qx.gates.CZ | qx.gates.CZ))
        assert channel.process_fidelity < channel.cycle.process_fidelity
        assert abs(channel.process_fidelity - exact) < 1e-3
        assert channel.process_infidelity == pytest.approx(1 - channel.process_fidelity)
        assert channel.average_gate_fidelity > channel.process_fidelity
        assert channel.average_gate_infidelity == pytest.approx(1 - channel.average_gate_fidelity)

    def test_json_round_trip(self):
        channel = _rx_channel(1e5, steps=2, coherence_time=T1)
        restored = CrosstalkCycleChannel.from_json(channel.to_json())
        assert restored == channel
        np.testing.assert_allclose(_matrix(restored.lowered_process()), _matrix(channel.lowered_process()), atol=1e-12)

    def test_equality(self):
        channel = _cz_channel(Coupling.zz(1, 2, 1e5))
        assert channel == _cz_channel(Coupling.zz(1, 2, 1e5))
        assert channel != _cz_channel(Coupling.zz(1, 2, 2e5))
        assert channel != _cz_channel(Coupling.zz(1, 2, 1e5), steps=2)
        assert channel != CrosstalkCycleChannel(channel.cycle, channel.couplings, CZ_TIME, max_fused_qubits=2)
        assert channel != channel.cycle

    def test_measurement_constituents_get_an_identity_frame(self):
        gate = _channel(RX(np.pi / 2, 0), RX_TIME)
        measurement = MeasurementChannel.from_readout_fidelity(MEASURE(1, None), fidelity=0.98)
        channel = CrosstalkCycleChannel(gate | measurement, (Coupling.zz(0, 1, 1e5),), RX_TIME)
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
        # ZZ(0,1) and XY(1,2) are fused into one unitary, which precedes the three gates.
        assert qubit_tuples == ((0, 1, 2), (0,), (1,), (2,))
        assert isinstance(ops[0], qx.Unitary)
        assert all(isinstance(op, qx.SuperOp) for op in ops[1:])

    def test_unfused_couplings_are_strang_split(self):
        channel = _rx_channel(1e5, coherence_time=T1, max_fused_qubits=2)
        ops, qubit_tuples, _ = expand_program(_program(channel), NoiseModel.from_channels([channel]))
        # Once toggled the two couplings do not commute: half, full, half, then the three gates.
        assert qubit_tuples == ((0, 1), (1, 2), (0, 1), (0,), (1,), (2,))
        assert [type(op) for op in ops[:3]] == [qx.Unitary] * 3

    def test_steps_repeat_the_crosstalk_layer(self):
        channel = _rx_channel(1e5, steps=3, coherence_time=T1)
        _, qubit_tuples, _ = expand_program(_program(channel), NoiseModel.from_channels([channel]))
        assert len(qubit_tuples) == 3 + 3

    def test_rejected_for_a_plain_gate(self):
        channel = _rx_channel(1e5)
        gate = Gate("RX", [np.pi / 2], [Qubit(0)])

        class Model:
            def get_channel(self, inst):
                return channel if inst == gate else None

        with pytest.raises(ValueError, match="CrosstalkCycleChannel for RX was not expanded"):
            expand_program(Program(gate), Model())


class TestSimulators:
    @pytest.mark.parametrize("max_subsystem_size", [0, 1, 2])
    def test_density_matrix_applies_the_lowered_process(self, max_subsystem_size):
        channel = _rx_channel(1e5, steps=2, coherence_time=T1)
        sim = DensityMatrixSimulator(
            _program(channel), noise_model=NoiseModel.from_channels([channel]), max_subsystem_size=max_subsystem_size
        )
        rho = np.asarray(sim.compute().matrix)
        np.testing.assert_allclose(rho, _evolved(channel.lowered_process(), sim.dims), atol=1e-10)

    def test_density_matrix_close_to_exact_evolution(self):
        channel = _cz_channel(Coupling.xy(1, 2, 1e5), steps=4, coherence_time=T1)
        # Start from a state the exchange acts on: |+>|1>|0>|+> would need state prep, so the
        # cycle is preceded by RX gates that are part of the program, not of the noise model.
        program = Program(RX(np.pi / 2, 0), RX(np.pi, 1), RX(np.pi / 2, 3)) + _program(channel)
        sim = DensityMatrixSimulator(program, noise_model=NoiseModel.from_channels([channel]))
        rho = np.asarray(sim.compute().matrix)

        ideal = Program(RX(np.pi / 2, 0), RX(np.pi, 1), RX(np.pi / 2, 3), I(2))
        prepared = DensityMatrixSimulator(ideal).compute()
        exact = np.asarray(qx.apply_superop_to_density_matrix(channel.exact_process(), prepared).matrix)
        without = np.asarray(
            qx.apply_superop_to_density_matrix(
                CrosstalkCycleChannel(channel.cycle, (), CZ_TIME).lowered_process(), prepared
            ).matrix
        )

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
        channel = _cz_channel(Coupling.zz(0, 1, 1e5), coherence_time=T1)
        sim = TrajectorySimulator(
            _program(channel), noise_model=NoiseModel.from_channels([channel]), max_subsystem_size=0
        )
        layout = sim._layout
        branches = list(zip(layout.bases, layout.base_max_k, layout.base_unitary, strict=True))
        # The coupling sits on the qubits of a noisy CZ, yet does not inherit its sixteen operators.
        assert ((0, 1), 1, True) in branches
        assert ((0, 1), 16, False) in branches
