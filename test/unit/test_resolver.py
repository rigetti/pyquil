"""Unit tests for the resolver pipeline."""

import jax.numpy as jnp
import numpy as np
import quax as qx

from pyquil.gates import CNOT, MEASURE, RESET, RX, RZ, H, X
from pyquil.noise._channels import (
    Channel,
    CycleChannel,
    MeasurementChannel,
    ResetChannel,
    SuperopChannel,
    SuperopResetChannel,
)
from pyquil.noise._noise_model import NoiseModel
from pyquil.quil import Program
from pyquil.quilatom import FormalArgument, MemoryReference, Qubit
from pyquil.quilbase import (
    Declare,
    DefCircuit,
    Gate,
    Measurement,
    ResetQubit,
)
from pyquil.simulation._resolver import (
    build_dag,
    expand_program,
    remap_qubits,
    resolve_program,
)

_EMPTY_PARAMS = jnp.array([], dtype=float)


# ──────────────────────────────────────────────────────────
# expand_program
# ──────────────────────────────────────────────────────────


class TestExpandProgram:
    def test_simple_gates(self):
        p = Program(H(0), X(1), CNOT(0, 1))
        ops, qubit_tuples, _ = expand_program(p)
        assert len(ops) == 3
        assert len(qubit_tuples) == 3
        # Physical qubit IDs
        assert qubit_tuples[0] == (0,)
        assert qubit_tuples[1] == (1,)
        assert qubit_tuples[2] == (0, 1)

    def test_fixed_gates_are_concrete(self):
        p = Program(H(0), X(1))
        ops, _, _ = expand_program(p)
        for op in ops:
            assert isinstance(op, qx.Unitary)

    def test_measurement_emitted(self):
        p = Program(Declare("ro", "BIT", 1), H(0), MEASURE(0, MemoryReference("ro", 0)))
        ops, qubit_tuples, _ = expand_program(p)
        assert len(ops) == 2
        # Measurement should be a concrete QuantumInstrument
        assert isinstance(ops[1], qx.QuantumInstrument)

    def test_reset_emitted(self):
        p = Program(RESET(), H(0))
        ops, qubit_tuples, _ = expand_program(p)
        # Bare RESET expands to one reset per qubit (only qubit 0 in this program)
        assert len(ops) == 2
        assert isinstance(ops[0], qx.SuperOp)

    def test_noise_channel_resolved(self):
        p = Program(X(0))
        ch = Channel.from_gate_fidelity(inst=X(0), fidelity=0.99)
        nm = NoiseModel.from_channels([ch])
        ops, _, _ = expand_program(p, nm)
        assert isinstance(ops[0], qx.SuperOp)

    def test_defcircuit_expansion_no_cycle_channel(self):
        q0, q1 = FormalArgument("q0"), FormalArgument("q1")
        dc = DefCircuit("MY_CYCLE", [], [q0, q1], [H(q0), CNOT(q0, q1)])
        p = Program(dc, Gate("MY_CYCLE", [], [Qubit(0), Qubit(1)]))
        ops, qubit_tuples, _ = expand_program(p)
        assert len(ops) == 2

    def test_cycle_channel_expansion(self):
        """CycleChannel constituents are emitted instead of DEFCIRCUIT body."""
        q0 = FormalArgument("q")
        dc = DefCircuit("SQC", [], [q0], [RX(0.1, q0), RZ(0.2, q0)])
        cycle_inst = Gate("SQC", [], [Qubit(0)])
        channels = tuple(
            Channel.from_depolarizing_constant(inst, depolarizing_constant=0.99) for inst in (RX(0.1, 0), RZ(0.2, 0))
        )
        nm = NoiseModel.from_channels([CycleChannel(inst=cycle_inst, defcircuit=dc, channels=channels)])
        p = Program(dc, cycle_inst)
        ops, qubit_tuples, _ = expand_program(p, nm)
        assert len(ops) == 2
        # All should be concrete noisy SuperOps
        for op in ops:
            assert isinstance(op, qx.SuperOp)

    def test_cycle_channel_applies_process_faithfully(self):
        """A ``CycleChannel`` constituent is simulated as its ``process`` verbatim.

        A channel's ``process`` is the source of truth for a cycle constituent: it must
        carry the gate (composed with any noise).  This pins that contract — a
        noiseless gate channel's ``process`` is emitted unchanged and equals the gate
        superoperator — so a noise model that puts an identity in ``process`` for a
        real gate (which silently drops it) is a *noise-model* error, caught at the
        point of construction, not something the resolver second-guesses.
        """
        q0 = FormalArgument("q")
        dc = DefCircuit("VCYC", [], [q0], [X(q0)])
        cycle_inst = Gate("VCYC", [], [Qubit(0)])
        # Noiseless X channel: the gate lives in the process (ideal_unitary == process action).
        gate_channel = SuperopChannel(
            inst=X(0), process=qx.to_superop(qx.gates.X), ideal_unitary=qx.gates.X
        )
        nm = NoiseModel.from_channels([CycleChannel(inst=cycle_inst, defcircuit=dc, channels=(gate_channel,))])
        p = Program(dc, cycle_inst)
        ops, _, _ = expand_program(p, nm)
        assert len(ops) == 1
        # Emitted verbatim: the process (an X superoperator) is what gets applied.
        assert jnp.allclose(qx.to_superop(ops[0]).matrix, qx.to_superop(qx.gates.X).matrix, atol=1e-6)

    def test_parameterized_gate_produces_callable(self):
        p = Program(Declare("theta", "REAL", 1), RZ(MemoryReference("theta", 0), 0))
        ops, _, _ = expand_program(p)
        assert len(ops) == 1
        assert callable(ops[0])
        result = ops[0](jnp.array([1.23]))
        assert isinstance(result, qx.Unitary)


# ──────────────────────────────────────────────────────────
# remap_qubits & build_dag
# ──────────────────────────────────────────────────────────


class TestRemapAndDag:
    def test_remap_qubits(self):
        qubit_tuples = [(3,), (5,), (3, 5)]
        qubit_indices = {3: 0, 5: 1}
        result = remap_qubits(qubit_tuples, qubit_indices)
        assert result == [(0,), (1,), (0, 1)]

    def test_build_dag_single_qubit_chain(self):
        qubit_tuples = [(0,), (0,), (0,)]
        dag = build_dag(qubit_tuples)
        assert dag.has_edge(0, 1)
        assert dag.has_edge(1, 2)
        assert not dag.has_edge(0, 2)

    def test_build_dag_independent_qubits(self):
        qubit_tuples = [(0,), (1,)]
        dag = build_dag(qubit_tuples)
        assert dag.number_of_edges() == 0

    def test_build_dag_multi_qubit(self):
        qubit_tuples = [(0,), (1,), (0, 1)]
        dag = build_dag(qubit_tuples)
        assert dag.has_edge(0, 2)
        assert dag.has_edge(1, 2)
        assert not dag.has_edge(0, 1)


# ──────────────────────────────────────────────────────────
# resolve_program (integration)
# ──────────────────────────────────────────────────────────


class TestResolveProgram:
    def test_basic_roundtrip(self):
        p = Program(H(0), CNOT(0, 1), X(1))
        res = resolve_program(p)
        ops = res.resolve(_EMPTY_PARAMS)
        assert len(ops) == 3
        assert all(isinstance(op, qx.Unitary) for op, _ in ops)
        assert res.dims == (2, 2)

    def test_with_noise(self):
        p = Program(X(0), H(1))
        ch = Channel.from_gate_fidelity(inst=X(0), fidelity=0.99)
        nm = NoiseModel.from_channels([ch])
        res = resolve_program(p, nm)
        ops = res.resolve(_EMPTY_PARAMS)
        assert len(ops) == 2
        assert isinstance(ops[0][0], qx.SuperOp)
        assert isinstance(ops[1][0], qx.Unitary)

    def test_parameterized(self):
        p = Program(Declare("theta", "REAL", 1), RZ(MemoryReference("theta", 0), 0))
        res = resolve_program(p)
        params = jnp.array([np.pi / 4])
        ops = res.resolve(params)
        assert len(ops) == 1
        assert isinstance(ops[0][0], qx.Unitary)

    def test_dag_structure(self):
        p = Program(H(0), X(0), CNOT(0, 1))
        dag = build_dag(resolve_program(p).subsystems)
        assert dag.has_edge(0, 1)
        assert dag.has_edge(1, 2)

    def test_measurement_and_reset(self):
        p = Program(Declare("ro", "BIT", 1), H(0), MEASURE(0, MemoryReference("ro", 0)))
        res = resolve_program(p)
        ops = res.resolve(_EMPTY_PARAMS)
        assert len(ops) == 2
        assert isinstance(ops[0][0], qx.Unitary)
        assert isinstance(ops[1][0], qx.QuantumInstrument)

    def test_qutrit_measurement_dimensions(self):
        p = Program(Gate("TX", [], [0]), Measurement(Qubit(0), None))
        res = resolve_program(p)
        ops = res.resolve(_EMPTY_PARAMS)
        assert res.dims == (3,)
        assert isinstance(ops[1][0], qx.QuantumInstrument)
        assert ops[1][0].dims == ((3,), (3,))


# ──────────────────────────────────────────────────────────
# Channel-flavor dispatch
# ──────────────────────────────────────────────────────────


class TestChannelFlavorDispatch:
    """Both flavors of gate and reset channel must reach the simulator.

    Gate channels come as ``Channel`` (Lindbladian-backed) or ``SuperopChannel`` (raw
    superoperator); resets as ``ResetChannel`` or ``SuperopResetChannel``. The resolver used to
    branch on the concrete Lindbladian classes, so the superoperator flavors fell through to the
    *ideal* operation and their noise vanished without any error -- and every derived channel
    operation (``@``, ``pauli_twirl``, ``to_coherent_channel``, ``to_stochastic_channel``)
    returns a ``SuperopChannel``, so that was the common case, not the exotic one.
    """

    @staticmethod
    def _emitted_superop(program, noise_model):
        ops, _, _ = expand_program(program, noise_model)
        assert len(ops) == 1
        return qx.to_superop(ops[0]).matrix

    def test_superop_gate_channel_noise_is_applied(self):
        channel = SuperopChannel.from_pauli_noise(X(0), {"Z": 0.1})
        emitted = self._emitted_superop(Program(X(0)), NoiseModel.from_channels([channel]))
        assert jnp.allclose(emitted, channel.process.matrix)
        # Explicitly not the ideal gate.
        assert not jnp.allclose(emitted, qx.to_superop(qx.gates.X).matrix)

    def test_lindbladian_gate_channel_noise_is_applied(self):
        channel = Channel.from_depolarizing_constant(X(0), 0.9)
        emitted = self._emitted_superop(Program(X(0)), NoiseModel.from_channels([channel]))
        assert jnp.allclose(emitted, channel.process.matrix)

    def test_superop_reset_channel_noise_is_applied(self):
        channel = SuperopResetChannel.from_reset_fidelity(ResetQubit(0), fidelity=0.9)
        emitted = self._emitted_superop(Program(RESET(0)), NoiseModel.from_channels([channel]))
        assert jnp.allclose(emitted, channel.process.matrix)
        # Explicitly not the ideal reset.
        assert not jnp.allclose(emitted, qx.to_superop(qx.gates.RESET(dim=2)).matrix)

    def test_lindbladian_reset_channel_noise_is_applied(self):
        channel = ResetChannel.from_amplitude_damping(ResetQubit(0), gamma=0.5)
        emitted = self._emitted_superop(Program(RESET(0)), NoiseModel.from_channels([channel]))
        assert jnp.allclose(emitted, channel.process.matrix)

    def test_derived_channels_reach_the_simulator(self):
        """The operations that produce SuperopChannel are the ones users actually reach for."""
        base = Channel.from_random_coherent_error(X(0), 0.97, rng=np.random.default_rng(0))
        for derived in (base.pauli_twirl(), base.to_coherent_channel(), base.to_stochastic_channel()):
            assert isinstance(derived, SuperopChannel)
            emitted = self._emitted_superop(Program(X(0)), NoiseModel.from_channels([derived]))
            assert jnp.allclose(emitted, derived.process.matrix)


class TestCycleMeasurementMode:
    def test_measurement_in_a_cycle_honors_superop_mode(self):
        """A MeasurementChannel inside a cycle must collapse to its total channel in superop mode.

        The superop pipeline is not meant to carry a QuantumInstrument; the cycle path used to
        emit one regardless of mode, unlike a standalone MEASURE.
        """
        gate = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        readout = MeasurementChannel.from_readout_fidelity(MEASURE(1, None), fidelity=0.95)
        cycle = gate | readout
        noise_model = NoiseModel.from_channels([cycle])
        program = Program(cycle.defcircuit, cycle.inst)

        superop_ops, _, _ = expand_program(program, noise_model, measurement="superop")
        assert all(isinstance(op, qx.SuperOp) for op in superop_ops)
        assert jnp.allclose(qx.to_superop(superop_ops[1]).matrix, readout.process.total_channel().matrix)

        instrument_ops, _, _ = expand_program(program, noise_model, measurement="instrument")
        assert isinstance(instrument_ops[1], qx.QuantumInstrument)

    def test_standalone_measurement_matches_cycle_behavior(self):
        """Whatever a bare MEASURE emits per mode, the cycle path must emit the same kind."""
        readout = MeasurementChannel.from_readout_fidelity(MEASURE(0, None), fidelity=0.95)
        noise_model = NoiseModel.from_channels([readout])
        for mode, expected in (("superop", qx.SuperOp), ("instrument", qx.QuantumInstrument)):
            ops, _, _ = expand_program(Program(MEASURE(0, None)), noise_model, measurement=mode)
            assert isinstance(ops[0], expected)
