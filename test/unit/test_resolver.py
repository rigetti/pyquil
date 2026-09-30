"""Unit tests for the resolver pipeline."""

import jax.numpy as jnp
import numpy as np
import pytest
import quax as qx

from pyquil.gates import (
    ADD,
    AND,
    CNOT,
    CONVERT,
    DELAY,
    EQ,
    EXCHANGE,
    FENCE,
    HALT,
    LOAD,
    MEASURE,
    MOVE,
    NOP,
    NOT,
    RESET,
    RX,
    RZ,
    SHIFT_PHASE,
    STORE,
    WAIT,
    H,
    X,
)
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
from pyquil.quilatom import FormalArgument, Frame, Label, MemoryReference, Qubit
from pyquil.quilbase import (
    Declare,
    DefCircuit,
    Gate,
    Jump,
    JumpTarget,
    JumpUnless,
    JumpWhen,
    Measurement,
    Pragma,
    ResetQubit,
)
from pyquil.simulation._circuit import Circuit, dependency_edges
from pyquil.simulation._resolver import (
    expand_program,
    remap_qubits,
    resolve_program,
)

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

    def test_defcircuit_body_rejects_literal_qubits(self):
        """A DEFCIRCUIT body must reference only its own formal arguments.

        Quil permits a literal qubit in a body, but such a qubit is invisible to
        ``Program.get_qubit_indices``, so it escapes the register the simulator sizes itself
        for. Rejecting it here beats the downstream KeyError.
        """
        q0 = FormalArgument("q0")
        dc = DefCircuit("LEAKY", [], [q0], [H(q0), X(Qubit(3))])
        p = Program(dc, Gate("LEAKY", [], [Qubit(0)]))
        with pytest.raises(ValueError, match="not one of its formal arguments"):
            expand_program(p)

    def test_defcircuit_rejects_arity_mismatch(self):
        q0, q1 = FormalArgument("q0"), FormalArgument("q1")
        dc = DefCircuit("PAIR", [], [q0, q1], [CNOT(q0, q1)])
        p = Program(dc, Gate("PAIR", [], [Qubit(0)]))
        with pytest.raises(ValueError, match="but DEFCIRCUIT PAIR declares 2"):
            expand_program(p)

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
        gate_channel = SuperopChannel(inst=X(0), process=qx.to_superop(qx.gates.X), ideal_unitary=qx.gates.X)
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
# remap_qubits
#
# Dependency-graph construction and merge planning are quax's; they are covered by
# ``tests/test_circuits.py`` there.  What stays here is the Quil-specific part: mapping
# physical qubit ids onto register indices.
# ──────────────────────────────────────────────────────────


class TestRemapQubits:
    def test_remap_qubits(self):
        qubit_tuples = [(3,), (5,), (3, 5)]
        qubit_indices = {3: 0, 5: 1}
        result = remap_qubits(qubit_tuples, qubit_indices)
        assert result == ((0,), (1,), (0, 1))

    def test_remap_rejects_a_qubit_outside_the_register(self):
        with pytest.raises(ValueError, match="but the simulated register is"):
            remap_qubits([(3,), (7,)], {3: 0})


# ──────────────────────────────────────────────────────────
# resolve_program (integration)
# ──────────────────────────────────────────────────────────


class TestResolveProgram:
    def test_basic_roundtrip(self):
        p = Program(H(0), CNOT(0, 1), X(1))
        res = resolve_program(p)
        ops = res.resolve()
        assert len(ops) == 3
        assert all(isinstance(op, qx.Unitary) for op, _ in ops)
        assert res.dims == (2, 2)

    def test_with_noise(self):
        p = Program(X(0), H(1))
        ch = Channel.from_gate_fidelity(inst=X(0), fidelity=0.99)
        nm = NoiseModel.from_channels([ch])
        res = resolve_program(p, nm)
        ops = res.resolve()
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

    def test_dependency_structure(self):
        p = Program(H(0), X(0), CNOT(0, 1))
        edges = dependency_edges(resolve_program(p).subsystems)
        assert (0, 1) in edges
        assert (1, 2) in edges

    def test_resolve_returns_a_circuit_on_the_program_register(self):
        p = Program(H(0), CNOT(0, 1))
        circuit = resolve_program(p).resolve()
        assert isinstance(circuit, Circuit)
        assert circuit.dims == (2, 2)
        assert circuit.subsystems == ((0,), (0, 1))

    def test_measurement_and_reset(self):
        p = Program(Declare("ro", "BIT", 1), H(0), MEASURE(0, MemoryReference("ro", 0)))
        res = resolve_program(p)
        ops = res.resolve()
        assert len(ops) == 2
        assert isinstance(ops[0][0], qx.Unitary)
        assert isinstance(ops[1][0], qx.QuantumInstrument)

    def test_qutrit_measurement_dimensions(self):
        p = Program(Gate("TX", [], [0]), Measurement(Qubit(0), None))
        res = resolve_program(p)
        ops = res.resolve()
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


class TestMeasurementRepresentation:
    """Expansion always emits an instrument; collapsing it is the backend's job.

    There used to be a ``measurement`` mode on expansion, chosen by the simulator family, so
    that the differentiable backends never had to carry an instrument. It was redundant:
    collapsing an instrument to its total channel is exactly what
    :meth:`~pyquil.simulation._circuit.Circuit.to_superops` does, so the mode is a representation change applied one
    layer too early. These tests pin the equivalence that let it be deleted.
    """

    def test_measurement_in_a_cycle_is_an_instrument(self):
        gate = Channel.from_depolarizing_constant(RX(np.pi / 2, 0), 0.99)
        readout = MeasurementChannel.from_readout_fidelity(MEASURE(1, None), fidelity=0.95)
        cycle = gate | readout
        noise_model = NoiseModel.from_channels([cycle])
        program = Program(cycle.defcircuit, cycle.inst)

        ops, _, _ = expand_program(program, noise_model)
        assert isinstance(ops[1], qx.QuantumInstrument)

    def test_standalone_measurement_matches_cycle_behavior(self):
        """A bare MEASURE and one reached through a cycle must emit the same kind of operator."""
        readout = MeasurementChannel.from_readout_fidelity(MEASURE(0, None), fidelity=0.95)
        noise_model = NoiseModel.from_channels([readout])
        ops, _, _ = expand_program(Program(MEASURE(0, None)), noise_model)
        assert isinstance(ops[0], qx.QuantumInstrument)

    @pytest.mark.parametrize("fidelity", [1.0, 0.95, 0.9])
    @pytest.mark.parametrize("asymmetry", [0.0, 0.3])
    def test_collapsing_an_instrument_gives_the_dephasing_channel(self, fidelity, asymmetry):
        """``to_superops`` reproduces what the old ``measurement="superop"`` mode emitted."""
        inst = MEASURE(0, None)
        readout = MeasurementChannel.from_readout_fidelity(inst, fidelity=fidelity, asymmetry=asymmetry)
        noise_model = NoiseModel.from_channels([readout])
        circuit = resolve_program(Program(H(0), inst), noise_model).resolve()

        assert isinstance(circuit[1][0], qx.QuantumInstrument)
        collapsed = circuit.to_superops()
        assert all(isinstance(op, qx.SuperOp) for op in collapsed.operators)
        assert jnp.allclose(collapsed[1][0].matrix, qx.to_superop(readout.process.total_channel()).matrix)

    def test_the_differentiable_family_keeps_no_instruments(self):
        """``_prepare_ops`` collapses them, which is why that family has nothing atomic."""
        from pyquil.simulation._simulator import DensityMatrixSimulator

        sim = DensityMatrixSimulator(Program(Declare("ro", "BIT", 1), H(0), MEASURE(0, ("ro", 0))))
        assert not any(isinstance(op, qx.QuantumInstrument) for op in sim._resolution.ops)
        # Nothing is pinned, so the measurement is free to merge with the gate before it.
        assert sim.plan.groups == (((0, 1), (0,)),)


# ──────────────────────────────────────────────────────────
# Instruction support
# ──────────────────────────────────────────────────────────


class TestInstructionSupport:
    """Control flow and classical memory raise; everything else non-quantum is ignored."""

    @pytest.mark.parametrize(
        "inst",
        [
            Jump(Label("end")),
            JumpWhen(Label("end"), MemoryReference("ro", 0)),
            JumpUnless(Label("end"), MemoryReference("ro", 0)),
            NOT(("ro", 0)),
            AND(("ro", 0), 1),
            ADD(("x", 0), 1.0),
            MOVE(("ro", 0), 1),
            EXCHANGE(("ro", 0), ("ro", 1)),
            CONVERT(("x", 0), ("ro", 0)),
            LOAD(("ro", 0), "ro", ("n", 0)),
            STORE("ro", ("n", 0), 1),
            EQ(("ro", 0), ("ro", 1), 1),
        ],
        ids=lambda inst: type(inst).__name__,
    )
    def test_control_flow_and_classical_memory_raise(self, inst):
        program = Program(
            Declare("ro", "BIT", 2),
            Declare("x", "REAL", 1),
            Declare("n", "INTEGER", 1),
            H(0),
            inst,
            JumpTarget(Label("end")),
        )
        with pytest.raises(ValueError, match="cannot be simulated"):
            expand_program(program)

    def test_non_quantum_instructions_are_ignored(self):
        """Declarations, pragmas, labels, HALT/NOP/WAIT and pulse-level Quil-T pass through silently."""
        frame = Frame([Qubit(0)], "rf")
        program = Program(
            Declare("ro", "BIT", 1),
            Pragma("INITIAL_REWIRING", freeform_string='"NAIVE"'),
            JumpTarget(Label("start")),
            H(0),
            FENCE(0),
            DELAY(0, 1e-8),
            SHIFT_PHASE(frame, 0.1),
            NOP,
            WAIT,
            HALT,
        )
        ops, subsystems, _ = expand_program(program)
        assert len(ops) == 1
        assert isinstance(ops[0], qx.Unitary)
        assert subsystems == ((0,),)


class TestParameterSlots:
    def test_repeated_reference_shares_a_slot(self):
        theta0 = MemoryReference("theta", 0)
        program = Program(Declare("theta", "REAL", 1), RX(theta0, 0), RZ(theta0, 1))
        ops, _, parameters = expand_program(program)
        assert parameters == (("theta", 0),)
        assert [op.arguments[0].slot_indices for op in ops] == [(0,), (0,)]

    def test_distinct_references_get_distinct_slots_in_first_use_order(self):
        program = Program(
            Declare("theta", "REAL", 2), RX(MemoryReference("theta", 1), 0), RZ(MemoryReference("theta", 0), 0)
        )
        _, _, parameters = expand_program(program)
        assert parameters == (("theta", 1), ("theta", 0))
        assert resolve_program(program).num_parameters == 2

    def test_parametric_gate_with_a_modifier_is_rejected(self):
        """The parametric path must reject modifiers too, not just the fixed-gate path."""
        program = Program(Declare("theta", "REAL", 1), RX(MemoryReference("theta", 0), 0).dagger())
        with pytest.raises(ValueError, match="modifiers are not supported.*DAGGER"):
            expand_program(program)
