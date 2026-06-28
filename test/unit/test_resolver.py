"""Unit tests for the resolver pipeline."""

import jax.numpy as jnp
import numpy as np
import quax as qx

from pyquil.gates import CNOT, MEASURE, RESET, RX, RZ, H, X
from pyquil.noise._channels import Channel, CycleChannel
from pyquil.noise._noise_model import NoiseModel
from pyquil.quil import Program
from pyquil.quilatom import FormalArgument, MemoryReference, Qubit
from pyquil.quilbase import (
    Declare,
    DefCircuit,
    Gate,
    Measurement,
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
