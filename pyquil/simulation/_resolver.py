##############################################################################
# Copyright 2016-2026 Rigetti Computing
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
##############################################################################
"""Shared infrastructure for the density-matrix and state-vector simulators.

This module provides the simulation preprocessing pipeline:

1. **Expander** — expands a program into a flat list of operators and physical
   qubit tuples, resolving noise channels, custom gates, and DEFCIRCUIT
   bodies.  Fixed (non-parameterized) operations are returned as concrete
   quax types; parameterized gates are returned as callables.
2. **Resolver** — converts a parameter vector into a list of
   ``(operator, subsystem)`` pairs using native quax types.
3. **Adapters** — convert resolved operations into the form expected by each
   simulator backend (``SuperOp`` for density matrices; ``Unitary``/``KrausMap``/
   ``QuantumInstrument`` for state-vector trajectories).
4. **Compressor** — merges adjacent operators via greedy edge contraction.
"""

from __future__ import annotations

import heapq
import logging
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from typing import Any, Literal, NamedTuple, TypeAlias, cast

import jax.numpy as jnp
import networkx as nx
import numpy as np
import quax as qx
from jax import Array

from pyquil.noise._channels import (
    ChannelBase,
    CycleChannel,
    MeasurementChannel,
    ResetChannelBase,
    get_custom_gates_from_program,
    get_instruction_unitary,
)
from pyquil.noise._noise_model import (
    NoiseModelLike,
)
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, Qubit, _contained_mrefs, substitute
from pyquil.quilbase import DefCircuit, Gate, Measurement, Reset, ResetQubit

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────

# A fixed (non-parameterized) operator — the most specific native quax type.
FixedOp: TypeAlias = qx.Unitary | qx.SuperOp | qx.KrausMap | qx.QuantumInstrument

# How ``MEASURE`` is represented during expansion.  The differentiable simulators
# (state-vector, density-matrix) treat a measurement as a plain dephasing
# ``SuperOp`` (``"superop"``); the trajectory simulators keep it as a sampled
# ``QuantumInstrument`` (``"instrument"``).  See :func:`resolve_for_differentiable` and
# :func:`resolve_for_trajectory`.
MeasurementMode: TypeAlias = Literal["superop", "instrument"]


class ParametricGate:
    """A parametric gate whose matrix depends on runtime parameters.

    Instances are callable: ``gate(params) -> qx.Unitary``.  They also expose
    the gate constructor and parameter layout so that the simulator can group
    gates by type and use ``jax.vmap`` for efficient batch construction.
    """

    __slots__ = ("gate_fn", "param_indices", "concrete_values")

    def __init__(
        self,
        gate_fn: Callable[..., qx.Unitary],
        param_indices: tuple[int, ...],
        concrete_values: tuple[float, ...],
    ) -> None:
        #: The quax gate constructor (e.g. ``qx.gates.RX``).
        self.gate_fn = gate_fn
        #: Per-argument index into the flat parameter vector, or ``-1``
        #: when that argument is a compile-time constant.
        self.param_indices = param_indices
        #: Per-argument concrete value (``nan`` for runtime-parametric slots).
        self.concrete_values = concrete_values

    def __call__(self, params: Array) -> qx.Unitary:
        resolved: list[Any] = []
        for pi, cv in zip(self.param_indices, self.concrete_values, strict=False):
            if pi >= 0:
                resolved.append(params[pi])
            else:
                resolved.append(cv)
        result = self.gate_fn(*resolved)
        if not isinstance(result, qx.Unitary):
            result = cast(Any, result)
            result = qx.Unitary.from_matrix(result.matrix, result.dims)
        return result


# An expanded item is either a fixed operator or a ParametricGate that
# resolves parameters into a Unitary.
ExpandedOp: TypeAlias = FixedOp | ParametricGate

# Resolved operations retain the most specific native quax type.
ResolvedOp = tuple[FixedOp, tuple[int, ...]]

# Trajectory operations for the state-vector simulator.
TrajectoryOp = tuple[qx.Unitary | qx.KrausMap | qx.QuantumInstrument, tuple[int, ...]]

# Density-matrix operations.
DensityMatrixOp = tuple[qx.SuperOp, tuple[int, ...]]


# ──────────────────────────────────────────────────────────
# DEFCIRCUIT expansion
# ──────────────────────────────────────────────────────────


def expand_defcircuit_body(
    inst: Gate,
    defcircuit: DefCircuit,
    circuit_definitions: dict[str, DefCircuit],
) -> Iterator[Gate | Measurement | ResetQubit | Reset]:
    """Yield concrete instructions from a DEFCIRCUIT invocation.

    Substitutes formal qubit/parameter arguments with the concrete values
    from ``inst``.  Handles nested DEFCIRCUITs via recursion.

    :param inst: The Gate that invokes the DEFCIRCUIT.
    :param defcircuit: The DefCircuit definition to expand.
    :param circuit_definitions: All known DEFCIRCUIT definitions (for nested expansion).
    :yields: Concrete instructions with physical qubits and resolved parameters.
    """
    if len(inst.qubits) != len(defcircuit.qubit_variables):
        raise ValueError(
            f"{inst.out()!r} passes {len(inst.qubits)} qubit(s) but DEFCIRCUIT "
            f"{defcircuit.name} declares {len(defcircuit.qubit_variables)} "
            f"({[str(a) for a in defcircuit.qubit_variables]})."
        )
    if len(inst.params) != len(defcircuit.parameters):
        raise ValueError(
            f"{inst.out()!r} passes {len(inst.params)} parameter(s) but DEFCIRCUIT "
            f"{defcircuit.name} declares {len(defcircuit.parameters)} "
            f"({[str(a) for a in defcircuit.parameters]})."
        )
    qarg_to_arg_map = {qarg: q for q, qarg in zip(inst.qubits, defcircuit.qubit_variables, strict=True)}
    parg_to_arg_map = {parg: param for param, parg in zip(inst.params, defcircuit.parameters, strict=True)}

    def resolve_qubit(qarg: Any) -> Any:
        """Map a formal argument to its concrete qubit.

        A DEFCIRCUIT body must reference only the circuit's own formal arguments. Quil permits
        a literal qubit in a body (``DEFCIRCUIT C q: X q; X 3``), but simulating one is a trap:
        the qubit is invisible to ``Program.get_qubit_indices``, so it silently escapes the
        register the simulator sizes itself for. Rejecting it here is clearer than the
        downstream failure.
        """
        if qarg not in qarg_to_arg_map:
            raise ValueError(
                f"DEFCIRCUIT {defcircuit.name} body references {qarg}, which is not one of its "
                f"formal arguments ({[str(a) for a in defcircuit.qubit_variables]}). Literal "
                "qubits in a DEFCIRCUIT body are not supported; parameterize the circuit over "
                "all the qubits it touches."
            )
        return qarg_to_arg_map[qarg]

    for circuit_inst in defcircuit.instructions:
        if isinstance(circuit_inst, Gate):
            circuit_inst = deepcopy(circuit_inst)
            circuit_inst.qubits = [resolve_qubit(qarg) for qarg in circuit_inst.qubits]
            if hasattr(circuit_inst, "params"):
                circuit_inst.params = [substitute(param, parg_to_arg_map) for param in circuit_inst.params]  # type: ignore[arg-type]
            if circuit_inst.name in circuit_definitions:
                yield from expand_defcircuit_body(
                    circuit_inst, circuit_definitions[circuit_inst.name], circuit_definitions
                )
            else:
                yield circuit_inst
        elif isinstance(circuit_inst, Measurement):
            circuit_inst = deepcopy(circuit_inst)
            circuit_inst.qubit = resolve_qubit(circuit_inst.qubit)
            yield circuit_inst
        elif isinstance(circuit_inst, ResetQubit):
            circuit_inst = deepcopy(circuit_inst)
            circuit_inst.qubit = resolve_qubit(circuit_inst.qubit)
            yield circuit_inst
        else:
            yield deepcopy(circuit_inst)  # type: ignore[misc]


# ══════════════════════════════════════════════════════════
# Expander
# ══════════════════════════════════════════════════════════


def _measure_registers(program: Program) -> set[str]:
    """Return the set of register names that are targets of MEASURE instructions."""
    regs: set[str] = set()
    for inst in program.instructions:
        if isinstance(inst, Measurement):
            cr = inst.classical_reg
            if cr is not None:
                regs.add(cr.name)
    return regs


def expand_program(
    program: Program,
    noise_model: NoiseModelLike | None = None,
    qubit_dimensions: Mapping[int, int] | None = None,
    *,
    measurement: MeasurementMode = "instrument",
) -> tuple[list[ExpandedOp], list[tuple[int, ...]], list[tuple[str, int]]]:
    """Expand a program into operators and physical qubit tuples.

    Fixed (non-parameterized) operations are returned as concrete quax types
    (``Unitary``, ``SuperOp``, ``QuantumInstrument``).  Only parameterized
    gates are returned as ``Callable[[Array], Unitary]``.

    DEFCIRCUIT invocations are expanded:

    * If a cycle invocation matches a :class:`CycleChannel` in the noise
      model, the cycle is expanded using the channel's constituent operators.
    * Otherwise the DEFCIRCUIT body is expanded via qubit/param substitution
      and each resulting instruction is resolved individually.

    The noise model is fully resolved during expansion: noisy gates become
    ``SuperOp``, noisy measurements become ``QuantumInstrument``, and noisy
    resets become ``SuperOp``.

    :param program: Quil program (may contain DEFCIRCUITs).
    :param noise_model: Optional noise model.
    :param qubit_dimensions: Optional mapping from physical qubit id to its
        Hilbert-space dimension. Used for ideal measurement and reset operators,
        whose quax constructors otherwise default to qubit dimension.
    :param measurement: How to represent ``MEASURE`` instructions.  ``"instrument"``
        (default) keeps a sampled ``QuantumInstrument``; ``"superop"`` emits the
        measurement's dephasing total channel as a ``SuperOp`` so no instrument
        is produced — used by the differentiable
        simulators.
    :return: Tuple of ``(ops, qubit_tuples, param_refs)`` where each op is
        either a concrete quax operator or a ``Callable[[Array], Unitary]``
        for parameterized gates, each qubit tuple contains physical qubit
        IDs, and ``param_refs`` is a list of ``(register_name, offset)``
        pairs for each scalar parameter in program order.
    """
    # Program-level derivations. These are independent of qubit dimensions, so
    # ``resolve_program``'s two passes each recompute them; measured at 1-4 ms (under 6% of a
    # resolve), which is not worth caching across the passes.
    circuit_definitions: dict[str, DefCircuit] = {
        inst.name: inst for inst in program.instructions if isinstance(inst, DefCircuit)
    }
    custom_gates = get_custom_gates_from_program(program) or None
    measure_regs = _measure_registers(program)
    all_qubits = sorted(program.get_qubit_indices())

    ops: list[ExpandedOp] = []
    qubit_tuples: list[tuple[int, ...]] = []
    param_refs: list[tuple[str, int]] = []
    param_counter = 0

    def _emit_op(op: ExpandedOp, qubits: tuple[int, ...]) -> None:
        ops.append(op)
        qubit_tuples.append(qubits)

    def _resolve_gate(inst: Gate) -> tuple[ExpandedOp, tuple[int, ...]]:
        """Resolve a single gate instruction to an operator or callable."""
        nonlocal param_counter
        qubits = tuple(inst.get_qubit_indices())

        # Check noise model first. Match on ChannelBase, not a concrete class: gate channels
        # come in both Lindbladian-backed (Channel) and raw-superoperator (SuperopChannel)
        # flavors, and every derived operation (composition, twirling, coherent/stochastic
        # splitting) returns the latter. Narrowing to one of them silently drops the other's
        # noise and simulates the ideal gate instead.
        channel = noise_model.get_channel(inst) if noise_model is not None else None
        if isinstance(channel, ChannelBase):
            return channel.process, qubits
        if isinstance(channel, CycleChannel):
            raise ValueError(f"CycleChannel for {inst.name} was not expanded before gate resolution.")

        # Parameterized gate → callable that resolves params at call time.
        if any(_contained_mrefs(p) for p in inst.params):  # type: ignore[arg-type]
            gate_name = inst.name
            if custom_gates is not None and gate_name in custom_gates:
                gate_def = custom_gates[gate_name]
            elif gate_name in qx.gates.QUANTUM_GATES:
                gate_def = qx.gates.QUANTUM_GATES[gate_name]
            else:
                raise KeyError(f"Unknown gate '{gate_name}'.")

            param_indices: list[int] = []
            concrete_values: list[float] = []
            for p in inst.params:
                mrefs = _contained_mrefs(p)  # type: ignore[arg-type]
                if not mrefs:
                    # A concrete number: a compile-time constant for this gate.
                    param_indices.append(-1)
                    concrete_values.append(float(np.real(p)))
                elif not isinstance(p, MemoryReference):
                    # An arithmetic expression over one or more memory regions, e.g.
                    # ``RX(theta[0] / 2) 0``. Each ParametricGate argument maps to a single
                    # slot of the flat parameter vector, which is what lets the simulator
                    # batch same-shaped gates under one ``jax.vmap``; an arbitrary
                    # expression would have to become part of that batching key.
                    raise ValueError(
                        f"Gate parameter {p} in {inst.out()!r} is an expression over memory "
                        f"region(s) {sorted(m.name for m in mrefs)}, which is not supported. "
                        "Pass the parameter directly (e.g. RX(theta[0]) with the division folded "
                        "into the value you bind), or substitute concrete values into the program "
                        "before simulating."
                    )
                elif p.name in measure_regs:
                    # Classically-conditioned angle: the value is only known mid-circuit.
                    raise ValueError(
                        f"Gate parameter {p} in {inst.out()!r} reads memory region "
                        f"'{p.name}', which is written by a MEASURE in this program. "
                        "Feed-forward (classically-conditioned) parameters are not supported."
                    )
                else:
                    param_indices.append(param_counter)
                    concrete_values.append(float("nan"))
                    param_refs.append((p.name, p.offset))
                    param_counter += 1

            return ParametricGate(gate_def, tuple(param_indices), tuple(concrete_values)), qubits

        # Fixed gate → resolve to Unitary now.
        unitary = get_instruction_unitary(inst, custom_gates=custom_gates)
        return unitary, qubits

    def _dimension_for(qubit: int) -> int:
        return qubit_dimensions.get(qubit, 2) if qubit_dimensions is not None else 2

    def _resolve_measurement(inst: Measurement) -> tuple[FixedOp, tuple[int, ...]]:
        """Resolve a measurement instruction.

        Under ``measurement="instrument"`` the result is a ``QuantumInstrument``
        (sampled by the trajectory simulators).  Under ``measurement="superop"``
        the instrument's dephasing total channel is returned as a ``SuperOp`` so
        the differentiable pipeline never has to carry — or merge — an instrument.
        """
        qubits = tuple(inst.get_qubit_indices())
        channel = noise_model.get_channel(inst) if noise_model is not None else None
        instrument = (
            channel.process
            if isinstance(channel, MeasurementChannel)
            else qx.gates.MEASURE(dim=_dimension_for(qubits[0]))
        )
        if measurement == "superop":
            return instrument.total_channel(), qubits
        return instrument, qubits

    def _resolve_reset_qubit(inst: ResetQubit) -> tuple[FixedOp, tuple[int, ...]]:
        """Resolve a targeted reset instruction."""
        qubits = tuple(inst.get_qubit_indices())  # type: ignore[arg-type]
        channel = noise_model.get_channel(inst) if noise_model is not None else None
        # ResetChannelBase, not ResetChannel: see the note in _resolve_gate. SuperopResetChannel
        # is the other reset flavor and must not fall through to the ideal reset.
        if isinstance(channel, ResetChannelBase):
            return channel.process, qubits
        return qx.gates.RESET(dim=_dimension_for(qubits[0])), qubits

    def _emit_instruction(inst: Gate | Measurement | ResetQubit | Reset) -> None:
        """Resolve and emit a single instruction."""
        match inst:
            case Gate():
                op, qubits = _resolve_gate(inst)
                _emit_op(op, qubits)
            case Measurement():
                op, qubits = _resolve_measurement(inst)
                _emit_op(op, qubits)
            case ResetQubit():
                op, qubits = _resolve_reset_qubit(inst)
                _emit_op(op, qubits)
            case Reset():
                # A global RESET is the same physical operation as a targeted one on every
                # qubit, so it must consult the noise model per qubit rather than always
                # emitting the ideal reset -- otherwise a model carrying reset channels is
                # silently ignored for `RESET` while being honoured for `RESET <q>`.
                for q in all_qubits:
                    op, qubits = _resolve_reset_qubit(ResetQubit(Qubit(q)))
                    _emit_op(op, qubits)

    for inst in program.instructions:
        if isinstance(inst, DefCircuit):
            continue

        if isinstance(inst, Gate) and inst.name in circuit_definitions:
            channel = noise_model.get_channel(inst) if noise_model is not None else None

            if isinstance(channel, CycleChannel):
                # Expand using the channel's constituent operators. A MeasurementChannel
                # constituent carries a QuantumInstrument, which must be collapsed to its total
                # channel under measurement="superop" exactly as a standalone MEASURE is -- the
                # differentiable pipeline is not meant to carry an instrument.
                for sub_ch in channel.channels:
                    # Use the channel's own `qubits` property rather than reaching through to
                    # `inst.get_qubit_indices()`, whose return type varies across the constituent
                    # families (a list for gates, a set-or-None for resets).
                    sub_qubits = tuple(sub_ch.qubits)
                    if measurement == "superop" and isinstance(sub_ch, MeasurementChannel):
                        _emit_op(sub_ch.process.total_channel(), sub_qubits)
                    else:
                        _emit_op(sub_ch.process, sub_qubits)
            else:
                # Expand DEFCIRCUIT body and resolve each instruction.
                for expanded_inst in expand_defcircuit_body(inst, circuit_definitions[inst.name], circuit_definitions):
                    _emit_instruction(expanded_inst)
        elif isinstance(inst, (Gate, Measurement, ResetQubit, Reset)):
            _emit_instruction(inst)

    return ops, qubit_tuples, param_refs


# ══════════════════════════════════════════════════════════
# DAG construction & qubit remapping
# ══════════════════════════════════════════════════════════


def remap_qubits(
    qubit_tuples: list[tuple[int, ...]],
    qubit_indices: dict[int, int],
) -> list[tuple[int, ...]]:
    """Remap physical qubit IDs to 0-based indices.

    :param qubit_tuples: List of physical qubit tuples from :func:`expand_program`.
    :param qubit_indices: Mapping from physical qubit id → 0-based index.
    :return: Remapped qubit tuples.
    """
    remapped: list[tuple[int, ...]] = []
    for qubits in qubit_tuples:
        missing = [q for q in qubits if q not in qubit_indices]
        if missing:
            raise ValueError(
                f"Operation on qubit(s) {missing} but the simulated register is "
                f"{sorted(qubit_indices)}. Pass qubits=[...] listing every qubit the program "
                "touches."
            )
        remapped.append(tuple(qubit_indices[q] for q in qubits))
    return remapped


def build_dag(qubit_tuples: list[tuple[int, ...]]) -> nx.DiGraph:
    """Build a dependency DAG from qubit tuples.

    Each node corresponds to one operation (indexed 0..N-1). An edge
    ``(u, v)`` exists when ``u`` and ``v`` act on a shared qubit and
    ``u`` precedes ``v`` in program order.

    :param qubit_tuples: Remapped qubit tuples (0-based indices).
    :return: DAG with node attribute ``"qubits"`` storing each node's qubit tuple.
    """
    dag: nx.DiGraph = nx.DiGraph()
    last_on_qubit: dict[int, int] = {}

    for idx, qubits in enumerate(qubit_tuples):
        dag.add_node(idx, qubits=qubits)
        for q in qubits:
            if q in last_on_qubit:
                dag.add_edge(last_on_qubit[q], idx)
            last_on_qubit[q] = idx

    return dag


def _infer_dims(resolved: list[ResolvedOp], n_qubits: int) -> tuple[int, ...]:
    """Infer per-qudit dimensions from resolved operators."""
    dims = [2] * n_qubits
    for op, subsystem in resolved:
        for q, d in zip(subsystem, op.dims[1], strict=False):
            dims[q] = max(dims[q], d)
    return tuple(dims)


# ══════════════════════════════════════════════════════════
# Resolver
# ══════════════════════════════════════════════════════════


class Resolution(NamedTuple):
    """Everything the simulators need from a program after expansion.

    :param dims: Inferred per-qudit dimensions (e.g. ``(2, 2, 3)``).
    :param ops: Expanded operators, one per DAG node, in program order.
    :param subsystems: 0-based qubit tuple each operator acts on.
    :param param_refs: ``(register_name, offset)`` for each scalar parameter.
    """

    dims: tuple[int, ...]
    ops: list[ExpandedOp]
    subsystems: list[tuple[int, ...]]
    param_refs: list[tuple[str, int]]

    def resolve(self, params: Array) -> list[ResolvedOp]:
        """Bind *params* to produce one concrete operator per expanded operation.

        Fixed operators pass straight through; :class:`ParametricGate` entries are called with
        the parameter vector to build their ``Unitary``.

        :param params: Flat parameter vector, laid out as ``param_refs``.
        :return: ``(operator, subsystem)`` pairs in program order.
        """
        return [
            (op(params) if isinstance(op, ParametricGate) else op, subsystem)
            for op, subsystem in zip(self.ops, self.subsystems, strict=True)
        ]


def resolve_program(
    program: Program,
    noise_model: NoiseModelLike | None = None,
    qubits: list[int] | None = None,
    dims: tuple[int, ...] | None = None,
    *,
    measurement: MeasurementMode = "instrument",
) -> Resolution:
    """Expand a program and build its parameter-resolving closure.

    Operators are returned in their most specific native type:

    * Ideal gates → ``qx.Unitary`` (parametric gates as a ``ParametricGate`` callable)
    * Noisy gates (``Channel``) → ``qx.SuperOp``
    * Expanded cycle gates with ``CycleChannel`` noise → constituent ``qx.SuperOp``
    * Measurements → ``qx.QuantumInstrument``
    * Noisy/ideal resets → ``qx.SuperOp``

    The program is expanded twice: first with default (qubit) register
    dimensions to infer each register's true dimension from the gates and noisy
    channels, then again with those dimensions so ideal measurement/reset
    instruments use the correct dimension.  Passing *dims* skips the first pass.

    :param program: Quil program (may contain DEFCIRCUITs and DEFGATEs).
    :param noise_model: Optional noise model.
    :param qubits: Optional explicit qubit list. If ``None``, inferred from the
        program. Use this when the simulator knows about qubits that don't
        appear in the program.
    :param dims: Optional pre-determined per-qudit dimensions.
    :param measurement: Measurement representation — see :func:`expand_program`.
        Prefer the :func:`resolve_for_differentiable` / :func:`resolve_for_trajectory`
        entry points, which pin the correct mode for each simulator family.
    :return: A :class:`Resolution`.
    """
    if qubits is None:
        qubits = sorted(program.get_qubit_indices())
    qubit_indices = {q: i for i, q in enumerate(qubits)}

    def expand(
        qubit_dimensions: Mapping[int, int] | None,
    ) -> tuple[list[ExpandedOp], list[tuple[int, ...]], list[tuple[str, int]]]:
        ops, phys_qubits, param_refs = expand_program(
            program, noise_model, qubit_dimensions=qubit_dimensions, measurement=measurement
        )
        return ops, remap_qubits(phys_qubits, qubit_indices), param_refs

    if dims is None:
        ops, subsystems, param_refs = expand(None)
        probe = Resolution(dims=(), ops=ops, subsystems=subsystems, param_refs=param_refs)
        dims = _infer_dims(probe.resolve(jnp.zeros(len(param_refs))), len(qubits))

    qubit_dimensions = {q: dims[i] for q, i in qubit_indices.items()}
    ops, subsystems, param_refs = expand(qubit_dimensions)
    return Resolution(dims, ops, subsystems, param_refs)


def resolve_for_differentiable(
    program: Program,
    noise_model: NoiseModelLike | None = None,
    qubits: list[int] | None = None,
    dims: tuple[int, ...] | None = None,
) -> Resolution:
    """Resolve *program* for the differentiable state-vector and density-matrix simulators.

    ``MEASURE`` becomes a dephasing superoperator so the pipeline stays differentiable.

    :param program: Quil program (may contain DEFCIRCUITs and DEFGATEs).
    :param noise_model: Optional noise model; instructions with no channel are ideal.
    :param qubits: Explicit register order. Defaults to the program's qubits, ascending.
    :param dims: Optional pre-determined per-qudit dimensions, in ``qubits`` order.
    :return: A :class:`Resolution` produced with ``measurement="superop"``.
    """
    return resolve_program(program, noise_model, qubits, dims, measurement="superop")


def resolve_for_trajectory(
    program: Program,
    noise_model: NoiseModelLike | None = None,
    qubits: list[int] | None = None,
    dims: tuple[int, ...] | None = None,
) -> Resolution:
    """Resolve *program* for the Monte-Carlo trajectory simulators.

    ``MEASURE`` stays a ``QuantumInstrument`` that can be sampled.

    :param program: Quil program (may contain DEFCIRCUITs and DEFGATEs).
    :param noise_model: Optional noise model; instructions with no channel are ideal.
    :param qubits: Explicit register order. Defaults to the program's qubits, ascending.
    :param dims: Optional pre-determined per-qudit dimensions, in ``qubits`` order.
    :return: A :class:`Resolution` produced with ``measurement="instrument"``.
    """
    return resolve_program(program, noise_model, qubits, dims, measurement="instrument")


def enumerate_bases(
    emit_order: list[tuple[int, list[int], tuple[int, ...]]],
) -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
    """Enumerate the distinct base subsystems produced by a compressor.

    The compressor's ``emit_order`` (see :func:`compressor_from_dag`) lists one
    ``(root, nodes, subsystem)`` entry per emitted group, in application order.  The
    merge structure depends only on the DAG, not on parameter values, so the base
    subsystems can be read straight off ``emit_order`` — no ``resolve``/``compress``
    probe is required.

    The differentiable simulators dispatch each compressed operation through a
    ``jax.lax.switch`` keyed by its base, so the number of *distinct* bases (rather
    than the number of operations) sets the size of the compiled graph.

    :param emit_order: The ``emit_order`` attribute of a compressor closure.
    :return: ``(bases, op_index)`` where ``bases`` is the distinct subsystems in
        first-seen order and ``op_index[k]`` is the base index of the ``k``-th
        emitted operation.
    """
    bases: list[tuple[int, ...]] = []
    sub_to_branch: dict[tuple[int, ...], int] = {}
    op_index: list[int] = []
    for _, _, subsystem in emit_order:
        if subsystem not in sub_to_branch:
            sub_to_branch[subsystem] = len(bases)
            bases.append(subsystem)
        op_index.append(sub_to_branch[subsystem])
    return bases, tuple(op_index)


# ══════════════════════════════════════════════════════════
# Adapters
#
# These adapters live outside the resolver intentionally. The resolver produces
# operators in their most specific native type (Unitary, SuperOp, KrausMap,
# QuantumInstrument). Each simulator backend then adapts these to its required
# representation. This separation keeps the resolver backend-agnostic and the
# per-op conversion cost (type dispatch + matrix reshape) is negligible compared
# to the actual simulation.
# ══════════════════════════════════════════════════════════


def adapt_for_density_matrix(
    ops: list[ResolvedOp],
) -> list[DensityMatrixOp]:
    """Convert resolved operations to ``(SuperOp, subsystem)`` pairs for density-matrix simulation.

    * ``Unitary`` → ``qx.to_superop(op)``
    * ``SuperOp`` → pass through
    * ``KrausMap`` → ``qx.to_superop(op)``
    * ``QuantumInstrument`` → ``qx.to_superop(op.total_channel())``

    :param ops: Resolved operations from :func:`build_resolver`.
    :return: List of ``(SuperOp, subsystem)`` pairs.
    """
    result: list[DensityMatrixOp] = []
    for op, subsystem in ops:
        # ``qx.to_superop`` is single-dispatch and idempotent on SuperOp, so it
        # covers Unitary/SuperOp/KrausMap directly; only an instrument needs its
        # total channel taken first.
        channel = op.total_channel() if isinstance(op, qx.QuantumInstrument) else op
        result.append((qx.to_superop(channel), subsystem))
    return result


def adapt_for_trajectory(
    ops: list[ResolvedOp],
    kraus_truncation_threshold: float = 1e-6,
) -> list[TrajectoryOp]:
    """Convert resolved operations to trajectory-compatible types.

    * ``Unitary`` → pass through
    * ``SuperOp`` → ``truncate_kraus(to_kraus(op))`` → ``KrausMap``
    * ``KrausMap`` → pass through
    * ``QuantumInstrument`` → pass through

    :param ops: Resolved operations from :func:`build_resolver`.
    :param kraus_truncation_threshold: Threshold for Kraus truncation.
    :return: List of ``(Unitary | KrausMap | QuantumInstrument, subsystem)`` pairs.
    """
    result: list[TrajectoryOp] = []
    for op, subsystem in ops:
        match op:
            case qx.SuperOp():
                km = qx.truncate_kraus(qx.to_kraus(op), atol=kraus_truncation_threshold)
                result.append((km, subsystem))
            case qx.Unitary() | qx.KrausMap() | qx.QuantumInstrument():
                result.append((op, subsystem))
            case _:
                raise TypeError(f"Cannot adapt operator of type {type(op).__name__} for trajectory simulation.")
    return result


# ══════════════════════════════════════════════════════════
# Compressor (greedy edge contraction)
# ══════════════════════════════════════════════════════════


def _merge_ops(
    ops_with_subsystems: list[ResolvedOp],
    merged_subsystem: tuple[int, ...],
    dims: tuple[int, ...],
) -> ResolvedOp:
    """Merge a sequence of operators into a single operator on the union subsystem.

    Each operator is embedded into the merged Hilbert space with :func:`quax.embed`
    and composed sequentially with ``@``.  Quax's operator ``@`` promotes mixed
    types automatically (requires ``rigetti-quax >= 0.6.5``), so an all-``Unitary``
    group yields a ``Unitary`` while a group containing any channel promotes to a
    ``SuperOp``.  Downstream adapters handle final conversion (e.g. to ``KrausMap``
    for trajectories).

    :param ops_with_subsystems: Ordered list of ``(operator, subsystem)`` pairs
        to merge (applied in order: first element is applied first).
    :param merged_subsystem: Sorted tuple of qubit indices for the merged operator.
    :param dims: Global per-qudit dimensions tuple.
    :return: A single ``(operator, merged_subsystem)`` pair.
    """
    target_dims = tuple(dims[q] for q in merged_subsystem)

    accumulated: FixedOp | None = None
    for op, subsystem in ops_with_subsystems:
        positions = tuple(merged_subsystem.index(q) for q in subsystem)
        embedded = qx.embed(op, target_dims=target_dims, positions=positions)
        accumulated = embedded if accumulated is None else embedded @ accumulated

    if accumulated is None:
        raise ValueError("Cannot merge an empty operation group.")
    return accumulated, merged_subsystem


class _UnionFind:
    """Simple union-find (disjoint set) data structure for node grouping."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    def make_set(self, x: int) -> None:
        self._parent[x] = x
        self._rank[x] = 0

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> int:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        return rx


def compressor_from_dag(
    dag: nx.DiGraph,
    max_subsystem_size: int,
    dims: tuple[int, ...] = (),
) -> Callable[[list[ResolvedOp]], list[ResolvedOp]]:
    """Build a compressor that merges operators via greedy edge contraction.

    The algorithm prioritises merging small gates into larger groups, which
    reduces the number of distinct subsystem shapes and therefore JIT
    compilation time.

    1. Build a priority queue of candidate edge merges sorted by resulting
       subsystem size (ascending), so 1-qubit gates are absorbed into
       neighbouring multi-qubit groups first.
    2. Greedily contract edges while the merged subsystem fits within
       ``max_subsystem_size``.
    3. Return a closure that receives the resolved operator list and produces
       a compressed operator list.

    :param dag: Program dependency DAG (nodes indexed 0..N-1, each with
        a ``"qubits"`` attribute).
    :param max_subsystem_size: Maximum number of qubits in a merged group.
        0 disables merging entirely.
    :param dims: Per-qudit dimensions tuple for embedding during merge.
    :return: A closure ``compress(ops) -> list[ResolvedOp]``.
    """
    n_original = dag.number_of_nodes()

    if max_subsystem_size == 0 or n_original == 0:
        passthrough_emit_order = [
            (nk, [nk], tuple(dag.nodes[nk]["qubits"])) for nk in nx.lexicographical_topological_sort(dag)
        ]

        def compress_passthrough(ops: list[ResolvedOp]) -> list[ResolvedOp]:
            return ops

        compress_passthrough.emit_order = passthrough_emit_order  # type: ignore[attr-defined]
        logger.info(
            "Compressor: %d ops (no merging), max_subsystem_size=0",
            n_original,
        )
        return compress_passthrough

    # --- Priority-queue based greedy edge contraction ---
    uf = _UnionFind()
    group_qubits: dict[int, set[int]] = {}

    for nk in dag.nodes:
        uf.make_set(nk)
        group_qubits[nk] = set(dag.nodes[nk]["qubits"])

    # Quotient graph over current group roots, kept in lock-step with the
    # union-find structure.  It starts as a copy of the dependency DAG and is
    # contracted whenever two groups merge.  It is the authority on whether a
    # candidate merge is *convex*: contracting two groups must not reorder any
    # operation that lies topologically between them (see ``_contraction_cycles``).
    quotient: nx.DiGraph = nx.DiGraph()
    quotient.add_nodes_from(dag.nodes)
    quotient.add_edges_from(dag.edges)

    def _contraction_cycles(root_a: int, root_b: int) -> bool:
        """Return ``True`` if merging two groups would create a cycle.

        The quotient graph is always a DAG, so contracting ``root_a`` and
        ``root_b`` introduces a cycle iff there is a directed path of length
        ``>= 2`` between them in *either* direction — i.e. some other group is
        sandwiched on a dependency path from one to the other.  Merging across
        such a node would force it to be reordered relative to the merged group,
        which is exactly what must be forbidden for any non-commuting operation.  A direct
        edge ``root_a -> root_b`` alone is fine; only an *indirect* path is a problem.
        """
        for src, dst in ((root_a, root_b), (root_b, root_a)):
            stack = [s for s in quotient.successors(src) if s != dst]
            seen = set(stack)
            while stack:
                node = stack.pop()
                if node == dst:
                    return True
                for nxt in quotient.successors(node):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
        return False

    def _contract_quotient(keep: int, drop: int) -> None:
        """Contract ``drop`` into ``keep`` in the quotient graph."""
        for pred in list(quotient.predecessors(drop)):
            if pred != keep:
                quotient.add_edge(pred, keep)
        for succ in list(quotient.successors(drop)):
            if succ != keep:
                quotient.add_edge(keep, succ)
        quotient.remove_node(drop)

    # Build initial candidate heap: (union_size, u, v)
    # Smaller union sizes are processed first.
    heap: list[tuple[int, int, int]] = []
    for u_node, v_node in dag.edges:
        union_size = len(group_qubits[u_node] | group_qubits[v_node])
        if union_size <= max_subsystem_size:
            heapq.heappush(heap, (union_size, u_node, v_node))

    while heap:
        _, u_node, v_node = heapq.heappop(heap)
        ru = uf.find(u_node)
        rv = uf.find(v_node)
        if ru == rv:
            continue
        union_qubits = group_qubits[ru] | group_qubits[rv]
        if len(union_qubits) > max_subsystem_size:
            continue
        # Reject non-convex merges: fusing two groups must not reorder anything that lies
        # topologically between them.  This is what protects operation order in general --
        # including around measurements -- so no separate barrier concept is needed.
        if _contraction_cycles(ru, rv):
            continue
        new_root = uf.union(ru, rv)
        group_qubits[new_root] = union_qubits
        old_root = rv if new_root == ru else ru
        if old_root in group_qubits:
            del group_qubits[old_root]
        _contract_quotient(new_root, old_root)

        # Re-enqueue edges from the newly merged group to its neighbours.
        for neighbour in (
            set(dag.successors(u_node))
            | set(dag.predecessors(u_node))
            | set(dag.successors(v_node))
            | set(dag.predecessors(v_node))
        ):
            rn = uf.find(neighbour)
            if rn == new_root:
                continue
            new_union_size = len(group_qubits[new_root] | group_qubits[rn])
            if new_union_size <= max_subsystem_size:
                heapq.heappush(heap, (new_union_size, u_node, neighbour))

    # --- Build merge plan ---
    # Order the *members within each group* by a lexicographical topological
    # sort of the original DAG (= program order), so ``_merge_ops`` composes
    # them in the order they appear in the program.
    topo_order = list(nx.lexicographical_topological_sort(dag))

    root_to_nodes: dict[int, list[int]] = {}
    for nk in topo_order:
        root = uf.find(nk)
        root_to_nodes.setdefault(root, []).append(nk)

    # A *merged* group's operator is built by ``_merge_ops``, which embeds its members
    # into the sorted union subsystem -- so sorted is the truth for those. A *singleton*
    # group, though, is emitted verbatim by ``compress`` below, so its operator is still
    # in the instruction's own operand order: ``CCNOT 2 1 0`` yields a matrix indexed
    # (2, 1, 0), not (0, 1, 2). Advertising a sorted subsystem for it would tell the
    # caller to apply that matrix to permuted qudits.
    #
    # The state-vector path re-derives each op's offsets inside its group
    # (``group_positions`` in ``_build_vectorized_operator_constructor``) and so was
    # correct either way, but the density-matrix path dispatches purely on the
    # subsystem advertised here -- and silently produced wrong states for any
    # non-ascending multi-qubit gate. Since a 3+ qubit gate can never merge under the
    # default ``max_subsystem_size=2``, it is always a singleton, so every reversed
    # ``CCNOT``/``CSWAP`` was affected.
    #
    # Reporting the raw operand order costs nothing (the embedding becomes the
    # identity) at the price of treating e.g. (0, 1) and (1, 0) as distinct bases,
    # i.e. one extra ``jax.lax.switch`` branch per operand ordering actually used.
    root_to_subsystem: dict[int, tuple[int, ...]] = {}
    for root, qubits in group_qubits.items():
        nodes = root_to_nodes[root]
        if len(nodes) == 1:
            root_to_subsystem[root] = tuple(dag.nodes[nodes[0]]["qubits"])
        else:
            root_to_subsystem[root] = tuple(sorted(qubits))

    # Emit the *groups* in a topological order of the **quotient** graph rather
    # than the original DAG.  A merged group can legitimately contain an op that
    # precedes a measurement in program order *together* with an op that depends on that
    # measurement — the merge is valid because the earlier op commutes with it, so it may be
    # applied afterwards.  But emitting the group at its earliest member's position (as an
    # earlier version did, by walking the original DAG) would place the *whole* group —
    # including the later op — before the measurement, silently applying post-measurement
    # gates first and corrupting the outcome.
    # A quotient topological sort respects every inter-group dependency, so a group is
    # emitted only after all groups it depends on.  The lexicographic key — each group's
    # minimum original node index — additionally keeps operations that cannot merge emitted
    # in program order: any ancestor group has a member preceding it in program order and
    # thus a strictly smaller minimum index.
    group_min_index: dict[int, int] = {}
    for nk in dag.nodes:
        root = uf.find(nk)
        if root not in group_min_index or nk < group_min_index[root]:
            group_min_index[root] = nk

    emit_order: list[tuple[int, list[int], tuple[int, ...]]] = []
    for root in nx.lexicographical_topological_sort(quotient, key=lambda r: group_min_index[r]):
        emit_order.append((root, root_to_nodes[root], root_to_subsystem[root]))

    # --- Log the compression statistics ---
    n_groups = len(emit_order)
    n_multi = sum(1 for _, nodes, _ in emit_order if len(nodes) > 1)
    subsystem_sizes = [len(sub) for _, _, sub in emit_order]
    avg_subsystem = sum(subsystem_sizes) / len(subsystem_sizes) if subsystem_sizes else 0.0
    max_sub = max(subsystem_sizes) if subsystem_sizes else 0

    logger.info(
        "Compressor: %d ops → %d groups (ratio=%.2f), "
        "%d merged groups, avg_subsystem=%.2f, max_subsystem=%d, max_subsystem_size=%d",
        n_original,
        n_groups,
        n_groups / n_original if n_original else 1.0,
        n_multi,
        avg_subsystem,
        max_sub,
        max_subsystem_size,
    )

    # --- Build compress closure ---
    def compress(ops: list[ResolvedOp]) -> list[ResolvedOp]:
        result: list[ResolvedOp] = []
        for _, nodes, subsystem in emit_order:
            if len(nodes) == 1:
                result.append(ops[nodes[0]])
            else:
                group_ops = [(ops[nk][0], ops[nk][1]) for nk in nodes]
                merged = _merge_ops(group_ops, subsystem, dims)
                result.append(merged)
        return result

    # Expose merge recipe: for each group, (nodes_in_topo_order, merged_subsystem).
    compress.emit_order = emit_order  # type: ignore[attr-defined]

    return compress
