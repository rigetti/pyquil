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
from collections.abc import Callable, Iterator
from copy import deepcopy
from typing import Any, TypeAlias, cast

import networkx as nx
import quax as qx
from jax import Array

from pyquil.noise._channels import (
    Channel,
    CycleChannel,
    MeasurementChannel,
    ResetChannel,
    get_custom_gates_from_program,
    get_instruction_unitary,
)
from pyquil.noise._noise_model import (
    NoiseModelLike,
)
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, substitute
from pyquil.quilbase import DefCircuit, Gate, Measurement, Reset, ResetQubit

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────

# A fixed (non-parameterized) operator — the most specific native quax type.
FixedOp: TypeAlias = qx.Unitary | qx.SuperOp | qx.KrausMap | qx.QuantumInstrument

# An expanded item is either a fixed operator or a callable that resolves
# parameters into a Unitary (only parameterized gates produce callables).
ExpandedOp: TypeAlias = FixedOp | Callable[[Array], qx.Unitary]

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
    qarg_to_arg_map = {qarg: q for q, qarg in zip(inst.qubits, defcircuit.qubit_variables, strict=False)}
    parg_to_arg_map = {parg: param for param, parg in zip(inst.params, defcircuit.parameters, strict=False)}

    for circuit_inst in defcircuit.instructions:
        if isinstance(circuit_inst, Gate):
            circuit_inst = deepcopy(circuit_inst)
            circuit_inst.qubits = [qarg_to_arg_map[qarg] for qarg in circuit_inst.qubits]  # type: ignore[index,misc]
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
            circuit_inst.qubit = qarg_to_arg_map[circuit_inst.qubit]  # type: ignore[index]
            yield circuit_inst
        elif isinstance(circuit_inst, ResetQubit):
            circuit_inst = deepcopy(circuit_inst)
            circuit_inst.qubit = qarg_to_arg_map[circuit_inst.qubit]  # type: ignore[index]
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
    :return: Tuple of ``(ops, qubit_tuples, param_refs)`` where each op is
        either a concrete quax operator or a ``Callable[[Array], Unitary]``
        for parameterized gates, each qubit tuple contains physical qubit
        IDs, and ``param_refs`` is a list of ``(register_name, offset)``
        pairs for each scalar parameter in program order.
    """
    # Derive circuit definitions and custom gates from the program.
    circuit_definitions: dict[str, DefCircuit] = {}
    for inst in program.instructions:
        if isinstance(inst, DefCircuit):
            circuit_definitions[inst.name] = inst

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

        # Check noise model first.
        channel = noise_model.get_channel(inst) if noise_model is not None else None
        if isinstance(channel, Channel):
            return channel.process, qubits
        if isinstance(channel, CycleChannel):
            raise ValueError(f"CycleChannel for {inst.name} was not expanded before gate resolution.")

        # Parameterized gate → callable that resolves params at call time.
        if any(isinstance(p, MemoryReference) for p in inst.params):
            gate_name = inst.name
            if custom_gates is not None and gate_name in custom_gates:
                gate_def = custom_gates[gate_name]
            elif gate_name in qx.gates.QUANTUM_GATES:
                gate_def = qx.gates.QUANTUM_GATES[gate_name]
            else:
                raise KeyError(f"Unknown gate '{gate_name}'.")

            param_indices: list[int] = []
            concrete_params = list(inst.params)
            for p in inst.params:
                if isinstance(p, MemoryReference) and p.name not in measure_regs:
                    param_indices.append(param_counter)
                    param_refs.append((p.name, p.offset))
                    param_counter += 1
                else:
                    param_indices.append(-1)

            def _make_param_callable(
                gdef: object,
                cp: list,
                pi: list[int],
            ) -> Callable[[Array], qx.Unitary]:
                def resolve_params(params: Array) -> qx.Unitary:
                    resolved: list[Any] = []
                    for p, pv in zip(cp, pi, strict=False):
                        if pv >= 0:
                            resolved.append(params[pv])
                        else:
                            resolved.append(float(p.real) if hasattr(p, "real") else float(p))
                    result = gdef(*resolved) if callable(gdef) else gdef
                    if not isinstance(result, qx.Unitary):
                        result = cast(Any, result)
                        result = qx.Unitary.from_matrix(result.matrix, result.dims)
                    return result

                return resolve_params

            return _make_param_callable(gate_def, concrete_params, param_indices), qubits

        # Fixed gate → resolve to Unitary now.
        unitary = get_instruction_unitary(inst, custom_gates=custom_gates)
        return unitary, qubits

    def _resolve_measurement(inst: Measurement) -> tuple[FixedOp, tuple[int, ...]]:
        """Resolve a measurement instruction."""
        qubits = tuple(inst.get_qubit_indices())
        channel = noise_model.get_channel(inst) if noise_model is not None else None
        if isinstance(channel, MeasurementChannel):
            return channel.process, qubits
        return qx.gates.MEASURE(), qubits

    def _resolve_reset_qubit(inst: ResetQubit) -> tuple[FixedOp, tuple[int, ...]]:
        """Resolve a targeted reset instruction."""
        qubits = tuple(inst.get_qubit_indices())  # type: ignore[arg-type]
        channel = noise_model.get_channel(inst) if noise_model is not None else None
        if isinstance(channel, ResetChannel):
            return channel.process, qubits
        return qx.gates.RESET(), qubits

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
                for q in all_qubits:
                    _emit_op(qx.gates.RESET(), (q,))

    for inst in program.instructions:
        if isinstance(inst, DefCircuit):
            continue

        if isinstance(inst, Gate) and inst.name in circuit_definitions:
            channel = noise_model.get_channel(inst) if noise_model is not None else None

            if isinstance(channel, CycleChannel):
                # Expand using the channel's constituent operators.
                for sub_ch in channel.channels:
                    sub_qubits = tuple(sub_ch.inst.get_qubit_indices())
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
    return [tuple(qubit_indices[q] for q in qubits) for qubits in qubit_tuples]


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


# ══════════════════════════════════════════════════════════
# Resolver
# ══════════════════════════════════════════════════════════


class Resolver:
    """Resolves a flat parameter vector into a list of (operator, subsystem) pairs.

    Constructed via :func:`resolver_from_program`. Call instances directly::

        resolver = resolver_from_program(program, ...)
        ops = resolver(params)

    :param dims: Inferred per-qudit dimensions (e.g. ``(2, 2, 3)``).
    """

    __slots__ = ("_resolve_fn", "dims")

    def __init__(self, resolve_fn: Callable[[Array], list[ResolvedOp]], dims: tuple[int, ...]) -> None:
        self._resolve_fn = resolve_fn
        self.dims = dims

    def __call__(self, params: Array) -> list[ResolvedOp]:
        return self._resolve_fn(params)


def resolver_from_program(
    program: Program,
    noise_model: NoiseModelLike | None = None,
    qubits: list[int] | None = None,
) -> tuple[Resolver, nx.DiGraph]:
    """Build a :class:`Resolver` and dependency DAG from a program.

    The resolver accepts a flat parameter vector and produces one
    ``(operator, subsystem)`` pair per operation, in program order.

    Operators are returned in their most specific native type:

    * Ideal gates → ``qx.Unitary``
    * Noisy gates (``Channel``) → ``qx.SuperOp``
    * Expanded cycle gates with ``CycleChannel`` noise → constituent ``qx.SuperOp``
    * Measurements → ``qx.QuantumInstrument``
    * Noisy resets (``ResetChannel``) → ``qx.SuperOp``
    * Ideal resets → ``qx.SuperOp``

    Custom gate definitions (DEFGATE) and circuit definitions (DEFCIRCUIT)
    are derived from the program automatically.

    :param program: Quil program (may contain DEFCIRCUITs and DEFGATEs).
    :param noise_model: Optional noise model.
    :param qubits: Optional explicit qubit list. If ``None``, inferred from
        the program. Use this when the simulator knows about qubits that
        don't appear in the program.
    :return: Tuple of ``(Resolver, dag)``.
    """
    # Phase 1: Expand into flat operators + physical qubit tuples.
    expanded_ops, phys_qubits, _param_refs = expand_program(program, noise_model)

    # Phase 2: Remap physical qubits to 0-based indices.
    if qubits is None:
        qubits = sorted(program.get_qubit_indices())
    qubit_indices = {q: i for i, q in enumerate(qubits)}
    mapped_qubits = remap_qubits(phys_qubits, qubit_indices)

    # Phase 3: Build dependency DAG.
    dag = build_dag(mapped_qubits)

    # Phase 4: Build the resolve closure.
    frozen_ops = list(zip(expanded_ops, mapped_qubits, strict=False))

    def resolve(params: Array) -> list[ResolvedOp]:
        return [
            (cast(Callable[[Array], qx.Unitary], item)(params) if callable(item) else item, subsystem)
            for item, subsystem in frozen_ops
        ]

    n_qubits = len(qubits)
    dims = (2,) * n_qubits

    return Resolver(resolve, dims=dims), dag


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
        match op:
            case qx.SuperOp():
                result.append((op, subsystem))
            case qx.QuantumInstrument():
                result.append((qx.to_superop(op.total_channel()), subsystem))
            case qx.Unitary() | qx.KrausMap():
                result.append((qx.to_superop(op), subsystem))
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

    Each operator is embedded into the merged Hilbert space and then composed
    sequentially using the ``@`` operator, which handles type promotion
    automatically (Unitary, SuperOp, KrausMap).

    For groups containing only ``Unitary`` operators, the result is a ``Unitary``.
    For groups containing any noisy operator (``SuperOp``, ``KrausMap``), all
    operators are promoted to ``SuperOp``, composed, and the result is returned
    as a ``SuperOp``.  Downstream adapters handle final conversion (e.g. to
    ``KrausMap`` for trajectories).

    :param ops_with_subsystems: Ordered list of ``(operator, subsystem)`` pairs
        to merge (applied in order: first element is applied first).
    :param merged_subsystem: Sorted tuple of qubit indices for the merged operator.
    :param dims: Global per-qudit dimensions tuple.
    :return: A single ``(operator, merged_subsystem)`` pair.
    """
    has_noisy = any(isinstance(op, (qx.KrausMap, qx.SuperOp)) for op, _ in ops_with_subsystems)

    target_dims = tuple(dims[q] for q in merged_subsystem)

    accumulated = None
    for op, subsystem in ops_with_subsystems:
        positions = tuple(merged_subsystem.index(q) for q in subsystem)

        if has_noisy:
            embedded = qx.embed(qx.to_superop(op), target_dims=target_dims, positions=positions)
        else:
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
    *,
    barrier_nodes: set[int] | None = None,
) -> Callable[[list[ResolvedOp]], list[ResolvedOp]]:
    """Build a compressor that merges operators via greedy edge contraction.

    The algorithm prioritises merging small gates into larger groups, which
    reduces the number of distinct subsystem shapes and therefore JIT
    compilation time.

    1. Classify each node as *mergeable* or *barrier* (e.g. measurement).
       Barrier nodes are never merged.
    2. Build a priority queue of candidate edge merges sorted by resulting
       subsystem size (ascending), so 1-qubit gates are absorbed into
       neighbouring multi-qubit groups first.
    3. Greedily contract edges while the merged subsystem fits within
       ``max_subsystem_size``.
    4. Return a closure that receives the resolved operator list and produces
       a compressed operator list.

    :param dag: Program dependency DAG (nodes indexed 0..N-1, each with
        a ``"qubits"`` attribute).
    :param max_subsystem_size: Maximum number of qubits in a merged group.
        0 disables merging entirely.
    :param dims: Per-qudit dimensions tuple for embedding during merge.
    :param barrier_nodes: Optional set of node indices that should not be
        merged (e.g. measurement nodes). If ``None``, all nodes are mergeable.
    :return: A closure ``compress(ops) -> list[ResolvedOp]``.
    """
    n_original = dag.number_of_nodes()

    if max_subsystem_size == 0 or n_original == 0:

        def compress_passthrough(ops: list[ResolvedOp]) -> list[ResolvedOp]:
            return ops

        logger.info(
            "Compressor: %d ops (no merging), max_subsystem_size=0",
            n_original,
        )
        return compress_passthrough

    if barrier_nodes is None:
        barrier_nodes = set()

    # --- Priority-queue based greedy edge contraction ---
    uf = _UnionFind()
    group_qubits: dict[int, set[int]] = {}

    for nk in dag.nodes:
        uf.make_set(nk)
        group_qubits[nk] = set(dag.nodes[nk]["qubits"])

    # Build initial candidate heap: (union_size, u, v)
    # Smaller union sizes are processed first.
    heap: list[tuple[int, int, int]] = []
    for u_node, v_node in dag.edges:
        if u_node in barrier_nodes or v_node in barrier_nodes:
            continue
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
        new_root = uf.union(ru, rv)
        group_qubits[new_root] = union_qubits
        old_root = rv if new_root == ru else ru
        if old_root in group_qubits:
            del group_qubits[old_root]

        # Re-enqueue edges from the newly merged group to its neighbours.
        for neighbour in (
            set(dag.successors(u_node))
            | set(dag.predecessors(u_node))
            | set(dag.successors(v_node))
            | set(dag.predecessors(v_node))
        ):
            rn = uf.find(neighbour)
            if rn == new_root or neighbour in barrier_nodes:
                continue
            new_union_size = len(group_qubits[new_root] | group_qubits[rn])
            if new_union_size <= max_subsystem_size:
                heapq.heappush(heap, (new_union_size, u_node, neighbour))

    # --- Build merge plan ---
    topo_order = list(nx.topological_sort(dag))

    root_to_nodes: dict[int, list[int]] = {}
    for nk in topo_order:
        root = uf.find(nk)
        root_to_nodes.setdefault(root, []).append(nk)

    root_to_subsystem: dict[int, tuple[int, ...]] = {}
    for root, qubits in group_qubits.items():
        root_to_subsystem[root] = tuple(sorted(qubits))

    emit_order: list[tuple[int, list[int], tuple[int, ...]]] = []
    emitted_roots: set[int] = set()
    for nk in topo_order:
        root = uf.find(nk)
        if root not in emitted_roots:
            emitted_roots.add(root)
            nodes = root_to_nodes[root]
            subsystem = root_to_subsystem[root]
            emit_order.append((root, nodes, subsystem))

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

    return compress
