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
"""
simulation._resolver module
----------------------------

Shared infrastructure for the density-matrix and state-vector simulators.

This module provides the three front-end stages of the simulation pipeline:

1. **Linearizer** — converts a ``MemoryMap`` into a flat JAX parameter vector.
2. **Resolver** — converts a parameter vector into a list of
   ``(operator, subsystem)`` pairs using native quax types.
3. **Adapters** — convert resolved operations into the form expected by each
   simulator backend (``SuperOp`` for density matrices; ``Unitary``/``KrausMap``/
   ``QuantumInstrument`` for state-vector trajectories).

It also provides shared utilities: DAG construction, dimension inference.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Set, Tuple, Union, cast

import jax.numpy as jnp
import networkx as nx
import quax as qx
from jax import Array

from pyquil.api import MemoryMap
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import DefCircuit, Gate, Measurement, Reset, ResetQubit

from pyquil.noise._channels import (
    Channel,
    CycleChannel,
    MeasurementChannel,
    ResetChannel,
    get_instruction_unitary,
)
from pyquil.noise._noise_model import (
    NoiseModelLike,
)
from pyquil.transform import expand_defcircuit_body

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────

# Resolved operations retain the most specific native quax type.
ResolvedOp = Tuple[Union[qx.Unitary, qx.SuperOp, qx.KrausMap, qx.QuantumInstrument], Tuple[int, ...]]
RecipeOp = Union[qx.Unitary, qx.SuperOp, qx.KrausMap, qx.QuantumInstrument]
RecipeCallable = Callable[[Array], RecipeOp]
Recipe = Tuple[Union[RecipeOp, RecipeCallable], Tuple[int, ...]]

# Trajectory operations for the state-vector simulator.
TrajectoryOp = Tuple[Union[qx.Unitary, qx.KrausMap, qx.QuantumInstrument], Tuple[int, ...]]

# Density-matrix operations.
DensityMatrixOp = Tuple[qx.SuperOp, Tuple[int, ...]]

# Custom gate definitions.
CustomGateMap = dict


# ══════════════════════════════════════════════════════════
# Linearizer
# ══════════════════════════════════════════════════════════


class Linearizer:
    """Converts a MemoryMap into a flat JAX parameter vector.

    Constructed via :func:`linearizer_from_program`. Call instances directly
    to perform the conversion::

        lin = linearizer_from_program(program)
        params = lin(memory_map)

    :param n_params: The number of scalar parameters in the vector.
    """

    __slots__ = ("_linearize_fn", "n_params")

    def __init__(self, linearize_fn: Callable[[MemoryMap], Array], n_params: int) -> None:
        self._linearize_fn = linearize_fn
        self.n_params = n_params

    def __call__(self, memory_map: MemoryMap) -> Array:
        return self._linearize_fn(memory_map)


def linearizer_from_program(program: Program) -> Linearizer:
    """Build a :class:`Linearizer` that converts a memory map to a flat JAX parameter vector.

    Walks the program to identify parameter registers (skipping ``"ro"`` and
    any register that is the target of a ``MEASURE`` instruction).  For each
    gate parameter that is a :class:`MemoryReference`, records ``(name, offset)``
    in program order.

    :param program: Expanded Quil program.
    :return: A :class:`Linearizer` instance.
    """
    # Find registers written to by MEASURE — these are output registers, not params
    measure_registers: Set[str] = set()
    for inst in program.instructions:
        if isinstance(inst, Measurement):
            cr = inst.classical_reg
            if cr is not None:
                measure_registers.add(cr.name)

    # Collect parameter references in program order
    param_refs: List[Tuple[str, int]] = []
    for inst in program.instructions:
        if isinstance(inst, Gate):
            for param in inst.params:
                if isinstance(param, MemoryReference):
                    if param.name not in measure_registers:
                        param_refs.append((param.name, param.offset))

    def linearize(memory_map: MemoryMap) -> Array:
        if not param_refs:
            return jnp.array([], dtype=float)
        values = [float(memory_map[name][offset]) for name, offset in param_refs]
        return jnp.array(values, dtype=float)

    return Linearizer(linearize, n_params=len(param_refs))


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

    def __init__(self, resolve_fn: Callable[[Array], List[ResolvedOp]], dims: Tuple[int, ...]) -> None:
        self._resolve_fn = resolve_fn
        self.dims = dims

    def __call__(self, params: Array) -> List[ResolvedOp]:
        return self._resolve_fn(params)


def _is_parameterized(inst: Gate) -> bool:
    """Check if a gate instruction has any MemoryReference parameters."""
    return any(isinstance(p, MemoryReference) for p in inst.params)


def _measure_registers(program: Program) -> Set[str]:
    """Return the set of register names that are targets of MEASURE instructions."""
    regs: Set[str] = set()
    for inst in program.instructions:
        if isinstance(inst, Measurement):
            cr = inst.classical_reg
            if cr is not None:
                regs.add(cr.name)
    return regs


def resolver_from_program(
    program: Program,
    noise_model: NoiseModelLike | None,
    qubit_indices: Dict[int, int],
    custom_gates: CustomGateMap | None,
) -> Tuple[Resolver, nx.DiGraph, List[int]]:
    """Build a :class:`Resolver`, DAG, and node order from a program.

    The resolver accepts a flat parameter vector and produces one
    ``(operator, subsystem)`` pair per DAG node, in ``node_order``.

    DEFCIRCUIT expansion is handled internally:

    * If a cycle invocation matches a :class:`CycleChannel` in the noise
      model, the cycle is expanded using the channel's constituent operators.
    * Otherwise the DEFCIRCUIT body is expanded via qubit/param substitution
      and each resulting instruction is resolved individually.

    The DAG is built simultaneously during instruction iteration.

    Operators are returned in their most specific native type:

    * Ideal gates → ``qx.Unitary``
    * Noisy gates (``Channel``) → ``qx.SuperOp``
    * Expanded cycle gates with ``CycleChannel`` noise → constituent ``qx.SuperOp``
    * Measurements → ``qx.QuantumInstrument``
    * Noisy resets (``ResetChannel``) → ``qx.SuperOp``
    * Ideal resets → ``qx.SuperOp``

    No type conversion (``to_kraus``, ``to_superop``) is performed here;
    that is the adapter's responsibility.

    :param program: Quil program (may contain DEFCIRCUITs).
    :param noise_model: Optional noise model.
    :param qubit_indices: Mapping from physical qubit id → 0-based index.
    :param custom_gates: Custom gate definitions.
    :return: Tuple of ``(Resolver, dag, node_order)``.
    """
    measure_regs = _measure_registers(program)

    # Extract DEFCIRCUIT definitions.
    circuit_definitions: Dict[str, DefCircuit] = {}
    for inst in program.instructions:
        if isinstance(inst, DefCircuit):
            circuit_definitions[inst.name] = inst

    # ── Expand instructions, building DAG and recipes in one pass ──

    dag = nx.DiGraph()
    node_order: List[int] = []
    last_on_qubit: Dict[int, int] = {}  # qubit_index → last node key

    # Flat lists populated during instruction iteration.
    expanded_insts: List[Gate | Measurement | ResetQubit | Reset] = []
    expanded_channels: List[Channel | MeasurementChannel | None] = []

    def _emit(inst: Gate | Measurement | ResetQubit | Reset, channel: Channel | MeasurementChannel | None = None) -> None:
        """Emit an instruction: add a DAG node and record the channel."""
        if isinstance(inst, Gate):
            qubits = tuple(qubit_indices[q] for q in inst.get_qubit_indices())
        elif isinstance(inst, Measurement):
            qubits = tuple(qubit_indices[q] for q in inst.get_qubit_indices())
        elif isinstance(inst, ResetQubit):
            qubits = tuple(qubit_indices[q] for q in inst.get_qubit_indices())  # type: ignore[union-attr]
        else:  # Reset
            qubits = tuple(sorted(qubit_indices.values()))
        node_key = len(expanded_insts)
        dag.add_node(node_key, inst=inst, qubits=qubits)
        node_order.append(node_key)
        for q in qubits:
            if q in last_on_qubit:
                dag.add_edge(last_on_qubit[q], node_key)
            last_on_qubit[q] = node_key
        expanded_insts.append(inst)
        expanded_channels.append(channel)

    def _lookup_and_emit(inst: Gate | Measurement | ResetQubit | Reset) -> None:
        """Look up noise channel for an instruction and emit it."""
        if isinstance(inst, Gate):
            ch = noise_model.get_channel(inst) if noise_model is not None else None
            if isinstance(ch, CycleChannel):
                ch = None
            _emit(inst, ch)
        elif isinstance(inst, Measurement):
            ch = noise_model.get_channel(inst) if noise_model is not None else None
            _emit(inst, ch if isinstance(ch, MeasurementChannel) else None)
        else:
            _emit(inst)

    # ── Main instruction loop ──

    for inst in program.instructions:
        if isinstance(inst, DefCircuit):
            continue

        if isinstance(inst, Gate) and inst.name in circuit_definitions:
            # DEFCIRCUIT invocation — check for CycleChannel.
            channel = noise_model.get_channel(inst) if noise_model is not None else None

            if isinstance(channel, CycleChannel):
                # Expand using constituent channels.
                for sub_ch in channel.channels:
                    _emit(sub_ch.inst, sub_ch)
            else:
                # No CycleChannel — expand the DEFCIRCUIT body and resolve individually.
                for expanded_inst in expand_defcircuit_body(inst, circuit_definitions[inst.name], circuit_definitions):
                    _lookup_and_emit(expanded_inst)
        elif isinstance(inst, (Gate, Measurement, ResetQubit, Reset)):
            _lookup_and_emit(inst)

    # ── Build recipes from expanded instructions ──

    # Assign parameter vector indices to each gate's MemoryReference params.
    param_counter = 0
    gate_param_indices: Dict[int, List[int]] = {}
    for idx in node_order:
        inst = expanded_insts[idx]
        if isinstance(inst, Gate):
            indices = []
            for param in inst.params:
                if isinstance(param, MemoryReference) and param.name not in measure_regs:
                    indices.append(param_counter)
                    param_counter += 1
                else:
                    indices.append(-1)
            gate_param_indices[idx] = indices

    # Pre-scan gate instructions to infer per-qudit dimensions.
    qudit_dims: Dict[int, int] = {}  # qubit_index → dimension
    for node_key in node_order:
        inst = expanded_insts[node_key]
        if isinstance(inst, Gate):
            subsystem = dag.nodes[node_key]["qubits"]
            channel = expanded_channels[node_key]
            if channel is None and noise_model is not None:
                channel = noise_model.get_channel(inst)
            if channel is not None and isinstance(channel, Channel):
                op_dims = channel.process.dims[0]
            elif channel is not None and isinstance(channel, CycleChannel):
                continue
            else:
                try:
                    unitary = get_instruction_unitary(inst, custom_gates=custom_gates)
                    op_dims = unitary.dims[0]
                except Exception:
                    continue
            for slot, dim in zip(subsystem, op_dims):
                if dim > qudit_dims.get(slot, 2):
                    qudit_dims[slot] = dim

    recipes: List[Recipe] = []

    for node_key in node_order:
        inst = expanded_insts[node_key]
        subsystem = dag.nodes[node_key]["qubits"]

        match inst:
            case Gate():
                channel = expanded_channels[node_key]
                if channel is None and noise_model is not None:
                    channel = noise_model.get_channel(inst)

                if channel is not None and isinstance(channel, Channel):
                    recipes.append((channel.process, subsystem))
                elif channel is not None and isinstance(channel, MeasurementChannel):
                    raise ValueError(f"MeasurementChannel cannot be applied to expanded gate {inst}.")
                elif channel is not None and isinstance(channel, CycleChannel):
                    raise ValueError(f"CycleChannel for {inst.name} was not expanded before resolver construction.")
                elif _is_parameterized(inst):
                    gate_name = inst.name
                    if custom_gates is not None and gate_name in custom_gates:
                        gate_def = custom_gates[gate_name]
                    elif gate_name in qx.gates.QUANTUM_GATES:
                        gate_def = qx.gates.QUANTUM_GATES[gate_name]
                    else:
                        raise KeyError(f"Unknown gate '{gate_name}'.")
                    pidx = gate_param_indices[node_key]
                    cparams = list(inst.params)

                    def _make_param_recipe(
                        gdef: object,
                        cp: list,
                        pi: List[int],
                    ) -> Callable[[Array], qx.Unitary]:
                        def recipe(params: Array) -> qx.Unitary:
                            resolved = []
                            for p, pv in zip(cp, pi):
                                if pv >= 0:
                                    resolved.append(params[pv])
                                else:
                                    resolved.append(float(p.real) if hasattr(p, "real") else float(p))
                            result = gdef(*resolved) if callable(gdef) else gdef  # type: ignore[operator]
                            if not isinstance(result, qx.Unitary):
                                result = cast(Any, result)
                                result = qx.Unitary.from_matrix(result.matrix, result.dims)
                            return result

                        return recipe

                    recipes.append((_make_param_recipe(gate_def, cparams, pidx), subsystem))

                else:
                    unitary = get_instruction_unitary(inst, custom_gates=custom_gates)
                    recipes.append((unitary, subsystem))

            case Measurement():
                meas_channel = expanded_channels[node_key]
                if meas_channel is None and noise_model is not None:
                    meas_channel = noise_model.get_channel(inst)
                if meas_channel is not None and isinstance(meas_channel, MeasurementChannel):
                    recipes.append((meas_channel.process, subsystem))
                elif meas_channel is not None and isinstance(meas_channel, Channel):
                    raise ValueError(f"Channel cannot be applied to expanded measurement {inst}.")
                else:
                    dim = qudit_dims.get(subsystem[0], 2)
                    recipes.append((qx.gates.MEASURE(dim=dim), subsystem))

            case ResetQubit():
                reset_channel = None
                if noise_model is not None:
                    reset_channel = noise_model.get_channel(inst)
                if reset_channel is not None and isinstance(reset_channel, ResetChannel):
                    recipes.append((reset_channel.process, subsystem))
                else:
                    dim = qudit_dims.get(subsystem[0], 2)
                    recipes.append((qx.gates.RESET(dim=dim), subsystem))

            case Reset():
                for _, q_idx in sorted(qubit_indices.items()):
                    dim = qudit_dims.get(q_idx, 2)
                    recipes.append((qx.gates.RESET(dim=dim), (q_idx,)))

    def resolve(params: Array) -> List[ResolvedOp]:
        ops: List[ResolvedOp] = []
        for op_or_fn, subsystem in recipes:
            if isinstance(op_or_fn, (qx.Unitary, qx.KrausMap, qx.SuperOp, qx.QuantumInstrument)):
                ops.append((op_or_fn, subsystem))
            else:
                ops.append((op_or_fn(params), subsystem))
        return ops

    # Compute per-qudit dimensions from the pre-scan.
    n_qubits = len(qubit_indices)
    dims = tuple(qudit_dims.get(i, 2) for i in range(n_qubits))

    return Resolver(resolve, dims=dims), dag, node_order


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
    ops: List[ResolvedOp],
) -> List[DensityMatrixOp]:
    """Convert resolved operations to ``(SuperOp, subsystem)`` pairs for density-matrix simulation.

    * ``Unitary`` → ``qx.to_superop(op)``
    * ``SuperOp`` → pass through
    * ``KrausMap`` → ``qx.to_superop(op)``
    * ``QuantumInstrument`` → ``qx.to_superop(op.total_channel())``

    :param ops: Resolved operations from :func:`build_resolver`.
    :return: List of ``(SuperOp, subsystem)`` pairs.
    """
    result: List[DensityMatrixOp] = []
    for op, subsystem in ops:
        if isinstance(op, qx.SuperOp):
            result.append((op, subsystem))
        elif isinstance(op, qx.QuantumInstrument):
            result.append((qx.to_superop(op.total_channel()), subsystem))
        else:
            # Unitary, KrausMap
            result.append((qx.to_superop(op), subsystem))
    return result


def adapt_for_trajectory(
    ops: List[ResolvedOp],
    kraus_truncation_threshold: float = 1e-6,
) -> List[TrajectoryOp]:
    """Convert resolved operations to trajectory-compatible types.

    * ``Unitary`` → pass through
    * ``SuperOp`` → ``truncate_kraus(to_kraus(op))`` → ``KrausMap``
    * ``KrausMap`` → pass through
    * ``QuantumInstrument`` → pass through

    :param ops: Resolved operations from :func:`build_resolver`.
    :param kraus_truncation_threshold: Threshold for Kraus truncation.
    :return: List of ``(Unitary | KrausMap | QuantumInstrument, subsystem)`` pairs.
    """
    result: List[TrajectoryOp] = []
    for op, subsystem in ops:
        if isinstance(op, qx.SuperOp):
            km = qx.truncate_kraus(qx.to_kraus(op), atol=kraus_truncation_threshold)
            result.append((km, subsystem))
        else:
            # Unitary, KrausMap, QuantumInstrument — pass through
            result.append((op, subsystem))  # type: ignore[arg-type]
    return result


# ══════════════════════════════════════════════════════════
# Compressor (greedy edge contraction)
# ══════════════════════════════════════════════════════════


def _merge_ops(
    ops_with_subsystems: List[ResolvedOp],
    merged_subsystem: Tuple[int, ...],
    dims: Tuple[int, ...],
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

    assert accumulated is not None
    return accumulated, merged_subsystem


class _UnionFind:
    """Simple union-find (disjoint set) data structure for node grouping."""

    def __init__(self) -> None:
        self._parent: Dict[int, int] = {}
        self._rank: Dict[int, int] = {}

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
    node_order: List[int],
    max_subsystem_size: int,
    dims: Tuple[int, ...] = (),
) -> Callable[[List[ResolvedOp]], List[ResolvedOp]]:
    """Build a compressor that merges operators via greedy edge contraction.

    The algorithm:
    1. Classify each node as *mergeable* (gate, reset) or *barrier* (measurement).
    2. Greedily contract DAG edges: for each edge ``(u, v)`` in topological
       order, merge the groups of ``u`` and ``v`` if both are mergeable and
       the union of their qubit sets fits within ``max_subsystem_size``.
    3. Build a merge plan mapping each group to its constituent nodes and
       merged qubit subsystem.
    4. Return a closure that receives the resolved operator list and produces
       a compressed operator list.

    :param dag: Program dependency DAG.
    :param node_order: Node keys in instruction order.
    :param max_subsystem_size: Maximum number of qubits in a merged group.
        0 disables merging entirely.
    :return: A closure ``compress(ops) -> List[ResolvedOp]``.
    """
    n_original = len(node_order)

    if max_subsystem_size == 0 or n_original == 0:
        # No merging — pass through
        def compress_passthrough(ops: List[ResolvedOp]) -> List[ResolvedOp]:
            return ops

        logger.info(
            "Compressor: %d ops (no merging), max_subsystem_size=0",
            n_original,
        )
        return compress_passthrough

    def _is_mergeable(node_key: int) -> bool:
        inst = dag.nodes[node_key]["inst"]
        return isinstance(inst, (Gate, Measurement, ResetQubit, Reset)) and not isinstance(inst, Measurement)

    # --- Greedy edge contraction ---
    uf = _UnionFind()
    group_qubits: Dict[int, Set[int]] = {}  # root → set of qubit indices

    for nk in node_order:
        uf.make_set(nk)
        group_qubits[nk] = set(dag.nodes[nk]["qubits"])

    topo_order = list(nx.topological_sort(dag))

    for u_node in topo_order:
        for v_node in dag.successors(u_node):
            ru = uf.find(u_node)
            rv = uf.find(v_node)
            if ru == rv:
                continue
            if not _is_mergeable(u_node) or not _is_mergeable(v_node):
                continue
            union_qubits = group_qubits[ru] | group_qubits[rv]
            if len(union_qubits) > max_subsystem_size:
                continue
            new_root = uf.union(ru, rv)
            group_qubits[new_root] = union_qubits
            # Clean up the non-root entry
            old_root = rv if new_root == ru else ru
            if old_root in group_qubits:
                del group_qubits[old_root]

    # --- Build merge plan ---
    root_to_nodes: Dict[int, List[int]] = {}
    for nk in topo_order:
        root = uf.find(nk)
        root_to_nodes.setdefault(root, []).append(nk)

    root_to_subsystem: Dict[int, Tuple[int, ...]] = {}
    for root, qubits in group_qubits.items():
        root_to_subsystem[root] = tuple(sorted(qubits))

    node_key_to_idx: Dict[int, int] = {nk: i for i, nk in enumerate(node_order)}

    emit_order: List[Tuple[int, List[int], Tuple[int, ...]]] = []
    emitted_roots: Set[int] = set()
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
    def compress(ops: List[ResolvedOp]) -> List[ResolvedOp]:
        result: List[ResolvedOp] = []
        for _, nodes, subsystem in emit_order:
            if len(nodes) == 1:
                idx = node_key_to_idx[nodes[0]]
                result.append(ops[idx])
            else:
                group_ops = [(ops[node_key_to_idx[nk]][0], ops[node_key_to_idx[nk]][1]) for nk in nodes]
                merged = _merge_ops(group_ops, subsystem, dims)
                result.append(merged)
        return result

    return compress


# ══════════════════════════════════════════════════════════
# Dimension inference
# ══════════════════════════════════════════════════════════


def infer_qudit_dims(
    operations: List[ResolvedOp] | List[TrajectoryOp] | List[DensityMatrixOp],
    n_qudits: int,
) -> Tuple[int, ...]:
    """Infer per-qudit dimensions from resolved operations.

    Starts with all registers at dimension 2 (qubit).  For each operation,
    checks the operator's dims and upgrades any slot whose operator dimension
    exceeds the current assignment.

    :param operations: Resolved list of ``(operator, subsystem)`` pairs.
    :param n_qudits: Number of qudit slots.
    :return: Tuple of per-qudit dimensions, e.g. ``(2, 3, 2)``.
    """
    qudit_dims: List[int] = [2] * n_qudits
    for op, subsystem in operations:
        # All quax operators expose dims as ((out_dims), (in_dims))
        op_dims = op.dims[0] if hasattr(op, "dims") else None
        if op_dims is None:
            continue
        for slot, dim in zip(subsystem, op_dims):
            if dim > qudit_dims[slot]:
                qudit_dims[slot] = dim
    return tuple(qudit_dims)
