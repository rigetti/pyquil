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
"""Unified program simulators backed by quax.

Three simulators share a common preprocessing pipeline:

* :class:`PureStateVectorSimulator` — gate-only programs (no noise,
  measurements, or resets).  Jit- and grad-friendly.
* :class:`DensityMatrixSimulator` — any program, optionally with noise.
  Jit- and grad-friendly.
* :class:`TrajectorySimulator` — Monte Carlo trajectory simulation for
  programs with measurements and resets, optionally with noise.

Each simulator is constructed from a :class:`~pyquil.quil.Program` and
exposes ``linearize``, ``resolve``, ``compress``, and ``compute`` methods.
The ``compute`` method is the main entry point and can be passed directly
to ``jax.jit`` or ``jax.grad``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import quax as qx
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from quax._apply import _sample_kraus_map_trajectory

from pyquil.api import MemoryMap
from pyquil.noise._noise_model import NoiseModelLike
from pyquil.quil import Program
from pyquil.quilbase import Measurement, Reset, ResetQubit
from pyquil.simulation._resolver import (
    ParametricGate,
    ResolvedOp,
    TrajectoryOp,
    adapt_for_density_matrix,
    adapt_for_trajectory,
    build_dag,
    compressor_from_dag,
    expand_program,
    remap_qubits,
)


def _pad_matrix(mat: Array, *target: int) -> Array:
    """Zero-pad the trailing dimensions of *mat* up to *target* sizes.

    Only the last ``len(target)`` axes are padded (top-left aligned); any
    leading (ensemble/stack) axes are left untouched.
    """
    if all(mat.shape[-len(target) + i] == t for i, t in enumerate(target)):
        return mat
    pad = [(0, 0)] * (mat.ndim - len(target)) + [
        (0, t - mat.shape[mat.ndim - len(target) + i]) for i, t in enumerate(target)
    ]
    return jnp.pad(mat, pad)


def _make_unitary_branch(
    base: tuple[int, ...],
    base_dims: tuple[int, ...],
    db: int,
) -> Callable[[Array, qx.StateVector], qx.StateVector]:
    """Build a ``jax.lax.switch`` branch that applies a unitary on *base*."""

    def branch(op_mat: Array, psi: qx.StateVector) -> qx.StateVector:
        unitary = qx.Unitary.from_matrix(op_mat[:db, :db], (base_dims, base_dims))
        return qx.targeted_apply_unitary(unitary, psi, base)

    return branch


def _make_superop_branch(
    base: tuple[int, ...],
    base_dims: tuple[int, ...],
    db2: int,
) -> Callable[[Array, qx.DensityMatrix], qx.DensityMatrix]:
    """Build a ``jax.lax.switch`` branch that applies a superoperator on *base*."""

    def branch(op_mat: Array, rho: qx.DensityMatrix) -> qx.DensityMatrix:
        superop = qx.SuperOp.from_matrix(op_mat[:db2, :db2], (base_dims, base_dims))
        return qx.targeted_apply_superop(superop, rho, base)

    return branch


def _make_kraus_trajectory_branch(
    base: tuple[int, ...],
    base_dims: tuple[int, ...],
    db: int,
) -> Callable[[Array, qx.StateVector, Array], tuple[qx.StateVector, Array]]:
    """Build a ``jax.lax.switch`` branch that samples a Kraus trajectory on *base*."""

    def branch(op_mat: Array, psi: qx.StateVector, key: Array) -> tuple[qx.StateVector, Array]:
        kraus_map = qx.KrausMap.from_matrix(op_mat[:, :db, :db], (base_dims, base_dims))
        return _sample_kraus_map_trajectory(kraus_map, psi, key, base)

    return branch


def _infer_dims(resolved: list[ResolvedOp], n_qubits: int) -> tuple[int, ...]:
    """Infer per-qudit dimensions from a resolved operation list.

    Each operator carries its own per-qudit input dimensions via
    ``op.dims[1]``.  A slot's dimension is the maximum dimension seen across
    every operation acting on it, defaulting to ``2`` (qubit) for slots that
    no operation touches.

    :param resolved: Resolved ``(operator, subsystem)`` pairs.
    :param n_qubits: Number of qudit slots.
    :return: Inferred per-qudit dimensions tuple.
    """
    dims = [2] * n_qubits
    for op, subsystem in resolved:
        op_dims = op.dims[1]
        for q, d in zip(subsystem, op_dims, strict=False):
            dims[q] = max(dims[q], d)
    return tuple(dims)


# ══════════════════════════════════════════════════════════
# Base class
# ══════════════════════════════════════════════════════════


class ProgramSimulator:
    """Base class for program simulators.

    Handles all shared preprocessing: circuit expansion, qubit ordering,
    building the linearizer, resolver, and compressor closures.

    Subclasses override :meth:`_validate` and :meth:`compute`.

    Instances are immutable after construction.
    """

    __slots__ = ("n_qubits", "qubits", "dims", "_linearize_fn", "_resolve_fn", "_compress_fn",
                 "bases", "op_index", "base_dims", "base_total_dim", "d_max", "_has_params",
                 "_expanded_ops", "_raw_subsystems")

    def __init__(
        self,
        program: Program,
        qubits: list[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 2,
        dims: tuple[int, ...] | None = None,
    ) -> None:
        self._validate(program)

        if qubits is None:
            qubits = sorted(program.get_qubit_indices())
        self.qubits = qubits
        self.n_qubits = len(qubits)

        qubit_indices = {q: i for i, q in enumerate(qubits)}

        # First expand with default ideal measurement/reset dimensions so we can
        # infer the register dimensions from gates and noisy channels.
        if dims is None:
            initial_expanded_ops, initial_phys_qubits, initial_param_refs = expand_program(
                program,
                noise_model,
            )
            initial_mapped_qubits = remap_qubits(initial_phys_qubits, qubit_indices)
            initial_frozen_ops = list(zip(initial_expanded_ops, initial_mapped_qubits, strict=False))

            def initial_resolve(params: Array) -> list[ResolvedOp]:
                return [
                    (cast(Callable[[Array], qx.Unitary], item)(params) if callable(item) else item, subsystem)
                    for item, subsystem in initial_frozen_ops
                ]

            self.dims = _infer_dims(initial_resolve(jnp.zeros(len(initial_param_refs))), self.n_qubits)
        else:
            self.dims = dims

        # Expand again with inferred dimensions so ideal measurement/reset
        # instruments are not stuck at qubit dimension.
        qubit_dimensions = {q: self.dims[i] for q, i in qubit_indices.items()}
        expanded_ops, phys_qubits, param_refs = expand_program(
            program,
            noise_model,
            qubit_dimensions=qubit_dimensions,
        )
        mapped_qubits = remap_qubits(phys_qubits, qubit_indices)
        dag = build_dag(mapped_qubits)
        frozen_ops = list(zip(expanded_ops, mapped_qubits, strict=False))

        def resolve(params: Array) -> list[ResolvedOp]:
            return [
                (cast(Callable[[Array], qx.Unitary], item)(params) if callable(item) else item, subsystem)
                for item, subsystem in frozen_ops
            ]

        # Build linearizer from parameter references discovered during expansion.
        def linearize(memory_map: MemoryMap) -> Array:
            if not param_refs:
                return jnp.array([], dtype=float)
            values = [float(memory_map[name][offset]) for name, offset in param_refs]
            return jnp.array(values, dtype=float)

        self._linearize_fn = linearize
        self._resolve_fn = resolve
        self._expanded_ops = tuple(expanded_ops)
        self._raw_subsystems = tuple(mapped_qubits)

        # Whether any gate matrix depends on a runtime parameter.  When it does
        # not, the compressed operator stack is a compile-time constant and can
        # be materialised eagerly (outside the traced graph), which avoids XLA
        # constant-folding/autotuning a large ``compose_operator`` subgraph — the
        # dominant JIT cost on accelerators for deep, literal-angle programs.
        self._has_params = bool(param_refs)

        # Derive barrier nodes: measurements (QuantumInstrument) should not
        # be merged by the compressor.
        barrier_nodes = {i for i, op in enumerate(expanded_ops) if isinstance(op, qx.QuantumInstrument)}

        self._compress_fn = compressor_from_dag(
            dag,
            max_subsystem_size,
            dims=self.dims,
            barrier_nodes=barrier_nodes,
        )

        # Enumerate the *base subsystems* produced by the compressor.  The merge
        # structure depends only on the DAG (not on parameter values), so a
        # structural probe with zero parameters yields exactly the subsystem
        # sequence that ``compress`` will produce for any parameters.  The
        # lax-loop ``compute`` methods dispatch each compressed operation through
        # a ``jax.lax.switch`` keyed by its base, so the number of distinct bases
        # (rather than the number of operations) determines the size of the
        # traced/compiled graph.
        probe = self._compress_fn(self._resolve_fn(jnp.zeros(len(param_refs))))
        self.bases = []
        sub_to_branch: dict[tuple[int, ...], int] = {}
        op_index: list[int] = []
        for _, subsystem in probe:
            if subsystem not in sub_to_branch:
                sub_to_branch[subsystem] = len(self.bases)
                self.bases.append(subsystem)
            op_index.append(sub_to_branch[subsystem])
        self.op_index = tuple(op_index)
        self.base_dims = [tuple(self.dims[q] for q in base) for base in self.bases]
        self.base_total_dim = [math.prod(d) for d in self.base_dims]
        self.d_max = max(self.base_total_dim) if self.base_total_dim else 1

    # -- hook for subclass validation ---------------------

    def _validate(self, program: Program) -> None:
        """Override to reject unsupported instructions."""

    # -- public pipeline methods --------------------------

    def linearize(self, memory_map: MemoryMap) -> Array:
        """Convert a memory map to a flat JAX parameter vector."""
        return self._linearize_fn(memory_map)

    def resolve(self, params: Array) -> list[ResolvedOp]:
        """Resolve parameters into one operator per DAG node."""
        return self._resolve_fn(params)

    def compress(self, resolved: list[ResolvedOp]) -> list[ResolvedOp]:
        """Merge operators via greedy edge contraction."""
        return self._compress_fn(resolved)

    def compute(self, params: Array, **kwargs: Any) -> Any:
        """Compute the simulation result.  Subclasses must override."""
        raise NotImplementedError


# ══════════════════════════════════════════════════════════
# Vectorized gate construction
# ══════════════════════════════════════════════════════════


def _embed_matrix_np(mat: np.ndarray, op_subsystem: tuple[int, ...],
                     group_subsystem: tuple[int, ...], dims: tuple[int, ...],
                     d_max: int) -> np.ndarray:
    """Embed a gate matrix into a larger subsystem (numpy, for constant ops).

    Computes the d_max×d_max padded matrix that applies ``mat`` on
    ``op_subsystem`` within the Hilbert space of ``group_subsystem``.
    """
    if op_subsystem == group_subsystem:
        return np.pad(mat, [(0, d_max - s) for s in mat.shape])
    import quax as qx  # noqa: F811 — local re-import for clarity
    target_dims = tuple(dims[q] for q in group_subsystem)
    positions = tuple(group_subsystem.index(q) for q in op_subsystem)
    op_dims = tuple(dims[q] for q in op_subsystem)
    op = qx.Unitary.from_matrix(jnp.array(mat), (op_dims, op_dims))
    embedded = qx.embed(op, target_dims=target_dims, positions=positions)
    result = np.asarray(embedded.matrix)
    return np.pad(result, [(0, d_max - s) for s in result.shape])


def _make_embed_fn(
    op_subsystem: tuple[int, ...],
    group_subsystem: tuple[int, ...],
    dims: tuple[int, ...],
    d_max: int,
) -> Callable[[Array], Array]:
    """Return a JIT-friendly function that embeds a gate matrix into a group subsystem.

    Uses simple Kronecker products rather than the full qx.embed machinery
    to minimize the traced graph size.
    """
    if op_subsystem == group_subsystem:
        D = math.prod(dims[q] for q in op_subsystem)
        pad_w = ((0, d_max - D),) * 2

        def _identity_embed(mat: Array) -> Array:
            return jnp.pad(mat, pad_w)

        return _identity_embed

    # General case: embed a tensor-format operator by placing its output/input
    # axes at the requested positions and identity tensors on untouched axes.
    target_dims = tuple(dims[q] for q in group_subsystem)
    op_dims = tuple(dims[q] for q in op_subsystem)
    positions = tuple(group_subsystem.index(q) for q in op_subsystem)
    n_group = len(group_subsystem)
    D = math.prod(target_dims)
    pad_w = ((0, d_max - D),) * 2
    n_op = len(op_subsystem)

    # For the common case: 1-qubit gate in a 2-qubit group
    if n_op == 1 and n_group == 2 and all(d == 2 for d in target_dims):
        pos = positions[0]
        I2 = jnp.eye(2, dtype=complex)
        if pos == 0:

            def _embed(mat: Array) -> Array:
                return jnp.pad(jnp.kron(mat, I2), pad_w)

            return _embed
        else:

            def _embed(mat: Array) -> Array:
                return jnp.pad(jnp.kron(I2, mat), pad_w)

            return _embed

    non_op_positions = [i for i in range(n_group) if i not in positions]
    identity_factors = [jnp.eye(target_dims[i], dtype=complex) for i in non_op_positions]

    # Example for op positions (0, 2) in a 3-qudit group:
    # op tensor axes are out0,out2,in0,in2; identity axes are out1,in1;
    # output order must be out0,out1,out2,in0,in1,in2.
    labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if 2 * n_group > len(labels):
        raise ValueError(f"Cannot build an einsum embedding for {n_group} subsystems.")
    out_labels = labels[:n_group]
    in_labels = labels[n_group : 2 * n_group]
    op_subscript = "".join(out_labels[p] for p in positions) + "".join(in_labels[p] for p in positions)
    identity_subscripts = [out_labels[p] + in_labels[p] for p in non_op_positions]
    embedded_subscript = out_labels + in_labels
    einsum_spec = ",".join([op_subscript, *identity_subscripts]) + f"->{embedded_subscript}"

    def _embed_general(mat: Array) -> Array:
        op_tensor = mat.reshape(op_dims + op_dims)
        embedded = jnp.einsum(einsum_spec, op_tensor, *identity_factors)
        return jnp.pad(embedded.reshape(D, D), pad_w)

    return _embed_general


def _build_vectorized_unitary_constructor(
    expanded_ops: tuple,
    raw_subsystems: tuple[tuple[int, ...], ...],
    emit_order: list,
    dims: tuple[int, ...],
    d_max: int,
) -> tuple[Callable[[Array], Array], np.ndarray, np.ndarray]:
    """Build a vectorized constructor that produces *embedded* gate matrices.

    Each raw gate matrix is embedded into its merge group's subsystem so that
    compression can be performed as a simple segmented matmul scan.

    Returns ``(build_fn, sort_order, group_boundaries)`` where:
    - ``build_fn(params)`` → ``(N_raw, d_max, d_max)`` array of embedded matrices
      in group-sorted order (ops within each group are consecutive).
    - ``sort_order[i]`` = raw op index for position *i* in the sorted array.
    - ``group_boundaries[g]`` = start index of group *g* in the sorted array.
      The final compressed op for group *g* is the product of sorted ops from
      ``group_boundaries[g]`` to ``group_boundaries[g+1]``.
    """
    n_ops = len(expanded_ops)

    # ── Compute group membership and embedding for each raw op ──
    # raw_to_group[i] = which compressed group raw op i belongs to
    # raw_to_embed_key[i] = (id(gate_fn_or_None), concrete_key, embed_key) for grouping
    raw_to_group = np.empty(n_ops, dtype=np.int32)
    raw_to_group_sub: list[tuple[int, ...]] = [() for _ in range(n_ops)]

    sorted_raw_indices: list[int] = []  # raw op indices in group-then-topo order
    group_boundaries: list[int] = [0]
    for group_idx, (_, nodes, subsystem) in enumerate(emit_order):
        for nk in nodes:
            raw_to_group[nk] = group_idx
            raw_to_group_sub[nk] = subsystem
            sorted_raw_indices.append(nk)
        group_boundaries.append(len(sorted_raw_indices))

    sort_order = np.array(sorted_raw_indices, dtype=np.int32)
    group_bounds = np.array(group_boundaries, dtype=np.int32)

    # ── Group ops by (gate_fn, concrete_layout, embedding_config) for vmap ──
    # The embedding_config is (op_subsystem, group_subsystem) which determines
    # how the raw matrix is expanded into the group's Hilbert space.
    param_groups: dict[tuple, tuple[Callable, tuple, tuple, list[int], list[list[int]], Callable]] = {}
    const_entries: list[tuple[int, np.ndarray]] = []  # (sorted_position, embedded_matrix)

    for sorted_pos, raw_idx in enumerate(sorted_raw_indices):
        eop = expanded_ops[raw_idx]
        op_sub = raw_subsystems[raw_idx]
        grp_sub = raw_to_group_sub[raw_idx]

        if isinstance(eop, ParametricGate):
            concrete_mask = tuple(j for j, pi in enumerate(eop.param_indices) if pi < 0)
            concrete_vals = tuple(eop.concrete_values[j] for j in concrete_mask)
            # Group by embedding TYPE not physical qubits: all embeddings with
            # the same (op_dims, target_dims, positions_within_group) produce
            # identical traced graphs, so they can share a single vmap.
            op_dims_key = tuple(dims[q] for q in op_sub)
            target_dims_key = tuple(dims[q] for q in grp_sub)
            positions_key = tuple(grp_sub.index(q) for q in op_sub)
            embed_key = (op_dims_key, target_dims_key, positions_key)
            key = (id(eop.gate_fn), concrete_mask, concrete_vals, embed_key)
            if key not in param_groups:
                embed_fn = _make_embed_fn(op_sub, grp_sub, dims, d_max)
                param_groups[key] = (eop.gate_fn, eop.param_indices, eop.concrete_values,
                                     [], [], embed_fn)
            param_groups[key][3].append(sorted_pos)
            param_groups[key][4].append([pi for pi in eop.param_indices if pi >= 0])
        else:
            mat = np.asarray(eop.matrix)
            embedded = _embed_matrix_np(mat, op_sub, grp_sub, dims, d_max)
            const_entries.append((sorted_pos, embedded))

    # ── Build vmapped constructors ──
    vmapped_specs: list[tuple[np.ndarray, Callable[[Array], Array]]] = []

    for gate_fn, template_pi, template_cv, positions, pidx_lists, embed_fn in param_groups.values():
        pos_arr = np.array(positions)

        parametric_slots = [j for j, pi in enumerate(template_pi) if pi >= 0]
        concrete_slots = [(j, cv) for j, (pi, cv) in enumerate(zip(template_pi, template_cv, strict=False)) if pi < 0]
        n_total = len(template_pi)

        def _make_batch(gf: Callable, ps: list[int], cs: list[tuple[int, float]],
                        nt: int, ef: Callable, pidx: Array) -> Callable[[Array], Array]:
            def _single(parametric_values: Array) -> Array:
                args: list[Any] = [None] * nt
                for slot, val in cs:
                    args[slot] = val
                for k, slot in enumerate(ps):
                    args[slot] = parametric_values[k]
                return ef(gf(*args).matrix)

            batched = jax.vmap(_single)

            def build(params: Array) -> Array:
                return batched(params[pidx])

            return build

        pidx_arr = jnp.array(pidx_lists)
        builder = _make_batch(gate_fn, parametric_slots, concrete_slots,
                              n_total, embed_fn, pidx_arr)
        vmapped_specs.append((pos_arr, builder))

    # ── Pre-build constant embedded matrices ──
    if const_entries:
        const_positions = np.array([p for p, _ in const_entries])
        const_stack = jnp.array(np.stack([m for _, m in const_entries]))
    else:
        const_positions = None
        const_stack = None

    # ── Build gather index for padded parallel composition ──
    n_groups = len(group_boundaries) - 1
    group_sizes = np.diff(group_boundaries)
    max_group_size = int(group_sizes.max()) if n_groups > 0 else 1

    # gather_idx[g, k] = sorted position of the k-th op in group g,
    # or n_ops (the identity sentinel position) for padding.
    gather_idx = np.full((n_groups, max_group_size), n_ops, dtype=np.int32)
    for g in range(n_groups):
        start, end = group_boundaries[g], group_boundaries[g + 1]
        size = end - start
        gather_idx[g, :size] = np.arange(start, end)
    gather_idx_jax = jnp.array(gather_idx)
    eye_mat = jnp.eye(d_max, dtype=complex)

    def build_compressed_stack(params: Array) -> Array:
        """Build all gate matrices and compose within each merge group.

        Returns ``(n_groups, d_max, d_max)`` — one compressed matrix per group.
        """
        # 1. Build all embedded gate matrices in group-sorted order.
        raw_mats = jnp.zeros((n_ops, d_max, d_max), dtype=complex)
        for pos_arr, builder in vmapped_specs:
            raw_mats = raw_mats.at[pos_arr].set(builder(params))
        if const_stack is not None:
            raw_mats = raw_mats.at[const_positions].set(const_stack)

        # 2. Append identity sentinel and gather into (n_groups, max_size, d, d).
        raw_mats_plus = jnp.concatenate([raw_mats, eye_mat[None]], axis=0)
        padded = raw_mats_plus[gather_idx_jax]  # (n_groups, max_size, d, d)

        # 3. Parallel fold: vmap a scan over all groups simultaneously.
        def group_product(mats: Array) -> Array:
            def body(acc: Array, mat: Array) -> tuple[Array, None]:
                return mat @ acc, None
            final, _ = jax.lax.scan(body, eye_mat, mats)
            return final

        return jax.vmap(group_product)(padded)  # (n_groups, d, d)

    return build_compressed_stack, sort_order, group_bounds


# ══════════════════════════════════════════════════════════
# Pure state-vector simulator
# ══════════════════════════════════════════════════════════


class PureStateVectorSimulator(ProgramSimulator):
    """Simulator for gate-only programs (no noise, measurements, or resets).

    All methods are jit- and grad-friendly::

        sim = PureStateVectorSimulator(program)
        params = sim.linearize(memory_map)
        psi = jax.jit(sim.compute)(params)
        U = jax.jit(sim.unitary)(params)
    """

    __slots__ = ("_psi0", "_branches", "_idx_arr", "_vmapped_build_fn")

    def __init__(
        self,
        program: Program,
        qubits: list[int] | None = None,
        *,
        max_subsystem_size: int = 2,
    ) -> None:
        super().__init__(program, qubits, noise_model=None, max_subsystem_size=max_subsystem_size)
        self._psi0 = qx.zero_state_vector(dims=self.dims)

        # Vectorized gate construction (vmap per gate type) followed by a
        # segmented matmul scan for compression.  This gives both fast
        # compilation (small traced graph) AND fast runtime (compressed
        # op count in the state-evolution scan).
        emit_order = getattr(self._compress_fn, "emit_order", [])
        build_fn, _sort_order, _group_bounds = _build_vectorized_unitary_constructor(
            self._expanded_ops, self._raw_subsystems, emit_order, self.dims, self.d_max,
        )
        self._vmapped_build_fn = build_fn

        self._branches = [
            _make_unitary_branch(base, base_dims, db)
            for base, base_dims, db in zip(self.bases, self.base_dims, self.base_total_dim, strict=True)
        ]
        self._idx_arr = jnp.asarray(self.op_index, dtype=jnp.int32)

    def _validate(self, program: Program) -> None:
        for inst in program.instructions:
            if isinstance(inst, Measurement):
                raise ValueError(f"PureStateVectorSimulator does not support measurements.  Found: {inst}")
            if isinstance(inst, (Reset, ResetQubit)):
                raise ValueError(f"PureStateVectorSimulator does not support resets.  Found: {inst}")

    def compute(self, params: Array) -> qx.StateVector:  # type: ignore[override]
        """Compute the final state vector.

        Operators are stacked into a single array and applied with a
        :func:`jax.lax.scan` whose body dispatches each operator to the right
        base subsystem via :func:`jax.lax.switch`.  This keeps the traced graph
        size proportional to the number of distinct base subsystems rather than
        the number of operations, dramatically reducing JIT compilation time
        for large programs.

        :param params: Flat parameter vector from :meth:`linearize`.
        :return: The final state vector.
        """
        # No operations (e.g. empty program) → the initial state is the result.
        if not self._branches:
            return self._psi0

        # Vectorized construction: build embedded matrices via vmap, then
        # compose within each merge group via a parallel fold.
        op_stack = self._vmapped_build_fn(params)
        branches = self._branches

        def body(psi: qx.StateVector, xs: tuple[Array, Array]) -> tuple[qx.StateVector, None]:
            op_mat, sidx = xs
            psi = jax.lax.switch(sidx, branches, op_mat, psi)
            return psi, None

        psi, _ = jax.lax.scan(body, self._psi0, (op_stack, self._idx_arr))
        return psi


    def __call__(self, params: Array) -> qx.StateVector:
        return self.compute(params)

    def unitary(self, params: Array) -> qx.Unitary:
        """Compute the full program unitary.

        :param params: Flat parameter vector from :meth:`linearize`.
        :return: The full unitary matrix.
        """
        resolved = self.resolve(params)
        compressed = self.compress(resolved)

        accumulated: qx.Unitary | None = None
        for op, subsystem in compressed:
            embedded = qx.embed(op, target_dims=self.dims, positions=subsystem)
            if accumulated is None:
                accumulated = embedded
            else:
                accumulated = embedded @ accumulated

        if accumulated is None:
            d = 1
            for dim in self.dims:
                d *= dim
            return qx.Unitary.from_matrix(jnp.eye(d, dtype=complex), (self.dims, self.dims))

        return accumulated


# ══════════════════════════════════════════════════════════
# Density-matrix simulator
# ══════════════════════════════════════════════════════════


class DensityMatrixSimulator(ProgramSimulator):
    """Density-matrix simulator for any program, optionally with noise.

    All methods are jit- and grad-friendly::

        sim = DensityMatrixSimulator(program, noise_model=noise_model)
        params = sim.linearize(memory_map)
        rho = jax.jit(sim.compute)(params)
    """

    __slots__ = ("_rho0", "_branches", "_idx_arr", "_const_op_stack")

    def __init__(
        self,
        program: Program,
        qubits: list[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 2,
    ) -> None:
        super().__init__(program, qubits, noise_model=noise_model, max_subsystem_size=max_subsystem_size)
        self._rho0 = qx.zero_state_matrix(dims=self.dims)
        self._branches = [
            _make_superop_branch(base, base_dims, db * db)
            for base, base_dims, db in zip(self.bases, self.base_dims, self.base_total_dim, strict=True)
        ]
        self._idx_arr = jnp.asarray(self.op_index, dtype=jnp.int32)

        # See :class:`PureStateVectorSimulator`: for parameter-free programs the
        # superoperator stack is constant, so build it eagerly to keep the traced
        # graph to just the scan over a concrete array.
        self._const_op_stack: Array | None = None
        if not self._has_params and self.op_index:
            self._const_op_stack = jax.block_until_ready(
                self._stack_superops(self.resolve(jnp.zeros(0)))
            )

    def _stack_superops(self, resolved: list[ResolvedOp]) -> Array:
        """Compress, promote each op to a SuperOp, and stack."""
        compressed = self.compress(resolved)
        superops = adapt_for_density_matrix(compressed)
        d_max2 = self.d_max * self.d_max
        mats = [_pad_matrix(superop.matrix, d_max2, d_max2) for superop, _ in superops]
        return jnp.stack(mats, axis=0)

    def compute(self, params: Array) -> qx.DensityMatrix:  # type: ignore[override]
        """Compute the final density matrix.

        Superoperators are stacked and applied with a :func:`jax.lax.scan`
        whose body dispatches to the correct base subsystem via
        :func:`jax.lax.switch`, keeping the compiled graph size proportional to
        the number of distinct base subsystems.

        :param params: Flat parameter vector from :meth:`linearize`.
        :return: The final density matrix.
        """
        if self._const_op_stack is not None:
            op_stack = self._const_op_stack
        else:
            resolved = self.resolve(params)
            if not resolved:
                return self._rho0
            op_stack = self._stack_superops(resolved)
        branches = self._branches

        def body(rho: qx.DensityMatrix, xs: tuple[Array, Array]) -> tuple[qx.DensityMatrix, None]:
            op_mat, sidx = xs
            rho = jax.lax.switch(sidx, branches, op_mat, rho)
            return rho, None

        rho, _ = jax.lax.scan(body, self._rho0, (op_stack, self._idx_arr))
        return rho

    def __call__(self, params: Array) -> qx.DensityMatrix:
        return self.compute(params)


# ══════════════════════════════════════════════════════════
# Trajectory simulator
# ══════════════════════════════════════════════════════════


class TrajectorySimulator(ProgramSimulator):
    """Monte Carlo trajectory simulator for programs with measurements and resets.

    The ``compute`` method requires a JAX PRNG key.  The number of
    trajectories is determined by the key shape: a scalar key runs one
    trajectory; a batch of keys ``jax.random.split(key, n)`` runs *n*
    trajectories in parallel::

        sim = TrajectorySimulator(program, noise_model=noise_model)
        params = sim.linearize(memory_map)

        # Single trajectory
        key = jax.random.key(0)
        psi, outcomes = sim.compute(params, key)

        # Batched trajectories
        keys = jax.random.split(jax.random.key(0), 100)
        psi_batch, outcomes_batch = sim.compute(params, keys)

    The ``sample`` method is a convenience wrapper that runs trajectories
    in batches and discards state vectors, returning only measurement
    outcomes.
    """

    __slots__ = ("_kraus_truncation_threshold", "_devices")

    def __init__(
        self,
        program: Program,
        qubits: list[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 2,
        kraus_truncation_threshold: float = 1e-6,
        devices: list[jax.Device] | None = None,
        dims: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(program, qubits, noise_model=noise_model, max_subsystem_size=max_subsystem_size, dims=dims)
        self._kraus_truncation_threshold = kraus_truncation_threshold
        self._devices = devices if devices is not None else jax.devices()

    def adapt(self, compressed: list[ResolvedOp]) -> list[TrajectoryOp]:
        """Convert compressed ops to trajectory-compatible types."""
        return adapt_for_trajectory(compressed, self._kraus_truncation_threshold)

    def compute(  # type: ignore[override]
        self,
        params: Array,
        key: Array,
    ) -> tuple[qx.StateVector, Array]:
        """Run trajectory simulation.

        :param params: Flat parameter vector from :meth:`linearize`.
        :param key: JAX PRNG key.  Scalar key → single trajectory.
            Batch of keys (from ``jax.random.split``) → batched trajectories.
        :return: Tuple of ``(state_vector, measurement_outcomes)``.
        """
        resolved = self.resolve(params)
        compressed = self.compress(resolved)
        operations = self.adapt(compressed)

        if key.ndim == 0:
            psi = qx.zero_state_vector(dims=self.dims)
        else:
            n_traj = key.shape[0]
            psi = qx.zero_state_vector(dims=self.dims, ensemble_size=(n_traj,))

        return _apply_trajectory_operations(operations, psi, key)

    def __call__(self, params: Array, key: Array) -> tuple[qx.StateVector, Array]:
        return self.compute(params, key)

    def sample(
        self,
        params: Array,
        num_trajectories: int = 1000,
        batch_size: int = 250,
        random_seed: int = 0,
    ) -> Array:
        """Run trajectory simulation in batches, returning only measurement outcomes.

        State vectors are discarded after each batch, making this scalable
        to arbitrarily many trajectories.  When multiple devices are
        available, each batch is sharded across them so that every device
        processes ``batch_size // n_devices`` trajectories concurrently.

        :param params: Flat parameter vector from :meth:`linearize`.
        :param num_trajectories: Total number of trajectories to simulate.
        :param batch_size: Maximum number of trajectories per batch
            (total across all devices).
        :param random_seed: Seed for the JAX PRNG.
        :return: Measurement outcomes with shape ``(num_trajectories, n_measurements)``.
        """
        resolved = self.resolve(params)
        compressed = self.compress(resolved)
        operations = self.adapt(compressed)

        _, all_outcomes = _run_batched_trajectories(
            operations,
            self.n_qubits,
            num_trajectories,
            batch_size,
            random_seed,
            keep_states=False,
            dims=self.dims,
            devices=self._devices,
        )

        if len(all_outcomes) == 1:
            return all_outcomes[0]
        return jnp.concatenate(all_outcomes, axis=0)


# ══════════════════════════════════════════════════════════
# Trajectory simulation internals
# ══════════════════════════════════════════════════════════


def _op_to_kraus_matrix(
    op: qx.Unitary | qx.KrausMap | qx.QuantumInstrument,
) -> tuple[Array, int, bool]:
    """Convert a single trajectory operator to a padded Kraus matrix.

    Every trajectory operator is expressed as a Kraus map so that a single,
    uniform ``jax.lax.switch`` branch (Kraus trajectory sampling) can handle
    all operation types:

    - ``qx.Unitary`` → a one-operator Kraus map.
    - ``qx.KrausMap`` → itself.
    - ``qx.QuantumInstrument`` → its outcome and Kraus axes are merged into a
      single Kraus axis (replicating the flattening in
      :func:`quax.targeted_apply_instrument_to_state_vector`).  The returned
      *divisor* is the number of Kraus operators per outcome, so the sampled
      Kraus index ``k`` decodes to the measurement outcome ``k // divisor``.

    :param op: The operator (already acting on its base subsystem).
    :return: ``(matrix, divisor, is_measurement)`` where ``matrix`` has shape
        ``(n_kraus, d, d)``.
    """
    match op:
        case qx.Unitary():
            return op.matrix[jnp.newaxis, :, :], 1, False
        case qx.KrausMap():
            return op.matrix, 1, False
        case qx.QuantumInstrument():
            kraus_mats = [
                qx.superop_to_kraus(op.outcome_superop(i)[0]).matrix
                for i in range(op.num_outcomes)
            ]
            n_kraus_per_outcome = kraus_mats[0].shape[-3]
            merged = jnp.concatenate(kraus_mats, axis=-3)
            return merged, n_kraus_per_outcome, True
        case _:
            raise TypeError(f"Unsupported operator type: {type(op)}")


def _apply_trajectory_operations(
    operations: list[TrajectoryOp],
    psi: qx.StateVector,
    key: Array,
) -> tuple[qx.StateVector, Array]:
    """Apply trajectory operations to a (batched) state vector via a JAX loop.

    Every operator is converted to a (zero-padded) Kraus map and stacked into a
    single array.  A :func:`jax.lax.fori_loop` then iterates over the stack,
    dispatching each operator to the correct base subsystem with a
    :func:`jax.lax.switch`.  Because only one loop body and one switch branch
    per distinct subsystem are traced, the compiled graph size scales with the
    number of distinct subsystems rather than the number of operations.

    Measurements are handled uniformly: a quantum instrument is flattened so
    that sampling a Kraus index also selects an outcome (``index // divisor``).
    Zero-padded Kraus operators have zero Born probability and are therefore
    never sampled.

    Key generation is sharding-friendly: per-operation keys are derived lazily
    via ``jax.random.fold_in`` so the key array is never materialised in full.

    :param operations: Ordered list of ``(operator, subsystem)`` pairs.
    :param psi: Initial state vector, optionally batched via ensemble dimension.
    :param key: JAX PRNG key (scalar) or per-trajectory key vector.
    :return: Tuple of ``(final_state_vector, measurement_outcomes)`` where
        measurement_outcomes has shape ``(*ensemble, n_measurements)`` with
        dtype int32.
    """
    ensemble_size = psi.ensemble_size

    if not operations:
        return psi, jnp.empty((*ensemble_size, 0), dtype=jnp.int32)

    # 1. Enumerate distinct subsystems → one switch branch each.
    distinct_subsystems: list[tuple[int, ...]] = []
    sub_to_branch: dict[tuple[int, ...], int] = {}
    for _, subsystem in operations:
        if subsystem not in sub_to_branch:
            sub_to_branch[subsystem] = len(distinct_subsystems)
            distinct_subsystems.append(subsystem)

    branches = [
        _make_kraus_trajectory_branch(
            subsystem,
            tuple(psi.dims[q] for q in subsystem),
            math.prod(psi.dims[q] for q in subsystem),
        )
        for subsystem in distinct_subsystems
    ]

    # 2. Convert every operator to a padded Kraus matrix and stack.
    kraus_mats: list[Array] = []
    divisors: list[int] = []
    measure_positions: list[int] = []
    branch_index: list[int] = []
    for i, (op, subsystem) in enumerate(operations):
        mat, divisor, is_measure = _op_to_kraus_matrix(op)
        kraus_mats.append(mat)
        divisors.append(divisor)
        branch_index.append(sub_to_branch[subsystem])
        if is_measure:
            measure_positions.append(i)

    max_k = max(mat.shape[0] for mat in kraus_mats)
    d_max = max(mat.shape[-1] for mat in kraus_mats)
    op_stack = jnp.stack([_pad_matrix(mat, max_k, d_max, d_max) for mat in kraus_mats], axis=0)
    branch_arr = jnp.asarray(branch_index, dtype=jnp.int32)

    # 3. Per-trajectory base keys.
    if ensemble_size:
        per_traj_keys = key if key.ndim > 0 else jax.random.split(key, ensemble_size[0])
    else:
        per_traj_keys = None

    n_ops = len(operations)
    sampled_init = jnp.zeros((n_ops, *ensemble_size), dtype=jnp.int32)

    def body(i: Array, carry: tuple[qx.StateVector, Array]) -> tuple[qx.StateVector, Array]:
        psi_c, sampled = carry
        if per_traj_keys is not None:
            op_key = jax.vmap(lambda k: jax.random.fold_in(k, i))(per_traj_keys)
        else:
            op_key = jax.random.fold_in(key, i)
        psi_c, sampled_idx = jax.lax.switch(branch_arr[i], branches, op_stack[i], psi_c, op_key)
        return psi_c, sampled.at[i].set(sampled_idx.astype(jnp.int32))

    psi, sampled = jax.lax.fori_loop(0, n_ops, body, (psi, sampled_init))

    if measure_positions:
        outcomes = jnp.stack([sampled[p] // divisors[p] for p in measure_positions], axis=-1)
    else:
        outcomes = jnp.empty((*ensemble_size, 0), dtype=jnp.int32)

    return psi, outcomes


def _make_mesh(devices: list[jax.Device] | None) -> Mesh | None:
    """Build a 1-D ``Mesh`` over *devices*, or ``None`` for single-device."""
    if devices is None:
        devices = jax.devices()
    if len(devices) <= 1:
        return None
    return Mesh(np.array(devices), axis_names=("traj",))


def _round_up_to(n: int, divisor: int) -> int:
    """Round *n* up to the nearest multiple of *divisor*."""
    return ((n + divisor - 1) // divisor) * divisor


def _run_batched_trajectories(
    operations: list[TrajectoryOp],
    n_qubits: int,
    num_trajectories: int,
    batch_size: int,
    random_seed: int,
    keep_states: bool = True,
    dims: tuple[int, ...] | None = None,
    devices: list[jax.Device] | None = None,
) -> tuple[list[qx.StateVector] | None, list[Array]]:
    """Run trajectory simulation in batches, optionally sharded across devices.

    When *devices* contains more than one device a :class:`jax.sharding.Mesh`
    is constructed and both the initial state vector and PRNG keys are sharded
    along the trajectory (ensemble) axis.  XLA's SPMD partitioner then
    distributes the work so that each device processes its own slice.
    """
    if dims is None:
        dims = (2,) * n_qubits

    mesh = _make_mesh(devices)
    n_devices = len(mesh.devices.flat) if mesh is not None else 1

    key = jax.random.key(random_seed)
    all_psis: list[qx.StateVector] = [] if keep_states else []
    all_outcomes: list[Array] = []

    remaining = num_trajectories
    while remaining > 0:
        this_batch = min(remaining, batch_size)

        # Pad to a multiple of n_devices so the shard split is even.
        padded_batch = _round_up_to(this_batch, n_devices) if n_devices > 1 else this_batch
        n_pad = padded_batch - this_batch

        key, batch_key = jax.random.split(key)

        if padded_batch == 1:
            psi = qx.zero_state_vector(dims=dims)
        else:
            psi = qx.zero_state_vector(dims=dims, ensemble_size=(padded_batch,))

        # Shard state and key across devices when a mesh is available.
        if mesh is not None:
            sharding = NamedSharding(mesh, PartitionSpec("traj"))  # type: ignore[no-untyped-call]
            psi = qx.StateVector.from_matrix(
                jax.device_put(psi.matrix, sharding),
                psi.dims,
            )
            # Split a per-trajectory key vector and shard it.
            batch_keys = jax.random.split(batch_key, padded_batch)
            batch_keys = jax.device_put(batch_keys, sharding)
        else:
            batch_keys = batch_key

        psi_out, outcomes = _apply_trajectory_operations(operations, psi, batch_keys)
        psi_out.matrix.block_until_ready()

        # Strip padding rows.
        if n_pad > 0:
            psi_out = qx.StateVector.from_matrix(
                psi_out.matrix[:this_batch],
                psi_out.dims,
            )
            outcomes = outcomes[:this_batch]

        if this_batch == 1 and padded_batch == 1:
            psi_out = qx.StateVector.from_matrix(
                psi_out.matrix[jnp.newaxis],
                psi_out.dims,
            )
            outcomes = outcomes[jnp.newaxis]

        if keep_states:
            all_psis.append(psi_out)
        all_outcomes.append(outcomes)
        remaining -= this_batch

    return (all_psis if keep_states else None), all_outcomes
