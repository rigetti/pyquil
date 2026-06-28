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
from dataclasses import dataclass, field
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
    resolve_program,
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
        return cast(tuple[qx.StateVector, Array], _sample_kraus_map_trajectory(kraus_map, psi, key, base))

    return branch


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

    __slots__ = (
        "n_qubits",
        "qubits",
        "dims",
        "_linearize_fn",
        "_resolve_fn",
        "_compress_fn",
        "bases",
        "op_index",
        "base_dims",
        "base_total_dim",
        "d_max",
        "_has_params",
        "_expanded_ops",
        "_raw_subsystems",
    )

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

        # Expand the program into operators, inferring register dimensions when
        # not supplied.  See :func:`resolve_program`.
        res = resolve_program(program, noise_model, qubits, dims=dims)
        self.dims = res.dims
        self._resolve_fn = res.resolve
        self._expanded_ops = tuple(res.ops)
        self._raw_subsystems = tuple(res.subsystems)
        param_refs = res.param_refs

        # Build linearizer from parameter references discovered during expansion.
        def linearize(memory_map: MemoryMap) -> Array:
            if not param_refs:
                return jnp.array([], dtype=float)
            values = [float(memory_map[name][offset]) for name, offset in param_refs]
            return jnp.array(values, dtype=float)

        self._linearize_fn = linearize

        dag = build_dag(res.subsystems)

        # Whether any gate matrix depends on a runtime parameter.  When it does
        # not, the compressed operator stack is a compile-time constant and can
        # be materialised eagerly (outside the traced graph), which avoids XLA
        # constant-folding/autotuning a large ``compose_operator`` subgraph — the
        # dominant JIT cost on accelerators for deep, literal-angle programs.
        self._has_params = bool(param_refs)

        # Derive barrier nodes: measurements (QuantumInstrument) should not
        # be merged by the compressor.
        barrier_nodes = {i for i, op in enumerate(res.ops) if isinstance(op, qx.QuantumInstrument)}

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
        self.bases: list[tuple[int, ...]] = []
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

    def _evolve(self, state: Any, op_stack: Array) -> Any:
        """Apply a stack of operator matrices to *state* via a scan + switch.

        Each operator is dispatched to the switch branch for its base subsystem
        (``self._branches``, keyed by ``self._idx_arr``), so the compiled graph
        size scales with the number of distinct base subsystems rather than the
        number of operations.  Used by the state-vector and density-matrix
        simulators, which differ only in their branch and state types.
        """
        branches = self._branches  # type: ignore[attr-defined]

        def body(state: Any, xs: tuple[Array, Array]) -> tuple[Any, None]:
            op_mat, sidx = xs
            return jax.lax.switch(sidx, branches, op_mat, state), None

        state, _ = jax.lax.scan(body, state, (op_stack, self._idx_arr))  # type: ignore[attr-defined]
        return state


# ══════════════════════════════════════════════════════════
# Vectorized gate construction
# ══════════════════════════════════════════════════════════


def _embed_constant_matrix(
    mat: Array, op_subsystem: tuple[int, ...], group_subsystem: tuple[int, ...], dims: tuple[int, ...], d_max: int
) -> Array:
    """Embed a constant gate matrix into its merge group, padded to ``d_max``.

    Computes the ``d_max × d_max`` matrix that applies ``mat`` on ``op_subsystem``
    within the Hilbert space of ``group_subsystem`` via :func:`quax.embed`.  This
    runs eagerly (once per parameter-free gate, outside any ``jit``); the result
    is closed over as a compile-time constant.  The final pad to ``d_max`` — the
    uniform stack width across all groups — is plain array padding, not a
    tensor-product embedding, so it has no quax equivalent.
    """
    op_dims = tuple(dims[q] for q in op_subsystem)
    target_dims = tuple(dims[q] for q in group_subsystem)
    positions = tuple(group_subsystem.index(q) for q in op_subsystem)
    op = qx.Unitary.from_matrix(jnp.asarray(mat), (op_dims, op_dims))
    embedded = qx.embed(op, target_dims=target_dims, positions=positions).matrix
    return jnp.pad(embedded, [(0, d_max - s) for s in embedded.shape])


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


@dataclass
class _GateBatch:
    """A set of gates sharing one constructor, concrete layout, and embedding.

    Members differ only in which entries of the parameter vector feed their
    free arguments, so all of them are built with a single ``jax.vmap``.  This
    keeps the traced graph proportional to the number of distinct gate *kinds*
    rather than the number of gates.
    """

    gate_fn: Callable[..., qx.Unitary]
    n_args: int
    #: ``(slot, value)`` for each compile-time-constant argument.
    concrete_args: tuple[tuple[int, float], ...]
    #: Embeds a raw gate matrix into its merge group, padded to ``d_max``.
    embed_fn: Callable[[Array], Array]
    #: Sorted-array positions this batch fills, one per member.
    positions: list[int] = field(default_factory=list)
    #: Parameter-vector index for each free argument, one list per member.
    param_indices: list[list[int]] = field(default_factory=list)

    def builder(self) -> Callable[[Array], Array]:
        """Return ``params -> (n_members, d_max, d_max)`` embedded gate matrices."""
        concrete = {slot for slot, _ in self.concrete_args}
        free_slots = [j for j in range(self.n_args) if j not in concrete]
        gate_fn, embed_fn, n_args, concrete_args = self.gate_fn, self.embed_fn, self.n_args, self.concrete_args
        param_indices = jnp.asarray(self.param_indices)  # (n_members, n_free)

        def single(free_values: Array) -> Array:
            args: list[Any] = [None] * n_args
            for slot, val in concrete_args:
                args[slot] = val
            for k, slot in enumerate(free_slots):
                args[slot] = free_values[k]
            return embed_fn(gate_fn(*args).matrix)

        batched = jax.vmap(single)
        return lambda params: batched(params[param_indices])


def _make_group_fold(group_start: list[int], n_ops: int, d_max: int) -> Callable[[Array], Array]:
    """Build the per-group matrix-product fold.

    ``fold(raw)`` takes ``(n_ops, d_max, d_max)`` embedded gate matrices laid out
    in group order and returns ``(n_groups, d_max, d_max)`` — the ordered matrix
    product of each group's gates.  Groups are gathered into a padded
    ``(n_groups, max_size, ...)`` array (short groups padded with an identity
    sentinel) so every group folds under a single ``jax.vmap``.
    """
    n_groups = len(group_start) - 1
    sizes = np.diff(group_start)
    max_size = int(sizes.max()) if n_groups else 1

    # gather[g, k] = sorted position of group g's k-th gate, or n_ops (the
    # identity sentinel appended in ``fold``) for padding.
    gather = np.full((n_groups, max_size), n_ops, dtype=np.int32)
    for g in range(n_groups):
        gather[g, : sizes[g]] = np.arange(group_start[g], group_start[g + 1])
    gather_jax = jnp.asarray(gather)
    eye = jnp.eye(d_max, dtype=complex)

    def group_product(mats: Array) -> Array:
        final, _ = jax.lax.scan(lambda acc, m: (m @ acc, None), eye, mats)
        return final

    def fold(raw: Array) -> Array:
        padded = jnp.concatenate([raw, eye[None]], axis=0)[gather_jax]  # (n_groups, max_size, d, d)
        return jax.vmap(group_product)(padded)

    return fold


def _build_vectorized_unitary_constructor(
    expanded_ops: tuple[Any, ...],
    raw_subsystems: tuple[tuple[int, ...], ...],
    emit_order: list[tuple[int, list[int], tuple[int, ...]]],
    dims: tuple[int, ...],
    d_max: int,
) -> Callable[[Array], Array]:
    """Build a JIT-friendly constructor for the compressed unitary stack.

    Returns ``build(params) -> (n_groups, d_max, d_max)``: one matrix per merge
    group, equal to ``compress(resolve(params))`` but assembled so the traced
    graph scales with the number of distinct gate *kinds* rather than the number
    of gates.  Each gate is embedded into its merge group's Hilbert space, then
    the gates of every group are folded together via :func:`_make_group_fold`.
    """
    n_ops = len(expanded_ops)

    # Lay raw ops out in group order: group g occupies sorted positions
    # [group_start[g], group_start[g + 1]).
    sorted_indices: list[int] = []
    group_subsystems: list[tuple[int, ...]] = []  # merge subsystem per sorted position
    group_start: list[int] = [0]
    for _, nodes, subsystem in emit_order:
        for nk in nodes:
            sorted_indices.append(nk)
            group_subsystems.append(subsystem)
        group_start.append(len(sorted_indices))

    # Plan how each op's embedded matrix is produced: parametric gates are
    # collected into vmapped batches; constant gates are embedded eagerly.
    batches: dict[tuple, _GateBatch] = {}
    const_positions: list[int] = []
    const_mats: list[Array] = []
    for pos, raw_idx in enumerate(sorted_indices):
        op = expanded_ops[raw_idx]
        op_sub = raw_subsystems[raw_idx]
        grp_sub = group_subsystems[pos]
        if isinstance(op, ParametricGate):
            # Key by embedding *type* (dims + positions within the group), not
            # physical qubits: embeddings that trace to the same graph share a vmap.
            embed_key = (
                tuple(dims[q] for q in op_sub),
                tuple(dims[q] for q in grp_sub),
                tuple(grp_sub.index(q) for q in op_sub),
            )
            concrete_args = tuple((j, op.concrete_values[j]) for j, pi in enumerate(op.param_indices) if pi < 0)
            key = (id(op.gate_fn), concrete_args, embed_key)
            batch = batches.get(key)
            if batch is None:
                batch = _GateBatch(
                    gate_fn=op.gate_fn,
                    n_args=len(op.param_indices),
                    concrete_args=concrete_args,
                    embed_fn=_make_embed_fn(op_sub, grp_sub, dims, d_max),
                )
                batches[key] = batch
            batch.positions.append(pos)
            batch.param_indices.append([pi for pi in op.param_indices if pi >= 0])
        else:
            const_positions.append(pos)
            const_mats.append(_embed_constant_matrix(op.matrix, op_sub, grp_sub, dims, d_max))

    builders = [(np.asarray(b.positions), b.builder()) for b in batches.values()]
    const_pos_arr = np.asarray(const_positions) if const_positions else None
    const_stack = jnp.stack(const_mats) if const_mats else None

    fold = _make_group_fold(group_start, n_ops, d_max)

    def build(params: Array) -> Array:
        raw = jnp.zeros((n_ops, d_max, d_max), dtype=complex)
        for positions, builder in builders:
            raw = raw.at[positions].set(builder(params))
        if const_stack is not None:
            raw = raw.at[const_pos_arr].set(const_stack)
        return fold(raw)

    return build


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
        self._vmapped_build_fn = _build_vectorized_unitary_constructor(
            self._expanded_ops,
            self._raw_subsystems,
            emit_order,
            self.dims,
            self.d_max,
        )

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
        return self._evolve(self._psi0, op_stack)

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
            d = math.prod(self.dims)
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
        # superoperator stack is constant, so it can be built once and reused to keep
        # the traced graph to just the scan over a concrete array. It is materialised
        # lazily on the first :meth:`compute` call rather than eagerly here, so
        # constructing the simulator stays cheap.
        self._const_op_stack: Array | None = None

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
        if not self._has_params and self.op_index:
            # Parameter-free program: build the constant superop stack once, then reuse.
            if self._const_op_stack is None:
                self._const_op_stack = self._stack_superops(self.resolve(jnp.zeros(0)))
            op_stack = self._const_op_stack
        else:
            resolved = self.resolve(params)
            if not resolved:
                return self._rho0
            op_stack = self._stack_superops(resolved)
        return self._evolve(self._rho0, op_stack)

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
            return qx.to_kraus(op).matrix, 1, False
        case qx.KrausMap():
            return op.matrix, 1, False
        case qx.QuantumInstrument():
            kraus_mats = [qx.superop_to_kraus(op.outcome_superop(i)[0]).matrix for i in range(op.num_outcomes)]
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
    all_psis: list[qx.StateVector] = []
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
