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
"""Program simulators backed by quax.

All simulators share the preprocessing in :class:`ProgramSimulator` (expansion,
dimension inference, ``linearize``/``resolve``/``compress``) and then split into
two families with different execution models:

* **Grad-able** (:class:`_GradableSimulator`) — jit/grad-friendly evolution of a
  compressed ``Unitary``/``SuperOp`` stack; measurements are dephasing SuperOps and
  there are no compressor barriers:

  * :class:`PureStateVectorSimulator` — gate-only programs (no noise, measurements,
    or resets).
  * :class:`DensityMatrixSimulator` — any program, optionally with noise.

* **Trajectory** (:class:`_TrajectorySimulator`) — Monte Carlo sampling of programs
  with measurements and resets; measurements stay as sampled QuantumInstruments:

  * :class:`TrajectorySimulator` — fixed-dimension, vectorized/batched trajectories.
  * :class:`DynamicTrajectorySimulator` — eager, per-trajectory dynamic qudit dims.

The ``compute`` method is the main entry point; for the grad-able family it can be
passed directly to ``jax.jit`` or ``jax.grad``.
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
    MeasurementMode,
    ParametricGate,
    ResolvedOp,
    TrajectoryOp,
    adapt_for_density_matrix,
    adapt_for_trajectory,
    build_dag,
    compressor_from_dag,
    enumerate_bases,
    resolve_for_gradable,
    resolve_for_trajectory,
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


# ══════════════════════════════════════════════════════════
# Base class
# ══════════════════════════════════════════════════════════


class ProgramSimulator:
    """Shared preprocessing base for program simulators.

    Handles the pipeline common to every backend: circuit expansion, qubit
    ordering, dimension inference, and building the ``linearize``/``resolve``/
    ``compress`` closures.  Two family bases specialise it — :class:`_GradableSimulator`
    (state-vector, density-matrix) and :class:`_TrajectorySimulator` (trajectory,
    dynamic trajectory) — supplying only the execution machinery each needs.

    Subclasses override :meth:`_validate` and :meth:`compute`.

    Instances are treated as immutable after construction.
    """

    def __init__(
        self,
        program: Program,
        qubits: list[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 2,
        dims: tuple[int, ...] | None = None,
        measurement: MeasurementMode = "instrument",
    ) -> None:
        self._validate(program)

        if qubits is None:
            qubits = sorted(program.get_qubit_indices())
        self.qubits = qubits
        self.n_qubits = len(qubits)

        # Expand the program into operators, inferring register dimensions when not
        # supplied.  The measurement mode selects the resolver family: the grad-able
        # simulators resolve measurements to dephasing SuperOps, the trajectory
        # simulators keep them as sampled QuantumInstruments.
        if measurement == "superop":
            res = resolve_for_gradable(program, noise_model, qubits, dims)
        else:
            res = resolve_for_trajectory(program, noise_model, qubits, dims)
        self.dims = res.dims
        self._resolve_fn = res.resolve
        self._expanded_ops = tuple(res.ops)
        self._raw_subsystems = tuple(res.subsystems)
        self._n_params = len(res.param_refs)
        param_refs = res.param_refs

        # Build linearizer from parameter references discovered during expansion.
        def linearize(memory_map: MemoryMap) -> Array:
            if not param_refs:
                return jnp.array([], dtype=float)
            values = [float(memory_map[name][offset]) for name, offset in param_refs]
            return jnp.array(values, dtype=float)

        self._linearize_fn = linearize

        dag = build_dag(res.subsystems)

        # Measurements (QuantumInstrument) must not be merged by the compressor.
        # Under the grad-able ("superop") mode no instruments are produced, so this
        # is naturally empty and the compressor is free to merge every operation.
        barrier_nodes = {i for i, op in enumerate(res.ops) if isinstance(op, qx.QuantumInstrument)}

        self._compress_fn = compressor_from_dag(
            dag,
            max_subsystem_size,
            dims=self.dims,
            barrier_nodes=barrier_nodes,
        )

    # -- hook for subclass validation ---------------------

    def _validate(self, program: Program) -> None:
        """Override to reject unsupported instructions."""

    # -- public pipeline methods --------------------------

    def linearize(self, memory_map: MemoryMap) -> Array:
        """Convert a memory map to a flat JAX parameter vector."""
        return self._linearize_fn(memory_map)

    def _default_params(self, params: Array | None) -> Array:
        """Return *params*, or the empty-memory-map vector when ``None``.

        Lets callers omit ``params`` for parameter-free programs (where the
        vector is empty).  For a parametric program the empty memory map raises
        on the missing register, which is clearer than silently using zeros.
        """
        return self.linearize({}) if params is None else params

    def resolve(self, params: Array) -> list[ResolvedOp]:
        """Resolve parameters into one operator per DAG node."""
        return self._resolve_fn(params)

    def compress(self, resolved: list[ResolvedOp]) -> list[ResolvedOp]:
        """Merge operators via greedy edge contraction."""
        return self._compress_fn(resolved)

    def compute(self, params: Array | None = None, **kwargs: Any) -> Any:
        """Compute the simulation result.  Subclasses must override."""
        raise NotImplementedError


# ══════════════════════════════════════════════════════════
# Grad-able family base (state-vector / density-matrix)
# ══════════════════════════════════════════════════════════


class _GradableSimulator(ProgramSimulator):
    """Base for the jit/grad-friendly state-vector and density-matrix simulators.

    Adds the compressed-stack evolution machinery.  It enumerates the distinct
    *base subsystems* the compressor emits (:func:`enumerate_bases`) and applies the
    operator stack with a :func:`jax.lax.scan` whose body dispatches each operator to
    the :func:`jax.lax.switch` branch for its base (``self._branches``, keyed by
    ``self._idx_arr``), so the compiled graph size scales with the number of distinct
    base subsystems rather than the number of operations.

    Measurements are dephasing SuperOps (``measurement="superop"``) and the compressor
    runs with no barriers, so former-measurement operators merge freely.
    """

    def __init__(
        self,
        program: Program,
        qubits: list[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 2,
        dims: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(
            program,
            qubits,
            noise_model=noise_model,
            max_subsystem_size=max_subsystem_size,
            dims=dims,
            measurement="superop",
        )

        # The merge structure depends only on the DAG (not on parameter values), so
        # the base subsystems can be read straight off the compressor's emit order —
        # no ``resolve``/``compress`` probe is required.
        self.bases, self.op_index = enumerate_bases(self._compress_fn.emit_order)  # type: ignore[attr-defined]
        self.base_dims = [tuple(self.dims[q] for q in base) for base in self.bases]
        self.base_total_dim = [math.prod(d) for d in self.base_dims]
        self.d_max = max(self.base_total_dim) if self.base_total_dim else 1
        self._idx_arr = jnp.asarray(self.op_index, dtype=jnp.int32)

        # Whether any gate matrix depends on a runtime parameter.  When it does not,
        # the compressed operator stack is a compile-time constant and can be
        # materialised eagerly (outside the traced graph), which avoids XLA
        # constant-folding/autotuning a large ``compose_operator`` subgraph — the
        # dominant JIT cost on accelerators for deep, literal-angle programs.
        self._has_params = self._n_params > 0

    def apply(self, state: Any, op_stack: Array) -> Any:
        """Apply a stack of operator matrices to *state* via a scan + switch.

        Each operator is dispatched to the switch branch for its base subsystem
        (``self._branches``, keyed by ``self._idx_arr``), so the compiled graph size
        scales with the number of distinct base subsystems rather than the number of
        operations.  The state-vector and density-matrix simulators differ only in
        their branch and state types.
        """
        branches = self._branches  # type: ignore[attr-defined]

        def body(state: Any, xs: tuple[Array, Array]) -> tuple[Any, None]:
            op_mat, sidx = xs
            return jax.lax.switch(sidx, branches, op_mat, state), None

        state, _ = jax.lax.scan(body, state, (op_stack, self._idx_arr))
        return state


# ══════════════════════════════════════════════════════════
# Vectorized gate construction
# ══════════════════════════════════════════════════════════


def _embed_unitary_to_group(
    op: qx.Unitary,
    target_dims: tuple[int, ...],
    positions: tuple[int, ...],
    d_max: int,
) -> Array:
    """Embed *op* into a merge group and pad to the uniform stack width ``d_max``.

    :func:`quax.embed` places ``op`` (whose qudits map to ``positions`` within the
    group) into the group Hilbert space ``target_dims``; the trailing pad to
    ``d_max`` — the stack width shared by every group — is plain array padding with
    no quax equivalent.  This is traceable, so it serves both the eager
    constant-gate path and the vmapped parametric path.
    """
    embedded = qx.embed(op, target_dims=target_dims, positions=positions).matrix
    return jnp.pad(embedded, [(0, d_max - s) for s in embedded.shape])


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
    #: Per-qudit dimensions of the merge group each member embeds into.
    target_dims: tuple[int, ...]
    #: Positions within the group occupied by the gate's qudits.
    group_positions: tuple[int, ...]
    #: Uniform stack width every embedded matrix is padded to.
    d_max: int
    #: Sorted-array positions this batch fills, one per member.
    positions: list[int] = field(default_factory=list)
    #: Parameter-vector index for each free argument, one list per member.
    param_indices: list[list[int]] = field(default_factory=list)

    def builder(self) -> Callable[[Array], Array]:
        """Return ``params -> (n_members, d_max, d_max)`` embedded gate matrices."""
        concrete = {slot for slot, _ in self.concrete_args}
        free_slots = [j for j in range(self.n_args) if j not in concrete]
        gate_fn, n_args, concrete_args = self.gate_fn, self.n_args, self.concrete_args
        target_dims, group_positions, d_max = self.target_dims, self.group_positions, self.d_max
        param_indices = jnp.asarray(self.param_indices)  # (n_members, n_free)

        def single(free_values: Array) -> Array:
            args: list[Any] = [None] * n_args
            for slot, val in concrete_args:
                args[slot] = val
            for k, slot in enumerate(free_slots):
                args[slot] = free_values[k]
            return _embed_unitary_to_group(gate_fn(*args), target_dims, group_positions, d_max)

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
        # Where the op's qudits sit within the merge group, and the group's dims.
        target_dims = tuple(dims[q] for q in grp_sub)
        group_positions = tuple(grp_sub.index(q) for q in op_sub)
        if isinstance(op, ParametricGate):
            # Key by embedding *type* (op dims + group dims + positions), not
            # physical qubits: embeddings that trace to the same graph share a vmap.
            embed_key = (tuple(dims[q] for q in op_sub), target_dims, group_positions)
            concrete_args = tuple((j, op.concrete_values[j]) for j, pi in enumerate(op.param_indices) if pi < 0)
            key = (id(op.gate_fn), concrete_args, embed_key)
            batch = batches.get(key)
            if batch is None:
                batch = _GateBatch(
                    gate_fn=op.gate_fn,
                    n_args=len(op.param_indices),
                    concrete_args=concrete_args,
                    target_dims=target_dims,
                    group_positions=group_positions,
                    d_max=d_max,
                )
                batches[key] = batch
            batch.positions.append(pos)
            batch.param_indices.append([pi for pi in op.param_indices if pi >= 0])
        else:
            const_positions.append(pos)
            const_mats.append(_embed_unitary_to_group(op, target_dims, group_positions, d_max))

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


class PureStateVectorSimulator(_GradableSimulator):
    """Simulator for gate-only programs (no noise, measurements, or resets).

    All methods are jit- and grad-friendly::

        sim = PureStateVectorSimulator(program)
        params = sim.linearize(memory_map)
        psi = jax.jit(sim.compute)(params)
        U = jax.jit(sim.unitary)(params)
    """

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

        # One switch branch per distinct base subsystem: it rebuilds a Unitary
        # from the padded matrix slice for its base and applies it to the state.
        def unitary_branch(
            base: tuple[int, ...], base_dims: tuple[int, ...], db: int
        ) -> Callable[[Array, qx.StateVector], qx.StateVector]:
            def branch(op_mat: Array, psi: qx.StateVector) -> qx.StateVector:
                unitary = qx.Unitary.from_matrix(op_mat[:db, :db], (base_dims, base_dims))
                return qx.targeted_apply_unitary(unitary, psi, base)

            return branch

        self._branches = [
            unitary_branch(base, base_dims, db)
            for base, base_dims, db in zip(self.bases, self.base_dims, self.base_total_dim, strict=True)
        ]

    def _validate(self, program: Program) -> None:
        for inst in program.instructions:
            if isinstance(inst, Measurement):
                raise ValueError(f"PureStateVectorSimulator does not support measurements.  Found: {inst}")
            if isinstance(inst, (Reset, ResetQubit)):
                raise ValueError(f"PureStateVectorSimulator does not support resets.  Found: {inst}")

    def compute(self, params: Array | None = None) -> qx.StateVector:  # type: ignore[override]
        """Compute the final state vector.

        Operators are stacked into a single array and applied with a
        :func:`jax.lax.scan` whose body dispatches each operator to the right
        base subsystem via :func:`jax.lax.switch`.  This keeps the traced graph
        size proportional to the number of distinct base subsystems rather than
        the number of operations, dramatically reducing JIT compilation time
        for large programs.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :return: The final state vector.
        """
        # No operations (e.g. empty program) → the initial state is the result.
        if not self._branches:
            return self._psi0

        # Vectorized construction: build embedded matrices via vmap, then
        # compose within each merge group via a parallel fold.
        op_stack = self._vmapped_build_fn(self._default_params(params))
        return self.apply(self._psi0, op_stack)

    def __call__(self, params: Array | None = None) -> qx.StateVector:
        return self.compute(params)

    def unitary(self, params: Array | None = None) -> qx.Unitary:
        """Compute the full program unitary.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :return: The full unitary matrix.
        """
        resolved = self.resolve(self._default_params(params))
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


class DensityMatrixSimulator(_GradableSimulator):
    """Density-matrix simulator for any program, optionally with noise.

    All methods are jit- and grad-friendly::

        sim = DensityMatrixSimulator(program, noise_model=noise_model)
        params = sim.linearize(memory_map)
        rho = jax.jit(sim.compute)(params)
    """

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

        # One switch branch per distinct base subsystem: it rebuilds a SuperOp
        # from the padded matrix slice for its base and applies it to the state.
        def superop_branch(
            base: tuple[int, ...], base_dims: tuple[int, ...], db2: int
        ) -> Callable[[Array, qx.DensityMatrix], qx.DensityMatrix]:
            def branch(op_mat: Array, rho: qx.DensityMatrix) -> qx.DensityMatrix:
                superop = qx.SuperOp.from_matrix(op_mat[:db2, :db2], (base_dims, base_dims))
                return qx.targeted_apply_superop(superop, rho, base)

            return branch

        self._branches = [
            superop_branch(base, base_dims, db * db)
            for base, base_dims, db in zip(self.bases, self.base_dims, self.base_total_dim, strict=True)
        ]

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

    def compute(self, params: Array | None = None) -> qx.DensityMatrix:  # type: ignore[override]
        """Compute the final density matrix.

        Superoperators are stacked and applied with a :func:`jax.lax.scan`
        whose body dispatches to the correct base subsystem via
        :func:`jax.lax.switch`, keeping the compiled graph size proportional to
        the number of distinct base subsystems.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :return: The final density matrix.
        """
        if not self._has_params and self.op_index:
            # Parameter-free program: build the constant superop stack once, then reuse.
            if self._const_op_stack is None:
                self._const_op_stack = self._stack_superops(self.resolve(jnp.zeros(0)))
            op_stack = self._const_op_stack
        else:
            resolved = self.resolve(self._default_params(params))
            if not resolved:
                return self._rho0
            op_stack = self._stack_superops(resolved)
        return self.apply(self._rho0, op_stack)

    def __call__(self, params: Array | None = None) -> qx.DensityMatrix:
        return self.compute(params)


# ══════════════════════════════════════════════════════════
# Trajectory family base
# ══════════════════════════════════════════════════════════


class _TrajectorySimulator(ProgramSimulator):
    """Base for the Monte-Carlo trajectory simulators.

    Resolves measurements as sampled ``QuantumInstrument`` operators
    (``measurement="instrument"``), keeps them out of merges via compressor
    barriers, and adapts the compressed operators to the trajectory-native
    ``Unitary``/``KrausMap``/``QuantumInstrument`` types.  Unlike the grad-able
    family it builds no base-subsystem switch table — each concrete simulator
    builds its own trajectory kernel.
    """

    def __init__(
        self,
        program: Program,
        qubits: list[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 2,
        kraus_truncation_threshold: float = 1e-6,
        dims: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(
            program,
            qubits,
            noise_model=noise_model,
            max_subsystem_size=max_subsystem_size,
            dims=dims,
            measurement="instrument",
        )
        self._kraus_truncation_threshold = kraus_truncation_threshold

    def adapt(self, compressed: list[ResolvedOp]) -> list[TrajectoryOp]:
        """Convert compressed ops to trajectory-compatible types."""
        return adapt_for_trajectory(compressed, self._kraus_truncation_threshold)


# ══════════════════════════════════════════════════════════
# Trajectory simulator
# ══════════════════════════════════════════════════════════


class TrajectorySimulator(_TrajectorySimulator):
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
        super().__init__(
            program,
            qubits,
            noise_model=noise_model,
            max_subsystem_size=max_subsystem_size,
            kraus_truncation_threshold=kraus_truncation_threshold,
            dims=dims,
        )
        self._devices = devices if devices is not None else jax.devices()

    def compute(  # type: ignore[override]
        self,
        params: Array | None = None,
        key: Array | None = None,
    ) -> tuple[qx.StateVector, Array]:
        """Run trajectory simulation.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :param key: JAX PRNG key (required).  Scalar key → single trajectory.
            Batch of keys (from ``jax.random.split``) → batched trajectories.
        :return: Tuple of ``(state_vector, measurement_outcomes)``.
        """
        if key is None:
            raise ValueError("TrajectorySimulator.compute requires a JAX PRNG key.")
        operations = self.adapt(self.compress(self.resolve(self._default_params(params))))

        if key.ndim == 0:
            psi = qx.zero_state_vector(dims=self.dims)
        else:
            n_traj = key.shape[0]
            psi = qx.zero_state_vector(dims=self.dims, ensemble_size=(n_traj,))

        return _build_trajectory_kernel(operations, self.dims)(psi, key)

    def __call__(self, params: Array | None = None, key: Array | None = None) -> tuple[qx.StateVector, Array]:
        return self.compute(params, key)

    def sample(
        self,
        params: Array | None = None,
        num_trajectories: int = 1000,
        batch_size: int = 250,
        random_seed: int = 0,
    ) -> Array:
        """Run trajectory simulation in batches, returning only measurement outcomes.

        State vectors are discarded after each batch, making this scalable
        to arbitrarily many trajectories.  When multiple devices are
        available, each batch is sharded across them so that every device
        processes ``batch_size // n_devices`` trajectories concurrently.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :param num_trajectories: Total number of trajectories to simulate.
        :param batch_size: Maximum number of trajectories per batch
            (total across all devices).
        :param random_seed: Seed for the JAX PRNG.
        :return: Measurement outcomes with shape ``(num_trajectories, n_measurements)``.
        """
        operations = self.adapt(self.compress(self.resolve(self._default_params(params))))

        _, all_outcomes = _run_batched_trajectories(
            operations,
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


TrajectoryRun = Callable[[qx.StateVector, Array], tuple[qx.StateVector, Array]]

#: A ``run(op_mat, psi, key) -> (psi, sampled_index)`` switch branch for one subsystem.
KrausBranch = Callable[[Array, qx.StateVector, Array], tuple[qx.StateVector, Array]]


@dataclass
class _KrausOpStack:
    """The batch-invariant, scan-ready form of a trajectory operation sequence.

    Every operation is expressed as a zero-padded Kraus matrix so the scan can index
    a single homogeneous stack (quax has no ragged/heterogeneous KrausMap stacking;
    operators live on different subsystems with different Kraus counts).  The padding
    to a uniform ``(max_k, d_max, d_max)`` is inherent to that single-``lax.scan``
    design — zero-padded Kraus operators carry zero Born probability and are simply
    never sampled — and each :data:`KrausBranch` re-slices ``[:, :db, :db]`` back to
    its subsystem before rebuilding a ``KrausMap``.
    """

    #: ``(n_ops, max_k, d_max, d_max)`` padded Kraus matrices, in application order.
    op_stack: Array
    #: ``(n_ops,)`` int32 — the :attr:`branches` index for each operation.
    branch_arr: Array
    #: One switch branch per distinct subsystem (Kraus trajectory sampling).
    branches: list[KrausBranch]
    #: Per-operation Kraus-count divisor; measurement outcome = ``sampled_index // divisor``.
    divisors: list[int]
    #: Indices (into the op sequence) of the measurement operations, in program order.
    measure_positions: list[int]

    @property
    def n_ops(self) -> int:
        return len(self.divisors)


def _build_kraus_op_stack(operations: list[TrajectoryOp], dims: tuple[int, ...]) -> _KrausOpStack:
    """Convert a trajectory operation sequence into a scan-ready :class:`_KrausOpStack`.

    This is all the batch-invariant work: enumerating distinct subsystems (one switch
    branch each), promoting each operator to its register dimension, converting it to a
    Kraus matrix (:func:`_op_to_kraus_matrix`), and padding everything to one uniform
    stack.  Separated from :func:`_build_trajectory_kernel` so the kernel itself only
    holds the scan / key-folding / outcome-decoding logic.

    :param operations: Ordered list of ``(operator, subsystem)`` pairs.
    :param dims: Per-qudit register dimensions (must match the ``psi`` passed to the kernel).
    """
    distinct_subsystems: list[tuple[int, ...]] = []
    sub_to_branch: dict[tuple[int, ...], int] = {}
    for _, subsystem in operations:
        if subsystem not in sub_to_branch:
            sub_to_branch[subsystem] = len(distinct_subsystems)
            distinct_subsystems.append(subsystem)

    def make_branch(base: tuple[int, ...]) -> KrausBranch:
        base_dims = tuple(dims[q] for q in base)
        db = math.prod(base_dims)

        def branch(op_mat: Array, psi: qx.StateVector, key: Array) -> tuple[qx.StateVector, Array]:
            kraus_map = qx.KrausMap.from_matrix(op_mat[:, :db, :db], (base_dims, base_dims))
            return cast(tuple[qx.StateVector, Array], _sample_kraus_map_trajectory(kraus_map, psi, key, base))

        return branch

    branches = [make_branch(subsystem) for subsystem in distinct_subsystems]

    kraus_mats: list[Array] = []
    divisors: list[int] = []
    measure_positions: list[int] = []
    branch_index: list[int] = []
    for i, (op, subsystem) in enumerate(operations):
        # Promote each operator to the register dimension on its subsystem (identity on the
        # higher levels). Without this, an op authored at a lower dimension than the register
        # (e.g. a qubit-dimension channel on a register promoted to qutrits by a leakage model)
        # would be zero-padded to ``d_max`` instead — silently wrong on the high levels, and a
        # reshape error when ``d_max`` is below the branch's dimension.
        target_dims = tuple(dims[q] for q in subsystem)
        if op.dims[0] != target_dims:
            op = qx.promote(op, target_dims)
        mat, divisor, is_measure = _op_to_kraus_matrix(op)
        kraus_mats.append(mat)
        divisors.append(divisor)
        branch_index.append(sub_to_branch[subsystem])
        if is_measure:
            measure_positions.append(i)

    max_k = max(mat.shape[0] for mat in kraus_mats)
    d_max = max(mat.shape[-1] for mat in kraus_mats)
    op_stack = jnp.stack([_pad_matrix(mat, max_k, d_max, d_max) for mat in kraus_mats], axis=0)
    return _KrausOpStack(
        op_stack=op_stack,
        branch_arr=jnp.asarray(branch_index, dtype=jnp.int32),
        branches=branches,
        divisors=divisors,
        measure_positions=measure_positions,
    )


def _build_trajectory_kernel(operations: list[TrajectoryOp], dims: tuple[int, ...]) -> TrajectoryRun:
    """Build a reusable, jitted trajectory kernel from *operations*.

    All batch-invariant work happens once in :func:`_build_kraus_op_stack`.  The
    returned ``run(psi, key)`` wraps the scan over that padded Kraus stack in
    :func:`jax.jit`, so repeated calls with matching ``psi``/``key`` shapes (e.g. one
    per sampling batch) reuse a single compilation instead of re-tracing — this is
    what turns the previously spiky, recompile-per-batch GPU usage into a single
    upfront compile.

    Measurements are handled uniformly by flattening a quantum instrument so that
    sampling a Kraus index also selects an outcome (``index // divisor``).
    Per-operation keys are derived lazily via ``jax.random.fold_in`` so the key array
    is never materialised in full (sharding-friendly).

    :param operations: Ordered list of ``(operator, subsystem)`` pairs.
    :param dims: Per-qudit register dimensions (must match the ``psi`` passed to
        ``run``).
    :return: ``run(psi, key) -> (final_state_vector, measurement_outcomes)`` where
        ``measurement_outcomes`` has shape ``(*ensemble, n_measurements)``,
        dtype int32.  ``key`` is a scalar PRNG key or a per-trajectory key vector.
    """
    if not operations:

        def run_empty(psi: qx.StateVector, key: Array) -> tuple[qx.StateVector, Array]:
            return psi, jnp.empty((*psi.ensemble_size, 0), dtype=jnp.int32)

        return run_empty

    stack = _build_kraus_op_stack(operations, dims)
    branches = stack.branches
    op_stack = stack.op_stack
    branch_arr = stack.branch_arr
    divisors = stack.divisors
    measure_positions = stack.measure_positions
    op_indices = jnp.arange(stack.n_ops, dtype=jnp.int32)

    @jax.jit
    def run(psi: qx.StateVector, key: Array) -> tuple[qx.StateVector, Array]:
        ensemble_size = psi.ensemble_size
        if ensemble_size:
            per_traj_keys = key if key.ndim > 0 else jax.random.split(key, ensemble_size[0])
        else:
            per_traj_keys = None

        def body(psi_c: qx.StateVector, xs: tuple[Array, Array, Array]) -> tuple[qx.StateVector, Array]:
            op_mat, bidx, i = xs
            if per_traj_keys is not None:
                op_key = jax.vmap(lambda k: jax.random.fold_in(k, i))(per_traj_keys)
            else:
                op_key = jax.random.fold_in(key, i)
            psi_c, sampled_idx = jax.lax.switch(bidx, branches, op_mat, psi_c, op_key)
            return psi_c, sampled_idx.astype(jnp.int32)

        psi_out, sampled = jax.lax.scan(body, psi, (op_stack, branch_arr, op_indices))

        if measure_positions:
            outcomes = jnp.stack([sampled[p] // divisors[p] for p in measure_positions], axis=-1)
        else:
            outcomes = jnp.empty((*ensemble_size, 0), dtype=jnp.int32)
        return psi_out, outcomes

    return run


def _apply_trajectory_operations(
    operations: list[TrajectoryOp],
    psi: qx.StateVector,
    key: Array,
) -> tuple[qx.StateVector, Array]:
    """Build a one-off trajectory kernel for *operations* and apply it to *psi*.

    Convenience wrapper around :func:`_build_trajectory_kernel` for callers that
    run a single (batch of) trajectories; batched sampling reuses one kernel
    across batches instead — see :func:`_run_batched_trajectories`.
    """
    return _build_trajectory_kernel(operations, psi.dims)(psi, key)


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
    num_trajectories: int,
    batch_size: int,
    random_seed: int,
    keep_states: bool = True,
    *,
    dims: tuple[int, ...],
    devices: list[jax.Device] | None = None,
) -> tuple[list[qx.StateVector] | None, list[Array]]:
    """Run trajectory simulation in batches, optionally sharded across devices.

    When *devices* contains more than one device a :class:`jax.sharding.Mesh`
    is constructed and both the initial state vector and PRNG keys are sharded
    along the trajectory (ensemble) axis.  XLA's SPMD partitioner then
    distributes the work so that each device processes its own slice.

    Every batch runs at the same width (:func:`_build_trajectory_kernel` compiled
    once, reused across batches) so the GPU sees one upfront compile rather than a
    recompile spike per batch; the final short batch is padded up to that width
    and its extra rows are sliced off.

    :param dims: Per-qudit register dimensions of the simulated system.
    """
    mesh = _make_mesh(devices)
    n_devices = len(mesh.devices.flat) if mesh is not None else 1
    sharding = NamedSharding(mesh, PartitionSpec("traj")) if mesh is not None else None  # type: ignore[no-untyped-call]

    # Build the jitted trajectory kernel once; reuse it for every batch.
    run = _build_trajectory_kernel(operations, dims)

    # Uniform per-batch width: full batches and the padded tail all share it, so
    # the kernel compiles exactly once.  Capped at the total so single-batch runs
    # don't pad past what's asked for; rounded to n_devices for even sharding.
    batch_width = _round_up_to(min(num_trajectories, batch_size), n_devices)

    key = jax.random.key(random_seed)
    all_psis: list[qx.StateVector] = []
    all_outcomes: list[Array] = []

    remaining = num_trajectories
    while remaining > 0:
        this_batch = min(remaining, batch_size)
        key, batch_key = jax.random.split(key)

        if batch_width == 1:
            psi = qx.zero_state_vector(dims=dims)
            batch_keys = batch_key
        else:
            psi = qx.zero_state_vector(dims=dims, ensemble_size=(batch_width,))
            batch_keys = batch_key

        # Shard state and key across devices when a mesh is available.
        if sharding is not None:
            psi = qx.StateVector.from_matrix(jax.device_put(psi.matrix, sharding), psi.dims)
            batch_keys = jax.device_put(jax.random.split(batch_key, batch_width), sharding)

        psi_out, outcomes = run(psi, batch_keys)
        psi_out.matrix.block_until_ready()

        # Strip padding rows down to this batch's real trajectory count.
        if batch_width > 1 and this_batch < batch_width:
            psi_out = qx.StateVector.from_matrix(psi_out.matrix[:this_batch], psi_out.dims)
            outcomes = outcomes[:this_batch]

        if this_batch == 1 and batch_width == 1:
            psi_out = qx.StateVector.from_matrix(psi_out.matrix[jnp.newaxis], psi_out.dims)
            outcomes = outcomes[jnp.newaxis]

        if keep_states:
            all_psis.append(psi_out)
        all_outcomes.append(outcomes)
        remaining -= this_batch

    return (all_psis if keep_states else None), all_outcomes


# ══════════════════════════════════════════════════════════
# Dynamic-shape trajectory simulator
# ══════════════════════════════════════════════════════════


def _dyn_apply(
    op: qx.Unitary | qx.KrausMap | qx.QuantumInstrument,
    psi: qx.StateVector,
    subsystem: tuple[int, ...],
    key: Array,
    squeeze_tol: float,
) -> tuple[qx.StateVector, Array | None]:
    """Apply one trajectory operator with dynamic per-subsystem dimensions.

    The reconciliation is *grow state → apply → squeeze state*:

    1. The state is grown via :func:`quax.promote` only where the operator exceeds it.
       Operators are applied at their authored dimension — they are never squeezed, since
       squeezing an operator is ill-defined: it acts non-trivially on levels the state may
       never populate (see :mod:`quax._squeeze`).
    2. The matching ``quax`` kernel is applied; it promotes the operator up to the state's
       dimensions, never shrinking it.
    3. The *state* is squeezed (a well-defined operation) to reclaim any leakage level the
       operator left empty.  So an ideal gate authored on a qutrit register transiently
       grows the state and then squeezes straight back, while a genuine leakage op leaves
       population behind that survives the squeeze.  The squeeze is skipped while every qudit
       is already at the qubit floor, so a purely no-leakage trajectory pays nothing for it.

    :param squeeze_tol: Tolerance for squeezing emptied leakage levels out of the state.
    :return: ``(state, outcome)`` where ``outcome`` is the sampled measurement
        result for an instrument, else ``None``.
    """
    current = tuple(psi.dims[q] for q in subsystem)
    target = tuple(max(c, e) for c, e in zip(current, op.dims[0], strict=True))
    if target != current:
        grown = list(psi.dims)
        for q, t in zip(subsystem, target, strict=True):
            grown[q] = t
        psi = qx.promote(psi, tuple(grown))

    if isinstance(op, qx.Unitary):
        psi, outcome = qx.targeted_apply_unitary(op, psi, subsystem), None
    elif isinstance(op, qx.KrausMap):
        psi, _ = _sample_kraus_map_trajectory(op, psi, key, subsystem)
        outcome = None
    elif isinstance(op, qx.QuantumInstrument):
        psi, outcome = qx.targeted_apply_instrument_to_state_vector(op, psi, key, subsystem)
    else:
        raise TypeError(f"DynamicTrajectorySimulator cannot apply operator of type {type(op).__name__}.")

    # Reclaim any leakage level the operator left empty by squeezing the *state*.  No qudit
    # can shrink below the qubit floor, so skip the work entirely while none has leaked.
    if any(d > 2 for d in psi.dims):
        psi = cast(qx.StateVector, qx.squeeze(psi, squeeze_tol))
    return psi, outcome


class DynamicTrajectorySimulator(_TrajectorySimulator):
    """Single-trajectory simulator with dynamically-sized qudit dimensions.

    Targets the **largest** leakage-aware registers.  Where the other simulators
    fix a global Hilbert-space shape, this one keeps a per-subsystem dimension
    vector that drifts at runtime: a qudit is grown to dimension 3 only when an
    operator can populate its leakage level, and squeezed back to 2 once that
    level empties (via :func:`quax.squeeze`).  In the realistic low-leakage
    regime only a handful of qudits occupy ``|2>`` at once, so the stored state
    stays far below the full ``3**n``.

    The simulation is **eager** — it applies one operator at a time and cannot be
    ``jax.jit``/``jax.grad``-compiled (the shapes are data-dependent) — and runs a
    single trajectory per :meth:`compute` call (scalar PRNG key, no ensemble).
    Squeeze is tolerance-based, so results carry a bounded truncation error set by
    ``squeeze_tol``.

    ``max_subsystem_size`` defaults to 1: chains of single-qudit gates on one line
    are still fused (no extra qudits pinned), but multi-qudit gates are left
    un-merged so that a leakage channel never pins its neighbours to dimension 3.

    Example::

        sim = DynamicTrajectorySimulator(program, noise_model=leakage_model)
        params = sim.linearize(memory_map)
        psi, outcomes = sim.compute(params, jax.random.key(0))
        shots = sim.sample(params, num_trajectories=1000)
    """

    def __init__(
        self,
        program: Program,
        qubits: list[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 1,
        kraus_truncation_threshold: float = 1e-6,
        squeeze_tol: float = 1e-9,
    ) -> None:
        super().__init__(
            program,
            qubits,
            noise_model=noise_model,
            max_subsystem_size=max_subsystem_size,
            kraus_truncation_threshold=kraus_truncation_threshold,
        )
        self._squeeze_tol = squeeze_tol

    def _validate(self, program: Program) -> None:
        """Measurements, resets, and noise are all supported (like TrajectorySimulator)."""

    def compute(  # type: ignore[override]
        self,
        params: Array | None = None,
        key: Array | None = None,
    ) -> tuple[qx.StateVector, Array]:
        """Run a single dynamic-shape trajectory.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :param key: Scalar JAX PRNG key for this trajectory (required).
        :return: ``(state_vector, measurement_outcomes)``.  The state's per-qudit
            dimensions reflect whatever leakage survived squeezing; outcomes has
            shape ``(n_measurements,)`` in program order.
        """
        if key is None:
            raise ValueError("DynamicTrajectorySimulator.compute requires a JAX PRNG key.")
        operations = self.adapt(self.compress(self.resolve(self._default_params(params))))

        psi = qx.zero_state_vector(dims=(2,) * self.n_qubits)
        outcomes: list[Array] = []
        # The state's shape changes as qudits grow and squeeze, so quax's jitted apply kernels
        # compile once per distinct (shape, subsystem) they see. In the low-leakage regime that
        # set is small and bounded — the deterministic no-leak shape sequence plus a few
        # one-qubit-leaked shapes — so the compilation is a fixed upfront cost that amortizes
        # over the many trajectories a sampling run takes. We therefore keep jit enabled.
        for idx, (op, subsystem) in enumerate(operations):
            psi, outcome = _dyn_apply(op, psi, tuple(subsystem), jax.random.fold_in(key, idx), self._squeeze_tol)
            if outcome is not None:
                outcomes.append(outcome)

        if outcomes:
            return psi, jnp.stack(outcomes, axis=-1).astype(jnp.int32)
        return psi, jnp.empty((0,), dtype=jnp.int32)

    def __call__(self, params: Array | None = None, key: Array | None = None) -> tuple[qx.StateVector, Array]:
        return self.compute(params, key)

    def sample(
        self,
        params: Array | None = None,
        num_trajectories: int = 1000,
        random_seed: int = 0,
    ) -> Array:
        """Run trajectories sequentially, returning only measurement outcomes.

        Dynamic per-trajectory shapes preclude ``vmap`` batching, so trajectories
        run one at a time.

        :param params: Flat parameter vector from :meth:`linearize`.
        :param num_trajectories: Number of trajectories to simulate.
        :param random_seed: Seed for the JAX PRNG.
        :return: Measurement outcomes with shape ``(num_trajectories, n_measurements)``.
        """
        key = jax.random.key(random_seed)
        outcomes = [self.compute(params, jax.random.fold_in(key, t))[1] for t in range(num_trajectories)]
        return jnp.stack(outcomes, axis=0)
