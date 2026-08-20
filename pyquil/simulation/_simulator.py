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
dimension inference, ``linearize``/``resolve``/``compress``).  This module provides
the **differentiable** family (:class:`_DifferentiableSimulator`) — jit/grad-friendly evolution
of a compressed ``Unitary``/``SuperOp`` stack, with measurements represented as dephasing
SuperOps:

* :class:`PureStateVectorSimulator` — gate-only programs (no noise, measurements,
  or resets).
* :class:`DensityMatrixSimulator` — any program, optionally with noise.

The ``compute`` method is the main entry point; for the differentiable family it can be
passed directly to ``jax.jit`` or ``jax.grad``.

.. warning::
    **Qubit ordering is big-endian here, unlike the rest of pyQuil.**  The first entry of
    the simulator's ``qubits`` list is the *most* significant subsystem of the returned
    state, so for ``qubits=[0, 1]`` the program ``X 0`` yields the basis state ``|10>``
    (index 2).  Every other simulator in pyQuil -- :class:`~pyquil.api.WavefunctionSimulator`,
    :class:`~pyquil.simulation.ReferenceWavefunctionSimulator`, the QVM -- is little-endian,
    where qubit 0 is the least significant bit and ``X 0`` gives ``|01>`` (index 1).

    This is a deliberate departure: tying subsystem order to the ``qubits`` list makes the
    state's axes read in the same order the register is written, which removes a persistent
    source of confusion when working with multi-qudit registers.  It does mean amplitudes
    must be reversed to compare against pyQuil's other simulators or against QVM readout
    bit order.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import quax as qx
from jax import Array

from pyquil.api import MemoryMap
from pyquil.noise._noise_model import NoiseModelLike
from pyquil.quil import Program
from pyquil.quilbase import Measurement, Reset, ResetQubit
from pyquil.simulation._resolver import (
    FixedOp,
    ParametricGate,
    ResolvedOp,
    build_dag,
    compressor_from_dag,
    enumerate_bases,
    resolve_for_differentiable,
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
    ``compress`` closures.  The :class:`_DifferentiableSimulator` family base specialises it
    for the state-vector and density-matrix simulators, supplying the execution
    machinery each needs.

    Subclasses override :meth:`_validate` and :meth:`compute`.

    Instances are treated as immutable after construction.

    .. note::
        This is deliberately a shared *base class* rather than a composed "simulation plan"
        object that each simulator holds. Composition would model the relationship more
        faithfully — the preprocessing is arguably a has-a, and a plan could be built once and
        evaluated several ways — but with only two consumers it would add a layer of indirection
        without paying for itself. Revisit if a third and fourth simulator arrive and genuinely
        want to share a prepared program.
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
        """Expand *program* into a compressed operator stack.

        :param program: The Quil program to simulate. May contain ``DEFGATE`` and
            ``DEFCIRCUIT`` definitions, which are expanded.
        :param qubits: Explicit register, in the order the state's subsystems will follow.
            Defaults to the program's qubits in ascending order. Pass this to include a
            qubit the program never names (an idle spectator), to fix a specific ordering,
            or when a ``DEFCIRCUIT`` body names a literal qubit -- those are not discovered
            automatically.
        :param noise_model: Optional noise model. Channels are looked up per instruction;
            instructions with no channel are simulated ideally.
        :param max_subsystem_size: Largest number of qudits the compressor may merge
            adjacent operations onto. Larger values mean fewer, bigger operators: usually
            faster to run and slower to compile. Purely a performance knob -- results are
            independent of it.
        :param dims: Per-qudit dimensions, in ``qubits`` order. Defaults to inferring them
            from the program's gates and channels (2 unless something says otherwise).
        :raises ValueError: If ``qubits`` contains duplicates.

        ``MEASURE`` is always resolved to a dephasing superoperator here: every simulator in
        this module is differentiable, and an instrument is not. The trajectory simulators
        select the instrument representation through
        :func:`~pyquil.simulation._resolver.resolve_for_trajectory` instead, so there is no
        mode to configure on this class.
        """
        self._validate(program)

        if qubits is None:
            qubits = sorted(program.get_qubit_indices())
        elif len(set(qubits)) != len(qubits):
            duplicates = sorted({q for q in qubits if qubits.count(q) > 1})
            raise ValueError(
                f"qubits contains duplicate entries {duplicates}: {qubits}. Each qubit must "
                "appear exactly once, since the list defines the register's subsystems."
            )
        self.qubits = qubits
        self.n_qubits = len(qubits)

        # Expand the program into operators, inferring register dimensions when not
        # supplied.  The differentiable simulators resolve measurements to dephasing SuperOps.
        res = resolve_for_differentiable(program, noise_model, qubits, dims)
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

        self._compress_fn = compressor_from_dag(dag, max_subsystem_size, dims=self.dims)

    # -- hook for subclass validation ---------------------

    def _validate(self, program: Program) -> None:
        """Override to reject unsupported instructions."""

    # -- public pipeline methods --------------------------

    def linearize(self, memory_map: MemoryMap) -> Array:
        """Convert a memory map to a flat JAX parameter vector.

        The vector's layout is fixed at construction (the order in which parametric gates
        were expanded), so it is the vector every ``compute``/``resolve`` call expects.

        :param memory_map: Values for each declared memory region, as passed to the QVM.
        :return: A flat ``float`` vector with one entry per runtime parameter.
        """
        return self._linearize_fn(memory_map)

    def _default_params(self, params: Array | None) -> Array:
        """Return *params*, validated against the program's parameter count.

        Lets callers omit ``params`` for parameter-free programs. Anything else -- a
        parametric program called with no vector, or a vector of the wrong length -- is a
        caller error, and is reported as one: the underlying ``jax`` indexing would
        otherwise fail with an opaque gather-out-of-range message, or silently ignore
        trailing values.

        :raises ValueError: If ``params`` has the wrong length, or is omitted for a
            program that takes parameters.
        """
        if params is None:
            if self._n_params:
                raise ValueError(
                    f"This program has {self._n_params} parameter(s); params cannot be omitted. "
                    "Build the vector with linearize(memory_map)."
                )
            return jnp.array([], dtype=float)

        params = jnp.asarray(params)
        if params.shape != (self._n_params,):
            raise ValueError(
                f"Expected {self._n_params} parameter(s) for this program, got shape "
                f"{tuple(params.shape)}. Build the vector with linearize(memory_map)."
            )
        return params

    def resolve(self, params: Array) -> list[ResolvedOp]:
        """Resolve parameters into one operator per DAG node.

        :param params: Flat parameter vector from :meth:`linearize`.
        :return: One ``(operator, subsystem)`` pair per expanded operation, in program order.
        """
        return self._resolve_fn(params)

    def compress(self, resolved: list[ResolvedOp]) -> list[ResolvedOp]:
        """Merge operators via greedy edge contraction.

        :param resolved: Operators from :meth:`resolve`.
        :return: Merged operators, one per compressor group, in application order. A merged
            group's subsystem is sorted; an unmerged operation keeps its own operand order.
        """
        return self._compress_fn(resolved)

    def compute(self, params: Array | None = None, **kwargs: Any) -> Any:
        """Compute the simulation result.  Subclasses must override.

        :param params: Flat parameter vector from :meth:`linearize`; omit for a
            parameter-free program.
        :param kwargs: Subclass-specific options.
        :return: The simulated result (a state vector, density matrix, ...).
        """
        raise NotImplementedError


# ══════════════════════════════════════════════════════════
# Differentiable family base (state-vector / density-matrix)
# ══════════════════════════════════════════════════════════


class _DifferentiableSimulator(ProgramSimulator):
    """Base for the jit/grad-friendly state-vector and density-matrix simulators.

    Adds the compressed-stack evolution machinery.  It enumerates the distinct
    *base subsystems* the compressor emits (:func:`enumerate_bases`) and applies the
    operator stack with a :func:`jax.lax.scan` whose body dispatches each operator to
    the :func:`jax.lax.switch` branch for its base (``self._branches``, keyed by
    ``self._idx_arr``), so the compiled graph size scales with the number of distinct
    base subsystems rather than the number of operations.

    Measurements are dephasing SuperOps, so they merge with neighbouring operations like any
    other superoperator; the compressor's convexity check is what preserves their ordering.
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
        """Set up the compressed-stack evolution machinery.

        Arguments are as :meth:`ProgramSimulator.__init__`.
        """
        super().__init__(
            program,
            qubits,
            noise_model=noise_model,
            max_subsystem_size=max_subsystem_size,
            dims=dims,
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

        :param state: The state to evolve (a ``StateVector`` or ``DensityMatrix``).
        :param op_stack: Operator matrices, one row per compressed operation, zero-padded
            to a common size and ordered as the compressor emits them.
        :return: The evolved state, of the same type as *state*.
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


def _embed_op_to_group(
    op: FixedOp,
    target_dims: tuple[int, ...],
    positions: tuple[int, ...],
    width: int,
    *,
    as_superop: bool,
) -> Array:
    """Embed *op* into a merge group, optionally lift to a superoperator, and pad to ``width``.

    :func:`quax.embed` places ``op`` (whose qudits map to ``positions`` within the group) into
    the group Hilbert space ``target_dims``.  With ``as_superop`` the embedded operator is then
    converted with :func:`quax.to_superop`; embedding and lifting commute for a unitary
    (``to_superop`` is a homomorphism), so doing the lift last lets the density-matrix path
    reuse this whole helper unchanged.  The trailing pad to ``width`` — the stack width shared
    by every group, ``d_max`` for unitaries and ``d_max**2`` for superoperators — is plain
    array padding with no quax equivalent.

    This is traceable, so it serves both the eager constant path and the vmapped parametric
    path.
    """
    embedded: FixedOp = qx.embed(op, target_dims=target_dims, positions=positions)
    if as_superop:
        embedded = qx.to_superop(embedded)
    matrix = embedded.matrix
    return jnp.pad(matrix, [(0, width - size) for size in matrix.shape])


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
    width: int
    #: Whether members are lifted to superoperators after embedding.
    as_superop: bool
    #: Sorted-array positions this batch fills, one per member.
    positions: list[int] = field(default_factory=list)
    #: Parameter-vector index for each free argument, one list per member.
    param_indices: list[list[int]] = field(default_factory=list)

    def builder(self) -> Callable[[Array], Array]:
        """Return ``params -> (n_members, width, width)`` embedded gate matrices."""
        concrete = {slot for slot, _ in self.concrete_args}
        free_slots = [j for j in range(self.n_args) if j not in concrete]
        gate_fn, n_args, concrete_args = self.gate_fn, self.n_args, self.concrete_args
        target_dims, group_positions = self.target_dims, self.group_positions
        width, as_superop = self.width, self.as_superop
        param_indices = jnp.asarray(self.param_indices)  # (n_members, n_free)

        def single(free_values: Array) -> Array:
            args: list[Any] = [None] * n_args
            for slot, val in concrete_args:
                args[slot] = val
            for k, slot in enumerate(free_slots):
                args[slot] = free_values[k]
            return _embed_op_to_group(gate_fn(*args), target_dims, group_positions, width, as_superop=as_superop)

        batched = jax.vmap(single)
        return lambda params: batched(params[param_indices])


def _make_group_fold(group_start: list[int], n_ops: int, width: int) -> Callable[[Array], Array]:
    """Build the per-group matrix-product fold.

    ``fold(raw)`` takes ``(n_ops, width, width)`` embedded matrices laid out in group order and
    returns ``(n_groups, width, width)`` — the ordered matrix product of each group's members.
    Groups are gathered into a padded ``(n_groups, max_size, ...)`` array (short groups padded
    with an identity sentinel) so every group folds under a single ``jax.vmap``.

    The fold is representation-agnostic: composing superoperators is the same left-multiplication
    as composing unitaries, so only ``width`` changes between the two simulators.
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
    eye = jnp.eye(width, dtype=complex)

    def group_product(mats: Array) -> Array:
        final, _ = jax.lax.scan(lambda acc, m: (m @ acc, None), eye, mats)
        return final

    def fold(raw: Array) -> Array:
        padded = jnp.concatenate([raw, eye[None]], axis=0)[gather_jax]  # (n_groups, max_size, d, d)
        return jax.vmap(group_product)(padded)

    return fold


def _build_vectorized_operator_constructor(
    expanded_ops: tuple[Any, ...],
    raw_subsystems: tuple[tuple[int, ...], ...],
    emit_order: list[tuple[int, list[int], tuple[int, ...]]],
    dims: tuple[int, ...],
    d_max: int,
    *,
    as_superop: bool,
) -> Callable[[Array], Array]:
    """Build a JIT-friendly constructor for the compressed operator stack.

    Returns ``build(params) -> (n_groups, width, width)`` with ``width = d_max`` for unitaries
    and ``d_max ** 2`` for superoperators: one matrix per merge group, equal to
    ``compress(resolve(params))`` but assembled so the traced graph scales with the number of
    distinct gate *kinds* rather than the number of gates.  Each operation is embedded into its
    merge group's Hilbert space, then each group's members are folded together via
    :func:`_make_group_fold`.

    This is the single most performance-critical piece of the module, and the reason for its
    complexity is measured rather than assumed.  Building the stack the obvious way — a Python
    comprehension over ``compress(resolve(params))`` — puts one traced operation per *gate* in
    the graph, and XLA compile time grows superlinearly: at 12 qubits x 20 layers that is
    **180 s versus 0.35 s**, a 518x difference (XLA itself prints "Very slow compile?").  The
    density-matrix simulator used the obvious construction until this was generalized, and
    compiled 51x slower than the state-vector one for parametric programs.

    :param expanded_ops: Operators from expansion, one per DAG node.
    :param raw_subsystems: Each operator's own qubit tuple, in operand order.
    :param emit_order: The compressor's ``(root, nodes, subsystem)`` groups, in emit order.
    :param dims: Per-qudit dimensions of the whole register.
    :param d_max: Largest group Hilbert-space dimension.
    :param as_superop: Lift every operation to a superoperator (density-matrix evolution).
    """
    n_ops = len(expanded_ops)
    width = d_max * d_max if as_superop else d_max

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
                    width=width,
                    as_superop=as_superop,
                )
                batches[key] = batch
            batch.positions.append(pos)
            batch.param_indices.append([pi for pi in op.param_indices if pi >= 0])
        else:
            # Constant operations are embedded once, eagerly. In superoperator mode this
            # also covers the non-unitary ops a noise model contributes (channel SuperOps,
            # and instrument total channels): ``qx.embed`` handles them, and ``to_superop``
            # is idempotent, so they need no special case.
            const_positions.append(pos)
            const_mats.append(_embed_op_to_group(op, target_dims, group_positions, width, as_superop=as_superop))

    builders = [(np.asarray(b.positions), b.builder()) for b in batches.values()]
    const_pos_arr = np.asarray(const_positions) if const_positions else None
    const_stack = jnp.stack(const_mats) if const_mats else None

    fold = _make_group_fold(group_start, n_ops, width)

    def build(params: Array) -> Array:
        raw = jnp.zeros((n_ops, width, width), dtype=complex)
        for positions, builder in builders:
            raw = raw.at[positions].set(builder(params))
        if const_stack is not None:
            raw = raw.at[const_pos_arr].set(const_stack)
        return fold(raw)

    return build


# ══════════════════════════════════════════════════════════
# Pure state-vector simulator
# ══════════════════════════════════════════════════════════


class PureStateVectorSimulator(_DifferentiableSimulator):
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
        """Prepare a pure-state simulator for a gate-only program.

        :param program: A Quil program of unitary operations only. Measurements, resets and
            noise channels are rejected -- including ones reached through a ``DEFCIRCUIT``.
            Use :class:`DensityMatrixSimulator` for those.
        :param qubits: Explicit register in state-subsystem order; see
            :meth:`ProgramSimulator.__init__`. Remember the ordering is big-endian.
        :param max_subsystem_size: Compressor merge budget; performance only. See
            :meth:`ProgramSimulator.__init__`.
        :raises ValueError: If the program contains a non-unitary operation.
        """
        super().__init__(program, qubits, noise_model=None, max_subsystem_size=max_subsystem_size)
        self._validate_expanded()
        self._psi0 = qx.zero_state_vector(dims=self.dims)

        # Vectorized gate construction (vmap per gate type) followed by a
        # segmented matmul scan for compression.  This gives both fast
        # compilation (small traced graph) AND fast runtime (compressed
        # op count in the state-evolution scan).
        emit_order = getattr(self._compress_fn, "emit_order", [])
        self._vmapped_build_fn = _build_vectorized_operator_constructor(
            self._expanded_ops,
            self._raw_subsystems,
            emit_order,
            self.dims,
            self.d_max,
            as_superop=False,
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

    def _validate_expanded(self) -> None:
        """Reject any expanded operation that is not a pure unitary.

        ``_validate`` sees only top-level instructions, so a ``MEASURE`` or ``RESET`` hidden
        in a ``DEFCIRCUIT`` body slips past it and would otherwise surface much later as a
        shape error from the unitary-only vectorized builder. Checking the expanded stack
        catches those, and anything else non-unitary a noise model or future instruction
        type might introduce.
        """
        for op, subsystem in zip(self._expanded_ops, self._raw_subsystems, strict=True):
            if not isinstance(op, (qx.Unitary, ParametricGate)):
                raise ValueError(
                    f"PureStateVectorSimulator supports unitary operations only, but the "
                    f"expanded program contains a {type(op).__name__} on qubit(s) "
                    f"{list(subsystem)}. Measurements, resets and noise channels require "
                    "DensityMatrixSimulator; note that these may come from a DEFCIRCUIT body."
                )

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


class DensityMatrixSimulator(_DifferentiableSimulator):
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
        """Prepare a density-matrix simulator.

        :param program: Any Quil program, including measurements and resets. ``MEASURE`` is
            applied as a dephasing channel: the resulting state is the correct *reduced*
            density matrix averaged over outcomes, but no classical outcome is recorded, so
            the register written by the measurement is not simulated.
        :param qubits: Explicit register in state-subsystem order; see
            :meth:`ProgramSimulator.__init__`. Remember the ordering is big-endian.
        :param noise_model: Optional noise model. Instructions with no channel are ideal.
        :param max_subsystem_size: Compressor merge budget; performance only. See
            :meth:`ProgramSimulator.__init__`.
        """
        super().__init__(program, qubits, noise_model=noise_model, max_subsystem_size=max_subsystem_size)
        self._rho0 = qx.zero_state_matrix(dims=self.dims)

        # Same vectorized construction as the state-vector simulator, lifted to
        # superoperators.  Building the stack per-operation instead costs ~50x in JIT
        # compile time for parametric programs; see
        # :func:`_build_vectorized_operator_constructor`.
        self._vmapped_build_fn = _build_vectorized_operator_constructor(
            self._expanded_ops,
            self._raw_subsystems,
            getattr(self._compress_fn, "emit_order", []),
            self.dims,
            self.d_max,
            as_superop=True,
        )

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
        # No operations (e.g. empty program) → the initial state is the result.
        if not self._branches:
            return self._rho0

        op_stack = self._vmapped_build_fn(self._default_params(params))
        return self.apply(self._rho0, op_stack)

    def __call__(self, params: Array | None = None) -> qx.DensityMatrix:
        return self.compute(params)
