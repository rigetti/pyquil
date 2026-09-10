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

.. warning::
    **Experimental.**  This module is deliberately private (``pyquil.simulation._simulator``)
    and its API is not stable: names, signatures and return types may change in any release
    before pyQuil 5.  It is published so the simulators can be exercised in real work and the
    design settled against that experience.

All simulators share the preprocessing in :class:`ProgramSimulator` (expansion, dimension
inference, ``linearize``/``resolve``/``compress``).  This module provides the
**differentiable** family (:class:`_DifferentiableSimulator`) — jit/grad-friendly evolution of
a compressed ``Unitary``/``SuperOp`` stack, with measurements collapsed to dephasing SuperOps:

* :class:`PureStateVectorSimulator` — gate-only programs (no noise, measurements,
  or resets).
* :class:`DensityMatrixSimulator` — any program, optionally with noise.

The ``compute`` method is the main entry point; for the differentiable family it can be
passed directly to ``jax.jit`` or ``jax.grad``, and returns a quax state on which any quax
metric can be evaluated inside the same transformation::

    sim = DensityMatrixSimulator(program, noise_model=noise_model)
    loss = lambda params: 1 - qx.fidelity(target, sim.compute(params))
    jax.grad(loss)(sim.linearize(memory_map))

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
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Generic, TypeVar, cast, final

import jax
import jax.numpy as jnp
import numpy as np
import quax as qx
from jax import Array

from pyquil.api import MemoryMap
from pyquil.noise._noise_model import NoiseModelLike
from pyquil.quil import Program
from pyquil.quilbase import Measurement, Reset, ResetQubit
from pyquil.simulation._circuit import Circuit, Group, MergePlan
from pyquil.simulation._resolver import (
    ExpandedOp,
    FixedOp,
    ParameterRef,
    ParametricGate,
    Resolution,
    resolve_program,
)

#: The state a differentiable simulator evolves.
StateT = TypeVar("StateT", qx.StateVector, qx.DensityMatrix)


# ══════════════════════════════════════════════════════════
# Base class
# ══════════════════════════════════════════════════════════


class ProgramSimulator(ABC):
    """Shared preprocessing base for program simulators.

    Handles the pipeline common to every backend: circuit expansion, qubit ordering,
    dimension inference, the parameter layout, and merge planning.  Subclasses implement
    :meth:`compute` and may override the hooks :meth:`_validate`, :meth:`_prepare_ops` and
    :meth:`_validate_ops`.  Everything else is ``@final``.

    A simulator is an **object constructed from a program**, not a function called on one,
    because efficient simulation builds several closures whose structure is fixed by the
    program (and noise model) but whose inputs are the runtime parameters.  Building them
    once lets the expensive analysis be shared across every evaluation -- a parameter sweep,
    a gradient, a batch of trajectories.

    Instances are immutable after construction.

    .. note::
        **Why an explicit** ``__init__`` **rather than a frozen dataclass with cached
        properties.**  The derived state here is eager *by requirement*, not habit.  The fused
        operator stack of a parameter-free program (``const_stack`` in
        :func:`_build_vectorized_operator_constructor`) has to be materialised outside any
        ``jax`` trace: built lazily on first use inside ``jax.jit(sim.compute)`` it becomes a
        traced constant and XLA constant-folds a per-gate composition subgraph, which is the
        180 s-versus-0.35 s compile-time regression measured on that function.  A
        ``cached_property`` is exactly such a lazy construction.  A frozen dataclass would
        therefore need a ``__post_init__`` that touches every cached property in dependency
        order to force it, and any property left off that list silently reintroduces the
        regression -- the same kind of implicit contract this class avoids elsewhere.  The
        constructor arguments (a ``Program``, a noise model) are also mutable and not
        meaningfully comparable, so the generated ``__eq__``/``__hash__`` would have to be
        disabled anyway.  The pure-data objects this class produces (:class:`Resolution`,
        :class:`~pyquil.simulation._circuit.MergePlan`,
        :class:`~pyquil.simulation._circuit.Circuit`) *are* frozen dataclasses.

    .. note::
        This is a shared *base class* rather than a composed "prepared program" that each
        simulator holds.  Composition models the relationship more faithfully, and is planned
        once all four simulators exist so the shared object is designed against every backend
        rather than two.
    """

    def __init__(
        self,
        program: Program,
        qubits: Sequence[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 2,
    ) -> None:
        """Expand *program* and plan its compressed operator stack.

        :param program: The Quil program to simulate. May contain ``DEFGATE`` and
            ``DEFCIRCUIT`` definitions, which are expanded.
        :param qubits: The register order: the state's subsystems follow this list, most
            significant first. Defaults to the program's qubits in ascending order. When
            given it must contain exactly the qubits the program acts on -- a qubit the program
            never touches would have no effect on the result, and one the list omits could not
            be simulated at all.
        :param noise_model: Optional noise model. Channels are looked up per instruction;
            instructions with no channel are simulated ideally.
        :param max_subsystem_size: Largest number of qudits the compressor may merge
            adjacent operations onto. Larger values mean fewer, bigger operators: usually
            faster to run and slower to compile. Purely a performance knob -- results are
            independent of it.
        :raises ValueError: If ``qubits`` contains duplicates or is not exactly the program's
            qubit set, or if the program contains an instruction the simulator does not
            support.
        """
        self._validate(program)
        self.qubits: tuple[int, ...] = self._register(program, qubits)

        # Expand the program into operators, inferring register dimensions.  Expansion is
        # backend-agnostic: a MEASURE always arrives as a QuantumInstrument, and a backend that
        # does not branch on outcomes collapses it in ``_prepare_ops``.
        resolution = resolve_program(program, noise_model, list(self.qubits))
        resolution = replace(resolution, ops=self._prepare_ops(resolution.ops))
        self._validate_ops(resolution.ops, resolution.subsystems)
        self._resolution = resolution
        self.dims: tuple[int, ...] = resolution.dims

        # Parameter layout: one slot per distinct memory reference, in slot order.
        self.parameters: tuple[ParameterRef, ...] = resolution.parameters
        self._slot_of: dict[ParameterRef, int] = {ref: i for i, ref in enumerate(self.parameters)}
        self._region_gathers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in dict.fromkeys(region for region, _ in self.parameters):
            pairs = [(slot, offset) for slot, (region, offset) in enumerate(self.parameters) if region == name]
            slots, offsets = zip(*pairs, strict=True)
            self._region_gathers[name] = (np.asarray(slots, dtype=np.int32), np.asarray(offsets, dtype=np.int32))

        # A surviving instrument has to stay addressable.  Fusing one into a neighbour would
        # preserve the circuit's overall channel, so the convexity check alone permits it --
        # but the outcome would no longer be observable.  A backend that collapsed its
        # instruments in ``_prepare_ops`` therefore has nothing atomic, and its measurements
        # merge like any other channel; one that kept them gets them pinned.
        atomic = tuple(i for i, op in enumerate(resolution.ops) if isinstance(op, qx.QuantumInstrument))

        # Merge planning is purely structural -- it needs the subsystems and nothing else.  The
        # plan is data: the vectorized stack builder reads its groups without ever
        # materialising a resolved operator.
        self.plan: MergePlan = MergePlan.greedy(resolution.subsystems, max_subsystem_size, atomic=atomic)

    @staticmethod
    def _register(program: Program, qubits: Sequence[int] | None) -> tuple[int, ...]:
        """Validate ``qubits`` against the program and return the register order."""
        program_qubits = set(program.get_qubit_indices())
        if qubits is None:
            return tuple(sorted(program_qubits))
        register = tuple(int(q) for q in qubits)
        if len(set(register)) != len(register):
            duplicates = sorted({q for q in register if register.count(q) > 1})
            raise ValueError(
                f"qubits contains duplicate entries {duplicates}: {list(register)}. Each qubit must "
                "appear exactly once, since the list defines the register's subsystems."
            )
        if set(register) != program_qubits:
            extra = sorted(set(register) - program_qubits)
            missing = sorted(program_qubits - set(register))
            raise ValueError(
                f"qubits must be exactly the qubits the program acts on, {sorted(program_qubits)}, in the "
                f"desired order; got {list(register)}"
                + (f" (not in the program: {extra})" if extra else "")
                + (f" (missing: {missing})" if missing else "")
                + "."
            )
        return register

    # -- hooks for subclasses -----------------------------

    def _validate(self, program: Program) -> None:  # noqa: B027 -- optional hook, not abstract
        """Reject unsupported top-level instructions before expansion.

        The default accepts everything :func:`~pyquil.simulation._resolver.expand_program`
        accepts.  Override to narrow, e.g. a unitary-only backend rejecting ``MEASURE``.
        """

    def _prepare_ops(self, ops: tuple[ExpandedOp, ...]) -> tuple[ExpandedOp, ...]:
        """Adapt the expanded operators to what this backend evolves.

        The default keeps them as expanded, which retains a ``MEASURE`` as a sampleable
        ``QuantumInstrument``.  Override to convert; see
        :meth:`_DifferentiableSimulator._prepare_ops`.
        """
        return ops

    def _validate_ops(  # noqa: B027 -- optional hook, not abstract
        self, ops: tuple[ExpandedOp, ...], subsystems: tuple[tuple[int, ...], ...]
    ) -> None:
        """Reject expanded operators this backend cannot evolve.

        Runs after :meth:`_prepare_ops` and before any stack is built, so a backend can check
        the *expanded* program -- which is where a ``MEASURE`` hidden in a ``DEFCIRCUIT`` body
        or a channel contributed by a noise model first becomes visible.  The default accepts
        everything.
        """

    # -- abstract interface -------------------------------

    @abstractmethod
    def compute(self, params: Array | None = None, **kwargs: Any) -> Any:
        """Compute the simulation result.

        :param params: Flat parameter vector from :meth:`linearize`; omit for a
            parameter-free program.
        :param kwargs: Backend-specific options.
        :return: The simulated result (a state vector, density matrix, ...).
        """

    # -- final public pipeline ----------------------------

    @property
    def num_parameters(self) -> int:
        """The length of the parameter vector :meth:`compute` expects."""
        return len(self.parameters)

    @final
    def parameter_index(self, name: str, offset: int = 0) -> int:
        """Return the slot of memory reference ``name[offset]`` in the parameter vector.

        Use this to read the component of a gradient that belongs to a given ``DECLARE``
        entry::

            grad = jax.grad(loss)(params)
            d_theta0 = grad[sim.parameter_index("theta", 0)]

        :param name: The memory region, as declared.
        :param offset: The index within the region.
        :return: The slot index.
        :raises KeyError: If the program's gates never read ``name[offset]``.
        """
        try:
            return self._slot_of[(name, offset)]
        except KeyError:
            known = ", ".join(f"{n}[{o}]" for n, o in self.parameters) or "none"
            raise KeyError(
                f"{name}[{offset}] is not a parameter of this program; its parameters are: {known}."
            ) from None

    @final
    def linearize(self, memory_map: MemoryMap) -> Array:
        """Convert a memory map to the flat parameter vector.

        The vector has one slot per distinct memory reference the program's gates read, in
        the order of :attr:`parameters`.  This is a pure gather over the region arrays, so it
        can be traced and differentiated: ``jax.grad(lambda t: loss(sim.compute(sim.linearize(
        {"theta": t}))))`` differentiates with respect to the declared memory directly.

        :param memory_map: Values for each declared memory region, as passed to the QVM.
        :return: A flat ``float`` vector with one entry per parameter slot.
        :raises KeyError: If a region the program reads is missing from ``memory_map``.
        :raises ValueError: If a region is too short for an offset the program reads.
        """
        values = jnp.zeros(len(self.parameters), dtype=float)
        for name, (slots, offsets) in self._region_gathers.items():
            if name not in memory_map:
                raise KeyError(f"memory_map has no region {name!r}; this program reads {sorted(self._region_gathers)}.")
            region = jnp.asarray(memory_map[name], dtype=float)
            if region.ndim != 1 or region.shape[0] <= int(offsets.max()):
                raise ValueError(
                    f"Region {name!r} has shape {tuple(region.shape)} but the program reads "
                    f"{name}[{int(offsets.max())}]."
                )
            values = values.at[slots].set(region[offsets])
        return values

    @final
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
        n = self.num_parameters
        if params is None:
            if n:
                raise ValueError(
                    f"This program has {n} parameter(s); params cannot be omitted. "
                    "Build the vector with linearize(memory_map)."
                )
            return jnp.array([], dtype=float)

        params = jnp.asarray(params)
        if params.shape != (n,):
            raise ValueError(
                f"Expected {n} parameter(s) for this program, got shape "
                f"{tuple(params.shape)}. Build the vector with linearize(memory_map)."
            )
        return params

    @final
    def resolve(self, params: Array) -> Circuit:
        """Resolve parameters into the program's circuit.

        :param params: Flat parameter vector from :meth:`linearize`.
        :return: A :class:`~pyquil.simulation._circuit.Circuit` with one operation per
            expanded operation, in program order.
        """
        return self._resolution.resolve(params)

    @final
    def compress(self, resolved: Circuit) -> Circuit:
        """Merge operations according to :attr:`plan`.

        :param resolved: The circuit from :meth:`resolve`.
        :return: The merged circuit, one operation per plan group, in application order. A
            merged group's subsystem is ascending; an unmerged operation keeps its own
            operand order.
        """
        return self.plan.apply(resolved)


# ══════════════════════════════════════════════════════════
# Differentiable family base (state-vector / density-matrix)
# ══════════════════════════════════════════════════════════


class _DifferentiableSimulator(ProgramSimulator, Generic[StateT]):
    """Base for the jit/grad-friendly state-vector and density-matrix simulators.

    Owns the whole evolution: it builds the fused operator stack with
    :func:`_build_vectorized_operator_constructor` and applies it with a :func:`jax.lax.scan`
    whose body dispatches each operator to the :func:`jax.lax.switch` branch for its base
    subsystem, so the compiled graph size scales with the number of distinct base subsystems
    (:attr:`~pyquil.simulation._circuit.MergePlan.bases`) rather than the number of operations.

    A concrete simulator supplies only its *representation*, through three abstract members:
    :attr:`_as_superop`, :meth:`_initial_state` and :meth:`_make_branch`.  :meth:`compute` is
    implemented here once and is final.

    Measurements are collapsed to dephasing SuperOps by :meth:`_prepare_ops`, so they merge
    with neighbouring operations like any other superoperator; the merge plan's convexity check
    is what preserves their ordering.
    """

    def __init__(
        self,
        program: Program,
        qubits: Sequence[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 2,
    ) -> None:
        """Set up the compressed-stack evolution machinery.

        Arguments are as :meth:`ProgramSimulator.__init__`.
        """
        super().__init__(program, qubits, noise_model=noise_model, max_subsystem_size=max_subsystem_size)

        # The merge structure depends only on the subsystems (not on parameter values), so
        # the base subsystems can be read straight off the plan -- no ``resolve``/``compress``
        # probe is required.
        self.bases: tuple[tuple[int, ...], ...] = self.plan.bases
        self.base_dims: tuple[tuple[int, ...], ...] = tuple(tuple(self.dims[q] for q in base) for base in self.bases)
        self.base_total_dim: tuple[int, ...] = tuple(math.prod(d) for d in self.base_dims)
        self.d_max: int = max(self.base_total_dim, default=1)
        self._idx_arr = jnp.asarray(self.plan.op_index, dtype=jnp.int32)

        # Vectorized gate construction (vmap per gate type) followed by a segmented matmul
        # fold for compression.  This gives both fast compilation (small traced graph) and
        # fast runtime (compressed op count in the state-evolution scan).
        self._build_stack: Callable[[Array], Array] = _build_vectorized_operator_constructor(
            self._resolution.ops,
            self._resolution.subsystems,
            self.plan.groups,
            self.dims,
            self.d_max,
            as_superop=self._as_superop,
        )

        # One switch branch per distinct base subsystem.  The base owns the table; the
        # concrete class only says how one branch rebuilds and applies its operator.
        self._branches: tuple[Callable[[Array, StateT], StateT], ...] = tuple(
            self._make_branch(base, base_dims, total_dim)
            for base, base_dims, total_dim in zip(self.bases, self.base_dims, self.base_total_dim, strict=True)
        )
        self._state0: StateT = self._initial_state()

    # -- abstract representation --------------------------

    @property
    @abstractmethod
    def _as_superop(self) -> bool:
        """Whether the stack holds superoperators (density matrix) or unitaries (state vector)."""

    @abstractmethod
    def _initial_state(self) -> StateT:
        """Return the all-zeros state on :attr:`dims`, in this backend's representation."""

    @abstractmethod
    def _make_branch(
        self, base: tuple[int, ...], base_dims: tuple[int, ...], total_dim: int
    ) -> Callable[[Array, StateT], StateT]:
        """Return the switch branch for one base subsystem.

        The branch receives one row of the fused stack -- a ``(width, width)`` matrix padded
        beyond ``total_dim`` (or ``total_dim ** 2`` for superoperators) -- and the current
        state, and returns the evolved state.

        :param base: Register indices the branch's operators act on.
        :param base_dims: Their per-qudit dimensions.
        :param total_dim: Their total Hilbert-space dimension.
        """

    # -- final implementation -----------------------------

    @final
    def _prepare_ops(self, ops: tuple[ExpandedOp, ...]) -> tuple[ExpandedOp, ...]:
        """Collapse every instrument to its total channel.

        Neither simulator in this family branches on a measurement outcome, and an instrument
        is not differentiable, so a ``MEASURE`` is evolved as the dephasing channel obtained by
        summing over its outcomes.  The resulting state is the correct outcome-averaged density
        matrix, but no classical outcome is recorded.

        This is :meth:`~pyquil.simulation._circuit.Circuit.to_superops` applied one operator
        early -- at construction rather than per ``resolve`` -- which is free, since an
        instrument never depends on a runtime parameter.  Doing it here rather than in expansion
        is what lets expansion stay backend-agnostic, and it is why this family has no atomic
        operations: once the instruments are gone there is nothing that must stay individually
        addressable.
        """
        return tuple(qx.to_superop(op.total_channel()) if isinstance(op, qx.QuantumInstrument) else op for op in ops)

    @final
    def compute(self, params: Array | None = None) -> StateT:  # type: ignore[override]
        """Compute the final state.

        The fused operator stack is built for *params* and applied with a :func:`jax.lax.scan`
        whose body dispatches each operator to the right base subsystem via
        :func:`jax.lax.switch`.  This keeps the traced graph size proportional to the number
        of distinct base subsystems rather than the number of operations, dramatically
        reducing JIT compilation time for large programs.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :return: The final state, in this backend's representation.
        """
        # No operations (e.g. empty program): the initial state is the result, and
        # ``lax.switch`` cannot be given zero branches.
        if not self._branches:
            return self._state0
        return self._apply(self._state0, self._build_stack(self._default_params(params)))

    @final
    def __call__(self, params: Array | None = None) -> StateT:
        """Alias for :meth:`compute`."""
        return self.compute(params)

    @final
    def _apply(self, state: StateT, op_stack: Array) -> StateT:
        """Apply a stack of operator matrices to *state* via a scan + switch.

        :param state: The state to evolve.
        :param op_stack: Operator matrices, one row per compressed operation, zero-padded
            to a common size and ordered as the plan emits them.
        :return: The evolved state.
        """
        branches = self._branches

        def body(state: StateT, xs: tuple[Array, Array]) -> tuple[StateT, None]:
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

    This is a mutable dataclass on purpose: it is an accumulator that
    :func:`_build_vectorized_operator_constructor` fills while it walks the plan, and it is
    consumed by :meth:`builder` immediately afterwards.
    """

    gate_fn: Callable[..., qx.Operator]
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
            gate = gate_fn(*args)
            if not isinstance(gate, qx.Unitary):
                gate = qx.Unitary.from_matrix(gate.matrix, gate.dims)
            return _embed_op_to_group(gate, target_dims, group_positions, width, as_superop=as_superop)

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
        final_mat, _ = jax.lax.scan(lambda acc, m: (m @ acc, None), eye, mats)
        return final_mat

    def fold(raw: Array) -> Array:
        padded = jnp.concatenate([raw, eye[None]], axis=0)[gather_jax]  # (n_groups, max_size, d, d)
        return jax.vmap(group_product)(padded)

    return fold


def _build_vectorized_operator_constructor(
    expanded_ops: tuple[ExpandedOp, ...],
    raw_subsystems: tuple[tuple[int, ...], ...],
    groups: tuple[Group, ...],
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

    :param expanded_ops: Operators from expansion, one per operation.
    :param raw_subsystems: Each operator's own qubit tuple, in operand order.
    :param groups: The plan's ``(operation indices, subsystem)`` groups, in application order.
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
    for nodes, subsystem in groups:
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


class PureStateVectorSimulator(_DifferentiableSimulator[qx.StateVector]):
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
        qubits: Sequence[int] | None = None,
        *,
        max_subsystem_size: int = 2,
    ) -> None:
        """Prepare a pure-state simulator for a gate-only program.

        :param program: A Quil program of unitary operations only. Measurements, resets and
            noise channels are rejected -- including ones reached through a ``DEFCIRCUIT``.
            Use :class:`DensityMatrixSimulator` for those.
        :param qubits: Register order; see :meth:`ProgramSimulator.__init__`. Remember the
            ordering is big-endian.
        :param max_subsystem_size: Compressor merge budget; performance only. See
            :meth:`ProgramSimulator.__init__`.
        :raises ValueError: If the program contains a non-unitary operation.
        """
        super().__init__(program, qubits, noise_model=None, max_subsystem_size=max_subsystem_size)

    @property
    def _as_superop(self) -> bool:
        return False

    def _validate(self, program: Program) -> None:  # noqa: B027 -- optional hook, not abstract
        for inst in program.instructions:
            if isinstance(inst, Measurement):
                raise ValueError(f"PureStateVectorSimulator does not support measurements.  Found: {inst}")
            if isinstance(inst, (Reset, ResetQubit)):
                raise ValueError(f"PureStateVectorSimulator does not support resets.  Found: {inst}")

    def _validate_ops(self, ops: tuple[ExpandedOp, ...], subsystems: tuple[tuple[int, ...], ...]) -> None:
        """Reject any expanded operation that is not a pure unitary.

        ``_validate`` sees only top-level instructions, so a ``MEASURE`` or ``RESET`` hidden
        in a ``DEFCIRCUIT`` body slips past it and would otherwise surface much later as a
        shape error from the unitary-only vectorized builder. Checking the expanded stack
        catches those, and anything else non-unitary a noise model or future instruction
        type might introduce.
        """
        for op, subsystem in zip(ops, subsystems, strict=True):
            if not isinstance(op, (qx.Unitary, ParametricGate)):
                raise ValueError(
                    f"PureStateVectorSimulator supports unitary operations only, but the "
                    f"expanded program contains a {type(op).__name__} on qubit(s) "
                    f"{list(subsystem)}. Measurements, resets and noise channels require "
                    "DensityMatrixSimulator; note that these may come from a DEFCIRCUIT body."
                )

    def _initial_state(self) -> qx.StateVector:
        return qx.zero_state_vector(dims=self.dims)

    def _make_branch(
        self, base: tuple[int, ...], base_dims: tuple[int, ...], total_dim: int
    ) -> Callable[[Array, qx.StateVector], qx.StateVector]:
        def branch(op_mat: Array, psi: qx.StateVector) -> qx.StateVector:
            unitary = qx.Unitary.from_matrix(op_mat[:total_dim, :total_dim], (base_dims, base_dims))
            return qx.targeted_apply_unitary(unitary, psi, base)

        return branch

    def unitary(self, params: Array | None = None) -> qx.Unitary:
        """Compute the full program unitary.

        This composes the merged circuit eagerly on the whole register and is exponentially
        expensive in the register size; it is meant for verification, not simulation.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :return: The full unitary matrix.
        """
        circuit = self.compress(self.resolve(self._default_params(params)))
        if circuit.num_ops == 0:
            d = math.prod(self.dims)
            return qx.Unitary.from_matrix(jnp.eye(d, dtype=complex), (self.dims, self.dims))
        return cast(qx.Unitary, circuit.compose())


# ══════════════════════════════════════════════════════════
# Density-matrix simulator
# ══════════════════════════════════════════════════════════


class DensityMatrixSimulator(_DifferentiableSimulator[qx.DensityMatrix]):
    """Density-matrix simulator for any program, optionally with noise.

    All methods are jit- and grad-friendly::

        sim = DensityMatrixSimulator(program, noise_model=noise_model)
        params = sim.linearize(memory_map)
        rho = jax.jit(sim.compute)(params)

    ``MEASURE`` is applied as a dephasing channel: the resulting state is the correct
    *reduced* density matrix averaged over outcomes, but no classical outcome is recorded, so
    the register written by the measurement is not simulated.  Arguments are as
    :meth:`ProgramSimulator.__init__`.
    """

    @property
    def _as_superop(self) -> bool:
        return True

    def _initial_state(self) -> qx.DensityMatrix:
        return qx.zero_state_matrix(dims=self.dims)

    def _make_branch(
        self, base: tuple[int, ...], base_dims: tuple[int, ...], total_dim: int
    ) -> Callable[[Array, qx.DensityMatrix], qx.DensityMatrix]:
        width = total_dim * total_dim

        def branch(op_mat: Array, rho: qx.DensityMatrix) -> qx.DensityMatrix:
            superop = qx.SuperOp.from_matrix(op_mat[:width, :width], (base_dims, base_dims))
            return qx.targeted_apply_superop(superop, rho, base)

        return branch


__all__ = [
    "DensityMatrixSimulator",
    "ProgramSimulator",
    "PureStateVectorSimulator",
    "Resolution",
]
