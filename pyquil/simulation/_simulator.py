##############################################################################
# Copyright 2026 Rigetti Computing
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
import warnings
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
from pyquil.simulation._circuit import Circuit, CircuitOp, Group, MergePlan
from pyquil.simulation._resolver import (
    ExpandedOp,
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


def _warn_if_matmul_precision_reduced() -> None:
    """Warn when an accelerator backend would run matrix products at reduced precision.

    On GPUs and TPUs JAX defaults to fast, reduced-precision matrix multiplication (TF32 or
    bfloat16 passes) unless ``jax_default_matmul_precision`` is ``"highest"``.  That is a poor
    fit for simulating quantum circuits, where errors compound over every operator applied.
    CPUs always multiply at full precision, so the flag is irrelevant there.
    """
    if jax.default_backend() == "cpu":
        return
    precision = jax.config.jax_default_matmul_precision # type: ignore
    if precision in ("highest", "float32"):
        return
    warnings.warn(
        f"JAX's default matmul precision is {precision!r} on the {jax.default_backend()!r} backend, which "
        "lets matrix products run at reduced (TF32/bfloat16) precision. Set "
        "jax.config.update('jax_default_matmul_precision', 'highest') (or JAX_DEFAULT_MATMUL_PRECISION=highest) "
        "before simulating.",
        stacklevel=3,
    )


class ProgramSimulator(ABC):
    """Shared preprocessing base for program simulators.

    Handles what every simulator does with a program before evolving anything: expanding it
    into operators, ordering the register, inferring qudit dimensions, laying out the runtime
    parameters, and planning which operators to merge.  Concrete simulators add the evolution
    itself through :meth:`compute`.

    A simulator is constructed **from a program** rather than called on one, because the
    expensive work — analysing the program and compiling the evolution — depends only on the
    program and noise model, not on the parameter values.  Doing it once lets a parameter
    sweep, a gradient or a batch of trajectories reuse it.

    Instances are immutable after construction.
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
            significant first. Defaults to the program's qubits in ascending order. If provided,
            it must contain exactly the qubits the program acts on.
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
        _warn_if_matmul_precision_reduced()
        self._validate(program)
        self.qubits: tuple[int, ...] = self._register(program, qubits)

        # Expand the program into operators, inferring register dimensions.  Expansion is
        # backend-agnostic: a MEASURE always arrives as a QuantumInstrument, and a backend that
        # does not branch on outcomes collapses it in ``_prepare_ops``.
        resolution = resolve_program(program, noise_model, list(self.qubits))
        resolution = replace(resolution, ops=self._prepare_ops(resolution.ops, resolution.subsystems))
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

    def _prepare_ops(
        self, ops: tuple[ExpandedOp, ...], subsystems: tuple[tuple[int, ...], ...]
    ) -> tuple[ExpandedOp, ...]:
        """Adapt the expanded operators to what this backend evolves.

        The default keeps them as expanded, which retains a ``MEASURE`` as a sampleable
        ``QuantumInstrument``.  Override to convert; see
        :meth:`_DifferentiableSimulator._prepare_ops`.  The returned tuple must have one entry
        per input operation, so that ``subsystems`` still lines up with it.

        :param ops: The expanded operators, in program order.
        :param subsystems: The register indices each operator acts on.
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
        the order of :attr:`parameters`.  The conversion is differentiable, so a loss can be
        differentiated with respect to the declared memory directly::

            jax.grad(lambda theta: loss(sim.compute(sim.linearize({"theta": theta}))))(theta)

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
    def resolve(self, params: Array | None = None) -> Circuit:
        """Resolve parameters into the program's circuit.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or pass ``None``)
            for a parameter-free program.
        :return: A :class:`~pyquil.simulation._circuit.Circuit` with one operation per
            expanded operation, in program order.
        """
        return self._resolution.resolve(self._default_params(params))

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

    The merged operators are built as one stack and applied to the initial state in a single
    compiled loop, so compile time depends on the number of distinct gate kinds and subsystem
    shapes rather than on the number of gates.  A concrete simulator supplies only its
    representation: whether operators are unitaries or superoperators (:attr:`_as_superop`),
    the initial state (:meth:`_initial_state`) and how one operator is applied
    (:meth:`_make_branch`).

    A measurement followed by further operations on its qudits is replaced by its total
    channel (:meth:`_prepare_ops`), so it merges with neighbouring operations like any other
    channel.  A *terminal* measurement -- one nothing acts on afterwards -- is held back
    instead: :meth:`compute` applies its total channel last, and the density-matrix backend
    reads the outcome distribution off the pre-measurement state in
    :meth:`DensityMatrixSimulator.outcome_probabilities`.
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
    def _prepare_ops(
        self, ops: tuple[ExpandedOp, ...], subsystems: tuple[tuple[int, ...], ...]
    ) -> tuple[ExpandedOp, ...]:
        """Collapse mid-circuit measurements to their total channel; hold terminal ones back.

        This family evolves the state deterministically, so a measurement whose qudits are
        acted on again later can only enter as the dephasing channel obtained by summing over
        its outcomes.  A terminal measurement -- one whose qudits nothing touches afterwards --
        is taken out of the operator stack (its slot becomes the identity, which the merge plan
        absorbs) and remembered in ``_terminal``.  :meth:`compute` applies its total channel
        after the stack, so the returned state is the same outcome-averaged state either way,
        while :meth:`DensityMatrixSimulator.outcome_probabilities` can still read the joint
        outcome distribution off the state just before the measurements.
        """
        prepared = list(ops)
        terminal: list[tuple[int, qx.QuantumInstrument, tuple[int, ...]]] = []
        for index, (op, subsystem) in enumerate(zip(ops, subsystems, strict=True)):
            if not isinstance(op, qx.QuantumInstrument):
                continue
            touched_later = any(set(subsystem) & set(later) for later in subsystems[index + 1 :])
            if touched_later:
                prepared[index] = qx.to_superop(op.total_channel())
            else:
                terminal.append((index, op, subsystem))
                in_dims = op.dims[1]
                identity = jnp.eye(math.prod(in_dims), dtype=complex)
                prepared[index] = qx.to_superop(qx.Unitary.from_matrix(identity, (in_dims, in_dims)))
        self._terminal: tuple[tuple[int, qx.QuantumInstrument, tuple[int, ...]], ...] = tuple(terminal)
        self._terminal_channels: tuple[tuple[qx.SuperOp, tuple[int, ...]], ...] = tuple(
            (qx.to_superop(op.total_channel()), subsystem) for _, op, subsystem in terminal
        )
        return tuple(prepared)

    @final
    def _pre_measurement_state(self, params: Array | None) -> StateT:
        """Evolve the initial state through the operator stack, terminal measurements excluded."""
        # No operations (e.g. empty program): the initial state is the result, and
        # ``lax.switch`` cannot be given zero branches.
        if not self._branches:
            return self._state0
        return self._apply(self._state0, self._build_stack(self._default_params(params)))

    @final
    def compute(self, params: Array | None = None) -> StateT:  # type: ignore[override]
        """Compute the final state.

        Builds the merged operators for *params* and applies them to the initial state.  The
        call can be wrapped in ``jax.jit``, ``jax.grad`` or ``jax.vmap``.

        A ``MEASURE`` contributes its total channel, so the returned density matrix is the
        state averaged over measurement outcomes; the outcome distribution itself is available
        from :meth:`DensityMatrixSimulator.outcome_probabilities`.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :return: The final state: a ``StateVector`` or a ``DensityMatrix``.
        """
        state = self._pre_measurement_state(params)
        for channel, subsystem in self._terminal_channels:
            # Only the density-matrix backend can hold an instrument, so ``state`` is a
            # ``DensityMatrix`` whenever this loop runs.
            state = cast(StateT, qx.targeted_apply_superop(channel, cast(qx.DensityMatrix, state), subsystem))
        return state

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
    op: CircuitOp,
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
    embedded: CircuitOp = qx.embed(op, target_dims=target_dims, positions=positions)
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

        This folds the merged circuit eagerly into one operator on the whole register and is exponentially
        expensive in the register size; it is meant for verification, not simulation.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :return: The full unitary matrix.
        """
        circuit = self.compress(self.resolve(self._default_params(params)))
        if circuit.num_ops == 0:
            d = math.prod(self.dims)
            return qx.Unitary.from_matrix(jnp.eye(d, dtype=complex), (self.dims, self.dims))
        return cast(qx.Unitary, circuit.full_operator())


# ══════════════════════════════════════════════════════════
# Density-matrix simulator
# ══════════════════════════════════════════════════════════


class DensityMatrixSimulator(_DifferentiableSimulator[qx.DensityMatrix]):
    """Density-matrix simulator for any program, optionally with noise.

    All methods are jit- and grad-friendly::

        sim = DensityMatrixSimulator(program, noise_model=noise_model)
        params = sim.linearize(memory_map)
        rho = jax.jit(sim.compute)(params)

    :meth:`compute` returns the density matrix averaged over measurement outcomes: a
    ``MEASURE`` enters as its total channel and no classical register is simulated.  The
    outcome statistics of the program's *terminal* measurements -- those nothing acts on
    afterwards, which is where a program normally reads out -- come from
    :meth:`outcome_probabilities`, which includes any readout error the noise model assigns
    to them::

        probs = jax.jit(sim.outcome_probabilities)(params)  # shape (2, 2) for two measured qubits
        probs[1, 0]  # P(qubit A -> 1, qubit B -> 0)

    Arguments are as :meth:`ProgramSimulator.__init__`.
    """

    def __init__(
        self,
        program: Program,
        qubits: Sequence[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 2,
    ) -> None:
        """Prepare a density-matrix simulator.

        Arguments are as :meth:`ProgramSimulator.__init__`.
        """
        super().__init__(program, qubits, noise_model=noise_model, max_subsystem_size=max_subsystem_size)
        for _, _, subsystem in self._terminal:
            if len(subsystem) != 1:
                raise NotImplementedError(
                    f"A measurement instrument on {len(subsystem)} qudits (register indices {subsystem}) is not "
                    "supported; outcome probabilities are only computed for single-qudit measurements."
                )
        # POVM element of each terminal measurement, ``E[k, b, a] = Tr[S_k(|a><b|)]``, so that
        # ``P(k) = sum_ab E[k, b, a] rho[a, b] = Tr[E_k rho]``.  Built by applying each
        # outcome's map to the matrix units, which needs no Kraus decomposition.
        self._povms: tuple[Array, ...] = tuple(_povm_elements(instrument) for _, instrument, _ in self._terminal)

    @property
    def _as_superop(self) -> bool:
        return True

    def _initial_state(self) -> qx.DensityMatrix:
        return qx.zero_state_matrix(dims=self.dims)

    @property
    def measured_qubits(self) -> tuple[int, ...]:
        """The qubit each axis of :meth:`outcome_probabilities` refers to, in program order."""
        return tuple(self.qubits[subsystem[0]] for _, _, subsystem in self._terminal)

    def outcome_probabilities(self, params: Array | None = None) -> Array:
        """Compute the joint distribution of the program's terminal measurement outcomes.

        Axis ``j`` of the result runs over the outcomes of the ``j``-th terminal measurement in
        program order, on qubit ``measured_qubits[j]``; ``probs[1, 0]`` is the probability that
        the first measured qubit reads ``1`` and the second ``0``.  Readout error assigned by
        the noise model is included, so on a noiseless ``|0>`` a measurement with 90% readout
        fidelity gives ``[0.9, 0.1]`` even though :meth:`compute` returns ``|0><0|``.

        Only terminal measurements count: a ``MEASURE`` followed by another operation on the
        same qubit is applied as its total channel and does not appear here.  Like
        :meth:`compute`, this is a pure JAX function of ``params``.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or pass ``None``)
            for a parameter-free program.
        :return: Real array of shape ``(n_1, ..., n_m)``, ``n_j`` being the number of outcomes
            of the ``j``-th terminal measurement.  The entries sum to one.
        :raises ValueError: If the program has no terminal measurement.
        """
        if not self._terminal:
            raise ValueError(
                "The program has no terminal measurement to read out. Add a MEASURE that is not followed by "
                "another operation on the same qubit, or use compute() for the state itself."
            )
        rho = self._pre_measurement_state(params).matrix
        n = len(self.dims)
        tensor = jnp.reshape(rho, self.dims + self.dims)
        # Label the row axis of qudit q as q and its column axis as n + q; a measured qudit's pair
        # is contracted with its POVM tensor, an unmeasured one is traced out by sharing a label.
        row_labels = list(range(n))
        col_labels = list(range(n, 2 * n))
        operands: list[Any] = []
        outcome_labels: list[int] = []
        for j, ((_, _, subsystem), povm) in enumerate(zip(self._terminal, self._povms, strict=True)):
            (qudit,) = subsystem
            label = 2 * n + j
            outcome_labels.append(label)
            operands += [povm, [label, col_labels[qudit], row_labels[qudit]]]
        measured = {subsystem[0] for _, _, subsystem in self._terminal}
        for qudit in range(n):
            if qudit not in measured:
                col_labels[qudit] = row_labels[qudit]
        probabilities = jnp.einsum(tensor, row_labels + col_labels, *operands, outcome_labels)
        return jnp.real(probabilities)

    def _make_branch(
        self, base: tuple[int, ...], base_dims: tuple[int, ...], total_dim: int
    ) -> Callable[[Array, qx.DensityMatrix], qx.DensityMatrix]:
        width = total_dim * total_dim

        def branch(op_mat: Array, rho: qx.DensityMatrix) -> qx.DensityMatrix:
            superop = qx.SuperOp.from_matrix(op_mat[:width, :width], (base_dims, base_dims))
            return qx.targeted_apply_superop(superop, rho, base)

        return branch


def _povm_elements(instrument: qx.QuantumInstrument) -> Array:
    """Return the POVM elements of *instrument* as an array ``E[k, b, a] = (E_k)_{ba}``.

    ``(E_k)_{ba} = Tr[S_k(|a><b|)]`` for the outcome-``k`` map ``S_k``, so ``Tr[E_k rho]`` is the
    probability of outcome ``k`` on ``rho``.  Each map is applied to the matrix units directly;
    no Kraus decomposition (and hence no eigendecomposition) is involved.
    """
    in_dims = instrument.dims[1]
    d = math.prod(in_dims)
    units = jnp.eye(d * d, dtype=complex).reshape(d * d, d, d)  # unit (a, b) at flat index a * d + b
    elements = []
    for k in range(instrument.num_outcomes):
        superop, _ = instrument.outcome_superop(k)

        def trace_after(unit: Array, superop: qx.SuperOp = superop) -> Array:
            return jnp.trace(
                qx.apply_superop_to_density_matrix(superop, qx.DensityMatrix.from_matrix(unit, in_dims)).matrix
            )

        traces = jax.vmap(trace_after)(units).reshape(d, d)  # traces[a, b] = Tr[S_k(|a><b|)]
        elements.append(traces.T)  # E[b, a]
    return jnp.stack(elements)


__all__ = [
    "DensityMatrixSimulator",
    "ProgramSimulator",
    "PureStateVectorSimulator",
    "Resolution",
]
