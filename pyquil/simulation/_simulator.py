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
inference, ``linearize``/``resolve``/``compress``).  Two families build on it:

* The **differentiable** family (:class:`_DifferentiableSimulator`) — jit/grad-friendly
  evolution of a compressed ``Unitary``/``SuperOp`` stack, with measurements collapsed to
  dephasing SuperOps: :class:`PureStateVectorSimulator` for gate-only programs and
  :class:`DensityMatrixSimulator` for any program, optionally with noise.
* The **trajectory** simulator (:class:`TrajectorySimulator`) — Monte-Carlo sampling of pure
  state vectors, with measurements kept as sampled instruments.

The ``compute`` method is the main entry point.  For the differentiable family it can be
passed directly to ``jax.jit`` or ``jax.grad``, and returns a quax state on which any quax
metric can be evaluated inside the same transformation::

    sim = DensityMatrixSimulator(program, noise_model=noise_model)
    loss = lambda params: 1 - qx.fidelity(target, sim.compute(params))
    jax.grad(loss)(sim.linearize(memory_map))

For the trajectory simulator it takes a PRNG key as well and returns the sampled measurement
outcomes alongside the state; ``sample`` returns outcomes alone, in batches.  That simulator is
``jit``-traceable but not differentiable (``jax.grad`` raises), the sampled Kraus index being a
discrete choice.

Both share the preprocessing pipeline ``resolve`` → ``compress`` → ``adapt``, which
``operations(params)`` runs end to end.

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
import secrets
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Generic, TypeAlias, TypeVar, cast, final

import jax
import jax.numpy as jnp
import numpy as np
import quax as qx
from jax import Array

from pyquil.api import MemoryMap
from pyquil.noise._noise_model import NoiseModelLike
from pyquil.quil import Program
from pyquil.quilbase import Measurement, Reset, ResetQubit
from pyquil.simulation._circuit import Circuit, CircuitOp, Group, MergePlan, Placement
from pyquil.simulation._resolver import (
    ExpandedOp,
    ParameterExpression,
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
    # JAX defines its flags dynamically, so they are read through ``config.values``: attribute
    # access works at runtime but is invisible to type checkers.
    precision = jax.config.values["jax_default_matmul_precision"]
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

    # Style note: explicit ``__init__`` rather than the frozen dataclass + ``cached_property``
    # pattern of pyQuil's noise code, because the derived JAX values (operator stacks, compiled
    # kernels) must be materialised outside any ``jax.jit`` trace.  Do not mix the two styles.
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
            faster to run and slower to compile. It does not change the simulated channel,
            the state, or the distribution of measurement outcomes. (For a trajectory
            simulator it does change which *unravelling* those outcomes are sampled from;
            see the architecture guide.)
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

    def adapt(self, compressed: Circuit) -> Circuit:
        """Convert a merged circuit to the representation this backend evolves.

        The default is the identity.  Override to convert; see
        :meth:`TrajectorySimulator.adapt`.  The differentiable family does not override it:
        its compiled path builds superoperators inside the vectorised stack constructor rather
        than by walking a circuit, so a hook here would never run on the path that matters.

        :param compressed: The circuit from :meth:`compress`.
        :return: The circuit in this backend's representation, one operation per input.
        """
        return compressed

    @final
    def operations(self, params: Array | None = None) -> Circuit:
        """Resolve, merge and adapt the program for *params*.

        The whole preprocessing pipeline in one call: :meth:`resolve`, then :meth:`compress`,
        then :meth:`adapt`.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or pass ``None``)
            for a parameter-free program.
        :return: The operations this backend evolves, in application order.
        """
        return self.adapt(self.compress(self.resolve(params)))


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
    def _pre_measurement_state(self, params: Array | None, initial_state: StateT | None = None) -> StateT:
        """Evolve the initial state through the operator stack, terminal measurements excluded."""
        state = self._state0 if initial_state is None else self._validated_initial_state(initial_state)
        # No operations (e.g. empty program): the initial state is the result, and
        # ``lax.switch`` cannot be given zero branches.
        if not self._branches:
            return state
        return self._apply(state, self._build_stack(self._default_params(params)))

    @final
    def _validated_initial_state(self, state: StateT) -> StateT:
        """Check that a caller-supplied initial state is this simulator's representation and shape.

        A mismatched state would otherwise fail deep inside the scan with an opaque shape error,
        or -- worse, for a state vector handed to the density-matrix backend -- broadcast silently.
        """
        if not isinstance(state, type(self._state0)):
            raise TypeError(
                f"initial_state must be a {type(self._state0).__name__} for {type(self).__name__}, "
                f"not a {type(state).__name__}."
            )
        if tuple(state.dims) != tuple(self.dims):
            raise ValueError(
                f"initial_state has dims {tuple(state.dims)} but this simulator's register is {tuple(self.dims)}."
            )
        return state

    @final
    def compute(self, params: Array | None = None, initial_state: StateT | None = None) -> StateT:  # type: ignore[override]
        """Compute the final state.

        Builds the merged operators for *params* and applies them to the initial state.  The
        call can be wrapped in ``jax.jit``, ``jax.grad`` or ``jax.vmap``.

        A ``MEASURE`` contributes its total channel, so the returned density matrix is the
        state averaged over measurement outcomes; the outcome distribution itself is available
        from :meth:`DensityMatrixSimulator.outcome_probabilities`.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :param initial_state: State to start from, in this simulator's representation and on
            this simulator's ``dims``.  Omit to start from the all-zero register.  This lets a
            circuit be split in two: evolve the shared prefix once, then run the varying tail
            from the state it produced, instead of re-running the prefix for every tail.
        :return: The final state: a ``StateVector`` or a ``DensityMatrix``.
        """
        state = self._pre_measurement_state(params, initial_state)
        for channel, subsystem in self._terminal_channels:
            # Only the density-matrix backend can hold an instrument, so ``state`` is a
            # ``DensityMatrix`` whenever this loop runs.
            state = cast(StateT, qx.targeted_apply_superop(channel, cast(qx.DensityMatrix, state), subsystem))
        return state

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
    """A set of gates sharing one constructor, argument layout, and embedding.

    Members differ only in which entries of the parameter vector feed their
    expression-valued arguments, so all of them are built with a single ``jax.vmap``.  This
    keeps the traced graph proportional to the number of distinct gate *kinds*
    rather than the number of gates.

    This is a mutable dataclass on purpose: it is an accumulator that
    :func:`_build_vectorized_operator_constructor` fills while it walks the plan, and it is
    consumed by :meth:`builder` immediately afterwards.
    """

    gate_fn: Callable[..., qx.Operator]
    n_args: int
    #: ``(position, value)`` for each literal argument.
    literal_args: tuple[tuple[int, float | complex], ...]
    #: ``(position, expression)`` for each expression-valued argument.  Every member's
    #: expressions have the same keys, so the first member's serve as the template: they are
    #: evaluated on the narrowed vector of slot values gathered for each member.
    expression_args: tuple[tuple[int, ParameterExpression], ...]
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
    #: Parameter-vector slots read by each member: the slot indices of its expression
    #: arguments, concatenated in argument order.
    slot_indices: list[list[int]] = field(default_factory=list)

    def builder(self) -> Callable[[Array], Array]:
        """Return ``params -> (n_members, width, width)`` embedded gate matrices."""
        gate_fn, n_args, literal_args, expression_args = (
            self.gate_fn,
            self.n_args,
            self.literal_args,
            self.expression_args,
        )
        target_dims, group_positions = self.target_dims, self.group_positions
        width, as_superop = self.width, self.as_superop
        slot_indices = jnp.asarray(self.slot_indices, dtype=jnp.int32)  # (n_members, n_slots)

        def single(slot_values: Array) -> Array:
            """Build one member's embedded matrix from the slot values it reads, in order."""
            args: list[Any] = [None] * n_args
            for position, value in literal_args:
                args[position] = value
            offset = 0
            for position, expression in expression_args:
                count = len(expression.slot_indices)
                args[position] = expression.evaluate(slot_values[offset : offset + count])
                offset += count
            gate = gate_fn(*args)
            if not isinstance(gate, qx.Unitary):
                gate = qx.Unitary.from_matrix(gate.matrix, gate.dims)
            return _embed_op_to_group(gate, target_dims, group_positions, width, as_superop=as_superop)

        batched = jax.vmap(single)
        return lambda params: batched(params[slot_indices])


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
            literal_args = tuple(
                (j, arg) for j, arg in enumerate(op.arguments) if not isinstance(arg, ParameterExpression)
            )
            expression_args = tuple(
                (j, arg) for j, arg in enumerate(op.arguments) if isinstance(arg, ParameterExpression)
            )
            # Expressions enter the key by *shape* only, so ``SIN(theta[0])`` and ``SIN(theta[1])``
            # share a batch and differ in the slots they read.
            key = (id(op.gate_fn), literal_args, tuple((j, arg.key) for j, arg in expression_args), embed_key)
            batch = batches.get(key)
            if batch is None:
                batch = _GateBatch(
                    gate_fn=op.gate_fn,
                    n_args=len(op.arguments),
                    literal_args=literal_args,
                    expression_args=expression_args,
                    target_dims=target_dims,
                    group_positions=group_positions,
                    width=width,
                    as_superop=as_superop,
                )
                batches[key] = batch
            batch.positions.append(pos)
            batch.slot_indices.append([slot for _, arg in expression_args for slot in arg.slot_indices])
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

    def outcome_probabilities(
        self, params: Array | None = None, initial_state: qx.DensityMatrix | None = None
    ) -> Array:
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
        :param initial_state: Density matrix to start from, as :meth:`compute`.
        :return: Real array of shape ``(n_1, ..., n_m)``, ``n_j`` being the number of outcomes
            of the ``j``-th terminal measurement.  The entries sum to one.
        :raises ValueError: If the program has no terminal measurement.
        """
        if not self._terminal:
            raise ValueError(
                "The program has no terminal measurement to read out. Add a MEASURE that is not followed by "
                "another operation on the same qubit, or use compute() for the state itself."
            )
        rho = self._pre_measurement_state(params, initial_state).matrix
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


# ══════════════════════════════════════════════════════════
# Trajectory simulator
# ══════════════════════════════════════════════════════════


class TrajectorySimulator(ProgramSimulator):
    """Monte Carlo trajectory simulator for programs with measurements and resets.

    Keeps every ``MEASURE`` as a sampleable ``QuantumInstrument`` (the default
    :meth:`~ProgramSimulator._prepare_ops`), which the merge plan therefore pins as ``atomic``,
    and adapts the merged circuit to the ``Unitary`` / ``KrausMap`` / ``QuantumInstrument``
    representation its sampling kernel consumes, with
    :meth:`~pyquil.simulation._circuit.Circuit.to_kraus_maps`.

    ``compute`` requires a JAX PRNG key.  The number of trajectories is determined by the key
    shape: a scalar key runs one trajectory; a batch of keys ``jax.random.split(key, n)`` runs
    *n* trajectories as a single ``vmap`` on one device::

        sim = TrajectorySimulator(program, noise_model=noise_model)
        params = sim.linearize(memory_map)

        # Single trajectory
        psi, outcomes = sim.compute(params, jax.random.key(0))

        # Batched trajectories
        keys = jax.random.split(jax.random.key(0), 100)
        psi_batch, outcomes_batch = sim.compute(params, keys)

    ``sample`` runs trajectories in batches and discards the state vectors, returning only the
    measurement outcomes.  It is also the multi-device entry point: each batch is spread over
    ``devices`` with :func:`jax.pmap`, one independent kernel replica per device.

    The sampling kernel is compiled once, at construction, from the merge plan alone: which
    subsystem each merged operation acts on, which operations are measurements, and how many
    Kraus operators each has are all properties of the program's structure, never of a
    parameter value.  Every ``compute`` and ``sample`` call rebuilds just the Kraus stack and
    reuses the compiled kernel.

    ``compute`` is ``jax.jit``- and ``jax.vmap``-traceable, so a parameter sweep can be
    compiled as a whole.  It is **not** differentiable: the sampled Kraus index is a discrete
    choice, so ``jax.grad`` raises ``NotImplementedError`` at trace time rather than returning
    a gradient that is silently zero.  Use :class:`DensityMatrixSimulator` for gradients.
    """

    def __init__(
        self,
        program: Program,
        qubits: Sequence[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 2,
        evolution_dtype: Any | None = None,
        devices: Sequence[Any] | None = None,
    ) -> None:
        """Prepare a trajectory simulator.

        :param program: Any Quil program, including measurements and resets.
        :param qubits: Register order; see :meth:`ProgramSimulator.__init__`.
        :param noise_model: Optional noise model. Instructions with no channel are ideal.
        :param max_subsystem_size: Compressor merge budget. Measurements are never merged,
            whatever the budget. It does not change the simulated channel or the outcome
            distribution, but it does change *which unravelling* the trajectories sample; see
            the architecture guide.
        :param evolution_dtype: Complex dtype the sampling kernel evolves in, e.g.
            ``jnp.complex64``. Preprocessing -- resolution, merging and the Choi
            eigendecomposition -- always runs at JAX's default precision, which must be 64-bit;
            only the Kraus stack and the state vector are cast. ``None`` (the default) keeps
            the JAX default throughout.
        :param devices: Devices :meth:`sample` runs data-parallel across, one independent
            replica each. Defaults to ``jax.devices()``.
        :raises ValueError: If ``evolution_dtype`` is not a complex dtype.
        """
        super().__init__(program, qubits, noise_model=noise_model, max_subsystem_size=max_subsystem_size)
        self.evolution_dtype = _complex_dtype(evolution_dtype)
        self._devices = tuple(devices) if devices is not None else tuple(jax.devices())
        self._layout = _KrausStackLayout.from_plan(self.plan, self._resolution.ops, self.dims)
        self._kernel = _build_trajectory_kernel(self._layout)
        self._batched_kernels: dict[int, Callable[[Array, Array], Array]] = {}

    @final
    def adapt(self, compressed: Circuit) -> Circuit:
        """Convert a merged circuit to the representation the trajectory kernel samples from.

        Dense superoperators become ``KrausMap`` operators; unitaries, Kraus maps and
        instruments pass through.  See
        :meth:`~pyquil.simulation._circuit.Circuit.to_kraus_maps`.
        """
        return compressed.to_kraus_maps()

    def compute(  # type: ignore[override]
        self,
        params: Array | None = None,
        key: Array | None = None,
        initial_state: qx.StateVector | None = None,
    ) -> tuple[qx.StateVector, Array]:
        """Run trajectory simulation.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :param key: JAX PRNG key (required).  Scalar key → single trajectory.
            Batch of keys (from ``jax.random.split``) → batched trajectories.
        :param initial_state: Pure state to start the trajectory from, on this simulator's
            ``dims``; omit to start from the all-zero register.  A trajectory carries a *pure*
            state, so a ``DensityMatrix`` is rejected.  The prefix/tail split this enables pays
            off mainly on the density-matrix path (see the architecture guide); a trajectory
            run has to run its trajectories either way.
        :return: Tuple of ``(state_vector, measurement_outcomes)``.  Outcome columns follow the
            ``MEASURE`` instructions in program order.  The state is at
            :attr:`evolution_dtype`.
        :raises ValueError: If ``key`` is omitted.
        """
        # ``key`` is required, but the abstract ``ProgramSimulator.compute`` takes only
        # ``params`` and ``**kwargs``, so typing it optional is what keeps this a signature
        # override mypy accepts.  Do not "fix" it to a required positional argument.
        if key is None:
            raise ValueError("TrajectorySimulator.compute requires a JAX PRNG key.")
        op_stack = self._op_stack(params)
        ensemble_size = () if key.ndim == 0 else (key.shape[0],)
        state = self._zero_state(ensemble_size) if initial_state is None else self._start_state(initial_state)
        return self._kernel(op_stack, state, key)

    @final
    def _start_state(self, state: qx.StateVector) -> qx.StateVector:
        """Check a caller-supplied initial state and cast it to the evolution dtype."""
        if not isinstance(state, qx.StateVector):
            raise TypeError(
                f"initial_state must be a StateVector, not a {type(state).__name__}. A trajectory "
                "carries a pure state; unravel a density matrix into eigenvectors first."
            )
        if tuple(state.dims) != tuple(self.dims):
            raise ValueError(
                f"initial_state has dims {tuple(state.dims)} but this simulator's register is {tuple(self.dims)}."
            )
        return qx.StateVector.from_matrix(state.matrix.astype(self.evolution_dtype), self.dims)

    @final
    def _op_stack(self, params: Array | None) -> Array:
        """Build the padded Kraus stack for *params*, at the evolution dtype."""
        return self._layout.stack(self.operations(params)).astype(self.evolution_dtype)

    @final
    def _zero_state(self, ensemble_size: tuple[int, ...]) -> qx.StateVector:
        """Build the all-zero register state, at the evolution dtype."""
        return _zero_state_vector(self.dims, ensemble_size, self.evolution_dtype)

    def sample(
        self,
        params: Array | None = None,
        num_trajectories: int = 1000,
        key: Array | None = None,
        *,
        batch_size: int | None = None,
    ) -> Array:
        """Run trajectory simulation in batches, returning only measurement outcomes.

        State vectors are discarded after each batch, so **device** memory is bounded by one
        batch however many trajectories are asked for.  The returned outcome array is not: it
        grows as ``num_trajectories * n_measurements * 4`` bytes in host memory.  When multiple
        devices are available the batch is run **data-parallel** via :func:`jax.pmap`: each
        device runs an independent replica of the trajectory kernel on its own slice of
        trajectories, with no cross-device communication.

        Each trajectory's randomness is derived from its *global* index, so the outcomes depend
        only on ``key`` and ``num_trajectories`` -- never on ``batch_size`` or on how many
        devices happen to be present.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :param num_trajectories: Total number of trajectories to simulate.
        :param key: JAX PRNG key.  Omit for fresh entropy, so that repeated calls accumulate
            independent samples; pass one to make a run reproducible.
        :param batch_size: Trajectories per device per batch.  With ``n`` devices each batch
            runs ``n * batch_size`` trajectories concurrently.  Purely a memory/performance
            knob: the results do not depend on it.  ``None`` (the default) derives a width from
            the kernel's per-trajectory footprint against a fixed device-memory budget
            (:func:`_default_batch_size`), capped at what ``num_trajectories`` needs.  Pass a
            value to pin one compiled width across calls of differing size, or to trade memory
            for throughput by hand.
        :return: Measurement outcomes with shape ``(num_trajectories, n_measurements)``.
        """
        op_stack = self._op_stack(params)
        if key is None:
            key = jax.random.key(secrets.randbits(63))
        if batch_size is None:
            batch_size = _default_batch_size(self._layout, self.evolution_dtype, num_trajectories, len(self._devices))
        if batch_size not in self._batched_kernels:
            self._batched_kernels[batch_size] = _batched_trajectory_kernel(
                self._kernel, self.dims, batch_size, dtype=self.evolution_dtype, devices=self._devices
            )
        batches = _run_batched_kernel(
            self._batched_kernels[batch_size], op_stack, num_trajectories, batch_size, key, len(self._devices)
        )
        if not batches:
            return jnp.empty((0, len(self._layout.measure_positions)), dtype=jnp.int32)
        return batches[0] if len(batches) == 1 else jnp.concatenate(batches, axis=0)


# ══════════════════════════════════════════════════════════
# Trajectory simulation internals
# ══════════════════════════════════════════════════════════


def _complex_dtype(dtype: Any | None) -> Any:
    """Validate an evolution dtype, defaulting to JAX's current complex type.

    :raises ValueError: If *dtype* is not a complex dtype.  Evolving a state in a real dtype
        would silently discard every phase, so it is rejected rather than coerced.
    """
    resolved = jnp.result_type(complex) if dtype is None else jnp.dtype(dtype)
    if not jnp.issubdtype(resolved, jnp.complexfloating):
        raise ValueError(f"evolution_dtype must be a complex dtype (jnp.complex64 or jnp.complex128), got {resolved}.")
    return resolved


def _zero_state_vector(dims: tuple[int, ...], ensemble_size: tuple[int, ...], dtype: Any) -> qx.StateVector:
    """Build the all-zero register state at *dtype*.

    ``qx.zero_state_vector`` takes no dtype and builds at JAX's default complex type, which is
    the preprocessing precision rather than the evolution precision; this keeps the two
    independent.  quax issue 59 asks for a ``dtype`` argument; drop this helper when it lands.
    """
    matrix = jnp.zeros((*ensemble_size, math.prod(dims)), dtype).at[..., 0].set(1)
    return qx.StateVector.from_matrix(matrix, dims)


def _trajectory_keys(key: Array, start: int, count: int) -> Array:
    """Per-trajectory PRNG keys for the global trajectory indices ``start .. start + count``.

    Deriving each key from its *global* index, rather than by splitting a chain per batch, is
    what makes :meth:`TrajectorySimulator.sample` return the same outcomes whatever the batch
    size and however many devices are present: trajectory *g* always gets ``fold_in(key, g)``.
    """
    return jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(start, start + count, dtype=jnp.uint32))


def _pad_matrix(mat: Array, max_k: int, dim: int) -> Array:
    """Zero-pad a ``(n_kraus, d, d)`` Kraus matrix out to ``(max_k, dim, dim)``.

    Padding is top-left aligned, so the operators keep their meaning on the leading block and
    the padding is exactly zero: a zero Kraus operator carries zero Born probability, and a
    zero row of the Kraus axis is never sampled.
    """
    if mat.shape == (max_k, dim, dim):
        return mat
    return jnp.pad(mat, ((0, max_k - mat.shape[0]), (0, dim - mat.shape[-2]), (0, dim - mat.shape[-1])))


def _op_to_kraus_matrix(op: CircuitOp) -> tuple[Array, int, bool]:
    """Convert a single trajectory operator to a bare Kraus *matrix*.

    A matrix rather than a ``KrausMap``, because the operators are zero-padded and stacked into
    the homogeneous array ``lax.scan`` needs; and every operator is expressed in Kraus form so a
    single, uniform ``jax.lax.switch`` branch (Kraus trajectory sampling) handles every type:

    - ``qx.Unitary`` → its own matrix, as a one-operator Kraus map.  Going through
      ``qx.to_kraus`` would diagonalise the Choi matrix to recover a rank-one channel: ``d²``
      operators of which one is non-zero, at the cost of an eigendecomposition per call.
    - ``qx.KrausMap`` → itself.
    - ``qx.QuantumInstrument`` → its outcome and Kraus axes are merged into a single Kraus axis
      (replicating the flattening in :func:`quax.targeted_apply_instrument_to_state_vector`).
      The returned *divisor* is the number of Kraus operators per outcome, so the sampled Kraus
      index ``k`` decodes to the measurement outcome ``k // divisor``.

    :param op: The operator (already acting on its base subsystem).
    :return: ``(matrix, divisor, is_measurement)`` where ``matrix`` has shape
        ``(n_kraus, d, d)``.
    :raises TypeError: For any other operator type, which :meth:`Circuit.to_kraus_maps` should
        already have converted.
    :raises ValueError: If an instrument's outcomes do not all carry the same number of Kraus
        operators, which would make the ``k // divisor`` decoding wrong.
    """
    match op:
        case qx.Unitary():
            return op.matrix[None], 1, False
        case qx.KrausMap():
            return op.matrix, 1, False
        case qx.QuantumInstrument():
            kraus_mats = [qx.superop_to_kraus(op.outcome_superop(i)[0]).matrix for i in range(op.num_outcomes)]
            # Decoding an outcome as ``k // divisor`` needs one divisor for every outcome, so
            # the per-outcome counts have to agree.  They always do today -- quax's Choi
            # decomposition returns a fixed d_out*d_in set -- and this guards the day it does
            # not, which would otherwise mis-label every outcome silently.
            counts = {mat.shape[-3] for mat in kraus_mats}
            if len(counts) != 1:
                raise ValueError(
                    f"Instrument outcomes carry differing Kraus counts {sorted(counts)}; the outcome decoding "
                    "requires one count per outcome."
                )
            return jnp.concatenate(kraus_mats, axis=-3), kraus_mats[0].shape[-3], True
        case _:
            raise TypeError(f"Unsupported operator type for trajectory sampling: {type(op).__name__}")


def _promote_to_register(op: CircuitOp, subsystem: tuple[int, ...], dims: tuple[int, ...]) -> CircuitOp:
    """Promote *op* to the register dimension on its subsystem (identity on the higher levels).

    Without this, an op authored at a lower dimension than the register (e.g. a qubit-dimension
    channel on a register promoted to qutrits by a leakage model) would be zero-padded to
    ``d_max`` instead — silently wrong on the high levels, and a reshape error when ``d_max``
    is below the branch's dimension.
    """
    target_dims = tuple(dims[q] for q in subsystem)
    return op if op.dims[0] == target_dims else qx.promote(op, target_dims)


#: An operation sequence the trajectory kernel accepts: a circuit, or bare placements.
_Operations: TypeAlias = Circuit | Sequence[Placement]

#: A ``run(op_stack, psi, key) -> (psi, outcomes)`` trajectory kernel.
_TrajectoryRun: TypeAlias = Callable[[Array, qx.StateVector, Array], tuple[qx.StateVector, Array]]

#: A ``branch(op_mat, psi, key) -> (psi, sampled_index)`` switch branch for one subsystem.
_KrausBranch: TypeAlias = Callable[[Array, qx.StateVector, Array], tuple[qx.StateVector, Array]]


def _kraus_count(members: Sequence[ExpandedOp], base_dim: int) -> tuple[int, int, bool]:
    """Derive a merged group's Kraus count from its members' *types* alone.

    Merging is structural and a noise channel is never parametric -- a channel is looked up by
    concrete instruction equality, so only ideal gates can carry a memory reference.  That
    makes every count below a property of the program, fixed for all parameter values, which is
    what lets the kernel be compiled once at construction with no probe:

    - all-unitary group → one Kraus operator, whatever the angles;
    - a group containing a channel → the Choi decomposition's fixed ``d²`` set;
    - an instrument → ``num_outcomes`` blocks of ``d²``.

    A *singleton* channel is measured at its own dimension rather than the register's, because
    :meth:`MergePlan.apply` passes a lone operation through untouched: a qubit-dimension channel
    on a register a leakage model promoted to qutrits decomposes at ``d = 2`` and is only then
    padded up by :func:`_promote_to_register`.

    :param members: The expanded operations the group fuses, in application order.
    :param base_dim: The total dimension of the group's subsystem in the register.
    :return: ``(kraus_count, divisor, is_measurement)``, matching :func:`_op_to_kraus_matrix`.
    """
    # A ParametricGate is a resolver callable, not a quax operator, so it has no ``dims``; the
    # unitary test has to come first.
    if all(isinstance(member, ParametricGate | qx.Unitary) for member in members):
        return 1, 1, False
    if len(members) == 1 and isinstance(members[0], qx.QuantumInstrument):
        per_outcome = base_dim**2
        return members[0].num_outcomes * per_outcome, per_outcome, True
    if len(members) == 1 and isinstance(members[0], qx.KrausMap):
        # A Kraus map passes through the conversion untouched, and nothing says its operator
        # count is the canonical d^2 -- a composed set can be larger -- so read it directly.
        return members[0].matrix.shape[-3], 1, False
    if len(members) == 1 and isinstance(members[0], qx.SuperOperator):
        own_dim = math.prod(members[0].dims[1])
        return own_dim**2, 1, False
    return base_dim**2, 1, False


@dataclass(frozen=True)
class _KrausStackLayout:
    """The structural, parameter-independent shape of a trajectory operation sequence.

    Every operation is expressed as a zero-padded Kraus matrix so the scan can index a single
    homogeneous ``(n_ops, max_k, d_max, d_max)`` stack (operators live on different subsystems
    with different Kraus counts, and there is no ragged stacking).  Zero-padded Kraus operators
    carry zero Born probability and are never sampled; each branch re-slices
    ``[:k_base, :db, :db]`` back to its own budget before rebuilding a ``KrausMap``.

    The layout is what the compiled kernel closes over.  :meth:`stack` produces the matching
    stack for one concrete operation sequence, and is the only per-parameter work.

    :param dims: Per-qudit register dimensions.
    :param subsystems: The subsystem each operation acts on, in application order.
    :param bases: The distinct subsystems, in first-seen order (one switch branch each).
    :param branch_index: For each operation, its index into ``bases``.
    :param base_max_k: Kraus operators each base needs, one entry per entry of ``bases``.
    :param divisors: Per-operation Kraus-count divisor; measurement outcome = ``index // divisor``.
    :param measure_positions: Indices of the measurement operations, in outcome-column order.
    :param d_max: Matrix dimension per stack row.
    """

    dims: tuple[int, ...]
    subsystems: tuple[tuple[int, ...], ...]
    bases: tuple[tuple[int, ...], ...]
    branch_index: tuple[int, ...]
    base_max_k: tuple[int, ...]
    divisors: tuple[int, ...]
    measure_positions: tuple[int, ...]
    d_max: int

    @property
    def n_ops(self) -> int:
        return len(self.subsystems)

    @property
    def max_k(self) -> int:
        """Kraus operators per stack row: the widest base, since the stack is homogeneous."""
        return max(self.base_max_k, default=1)

    @classmethod
    def from_plan(
        cls,
        plan: MergePlan,
        ops: Sequence[ExpandedOp],
        dims: tuple[int, ...],
    ) -> _KrausStackLayout:
        """Derive the layout from the merge plan, without resolving any parameter.

        Everything the kernel needs is structural (see :func:`_kraus_count`), so the layout is
        read straight off the plan -- including the Kraus counts, which is why a simulator can
        compile its kernel at construction and never re-derive the shape.

        :param plan: The merge plan; supplies the groups, bases and branch indices directly.
        :param ops: The expanded operations, in program order, indexed by the plan's groups.
        :param dims: Per-qudit register dimensions.
        """
        dims = tuple(dims)
        base_of = {base: i for i, base in enumerate(plan.bases)}
        base_max_k = [1] * len(plan.bases)
        divisors: list[int] = []
        measure_positions: list[int] = []
        for position, (nodes, subsystem) in enumerate(plan.groups):
            base_dim = math.prod(dims[q] for q in subsystem)
            count, divisor, is_measure = _kraus_count([ops[node] for node in nodes], base_dim)
            index = base_of[subsystem]
            base_max_k[index] = max(base_max_k[index], count)
            divisors.append(divisor)
            if is_measure:
                measure_positions.append(position)
        # Outcome columns follow the MEASURE instructions in *program* order.  The plan may emit
        # two independent measurements in either order (see the warning on ``MergePlan``), so
        # each position is ordered by the program index its group carries.
        measure_positions.sort(key=lambda position: plan.groups[position][0][0])
        return cls(
            dims=dims,
            subsystems=tuple(subsystem for _, subsystem in plan.groups),
            bases=plan.bases,
            branch_index=plan.op_index,
            base_max_k=tuple(base_max_k),
            divisors=tuple(divisors),
            measure_positions=tuple(measure_positions),
            d_max=max((math.prod(dims[q] for q in base) for base in plan.bases), default=1),
        )

    @classmethod
    def from_operations(
        cls,
        operations: _Operations,
        dims: tuple[int, ...],
        program_index: Sequence[int] | None = None,
    ) -> _KrausStackLayout:
        """Read the layout off one concrete operation sequence, by measuring its matrices.

        :meth:`from_plan` is what the simulators use; this is for callers holding bare
        placements rather than a plan, and it is the oracle the structural derivation is tested
        against -- the two must agree on any circuit a plan produced.

        :param operations: The operations, in application order.
        :param dims: Per-qudit register dimensions.
        :param program_index: For each operation, the index that orders measurement outcome
            columns -- typically the original program index of the group's operation, so that
            columns follow ``MEASURE`` instructions in program order even when the merge plan
            emitted independent measurements in a different order. Defaults to application
            order.
        :raises ValueError: If ``program_index`` does not have one entry per operation.
        """
        dims = tuple(dims)
        subsystems: list[tuple[int, ...]] = []
        bases: dict[tuple[int, ...], int] = {}
        branch_index: list[int] = []
        base_max_k: list[int] = []
        divisors: list[int] = []
        measure_positions: list[int] = []
        d_max = 1
        for i, (op, subsystem) in enumerate(operations):
            subsystem = tuple(subsystem)
            mat, divisor, is_measure = _op_to_kraus_matrix(_promote_to_register(op, subsystem, dims))
            subsystems.append(subsystem)
            if subsystem not in bases:
                bases[subsystem] = len(bases)
                base_max_k.append(1)
            index = bases[subsystem]
            branch_index.append(index)
            base_max_k[index] = max(base_max_k[index], mat.shape[0])
            divisors.append(divisor)
            if is_measure:
                measure_positions.append(i)
            d_max = max(d_max, mat.shape[-1])
        if program_index is not None:
            if len(program_index) != len(subsystems):
                raise ValueError(f"program_index has {len(program_index)} entries for {len(subsystems)} operation(s).")
            measure_positions.sort(key=lambda position: program_index[position])
        return cls(
            dims=dims,
            subsystems=tuple(subsystems),
            bases=tuple(bases),
            branch_index=tuple(branch_index),
            base_max_k=tuple(base_max_k),
            divisors=tuple(divisors),
            measure_positions=tuple(measure_positions),
            d_max=d_max,
        )

    def stack(self, operations: _Operations) -> Array:
        """Build the ``(n_ops, max_k, d_max, d_max)`` Kraus stack for one operation sequence.

        :raises ValueError: If *operations* does not fit this layout, which means it was not
            produced from the same program.
        """
        if len(operations) != self.n_ops:
            raise ValueError(f"Layout covers {self.n_ops} operation(s) but {len(operations)} were given.")
        max_k = self.max_k
        mats: list[Array] = []
        for i, (op, subsystem) in enumerate(operations):
            if tuple(subsystem) != self.subsystems[i]:
                raise ValueError(f"Operation {i} acts on {tuple(subsystem)}; the layout expects {self.subsystems[i]}.")
            mat, _, _ = _op_to_kraus_matrix(_promote_to_register(op, self.subsystems[i], self.dims))
            budget = self.base_max_k[self.branch_index[i]]
            if mat.shape[0] > budget or mat.shape[-1] > self.d_max:
                raise ValueError(
                    f"Operation {i} has {mat.shape[0]} Kraus operator(s) of dimension {mat.shape[-1]}, "
                    f"but the layout allows {budget} of dimension {self.d_max}."
                )
            mats.append(_pad_matrix(mat, max_k, self.d_max))
        return jnp.stack(mats, axis=0)


def _kraus_branch(
    base: tuple[int, ...],
    base_dims: tuple[int, ...],
    base_max_k: int,
    op_mat: Array,
    psi: qx.StateVector,
    key: Array,
) -> tuple[qx.StateVector, Array]:
    """Apply one padded stack row to *psi* as a Kraus map on *base*; return the sampled index too.

    The scanned stack has to be homogeneous, but a branch need only look at its own base's
    budget: slicing the Kraus axis here keeps a one-qubit gate from being applied as if it were
    the widest merged channel in the circuit.  ``sample_kraus_map_trajectory`` rather than
    ``targeted_apply_kraus_map_trajectory`` because the kernel needs the sampled index to
    decode measurement outcomes.

    The first three arguments are layout facts, bound with :func:`functools.partial` by
    :func:`_build_trajectory_kernel`; the last three are traced.
    """
    db = math.prod(base_dims)
    kraus_map = qx.KrausMap.from_matrix(op_mat[:base_max_k, :db, :db], (base_dims, base_dims))
    return cast(tuple[qx.StateVector, Array], qx.sample_kraus_map_trajectory(kraus_map, psi, key, base))


def _fold_keys(keys: Array, i: Array) -> Array:
    """Per-operation key(s) for operation index *i*: a scalar key folds once, a key vector row-wise."""
    if keys.ndim == 0:
        return jax.random.fold_in(keys, i)
    return jax.vmap(lambda k: jax.random.fold_in(k, i))(keys)


def _scan_body(
    branches: Sequence[_KrausBranch], keys: Array, psi: qx.StateVector, xs: tuple[Array, Array, Array]
) -> tuple[qx.StateVector, Array]:
    """One ``lax.scan`` step: dispatch stack row *xs* to the branch for its base subsystem.

    *branches* is bound by the builder and *keys* by :func:`_run_trajectory` from its own trace.
    """
    op_mat, branch_index, i = xs
    psi, sampled = jax.lax.switch(branch_index, branches, op_mat, psi, _fold_keys(keys, i))
    return psi, sampled.astype(jnp.int32)


def _run_trajectory(
    layout: _KrausStackLayout,
    branches: Sequence[_KrausBranch],
    op_stack: Array,
    psi: qx.StateVector,
    key: Array,
) -> tuple[qx.StateVector, Array]:
    """Run the trajectory kernel body; :func:`_build_trajectory_kernel` states the contract.

    *layout* and *branches* are bound by the builder; ``op_stack``, ``psi`` and ``key`` are
    traced arguments.
    """
    ensemble_size = psi.ensemble_size
    if ensemble_size and key.ndim == 0:
        key = jax.random.split(key, ensemble_size[0])
    # Build-time constants stay NumPy: they are baked into the executable as literals, so there
    # is no reason to allocate a device array for them.
    xs = (op_stack, np.asarray(layout.branch_index, dtype=np.int32), np.arange(layout.n_ops, dtype=np.int32))
    psi, sampled = jax.lax.scan(partial(_scan_body, branches, key), psi, xs)
    if layout.measure_positions:
        outcomes = jnp.stack([sampled[p] // layout.divisors[p] for p in layout.measure_positions], axis=-1)
    else:
        outcomes = jnp.empty((*ensemble_size, 0), dtype=jnp.int32)
    return psi, outcomes


def _run_empty(op_stack: Array, psi: qx.StateVector, key: Array) -> tuple[qx.StateVector, Array]:
    """Pass the state through unchanged: the kernel for a program with no operations."""
    return psi, jnp.empty((*psi.ensemble_size, 0), dtype=jnp.int32)


def _reject_differentiation(primals: tuple[Any, ...], tangents: tuple[Any, ...]) -> tuple[Any, Any]:
    """Refuse to differentiate the kernel (its ``custom_jvp`` rule): an error, not a silent zero."""
    raise NotImplementedError(
        "TrajectorySimulator.compute is not differentiable: the sampled Kraus index is a discrete "
        "choice, so its gradient would be identically zero. Use DensityMatrixSimulator for gradients."
    )


def _build_trajectory_kernel(layout: _KrausStackLayout) -> _TrajectoryRun:
    """Build the jitted trajectory kernel for a layout.

    The returned ``run(op_stack, psi, key)`` scans the padded Kraus stack with a
    :func:`jax.lax.switch` per operation dispatching on its base subsystem.  Because the stack
    is an *argument*, one compilation serves every parameter value (and every batch) with the
    same ``psi``/``key`` shapes.

    Measurements are handled uniformly by flattening a quantum instrument so that sampling a
    Kraus index also selects an outcome (``index // divisor``).  Per-operation keys are derived
    lazily via ``jax.random.fold_in`` so the key array is never materialised in full.

    **What the kernel closes over.**  Only *layout* and the switch branches derived from it,
    bound explicitly with :func:`functools.partial` (:func:`_kraus_branch`,
    :func:`_run_trajectory`) so the captured set is readable from the signatures.  A capture is
    read once at trace time and frozen into the executable, which sets two rules: anything that
    varies per call -- the stack, the state, the key -- is an argument, never a capture; and
    nothing outside the layout (``self``, a config read, a clock) is captured, so two kernels
    built from equal layouts are interchangeable.  Background: the JAX guides on
    `jit compilation <https://docs.jax.dev/en/latest/jit-compilation.html>`_ and
    `common gotchas <https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_.

    The kernel carries a ``custom_jvp`` rule that raises, so ``jax.grad`` fails at trace time
    instead of returning a gradient that is silently zero.

    :return: ``run(op_stack, psi, key) -> (final_state_vector, measurement_outcomes)`` where
        ``measurement_outcomes`` has shape ``(*ensemble, n_measurements)``, dtype int32.
        ``key`` is a scalar PRNG key or a per-trajectory key vector.
    """
    if layout.n_ops == 0:
        return _run_empty
    branches = tuple(
        partial(_kraus_branch, base, tuple(layout.dims[q] for q in base), k)
        for base, k in zip(layout.bases, layout.base_max_k, strict=True)
    )
    run = jax.custom_jvp(partial(_run_trajectory, layout, branches))
    run.defjvp(_reject_differentiation)
    return cast(_TrajectoryRun, jax.jit(run))


def _batched_trajectory_kernel(
    kernel: _TrajectoryRun,
    dims: tuple[int, ...],
    per_device: int,
    *,
    dtype: Any,
    devices: Sequence[Any],
) -> Callable[[Array, Array], Array]:
    """Wrap a trajectory kernel in a data-parallel :func:`jax.pmap` over *devices*.

    Each device runs an independent replica of the kernel on ``per_device`` trajectories.
    Trajectories are statistically independent, so no cross-device communication is required:
    per-device memory equals a single-device run.  The zero state is built *inside* the mapped
    function so the full ``(n_devices, per_device, hilbert)`` array is never allocated on one
    device, and only the outcomes are returned, so the final state vectors are freeable
    intermediates that are never gathered back to the host.

    :param kernel: The compiled single-batch kernel to replicate.
    :param dims: Per-qudit register dimensions.
    :param per_device: Trajectories each device runs per call.
    :param dtype: Complex dtype to allocate the per-device zero state in.
    :param devices: The devices to map over.
    :return: ``pkernel(op_stack, device_keys)`` taking the Kraus stack (broadcast to every
        device) and a ``(n_devices, per_device)`` key array, returning outcomes only.
    """

    def run_replica(op_stack: Array, device_keys: Array) -> Array:
        psi = _zero_state_vector(dims, (per_device,), dtype)
        return kernel(op_stack, psi, device_keys)[1]

    return cast(Callable[[Array, Array], Array], jax.pmap(run_replica, in_axes=(None, 0), devices=list(devices)))


def _run_batched_kernel(
    pkernel: Callable[[Array, Array], Array],
    op_stack: Array,
    num_trajectories: int,
    per_device: int,
    key: Array,
    n_devices: int,
) -> list[Array]:
    """Drive a pmapped kernel over ``num_trajectories`` in fixed-width calls.

    Every call runs at the same width (``n_devices * per_device``) so the compiled kernel is
    reused; the final short call is padded up to that width and its extra rows are sliced off.
    Keys come from :func:`_trajectory_keys` on the *global* trajectory index, so the outcomes
    do not depend on how the work was divided into calls or devices.

    :return: One outcome array per call, already trimmed to the trajectories asked for.
    """
    per_call = n_devices * per_device
    batches: list[Array] = []
    for start in range(0, num_trajectories, per_call):
        this_call = min(per_call, num_trajectories - start)
        batch_keys = _trajectory_keys(key, start, per_call).reshape(n_devices, per_device)
        # pmap re-adds the leading device axis: (n_devices, per_device, n_meas).  Flatten it
        # back to a 1-D ensemble to preserve the return contract.
        batches.append(pkernel(op_stack, batch_keys).reshape(per_call, -1)[:this_call])
    return batches


#: Device memory one ``sample`` batch may occupy, in bytes, when ``batch_size`` is derived.
_BATCH_MEMORY_BUDGET = 256 * 2**20

#: Working-set multiplier over the ``max_k`` candidate states a Kraus application materialises.
_BATCH_WORKING_SET_FACTOR = 4


def _default_batch_size(layout: _KrausStackLayout, dtype: Any, num_trajectories: int, n_devices: int) -> int:
    """Pick a per-device batch width for :meth:`TrajectorySimulator.sample`.

    Applying a ``max_k``-operator Kraus map to a state materialises ``max_k`` candidate states,
    so a trajectory's working set is about ``max_k * hilbert_dim * itemsize`` bytes, times a
    slack factor for the scan carry and the norms.  The width is what fits that into
    :data:`_BATCH_MEMORY_BUDGET`, never below one trajectory, and never above the
    ``ceil(num_trajectories / n_devices)`` the run needs -- so a small run does not pad itself
    out to a wide batch.  It is a heuristic: XLA's real footprint is not knowable from here, and
    ``batch_size`` stays available to override it.
    """
    hilbert_dim = math.prod(layout.dims)
    per_trajectory = layout.max_k * hilbert_dim * jnp.dtype(dtype).itemsize * _BATCH_WORKING_SET_FACTOR
    fits = max(1, _BATCH_MEMORY_BUDGET // per_trajectory)
    needed = max(1, -(-num_trajectories // n_devices))
    return int(min(fits, needed))


__all__ = [
    "DensityMatrixSimulator",
    "ProgramSimulator",
    "PureStateVectorSimulator",
    "Resolution",
    "TrajectorySimulator",
]
