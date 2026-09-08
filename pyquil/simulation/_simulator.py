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

Each simulator here is a thin **wrapper** around its quax counterpart:
:func:`~pyquil.simulation._quax_backend.transpile` turns the ``Program`` into a
:class:`quax.Circuit`, and the quax simulator does the rest.  Everything that is not
Quil-specific — merge planning, vectorised operator-stack construction, evolution — lives in
quax and is shared with any other front end.

* :class:`PureStateVectorSimulator` — gate-only programs (no noise, measurements or resets).
* :class:`DensityMatrixSimulator` — any program, optionally with noise.

Both are jit- and grad-friendly; ``compute`` can be passed straight to ``jax.jit`` or
``jax.grad``.

Wrapping rather than subclassing is deliberate.  A quax simulator is a frozen, prepared object
built from a circuit, and a pyQuil simulator is that plus the Quil-facing pipeline
(``linearize``/``resolve``/``compress``).  Inheriting would force the two lifecycles together
and expose quax's constructor as pyQuil's; holding one keeps the boundary a boundary.

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

from typing import TYPE_CHECKING, Any

import quax as qx
from jax import Array

from pyquil.quilbase import Measurement, Reset, ResetQubit
from pyquil.simulation._quax_backend import ParameterBinding, Transpilation, transpile
from pyquil.simulation._resolver import ResolvedOp

if TYPE_CHECKING:
    from pyquil.api import MemoryMap
    from pyquil.noise._noise_model import NoiseModelLike
    from pyquil.quil import Program


# ══════════════════════════════════════════════════════════
# Base class
# ══════════════════════════════════════════════════════════


class ProgramSimulator:
    """Shared Quil-facing pipeline for the quax-backed program simulators.

    Holds the transpiled circuit and the quax simulator built from it, and exposes the
    ``linearize``/``resolve``/``compress``/``compute`` pipeline in terms of them.  Subclasses
    supply the quax simulator and any Quil-level validation.

    Instances are treated as immutable after construction.
    """

    #: The quax simulator this wraps.  Subclasses set it during construction.
    _simulator: qx.Simulator

    def __init__(
        self,
        program: Program,
        qubits: list[int] | None = None,
        *,
        noise_model: NoiseModelLike | None = None,
        max_subsystem_size: int = 2,
        dims: tuple[int, ...] | None = None,
    ) -> None:
        """Transpile *program* and prepare the quax simulator for it.

        :param program: The Quil program to simulate. May contain ``DEFGATE`` and
            ``DEFCIRCUIT`` definitions, which are expanded.
        :param qubits: Explicit register, in the order the state's subsystems will follow.
            Defaults to the program's qubits in ascending order. Pass this to include a
            qubit the program never names (an idle spectator), to fix a specific ordering,
            or when a ``DEFCIRCUIT`` body names a literal qubit -- those are not discovered
            automatically.
        :param noise_model: Optional noise model. Channels are looked up per instruction;
            instructions with no channel are simulated ideally.
        :param max_subsystem_size: Largest number of qudits the merge planner may fuse
            adjacent operations onto. Larger values mean fewer, bigger operators: usually
            faster to run and slower to compile. Purely a performance knob -- results are
            independent of it.
        :param dims: Per-qudit dimensions, in ``qubits`` order. Defaults to inferring them
            from the program's gates and channels (2 unless something says otherwise).
        :raises ValueError: If ``qubits`` contains duplicates.
        """
        self._validate(program)
        self._transpilation = transpile(program, qubits, noise_model=noise_model, dims=dims, measurement="instrument")
        self.qubits = list(self._transpilation.qubits)
        self.n_qubits = len(self.qubits)
        self._max_subsystem_size = max_subsystem_size

    # -- hook for subclass validation ---------------------

    def _validate(self, program: Program) -> None:
        """Override to reject unsupported instructions."""

    # -- what the wrapper is made of ----------------------

    @property
    def transpilation(self) -> Transpilation:
        """The transpiled circuit, register and parameter binding."""
        return self._transpilation

    @property
    def binding(self) -> ParameterBinding:
        """The map between Quil memory references and quax parameter slots."""
        return self._transpilation.binding

    @property
    def circuit(self) -> qx.Circuit:
        """The transpiled circuit, before any simulator-specific preparation."""
        return self._transpilation.circuit

    @property
    def simulator(self) -> qx.Simulator:
        """The quax simulator doing the work."""
        return self._simulator

    @property
    def dims(self) -> tuple[int, ...]:
        """Per-qudit dimensions of the register, in ``qubits`` order."""
        return self._transpilation.circuit.dims

    @property
    def bases(self) -> list[tuple[int, ...]]:
        """The distinct subsystems the merged operations act on, in first-seen order."""
        return list(self._simulator.bases)

    @property
    def op_index(self) -> tuple[int, ...]:
        """The index into :attr:`bases` of each merged operation, in application order."""
        return self._simulator.plan.op_index

    @property
    def base_dims(self) -> list[tuple[int, ...]]:
        """Per-qudit dimensions of each base subsystem."""
        return list(self._simulator.base_dims)

    @property
    def d_max(self) -> int:
        """The largest base Hilbert-space dimension."""
        return self._simulator.d_max

    # -- public pipeline methods --------------------------

    def linearize(self, memory_map: MemoryMap | None) -> Array:
        """Convert a memory map to a flat JAX parameter vector.

        The vector's layout is fixed at construction (the order in which parametric gates
        were expanded), so it is the vector every ``compute``/``resolve`` call expects.

        :param memory_map: Values for each declared memory region, as passed to the QVM.
        :return: A flat ``float`` vector with one entry per runtime parameter.
        """
        return self._transpilation.binding.linearize(memory_map)

    def _default_params(self, params: Array | None) -> Array:
        """Return *params*, validated against the program's parameter count.

        Lets callers omit ``params`` for parameter-free programs. Anything else -- a
        parametric program called with no vector, or a vector of the wrong length -- is a
        caller error, and is reported as one.

        :raises ValueError: If ``params`` has the wrong length, or is omitted for a
            program that takes parameters.
        """
        return self._prepared_circuit.validate_params(params)

    @property
    def _prepared_circuit(self) -> qx.Circuit:
        """The circuit the quax simulator actually evolves, after any representation change."""
        return self._simulator.prepared_circuit

    def resolve(self, params: Array) -> list[ResolvedOp]:
        """Resolve parameters into one operator per expanded operation.

        Building every gate is exactly what the simulation path avoids -- it is what makes
        the compiled graph scale with the number of gate *kinds* rather than the number of
        gates -- so this is an inspection API, not a step in ``compute``.

        :param params: Flat parameter vector from :meth:`linearize`.
        :return: One ``(operator, subsystem)`` pair per expanded operation, in program order.
        """
        return list(self._prepared_circuit.to_constant_circuit(params).ops)

    def compress(self, resolved: list[ResolvedOp]) -> list[ResolvedOp]:
        """Merge operators via greedy edge contraction.

        :param resolved: Operators from :meth:`resolve`.
        :return: Merged operators, one per merge group, in application order. A merged
            group's subsystem is sorted; an unmerged operation keeps its own operand order.
        """
        constant = qx.ConstantCircuit(dims=self.dims, ops=tuple(resolved))
        return list(self._simulator.plan.apply(constant).ops)

    def compute(self, params: Array | None = None, **kwargs: Any) -> Any:
        """Compute the simulation result.  Subclasses must override.

        :param params: Flat parameter vector from :meth:`linearize`; omit for a
            parameter-free program.
        :param kwargs: Subclass-specific options.
        :return: The simulated result (a state vector, density matrix, ...).
        """
        raise NotImplementedError


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
        :param max_subsystem_size: Merge budget; performance only. See
            :meth:`ProgramSimulator.__init__`.
        :raises ValueError: If the program contains a non-unitary operation.
        """
        super().__init__(program, qubits, noise_model=None, max_subsystem_size=max_subsystem_size)
        try:
            self._simulator = qx.StateVectorSimulator(circuit=self.circuit, max_subsystem_size=max_subsystem_size)
        except qx.CircuitError as error:
            raise self._non_unitary_error(error) from error

    def _validate(self, program: Program) -> None:
        for inst in program.instructions:
            if isinstance(inst, Measurement):
                raise ValueError(f"PureStateVectorSimulator does not support measurements.  Found: {inst}")
            if isinstance(inst, (Reset, ResetQubit)):
                raise ValueError(f"PureStateVectorSimulator does not support resets.  Found: {inst}")

    def _non_unitary_error(self, error: qx.CircuitError) -> ValueError:
        """Re-raise quax's rejection with the Quil context quax cannot know about.

        ``_validate`` sees only top-level instructions, so a ``MEASURE`` or ``RESET`` hidden in
        a ``DEFCIRCUIT`` body slips past it and is caught here instead, once the program has
        been expanded.  quax reports *what* is wrong and *where in the circuit*; only pyQuil
        can say that the operation may have come from a ``DEFCIRCUIT`` body.
        """
        if error.kind is not qx.CircuitErrorKind.NON_UNITARY_OP:
            return ValueError(str(error))
        operation = self.circuit.ops[error.op_index][0] if error.op_index is not None else None
        subsystem = list(error.subsystem) if error.subsystem is not None else []
        return ValueError(
            f"PureStateVectorSimulator supports unitary operations only, but the "
            f"expanded program contains a {type(operation).__name__} on qubit(s) "
            f"{subsystem}. Measurements, resets and noise channels require "
            "DensityMatrixSimulator; note that these may come from a DEFCIRCUIT body."
        )

    def compute(self, params: Array | None = None) -> qx.StateVector:  # type: ignore[override]
        """Compute the final state vector.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :return: The final state vector.
        """
        return self._simulator.compute(params)

    def __call__(self, params: Array | None = None) -> qx.StateVector:
        return self.compute(params)

    def unitary(self, params: Array | None = None) -> qx.Unitary:
        """Compute the full program unitary.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :return: The full unitary matrix.
        """
        return self._simulator.unitary(params)


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
        :param max_subsystem_size: Merge budget; performance only. See
            :meth:`ProgramSimulator.__init__`.
        """
        super().__init__(program, qubits, noise_model=noise_model, max_subsystem_size=max_subsystem_size)
        self._simulator = qx.DensityMatrixSimulator(circuit=self.circuit, max_subsystem_size=max_subsystem_size)

    def compute(self, params: Array | None = None) -> qx.DensityMatrix:  # type: ignore[override]
        """Compute the final density matrix.

        :param params: Flat parameter vector from :meth:`linearize`.  Omit (or
            pass ``None``) for a parameter-free program.
        :return: The final density matrix.
        """
        return self._simulator.compute(params)

    def __call__(self, params: Array | None = None) -> qx.DensityMatrix:
        return self.compute(params)
