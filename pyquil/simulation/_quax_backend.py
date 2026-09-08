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
"""Transpilation from a Quil :class:`~pyquil.quil.Program` to a :class:`quax.Circuit`.

This is the whole of pyQuil's side of the simulator boundary.  Everything downstream of a
``qx.Circuit`` — merge planning, operator-stack construction, evolution — lives in quax and
knows nothing about Quil.  Everything upstream — ``DEFGATE``/``DEFCIRCUIT`` expansion, qubit
ordering, dimension inference, noise-channel lookup, memory-map layout — lives here and never
leaves.

The transpiler is deliberately thin: :func:`~pyquil.simulation._resolver.resolve_program` does
the expansion exactly as it always has, and :func:`transpile` only rephrases its output in
quax's vocabulary.

**Parameters.** Quil addresses parameters by memory reference; quax addresses them by position
in a flat vector, and requires that every gate argument own an *independent* slot so that a
gradient entry refers to one gate occurrence rather than an implicit sum over several.  A
program that reuses ``theta[0]`` across twenty gates therefore maps that one value onto twenty
slots.  :class:`ParameterBinding` owns that mapping in both directions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, final

import jax.numpy as jnp
import numpy as np
import quax as qx
from jax import Array

from pyquil.simulation._resolver import (
    MeasurementMode,
    ParametricGate,
    Resolution,
    resolve_program,
)

if TYPE_CHECKING:
    from pyquil.api import MemoryMap
    from pyquil.noise._noise_model import NoiseModelLike
    from pyquil.quil import Program


@final
@dataclass(frozen=True)
class ParameterBinding:
    """The map between Quil memory references and quax parameter slots.

    ``refs[k]`` is the ``(region, offset)`` that feeds slot ``k``.  Duplicates are expected and
    correct: each *occurrence* of a memory reference in the program owns its own slot, which is
    what makes a per-slot gradient unambiguous.

    Two directions are available, and they are not redundant:

    * :meth:`linearize` reads a :class:`~pyquil.api.MemoryMap` eagerly into the slot vector.
      This is the classic entry point and what :meth:`Simulator.compute` expects.
    * :meth:`scatter` does the same thing *differentiably*, from a vector in memory-reference
      space.  Composing through it gives a gradient with respect to each distinct
      ``theta[i]`` for free, because the VJP of a gather is a segment sum — the reuse is
      summed by the chain rule rather than by hand.

    :param refs: One ``(region, offset)`` per slot, in slot order.
    """

    refs: tuple[tuple[str, int], ...]

    @property
    def num_params(self) -> int:
        """The number of slots."""
        return len(self.refs)

    @cached_property
    def memory_layout(self) -> tuple[tuple[str, int], ...]:
        """The distinct memory references this program reads, in first-seen order.

        The coordinate system :meth:`scatter` expects and :meth:`accumulate` returns.
        """
        seen: dict[tuple[str, int], None] = {}
        for ref in self.refs:
            seen.setdefault(ref, None)
        return tuple(seen)

    @cached_property
    def slots_by_ref(self) -> Mapping[tuple[str, int], tuple[int, ...]]:
        """Every slot each memory reference feeds."""
        out: dict[tuple[str, int], list[int]] = {}
        for slot, ref in enumerate(self.refs):
            out.setdefault(ref, []).append(slot)
        return {ref: tuple(slots) for ref, slots in out.items()}

    @cached_property
    def _gather_index(self) -> np.ndarray:
        """Slot -> index into :attr:`memory_layout`."""
        position = {ref: i for i, ref in enumerate(self.memory_layout)}
        return np.asarray([position[ref] for ref in self.refs], dtype=np.int32)

    def linearize(self, memory_map: MemoryMap | None) -> Array:
        """Read a memory map into the flat slot vector.

        :param memory_map: Values for each declared memory region, as passed to the QVM.
        :return: A flat ``float`` vector with one entry per slot.
        """
        if not self.refs:
            return jnp.array([], dtype=float)
        if memory_map is None:
            raise ValueError(f"This program has {self.num_params} parameter(s); a memory map is required.")
        return jnp.array([float(memory_map[name][offset]) for name, offset in self.refs], dtype=float)

    def memory_vector(self, memory_map: MemoryMap) -> Array:
        """Read a memory map into :attr:`memory_layout` order, one entry per *distinct* reference.

        The input :meth:`scatter` expects, so the round trip is symmetric and a caller never
        has to build that vector by hand.

        :param memory_map: Values for each declared memory region.
        :return: One value per distinct memory reference.
        """
        return jnp.array([float(memory_map[name][offset]) for name, offset in self.memory_layout], dtype=float)

    def scatter(self, memory_vector: Array) -> Array:
        """Expand a memory-space vector to the slot vector, differentiably.

        :param memory_vector: One value per entry of :attr:`memory_layout`.
        :return: The flat slot vector.
        """
        return jnp.asarray(memory_vector)[self._gather_index]

    def accumulate(self, slot_gradient: Array) -> dict[tuple[str, int], float]:
        """Sum a per-slot gradient back into per-memory-reference gradients.

        Only needed for a gradient obtained *without* composing through :meth:`scatter`; that
        route already performs this sum exactly. Provided so the arithmetic is never left to a
        caller to reinvent.

        :param slot_gradient: One entry per slot, as ``jax.grad`` of a slot-space function.
        :return: The summed gradient, keyed by memory reference.
        """
        slot_gradient = jnp.asarray(slot_gradient)
        return {ref: float(jnp.sum(slot_gradient[jnp.asarray(slots)])) for ref, slots in self.slots_by_ref.items()}


@final
@dataclass(frozen=True)
class Transpilation:
    """A Quil program rendered as a quax circuit, with the parameter map that goes with it.

    :param circuit: The circuit, ready to hand to a quax simulator.
    :param qubits: The register, in the order the state's subsystems follow.
    :param binding: The map between Quil memory and the circuit's parameter slots.
    """

    circuit: qx.Circuit
    qubits: tuple[int, ...]
    binding: ParameterBinding

    @property
    def dims(self) -> tuple[int, ...]:
        """Per-qudit dimensions of the register, in :attr:`qubits` order."""
        return self.circuit.dims

    def linearize(self, memory_map: MemoryMap | None) -> Array:
        """Shorthand for ``binding.linearize(memory_map)``."""
        return self.binding.linearize(memory_map)


def _to_parameterized_gate(gate: ParametricGate) -> qx.ParameterizedGate:
    """Rephrase a resolver ``ParametricGate`` in quax's argument vocabulary.

    The resolver encodes each argument across two parallel tuples, using ``-1`` and ``nan`` as
    "not this one" sentinels; quax gives each argument a single tagged value.  The translation
    is total, because exactly one of the two sentinels is absent for every argument.
    """
    arguments: list[qx.GateArgument] = []
    for slot, value in zip(gate.param_indices, gate.concrete_values, strict=True):
        arguments.append(qx.Slot(index=slot) if slot >= 0 else qx.Constant(value=float(value)))
    return qx.ParameterizedGate(gate_fn=gate.gate_fn, arguments=tuple(arguments))


def circuit_from_resolution(resolution: Resolution) -> qx.Circuit:
    """Build a :class:`quax.Circuit` from an expanded program.

    :param resolution: The resolver's output.
    :return: The circuit, with one operation per expanded instruction.
    """
    ops: list[qx.Placement] = []
    for op, subsystem in zip(resolution.ops, resolution.subsystems, strict=True):
        placed = _to_parameterized_gate(op) if isinstance(op, ParametricGate) else op
        ops.append((placed, tuple(subsystem)))
    return qx.Circuit(dims=resolution.dims, ops=tuple(ops), num_params=len(resolution.param_refs))


def transpile(
    program: Program,
    qubits: list[int] | None = None,
    *,
    noise_model: NoiseModelLike | None = None,
    dims: tuple[int, ...] | None = None,
    measurement: MeasurementMode = "instrument",
) -> Transpilation:
    """Transpile a Quil program into a quax circuit.

    ``DEFGATE`` and ``DEFCIRCUIT`` definitions are expanded, noise channels are resolved against
    *noise_model*, and register dimensions are inferred from the resulting operators unless
    *dims* is given.

    ``MEASURE`` becomes a :class:`quax.QuantumInstrument` by default, which is the faithful
    representation; a simulator that cannot evolve one collapses it to its total channel itself
    (:class:`quax.DensityMatrixSimulator`) or rejects it (:class:`quax.StateVectorSimulator`).

    .. warning::
        **The register is big-endian.** The first entry of *qubits* is the *most* significant
        subsystem of the resulting state, so for ``qubits=[0, 1]`` the program ``X 0`` yields
        ``|10>`` (index 2).  Every other simulator in pyQuil is little-endian.  This is a
        deliberate departure: tying subsystem order to the ``qubits`` list makes the state's
        axes read in the order the register is written.  Amplitudes must be reversed to compare
        against pyQuil's other simulators or against QVM readout bit order.

    :param program: The Quil program.
    :param qubits: Explicit register, in state-subsystem order.  Defaults to the program's
        qubits in ascending order.  Pass this to include an idle spectator, or to fix an order.
    :param noise_model: Optional noise model; instructions with no channel are ideal.
    :param dims: Per-qudit dimensions in *qubits* order.  Defaults to inferring them.
    :param measurement: How ``MEASURE`` is represented; see
        :func:`~pyquil.simulation._resolver.resolve_program`.
    :return: The circuit, register and parameter binding.
    :raises ValueError: If *qubits* contains duplicates.
    """
    if qubits is not None and len(set(qubits)) != len(qubits):
        duplicates = sorted({q for q in qubits if qubits.count(q) > 1})
        raise ValueError(
            f"qubits contains duplicate entries {duplicates}: {qubits}. Each qubit must "
            "appear exactly once, since the list defines the register's subsystems."
        )
    resolution = resolve_program(program, noise_model, qubits, dims, measurement=measurement)
    register = tuple(qubits) if qubits is not None else tuple(sorted(program.get_qubit_indices()))
    return Transpilation(
        circuit=circuit_from_resolution(resolution),
        qubits=register,
        binding=ParameterBinding(refs=tuple(resolution.param_refs)),
    )
