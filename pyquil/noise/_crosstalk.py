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
r"""Cycle channels with crosstalk between the operations of a cycle.

A :class:`~pyquil.noise._channels.CycleChannel` is a tensor product of independent channels. A
:class:`CrosstalkCycleChannel` adds a crosstalk Hamiltonian :math:`V = \sum_e V_e`, a
:class:`~pyquil.paulis.PauliSum` such as static ZZ or XY exchange, that acts across those operations
for the duration of the cycle. The joint generator

.. math::

    \mathcal{L} = \sum_i \mathcal{L}_i - i [V, \cdot\,]

is a sum of local terms and so scales with the number of qubits, but its superoperator does not.
:meth:`CrosstalkCycleChannel.operations` lowers it to a short list of small operators instead:

1. **Toggling frame.** In the interaction picture of the ideal gates, :math:`U_0(s) = \bigotimes_i
   e^{-i H_i s}`, the cycle is :math:`U_0(T)\,\mathcal{T}\exp(-i\int_0^T \tilde V(s)\,ds)` with
   :math:`\tilde V_e(s) = U_0(s)^\dagger V_e U_0(s)`. Each :math:`\tilde V_e` acts only on the
   operations its term touches.
2. **First-order Magnus.** The cycle is cut into ``steps`` slices and :math:`\tilde V_e` is replaced
   on each by its exact time average (:func:`quax.interaction_picture_average`).
3. **Support.** Qudits on which an averaged term acts as the identity are dropped
   (:func:`quax.support`), so a term that commutes with its gates, such as ZZ during CZ, stays as
   small as it was written.
4. **Symmetric Trotter** across the terms of a slice, skipped when they all commute.

The crosstalk unitaries are applied first, followed by the constituents' own processes, unchanged.
The simulator's merge plan fuses them with their neighbours like any other operation.

With coupling angle :math:`\epsilon = \lVert V\rVert T` and decoherence :math:`\gamma T`, the error is
:math:`O(\epsilon^2/\mathrm{steps}^2)` from the Magnus truncation, :math:`O(\epsilon^3/\mathrm{steps}^2)`
from the Trotter splitting and :math:`O(\epsilon\,\gamma T)` from using the ideal (unitary) frame.
Terms that commute with the gates and with each other incur only the last.

.. note::
    The frame assumes each gate is driven at a constant rate for the whole cycle: its gate
    Hamiltonian is stretched from ``gate_time`` to the cycle's ``duration``. Measurement and reset
    constituents have no frame, so crosstalk on them is applied as written, before the operation.
    The crosstalk acts on the computational subspace; on a qudit with leakage levels it is
    zero-padded.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from numbers import Number
from typing import TypeAlias, cast, final

import jax.numpy as jnp
import numpy as np
import quax as qx

from pyquil.noise._channels import Channel, ChannelBase, CycleChannel, CycleConstituent
from pyquil.paulis import PauliSum, PauliTerm, sX, sY, sZ
from pyquil.quilbase import DefCircuit, Gate

CycleOperation: TypeAlias = tuple[qx.Unitary | qx.SuperOp | qx.QuantumInstrument, tuple[int, ...]]
"""An operator and the physical qubits it acts on, in the form the resolver emits."""

_Term: TypeAlias = tuple[qx.Observable, tuple[int, ...]]
"""A Hamiltonian term and the physical qubits it acts on, in the order of its dims."""

_COMMUTATOR_RTOL = math.sqrt(np.finfo(float).eps)
"""Commutator size, relative to the product of the terms' norms, below which two terms commute."""

_MAX_DENSE_DIMENSION = 32
"""Largest Hilbert-space dimension for which :meth:`CrosstalkCycleChannel.exact_process` runs by default."""


# ══════════════════════════════════════════════════════════
# Crosstalk Hamiltonians
# ══════════════════════════════════════════════════════════


def zz_crosstalk(qubit_0: int, qubit_1: int, rate: float) -> PauliSum:
    r"""Build static ZZ crosstalk :math:`\tfrac{2\pi\zeta}{4}(Z_0Z_1 - Z_0 - Z_1)`.

    ``rate`` is :math:`\zeta`, the frequency shift of :math:`|11\rangle` relative to the other basis
    states, in cycles per unit of the cycle's ``duration`` (Hz for a duration in seconds). Up to a
    global phase this is :math:`2\pi\zeta\, |11\rangle\langle 11|`.
    """
    scale = float(2 * np.pi * rate / 4)
    return scale * sZ(qubit_0) * sZ(qubit_1) - scale * sZ(qubit_0) - scale * sZ(qubit_1)


def xy_crosstalk(qubit_0: int, qubit_1: int, rate: float) -> PauliSum:
    r"""Build exchange crosstalk :math:`\tfrac{2\pi g}{2}(X_0X_1 + Y_0Y_1) = 2\pi g\,(|01\rangle\langle 10| + \mathrm{h.c.})`.

    ``rate`` is :math:`g`, in cycles per unit of the cycle's ``duration``.
    """
    scale = float(2 * np.pi * rate / 2)
    return scale * sX(qubit_0) * sX(qubit_1) + scale * sY(qubit_0) * sY(qubit_1)


def _pauli_terms(crosstalk: PauliSum) -> tuple[_Term, ...]:
    """Group the non-identity terms of *crosstalk* by the qubits they act on.

    A term on a subset of another term's qubits joins that term's group, so the single-qubit
    terms of :func:`zz_crosstalk` stay with their ZZ term.

    :return: For each group, its Hamiltonian and its qubits, in increasing order. Identity terms are
        a global phase and are dropped.
    """
    from pyquil.simulation.tools import lifted_pauli

    def support(term: PauliTerm) -> tuple[int, ...]:
        return tuple(sorted(cast(int, q) for q in term.get_qubits()))

    groups: dict[tuple[int, ...], list[PauliTerm]] = {}
    for term in sorted(crosstalk.terms, key=lambda term: -len(support(term))):
        qubits = support(term)
        if qubits:
            home = next((group for group in groups if set(qubits) <= set(group)), qubits)
            groups.setdefault(home, []).append(term)

    def observable(qubits: tuple[int, ...], terms: list[PauliTerm]) -> qx.Observable:
        # lifted_pauli puts the last qubit it is given first (most significant); quax is big-endian.
        matrix = jnp.asarray(lifted_pauli(PauliSum(terms), list(reversed(qubits))))
        dims = (2,) * len(qubits)
        return qx.Observable.from_matrix(matrix, (dims, dims))

    return tuple((observable(qubits, terms), qubits) for qubits, terms in groups.items())


def _validate_crosstalk(crosstalk: PauliSum) -> None:
    for term in crosstalk.terms:
        coefficient = term.coefficient
        if not isinstance(coefficient, Number) or complex(coefficient).imag != 0:
            raise ValueError(f"Crosstalk coefficients must be real numbers, got {coefficient} on {term}.")
        if not all(isinstance(q, int) for q in term.get_qubits()):
            raise ValueError(f"Crosstalk must act on integer qubits, got {term}.")


# ══════════════════════════════════════════════════════════
# Lowering helpers
# ══════════════════════════════════════════════════════════


def _on_common_qubits(a: _Term, b: _Term) -> tuple[np.ndarray, np.ndarray]:
    """Embed the matrices of two terms on the union of their qubits."""
    (ha, qa), (hb, qb) = a, b
    qubits = tuple(dict.fromkeys(qa + qb))
    dim_of = dict(zip(qa + qb, ha.dims[0] + hb.dims[0], strict=True))
    dims = tuple(dim_of[q] for q in qubits)
    ma = np.asarray(qx.embed(ha, dims, tuple(qubits.index(q) for q in qa)).matrix)
    mb = np.asarray(qx.embed(hb, dims, tuple(qubits.index(q) for q in qb)).matrix)
    return ma, mb


def _commute(a: _Term, b: _Term) -> bool:
    """Whether two terms commute; terms on disjoint qubits always do."""
    if not set(a[1]) & set(b[1]):
        return True
    ma, mb = _on_common_qubits(a, b)
    return bool(np.linalg.norm(ma @ mb - mb @ ma) <= _COMMUTATOR_RTOL * np.linalg.norm(ma) * np.linalg.norm(mb))


def _strang(terms: Sequence[_Term], time: float) -> list[tuple[qx.Unitary, tuple[int, ...]]]:
    """Evolve the sum of *terms* for *time*: exactly when they commute, else by a symmetric Trotter step."""
    if all(_commute(a, b) for i, a in enumerate(terms) for b in terms[i + 1 :]):
        return [(qx.evolve(h, time), qubits) for h, qubits in terms]
    *outer, (last, last_qubits) = terms
    halves = [(qx.evolve(h, time / 2), qubits) for h, qubits in outer]
    return [*halves, (qx.evolve(last, time), last_qubits), *reversed(halves)]


def _covariance(a: _Term, b: _Term) -> float:
    r"""Covariance :math:`\mathrm{tr}(AB)/d - \mathrm{tr}A\,\mathrm{tr}B/d^2` of two Hermitian terms in the maximally mixed state."""
    ma, mb = _on_common_qubits(a, b)
    d = ma.shape[0]
    return float(np.real(np.trace(ma @ mb) / d - np.trace(ma) * np.trace(mb) / d**2))


def _constituent_dims(channel: CycleConstituent) -> tuple[int, ...]:
    """Per-qudit dimensions of a cycle constituent, in the order of its ``qubits``."""
    if isinstance(channel, ChannelBase):
        return channel.dims
    return tuple(int(d) for d in channel.process.dims[0])


# ══════════════════════════════════════════════════════════
# Crosstalk cycle channel
# ══════════════════════════════════════════════════════════


@final
@dataclass(frozen=True)
class CrosstalkCycleChannel:
    """A cycle channel whose operations are coupled by a crosstalk Hamiltonian.

    See the module docstring for the model and for how :meth:`operations` approximates it.
    """

    cycle: CycleChannel
    """The independent (crosstalk-free) channels of the cycle."""

    crosstalk: PauliSum
    """The crosstalk Hamiltonian on qubits of the cycle, as an angular frequency in the reciprocal of
    the time unit of ``duration`` (rad/s for a duration in seconds). See :func:`zz_crosstalk` and
    :func:`xy_crosstalk`."""

    duration: float
    """Length of the cycle."""

    steps: int = 1
    """Number of time slices the toggling-frame average is taken over. More slices are more
    accurate and emit proportionally more operators; terms that commute with their gates are exact
    at one."""

    def __post_init__(self) -> None:
        """Validate the crosstalk against the cycle."""
        if isinstance(self.crosstalk, PauliTerm):
            object.__setattr__(self, "crosstalk", PauliSum([self.crosstalk]))
        if not isinstance(self.crosstalk, PauliSum):
            raise TypeError(f"Crosstalk must be a PauliSum or PauliTerm, got {type(self.crosstalk).__name__}.")
        if self.duration <= 0:
            raise ValueError(f"Cycle duration must be positive, got {self.duration}.")
        if self.steps < 1:
            raise ValueError(f"steps must be at least 1, got {self.steps}.")
        _validate_crosstalk(self.crosstalk)
        missing = sorted(set(cast(list[int], self.crosstalk.get_qubits())) - set(self._qubit_dims))
        if missing:
            raise ValueError(f"Crosstalk acts on qubit(s) {missing} outside the cycle.")

    # ──────────────────────────────────────────────
    # Delegation to the underlying cycle
    # ──────────────────────────────────────────────

    @property
    def inst(self) -> Gate:
        """The cycle invocation the channel applies to."""
        return self.cycle.inst

    @property
    def defcircuit(self) -> DefCircuit:
        """The ``DEFCIRCUIT`` of the cycle."""
        return self.cycle.defcircuit

    @property
    def channels(self) -> tuple[CycleConstituent, ...]:
        """The constituent channels, one per operation of the cycle."""
        return self.cycle.channels

    @property
    def qubits(self) -> list[int]:
        """All qubits in the cycle."""
        return self.cycle.qubits

    @cached_property
    def _qubit_dims(self) -> dict[int, int]:
        return {q: d for ch in self.channels for q, d in zip(ch.qubits, _constituent_dims(ch), strict=True)}

    # ──────────────────────────────────────────────
    # Lowering
    # ──────────────────────────────────────────────

    def _frame_hamiltonian(self, channel: CycleConstituent) -> qx.Observable:
        """Hamiltonian that generates a constituent's ideal operation over the cycle duration."""
        if isinstance(channel, Channel):
            return channel.gate_hamiltonian * (channel.gate_time / self.duration)
        if isinstance(channel, ChannelBase):
            return qx.unitary_to_hamiltonian(channel.ideal_unitary) * (1.0 / self.duration)
        dims = _constituent_dims(channel)
        dimension = math.prod(dims)
        return qx.Observable.from_matrix(jnp.zeros((dimension, dimension), dtype=complex), (dims, dims))

    def _averaged_terms(self, steps: int) -> tuple[tuple[_Term, ...], ...]:
        """Average each crosstalk term in the toggling frame over each of *steps* slices, on its support."""
        block_of = {q: index for index, ch in enumerate(self.channels) for q in ch.qubits}
        frames = [self._frame_hamiltonian(ch) for ch in self.channels]
        edges = jnp.linspace(0.0, self.duration, steps + 1)

        slices: list[list[_Term]] = [[] for _ in range(steps)]
        for bare, term_qubits in _pauli_terms(self.crosstalk):
            blocks = list(dict.fromkeys(block_of[q] for q in term_qubits))
            qubits = tuple(q for b in blocks for q in self.channels[b].qubits)
            dims = tuple(self._qubit_dims[q] for q in qubits)

            def positions(of: Sequence[int], qubits: tuple[int, ...] = qubits) -> tuple[int, ...]:
                return tuple(qubits.index(q) for q in of)

            frame = sum(
                (qx.embed(frames[b], dims, positions(self.channels[b].qubits)) for b in blocks[1:]),
                start=qx.embed(frames[blocks[0]], dims, positions(self.channels[blocks[0]].qubits)),
            )
            averages = qx.interaction_picture_average(qx.embed(bare, dims, positions(term_qubits)), frame, edges)
            for index in range(steps):
                average = qx.Observable.from_matrix(averages.matrix[index], (dims, dims))
                kept = qx.support(average)
                # A term reduced to a multiple of the identity is a global phase.
                if kept:
                    slices[index].append((qx.restrict(average, kept), tuple(qubits[k] for k in kept)))
        return tuple(tuple(terms) for terms in slices)

    @cached_property
    def _crosstalk_operations(self) -> tuple[tuple[qx.Unitary, tuple[int, ...]], ...]:
        """The crosstalk unitaries, in the order they are applied."""
        step = self.duration / self.steps
        return tuple(op for terms in self._averaged_terms(self.steps) if terms for op in _strang(terms, step))

    def operations(self) -> tuple[CycleOperation, ...]:
        """Lower the cycle to a sequence of small operators, in the order they are applied.

        The crosstalk unitaries come first, followed by each constituent's own ``process``.
        """
        constituents = tuple((op, tuple(ch.qubits)) for op, ch in zip(self.cycle.operator, self.channels, strict=True))
        return (*self._crosstalk_operations, *constituents)

    def exact_process(self, max_dimension: int = _MAX_DENSE_DIMENSION) -> qx.SuperOp:
        """Evolve the joint generator, crosstalk included, into one superoperator on :attr:`qubits`.

        The reference :meth:`operations` approximates; its cost grows as the fourth power of the
        cycle's dimension, so it is limited to small cycles. Every constituent must be a
        :class:`~pyquil.noise._channels.Channel`, since the crosstalk acts *during* the gates.
        """
        qubits = self.qubits
        dims = tuple(self._qubit_dims[q] for q in qubits)
        dimension = math.prod(dims)
        if dimension > max_dimension:
            raise ValueError(
                f"A dense process on dimension {dimension} exceeds max_dimension={max_dimension}; the joint "
                "superoperator grows as the fourth power of the dimension. Use operations() instead."
            )

        def positions(of: Sequence[int]) -> tuple[int, ...]:
            return tuple(qubits.index(q) for q in of)

        hamiltonian = qx.Observable.from_matrix(jnp.zeros((dimension, dimension), dtype=complex), (dims, dims))
        jumps = [jnp.zeros((1, dimension, dimension), dtype=complex)]
        for channel in self.channels:
            if not isinstance(channel, Channel):
                raise TypeError(
                    f"exact_process needs a generator for every operation, but {channel.inst} is a "
                    f"{type(channel).__name__}."
                )
            # Stretch the generator so that it produces the same process over the cycle duration.
            scale = channel.gate_time / self.duration
            lindbladian = channel.lindbladian
            if lindbladian.hamiltonian is not None:
                hamiltonian = hamiltonian + qx.embed(lindbladian.hamiltonian * scale, dims, positions(channel.qubits))
            embedded = qx.embed(lindbladian.jump_operators, dims, positions(channel.qubits))
            jumps.append(math.sqrt(scale) * embedded.matrix)
        for bare, term_qubits in _pauli_terms(self.crosstalk):
            hamiltonian = hamiltonian + qx.embed(bare, dims, positions(term_qubits))
        generator = qx.Lindbladian(
            hamiltonian=qx.Observable.from_matrix(hamiltonian.matrix, (dims, dims)),
            jump_operators=qx.Operator.from_matrix(jnp.concatenate(jumps), (dims, dims)),
        )
        return cast(qx.SuperOp, qx.evolve(generator, self.duration))

    # ──────────────────────────────────────────────
    # Fidelities
    # ──────────────────────────────────────────────

    @cached_property
    def process_fidelity(self) -> float:
        r"""Estimated process fidelity of the cycle's gates, crosstalk included.

        The crosstalk is a coherent error :math:`e^{-i\Omega}` in the frame of the gates, with
        :math:`\Omega = \sum_e \Omega_e` the whole-cycle Magnus generator of its terms. To second
        order its process fidelity is :math:`1 - \mathrm{Var}(\Omega)` in the maximally mixed state,
        where terms on disjoint qubits do not covary; this does not depend on ``steps``. The result
        multiplies :attr:`CycleChannel.process_fidelity`, which is correct to first order in the
        two infidelities. Reset and measurement channels are skipped, as there.
        """
        (terms,) = self._averaged_terms(1)
        generators = [(h * self.duration, qubits) for h, qubits in terms]
        variance = sum(
            (1 if i == j else 2) * _covariance(a, b)
            for i, a in enumerate(generators)
            for j, b in enumerate(generators[: i + 1])
            if set(a[1]) & set(b[1])
        )
        return self.cycle.process_fidelity * max(0.0, 1.0 - variance)

    @cached_property
    def process_infidelity(self) -> float:
        """``1 - process_fidelity``."""
        return 1.0 - self.process_fidelity

    @cached_property
    def average_gate_fidelity(self) -> float:
        """Average gate fidelity of the cycle's gates, converted once from :attr:`process_fidelity`."""
        dims = tuple(dim for ch in self.channels if isinstance(ch, ChannelBase) for dim in ch.dims)
        if not dims:
            return 1.0
        return float(qx.process_fidelity_to_average_fidelity(self.process_fidelity, dims=dims))

    @cached_property
    def average_gate_infidelity(self) -> float:
        """``1 - average_gate_fidelity``."""
        return 1.0 - self.average_gate_fidelity

    # ──────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize the channel to a JSON string."""
        data = {
            "cycle": self.cycle.to_json(),
            "crosstalk": self.crosstalk.compact_str(),
            "duration": self.duration,
            "steps": self.steps,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> CrosstalkCycleChannel:
        """Deserialize a channel from a JSON string produced by :meth:`to_json`."""
        data = json.loads(json_str)
        return cls(
            cycle=CycleChannel.from_json(data["cycle"]),
            crosstalk=PauliSum.from_compact_str(data["crosstalk"]),
            duration=data["duration"],
            steps=data["steps"],
        )

    # ──────────────────────────────────────────────
    # Dunder methods
    # ──────────────────────────────────────────────

    def __str__(self) -> str:
        """Return a simplified string representation showing the cycle and its estimated fidelity."""
        return f"<{self.inst.out()} + crosstalk ~ ({100 * self.process_fidelity:.2f}%)>"

    def __eq__(self, other: object) -> bool:
        """Check equality by cycle, crosstalk, duration and steps."""
        if not isinstance(other, CrosstalkCycleChannel):
            return False
        return (
            self.cycle == other.cycle
            and self.crosstalk == other.crosstalk
            and self.duration == other.duration
            and self.steps == other.steps
        )

    __hash__ = None  # type: ignore[assignment]


__all__ = ["CrosstalkCycleChannel", "CycleOperation", "xy_crosstalk", "zz_crosstalk"]
