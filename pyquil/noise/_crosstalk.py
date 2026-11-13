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
:class:`CrosstalkCycleChannel` adds :class:`Coupling` terms -- small Hamiltonians such as ZZ or XY
exchange -- that act across those operations for the duration of the cycle. The joint generator

.. math::

    \mathcal{L} = \sum_i \mathcal{L}_i - i \sum_e [V_e, \cdot\,]

is a sum of local terms and so scales with the number of qubits, but its superoperator does not.
:meth:`CrosstalkCycleChannel.operations` lowers it to a short list of small operators instead:

1. **Toggling frame.** In the interaction picture of the ideal gates, :math:`U_0(s) = \bigotimes_i
   e^{-i H_i s}`, the cycle is :math:`U_0(T)\,\mathcal{T}\exp(-i\int_0^T \tilde V(s)\,ds)` with
   :math:`\tilde V_e(s) = U_0(s)^\dagger V_e U_0(s)`. Each :math:`\tilde V_e` acts only on the
   operations its coupling touches.
2. **First-order Magnus.** The cycle is cut into ``steps`` slices and :math:`\tilde V_e` is replaced
   on each by its exact time average (closed form in the eigenbasis of the frame Hamiltonian).
3. **Support pruning.** Qudits on which an averaged term acts as the identity are dropped, so a
   coupling that commutes with its gates (ZZ during CZ) stays as small as it was written.
4. **Fusion.** The terms of a slice are summed into clusters of at most ``max_fused_qubits``
   qubits, each exponentiated exactly as one unitary.
5. **Symmetric Trotter** across the clusters of a slice, skipped when they all commute.

The crosstalk unitaries are applied first, followed by the constituents' own processes, unchanged.

With coupling angle :math:`\epsilon = \lVert V\rVert T` and decoherence :math:`\gamma T`, the error is
:math:`O(\epsilon^2/\mathrm{steps}^2)` from the Magnus truncation, :math:`O(\epsilon^3/\mathrm{steps}^2)`
from the Trotter splitting and :math:`O(\epsilon\,\gamma T)` from using the ideal (unitary) frame.
Couplings that commute with the gates and with each other incur only the last.

.. note::
    The frame assumes each gate is driven at a constant rate for the whole cycle,
    :math:`H_i = i\log(U_i)/T`. This is the same assumption :class:`~pyquil.noise._channels.Channel`
    makes for its ``gate_hamiltonian``. Measurement and reset constituents have no frame (their
    couplings are applied as written, before the operation).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, TypeAlias, cast, final

import jax.numpy as jnp
import numpy as np
import quax as qx
from scipy.linalg import expm as scipy_expm

from pyquil.noise._channels import (
    Channel,
    ChannelBase,
    CycleChannel,
    CycleConstituent,
    _pack_operator,
    _unpack_complex_array,
    _unpack_dims,
)
from pyquil.quilbase import DefCircuit, Gate

CycleOperation: TypeAlias = tuple[qx.Unitary | qx.SuperOp | qx.QuantumInstrument, tuple[int, ...]]
"""An operator and the physical qubits it acts on, in the form the resolver emits."""

_IDENTITY_TOLERANCE = 1e-12
"""Relative tolerance below which an averaged coupling is taken to act trivially on a qudit."""

_MAX_DENSE_DIMENSION = 32
"""Largest Hilbert-space dimension for which the dense reference processes are built by default."""


# ══════════════════════════════════════════════════════════
# Dense helpers (NumPy, big-endian like quax)
# ══════════════════════════════════════════════════════════


def _embed_matrix(matrix: np.ndarray, positions: Sequence[int], dims: Sequence[int]) -> np.ndarray:
    """Embed an operator acting on ``positions`` (in that order) into a register with ``dims``."""
    n = len(dims)
    rest = [i for i in range(n) if i not in positions]
    order = [*positions, *rest]
    shape = [dims[i] for i in order]
    rest_dim = int(np.prod([dims[i] for i in rest], dtype=int))
    tensor = np.kron(matrix, np.eye(rest_dim)).reshape(shape + shape)
    perm = list(np.argsort(order))
    total = int(np.prod(dims, dtype=int))
    return tensor.transpose(perm + [n + p for p in perm]).reshape(total, total)


def _prune_support(
    matrix: np.ndarray, qubits: tuple[int, ...], dims: tuple[int, ...]
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Drop every qudit on which ``matrix`` acts as the identity.

    :return: The reduced matrix with the qubits and dims it still acts on. An operator proportional
        to the identity comes back as a ``1x1`` matrix on no qubits.
    """
    scale = max(float(np.linalg.norm(matrix)), 1.0)
    index = 0
    while index < len(qubits):
        n = len(dims)
        tensor = matrix.reshape(dims + dims)
        # Bring the candidate qudit's (out, in) axes to the front: tensor[a, a', rest, rest'].
        others = [i for i in range(n) if i != index]
        tensor = tensor.transpose([index, n + index, *others, *[n + i for i in others]])
        reduced = tensor[0, 0]
        expected = np.einsum("ab,...->ab...", np.eye(dims[index]), reduced)
        if np.linalg.norm(tensor - expected) <= _IDENTITY_TOLERANCE * scale:
            qubits = qubits[:index] + qubits[index + 1 :]
            dims = dims[:index] + dims[index + 1 :]
            total = int(np.prod(dims, dtype=int))
            matrix = reduced.reshape(total, total)
        else:
            index += 1
    return matrix, qubits, dims


def _slice_average(coupling: np.ndarray, frame: np.ndarray, start: float, stop: float) -> np.ndarray:
    r"""Average :math:`e^{iH_0 s} V e^{-iH_0 s}` over ``s`` in ``[start, stop]``, exactly.

    In the eigenbasis of the frame Hamiltonian the toggled coupling is :math:`V_{mn}
    e^{i\omega_{mn}s}` with :math:`\omega_{mn} = E_m - E_n`, whose average is :math:`V_{mn}
    e^{i\omega_{mn}(a+b)/2}\,\mathrm{sinc}(\omega_{mn}(b-a)/2)`.
    """
    energies, vectors = np.linalg.eigh(frame)
    omega = energies[:, None] - energies[None, :]
    # np.sinc(x) is sin(pi x) / (pi x).
    factor = np.exp(1j * omega * (start + stop) / 2) * np.sinc(omega * (stop - start) / (2 * np.pi))
    rotated = vectors.conj().T @ coupling @ vectors
    averaged = vectors @ (rotated * factor) @ vectors.conj().T
    return np.asarray((averaged + averaged.conj().T) / 2)


@dataclass(frozen=True)
class _Term:
    """An averaged coupling: a Hermitian matrix on a few qubits."""

    matrix: np.ndarray
    qubits: tuple[int, ...]
    dims: tuple[int, ...]

    def unitary(self, time: float) -> qx.Unitary:
        return qx.Unitary.from_matrix(jnp.asarray(scipy_expm(-1j * time * self.matrix)), (self.dims, self.dims))


def _all_commute(terms: Sequence[_Term]) -> bool:
    """Whether every pair of terms with overlapping support commutes."""
    for i, a in enumerate(terms):
        for b in terms[i + 1 :]:
            if not set(a.qubits) & set(b.qubits):
                continue
            qubits = tuple(dict.fromkeys(a.qubits + b.qubits))
            dim_of = dict(zip(a.qubits + b.qubits, a.dims + b.dims, strict=True))
            dims = [dim_of[q] for q in qubits]
            ma = _embed_matrix(a.matrix, [qubits.index(q) for q in a.qubits], dims)
            mb = _embed_matrix(b.matrix, [qubits.index(q) for q in b.qubits], dims)
            scale = max(float(np.linalg.norm(ma) * np.linalg.norm(mb)), 1e-300)
            if np.linalg.norm(ma @ mb - mb @ ma) > _IDENTITY_TOLERANCE * scale:
                return False
    return True


def _fuse_terms(terms: Sequence[_Term], max_qubits: int) -> tuple[_Term, ...]:
    """Sum terms into as few terms as possible, each on at most ``max_qubits`` qubits.

    Terms summed together are exponentiated exactly rather than split, and a simulator pays per
    operator far more than per qubit of a small operator, so a few wide unitaries beat many
    two-qubit ones. Clusters are grown greedily around the qubit with the most terms left, which
    turns a lattice's couplers into stars. A term wider than ``max_qubits`` is kept on its own.
    """
    remaining = list(terms)
    fused: list[_Term] = []
    while remaining:
        counts: dict[int, int] = {}
        for term in remaining:
            for q in term.qubits:
                counts[q] = counts.get(q, 0) + 1
        centre = max(counts, key=lambda q: counts[q])
        # The centre's terms come first; a second pass then picks up anything else that fits.
        ordered = sorted(remaining, key=lambda term: centre not in term.qubits)
        members: list[_Term] = [ordered[0]]
        dim_of = dict(zip(ordered[0].qubits, ordered[0].dims, strict=True))
        for term in ordered[1:]:
            union = {**dim_of, **dict(zip(term.qubits, term.dims, strict=True))}
            if len(union) <= max_qubits and set(term.qubits) & set(dim_of):
                members.append(term)
                dim_of = union
        chosen = {id(term) for term in members}
        remaining = [term for term in remaining if id(term) not in chosen]

        qubits = tuple(dim_of)
        dims = tuple(dim_of[q] for q in qubits)
        matrix = np.zeros((int(np.prod(dims, dtype=int)),) * 2, dtype=complex)
        for term in members:
            matrix += _embed_matrix(term.matrix, [qubits.index(q) for q in term.qubits], dims)
        fused.append(_Term(matrix, qubits, dims))
    return tuple(fused)


def _constituent_dims(channel: CycleConstituent) -> tuple[int, ...]:
    """Per-qudit dimensions of a cycle constituent, in the order of its ``qubits``."""
    if isinstance(channel, ChannelBase):
        return channel.dims
    return tuple(int(d) for d in channel.process.dims[0])


# ══════════════════════════════════════════════════════════
# Coupling
# ══════════════════════════════════════════════════════════


@final
@dataclass(frozen=True)
class Coupling:
    """A Hamiltonian term coupling a few qubits for the duration of a cycle.

    The Hamiltonian is an angular frequency in the reciprocal of the time unit used for the
    cycle's ``duration``: rad/s for a duration in seconds.
    """

    qubits: tuple[int, ...]
    """The physical qubits the term acts on."""

    hamiltonian: qx.Observable
    """The coupling Hamiltonian, on the qubits in the order given."""

    def __post_init__(self) -> None:
        """Validate the qubits against the Hamiltonian."""
        object.__setattr__(self, "qubits", tuple(int(q) for q in self.qubits))
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"Coupling qubits must be distinct, got {self.qubits}.")
        if len(self.qubits) != len(self.dims):
            raise ValueError(
                f"Coupling on qubits {self.qubits} needs a Hamiltonian on {len(self.qubits)} qudit(s), "
                f"got dims {self.dims}."
            )
        if np.asarray(self.hamiltonian.matrix).ndim != 2:
            raise ValueError("Coupling Hamiltonians must not carry ensemble dimensions.")

    @property
    def dims(self) -> tuple[int, ...]:
        """Per-qudit dimensions of the Hamiltonian."""
        return tuple(int(d) for d in self.hamiltonian.dims[0])

    @classmethod
    def from_matrix(cls, qubits: Sequence[int], matrix: Any, dims: Sequence[int] | None = None) -> Coupling:
        """Build a coupling from a Hermitian matrix (big-endian over ``qubits``).

        :param qubits: The physical qubits the term acts on.
        :param matrix: Hermitian matrix in angular-frequency units.
        :param dims: Per-qudit dimensions; qubits when omitted.
        """
        dims_tuple = tuple(dims) if dims is not None else (2,) * len(qubits)
        array = np.asarray(matrix, dtype=complex)
        if not np.allclose(array, array.conj().T):
            raise ValueError("A coupling Hamiltonian must be Hermitian.")
        return cls(tuple(qubits), qx.Observable.from_matrix(jnp.asarray(array), (dims_tuple, dims_tuple)))

    @classmethod
    def zz(cls, qubit_0: int, qubit_1: int, rate_hz: float, dims: tuple[int, int] = (2, 2)) -> Coupling:
        r"""Build a static ZZ coupling :math:`2\pi\zeta\, n \otimes n`.

        ``rate_hz`` is :math:`\zeta`, the frequency shift of ``|11>`` relative to ``|01>`` and
        ``|10>``. Written with number operators so it extends to qutrits; on qubits it equals
        :math:`(2\pi\zeta/4)(ZZ - ZI - IZ + II)`.
        """
        n0, n1 = np.diag(np.arange(dims[0])), np.diag(np.arange(dims[1]))
        return cls.from_matrix((qubit_0, qubit_1), 2 * np.pi * rate_hz * np.kron(n0, n1), dims)

    @classmethod
    def xy(cls, qubit_0: int, qubit_1: int, rate_hz: float, dims: tuple[int, int] = (2, 2)) -> Coupling:
        r"""Build an exchange coupling :math:`2\pi g\,(|01\rangle\langle 10| + \mathrm{h.c.})`.

        ``rate_hz`` is :math:`g`; on qubits the term equals :math:`(2\pi g/2)(XX + YY)`.
        """
        exchange = np.zeros((dims[0] * dims[1], dims[0] * dims[1]), dtype=complex)
        exchange[1, dims[1]] = 1.0  # |01><10|
        return cls.from_matrix((qubit_0, qubit_1), 2 * np.pi * rate_hz * (exchange + exchange.T), dims)

    def to_json(self) -> str:
        """Serialize the coupling to a JSON string."""
        return json.dumps({"qubits": list(self.qubits), "hamiltonian": _pack_operator(self.hamiltonian)})

    @classmethod
    def from_json(cls, json_str: str) -> Coupling:
        """Deserialize a coupling from a JSON string produced by :meth:`to_json`."""
        data = json.loads(json_str)
        ham = data["hamiltonian"]
        hamiltonian = qx.Observable.from_matrix(_unpack_complex_array(ham), _unpack_dims(ham["dims"]))
        return cls(tuple(data["qubits"]), hamiltonian)

    def __eq__(self, other: object) -> bool:
        """Compare by qubits and Hamiltonian matrix."""
        if not isinstance(other, Coupling):
            return False
        return (
            self.qubits == other.qubits
            and self.dims == other.dims
            and bool(np.allclose(self.hamiltonian.matrix, other.hamiltonian.matrix))
        )

    __hash__ = None  # type: ignore[assignment]


# ══════════════════════════════════════════════════════════
# Crosstalk cycle channel
# ══════════════════════════════════════════════════════════


@final
@dataclass(frozen=True)
class CrosstalkCycleChannel:
    """A cycle channel whose operations are coupled by crosstalk Hamiltonians.

    See the module docstring for the model and for how :meth:`operations` approximates it.
    """

    cycle: CycleChannel
    """The independent (crosstalk-free) channels of the cycle."""

    couplings: tuple[Coupling, ...]
    """The crosstalk terms, each on qubits of the cycle."""

    duration: float
    """Length of the cycle, in the time unit of the couplings."""

    steps: int = 1
    """Number of time slices the toggling-frame average is taken over. More slices are more
    accurate and emit proportionally more operators; couplings that commute with their gates are
    exact at one."""

    max_fused_qubits: int = 4
    """Widest unitary the couplings of a slice are fused into. Couplings fused together are
    exponentiated exactly and applied as one operator; the rest are Trotterized against each other."""

    def __post_init__(self) -> None:
        """Validate the couplings against the cycle."""
        object.__setattr__(self, "couplings", tuple(self.couplings))
        if self.duration <= 0:
            raise ValueError(f"Cycle duration must be positive, got {self.duration}.")
        if self.steps < 1:
            raise ValueError(f"steps must be at least 1, got {self.steps}.")
        if self.max_fused_qubits < 1:
            raise ValueError(f"max_fused_qubits must be at least 1, got {self.max_fused_qubits}.")
        dim_of = self._qubit_dims
        for coupling in self.couplings:
            missing = [q for q in coupling.qubits if q not in dim_of]
            if missing:
                raise ValueError(f"Coupling on {coupling.qubits} acts on qubit(s) {missing} outside the cycle.")
            expected = tuple(dim_of[q] for q in coupling.qubits)
            if coupling.dims != expected:
                raise ValueError(
                    f"Coupling on {coupling.qubits} has dims {coupling.dims}, but the cycle's channels have "
                    f"dims {expected} on those qubits."
                )

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

    def _frame_hamiltonian(self, channel: CycleConstituent) -> np.ndarray:
        """Hamiltonian that generates a constituent's ideal operation over the cycle duration."""
        dimension = int(np.prod(_constituent_dims(channel), dtype=int))
        if not isinstance(channel, ChannelBase):
            return np.zeros((dimension, dimension), dtype=complex)
        return np.asarray(qx.unitary_to_hamiltonian(channel.ideal_unitary).matrix) / self.duration

    @cached_property
    def _slices(self) -> tuple[tuple[_Term, ...], ...]:
        """The averaged, support-pruned and fused coupling terms of each time slice."""
        block_of = {q: index for index, ch in enumerate(self.channels) for q in ch.qubits}
        frames = [self._frame_hamiltonian(ch) for ch in self.channels]
        edges = np.linspace(0.0, self.duration, self.steps + 1)

        slices: list[list[_Term]] = [[] for _ in range(self.steps)]
        for coupling in self.couplings:
            blocks = list(dict.fromkeys(block_of[q] for q in coupling.qubits))
            qubits = tuple(q for b in blocks for q in self.channels[b].qubits)
            dims = tuple(self._qubit_dims[q] for q in qubits)
            frame = np.zeros((int(np.prod(dims, dtype=int)),) * 2, dtype=complex)
            for b in blocks:
                frame += _embed_matrix(frames[b], [qubits.index(q) for q in self.channels[b].qubits], dims)
            bare = _embed_matrix(
                np.asarray(coupling.hamiltonian.matrix), [qubits.index(q) for q in coupling.qubits], dims
            )
            for index in range(self.steps):
                averaged = _slice_average(bare, frame, float(edges[index]), float(edges[index + 1]))
                matrix, term_qubits, term_dims = _prune_support(averaged, qubits, dims)
                # A term reduced to a scalar is a global phase.
                if term_qubits:
                    slices[index].append(_Term(matrix, term_qubits, term_dims))
        return tuple(_fuse_terms(terms, self.max_fused_qubits) for terms in slices)

    @cached_property
    def _crosstalk_operations(self) -> tuple[tuple[qx.Unitary, tuple[int, ...]], ...]:
        """The crosstalk unitaries, in the order they are applied."""
        step = self.duration / self.steps
        operations: list[tuple[qx.Unitary, tuple[int, ...]]] = []
        for terms in self._slices:
            if not terms:
                continue
            if _all_commute(terms):
                operations.extend((term.unitary(step), term.qubits) for term in terms)
                continue
            # Symmetric (Strang) splitting; the two half steps of the last term are fused.
            halves = [(term.unitary(step / 2), term.qubits) for term in terms[:-1]]
            operations.extend(halves)
            operations.append((terms[-1].unitary(step), terms[-1].qubits))
            operations.extend(reversed(halves))
        return tuple(operations)

    def operations(self) -> tuple[CycleOperation, ...]:
        """Lower the cycle to a sequence of small operators, in the order they are applied.

        The crosstalk unitaries come first, followed by each constituent's own ``process``.
        """
        constituents: list[CycleOperation] = [(ch.process, tuple(ch.qubits)) for ch in self.channels]
        return (*self._crosstalk_operations, *constituents)

    # ──────────────────────────────────────────────
    # Dense reference processes (small registers only)
    # ──────────────────────────────────────────────

    def _dense_dims(self, max_dimension: int) -> tuple[int, ...]:
        dims = tuple(self._qubit_dims[q] for q in self.qubits)
        dimension = int(np.prod(dims, dtype=int))
        if dimension > max_dimension:
            raise ValueError(
                f"A dense process on dimension {dimension} exceeds max_dimension={max_dimension}; the joint "
                "superoperator grows as the fourth power of the dimension. Use operations() instead."
            )
        return dims

    def lowered_process(self, max_dimension: int = _MAX_DENSE_DIMENSION) -> qx.SuperOp:
        """Compose :meth:`operations` into one superoperator on :attr:`qubits`.

        This is the channel a simulator actually applies. Intended for checking the approximation
        on small cycles.
        """
        dims = self._dense_dims(max_dimension)
        qubits = self.qubits
        total: qx.SuperOp | None = None
        for op, op_qubits in self.operations():
            if isinstance(op, qx.QuantumInstrument):
                raise TypeError("A dense process is only defined for cycles of gates and resets.")
            embedded = qx.embed(qx.to_superop(op), dims, tuple(qubits.index(q) for q in op_qubits))
            total = embedded if total is None else embedded @ total
        if total is None:
            raise ValueError("The cycle has no operations.")
        return total

    def exact_process(self, max_dimension: int = _MAX_DENSE_DIMENSION) -> qx.SuperOp:
        """Evolve the joint generator, couplings included, into one superoperator on :attr:`qubits`.

        The reference :meth:`operations` approximates. Every constituent must be a Lindbladian-backed
        :class:`~pyquil.noise._channels.Channel`, since the couplings act *during* the gates.
        """
        dims = self._dense_dims(max_dimension)
        qubits = self.qubits
        dimension = int(np.prod(dims, dtype=int))
        hamiltonian = np.zeros((dimension, dimension), dtype=complex)
        jumps: list[np.ndarray] = []
        for channel in self.channels:
            if not isinstance(channel, Channel):
                raise TypeError(
                    f"exact_process needs a generator for every operation, but {channel.inst} is a "
                    f"{type(channel).__name__}."
                )
            positions = [qubits.index(q) for q in channel.qubits]
            # Stretch the generator so that it produces the same process over the cycle duration.
            scale = channel.gate_time / self.duration
            if channel.lindbladian.hamiltonian is not None:
                local = np.asarray(channel.lindbladian.hamiltonian.matrix)
                hamiltonian += scale * _embed_matrix(local, positions, dims)
            for jump in np.asarray(channel.lindbladian.jump_operators.matrix):
                jumps.append(np.sqrt(scale) * _embed_matrix(jump, positions, dims))
        for coupling in self.couplings:
            positions = [qubits.index(q) for q in coupling.qubits]
            hamiltonian += _embed_matrix(np.asarray(coupling.hamiltonian.matrix), positions, dims)
        if not jumps:
            jumps.append(np.zeros((dimension, dimension), dtype=complex))
        lindbladian = qx.Lindbladian(
            hamiltonian=qx.Observable.from_matrix(jnp.asarray(hamiltonian), (dims, dims)),
            jump_operators=qx.Operator.from_matrix(jnp.asarray(np.stack(jumps)), (dims, dims)),
        )
        return cast(qx.SuperOp, qx.evolve(lindbladian, self.duration))

    # ──────────────────────────────────────────────
    # Fidelities
    # ──────────────────────────────────────────────

    @cached_property
    def process_fidelity(self) -> float:
        """Estimated process fidelity of the cycle's gates, crosstalk included.

        The product of the constituents' process fidelities and of each crosstalk unitary's
        fidelity to the identity. Unlike :attr:`CycleChannel.process_fidelity` this is an
        *estimate*: the factors do not act on disjoint qubits, so the product is correct to first
        order in the infidelities. Reset and measurement channels are skipped, as there.
        """
        fidelity = self.cycle.process_fidelity
        for unitary, _ in self._crosstalk_operations:
            matrix = np.asarray(unitary.matrix)
            fidelity *= float(abs(np.trace(matrix)) ** 2) / matrix.shape[0] ** 2
        return fidelity

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
            "couplings": [coupling.to_json() for coupling in self.couplings],
            "duration": self.duration,
            "steps": self.steps,
            "max_fused_qubits": self.max_fused_qubits,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> CrosstalkCycleChannel:
        """Deserialize a channel from a JSON string produced by :meth:`to_json`."""
        data = json.loads(json_str)
        return cls(
            cycle=CycleChannel.from_json(data["cycle"]),
            couplings=tuple(Coupling.from_json(c) for c in data["couplings"]),
            duration=data["duration"],
            steps=data["steps"],
            max_fused_qubits=data["max_fused_qubits"],
        )

    # ──────────────────────────────────────────────
    # Dunder methods
    # ──────────────────────────────────────────────

    def __str__(self) -> str:
        """Return a simplified string representation showing the cycle and its estimated fidelity."""
        return f"<{self.inst.out()} + {len(self.couplings)} coupling(s) ~ ({100 * self.process_fidelity:.2f}%)>"

    def __eq__(self, other: object) -> bool:
        """Check equality by cycle, couplings, duration, and lowering parameters."""
        if not isinstance(other, CrosstalkCycleChannel):
            return False
        return (
            self.cycle == other.cycle
            and self.couplings == other.couplings
            and self.duration == other.duration
            and self.steps == other.steps
            and self.max_fused_qubits == other.max_fused_qubits
        )

    __hash__ = None  # type: ignore[assignment]


__all__ = ["Coupling", "CrosstalkCycleChannel", "CycleOperation"]
