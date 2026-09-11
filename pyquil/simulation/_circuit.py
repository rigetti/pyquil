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
"""Straight-line circuits over a qudit register, and structural merge planning.

A :class:`Circuit` is an ordered sequence of concrete quax operators — unitaries, channels and
measurement instruments — each placed on a subsystem of a fixed qudit register.  It is not a
program: there are no gate names, no free parameters, no classical memory and no control flow.
The Quil expander in :mod:`pyquil.simulation._resolver` produces one from a program; this module
works on the operators alone.

A :class:`MergePlan` decides which neighbouring operations of a circuit may be fused into a
single larger operator without changing the circuit's action.  Fusing reduces the number of
operator applications a simulator performs, and because a plan depends only on *which* qudits
each operation touches, one plan serves every parameter value of a parametric circuit.

    >>> circuit = Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (0, 1)), (qx.gates.X, (1,))])
    >>> plan = MergePlan.greedy(circuit.subsystems, max_subsystem_size=2)
    >>> plan.apply(circuit).num_ops
    1
"""

# Nothing in this module is specific to Quil, and it is staged to move into quax: ``Circuit`` and
# ``MergePlan`` are the shape proposed for quax's own circuit type.  They live in pyQuil for now so
# the simulator API can settle against real use before the seam is fixed.  One deliberate
# difference: quax carries no graph dependency and would reimplement a small DAG for planning,
# whereas pyQuil already depends on networkx and uses it for the dependency graph, the contracted
# quotient graph, the union-find and the topological sort.

from __future__ import annotations

import heapq
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property, reduce
from operator import mul
from typing import TypeAlias

import jax
import networkx as nx
import quax as qx
from jax import Array
from networkx.utils import UnionFind

#: An operator that may appear in a circuit.  ``Unitary`` covers ``Involution`` (and hence the
#: constant gates); ``SuperOperator`` covers ``SuperOp``, ``KrausMap``, ``Choi`` and
#: ``PauliLiouville``.  ``Lindbladian`` is excluded: it is a generator, not an operation.
CircuitOp: TypeAlias = qx.Unitary | qx.SuperOperator | qx.QuantumInstrument

#: One operation: an operator together with the register indices it acts on, in operand order.
Placement: TypeAlias = tuple[CircuitOp, tuple[int, ...]]

#: A group in a merge plan: the operation indices it fuses, and the subsystem it acts on.
Group: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]


def dependency_edges(subsystems: Sequence[tuple[int, ...]]) -> tuple[tuple[int, int], ...]:
    """Return the dependency edges induced by a sequence of subsystems.

    An edge ``(u, v)`` means operation ``u`` must be applied before operation ``v`` because
    they share a qudit and ``u`` comes first.  Only the immediate predecessor on each qudit is
    recorded; longer chains are implied.  Every edge runs from a lower index to a higher one,
    so application order is always a valid ordering of the graph.

    :param subsystems: One tuple of register indices per operation, in application order.
    :return: Edges as ``(predecessor, successor)`` pairs.
    """
    edges: list[tuple[int, int]] = []
    last_on_qudit: dict[int, int] = {}
    for index, subsystem in enumerate(subsystems):
        for qudit in subsystem:
            previous = last_on_qudit.get(qudit)
            if previous is not None:
                edges.append((previous, index))
            last_on_qudit[qudit] = index
    return tuple(edges)


def dependency_graph(subsystems: Sequence[tuple[int, ...]]) -> nx.DiGraph:
    """Return the dependency DAG of a sequence of subsystems as a :class:`networkx.DiGraph`.

    One node per operation (``0 .. n - 1``), with the edges of :func:`dependency_edges`.

    :param subsystems: One tuple of register indices per operation, in application order.
    :return: The dependency DAG.
    """
    dag: nx.DiGraph = nx.DiGraph()
    dag.add_nodes_from(range(len(subsystems)))
    dag.add_edges_from(dependency_edges(subsystems))
    return dag


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Circuit:
    """An ordered sequence of concrete operators placed on a qudit register.

    A circuit describes one *straight-line block*: the operations are applied in order, with
    no branching.  Anything that requires branching on a measurement outcome is the caller's
    concern, and is naturally expressed as several circuits.

    Operators are stored exactly as given, including their operand order, so a ``CNOT`` placed
    on ``(1, 0)`` keeps its control and target the way the caller wrote them.

    The register's dimensions are an upper bound, not an exact match: an operator may act on a
    subsystem whose register dimension is *larger* than its own, in which case
    :func:`quax.embed` promotes it when needed.  This is what lets a qubit gate sit in a
    qutrit register.

    :param dims: Per-qudit dimensions of the register, e.g. ``(2, 2, 3)``.
    :param ops: The operations, each an ``(operator, subsystem)`` pair, in application order.
    """

    dims: tuple[int, ...]
    ops: tuple[Placement, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", tuple(int(d) for d in self.dims))
        object.__setattr__(self, "ops", tuple((op, tuple(int(q) for q in sub)) for op, sub in self.ops))

        if any(d < 1 for d in self.dims):
            raise ValueError(f"Register dimensions must be positive, got {self.dims}.")

        num_qudits = len(self.dims)
        for index, (op, subsystem) in enumerate(self.ops):
            out_of_range = [q for q in subsystem if not 0 <= q < num_qudits]
            if out_of_range:
                raise ValueError(
                    f"Operation {index} acts on qudit(s) {out_of_range}, outside a register of "
                    f"{num_qudits} qudit(s) with dims={self.dims}."
                )
            if len(set(subsystem)) != len(subsystem):
                raise ValueError(f"Operation {index} names a qudit more than once: {subsystem}.")

            op_dims = op.dims[1]
            if len(op_dims) != len(subsystem):
                raise ValueError(
                    f"Operation {index} acts on {len(op_dims)} qudit(s) but is placed on "
                    f"{len(subsystem)} register position(s) {subsystem}."
                )
            too_large = [(q, d, self.dims[q]) for q, d in zip(subsystem, op_dims, strict=True) if d > self.dims[q]]
            if too_large:
                detail = ", ".join(f"qudit {q}: operator dim {d} > register dim {rd}" for q, d, rd in too_large)
                raise ValueError(f"Operation {index} does not fit the register ({detail}).")

    # ----- pytree -----

    def tree_flatten(self) -> tuple[tuple[CircuitOp, ...], tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]]:
        return self.operators, (self.dims, self.subsystems)

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children: Iterable[CircuitOp]) -> Circuit:
        dims, subsystems = aux_data
        # Bypass ``__post_init__``: unflattening happens with traced (and sometimes
        # placeholder) children, whose ``dims`` are not meaningful to validate.
        circuit = object.__new__(cls)
        object.__setattr__(circuit, "dims", dims)
        object.__setattr__(circuit, "ops", tuple(zip(children, subsystems, strict=True)))
        return circuit

    # ----- display -----

    def __str__(self) -> str:
        return f"Circuit(dims={self.dims}, num_ops={self.num_ops})"

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self) -> Iterator[Placement]:
        return iter(self.ops)

    def __getitem__(self, index: int) -> Placement:
        """Return one operation as an ``(operator, subsystem)`` pair.

        Slicing is not supported: a slice of a circuit would still carry the whole register,
        which is rarely what a caller means.  Build one explicitly with :meth:`with_ops`.
        """
        return self.ops[index]

    # ----- structure -----

    @property
    def num_ops(self) -> int:
        """The number of operations."""
        return len(self.ops)

    @property
    def num_qudits(self) -> int:
        """The number of qudits in the register."""
        return len(self.dims)

    @property
    def dim(self) -> int:
        """The total Hilbert-space dimension of the register."""
        return reduce(mul, self.dims, 1)

    @cached_property
    def subsystems(self) -> tuple[tuple[int, ...], ...]:
        """The register indices each operation acts on, in operand order."""
        return tuple(subsystem for _, subsystem in self.ops)

    @cached_property
    def operators(self) -> tuple[CircuitOp, ...]:
        """The operators, in application order."""
        return tuple(op for op, _ in self.ops)

    # ----- construction -----

    @staticmethod
    def infer_dims(
        ops: Sequence[Placement],
        num_qudits: int | None = None,
        default_dim: int = 2,
    ) -> tuple[int, ...]:
        """Infer register dimensions as the largest dimension each qudit is acted on with.

        ``default_dim`` applies only to qudits that no operation touches: a qudit an operation
        acts on takes its dimension from the operators, so a qubit gate does not silently
        widen its slot.  Pass ``dims`` to :class:`Circuit` directly to size a register the
        operators do not determine — an all-qutrit register holding qubit gates, say.

        :param ops: The operations, as ``(operator, subsystem)`` pairs.
        :param num_qudits: Register size.  Defaults to one past the largest index used.
        :param default_dim: Dimension for a qudit no operation touches.
        :return: Per-qudit dimensions.
        """
        if num_qudits is None:
            num_qudits = 1 + max((q for _, sub in ops for q in sub), default=-1)
        dims = [0] * num_qudits
        for op, subsystem in ops:
            for qudit, d in zip(subsystem, op.dims[1], strict=True):
                dims[qudit] = max(dims[qudit], d)
        return tuple(d or default_dim for d in dims)

    @classmethod
    def from_ops(
        cls,
        ops: Sequence[Placement],
        num_qudits: int | None = None,
        default_dim: int = 2,
    ) -> Circuit:
        """Build a circuit, inferring the register dimensions from the operators.

        :param ops: The operations, as ``(operator, subsystem)`` pairs, in application order.
        :param num_qudits: Register size.  Defaults to one past the largest index used.
        :param default_dim: Dimension for a qudit no operation touches.
        :return: The circuit.
        """
        return cls(dims=cls.infer_dims(ops, num_qudits, default_dim), ops=tuple(ops))

    def with_ops(self, ops: Sequence[Placement]) -> Circuit:
        """Return a circuit with the same register and different operations."""
        return Circuit(dims=self.dims, ops=tuple(ops))

    # ----- representation changes -----

    def to_superops(self) -> Circuit:
        """Convert every operation to a :class:`~quax.SuperOp`.

        Unitaries, Kraus maps, Choi matrices and Pauli-Liouville matrices are converted
        directly.  A :class:`~quax.QuantumInstrument` is replaced by its total channel, which
        discards the outcome labels — the resulting circuit describes the unconditioned
        evolution, which is what density-matrix evolution needs.

        :return: A circuit whose operators are all ``SuperOp``.
        """
        converted: list[Placement] = []
        for op, subsystem in self.ops:
            channel = op.total_channel() if isinstance(op, qx.QuantumInstrument) else op
            converted.append((qx.to_superop(channel), subsystem))
        return self.with_ops(converted)

    def to_kraus_maps(self, atol: float = 1e-6) -> Circuit:
        """Convert channels to truncated :class:`~quax.KrausMap` operators.

        ``SuperOp``, ``Choi`` and ``PauliLiouville`` operations become ``KrausMap`` operations
        with negligible Kraus operators dropped.  ``Unitary``, ``KrausMap`` and
        ``QuantumInstrument`` operations pass through unchanged: each is already applicable to
        a state vector, deterministically or by sampling, so lifting them would only cost
        precision and memory.

        The decomposition diagonalises each channel's Choi matrix and keeps the eigenvectors
        whose eigenvalues exceed ``atol``.  At JAX's default 32-bit precision that threshold
        sits at the resolution of the arithmetic itself, so a warning is issued when a channel
        is decomposed with ``jax_enable_x64`` off; enable 64-bit mode before building or
        converting noise models.

        :param atol: Kraus operators with smaller norm are discarded.
        :return: A circuit with no dense superoperators.
        """
        converted: list[Placement] = []
        decomposed = False
        for op, subsystem in self.ops:
            if isinstance(op, qx.SuperOperator) and not isinstance(op, qx.KrausMap):
                if not decomposed and not jax.config.jax_enable_x64:
                    warnings.warn(
                        "Kraus decomposition at 32-bit precision: JAX's jax_enable_x64 flag is off, so the "
                        f"eigendecomposition and the atol={atol} truncation run at float32 resolution. "
                        "Enable 64-bit mode (jax.config.update('jax_enable_x64', True), or JAX_ENABLE_X64=1) "
                        "before building or converting noise models.",
                        stacklevel=2,
                    )
                decomposed = True
                converted.append((qx.truncate_kraus(qx.to_kraus(op), atol=atol), subsystem))
            else:
                converted.append((op, subsystem))
        return self.with_ops(converted)

    # ----- algebra -----

    def full_operator(self) -> CircuitOp:
        """Fold the whole circuit into a single operator on the full register.

        Each operation is embedded into the register's Hilbert space and the embedded operators
        are multiplied in application order.  The result is a ``Unitary`` when every operation
        is unitary and a superoperator as soon as one is not.

        This is exponentially expensive in the register size and is intended for verification
        and analysis, not for simulation.

        :return: The operator of the whole circuit, acting on all ``num_qudits`` qudits.
        :raises ValueError: If the circuit is empty.
        :raises TypeError: If any operation is a ``QuantumInstrument``, which has no single
            operator — call :meth:`to_superops` first to use the total channel instead.
        """
        if not self.ops:
            raise ValueError("An empty circuit has no operator.")
        instruments = [i for i, (op, _) in enumerate(self.ops) if isinstance(op, qx.QuantumInstrument)]
        if instruments:
            raise TypeError(
                f"Operation(s) {instruments} are QuantumInstruments, which have no single operator. "
                "Call to_superops() first to use their total channels instead."
            )
        return _merge(self.ops, tuple(range(self.num_qudits)), self.dims)


def _merge(
    ops: Sequence[Placement],
    subsystem: tuple[int, ...],
    dims: tuple[int, ...],
) -> CircuitOp:
    """Embed each operation into ``subsystem`` and compose them in application order.

    ``@`` promotes mixed operator types, so an all-unitary group folds to a ``Unitary`` while
    a group containing any channel folds to a superoperator.

    :param ops: The operations to merge, in application order.
    :param subsystem: Register indices of the merged operator, ascending.
    :param dims: Per-qudit dimensions of the whole register.
    :return: The composed operator, acting on ``subsystem``.
    """
    target_dims = tuple(dims[q] for q in subsystem)
    accumulated: CircuitOp | None = None
    for op, op_subsystem in ops:
        positions = tuple(subsystem.index(q) for q in op_subsystem)
        embedded = qx.embed(op, target_dims=target_dims, positions=positions)
        accumulated = embedded if accumulated is None else embedded @ accumulated
    if accumulated is None:
        raise ValueError("Cannot merge an empty operation group.")
    return accumulated


def _contraction_creates_cycle(quotient: nx.DiGraph, a: int, b: int) -> bool:
    """Return ``True`` if contracting ``a`` and ``b`` in the quotient DAG would create a cycle.

    The quotient is always a DAG, so contracting two nodes introduces a cycle iff there is a
    directed path of length two or more between them in *either* direction — some other group
    is sandwiched on a dependency path from one to the other, and merging across it would force
    it to be reordered.  A direct edge alone is fine, so it is set aside before asking networkx
    whether any other path connects the two.
    """
    direct = [(u, v) for u, v in ((a, b), (b, a)) if quotient.has_edge(u, v)]
    quotient.remove_edges_from(direct)
    try:
        return bool(nx.has_path(quotient, a, b) or nx.has_path(quotient, b, a))
    finally:
        quotient.add_edges_from(direct)


@dataclass(frozen=True)
class MergePlan:
    """A structural recipe for fusing a circuit's operations into groups.

    A plan depends only on *which* subsystems the operations act on — never on the operators
    themselves, nor on any parameter value — so it is computed once per circuit structure and
    reused.  :meth:`apply` performs the merge on a concrete circuit; the simulators instead read
    :attr:`groups`, :attr:`bases` and :attr:`op_index` and build their fused operators directly.

    Groups are listed in an order that respects every dependency, and each group lists its
    members in application order.  A group of one operation keeps that operation's own operand
    order; a group of several is recorded on the ascending union of its members' subsystems,
    which is the order :meth:`apply` embeds them into.

    .. warning::
        A group's *position* is not its application order.  Two operations that share no qudit
        commute, so merging can legitimately swap them: a group blocked behind a wide merge may
        be emitted after an independent operation that came later in the original circuit.  Any
        caller that needs to label results per operation — measurement outcome columns, say —
        must use the operation indices in :attr:`groups`, never the group's position.

    :param groups: One ``(operation indices, subsystem)`` pair per group, in application order.
    :param num_ops: The number of operations the plan covers.
    """

    groups: tuple[Group, ...]
    num_ops: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "groups",
            tuple((tuple(int(n) for n in nodes), tuple(int(q) for q in sub)) for nodes, sub in self.groups),
        )
        covered = [node for nodes, _ in self.groups for node in nodes]
        if sorted(covered) != list(range(self.num_ops)):
            raise ValueError(
                f"A merge plan must cover each of the {self.num_ops} operation(s) exactly once; "
                f"got {len(covered)} entries covering {len(set(covered))} distinct operations."
            )

    def __len__(self) -> int:
        return len(self.groups)

    def __str__(self) -> str:
        return f"MergePlan(num_ops={self.num_ops}, num_groups={self.num_groups}, num_bases={len(self.bases)})"

    @property
    def num_groups(self) -> int:
        """The number of groups, i.e. the number of operations after merging."""
        return len(self.groups)

    @cached_property
    def bases(self) -> tuple[tuple[int, ...], ...]:
        """The distinct subsystems the groups act on, in first-seen order.

        A simulator that dispatches each fused operation through a ``jax.lax.switch`` needs one
        branch per base, so the size of its compiled graph scales with the number of bases
        rather than the number of operations.
        """
        seen: dict[tuple[int, ...], int] = {}
        for _, subsystem in self.groups:
            if subsystem not in seen:
                seen[subsystem] = len(seen)
        return tuple(seen)

    @cached_property
    def op_index(self) -> tuple[int, ...]:
        """The index into :attr:`bases` of each group's subsystem, in application order."""
        lookup = {subsystem: i for i, subsystem in enumerate(self.bases)}
        return tuple(lookup[subsystem] for _, subsystem in self.groups)

    @property
    def compression_ratio(self) -> float:
        """Groups per original operation; 1.0 when nothing was merged."""
        return self.num_groups / self.num_ops if self.num_ops else 1.0

    @classmethod
    def trivial(cls, subsystems: Sequence[tuple[int, ...]]) -> MergePlan:
        """Return the plan that merges nothing, keeping every operation as written.

        :param subsystems: One tuple of register indices per operation, in application order.
        :return: A plan with one single-operation group per operation.
        """
        return cls(groups=tuple(((i,), tuple(sub)) for i, sub in enumerate(subsystems)), num_ops=len(subsystems))

    # ``from_partition`` is the seam for further merge strategies.  Each of these is a few lines
    # on the dependency DAG with networkx, and none is implemented until a use case asks for it:
    #   * layer-wise fusion from ``nx.topological_generations`` -- fuse per qudit pair across
    #     consecutive layers, mirroring the cycle structure of ``CycleChannel`` noise models;
    #   * light-cone pruning from ``nx.ancestors`` of the measured operations -- drop operations
    #     that cannot affect a terminal measurement;
    #   * independent registers from ``nx.weakly_connected_components`` -- simulate disconnected
    #     subregisters separately;
    #   * round-based pairwise merging from ``nx.max_weight_matching`` -- a non-greedy
    #     alternative to smallest-union-first.
    # Tensor-network contraction ordering (opt_einsum, cotengra) optimises a different objective,
    # a full contraction rather than a fused stack for a jitted scan, and is not a merge plan.
    @classmethod
    def from_partition(
        cls,
        subsystems: Sequence[tuple[int, ...]],
        groups: Iterable[Iterable[int]],
        *,
        max_subsystem_size: int | None = None,
    ) -> MergePlan:
        """Build a plan from a partition of the operations into groups.

        Any merge strategy reduces to this: decide which operations belong together, and let the
        plan validate and order the result.  A group is accepted only if it is *convex* — no
        operation outside the group lies on a dependency path between two of its members — since
        fusing across such an operation would reorder it.  Groups are emitted in an order that
        respects every dependency, ties broken by each group's earliest member, so operations
        that are not merged keep their relative program order.

        :param subsystems: One tuple of register indices per operation, in application order.
        :param groups: The partition: each operation index appears in exactly one group.
        :param max_subsystem_size: If given, a merged group may span at most this many qudits.
            A group of one operation is never bounded by it.
        :return: The plan.
        :raises ValueError: If ``groups`` is not a partition of the operations, a group is not
            convex, or a merged group exceeds ``max_subsystem_size``.
        """
        subsystems = tuple(tuple(int(q) for q in sub) for sub in subsystems)
        num_ops = len(subsystems)
        blocks = [tuple(sorted(int(n) for n in group)) for group in groups]
        covered = sorted(node for block in blocks for node in block)
        if any(not block for block in blocks) or covered != list(range(num_ops)):
            raise ValueError(
                f"groups must partition the {num_ops} operation(s) into non-empty groups; got "
                f"{len(covered)} entries covering {len(set(covered))} distinct operations."
            )

        qudits = {block: set().union(*(set(subsystems[node]) for node in block)) for block in blocks}
        if max_subsystem_size is not None:
            too_wide = [block for block in blocks if len(block) > 1 and len(qudits[block]) > max_subsystem_size]
            if too_wide:
                raise ValueError(f"Group(s) {too_wide} span more than max_subsystem_size={max_subsystem_size} qudits.")

        quotient = nx.quotient_graph(
            dependency_graph(subsystems),
            [set(block) for block in blocks],
            create_using=nx.DiGraph(),
        )
        if not nx.is_directed_acyclic_graph(quotient):
            involved = sorted({tuple(sorted(block)) for edge in nx.find_cycle(quotient) for block in edge[:2]})
            raise ValueError(
                f"Group(s) {involved} are not convex: an operation in one depends on the other and vice "
                "versa, so merging them would reorder an operation that lies between them."
            )

        plan_groups: list[Group] = []
        for block in nx.lexicographical_topological_sort(quotient, key=min):
            nodes = tuple(sorted(block))
            subsystem = subsystems[nodes[0]] if len(nodes) == 1 else tuple(sorted(qudits[nodes]))
            plan_groups.append((nodes, subsystem))
        return cls(groups=tuple(plan_groups), num_ops=num_ops)

    @classmethod
    def greedy(
        cls,
        subsystems: Sequence[tuple[int, ...]],
        max_subsystem_size: int,
        *,
        atomic: Iterable[int] = (),
    ) -> MergePlan:
        """Plan a merge by greedy edge contraction, smallest candidate union first.

        Merging small operations into larger neighbours reduces the number of distinct
        subsystem shapes, and therefore the size of a simulator's compiled graph, more than it
        reduces the operation count.  So candidates are contracted in ascending order of the
        subsystem size they would produce, which absorbs single-qudit operations into
        multi-qudit neighbours first.

        Two groups are contracted only when the union fits within ``max_subsystem_size`` *and*
        the contraction is convex — no operation that depends on one group and is depended on
        by the other gets reordered.  Convexity is what makes merging safe in the presence of
        non-commuting operations.

        :param subsystems: One tuple of register indices per operation, in application order.
        :param max_subsystem_size: The largest number of qudits a group may span.  ``0``
            disables merging, giving :meth:`trivial`.
        :param atomic: Operations that must never be merged.  Use this for any operation a
            caller needs to interact with individually — a
            :class:`~quax.QuantumInstrument` whose outcome is sampled, for instance, is
            unobservable once fused into a neighbour, even though fusing it would preserve the
            circuit's overall channel.  An atomic operation keeps its place relative to
            everything it depends on, but not relative to independent operations; see the
            warning on :class:`MergePlan`.
        :return: The plan.
        """
        num_ops = len(subsystems)
        if max_subsystem_size <= 0 or num_ops == 0:
            return cls.trivial(subsystems)

        atomic_nodes = frozenset(int(n) for n in atomic)
        out_of_range = sorted(n for n in atomic_nodes if not 0 <= n < num_ops)
        if out_of_range:
            raise ValueError(f"atomic contains operation index(es) {out_of_range} outside 0..{num_ops - 1}.")

        dag = dependency_graph(subsystems)
        union_find = UnionFind(range(num_ops))
        # The quotient graph over current group representatives.  It starts as a copy of the
        # dependency DAG and is contracted whenever two groups merge, so it is the authority on
        # whether a candidate merge is convex.
        quotient: nx.DiGraph = dag.copy()
        group_qudits: dict[int, set[int]] = {i: set(sub) for i, sub in enumerate(subsystems)}

        # Candidate heap keyed by the size of the union a contraction would produce.
        candidates: list[tuple[int, int, int]] = []
        for u, v in dag.edges:
            if u in atomic_nodes or v in atomic_nodes:
                continue
            union_size = len(group_qudits[u] | group_qudits[v])
            if union_size <= max_subsystem_size:
                heapq.heappush(candidates, (union_size, u, v))

        while candidates:
            _, u, v = heapq.heappop(candidates)
            root_u, root_v = union_find[u], union_find[v]
            if root_u == root_v:
                continue
            union_qudits = group_qudits[root_u] | group_qudits[root_v]
            if len(union_qudits) > max_subsystem_size:
                continue
            if _contraction_creates_cycle(quotient, root_u, root_v):
                continue

            union_find.union(root_u, root_v)
            new_root = union_find[root_u]
            dropped = root_v if new_root == root_u else root_u
            group_qudits[new_root] = union_qudits
            del group_qudits[dropped]
            nx.contracted_nodes(quotient, new_root, dropped, self_loops=False, copy=False)

            for neighbour in set(nx.all_neighbors(dag, u)) | set(nx.all_neighbors(dag, v)):
                if neighbour in atomic_nodes:
                    continue
                root_neighbour = union_find[neighbour]
                if root_neighbour == new_root:
                    continue
                union_size = len(group_qudits[new_root] | group_qudits[root_neighbour])
                if union_size <= max_subsystem_size:
                    heapq.heappush(candidates, (union_size, u, neighbour))

        # The contraction loop above is the only strategy-specific part; validation and emission
        # order are shared with every other strategy through ``from_partition``.
        members: dict[int, list[int]] = {}
        for node in range(num_ops):
            members.setdefault(union_find[node], []).append(node)
        return cls.from_partition(subsystems, members.values())

    def apply(self, circuit: Circuit) -> Circuit:
        """Merge a circuit's operations according to this plan.

        A single-operation group passes its operator through untouched.  A multi-operation
        group is embedded into the group's subsystem and composed, so an all-unitary group
        yields a ``Unitary`` and a group containing any channel yields a superoperator.

        :param circuit: The circuit to merge.  Its operation count must match
            :attr:`num_ops`, and its subsystems are expected to be the ones the plan was
            built from.
        :return: The merged circuit, on the same register.
        :raises ValueError: If the circuit's operation count does not match the plan.
        :raises TypeError: If a multi-operation group contains a ``QuantumInstrument``.
        """
        if circuit.num_ops != self.num_ops:
            raise ValueError(
                f"This plan covers {self.num_ops} operation(s) but the circuit has "
                f"{circuit.num_ops}. Rebuild the plan from circuit.subsystems."
            )
        merged: list[Placement] = []
        for nodes, subsystem in self.groups:
            if len(nodes) == 1:
                merged.append(circuit.ops[nodes[0]])
                continue
            group_ops = [circuit.ops[node] for node in nodes]
            instruments = [
                node for node, (op, _) in zip(nodes, group_ops, strict=True) if isinstance(op, qx.QuantumInstrument)
            ]
            if instruments:
                raise TypeError(
                    f"Group {nodes} contains QuantumInstrument operation(s) {instruments}, which "
                    "cannot be merged: fusing an instrument into a neighbour discards the outcome. "
                    "Pass those indices as MergePlan.greedy(..., atomic=...)."
                )
            merged.append((_merge(group_ops, subsystem, circuit.dims), subsystem))
        return circuit.with_ops(merged)


def random_circuit(
    dims: tuple[int, ...],
    num_ops: int,
    key: Array,
    *,
    max_arity: int = 2,
    channel_probability: float = 0.0,
    kraus_rank: int = 2,
) -> Circuit:
    """Generate a random circuit over a register.

    Each operation gets a uniformly random arity up to ``max_arity``, a uniformly random
    subsystem of that arity, and a Haar-random unitary — or, with probability
    ``channel_probability``, a BCSZ-random channel as a ``SuperOp``.

    :param dims: Per-qudit dimensions of the register.
    :param num_ops: The number of operations to generate.
    :param key: A JAX PRNG key.
    :param max_arity: The largest number of qudits one operation may act on.  Clipped to the
        register size.
    :param channel_probability: The probability that an operation is a channel rather than a
        unitary.
    :param kraus_rank: The Kraus rank of generated channels.
    :return: The circuit.
    """
    num_qudits = len(dims)
    if num_qudits == 0:
        raise ValueError("Cannot generate a circuit over an empty register.")
    max_arity = min(max_arity, num_qudits)
    if max_arity < 1:
        raise ValueError(f"max_arity must be at least 1, got {max_arity}.")

    ops: list[Placement] = []
    for _ in range(num_ops):
        key, arity_key, subsystem_key, op_key, kind_key = jax.random.split(key, 5)
        arity = int(jax.random.randint(arity_key, (), 1, max_arity + 1))
        subsystem = tuple(int(q) for q in jax.random.choice(subsystem_key, num_qudits, (arity,), replace=False))
        op_dims = tuple(dims[q] for q in subsystem)
        if channel_probability > 0.0 and float(jax.random.uniform(kind_key)) < channel_probability:
            op: CircuitOp = qx.to_superop(qx.random_choi((op_dims, op_dims), rank=kraus_rank, key=op_key))
        else:
            op = qx.random_unitary((op_dims, op_dims), key=op_key)
        ops.append((op, subsystem))
    return Circuit(dims=tuple(dims), ops=tuple(ops))
