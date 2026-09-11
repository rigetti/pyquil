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

"""Tests for straight-line circuits and structural merge planning.

The suite is deliberately split three ways, because the three parts have very different
costs and catch very different bugs:

* **Structural** tests exercise :class:`~pyquil.simulation._circuit.MergePlan` on nothing but subsystem lists.
  No operator is ever built, so they run in microseconds and can sweep hundreds of random
  circuit shapes.  Almost every way the greedy contraction can go wrong — losing an
  operation, merging past a dependency, reordering a barrier — is visible here.
* **Algebraic** tests build operators and check the one property that defines correctness:
  merging must not change the channel the circuit represents.  Comparison is exact
  (superoperator matrices, not fidelities), so no statistical thresholds are involved.
* **Interface** tests cover validation, representation changes and pytree behaviour.
"""

import itertools
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import quax as qx

from pyquil.simulation._circuit import Circuit, MergePlan, dependency_edges, random_circuit

# ---------- helpers ----------


def assert_same_channel(actual, expected, atol: float = 1e-12) -> None:
    """Assert two operators describe the same channel, comparing superoperator matrices.

    Comparing through :func:`quax.to_superop` rather than a fidelity keeps the check exact and
    type-agnostic: a ``Unitary`` and the ``SuperOp`` it lifts to compare equal.  It is also far
    more sensitive than ``process_fidelity``, which loses about eight digits going through the
    Choi matrix and its square root.

    The default tolerance is deliberately tight.  Merging is exact up to floating-point
    associativity, so at the 64-bit precision this suite requires the observed residual is
    around ``1e-16``; anything approaching ``1e-12`` is a real defect, not accumulated error.
    """
    actual_superop = qx.to_superop(actual)
    expected_superop = qx.to_superop(expected)
    assert actual_superop.dims == expected_superop.dims
    difference = float(jnp.max(jnp.abs(actual_superop.matrix - expected_superop.matrix)))
    assert difference < atol, f"channels differ by {difference:g}"


def random_subsystems(
    rng: np.random.Generator,
    num_ops: int,
    num_qudits: int,
    max_arity: int = 2,
) -> list[tuple[int, ...]]:
    """Generate a random list of subsystems — the only input a merge plan needs."""
    subsystems: list[tuple[int, ...]] = []
    for _ in range(num_ops):
        arity = int(rng.integers(1, min(max_arity, num_qudits) + 1))
        subsystems.append(tuple(int(q) for q in rng.choice(num_qudits, size=arity, replace=False)))
    return subsystems


def group_index_of_op(plan: MergePlan) -> dict[int, int]:
    """Map each operation index to the index of the group that emits it."""
    return {node: g for g, (nodes, _) in enumerate(plan.groups) for node in nodes}


def assert_plan_is_well_formed(plan: MergePlan, subsystems, max_subsystem_size=None, atomic=()) -> None:
    """Assert every structural invariant a plan must satisfy, using no operators.

    A plan must partition the operations, keep each group within the size budget, list each
    group's members in application order, hold every dependency in the emitted group order,
    and leave atomic operations unmerged.
    """
    covered = [node for nodes, _ in plan.groups for node in nodes]
    assert sorted(covered) == list(range(len(subsystems))), "plan must cover every operation once"

    group_of = group_index_of_op(plan)
    for nodes, subsystem in plan.groups:
        assert list(nodes) == sorted(nodes), f"group {nodes} members must be in application order"
        if len(nodes) == 1:
            assert subsystem == subsystems[nodes[0]], "a lone operation keeps its own operand order"
        else:
            expected = tuple(sorted({q for node in nodes for q in subsystems[node]}))
            assert subsystem == expected, "a merged group spans the ascending union of its members"
        # The budget bounds the union a *merge* may create.  A single operation is never
        # bounded by it: a two-qudit gate is a legal group of one even when the budget is 1.
        if max_subsystem_size and len(nodes) > 1:
            assert len(subsystem) <= max_subsystem_size, f"group {nodes} exceeds the size budget"

    # Convexity: no dependency may point backwards in the emitted group order.
    for u, v in dependency_edges(subsystems):
        assert group_of[u] <= group_of[v], f"dependency {u} -> {v} is inverted by the plan"

    for node in atomic:
        nodes, _ = plan.groups[group_of[node]]
        assert nodes == (node,), f"atomic operation {node} was merged into {nodes}"

    # bases / op_index must agree with the groups they summarise.
    assert len(plan.op_index) == plan.num_groups
    for g, (_, subsystem) in enumerate(plan.groups):
        assert plan.bases[plan.op_index[g]] == subsystem
    assert len(plan.bases) == len(set(plan.bases))


DIMS = [(2, 2), (2, 2, 2), (2, 3), (2, 2, 3), (3, 3)]


# ══════════════════════════════════════════════════════════
# dependency_edges
# ══════════════════════════════════════════════════════════


class TestDependencyEdges:
    def test_single_qudit_chain(self):
        assert dependency_edges([(0,), (0,), (0,)]) == ((0, 1), (1, 2))

    def test_independent_qudits(self):
        assert dependency_edges([(0,), (1,)]) == ()

    def test_multi_qudit_operation_depends_on_both(self):
        edges = dependency_edges([(0,), (1,), (0, 1)])
        assert set(edges) == {(0, 2), (1, 2)}

    def test_only_the_immediate_predecessor_is_recorded(self):
        # 0 -> 1 -> 2 on the same qudit; the transitive edge 0 -> 2 is implied, not listed.
        assert (0, 2) not in dependency_edges([(0,), (0,), (0,)])

    def test_empty(self):
        assert dependency_edges([]) == ()

    @pytest.mark.parametrize("num_qudits", [1, 2, 5])
    def test_edges_always_point_forwards(self, num_qudits):
        """Application order is always a topological order, which the planner relies on."""
        rng = np.random.default_rng(20260902)
        for _ in range(50):
            subsystems = random_subsystems(rng, 12, num_qudits, max_arity=min(3, num_qudits))
            assert all(u < v for u, v in dependency_edges(subsystems))


# ══════════════════════════════════════════════════════════
# Circuit — construction, validation, interface
# ══════════════════════════════════════════════════════════


class TestCircuitConstruction:
    def test_from_ops_infers_qubit_dims(self):
        circuit = Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (0, 1))])
        assert circuit.dims == (2, 2)
        assert circuit.num_ops == 2
        assert circuit.num_qudits == 2
        assert circuit.dim == 4

    def test_from_ops_infers_mixed_dims(self):
        circuit = Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.MEASURE(dim=3), (1,))])
        assert circuit.dims == (2, 3)

    def test_from_ops_pads_untouched_qudits_with_the_default(self):
        circuit = Circuit.from_ops([(qx.gates.H, (0,))], num_qudits=3, default_dim=3)
        assert circuit.dims == (2, 3, 3)

    def test_infer_dims_takes_the_largest_dimension_per_qudit(self):
        ops = [(qx.gates.H, (0,)), (qx.gates.TRX01(0.3), (0,))]
        assert Circuit.infer_dims(ops) == (3,)

    def test_infer_dims_applies_the_default_only_to_untouched_qudits(self):
        ops = [(qx.gates.H, (0,))]
        assert Circuit.infer_dims(ops, num_qudits=2, default_dim=3) == (2, 3)

    def test_subsystems_and_operators_round_trip(self):
        ops = [(qx.gates.H, (0,)), (qx.gates.CNOT, (1, 0))]
        circuit = Circuit.from_ops(ops)
        assert circuit.subsystems == ((0,), (1, 0))
        assert circuit.operators == (qx.gates.H, qx.gates.CNOT)
        assert list(circuit) == [(op, sub) for op, sub in ops]
        assert len(circuit) == 2
        assert circuit[0] == (qx.gates.H, (0,))
        assert circuit[-1] == (qx.gates.CNOT, (1, 0))

    def test_operand_order_is_preserved(self):
        circuit = Circuit.from_ops([(qx.gates.CNOT, (1, 0))])
        assert circuit.subsystems == ((1, 0),)

    def test_with_ops_keeps_the_register(self):
        circuit = Circuit(dims=(2, 2, 3), ops=((qx.gates.H, (0,)),))
        replaced = circuit.with_ops([(qx.gates.X, (1,))])
        assert replaced.dims == (2, 2, 3)
        assert replaced.subsystems == ((1,),)

    def test_accepts_lists_and_normalises_to_tuples(self):
        # Deliberately passing lists where tuples are declared: construction normalises them,
        # so a hand-written literal need not be punctuated exactly right.
        circuit = Circuit(dims=cast(Any, [2, 2]), ops=cast(Any, [(qx.gates.H, [0])]))
        assert circuit.dims == (2, 2)
        assert circuit.subsystems == ((0,),)

    def test_empty_circuit_is_allowed(self):
        circuit = Circuit(dims=(2,), ops=())
        assert circuit.num_ops == 0
        assert circuit.subsystems == ()

    def test_str(self):
        assert str(Circuit.from_ops([(qx.gates.H, (0,))])) == "Circuit(dims=(2,), num_ops=1)"


class TestCircuitValidation:
    def test_rejects_a_qudit_outside_the_register(self):
        with pytest.raises(ValueError, match="outside a register"):
            Circuit(dims=(2, 2), ops=((qx.gates.H, (5,)),))

    def test_rejects_a_repeated_qudit(self):
        with pytest.raises(ValueError, match="more than once"):
            Circuit(dims=(2, 2), ops=((qx.gates.CNOT, (1, 1)),))

    def test_rejects_an_arity_mismatch(self):
        with pytest.raises(ValueError, match="but is placed on"):
            Circuit(dims=(2, 2), ops=((qx.gates.CNOT, (0,)),))

    def test_rejects_an_operator_larger_than_its_register_slot(self):
        with pytest.raises(ValueError, match="does not fit the register"):
            Circuit(dims=(2, 2), ops=((qx.gates.MEASURE(dim=3), (0,)),))

    def test_rejects_a_non_positive_dimension(self):
        with pytest.raises(ValueError, match="must be positive"):
            Circuit(dims=(2, 0), ops=())

    def test_allows_an_operator_smaller_than_its_register_slot(self):
        """A qubit gate in a qutrit register is legal; embed promotes it."""
        circuit = Circuit(dims=(3, 3), ops=((qx.gates.H, (0,)),))
        composed = circuit.full_operator()
        assert composed.dims == ((3, 3), (3, 3))


class TestCircuitPytree:
    def test_flatten_exposes_operators_as_leaves(self):
        circuit = Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (0, 1))])
        leaves, treedef = jax.tree_util.tree_flatten(circuit)
        # Each operator is itself a pytree of one array.
        assert len(leaves) == 2
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored.dims == circuit.dims
        assert restored.subsystems == circuit.subsystems
        assert_same_channel(restored.full_operator(), circuit.full_operator())

    def test_structure_is_static_and_operators_are_traced(self):
        """A circuit can be closed over by ``jit``, with its operators as traced values."""

        def composed_matrix(circuit):
            return circuit.full_operator().matrix

        circuit = Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (0, 1))])
        eager = composed_matrix(circuit)
        compiled = jax.jit(composed_matrix)(circuit)
        assert jnp.allclose(eager, compiled)

    def test_tree_map_over_operators(self):
        circuit = Circuit.from_ops([(qx.gates.H, (0,))])
        doubled = jax.tree_util.tree_map(lambda x: 2.0 * x, circuit)
        assert jnp.allclose(doubled.operators[0].matrix, 2.0 * qx.gates.H.matrix)


# ══════════════════════════════════════════════════════════
# Circuit — representation changes and composition
# ══════════════════════════════════════════════════════════


class TestToSuperops:
    def test_converts_unitaries(self):
        circuit = Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (0, 1))]).to_superops()
        assert all(isinstance(op, qx.SuperOp) for op in circuit.operators)

    def test_preserves_placement(self):
        circuit = Circuit.from_ops([(qx.gates.CNOT, (1, 0))]).to_superops()
        assert circuit.subsystems == ((1, 0),)

    def test_replaces_an_instrument_with_its_total_channel(self):
        instrument = qx.gates.MEASURE(dim=2)
        circuit = Circuit.from_ops([(instrument, (0,))]).to_superops()
        assert isinstance(circuit.operators[0], qx.SuperOp)
        assert_same_channel(circuit.operators[0], instrument.total_channel())

    def test_is_idempotent(self):
        once = Circuit.from_ops([(qx.gates.H, (0,))]).to_superops()
        assert_same_channel(once.to_superops().full_operator(), once.full_operator())

    def test_preserves_the_whole_channel(self):
        circuit = random_circuit((2, 2), 8, jax.random.key(3), channel_probability=0.5)
        assert_same_channel(circuit.to_superops().full_operator(), circuit.full_operator())


class TestToKrausMaps:
    def test_converts_superops_to_kraus_maps(self):
        circuit = Circuit.from_ops([(qx.gates.H, (0,))]).to_superops().to_kraus_maps()
        assert all(isinstance(op, qx.KrausMap) for op in circuit.operators)

    def test_passes_unitaries_and_instruments_through(self):
        ops = [(qx.gates.H, (0,)), (qx.gates.MEASURE(dim=2), (1,))]
        circuit = Circuit.from_ops(ops).to_kraus_maps()
        assert isinstance(circuit.operators[0], qx.Unitary)
        assert isinstance(circuit.operators[1], qx.QuantumInstrument)

    def test_preserves_the_channel(self):
        circuit = random_circuit((2, 2), 8, jax.random.key(5), channel_probability=1.0)
        assert_same_channel(circuit.to_kraus_maps().full_operator(), circuit.full_operator())

    def test_truncation_drops_negligible_kraus_operators(self):
        unitary_channel = Circuit.from_ops([(qx.gates.H, (0,))]).to_superops()
        truncated = unitary_channel.to_kraus_maps(atol=1e-6).operators[0]
        # A unitary channel has Kraus rank 1, whatever the superoperator's dense shape.
        # A Kraus map's matrix is (*ensemble, num_kraus, d_out, d_in).
        assert truncated.matrix.shape[-3] == 1


class TestCompose:
    def test_matches_a_hand_composed_product(self):
        circuit = Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.X, (0,))])
        expected = qx.gates.X @ qx.gates.H
        assert_same_channel(circuit.full_operator(), expected)

    def test_applies_operations_in_order(self):
        """H then X differs from X then H, so composition order is observable."""
        forwards = Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.X, (0,))]).full_operator()
        backwards = Circuit.from_ops([(qx.gates.X, (0,)), (qx.gates.H, (0,))]).full_operator()
        assert not jnp.allclose(forwards.matrix, backwards.matrix)

    def test_stays_unitary_for_unitary_circuits(self):
        circuit = Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (0, 1))])
        assert isinstance(circuit.full_operator(), qx.Unitary)

    def test_promotes_to_a_superoperator_when_any_operation_is_a_channel(self):
        circuit = Circuit.from_ops([(qx.gates.H, (0,))]).to_superops()
        assert isinstance(circuit.full_operator(), qx.SuperOperator)

    def test_spans_the_whole_register_including_idle_qudits(self):
        circuit = Circuit(dims=(2, 2, 2), ops=((qx.gates.H, (0,)),))
        assert circuit.full_operator().dims == ((2, 2, 2), (2, 2, 2))

    def test_rejects_an_empty_circuit(self):
        with pytest.raises(ValueError, match="empty circuit"):
            Circuit(dims=(2,), ops=()).full_operator()

    def test_rejects_instruments_and_says_what_to_do(self):
        circuit = Circuit.from_ops([(qx.gates.MEASURE(dim=2), (0,))])
        with pytest.raises(TypeError, match="to_superops"):
            circuit.full_operator()


# ══════════════════════════════════════════════════════════
# MergePlan — structural (no operators built)
# ══════════════════════════════════════════════════════════


class TestTrivialPlan:
    def test_keeps_every_operation_separate(self):
        subsystems = [(0,), (0, 1), (1,)]
        plan = MergePlan.trivial(subsystems)
        assert plan.groups == (((0,), (0,)), ((1,), (0, 1)), ((2,), (1,)))
        assert plan.num_groups == 3
        assert plan.compression_ratio == 1.0

    def test_preserves_operand_order(self):
        plan = MergePlan.trivial([(1, 0)])
        assert plan.groups == (((0,), (1, 0)),)

    def test_bases_are_deduplicated_in_first_seen_order(self):
        plan = MergePlan.trivial([(0,), (0, 1), (0,), (1,)])
        assert plan.bases == ((0,), (0, 1), (1,))
        assert plan.op_index == (0, 1, 0, 2)

    def test_empty(self):
        plan = MergePlan.trivial([])
        assert plan.groups == ()
        assert plan.compression_ratio == 1.0


class TestPlanValidation:
    def test_rejects_a_plan_that_misses_an_operation(self):
        with pytest.raises(ValueError, match="exactly once"):
            MergePlan(groups=(((0,), (0,)),), num_ops=2)

    def test_rejects_a_plan_that_covers_an_operation_twice(self):
        with pytest.raises(ValueError, match="exactly once"):
            MergePlan(groups=(((0,), (0,)), ((0,), (0,))), num_ops=1)

    def test_rejects_an_out_of_range_atomic_index(self):
        with pytest.raises(ValueError, match="outside 0..1"):
            MergePlan.greedy([(0,), (0,)], 2, atomic=[7])

    def test_str(self):
        plan = MergePlan.trivial([(0,), (1,)])
        assert str(plan) == "MergePlan(num_ops=2, num_groups=2, num_bases=2)"


class TestFromPartition:
    """``from_partition`` is the constructor every strategy reduces to."""

    def test_reproduces_the_greedy_plan_from_its_partition(self):
        rng = np.random.default_rng(20260910)
        for _ in range(50):
            subsystems = random_subsystems(rng, 20, 5, max_arity=2)
            greedy = MergePlan.greedy(subsystems, 2)
            rebuilt = MergePlan.from_partition(subsystems, [nodes for nodes, _ in greedy.groups])
            assert rebuilt.groups == greedy.groups

    def test_orders_groups_topologically_with_program_order_ties(self):
        # Three independent single-qudit chains, handed over out of order.
        subsystems = [(0,), (1,), (2,), (0,), (1,), (2,)]
        plan = MergePlan.from_partition(subsystems, [[2, 5], [0, 3], [1, 4]])
        assert plan.groups == (((0, 3), (0,)), ((1, 4), (1,)), ((2, 5), (2,)))

    def test_a_dependent_group_is_emitted_after_what_it_depends_on(self):
        subsystems = [(0,), (0, 1), (1,)]
        plan = MergePlan.from_partition(subsystems, [[2], [0, 1]])
        assert plan.groups == (((0, 1), (0, 1)), ((2,), (1,)))

    def test_a_lone_operation_keeps_its_operand_order(self):
        plan = MergePlan.from_partition([(1, 0), (0,)], [[0], [1]])
        assert plan.groups[0] == ((0,), (1, 0))

    def test_rejects_a_non_convex_group(self):
        # 0 and 2 sandwich 1 on qudit 0; merging them would reorder 1.
        with pytest.raises(ValueError, match="not convex"):
            MergePlan.from_partition([(0,), (0,), (0,)], [[0, 2], [1]])

    def test_rejects_a_group_over_the_budget(self):
        with pytest.raises(ValueError, match="max_subsystem_size=2"):
            MergePlan.from_partition([(0,), (1,), (2,)], [[0, 1, 2]], max_subsystem_size=2)

    def test_a_lone_operation_is_never_bounded_by_the_budget(self):
        plan = MergePlan.from_partition([(0, 1, 2)], [[0]], max_subsystem_size=1)
        assert plan.num_groups == 1

    @pytest.mark.parametrize("groups", [[[0], [1]], [[0, 1, 2], [2]], [[0], [], [1, 2]]])
    def test_rejects_a_non_partition(self, groups):
        with pytest.raises(ValueError, match="must partition"):
            MergePlan.from_partition([(0,), (1,), (2,)], groups)

    def test_empty(self):
        assert MergePlan.from_partition([], []).groups == ()

    def test_random_partitions_are_well_formed_or_rejected_for_the_right_reason(self):
        """Any convex partition gives a well-formed plan; a rejection names the non-convex groups."""
        rng = np.random.default_rng(7)
        for _ in range(100):
            subsystems = random_subsystems(rng, 12, 4, max_arity=2)
            labels = rng.integers(0, 5, size=12)
            partition = [list(np.flatnonzero(labels == g)) for g in range(5) if (labels == g).any()]
            try:
                plan = MergePlan.from_partition(subsystems, partition)
            except ValueError as error:
                assert "not convex" in str(error)
                continue
            assert_plan_is_well_formed(plan, subsystems)


class TestGreedyPlanCases:
    def test_a_chain_collapses_to_one_group(self):
        subsystems = [(0,), (0, 1), (1,)]
        plan = MergePlan.greedy(subsystems, max_subsystem_size=2)
        assert plan.groups == (((0, 1, 2), (0, 1)),)
        assert_plan_is_well_formed(plan, subsystems, 2)

    def test_zero_budget_disables_merging(self):
        subsystems = [(0,), (0, 1), (1,)]
        assert MergePlan.greedy(subsystems, 0).groups == MergePlan.trivial(subsystems).groups

    def test_a_budget_of_one_merges_only_within_a_qudit(self):
        subsystems = [(0,), (0,), (0, 1)]
        plan = MergePlan.greedy(subsystems, max_subsystem_size=1)
        assert plan.groups == (((0, 1), (0,)), ((2,), (0, 1)))
        assert_plan_is_well_formed(plan, subsystems, 1)

    def test_independent_operations_are_not_merged(self):
        """With no shared qudit there is no dependency edge, so nothing is a candidate."""
        subsystems = [(0,), (1,), (2,)]
        plan = MergePlan.greedy(subsystems, max_subsystem_size=3)
        assert plan.num_groups == 3

    def test_an_atomic_operation_is_never_merged(self):
        subsystems = [(0,), (0,), (0,)]
        plan = MergePlan.greedy(subsystems, max_subsystem_size=2, atomic=[1])
        assert plan.groups == (((0,), (0,)), ((1,), (0,)), ((2,), (0,)))
        assert_plan_is_well_formed(plan, subsystems, 2, atomic=[1])

    def test_operations_are_not_merged_across_an_atomic_operation(self):
        """The whole point of ``atomic``: a measurement must stay where it is."""
        subsystems = [(0,), (0,), (0,), (0,)]
        plan = MergePlan.greedy(subsystems, max_subsystem_size=2, atomic=[2])
        group_of = group_index_of_op(plan)
        assert group_of[1] < group_of[2] < group_of[3]

    def test_convexity_forbids_swallowing_a_sandwiched_operation(self):
        """Operations 0 and 2 would fit the budget together, but operation 1 is in the way.

        Operation 1 lies on a dependency path between them and cannot be absorbed as well
        (the three-qudit union exceeds the budget), so merging 0 with 2 would reorder it.
        """
        subsystems = [(0, 1), (1, 2), (0, 1)]
        plan = MergePlan.greedy(subsystems, max_subsystem_size=2)
        assert plan.num_groups == 3
        assert_plan_is_well_formed(plan, subsystems, 2)

    def test_a_sandwiched_operation_may_be_merged_when_it_fits(self):
        """With room for all three there is nothing left outside the group to reorder."""
        subsystems = [(0, 1), (1, 2), (0, 1)]
        plan = MergePlan.greedy(subsystems, max_subsystem_size=3)
        assert plan.groups == (((0, 1, 2), (0, 1, 2)),)

    def test_a_whole_diamond_may_collapse(self):
        """Absorbing every branch of a diamond strands nothing, so it is a legal merge."""
        subsystems = [(0, 1), (0,), (1,), (0, 1)]
        plan = MergePlan.greedy(subsystems, max_subsystem_size=2)
        assert plan.groups == (((0, 1, 2, 3), (0, 1)),)
        assert_plan_is_well_formed(plan, subsystems, 2)

    def test_dependent_atomic_operations_stay_in_application_order(self):
        subsystems = [(0,), (0,), (0,), (0,)]
        atomic = [1, 2]
        plan = MergePlan.greedy(subsystems, max_subsystem_size=2, atomic=atomic)
        group_of = group_index_of_op(plan)
        assert group_of[1] < group_of[2]
        assert_plan_is_well_formed(plan, subsystems, 2, atomic=atomic)

    def test_independent_atomic_operations_may_be_reordered(self):
        """Emission position is not application order, and must not be used as an identity.

        Operations 1 and 2 are both atomic and share no qudit.  Operation 0 merges with
        operation 3, and that group must follow operation 2 (it depends on it) and precede
        operation 1 (which depends on it) — so operation 2 is emitted first and operation 1
        last, inverting their application order.  Both are physically free to move, but a
        caller that labelled measurement outcomes by emission position would mislabel them.
        ``groups`` carries each operation's original index for exactly this reason.
        """
        subsystems = [(3, 1), (1, 2), (0,), (0, 3)]
        atomic = [1, 2]
        plan = MergePlan.greedy(subsystems, max_subsystem_size=3, atomic=atomic)
        assert_plan_is_well_formed(plan, subsystems, 3, atomic=atomic)

        group_of = group_index_of_op(plan)
        assert group_of[2] < group_of[1], "this case exists to exhibit the reordering"

        # Application order is always recoverable: an atomic group holds exactly its operation.
        emitted = [nodes[0] for nodes, _ in plan.groups if len(nodes) == 1 and nodes[0] in atomic]
        assert sorted(emitted) == atomic

    def test_small_operations_are_absorbed_into_larger_neighbours(self):
        """Candidates are ordered by resulting size, which keeps the base count down."""
        subsystems = [(0,), (1,), (0, 1), (2,), (1, 2)]
        plan = MergePlan.greedy(subsystems, max_subsystem_size=2)
        assert plan.num_groups < len(subsystems)
        assert len(plan.bases) <= plan.num_groups
        assert_plan_is_well_formed(plan, subsystems, 2)

    def test_is_deterministic(self):
        subsystems = [(0,), (0, 1), (1,), (1, 2), (2,), (0, 2)]
        first = MergePlan.greedy(subsystems, 2)
        second = MergePlan.greedy(subsystems, 2)
        assert first.groups == second.groups


class TestGreedyPlanProperties:
    """Sweep random circuit shapes.  No operator is built, so this is fast and broad."""

    @pytest.mark.parametrize(
        ("num_qudits", "max_subsystem_size"),
        list(itertools.product([1, 2, 3, 5, 8], [0, 1, 2, 3, 4])),
    )
    def test_plans_are_well_formed(self, num_qudits, max_subsystem_size):
        rng = np.random.default_rng(hash((num_qudits, max_subsystem_size)) % 2**32)
        for _ in range(25):
            num_ops = int(rng.integers(0, 25))
            subsystems = random_subsystems(rng, num_ops, num_qudits, max_arity=min(3, num_qudits))
            plan = MergePlan.greedy(subsystems, max_subsystem_size)
            assert_plan_is_well_formed(plan, subsystems, max_subsystem_size)

    @pytest.mark.parametrize("num_qudits", [2, 4, 6])
    def test_plans_with_atomic_operations_are_well_formed(self, num_qudits):
        rng = np.random.default_rng(num_qudits)
        for _ in range(25):
            subsystems = random_subsystems(rng, 20, num_qudits, max_arity=min(2, num_qudits))
            atomic = sorted(int(i) for i in rng.choice(20, size=4, replace=False))
            plan = MergePlan.greedy(subsystems, 3, atomic=atomic)
            assert_plan_is_well_formed(plan, subsystems, 3, atomic=atomic)
            # Atomic operations that depend on one another keep their relative order; ones
            # that do not may be emitted in either order.  See
            # ``test_independent_atomic_operations_may_be_reordered``.
            group_of = group_index_of_op(plan)
            for earlier, later in itertools.combinations(atomic, 2):
                if set(subsystems[earlier]) & set(subsystems[later]):
                    assert group_of[earlier] < group_of[later]

    @pytest.mark.parametrize("num_qudits", [2, 4])
    def test_a_larger_budget_never_produces_more_groups(self, num_qudits):
        rng = np.random.default_rng(7 + num_qudits)
        for _ in range(25):
            subsystems = random_subsystems(rng, 20, num_qudits, max_arity=min(2, num_qudits))
            counts = [MergePlan.greedy(subsystems, k).num_groups for k in range(1, num_qudits + 1)]
            assert counts == sorted(counts, reverse=True)

    def test_a_full_budget_collapses_a_connected_circuit(self):
        subsystems = [(0,), (0, 1), (1,), (1, 2), (2,)]
        plan = MergePlan.greedy(subsystems, max_subsystem_size=3)
        assert plan.num_groups == 1


# ══════════════════════════════════════════════════════════
# MergePlan — algebraic (merging preserves the channel)
# ══════════════════════════════════════════════════════════


class TestMergePreservesTheChannel:
    """The property that defines correctness, checked exactly rather than statistically."""

    @pytest.mark.parametrize("dims", DIMS)
    @pytest.mark.parametrize("max_subsystem_size", [0, 1, 2, 3])
    def test_unitary_circuits(self, dims, max_subsystem_size):
        circuit = random_circuit(dims, 12, jax.random.key(11))
        plan = MergePlan.greedy(circuit.subsystems, max_subsystem_size)
        assert_plan_is_well_formed(plan, circuit.subsystems, max_subsystem_size)
        assert_same_channel(plan.apply(circuit).full_operator(), circuit.full_operator())

    @pytest.mark.parametrize("dims", DIMS)
    @pytest.mark.parametrize("max_subsystem_size", [0, 2, 3])
    def test_noisy_circuits(self, dims, max_subsystem_size):
        circuit = random_circuit(dims, 12, jax.random.key(13), channel_probability=0.5)
        plan = MergePlan.greedy(circuit.subsystems, max_subsystem_size)
        assert_same_channel(plan.apply(circuit).full_operator(), circuit.full_operator())

    @pytest.mark.parametrize("max_subsystem_size", [2, 3])
    def test_channel_only_circuits(self, max_subsystem_size):
        circuit = random_circuit((2, 2, 2), 10, jax.random.key(17), channel_probability=1.0)
        plan = MergePlan.greedy(circuit.subsystems, max_subsystem_size)
        assert_same_channel(plan.apply(circuit).full_operator(), circuit.full_operator())

    @pytest.mark.parametrize("max_subsystem_size", [2, 3])
    def test_mixed_arity_circuits(self, max_subsystem_size):
        circuit = random_circuit((2, 2, 2, 2), 16, jax.random.key(19), max_arity=3)
        plan = MergePlan.greedy(circuit.subsystems, max_subsystem_size)
        assert_same_channel(plan.apply(circuit).full_operator(), circuit.full_operator())

    def test_merging_actually_happens_in_these_cases(self):
        """Guard against the invariant passing because nothing was ever merged."""
        circuit = random_circuit((2, 2), 12, jax.random.key(11))
        plan = MergePlan.greedy(circuit.subsystems, 2)
        assert plan.num_groups < circuit.num_ops
        assert any(len(nodes) > 1 for nodes, _ in plan.groups)

    def test_reversed_operand_order_is_respected_in_a_merge(self):
        """A merge embeds each member by position, so ``CNOT (1, 0)`` must not become (0, 1)."""
        forwards = Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (0, 1))])
        backwards = Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (1, 0))])
        merged_forwards = MergePlan.greedy(forwards.subsystems, 2).apply(forwards)
        merged_backwards = MergePlan.greedy(backwards.subsystems, 2).apply(backwards)
        assert_same_channel(merged_forwards.full_operator(), forwards.full_operator())
        assert_same_channel(merged_backwards.full_operator(), backwards.full_operator())
        assert not jnp.allclose(
            merged_forwards.full_operator().matrix,
            merged_backwards.full_operator().matrix,
        )

    def test_a_merged_group_of_unitaries_stays_unitary(self):
        circuit = Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (0, 1))])
        merged = MergePlan.greedy(circuit.subsystems, 2).apply(circuit)
        assert isinstance(merged.operators[0], qx.Unitary)

    def test_a_merged_group_containing_a_channel_becomes_a_superoperator(self):
        ops = [(qx.gates.H, (0,)), (qx.to_superop(qx.gates.X), (0,))]
        circuit = Circuit.from_ops(ops)
        merged = MergePlan.greedy(circuit.subsystems, 1).apply(circuit)
        assert merged.num_ops == 1
        assert isinstance(merged.operators[0], qx.SuperOperator)


class TestMergeWithInstruments:
    def test_an_atomic_instrument_survives_merging_around_it(self):
        ops = [
            (qx.gates.H, (0,)),
            (qx.gates.MEASURE(dim=2), (0,)),
            (qx.gates.X, (0,)),
        ]
        circuit = Circuit.from_ops(ops)
        plan = MergePlan.greedy(circuit.subsystems, 2, atomic=[1])
        merged = plan.apply(circuit)
        assert merged.num_ops == 3
        assert isinstance(merged.operators[1], qx.QuantumInstrument)
        assert_same_channel(merged.to_superops().full_operator(), circuit.to_superops().full_operator())

    def test_merging_an_instrument_is_refused_with_a_pointer_to_atomic(self):
        circuit = Circuit.from_ops([(qx.gates.MEASURE(dim=2), (0,)), (qx.gates.X, (0,))])
        plan = MergePlan.greedy(circuit.subsystems, 2)
        assert plan.num_groups == 1  # nothing told the planner to keep the instrument apart
        with pytest.raises(TypeError, match="atomic"):
            plan.apply(circuit)

    def test_the_unconditioned_channel_is_preserved_around_an_instrument(self):
        circuit = random_circuit((2, 2), 8, jax.random.key(23))
        with_measurement = circuit.with_ops(
            list(circuit.ops[:4]) + [(qx.gates.MEASURE(dim=2), (0,))] + list(circuit.ops[4:])
        )
        plan = MergePlan.greedy(with_measurement.subsystems, 2, atomic=[4])
        merged = plan.apply(with_measurement)
        assert_same_channel(merged.to_superops().full_operator(), with_measurement.to_superops().full_operator())


class TestApplyValidation:
    def test_rejects_a_circuit_of_the_wrong_length(self):
        circuit = Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.X, (0,))])
        plan = MergePlan.trivial([(0,)])
        with pytest.raises(ValueError, match="covers 1 operation"):
            plan.apply(circuit)

    def test_a_trivial_plan_returns_the_operations_untouched(self):
        circuit = Circuit.from_ops([(qx.gates.H, (0,)), (qx.gates.CNOT, (1, 0))])
        applied = MergePlan.trivial(circuit.subsystems).apply(circuit)
        assert applied.ops == circuit.ops

    def test_result_subsystems_match_the_plan(self):
        circuit = random_circuit((2, 2, 2), 10, jax.random.key(29))
        plan = MergePlan.greedy(circuit.subsystems, 2)
        merged = plan.apply(circuit)
        assert merged.subsystems == tuple(subsystem for _, subsystem in plan.groups)
        assert merged.dims == circuit.dims


# ══════════════════════════════════════════════════════════
# random_circuit
# ══════════════════════════════════════════════════════════


class TestRandomCircuit:
    @pytest.mark.parametrize("dims", DIMS)
    def test_shape_and_placement(self, dims):
        circuit = random_circuit(dims, 10, jax.random.key(0), max_arity=2)
        assert circuit.dims == dims
        assert circuit.num_ops == 10
        assert all(1 <= len(sub) <= 2 for sub in circuit.subsystems)
        assert all(len(set(sub)) == len(sub) for sub in circuit.subsystems)

    def test_is_reproducible(self):
        first = random_circuit((2, 2), 6, jax.random.key(1))
        second = random_circuit((2, 2), 6, jax.random.key(1))
        assert first.subsystems == second.subsystems
        assert jnp.allclose(first.operators[0].matrix, second.operators[0].matrix)

    def test_unitary_by_default(self):
        circuit = random_circuit((2, 2), 8, jax.random.key(2))
        assert all(isinstance(op, qx.Unitary) for op in circuit.operators)

    def test_all_channels_when_asked(self):
        circuit = random_circuit((2, 2), 8, jax.random.key(2), channel_probability=1.0)
        assert all(isinstance(op, qx.SuperOp) for op in circuit.operators)

    def test_arity_is_clipped_to_the_register(self):
        circuit = random_circuit((2,), 4, jax.random.key(4), max_arity=3)
        assert all(sub == (0,) for sub in circuit.subsystems)

    def test_rejects_an_empty_register(self):
        with pytest.raises(ValueError, match="empty register"):
            random_circuit((), 1, jax.random.key(0))
