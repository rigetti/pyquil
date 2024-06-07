##############################################################################
# Copyright 2016-2019 Rigetti Computing
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
import functools
import itertools
from collections.abc import Iterable, Sequence
from operator import mul
from typing import Union, cast

import networkx as nx
from networkx.algorithms.approximation.clique import clique_removal

from pyquil.experiment._main import Experiment
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import ExperimentSetting, TensorProductState, _OneQState
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.paulis import PauliTerm, sI
from pyquil.quil import Program


def get_results_by_qubit_groups(
    results: Iterable[ExperimentResult], qubit_groups: Sequence[Sequence[int]]
) -> dict[tuple[int, ...], list[ExperimentResult]]:
    """Organizes ExperimentResults by the group of qubits on which the observable of the result acts.

    Each experiment result will be associated with a qubit group key if the observable of the
    result.setting acts on a subset of the qubits in the group. If the result does not act on a
    subset of qubits of any given group then the result is ignored.

    Note that for groups of qubits which are not pairwise disjoint, one result may be associated to
    multiple groups.

    :param qubit_groups: groups of qubits for which you want the pertinent results.
    :param results: ExperimentResults from running an Experiment
    :return: a dictionary whose keys are individual groups of qubits (as sorted tuples). The
        corresponding value is the list of experiment results whose observables measure some
        subset of that qubit group. The result order is maintained within each group.
    """
    tuple_groups = [tuple(sorted(group)) for group in qubit_groups]
    results_by_qubit_group: dict[tuple[int, ...], list[ExperimentResult]] = {group: [] for group in tuple_groups}
    for res in results:
        res_qs = res.setting.out_operator.get_qubits()

        for group in tuple_groups:
            if set(res_qs).issubset(set(group)):
                results_by_qubit_group[group].append(res)

    return results_by_qubit_group


def merge_disjoint_experiments(experiments: list[Experiment], group_merged_settings: bool = True) -> Experiment:
    """Merge the experiments into one that runs all individual programs and includes all combined settings.

    A group of Experiments whose programs operate on disjoint sets of qubits can be
    'parallelized' so that the total number of runs can be reduced after grouping the settings.
    Settings which act on disjoint sets of qubits can be automatically estimated from the same
    run on the quantum computer.

    If any experiment programs act on a shared qubit they cannot be thoughtlessly composed since
    the order of operations on the shared qubit may have a significant impact on the program
    behaviour; therefore we do not recommend using this method if this is the case.

    Even when the individual experiments act on disjoint sets of qubits you must be
    careful not to associate 'parallel' with 'simultaneous' execution. Physically the gates
    specified in a pyquil Program occur as soon as resources are available; meanwhile, measurement
    happens only after all gates. There is no specification of the exact timing of gates beyond
    their causal relationships. Therefore, while grouping experiments into parallel operation can
    be quite beneficial for time savings, do not depend on any simultaneous execution of gates on
    different qubits, and be wary of the fact that measurement happens only after all gates have
    finished.

    Note that to get the time saving benefits the settings must be grouped on the merged
    experiment--by default this is done before returning the experiment.

    :param experiments: a group of experiments to combine into a single experiment
    :param group_merged_settings: By default group the settings of the merged experiment.
    :return: a single experiment that runs the summed program and all settings.
    """
    used_qubits: set[int] = set()
    for expt in experiments:
        if expt.program.get_qubits().intersection(used_qubits):
            raise ValueError(
                "Experiment programs act on some shared set of qubits and cannot be " "merged unambiguously."
            )
        used_qubits = used_qubits.union(cast(set[int], expt.program.get_qubits()))

    # get a flat list of all settings, to be regrouped later
    all_settings = [setting for expt in experiments for simult_settings in expt for setting in simult_settings]
    merged_program = sum([expt.program for expt in experiments], Program())
    merged_program.wrap_in_numshots_loop(max([expt.program.num_shots for expt in experiments]))

    symm_levels = [expt.symmetrization for expt in experiments]
    symm_level = max(symm_levels)
    if SymmetrizationLevel.EXHAUSTIVE in symm_levels:
        symm_level = SymmetrizationLevel.EXHAUSTIVE
    merged_expt = Experiment(all_settings, merged_program, symmetrization=symm_level)

    if group_merged_settings:
        merged_expt = group_settings(merged_expt)

    return merged_expt


def construct_tpb_graph(experiments: Experiment) -> nx.Graph:
    """Construct a graph where an edge signifies two experiments are diagonal in a TPB."""
    g = nx.Graph()
    for expt in experiments:
        if len(expt) != 1:
            raise ValueError("There must be a single set of ExperimentSettings for each Experiment.")

        unpacked_expt = expt[0]

        if unpacked_expt not in g:
            g.add_node(unpacked_expt, count=1)
        else:
            g.nodes[unpacked_expt]["count"] += 1

    for expt1, expt2 in itertools.combinations(experiments, r=2):
        unpacked_expt1 = expt1[0]
        unpacked_expt2 = expt2[0]

        if unpacked_expt1 == unpacked_expt2:
            continue

        max_weight_in = _max_weight_state([unpacked_expt1.in_state, unpacked_expt2.in_state])
        max_weight_out = _max_weight_operator([unpacked_expt1.out_operator, unpacked_expt2.out_operator])
        if max_weight_in is not None and max_weight_out is not None:
            g.add_edge(unpacked_expt1, unpacked_expt2)

    return g


def group_settings_clique_removal(experiments: Experiment) -> Experiment:
    """Group experiments that are diagonal in a shared tensor product basis (TPB) to minimize number of QPU runs.

    This function uses a graph clique removal algorithm.

    :param experiments: a tomography experiment
    :return: a tomography experiment with all the same settings, just grouped according to shared
        TPBs.
    """
    g = construct_tpb_graph(experiments)
    _, cliqs = clique_removal(g)
    new_cliqs: list[list[ExperimentSetting]] = []
    for cliq in cliqs:
        new_cliq: list[ExperimentSetting] = []
        for expt in cliq:
            # duplicate `count` times
            new_cliq += [expt] * g.nodes[expt]["count"]

        new_cliqs += [new_cliq]

    return Experiment(
        new_cliqs,
        program=experiments.program,
        symmetrization=experiments.symmetrization,
    )


def _max_weight_operator(ops: Iterable[PauliTerm]) -> Union[None, PauliTerm]:
    """Construct a PauliTerm operator by taking the non-identity single-qubit operator at each qubit position.

    This function will return ``None`` if the input operators do not share a natural tensor product basis.

    For example, the max_weight_operator of ["XI", "IZ"] is "XZ". Asking for the max weight
    operator of something like ["XI", "ZI"] will return None.
    """
    mapping: dict[int, str] = dict()
    for op in ops:
        for idx, op_str in op:
            if not isinstance(idx, int):
                raise ValueError(f"Expected index to be int but got {type(idx)}")
            if idx in mapping:
                if mapping[idx] != op_str:
                    return None
            else:
                mapping[idx] = op_str
    op = functools.reduce(mul, (PauliTerm(op, q) for q, op in mapping.items()), sI())
    return op


def _max_weight_state(states: Iterable[TensorProductState]) -> Union[None, TensorProductState]:
    """Construct a TensorProductState by taking the single-qubit state at each qubit position.

    This function will return ``None`` if the input states are not compatible

    For example, the max_weight_state of ["(+X, q0)", "(-Z, q1)"] is "(+X, q0; -Z q1)". Asking for
    the max weight state of something like ["(+X, q0)", "(+Z, q0)"] will return None.
    """
    mapping: dict[int, _OneQState] = dict()
    for state in states:
        for oneq_state in state.states:
            if oneq_state.qubit in mapping:
                if mapping[oneq_state.qubit] != oneq_state:
                    return None
            else:
                mapping[oneq_state.qubit] = oneq_state
    return TensorProductState(list(mapping.values()))


def _max_tpb_overlap(
    tomo_expt: Experiment,
) -> dict[ExperimentSetting, list[ExperimentSetting]]:
    """Given an input Experiment, provide a dictionary indicating which ExperimentSettings share a tensor product basis.

    :param tomo_expt: Experiment, from which to group ExperimentSettings that share a tpb
        and can be run together
    :return: dictionary keyed with ExperimentSetting (specifying a tpb), and with each value being a
            list of ExperimentSettings (diagonal in that tpb)
    """
    # initialize empty dictionary
    diagonal_sets: dict[ExperimentSetting, list[ExperimentSetting]] = {}
    # loop through ExperimentSettings of the Experiment
    for expt_setting in tomo_expt:
        # no need to group already grouped Experiment
        if len(expt_setting) != 1:
            raise ValueError("ExperimentSettings should not be grouped before calling this function.")

        unpacked_expt_setting = expt_setting[0]
        # calculate max overlap of expt_setting with keys of diagonal_sets
        # keep track of whether a shared tpb was found
        found_tpb = False
        # loop through dict items
        for es, es_list in diagonal_sets.items():
            trial_es_list = es_list + [unpacked_expt_setting]
            diag_in_term = _max_weight_state(expst.in_state for expst in trial_es_list)
            diag_out_term = _max_weight_operator(expst.out_operator for expst in trial_es_list)
            # max_weight_xxx returns None if the set of xxx's don't share a TPB, so the following
            # conditional is True if expt_setting can be inserted into the current es_list.
            if diag_in_term is not None and diag_out_term is not None:
                found_tpb = True
                if len(diag_in_term) < len(es.in_state):
                    raise ValueError("Highest weight in-state can't be smaller than the given in-state")
                if len(diag_out_term) < len(es.out_operator):
                    raise ValueError("Highest weight out-PauliTerm can't be smaller than the given out-PauliTerm")

                # update the diagonalizing basis (key of dict) if necessary
                if len(diag_in_term) > len(es.in_state) or len(diag_out_term) > len(es.out_operator):
                    del diagonal_sets[es]
                    new_es = ExperimentSetting(diag_in_term, diag_out_term)
                    diagonal_sets[new_es] = trial_es_list
                else:
                    diagonal_sets[es] = trial_es_list
                break

        if not found_tpb:
            # made it through entire dict without finding any ExperimentSetting with shared tpb,
            # so need to make a new item
            diagonal_sets[unpacked_expt_setting] = [unpacked_expt_setting]

    return diagonal_sets


def group_settings_greedy(tomo_expt: Experiment) -> Experiment:
    """Greedy method to group ExperimentSettings in a given Experiment.

    :param tomo_expt: Experiment to group ExperimentSettings within
    :return: Experiment, with grouped ExperimentSettings according to whether
        it consists of PauliTerms diagonal in the same tensor product basis
    """
    diag_sets = _max_tpb_overlap(tomo_expt)
    grouped_expt_settings_list = list(diag_sets.values())
    grouped_tomo_expt = Experiment(
        grouped_expt_settings_list,
        program=tomo_expt.program,
        symmetrization=tomo_expt.symmetrization,
    )
    return grouped_tomo_expt


def group_settings(experiments: Experiment, method: str = "greedy") -> Experiment:
    """Group experiments that are diagonal in a shared tensor product basis (TPB) to minimize number of QPU runs.

    .. rubric:: Background

    Given some PauliTerm operator, the 'natural' tensor product basis to
    diagonalize this term is the one which diagonalizes each Pauli operator in the
    product term-by-term.

    For example, X(1) * Z(0) would be diagonal in the 'natural' tensor product basis
    ``{(|0> +/- |1>)/Sqrt[2]} * {|0>, |1>}``, whereas Z(1) * X(0) would be diagonal
    in the 'natural' tpb ``{|0>, |1>} * {(|0> +/- |1>)/Sqrt[2]}``. The two operators
    commute but are not diagonal in each others 'natural' tpb (in fact, they are
    anti-diagonal in each others 'natural' tpb). This function tests whether two
    operators given as PauliTerms are both diagonal in each others 'natural' tpb.

    Note that for the given example of X(1) * Z(0) and Z(1) * X(0), we can construct
    the following basis which simultaneously diagonalizes both operators::

      -- |0>' = |0> (|+>) + |1> (|->)
      -- |1>' = |0> (|+>) - |1> (|->)
      -- |2>' = |0> (|->) + |1> (|+>)
      -- |3>' = |0> (-|->) + |1> (|+>)

    In this basis, X Z looks like diag(1, -1, 1, -1), and Z X looks like diag(1, 1, -1, -1).
    Notice however that this basis cannot be constructed with single-qubit operations, as each
    of the basis vectors are entangled states.


    .. rubric:: Methods

    The "greedy" method will keep a running set of 'buckets' into which grouped ExperimentSettings
    will be placed. Each new ExperimentSetting considered is assigned to the first applicable
    bucket and a new bucket is created if there are no applicable buckets.

    The "clique-removal" method maps the term grouping problem onto Max Clique graph problem.
    This method constructs a NetworkX graph where an edge exists between two settings that
    share an nTPB and then uses networkx's algorithm for clique removal. This method can give
    you marginally better groupings in certain circumstances, but constructing the
    graph is pretty slow so "greedy" is the default.

    :param experiments: a tomography experiment
    :param method: method used for grouping; the allowed methods are one of
        ['greedy', 'clique-removal']
    :return: a tomography experiment with all the same settings, just grouped according to shared
        TPBs.
    """
    if method == "greedy":
        return group_settings_greedy(experiments)
    elif method == "clique-removal":
        return group_settings_clique_removal(experiments)
    else:
        allowed_methods = ["greedy", "clique-removal"]
        raise ValueError(f"'method' should be one of {allowed_methods}.")
