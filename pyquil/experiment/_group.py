from typing import Iterable, Sequence, Dict, Tuple, List
from pyquil.experiment import ExperimentResult


def get_results_by_qubit_groups(
    results: Iterable[ExperimentResult], qubit_groups: Sequence[Sequence[int]]
) -> Dict[Tuple[int, ...], List[ExperimentResult]]:
    """
    Organizes ExperimentResults by the group of qubits on which the observable of the result acts.

    Each experiment result will be associated with a qubit group key if the observable of the
    result.setting acts on a subset of the qubits in the group. If the result does not act on a
    subset of qubits of any given group then the result is ignored.

    Note that for groups of qubits which are not pairwise disjoint, one result may be associated to
    multiple groups.

    :param qubit_groups: groups of qubits for which you want the pertinent results.
    :param results: ExperimentResults from running an TomographyExperiment
    :return: a dictionary whose keys are individual groups of qubits (as sorted tuples). The
        corresponding value is the list of experiment results whose observables measure some
        subset of that qubit group. The result order is maintained within each group.
    """
    qubit_groups = [tuple(sorted(group)) for group in qubit_groups]
    results_by_qubit_group = {group: [] for group in qubit_groups}
    for res in results:
        res_qs = res.setting.out_operator.get_qubits()

        for group in qubit_groups:
            if set(res_qs).issubset(set(group)):
                results_by_qubit_group[group].append(res)

    return results_by_qubit_group
