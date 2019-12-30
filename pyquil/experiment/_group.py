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
from typing import Iterable, Sequence, Dict, Tuple, List
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._main import TomographyExperiment
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.operator_estimation import group_experiments
from pyquil import Program


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


def merge_disjoint_experiments(
    experiments: List[TomographyExperiment], group_merged_settings: bool = True
) -> TomographyExperiment:
    """
    Merges the list of experiments into a single experiment that runs the sum of the individual
    experiment programs and contains all of the combined experiment settings.

    A group of TomographyExperiments whose programs operate on disjoint sets of qubits can be
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
    used_qubits = set()
    for expt in experiments:
        if expt.program.get_qubits().intersection(used_qubits):
            raise ValueError(
                "Experiment programs act on some shared set of qubits and cannot be "
                "merged unambiguously."
            )
        used_qubits = used_qubits.union(expt.program.get_qubits())

    # get a flat list of all settings, to be regrouped later
    all_settings = [
        setting for expt in experiments for simult_settings in expt for setting in simult_settings
    ]
    merged_program = sum([expt.program for expt in experiments], Program())
    merged_program.wrap_in_numshots_loop(max([expt.program.num_shots for expt in experiments]))

    symm_levels = [expt.symmetrization for expt in experiments]
    symm_level = max(symm_levels)
    if SymmetrizationLevel.EXHAUSTIVE in symm_levels:
        symm_level = SymmetrizationLevel.EXHAUSTIVE
    merged_expt = TomographyExperiment(all_settings, merged_program, symmetrization=symm_level)

    if group_merged_settings:
        merged_expt = group_experiments(merged_expt)

    return merged_expt
