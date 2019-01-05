import itertools
import json
import logging
import sys
from json import JSONEncoder
from math import pi
from typing import List, Union, Iterable, Dict
from functools import reduce
import networkx as nx
import numpy as np
from networkx.algorithms.approximation.clique import clique_removal

from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.gates import *
from pyquil.paulis import PauliTerm, is_identity

if sys.version_info < (3, 7):
    from pyquil.external.dataclasses import dataclass
else:
    from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentSetting:
    """
    Input and output settings for a tomography-like experiment.

    Many near-term quantum algorithms take the following form:

     - Start in a pauli state
     - Prepare some ansatz
     - Measure it w.r.t. pauli operators

    Where we typically use a large number of (start, measure) pairs but keep the ansatz preparation
    program consistent. This class represents the (start, measure) pairs. Typically a large
    number of these :py:class:`ExperimentSetting` objects will be created and grouped into
    a :py:class:`TomographyExperiment`.
    """
    in_operator: PauliTerm
    out_operator: PauliTerm

    def __str__(self):
        return f'{self.in_operator.compact_str()}→{self.out_operator.compact_str()}'

    def __repr__(self):
        return f'ExperimentSetting[{self}]'

    def serializable(self):
        return str(self)

    @classmethod
    def from_str(cls, s: str):
        """The opposite of str(expt)"""
        instr, outstr = s.split('→')
        return ExperimentSetting(in_operator=PauliTerm.from_compact_str(instr),
                                 out_operator=PauliTerm.from_compact_str(outstr))


def _abbrev_program(program: Program, max_len=10):
    """Create an abbreviated string representation of a Program.

    This will join all instructions onto a single line joined by '; '. If the number of
    instructions exceeds ``max_len``, some will be excluded from the string representation.
    """
    program_lines = program.out().splitlines()
    if max_len is not None and len(program_lines) > max_len:
        first_n = max_len // 2
        last_n = max_len - first_n
        excluded = len(program_lines) - max_len
        program_lines = (program_lines[:first_n] + [f'... {excluded} instrs not shown ...']
                         + program_lines[-last_n:])

    return '; '.join(program_lines)


class TomographyExperiment:
    """
    A tomography-like experiment.

    Many near-term quantum algorithms involve:

     - some limited state preparation
     - enacting a quantum process (like in tomography) or preparing a variational ansatz state
       (like in VQE)
     - measuring observables of the state.

    Where we typically use a large number of (state_prep, measure) pairs but keep the ansatz
    program consistent. This class stores the ansatz program as a :py:class:`~pyquil.Program`
    and maintains a list of :py:class:`ExperimentSetting` objects which each represent a
    (state_prep, measure) pair.

    Settings diagonalized by a shared tensor product basis (TPB) can (optionally) be estimated
    simultaneously. Therefore, this class is backed by a list of list of ExperimentSettings.
    Settings sharing an inner list will be estimated simultaneously. If you don't want this,
    provide a list of length-1-lists. As a convenience, if you pass a 1D list to the constructor
    will expand it to a list of length-1-lists.

    This class will not group settings for you. Please see :py:func:`group_experiments` for
    a function that will automatically process a TomographyExperiment to group Experiments sharing
    a TPB.
    """

    def __init__(self,
                 settings: Union[List[ExperimentSetting], List[List[ExperimentSetting]]],
                 program: Program,
                 qubits: List[int]):
        if len(settings) == 0:
            settings = []
        else:
            if isinstance(settings[0], ExperimentSetting):
                # convenience wrapping in lists of length 1
                settings = [[expt] for expt in settings]

        self._settings = settings  # type: List[List[ExperimentSetting]]
        self.program = program
        self.qubits = qubits

    def __len__(self):
        return len(self._settings)

    def __getitem__(self, item):
        return self._settings[item]

    def __setitem__(self, key, value):
        self._settings[key] = value

    def __delitem__(self, key):
        self._settings.__delitem__(key)

    def __iter__(self):
        yield from self._settings

    def __reversed__(self):
        yield from reversed(self._settings)

    def __contains__(self, item):
        return item in self._settings

    def append(self, expts):
        if not isinstance(expts, list):
            expts = [expts]
        return self._settings.append(expts)

    def count(self, expt):
        return self._settings.count(expt)

    def index(self, expt, start=None, stop=None):
        return self._settings.index(expt, start, stop)

    def extend(self, expts):
        return self._settings.extend(expts)

    def insert(self, index, expt):
        return self._settings.insert(index, expt)

    def pop(self, index=None):
        return self._settings.pop(index)

    def remove(self, expt):
        return self._settings.remove(expt)

    def reverse(self):
        return self._settings.reverse()

    def sort(self, key=None, reverse=False):
        return self._settings.sort(key, reverse)

    def setting_strings(self):
        yield from ('{i}: {st_str}'.format(i=i, st_str=', '.join(str(setting)
                                                                 for setting in settings))
                    for i, settings in enumerate(self._settings))

    def settings_string(self, abbrev_after=None):
        setting_strs = list(self.setting_strings())
        if abbrev_after is not None and len(setting_strs) > abbrev_after:
            first_n = abbrev_after // 2
            last_n = abbrev_after - first_n
            excluded = len(setting_strs) - abbrev_after
            setting_strs = (setting_strs[:first_n] + [f'... {excluded} not shown ...',
                                                      '... use e.settings_string() for all ...']
                            + setting_strs[-last_n:])
        return '\n'.join(setting_strs)

    def __str__(self):
        return _abbrev_program(self.program) + '\n' + self.settings_string(abbrev_after=20)

    def serializable(self):
        return {
            'type': 'TomographyExperiment',
            'settings': self._settings,
            'program': self.program.out(),
            'qubits': self.qubits,
        }

    def __eq__(self, other):
        if not isinstance(other, TomographyExperiment):
            return False
        return self.serializable() == other.serializable()


class OperatorEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, ExperimentSetting):
            return o.serializable()
        if isinstance(o, TomographyExperiment):
            return o.serializable()
        if isinstance(o, ExperimentResult):
            return o.serializable()
        return o


def to_json(fn, obj):
    """Convenience method to save pyquil.operator_estimation objects as a JSON file.

    See :py:func:`read_json`.
    """
    with open(fn, 'w') as f:
        json.dump(obj, f, cls=OperatorEncoder, indent=2)
    return fn


def _operator_object_hook(obj):
    if 'type' in obj and obj['type'] == 'TomographyExperiment':
        return TomographyExperiment([[ExperimentSetting.from_str(s) for s in settings]
                                     for settings in obj['settings']],
                                    program=Program(obj['program']),
                                    qubits=obj['qubits'])
    return obj


def read_json(fn):
    """Convenience method to read pyquil.operator_estimation objects from a JSON file.

    See :py:func:`to_json`.
    """
    with open(fn) as f:
        return json.load(f, object_hook=_operator_object_hook)


def _local_pauli_eig_prep(op: str, idx: int):
    """
    Generate gate sequence to prepare a the +1 eigenstate of a Pauli operator, assuming
    we are starting from the |00..00> ground state.

    :param op: A string representation of the Pauli operator whose +1 eigenstate we'd like to
        prepare.
    :param idx: The index of the qubit that the preparation is acting on
    :return: A program which will prepare the requested state.
    """
    if op == 'X':
        return Program(RY(pi / 2, idx))
    elif op == 'Y':
        return Program(RX(-pi / 2, idx))
    elif op == 'Z':
        return Program()

    raise ValueError(f'Unknown operation {op}')


def _local_pauli_eig_meas(op, idx):
    """
    Generate gate sequence to measure in the eigenbasis of a Pauli operator, assuming
    we are only able to measure in the Z eigenbasis.

    """
    if op == 'X':
        return Program(RY(-pi / 2, idx))
    elif op == 'Y':
        return Program(RX(pi / 2, idx))
    elif op == 'Z':
        return Program()
    raise ValueError(f'Unknown operation {op}')


def _ops_diagonal_in_tpb(op_code1: str, op_code2: str):
    """
    Given two op strings (I, X, Y, or Z) return whether they are diagonal in a tensor product basis.

    I.e. are they the same or is one of them 'I'.
    """
    if op_code1 not in ['X', 'Y', 'Z', 'I']:
        raise ValueError(f"Unknown op_code {op_code1}")
    if op_code2 not in ['X', 'Y', 'Z', 'I']:
        raise ValueError(f"Unknown op_code {op_code2}")

    if op_code1 == op_code2:
        # Same op
        return True
    elif op_code1 == 'I' or op_code2 == 'I':
        # I commutes with everything
        return True
    else:
        # Otherwise, they do not commute.
        return False


def _all_qubits_diagonal_in_tpb(op1: PauliTerm, op2: PauliTerm):
    """
    Compare all qubits between two PauliTerms to see if they are all diagonal in an
    overall shared tensor product basis.
    """
    all_qubits = set(op1.get_qubits()) | set(op2.get_qubits())
    return all(_ops_diagonal_in_tpb(op1[q], op2[q]) for q in all_qubits)


def construct_tpb_graph(experiments: TomographyExperiment):
    """
    Construct a graph where an edge signifies two experiments are diagonal in a TPB.
    """
    g = nx.Graph()
    for expt in experiments:
        assert len(expt) == 1, 'already grouped?'
        expt = expt[0]

        if expt not in g:
            g.add_node(expt, count=1)
        else:
            g.nodes[expt]['count'] += 1

    for expt1, expt2 in itertools.combinations(experiments, r=2):
        expt1 = expt1[0]
        expt2 = expt2[0]

        if expt1 == expt2:
            continue

        if (_all_qubits_diagonal_in_tpb(expt1.in_operator, expt2.in_operator)
                and _all_qubits_diagonal_in_tpb(expt1.out_operator, expt2.out_operator)):
            g.add_edge(expt1, expt2)

    return g


def group_experiments_clique_removal(experiments: TomographyExperiment) -> TomographyExperiment:
    """
    Group experiments that are diagonal in a shared tensor product basis (TPB) to minimize number
    of QPU runs, using a graph clique removal algorithm.

    :param experiments: an tomography experiment
    :return: an tomography experiment with all the same settings, just grouped according to shared
        TPBs.
    """
    g = construct_tpb_graph(experiments)
    _, cliqs = clique_removal(g)
    new_cliqs = []
    for cliq in cliqs:
        new_cliq = []
        for expt in cliq:
            # duplicate `count` times
            new_cliq += [expt] * g.nodes[expt]['count']

        new_cliqs += [new_cliq]

    return TomographyExperiment(new_cliqs, program=experiments.program, qubits=experiments.qubits)


def group_experiments(experiments: TomographyExperiment, method='greedy') -> TomographyExperiment:
    """
    Group experiments that are diagonal in a shared tensor product basis (TPB) to minimize number
    of QPU runs, using a specified method (greedy method by default)

    :param experiments: an tomography experiment
    :param method: method used for grouping
    :return: an tomography experiment with all the same settings, just grouped according to shared
        TPBs.
    """
    allowed_methods = ['greedy', 'clique-removal']
    assert method in allowed_methods, f"'method' should be one of {allowed_methods}."
    if method == 'greedy':
        return group_experiments_greedy(experiments)
    elif method == 'clique-removal':
        return group_experiments_clique_removal(experiments)


@dataclass(frozen=True)
class ExperimentResult:
    """An expectation and standard deviation for the measurement of one experiment setting
    in a tomographic experiment.
    """

    setting: ExperimentSetting
    expectation: Union[float, complex]
    stddev: float

    def __str__(self):
        return f'{self.setting}: {self.expectation} +- {self.stddev}'

    def __repr__(self):
        return f'ExperimentResult[{self}]'

    def serializable(self):
        return {
            'type': 'ExperimentResult',
            'setting': self.setting,
            'expectation': self.expectation,
            'stddev': self.stddev,
        }


def _validate_all_diagonal_in_tpb(ops: Iterable[PauliTerm]) -> Dict[int, str]:
    """Each non-identity qubit should result in the same op_str among all operations. Return
    said mapping.
    """
    mapping = dict()  # type: Dict[int, str]
    for op in ops:
        for idx, op_str in op:
            if idx in mapping:
                assert mapping[idx] == op_str, 'Improper grouping of operators'
            else:
                mapping[idx] = op_str
    return mapping


def measure_observables(qc: QuantumComputer, tomo_experiment: TomographyExperiment, n_shots=1000,
                        progress_callback=None, active_reset=False):
    """
    Measure all the observables in an TomographyExperiment.

    :param qc: A QuantumComputer which can run quantum programs
    :param tomo_experiment: A suite of tomographic observables to measure
    :param n_shots: The number of shots to take per ExperimentSetting
    :param progress_callback: If not None, this function is called each time a group of
        settings is run with arguments ``f(i, len(tomo_experiment)`` such that the progress
        is ``i / len(tomo_experiment)``.
    :param active_reset: Whether to actively reset qubits instead of waiting several
        times the coherence length for qubits to decay to |0> naturally. Setting this
        to True is much faster but there is a ~1% error per qubit in the reset operation.
        Thermal noise from "traditional" reset is not routinely characterized but is of the same
        order.
    """
    for i, settings in enumerate(tomo_experiment):
        # Outer loop over a collection of grouped settings for which we can simultaneously
        # estimate.
        log.info(f"Collecting bitstrings for the {len(settings)} settings: {settings}")

        # 1.1 Prepare a state according to setting.in_operator
        total_prog = Program()
        if active_reset:
            total_prog += RESET()
        in_mapping = _validate_all_diagonal_in_tpb(setting.in_operator for setting in settings)
        for idx, op_str in in_mapping.items():
            total_prog += _local_pauli_eig_prep(op_str, idx)

        # 1.2 Add in the program
        total_prog += tomo_experiment.program

        # 1.3 Measure the state according to setting.out_operator
        out_mapping = _validate_all_diagonal_in_tpb(setting.out_operator for setting in settings)
        for idx, op_str in out_mapping.items():
            total_prog += _local_pauli_eig_meas(op_str, idx)

        # 2. Run the experiment
        bitstrings = qc.run_and_measure(total_prog, n_shots)
        if progress_callback is not None:
            progress_callback(i, len(tomo_experiment))

        # 3. Post-process
        # 3.1 First transform bits to eigenvalues; ie (+1, -1)
        obs_strings = {q: 1 - 2 * bitstrings[q] for q in bitstrings}

        # Inner loop over the grouped settings. They only differ in which qubits' measurements
        # we include in the post-processing. For example, if `settings` is Z1, Z2, Z1Z2 and we
        # measure (n_shots, n_qubits=2) obs_strings then the full operator value involves selecting
        # either the first column, second column, or both and multiplying along the row.
        for setting in settings:
            # 3.2 Special case for measuring the "identity" operator, which doesn't make much
            #     sense but should happen perfectly.
            if is_identity(setting.out_operator):
                yield ExperimentResult(
                    setting=setting,
                    expectation=1.0,
                    stddev=0.0,
                )
                continue

            # 3.3 Get the term's coefficient so we can multiply it in later.
            assert setting.in_operator.coefficient == 1, 'in_operator should specify a state and ' \
                                                         'therefore cannot have a coefficient'
            coeff = complex(setting.out_operator.coefficient)
            if not np.isclose(coeff.imag, 0):
                raise ValueError(f"{setting}'s out_operator has a complex coefficient.")
            coeff = coeff.real

            # 3.4 Pick columns corresponding to qubits with a non-identity out_operation and stack
            #     into an array of shape (n_shots, n_measure_qubits)
            my_obs_strings = np.vstack(obs_strings[q] for q, op_str in setting.out_operator).T

            # 3.6 Multiply row-wise to get operator values. Do statistics. Yield result.
            obs_vals = coeff * np.prod(my_obs_strings, axis=1)
            obs_mean = np.mean(obs_vals)
            obs_var = np.var(obs_vals) / n_shots
            yield ExperimentResult(
                setting=setting,
                expectation=np.asscalar(obs_mean),
                stddev=np.asscalar(np.sqrt(obs_var)),
            )


def diagonal_basis_commutes(pauli_a: PauliTerm, pauli_b: PauliTerm):
    """
    Test if `pauli_a` and `pauli_b` are diagonal in the same tensor product basis.

    Elaboration: Given some PauliTerm, the 'natural' tensor product basis (tpb) to
    diagonalize this term is the one which diagonalizes each Pauli operator in the
    product term-by-term.
    
    For example, X(1) * Z(0) would be diagonal in the 'natural' tensor product basis
    {(|0> +/- |1>)/Sqrt[2]} * {|0>, |1>}, whereas Z(1) * X(0) would be diagonal
    in the 'natural' tpb {|0>, |1>} * {(|0> +/- |1>)/Sqrt[2]}. The two operators
    commute but are not diagonal in each others 'natural' tpb (in fact, they are
    anti-diagonal in each others 'natural' tpb). This function tests whether two
    operators given as PauliTerms are both diagonal in each others 'natural' tpb.

    Note that for the given example of X(1) * Z(0) and Z(1) * X(0), we can construct
    the following basis which simultaneously diagonalizes both operators:

      -- 2 * |0>' = |0> (|0> + |1>) + |1> (|0> - |1>)
      -- 2 * |1>' = |0> (|0> + |1>) - |1> (|0> - |1>)
      -- 2 * |2>' = |0> (|0> - |1>) + |1> (|0> + |1>)
      -- 2 * |3>' = |0> (-|0> + |1>) + |1> (|0> + |1>)

    In this basis, X Z looks like diag(1, -1, 1, -1), and Z X looks like diag(1, 1, -1, -1)

    :param pauli_a: PauliTerm to check diagonality of in the natural tpb of `pauli_b`
    :param pauli_b: PauliTerm to check diagonality of in the natural tpb of `pauli_a`
    :return: Boolean of diagonality in each others natural tpb
    """
    overlapping_active_qubits = set(pauli_a.get_qubits()) & set(pauli_b.get_qubits())
    for qubit_index in overlapping_active_qubits:
        if (pauli_a[qubit_index] != 'I' and pauli_b[qubit_index] != 'I' and
                pauli_a[qubit_index] != pauli_b[qubit_index]):
            return False

    return True


def get_diagonalizing_basis(list_of_pauli_terms: List[PauliTerm]):
    """
    Find the Pauli Term with the most non-identity terms
    :param list_of_pauli_terms: List of Pauli terms to check
    :return: The highest weight Pauli Term
    """
    qubit_ops = set(reduce(lambda x, y: x + y,
                           [list(term._ops.items()) for term in list_of_pauli_terms]))
    qubit_ops = sorted(list(qubit_ops), key=lambda x: x[0])

    return PauliTerm.from_list(list(map(lambda x: tuple(reversed(x)), qubit_ops)))


def _max_key_overlap_term_pair(expt_setting: ExperimentSetting, diagonal_sets: Dict):
    """
    Calculate the max overlap of an ExperimentSetting with keys of diagonal_sets.
    Returns a different key if we find any collisions. If no collisions are found,
    then the ExperimentSetting is added (as a tuple) and the key is updated so it has the
    largest weight.
    :param expt_setting: ExperimentSetting
    :param diagonal_sets: dictionary with
        (key, value): ((diagonal_in_basis, diagonal_out_basis): list of tuples of PauliTerms
                                                    that are diagonal in those bases)
    :return: the updated diagonal_sets dictionary updated with the input expt_setting placed
            appropriately into it
    """
    # construct a tuple out of the in and out operators of the ExperimentSetting
    tup_pauli_terms = tuple(([expt_setting.in_operator], [expt_setting.out_operator]))
    # loop through keys and determine if in_operator is diagonal in the same tpb as in_key,
    # and out_operator the same with out_key
    for key in list(diagonal_sets.keys()):
        # obtain the in and out keys
        in_key = key[0]
        out_key = key[1]
        # construct PauliTerms from in and out keys
        pauli_in_from_key = PauliTerm.from_list([x[::-1] for x in in_key])
        pauli_out_from_key = PauliTerm.from_list([x[::-1] for x in out_key])
        # determine if in_operator is diagonal in the same tpb as the PauliTerm constructed
        # from the in_key; do the same for the out_operator and the out_key
        b_in_commutes = diagonal_basis_commutes(tup_pauli_terms[0][0], pauli_in_from_key)
        b_out_commutes = diagonal_basis_commutes(tup_pauli_terms[1][0], pauli_out_from_key)
        # update the dict value if both the pairs are diagonal in the same tpb
        # (note: the two tpbs need not be the same)
        if b_in_commutes and b_out_commutes:
            # updated the in and out sets comprising the dict's value
            updated_pauli_in_set = diagonal_sets[key][0] + tup_pauli_terms[0]
            updated_pauli_out_set = diagonal_sets[key][1] + tup_pauli_terms[1]
            updated_pauli_set = (updated_pauli_in_set, updated_pauli_out_set)
            # obtain the diagonalizing bases for both the updated in and out sets
            diagonalizing_in_term = get_diagonalizing_basis(updated_pauli_in_set)
            diagonalizing_out_term = get_diagonalizing_basis(updated_pauli_out_set)
            # update the diagonalizing basis (key of dict) if necessary
            if len(diagonalizing_in_term) > len(in_key) or len(diagonalizing_out_term) > len(out_key):
                del diagonal_sets[key]
                new_in_key = tuple(sorted(diagonalizing_in_term._ops.items(), key=lambda x: x[0]))
                new_out_key = tuple(sorted(diagonalizing_out_term._ops.items(), key=lambda x: x[0]))
                new_key = (new_in_key, new_out_key)
                diagonal_sets[new_key] = updated_pauli_set
            else:
                diagonal_sets[key] = updated_pauli_set
            return diagonal_sets

    # made it through all keys and sets so need to make a new set
    # always need to sort because new pauli term functionality
    new_in_key = tuple(sorted(tup_pauli_terms[0][0]._ops.items(), key=lambda x: x[0]))
    new_out_key = tuple(sorted(tup_pauli_terms[1][0]._ops.items(), key=lambda x: x[0]))
    new_key = (new_in_key, new_out_key)
    diagonal_sets[new_key] = tup_pauli_terms
    return diagonal_sets


def commuting_sets_by_zbasis_tomo_expt(tomoexpt: TomographyExperiment):
    """
    Given an input TomographyExperiment, provide a dictionary indicating which ExperimentSettings
    share a tensor product basis

    :param tomoexpt: TomographyExperiment, from which to group ExperimentSettings that share a tpb
        and can be run together
    :return: dict with (key, value): ((diagonal_in_basis, diagonal_out_basis): list of tuples of PauliTerms
                                                    that are diagonal in those bases)
    """
    diagonal_sets = {}
    for expt_setting in tomoexpt:
        assert len(expt_setting) == 1, 'already grouped?'
        expt_setting = expt_setting[0]
        diagonal_sets = _max_key_overlap_term_pair(expt_setting, diagonal_sets)
    return diagonal_sets


def tomo_expt_from_diagonal_sets(diagonal_sets: Dict, tomoexpt: TomographyExperiment):
    """
    Construct a grouped version of TomographyExperiment, given a dictionary
    that contains the grouping information

    :param diagonal_sets: dict containing the diagonal sets of tomoexpt, in the form
        (key, value): ((diagonal_in_basis, diagonal_out_basis): list of tuples of PauliTerms
                                                    that are diagonal in those bases)
    :param tomoexpt: the original experiment suite, ungrouped by diagonal bases
    :return: TomographyExperiment, grouped by ExperimentSettings that are diagonal in the
        same tensor product basis
    """
    l_expt_settings = []
    for v in diagonal_sets.values():
        expt_settings = [ExperimentSetting(v[0][i], v[1][i]) for i in range(len(v[0]))]
        l_expt_settings.append(expt_settings)
    my_tomo_expt = TomographyExperiment(l_expt_settings, program=tomoexpt.program, qubits=tomoexpt.qubits)

    return my_tomo_expt


def group_experiments_greedy(tomo_expt: TomographyExperiment):
    """
    Greedy method to group ExperimentSettings in a given TomographyExperiment

    :param tomo_expt: TomographyExperiment to group ExperimentSettings within
    :return: TomographyExperiment, with grouped ExperimentSettings according to whether
        it consists of PauliTerms diagonal in the same tensor product basis
    """
    diag_sets = commuting_sets_by_zbasis_tomo_expt(tomo_expt)
    grouped_tomo_expt = tomo_expt_from_diagonal_sets(diag_sets, tomo_expt)
    return grouped_tomo_expt
