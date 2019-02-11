import itertools
import json
import logging
import sys
from json import JSONEncoder
from math import pi
from typing import List, Union, Iterable, Dict, Tuple
import networkx as nx
import numpy as np
from networkx.algorithms.approximation.clique import clique_removal
from functools import reduce
from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.gates import *
from pyquil.paulis import (PauliSum, PauliTerm, commuting_sets, sI,
                           term_with_coeff, is_identity)
from pyquil.quilbase import (Measurement, Pragma, Gate)
from math import pi

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
        json.dump(obj, f, cls=OperatorEncoder, indent=2, ensure_ascii=False)
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
    we are only able to measure in the Z eigenbasis. (Note: The unitary operations of this
    Program are essentially the Hermitian conjugates of those in :py:func:`_local_pauli_eig_prep`)

    """
    if op == 'X':
        return Program(RY(-pi / 2, idx))
    elif op == 'Y':
        return Program(RX(pi / 2, idx))
    elif op == 'Z':
        return Program()
    raise ValueError(f'Unknown operation {op}')


def _ops_commute(op_code1: str, op_code2: str):
    """
    Given two 1q op strings (I, X, Y, or Z), determine whether they commute

    :param op_code1: First operation
    :param op_code2: Second operation
    :return: Boolean specifying whether the two operations commute
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
    overall shared tensor product basis. More concretely, test if ``op1`` and
    ``op2`` are diagonal in each others' "natural" tensor product basis.

    Given some PauliTerm, the 'natural' tensor product basis (tpb) to
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

      -- |0>' = |0> (|+>) + |1> (|->)
      -- |1>' = |0> (|+>) - |1> (|->)
      -- |2>' = |0> (|->) + |1> (|+>)
      -- |3>' = |0> (-|->) + |1> (|+>)

    In this basis, X Z looks like diag(1, -1, 1, -1), and Z X looks like diag(1, 1, -1, -1).
    Notice however that this basis cannot be constructed with single-qubit operations, as each
    of the basis vectors are entangled states.

    :param op1: PauliTerm to check diagonality of in the natural tpb of ``op2``
    :param op2: PauliTerm to check diagonality of in the natural tpb of ``op1``
    :return: Boolean of diagonality in each others natural tpb
    """
    all_qubits = set(op1.get_qubits()) & set(op2.get_qubits())
    return all(_ops_commute(op1[q], op2[q]) for q in all_qubits)


def _expt_settings_diagonal_in_tpb(es1: ExperimentSetting, es2: ExperimentSetting):
    """
    Extends the concept of being diagonal in the same tpb (see :py:func:_all_qubits_diagonal_in_tpb)
    to ExperimentSettings, by determining if the pairs of in_operators and out_operators are
    separately diagonal in the same tpb

    :param es1: ExperimentSetting to check diagonality of in the natural tpb of ``es2``
    :param es2: ExperimentSetting to check diagonality of in the natural tpb of ``es1``
    :return: Boolean of diagonality in each others natural tpb
    """
    in_dtpb = _all_qubits_diagonal_in_tpb(es1.in_operator, es2.in_operator)
    out_dtpb = _all_qubits_diagonal_in_tpb(es1.out_operator, es2.out_operator)
    return in_dtpb and out_dtpb


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

        if _expt_settings_diagonal_in_tpb(expt1, expt2):
            g.add_edge(expt1, expt2)

    return g


def group_experiments_clique_removal(experiments: TomographyExperiment) -> TomographyExperiment:
    """
    Group experiments that are diagonal in a shared tensor product basis (TPB) to minimize number
    of QPU runs, using a graph clique removal algorithm.

    :param experiments: a tomography experiment
    :return: a tomography experiment with all the same settings, just grouped according to shared
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


def _get_diagonalizing_basis(ops: Iterable[PauliTerm]) -> PauliTerm:
    """
    Find the Pauli Term with the most non-identity terms

    :param ops: Iterable of PauliTerms to check
    :return: The highest weight PauliTerm from the input iterable
    """
    # obtain qubit: operation mapping
    dict_pt = _validate_all_diagonal_in_tpb(ops)
    # convert this mapping to PauliTerm
    pt = sI()
    for q, op in dict_pt.items():
        pt *= PauliTerm(op, q)
    return pt


def _max_tpb_overlap(tomo_expt: TomographyExperiment):
    """
    Given an input TomographyExperiment, provide a dictionary indicating which ExperimentSettings
    share a tensor product basis

    :param tomo_expt: TomographyExperiment, from which to group ExperimentSettings that share a tpb
        and can be run together
    :return: dictionary keyed with ExperimentSetting (specifying a tpb), and with each value being a
            list of ExperimentSettings (diagonal in that tpb)
    """
    # initialize empty dictionary
    diagonal_sets = {}
    # loop through ExperimentSettings of the TomographyExperiment
    for expt_setting in tomo_expt:
        # no need to group already grouped TomographyExperiment
        assert len(expt_setting) == 1, 'already grouped?'
        expt_setting = expt_setting[0]
        # calculate max overlap of expt_setting with keys of diagonal_sets
        # keep track of whether a shared tpb was found
        found_tpb = False
        # loop through dict items
        for es, es_list in diagonal_sets.items():
            # update the dict value if es is diagonal in the same tpb as expt_setting
            if _expt_settings_diagonal_in_tpb(es, expt_setting):
                # shared tpb was found
                found_tpb = True
                # determine the updated list of ExperimentSettings
                updated_es_list = es_list + [expt_setting]
                # obtain the diagonalizing bases for both the updated in and out sets
                diag_in_term = _get_diagonalizing_basis([expst.in_operator for expst in updated_es_list])
                diag_out_term = _get_diagonalizing_basis([expst.out_operator for expst in updated_es_list])
                assert len(diag_in_term) >= len(es.in_operator), \
                    "Highest weight in-PauliTerm can't be smaller than the given in-PauliTerm"
                assert len(diag_out_term) >= len(es.out_operator), \
                    "Highest weight out-PauliTerm can't be smaller than the given out-PauliTerm"
                # update the diagonalizing basis (key of dict) if necessary
                if len(diag_in_term) > len(es.in_operator) or len(diag_out_term) > len(es.out_operator):
                    del diagonal_sets[es]
                    new_es = ExperimentSetting(diag_in_term, diag_out_term)
                    diagonal_sets[new_es] = updated_es_list
                else:
                    diagonal_sets[es] = updated_es_list
                break

        if not found_tpb:
            # made it through entire dict without finding any ExperimentSetting with shared tpb,
            # so need to make a new item
            diagonal_sets[expt_setting] = [expt_setting]

    return diagonal_sets


def group_experiments_greedy(tomo_expt: TomographyExperiment):
    """
    Greedy method to group ExperimentSettings in a given TomographyExperiment

    :param tomo_expt: TomographyExperiment to group ExperimentSettings within
    :return: TomographyExperiment, with grouped ExperimentSettings according to whether
        it consists of PauliTerms diagonal in the same tensor product basis
    """
    diag_sets = _max_tpb_overlap(tomo_expt)
    grouped_expt_settings_list = list(diag_sets.values())
    grouped_tomo_expt = TomographyExperiment(grouped_expt_settings_list, program=tomo_expt.program,
                                             qubits=tomo_expt.qubits)
    return grouped_tomo_expt


def group_experiments(experiments: TomographyExperiment, method: str = 'greedy') -> TomographyExperiment:
    """
    Group experiments that are diagonal in a shared tensor product basis (TPB) to minimize number
    of QPU runs, using a specified method (greedy method by default)

    :param experiments: a tomography experiment
    :param method: method used for grouping; the allowed methods are one of
        ['greedy', 'clique-removal']
    :return: a tomography experiment with all the same settings, just grouped according to shared
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
    total_counts: int

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
            'total_counts': self.total_counts,
        }


def measure_observables(qc: QuantumComputer, tomo_experiment: TomographyExperiment, n_shots=1000,
                        progress_callback=None, active_reset=False):
    """
    Measure all the observables in a TomographyExperiment.

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
            # 3.2 Get the term's coefficient so we can multiply it in later.
            assert setting.in_operator.coefficient == 1, 'in_operator should specify a state and ' \
                                                         'therefore cannot have a coefficient'
            coeff = complex(setting.out_operator.coefficient)
            if not np.isclose(coeff.imag, 0):
                raise ValueError(f"{setting}'s out_operator has a complex coefficient.")
            coeff = coeff.real

            # 3.3 Special case for measuring the "identity" operator, which doesn't make much
            #     sense but should happen perfectly.
            if is_identity(setting.out_operator):
                yield ExperimentResult(
                    setting=setting,
                    expectation=coeff,
                    stddev=0.0,
                    total_counts=n_shots,
                )
                continue

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
                total_counts=n_shots,
            )


class DiagonalNTPBError(ValueError):
    """
    Raised error when two terms are not diagonal in each others'
    natural tensor product basis (NTPB)
    """
    pass


def remove_identity(psum: PauliSum) -> Tuple[PauliSum]:
    """
    Remove the identity term from a Pauli sum

    :param PauliSum psum: PauliSum object to remove identity
    :return: The new PauliSum and the identity term.
    """
    new_psum = []
    identity_terms = []
    for term in psum:
        if not is_identity(term):
            new_psum.append(term)
        else:
            identity_terms.append(term)
    return sum(new_psum), sum(identity_terms)


def remove_imaginary(pauli_sum: PauliSum) -> PauliSum:
    """
    Remove the imaginary component of each term in a Pauli sum

    :param pauli_sum: The Pauli sum to process.
    :return: a purely Hermitian PauliSum
    """
    if not isinstance(pauli_sum, PauliSum):
        raise TypeError("not a pauli sum. please give me one")
    new_term = sI(0) * 0.0
    for term in pauli_sum:
        new_term += term_with_coeff(term, term.coefficient.real)

    return new_term


def get_rotation_program_measure(pauli_term: PauliTerm) -> Program:
    """
    Generate a rotation program that diagonalizes a PauliTerm in order
    to measure it out. The unitary operations in this Program correspond to the Hermitian conjugate
    of change of basis transformation from the computational basis to the "natural"
    tensor product basis of the given `pauli_term`

    :param pauli_term: The Pauli term used to generate diagonalizing
                                 one-qubit rotations.
    :return: The rotation Program.
    """
    meas_basis_change = Program()
    for index, gate in pauli_term:
        meas_basis_change.inst(_local_pauli_eig_meas(gate, index))

    return meas_basis_change


def get_parity(pauli_terms, bitstring_results):
    r"""
    Calculate the eigenvalues of Pauli operators given results of projective measurements

    The single-qubit projective measurement results (elements of
    `bitstring_results`) are listed in physical-qubit-label numerical order.

    An example:

    Consider a Pauli term Z1 Z5 Z6 I7 and a collection of single-qubit
    measurement results corresponding to measurements in the z-basis on qubits
    {1, 5, 6, 7}. Each element of bitstring_results is an element of
    :math:`\{0, 1\}^{\otimes 4}`.  If [0, 0, 1, 0] and [1, 0, 1, 1]
    are the two projective measurement results in `bitstring_results` then
    this method returns a 1 x 2 numpy array with values [[-1, 1]]

    :param List pauli_terms: A list of Pauli terms operators to use
    :param bitstring_results: A list of projective measurement results.  Each
                              element is a list of single-qubit measurements.
    :return: Array (m x n) of {+1, -1} eigenvalues for the m-operators in
             `pauli_terms` associated with the n measurement results.
    :rtype: np.ndarray
    """
    qubit_set = []
    for term in pauli_terms:
        qubit_set.extend(list(term.get_qubits()))
    active_qubit_indices = sorted(list(set(qubit_set)))
    index_mapper = dict(zip(active_qubit_indices,
                            range(len(active_qubit_indices))))

    results = np.zeros((len(pauli_terms), len(bitstring_results)))

    # convert to array so we can fancy index into it later.
    # list() is needed to cast because we don't really want a map object
    bitstring_results = list(map(np.array, bitstring_results))
    for row_idx, term in enumerate(pauli_terms):
        memory_index = np.array(list(map(lambda x: index_mapper[x],
                                         sorted(term.get_qubits()))))

        results[row_idx, :] = [-2 * (sum(x[memory_index]) % 2)
                               + 1 for x in bitstring_results]
    return results


@dataclass(frozen=True)
class EstimationResult:
    """
    A dataclass describing Monte-Carlo averaging results.

    :param expected_value: expected value of the PauliSum
    :param pauli_expectations:  individual expected values of elements in the
                                PauliSum.  These values are scaled by the
                                coefficients associated with each PauliSum.
    :param covariance: The covariance matrix computed from the shot data.
    :param variance: Sample variance computed from the covariance matrix and
                     number of shots taken to obtain the data.
    :param n_shots: Number of readouts collected.
    """

    expected_value: float
    pauli_expectations: List[float]
    covariance: np.array
    variance: float
    n_shots: int

    def __str__(self):
        return f"Expected value: {self.expected_value}, variance: {self.variance}, shots: {self.n_shots}"

    def __repr__(self):
        return f"EstimationResult[{self}]"

    def serializable(self):
        return {
            'type': 'EstimationResult',
            'expected_value': self.expected_value,
            'pauli_expectations': self.pauli_expectations,
            'covariance': self.covariance,
            'variance': self.variance,
            'n_shots': self.n_shots
        }


def estimate_pauli_sum(pauli_terms,
                       basis_transform_dict,
                       program,
                       variance_bound,
                       quantum_resource,
                       diagonal_ntpb_check=True,
                       symmetrize=True,
                       rand_samples=16):
    r"""
    Estimate the mean of a sum of pauli terms to set variance

    The sample variance is calculated by

    .. math::
        \begin{align}
        \mathrm{Var}[\hat{\langle H \rangle}] = \sum_{i, j}h_{i}h_{j}
        \mathrm{Cov}(\hat{\langle P_{i} \rangle}, \hat{\langle P_{j} \rangle})
        \end{align}

    The expectation value of each Pauli operator (term and coefficient) is
    also returned.  It can be accessed through the named-tuple field
    `pauli_expectations'.

    :param pauli_terms: list of pauli terms to measure simultaneously or a
                        PauliSum object
    :param basis_transform_dict: basis transform dictionary where the key is
                                 the qubit index and the value is the basis to
                                 rotate into. Valid basis is [I, X, Y, Z].
    :param program: program generating a state to sample from.  The program
                    is deep copied to ensure no mutation of gates or program
                    is perceived by the user.
    :param variance_bound:  Bound on the variance of the estimator for the
                            PauliSum. Remember this is the SQUARE of the
                            standard error!
    :param quantum_resource: quantum abstract machine object
    :param Bool diagonal_ntpb_check: Optional flag toggling a safety check
                                   ensuring all terms in `pauli_terms`
                                   commute with each other
    :param Bool symmetrize: Optional flag toggling symmetrization of readout
    :param Int rand_samples: number of random realizations for readout symmetrization
    :return: estimated expected value, expected value of each Pauli term in
             the sum, covariance matrix, variance of the estimator, and the
             number of shots taken.  The objected returned is a named tuple with
             field names as follows: expected_value, pauli_expectations,
             covariance, variance, n_shots.
             `expected_value' == coef_vec.dot(pauli_expectations)
    :rtype: EstimationResult
    """
    if not isinstance(pauli_terms, (list, PauliSum)):
        raise TypeError("pauli_terms needs to be a list or a PauliSum")

    if isinstance(pauli_terms, PauliSum):
        pauli_terms = pauli_terms.terms

    # check if each term commutes with everything
    if diagonal_ntpb_check:
        try:
            _validate_all_diagonal_in_tpb(pauli_terms)
        except AssertionError:
            raise DiagonalNTPBError("Not all terms are diagonal in each others'"
                                    + "natural tensor product basis, as expected")

    program = program.copy()
    pauli_for_rotations = PauliTerm.from_list(
        [(value, key) for key, value in basis_transform_dict.items()])

    program += get_rotation_program_measure(pauli_for_rotations)

    qubits = sorted(list(basis_transform_dict.keys()))
    if symmetrize:
        theta = program.declare("ro_symmetrize", "REAL", len(qubits))
        for (idx, q) in enumerate(qubits):
            program += RX(theta[idx], q)

    ro = program.declare("ro", "BIT", memory_size=len(qubits))
    for num, qubit in enumerate(qubits):
        program.inst(MEASURE(qubit, ro[num]))

    coeff_vec = np.array(
        list(map(lambda x: x.coefficient, pauli_terms))).reshape((-1, 1))

    # upper bound on samples given by IV of arXiv:1801.03524
    num_sample_ubound = 10 * int(np.ceil(np.sum(np.abs(coeff_vec)) ** 2 / variance_bound))
    if num_sample_ubound <= 2:
        raise ValueError("Something happened with our calculation of the max sample")

    standard_numshots = 10000
    if symmetrize:
        if min(standard_numshots, num_sample_ubound) // rand_samples == 0:
            raise ValueError(f"The number of shots must be larger than {rand_samples}.")

        program = program.wrap_in_numshots_loop(min(standard_numshots, num_sample_ubound) // rand_samples)
    else:
        program = program.wrap_in_numshots_loop(min(standard_numshots, num_sample_ubound))

    compiled_prog = quantum_resource.compiler.quil_to_native_quil(program)
    binary = quantum_resource.compiler.native_quil_to_executable(compiled_prog)

    results = None
    sample_variance = np.infty
    number_of_samples = 0
    tresults = np.zeros((0, len(qubits)))
    while (sample_variance > variance_bound and number_of_samples < num_sample_ubound):
        if symmetrize:
            # for some number of times sample random bit string
            for r in range(rand_samples):
                rand_flips = np.random.randint(low=0, high=2, size=len(qubits))
                temp_results = quantum_resource.run(binary, memory_map={'ro_symmetrize': np.pi * rand_flips})
                tresults = np.vstack((tresults, rand_flips ^ temp_results))
        else:
            tresults = quantum_resource.run(binary)

        number_of_samples += len(tresults)
        parity_results = get_parity(pauli_terms, tresults)

        # Note: easy improvement would be to update mean and variance on the fly
        # instead of storing all these results.
        if results is None:
            results = parity_results
        else:
            results = np.hstack((results, parity_results))

        # calculate the expected values....
        covariance_mat = np.cov(results, ddof=1)
        sample_variance = coeff_vec.T.dot(covariance_mat).dot(coeff_vec) / (results.shape[1] - 1)

    return EstimationResult(expected_value=coeff_vec.T.dot(np.mean(results, axis=1)),
                            pauli_expectations=np.multiply(coeff_vec.flatten(), np.mean(results, axis=1).flatten()),
                            covariance=covariance_mat,
                            variance=sample_variance,
                            n_shots=results.shape[1])


#########
#
# API
#
#########
def estimate_locally_commuting_operator(program,
                                        pauli_sum,
                                        variance_bound,
                                        quantum_resource,
                                        symmetrize=True):
    """
    Estimate the expected value of a Pauli sum to fixed precision.

    :param program: state preparation program
    :param pauli_sum: pauli sum of operators to estimate expected value
    :param variance_bound: variance bound on the estimator
    :param quantum_resource: quantum abstract machine object
    :param symmetrize: flag that determines whether readout is symmetrized or not
    :return: expected value, estimator variance, total number of experiments
    """
    pauli_sum, identity_term = remove_identity(pauli_sum)

    expected_value = 0
    if isinstance(identity_term, int):
        if np.isclose(identity_term, 0):
            expected_value = 0
        else:
            expected_value = identity_term

    elif isinstance(identity_term, (PauliTerm, PauliSum)):
        if isinstance(identity_term, PauliTerm):
            expected_value = identity_term.coefficient
        else:
            expected_value = identity_term[0].coefficient
    else:
        raise TypeError("identity_term must be a PauliTerm or integer. We got {}".format(type(identity_term)))

    # check if pauli_sum didn't get killed...or we gave an identity term
    if isinstance(pauli_sum, int):
        # we have no estimation work to do...just return the identity value
        return expected_value, 0, 0

    psets = group_terms_greedy(pauli_sum)
    variance_bound_per_set = variance_bound / len(psets)

    total_shots = 0
    estimator_variance = 0
    for qubit_op_key, pset in psets.items():
        results = estimate_pauli_sum(pset,
                                     dict(qubit_op_key),
                                     program,
                                     variance_bound_per_set,
                                     quantum_resource,
                                     diagonal_ntpb_check=False,
                                     symmetrize=symmetrize)
        assert results.variance < variance_bound_per_set

        expected_value += results.expected_value
        total_shots += results.n_shots
        estimator_variance += results.variance

    return expected_value, estimator_variance, total_shots


def estimate_general_psum(program, pauli_sum, variance_bound, quantum_resource,
                          sequential=False):
    """
    Estimate the expected value of a Pauli sum to fixed precision.

    :param program: state preparation program
    :param pauli_sum: pauli sum of operators to estimate expected value
    :param variance_bound: variance bound on the estimator
    :param quantum_resource: quantum abstract machine object
    :return: expected value, estimator variance, total number of experiments
    """
    if sequential:
        expected_value = 0
        estimator_variance = 0
        total_shots = 0
        variance_bound_per_term = variance_bound / len(pauli_sum)
        for term in pauli_sum:
            exp_v, exp_var, exp_shots = estimate_locally_commuting_operator(
                program, PauliSum([term]), variance_bound_per_term, quantum_resource)
            expected_value += exp_v
            estimator_variance += exp_var
            total_shots += exp_shots
        return expected_value, estimator_variance, total_shots
    else:
        return estimate_locally_commuting_operator(program, pauli_sum,
                                                   variance_bound,
                                                   quantum_resource)


def estimate_general_psum_symmeterized(program, pauli_sum, variance_bound,
                                       quantum_resource, confusion_mat_dict=None,
                                       sequential=False):
    """
    Estimate the expected value of a Pauli sum to fixed precision.

    :param program: state preparation program
    :param pauli_sum: pauli sum of operators to estimate expected value
    :param variance_bound: variance bound on the estimator
    :param quantum_resource: quantum abstract machine object
    :return: expected value, estimator variance, total number of experiments
    """
    if sequential:
        expected_value = 0
        estimator_variance = 0
        total_shots = 0
        variance_bound_per_term = variance_bound / len(pauli_sum)
        for term in pauli_sum:
            exp_v, exp_var, exp_shots = \
                estimate_locally_commuting_operator_symmeterized(
                    program, PauliSum([term]), variance_bound_per_term,
                    quantum_resource, confusion_mat_dict=confusion_mat_dict)
            expected_value += exp_v
            estimator_variance += exp_var
            total_shots += exp_shots

        return expected_value, estimator_variance, total_shots

    else:
        return estimate_locally_commuting_operator_symmeterized(
            program, pauli_sum, variance_bound, quantum_resource,
            confusion_mat_dict=confusion_mat_dict)


def estimate_locally_commuting_operator_symmeterized(program, pauli_sum,
                                                     variance_bound,
                                                     quantum_resource,
                                                     confusion_mat_dict=None):
    """
    Estimate the expected value of a Pauli sum to fixed precision.

    Pauli sum can be a sum of non-commuting terms.  This routine groups the
    terms into sets of locally commuting operators.  This routine uses
    symmeterized readout.

    :param program: state preparation program
    :param pauli_sum: pauli sum of operators to estimate expected value
    :param variance_bound: variance bound on the estimator
    :param quantum_resource: quantum abstract machine object
    :return: expected value, estimator variance, total number of experiments
    """
    pauli_sum, identity_term = remove_identity(pauli_sum)

    expected_value = 0
    if isinstance(identity_term, int):
        if np.isclose(identity_term, 0):
            expected_value = 0
        else:
            expected_value = identity_term

    elif isinstance(identity_term, (PauliTerm, PauliSum)):
        if isinstance(identity_term, PauliTerm):
            expected_value = identity_term.coefficient
        else:
            expected_value = identity_term[0].coefficient
    else:
        raise TypeError("identity_term must be a PauliTerm or integer. We got type {}".format(type(identity_term)))

    # check if pauli_sum didn't get killed...or we gave an identity term
    if isinstance(pauli_sum, int):
        # we have no estimation work to do...just return the identity value
        return expected_value, 0, 0

    psets = group_terms_greedy(pauli_sum)
    variance_bound_per_set = variance_bound / len(psets)
    total_shots = 0
    estimator_variance = 0

    for qubit_op_key, pset in psets.items():
        results = estimate_pauli_sum_symmeterized(pset, dict(qubit_op_key), program,
                                                  variance_bound_per_set,
                                                  quantum_resource,
                                                  diagonal_ntpb_check=False,
                                                  confusion_mat_dict=confusion_mat_dict)

        assert results.variance < variance_bound_per_set
        expected_value += results.expected_value
        total_shots += results.n_shots
        estimator_variance += results.variance

    return expected_value, estimator_variance, total_shots


def group_terms_greedy(pauli_sums):
    """
    Computes commuting sets based on terms having the same diagonal basis
    Following the technique outlined in the appendix of arXiv:1704.05018.
    :param pauli_sums: PauliSum object to group
    :return: dictionary where key value pair is a tuple corresponding to the
             basis and a list of PauliTerms associated with that basis.
    """
    expt_settings = [ExperimentSetting(sI(), pt) for pt in pauli_sums]
    tomo_expt = TomographyExperiment(expt_settings, Program(), [])
    d_expt = _max_tpb_overlap(tomo_expt)
    d_tpb = {tuple(sorted(k.out_operator.operations_as_set())): [es.out_operator for es in v] for k, v in d_expt.items()}
    return d_tpb


def estimate_pauli_sum_symmeterized(pauli_terms,
                                    basis_transform_dict,
                                    program,
                                    variance_bound,
                                    quantum_resource,
                                    diagonal_ntpb_check=True,
                                    confusion_mat_dict=None):
    r"""
    Estimate the mean of a sum of pauli terms to set variance with symmeterized
    readout

    The sample variance is calculated by

    .. math::
        \begin{align}
        \mathrm{Var}[\hat{\langle H \rangle}] = \sum_{i, j}h_{i}h_{j}
        \mathrm{Cov}(\hat{\langle P_{i} \rangle}, \hat{\langle P_{j} \rangle})
        \end{align}

    The expectation value of each Pauli operator (term and coefficient) is
    also returned.  It can be accessed through the named-tuple field
    `pauli_expectations'.

    :param pauli_terms: list of pauli terms to measure simultaneously or a
                        PauliSum object
    :param basis_transform_dict: basis transform dictionary where the key is
                                 the qubit index and the value is the basis to
                                 rotate into. Valid basis is [I, X, Y, Z].
    :param program: program generating a state to sample from.  The program
                    is deep copied to ensure no mutation of gates or program
                    is perceived by the user.
    :param variance_bound:  Bound on the variance of the estimator for the
                            PauliSum. Remember this is the SQUARE of the
                            standard error!
    :param quantum_resource: quantum abstract machine object
    :param Bool diagonal_ntpb_check: Optional flag toggling a safety check
                                   ensuring all terms in `pauli_terms`
                                   commute with each other
    :return: estimated expected value, expected value of each Pauli term in
             the sum, covariance matrix, variance of the estimator, and the
             number of shots taken.  The objected returned is a named tuple with
             field names as follows: expected_value, pauli_expectations,
             covariance, variance, n_shots.
             `expected_value' == coef_vec.dot(pauli_expectations)
    :rtype: EstimationResult
    """
    if not isinstance(pauli_terms, (list, PauliSum)):
        raise TypeError("pauli_terms needs to be a list or a PauliSum")

    if isinstance(pauli_terms, PauliSum):
        pauli_terms = pauli_terms.terms

    # check if each term commutes with everything
    if diagonal_ntpb_check:
        try:
            _validate_all_diagonal_in_tpb(pauli_terms)
        except AssertionError:
            raise DiagonalNTPBError("Not all terms are diagonal in each others'"
                                    + "natural tensor product basis, as expected")

    program = program.copy()
    pauli_for_rotations = PauliTerm.from_list(
        [(value, key) for key, value in basis_transform_dict.items()])

    program += get_rotation_program_measure(pauli_for_rotations)

    qubits = sorted(list(basis_transform_dict.keys()))
    ro = program.declare("ro", "BIT", memory_size=len(qubits))
    for num, qubit in enumerate(qubits):
        program.inst(MEASURE(qubit, ro[num]))

    coeff_vec = np.array(
        list(map(lambda x: x.coefficient, pauli_terms))).reshape((-1, 1))

    results = None
    sample_variance = np.infty
    number_of_samples = 0

    # get confusion matrices
    if confusion_mat_dict is None:
        max_coeff = np.max(np.abs(coeff_vec))
        num_sample_ubound = max(int(10 * (max_coeff**2) * (len(pauli_terms)**2) / variance_bound), 1000)
        confusion_mat_dict = get_confusion_matrices(quantum_resource, qubits, num_sample_ubound)

    # create a rescale vector.  the rescale vector would take the expected value
    # of each pauli operator and rescale by the appropriate symmeterized
    # measurement adjustment. if perfect readout--i.e. p(0|0) = p(1|1) = 1 then
    # we should get the identity matrix back
    meas_adjustments = np.ones(len(pauli_terms))
    for pidx, term in enumerate(pauli_terms):
        term_qubits = term.get_qubits()
        rescale_coeffs = [1 / (confusion_mat_dict[x][0, 0] + confusion_mat_dict[x][1, 1] - 1) for x in term_qubits]
        meas_adjustments[pidx] *= reduce(lambda x, y: x * y, rescale_coeffs)

    max_coeff = np.max(np.abs(coeff_vec * meas_adjustments))
    num_sample_ubound = max(int(10 * (max_coeff**2) * (len(pauli_terms)**2) / variance_bound), 1000)
    if num_sample_ubound <= 2:
        raise ValueError("Something happened with our calculation of the max sample")

    standard_numshots = 10000

    program = program.wrap_in_numshots_loop(min(standard_numshots, num_sample_ubound + (num_sample_ubound % 2)))
    binary = quantum_resource.compiler.native_quil_to_executable(program)

    while (sample_variance > variance_bound
            and number_of_samples < num_sample_ubound):
        tresults = quantum_resource.run(binary)
        number_of_samples += len(tresults)

        parity_results = get_parity(pauli_terms, tresults)

        # Note: easy improvement would be to update mean and variance on the fly
        # instead of storing all these results.
        if results is None:
            results = parity_results
        else:
            results = np.hstack((results, parity_results))

        # calculate the expected values....
        # this coeff matrix is for Cov(aX, bY) = a * b * Cov(X, Y)
        # symmeterized readout is rescaling each random variable by a constant
        # so we need to calculate the adjusted covariance matrix we compute
        # the hadamard product between the outer product of the coeff adjustement
        # vector and the covariance matrix
        coeff_mat_adjusted = np.outer(meas_adjustments, meas_adjustments)
        covariance_mat = np.cov(results, ddof=1)
        sample_variance = coeff_vec.T.dot(covariance_mat * coeff_mat_adjusted).dot(coeff_vec) / (results.shape[1] - 1)

    return EstimationResult(expected_value=coeff_vec.T.dot(np.multiply(meas_adjustments, np.mean(results, axis=1))),
                            pauli_expectations=np.multiply(coeff_vec.flatten(), np.multiply(meas_adjustments, np.mean(results, axis=1))).flatten(),
                            covariance=np.multiply(covariance_mat, coeff_mat_adjusted),
                            variance=sample_variance,
                            n_shots=results.shape[1])


def get_confusion_matrices(quantum_resource, qubits, num_sample_ubound):
    """
    Get the confusion matrices from the quantum resource given number of samples

    This allows the user to change the accuracy at which they estimate the
    confusion matrix

    :param quantum_resource: Quantum Abstract Machine connection object
    :param qubits: qubits to measure 1-qubit readout confusion matrices
    :return: dictionary of confusion matrices indexed by the qubit label
    :rtype: dict
    """
    confusion_mat_dict = {q: estimate_confusion_matrix(quantum_resource, q, num_sample_ubound) for q in qubits}
    return confusion_mat_dict


def get_confusion_matrix_programs(qubit):
    """Construct programs for measuring a confusion matrix.

    This is a fancy way of saying "measure |0>"  and "measure |1>".

    :returns: program that should measure |0>, program that should measure |1>.
    """
    zero_meas = Program()
    ro = zero_meas.declare('ro', 'BIT', 1)
    zero_meas += I(qubit)
    zero_meas += I(qubit)
    zero_meas += MEASURE(qubit, ro[0])

    # prepare one and get statistics
    one_meas = Program()
    ro = one_meas.declare('ro', 'BIT', 1)
    one_meas += I(qubit)
    one_meas += RX(pi, qubit)
    one_meas += MEASURE(qubit, ro[0])

    return zero_meas, one_meas


def estimate_confusion_matrix(qam: 'QuantumComputer', qubit: int, samples=10000):
    """Estimate the readout confusion matrix for a given qubit.

    :param qam: The quantum computer to estimate the confusion matrix.
    :param qubit: The actual physical qubit to measure
    :param samples: The number of shots to take. This function runs two programs, so
        the total number of shots taken will be twice this number.
    """
    zero_meas, one_meas = get_confusion_matrix_programs(qubit)
    zero_meas.wrap_in_numshots_loop(samples)
    should_be_0 = qam.run(zero_meas)

    one_meas.wrap_in_numshots_loop(samples)
    should_be_1 = qam.run(one_meas)
    # should_be_0 = qam.run(zero_meas, [0], samples)
    # should_be_1 = qam.run(one_meas, [0], samples)
    p00 = 1 - np.mean(should_be_0)
    p11 = np.mean(should_be_1)

    return np.array([[p00, 1 - p00],
                     [1 - p11, p11]])
