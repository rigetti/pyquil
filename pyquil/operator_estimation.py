import functools
import itertools
import json
import logging
import re
import sys
import warnings
from json import JSONEncoder
from math import pi
from operator import mul
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
class _OneQState:
    """
    A description of a named one-qubit quantum state.

    This can be used to generate pre-rotations for quantum process tomography. For example,
    X0_14 will generate the +1 eigenstate of the X operator on qubit 14. X1_14 will generate the
    -1 eigenstate. SIC0_14 will generate the 0th SIC-basis state on qubit 14.
    """
    label: str
    index: int
    qubit: int

    def __str__(self):
        return f'{self.label}{self.index}_{self.qubit}'

    @classmethod
    def from_str(cls, s):
        ma = re.match(r'\s*(\w+)(\d+)_(\d+)\s*', s)
        if ma is None:
            raise ValueError(f"Couldn't parse '{s}'")
        return _OneQState(
            label=ma.group(1),
            index=int(ma.group(2)),
            qubit=int(ma.group(3)),
        )


@dataclass(frozen=True)
class TensorProductState:
    """
    A description of a multi-qubit quantum state that is a tensor product of many _OneQStates
    states.
    """
    states: Tuple[_OneQState]

    def __init__(self, states=None):
        if states is None:
            states = tuple()
        object.__setattr__(self, 'states', tuple(states))

    def __mul__(self, other):
        return TensorProductState(self.states + other.states)

    def __str__(self):
        return ' * '.join(str(s) for s in self.states)

    def __repr__(self):
        return f'TensorProductState[{self}]'

    def __getitem__(self, qubit):
        """Return the _OneQState at the given qubit."""
        for oneq_state in self.states:
            if oneq_state.qubit == qubit:
                return oneq_state
        raise IndexError()

    def __iter__(self):
        yield from self.states

    def __len__(self):
        return len(self.states)

    def states_as_set(self):
        return frozenset(self.states)

    def __eq__(self, other):
        if not isinstance(other, TensorProductState):
            return False

        return self.states_as_set() == other.states_as_set()

    def __hash__(self):
        return hash(self.states_as_set())

    @classmethod
    def from_str(cls, s):
        if s == '':
            return TensorProductState()
        return TensorProductState(tuple(_OneQState.from_str(x) for x in s.split('*')))


def SIC0(q):
    return TensorProductState((_OneQState('SIC', 0, q),))


def SIC1(q):
    return TensorProductState((_OneQState('SIC', 1, q),))


def SIC2(q):
    return TensorProductState((_OneQState('SIC', 2, q),))


def SIC3(q):
    return TensorProductState((_OneQState('SIC', 3, q),))


def plusX(q):
    return TensorProductState((_OneQState('X', 0, q),))


def minusX(q):
    return TensorProductState((_OneQState('X', 1, q),))


def plusY(q):
    return TensorProductState((_OneQState('Y', 0, q),))


def minusY(q):
    return TensorProductState((_OneQState('Y', 1, q),))


def plusZ(q):
    return TensorProductState((_OneQState('Z', 0, q),))


def minusZ(q):
    return TensorProductState((_OneQState('Z', 1, q),))


def zeros_state(qubits: Iterable[int]):
    return TensorProductState(_OneQState('Z', 0, q) for q in qubits)


@dataclass(frozen=True, init=False)
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
    in_state: TensorProductState
    out_operator: PauliTerm

    def __init__(self, in_state: TensorProductState, out_operator: PauliTerm):
        # For backwards compatibility, handle in_state specified by PauliTerm.
        if isinstance(in_state, PauliTerm):
            warnings.warn("Please specify in_state as a TensorProductState",
                          DeprecationWarning, stacklevel=2)

            if is_identity(in_state):
                in_state = TensorProductState()
            else:
                in_state = TensorProductState([
                    _OneQState(label=pauli_label, index=0, qubit=qubit)
                    for qubit, pauli_label in in_state._ops.items()
                ])

        object.__setattr__(self, 'in_state', in_state)
        object.__setattr__(self, 'out_operator', out_operator)

    @property
    def in_operator(self):
        warnings.warn("ExperimentSetting.in_operator is deprecated in favor of in_state",
                      stacklevel=2)

        # Backwards compat
        pt = sI()
        for oneq_state in self.in_state.states:
            if oneq_state.label not in ['X', 'Y', 'Z']:
                raise ValueError(f"Can't shim {oneq_state.label} into a pauli term. Use in_state.")
            if oneq_state.index != 0:
                raise ValueError(f"Can't shim {oneq_state} into a pauli term. Use in_state.")

            pt *= PauliTerm(op=oneq_state.label, index=oneq_state.qubit)

        return pt

    def __str__(self):
        return f'{self.in_state}→{self.out_operator.compact_str()}'

    def __repr__(self):
        return f'ExperimentSetting[{self}]'

    def serializable(self):
        return str(self)

    @classmethod
    def from_str(cls, s: str):
        """The opposite of str(expt)"""
        instr, outstr = s.split('→')
        return ExperimentSetting(in_state=TensorProductState.from_str(instr),
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


def _one_q_sic_prep(index, qubit):
    """Prepare the index-th SIC basis state."""
    if index == 0:
        return Program()

    theta = 2 * np.arccos(1 / np.sqrt(3))
    zx_plane_rotation = Program([
        RX(-pi / 2, qubit),
        RZ(theta - pi, qubit),
        RX(-pi / 2, qubit),
    ])

    if index == 1:
        return zx_plane_rotation

    elif index == 2:
        return zx_plane_rotation + RZ(-2 * pi / 3, qubit)

    elif index == 3:
        return zx_plane_rotation + RZ(2 * pi / 3, qubit)

    raise ValueError(f'Bad SIC index: {index}')


def _one_q_pauli_prep(label, index, qubit):
    """Prepare the index-th eigenstate of the pauli operator given by label."""
    if index not in [0, 1]:
        raise ValueError(f'Bad Pauli index: {index}')

    if label == 'X':
        if index == 0:
            return Program(RY(pi / 2, qubit))
        else:
            return Program(RY(-pi / 2, qubit))

    elif label == 'Y':
        if index == 0:
            return Program(RX(-pi / 2, qubit))
        else:
            return Program(RX(pi / 2, qubit))

    elif label == 'Z':
        if index == 0:
            return Program()
        else:
            return Program(RX(pi, qubit))

    raise ValueError(f'Bad Pauli label: {label}')


def _one_q_state_prep(oneq_state: _OneQState):
    """Prepare a one qubit state.

    Either SIC[0-3], X[0-1], Y[0-1], or Z[0-1].
    """
    label = oneq_state.label
    if label == 'SIC':
        return _one_q_sic_prep(oneq_state.index, oneq_state.qubit)
    elif label in ['X', 'Y', 'Z']:
        return _one_q_pauli_prep(label, oneq_state.index, oneq_state.qubit)
    else:
        raise ValueError(f"Bad state label: {label}")


def _local_pauli_eig_meas(op, idx):
    """
    Generate gate sequence to measure in the eigenbasis of a Pauli operator, assuming
    we are only able to measure in the Z eigenbasis. (Note: The unitary operations of this
    Program are essentially the Hermitian conjugates of those in :py:func:`_one_q_pauli_prep`)

    """
    if op == 'X':
        return Program(RY(-pi / 2, idx))
    elif op == 'Y':
        return Program(RX(pi / 2, idx))
    elif op == 'Z':
        return Program()
    raise ValueError(f'Unknown operation {op}')


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

        max_weight_in = _max_weight_state([expt1.in_state, expt2.in_state])
        max_weight_out = _max_weight_operator([expt1.out_operator, expt2.out_operator])
        if max_weight_in is not None and max_weight_out is not None:
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


def _max_weight_operator(ops: Iterable[PauliTerm]) -> Union[None, PauliTerm]:
    """Construct a PauliTerm operator by taking the non-identity single-qubit operator at each
    qubit position.

    This function will return ``None`` if the input operators do not share a natural tensor
    product basis.

    For example, the max_weight_operator of ["XI", "IZ"] is "XZ". Asking for the max weight
    operator of something like ["XI", "ZI"] will return None.
    """
    mapping = dict()  # type: Dict[int, str]
    for op in ops:
        for idx, op_str in op:
            if idx in mapping:
                if mapping[idx] != op_str:
                    return None
            else:
                mapping[idx] = op_str
    op = functools.reduce(mul, (PauliTerm(op, q) for q, op in mapping.items()), sI())
    return op


def _max_weight_state(states: Iterable[TensorProductState]) -> Union[None, TensorProductState]:
    """Construct a TensorProductState by taking the single-qubit state at each
    qubit position.

    This function will return ``None`` if the input states are not compatible

    For example, the max_weight_state of ["(+X, q0)", "(-Z, q1)"] is "(+X, q0; -Z q1)". Asking for
    the max weight state of something like ["(+X, q0)", "(+Z, q0)"] will return None.
    """
    mapping = dict()  # type: Dict[int, _OneQState]
    for state in states:
        for oneq_state in state.states:
            if oneq_state.qubit in mapping:
                if mapping[oneq_state.qubit] != oneq_state:
                    return None
            else:
                mapping[oneq_state.qubit] = oneq_state
    return TensorProductState(list(mapping.values()))


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
            trial_es_list = es_list + [expt_setting]
            diag_in_term = _max_weight_state(expst.in_state for expst in trial_es_list)
            diag_out_term = _max_weight_operator(expst.out_operator for expst in trial_es_list)
            # max_weight_xxx returns None if the set of xxx's don't share a TPB, so the following
            # conditional is True if expt_setting can be inserted into the current es_list.
            if diag_in_term is not None and diag_out_term is not None:
                found_tpb = True
                assert len(diag_in_term) >= len(es.in_state), \
                    "Highest weight in-state can't be smaller than the given in-state"
                assert len(diag_out_term) >= len(es.out_operator), \
                    "Highest weight out-PauliTerm can't be smaller than the given out-PauliTerm"

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


def group_experiments(experiments: TomographyExperiment,
                      method: str = 'greedy') -> TomographyExperiment:
    """
    Group experiments that are diagonal in a shared tensor product basis (TPB) to minimize number
    of QPU runs.

    Background
    ----------

    Given some PauliTerm operator, the 'natural' tensor product basis to
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


    Methods
    -------

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

    In the case of readout error calibration, we also include
    expectation, standard deviation and count for the calibration results, as well as the
    expectation and standard deviation for the corrected results.
    """

    setting: ExperimentSetting
    expectation: Union[float, complex]
    stddev: Union[float, complex]
    total_counts: int
    raw_expectation: Union[float, complex] = None
    raw_stddev: float = None
    calibration_expectation: Union[float, complex] = None
    calibration_stddev: Union[float, complex] = None
    calibration_counts: int = None

    def __str__(self):
        return f'{self.setting}: {self.expectation} +- {self.stddev}'

    def __repr__(self):
        return f'ExperimentResult[{self}]'

    def serializable(self):
        return {
            'type': 'ExperimentResult',
            'setting': self.setting,
            'expectation': self.corrected_expectation,
            'stddev': self.corrected_stddev,
            'total_counts': self.total_counts,
            'raw_expectation': self.expectation,
            'raw_stddev': self.stddev,
            'calibration_expectation': self.calibration_expectation,
            'calibration_stddev': self.calibration_stddev,
            'calibration_counts': self.calibration_counts,
        }


def measure_observables(qc: QuantumComputer, tomo_experiment: TomographyExperiment,
                        n_shots: int = 1000, progress_callback=None, active_reset=False,
                        readout_symmetrize: str = None, calibrate_readout: str = None):
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
    :param readout_symmetrize: Method used to symmetrize the readout errors, i.e. set
        p(0|1) = p(1|0). For uncorrelated readout errors, this can be achieved by randomly
        selecting between the POVMs {X.D1.X, X.D0.X} and {D0, D1} (where both D0 and D1 are
        diagonal). However, here we currently support exhaustive symmetrization and loop through
        all possible 2^n POVMs {X/I . POVM . X/I}^n, and obtain symmetrization more generally,
        i.e. set p(00|00) = p(01|01) = .. = p(11|11), as well as p(00|01) = p(01|00) etc. If this
        is None, no symmetrization is performed. The exhaustive method can be specified by setting
        this variable to 'exhaustive'
    :param calibrate_readout: Method used to calibrate the readout results. Currently, the only
        method supported is normalizing against the operator's expectation value in its +1
        eigenstate, which can be specified by setting this variable to 'plus-eig'. The preceding
        symmetrization and this step together yield a more accurate estimation of the observable.
    """
    # calibration readout only works with symmetrization turned on
    if calibrate_readout is not None and readout_symmetrize is None:
        raise ValueError("Readout calibration only works with readout symmetrization turned on")

    # Outer loop over a collection of grouped settings for which we can simultaneously
    # estimate.
    for i, settings in enumerate(tomo_experiment):

        log.info(f"Collecting bitstrings for the {len(settings)} settings: {settings}")

        # 1.1 Prepare a state according to the amalgam of all setting.in_state
        total_prog = Program()
        if active_reset:
            total_prog += RESET()
        max_weight_in_state = _max_weight_state(setting.in_state for setting in settings)
        for oneq_state in max_weight_in_state.states:
            total_prog += _one_q_state_prep(oneq_state)

        # 1.2 Add in the program
        total_prog += tomo_experiment.program

        # 1.3 Measure the state according to setting.out_operator
        max_weight_out_op = _max_weight_operator(setting.out_operator for setting in settings)
        for qubit, op_str in max_weight_out_op:
            total_prog += _local_pauli_eig_meas(op_str, qubit)

        if readout_symmetrize == 'exhaustive':
            # 1.4 Symmetrize -- flip qubits pre-measurement
            qubits = max_weight_out_op.get_qubits()
            ops_strings = _ops_strs_symmetrize(qubits)
            list_bitstrings_symm = []
            for ops_str in ops_strings:
                total_prog_symm = total_prog.copy()
                prog_symm = _ops_str_to_prog(ops_str, qubits)
                total_prog_symm += prog_symm
                # 2. Run the experiment
                n_shots_symm = n_shots // len(ops_strings)
                bitstrings_symm = qc.run_and_measure(total_prog_symm, n_shots_symm)
                # 2.1 Flip the results post-measurement
                d_flips_symm = _ops_str_to_flips(ops_str, qubits)
                for qubit, bs_results in bitstrings_symm.items():
                    bitstrings_symm[qubit] = bs_results ^ d_flips_symm.get(qubit, 0)
                # 2.2 Gather together the symmetrized results into list
                list_bitstrings_symm.append(bitstrings_symm)

            # 2.3 Gather together all the symmetrized results
            bitstrings = reduce(lambda d1, d2: _stack_dicts(d1, d2), list_bitstrings_symm)

        elif readout_symmetrize is None:
            # 2. Run the experiment
            bitstrings = qc.run_and_measure(total_prog, n_shots)

        else:
            raise ValueError("Readout symmetrization method must be either 'exhaustive' or None")

        if progress_callback is not None:
            progress_callback(i, len(tomo_experiment))

        # 3. Post-process
        # Inner loop over the grouped settings. They only differ in which qubits' measurements
        # we include in the post-processing. For example, if `settings` is Z1, Z2, Z1Z2 and we
        # measure (n_shots, n_qubits=2) obs_strings then the full operator value involves selecting
        # either the first column, second column, or both and multiplying along the row.
        for setting in settings:
            # 3.1 Get the term's coefficient so we can multiply it in later.
            coeff = complex(setting.out_operator.coefficient)
            if not np.isclose(coeff.imag, 0):
                raise ValueError(f"{setting}'s out_operator has a complex coefficient.")
            coeff = coeff.real

            # 3.2 Special case for measuring the "identity" operator, which doesn't make much
            #     sense but should happen perfectly.
            if is_identity(setting.out_operator):
                yield ExperimentResult(
                    setting=setting,
                    expectation=coeff,
                    stddev=0.0,
                    total_counts=n_shots,
                )
                continue

            # 3.3 Obtain statistics from result of experiment
            obs_mean, obs_var = _stats_from_measurements(bitstrings, setting, n_shots, coeff)

            if calibrate_readout == 'plus-eig':
                # 4 Readout calibration
                # 4.1 Prepare the +1 eigenstate for the out operator
                calibr_prog = Program()
                for q, op in setting.out_operator.operations_as_set():
                    calibr_prog += _one_q_pauli_prep(label=op, index=0, qubit=q)
                # 4.2 Measure the out operator in this state
                for q, op in setting.out_operator.operations_as_set():
                    calibr_prog += _local_pauli_eig_meas(op, q)
                calibr_shots = n_shots
                calibr_results = qc.run_and_measure(calibr_prog, calibr_shots)
                # 4.3 Obtain statistics from the measurement process
                obs_calibr_mean, obs_calibr_var = _stats_from_measurements(calibr_results, setting, calibr_shots)
                # 4.4 Calibrate the readout results
                corrected_mean = obs_mean / obs_calibr_mean
                corrected_var = ratio_variance(obs_mean, obs_var, obs_calibr_mean, obs_calibr_var)

                yield ExperimentResult(
                    setting=setting,
                    expectation=corrected_mean.item(),
                    stddev=np.sqrt(corrected_var).item(),
                    total_counts=n_shots,
                    raw_expectation=obs_mean.item(),
                    raw_stddev=np.sqrt(obs_var).item(),
                    calibration_expectation=obs_calibr_mean.item(),
                    calibration_stddev=np.sqrt(obs_calibr_var).item(),
                    calibration_counts=calibr_shots,
                )

            elif calibrate_readout is None:
                # No calibration
                yield ExperimentResult(
                    setting=setting,
                    expectation=obs_mean.item(),
                    stddev=np.sqrt(obs_var).item(),
                    total_counts=n_shots,
                )

            else:
                raise ValueError("Calibration readout method must be either 'plus-eig' or None")


def _ops_strs_symmetrize(qubits: List[int]) -> List[str]:
    """
    :param qubits: list specifying the qubits whose readout errors we wish to symmetrize
    :return: list with the operation strings necessary for exhaustive symmetrization
    """
    ops_strings = []
    for prod in itertools.product(['I', 'X'], repeat=len(qubits)):
        ops_strings.append(''.join(prod))
    return ops_strings


def _ops_str_to_prog(ops_str: str, qubits: List[int]) -> Program:
    """
    :param ops_str: string specifying the operation to be carried out on `qubits`
    :param qubits: list specifying the qubits to be carried operations on
    :return: Program with the operations specified in `ops_str` on the qubits specified in `qubits`
    """
    assert len(ops_str) == len(qubits), "Mismatch of qubits and operations"
    prog = Program()
    for i, op_ch in enumerate(ops_str):
        if op_ch == 'I':
            continue
        elif op_ch == 'X':
            prog += Program(X(qubits[i]))
        else:
            raise ValueError("ops_strings_ should only consist of 'I's and/or 'X's")
    return prog


def _ops_str_to_flips(ops_str: str, qubits: List[int]) -> Dict:
    """
    :param ops_str: string specifying the operation to be carried out on `qubits`
    :param qubits: list specifying the qubits to be carried operations on
    :return: Dict specyfing whether to flip the readout results or not, depending on
        the operations specified in `ops_str`, which in turn are operating on the
        qubits specified in `qubits`
    """
    d_flip = {}
    for i, op_ch in enumerate(ops_str):
        q = qubits[i]
        if op_ch == 'I':
            d_flip[q] = 0
        elif op_ch == 'X':
            d_flip[q] = 1
        else:
            raise ValueError("ops_strings_ should only consist of 'I's and/or 'X's")
    return d_flip


def _stack_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Given two dicts representing qubit measurements, keyed by integers (representing qubits)
    and valued by 1-dimensional numpy arrays (representing measurements), this helper function
    horizontally stacks the measurement results from the two dicts to produce a single
    1-dimensional numpy array representing the measurement results in a single dict.

    :param dict1: Dict keyed with integer specifying qubit, valued by 1-dimensional
        numpy array specifying readout results
    :param dict2: Dict keyed with integer specifying qubit, valued by 1-dimensional
        numpy array specifying readout results
    :return: Dict keyed with integer specifying qubit, valued by 1-dimensional
        numpy array specifying readout results gathered from both `dict1` and `dict2`
    """
    assert set(dict1.keys()) == set(dict2.keys()), "Dictionaries must have same keys"
    assert set(len(v.shape) for v in dict1.values()) == \
           set(len(v.shape) for v in dict2.values()), "Arrays in dict values must have same dimension"
    dict_combined = {}
    for k, v in dict1.items():
        dict_combined[k] = np.hstack([v, dict2[k]])
    return dict_combined


def _stats_from_measurements(d_results: dict, setting: ExperimentSetting,
                             n_shots: int, coeff: float = 1.0) -> Tuple[float]:
    """
    :param d_results: results from running `qc.run_and_measure()`
    :param setting: ExperimentSetting
    :param n_shots: number of shots in the measurement process
    :param coeff: coefficient of the operator being estimated
    :return: tuple specifying (mean, variance)
    """
    # Transform bits to eigenvalues; ie (+1, -1)
    obs_strings = {q: 1 - 2 * d_results[q] for q in d_results}
    # Pick columns corresponding to qubits with a non-identity out_operation and stack
    # into an array of shape (n_shots, n_measure_qubits)
    my_obs_strings = np.vstack([obs_strings[q] for q, op_str in setting.out_operator]).T
    # Multiply row-wise to get operator values. Do statistics. Return result.
    obs_vals = coeff * np.prod(my_obs_strings, axis=1)
    obs_mean = np.mean(obs_vals)
    obs_var = np.var(obs_vals) / n_shots

    return obs_mean, obs_var


def ratio_variance(a: Union[float, np.ndarray],
                   var_a: Union[float, np.ndarray],
                   b: Union[float, np.ndarray],
                   var_b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Given random variables 'A' and 'B', compute the variance on the ratio Y = A/B. Denote the
    mean of the random variables as a = E[A] and b = E[B] while the variances are var_a = Var[A]
    and var_b = Var[B] and the covariance as Cov[A,B]. The following expression approximates the
    variance of Y

    Var[Y] \approx (a/b) ^2 * ( var_a /a^2 + var_b / b^2 - 2 * Cov[A,B]/(a*b) )

    We assume the covariance of A and B is negligible, resting on the assumption that A and B
    are independently measured. The expression above rests on the assumption that B is non-zero,
    an assumption which we expect to hold true in most cases, but makes no such assumptions
    about A. If we allow E[A] = 0, then calculating the expression above via numpy would complain
    about dividing by zero. Instead, we can re-write the above expression as

    Var[Y] \approx var_a /b^2 + (a^2 * var_b) / b^4

    where we have dropped the covariance term as noted above.

    See the following for more details:
      - https://doi.org/10.1002/(SICI)1097-0320(20000401)39:4<300::AID-CYTO8>3.0.CO;2-O
      - http://www.stat.cmu.edu/~hseltman/files/ratio.pdf
      - https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables

    :param a: Mean of 'A', to be used as the numerator in a ratio.
    :param var_a: Variance in 'A'
    :param b: Mean of 'B', to be used as the numerator in a ratio.
    :param var_b: Variance in 'B'
    """
    return var_a / b**2 + (a**2 * var_b) / b**4
