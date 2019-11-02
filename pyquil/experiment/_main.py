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
"""
Schema definition of a TomographyExperiment, which is a collection of ExperimentSetting
objects and a main program body (or ansatz). This schema is widely useful for defining
and executing many common types of algorithms / applications, including state and process
tomography, and the variational quantum eigensolver.
"""
import json
import logging
import warnings
from json import JSONEncoder
from enum import IntEnum
from typing import List, Union, Optional

from pyquil import Program
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import ExperimentSetting
from pyquil.quilbase import DefPermutationGate, Reset


log = logging.getLogger(__name__)


class SymmetrizationLevel(IntEnum):
    EXHAUSTIVE = -1
    NONE = 0
    OA_STRENGTH_1 = 1
    OA_STRENGTH_2 = 2
    OA_STRENGTH_3 = 3


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

    return '   ' + '\n   '.join(program_lines)


def _remove_reset_from_program(program: Program) -> Program:
    """
    Trim the RESET from a program because in measure_observables it is re-added.

    :param program: Program to remove RESET(s) from.
    :return: Trimmed Program.
    """
    definitions = [gate for gate in program.defined_gates]

    p = Program([inst for inst in program if not isinstance(inst, Reset)])

    for definition in definitions:
        if isinstance(definition, DefPermutationGate):
            p.inst(DefPermutationGate(definition.name, list(definition.permutation)))
        else:
            p.defgate(definition.name, definition.matrix, definition.parameters)
    return p


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

    :ivar settings: The collection of ExperimentSetting objects that define this experiment.
    :ivar program: The main program body of this experiment. Also determines the ``shots``
        and ``reset`` instance variables. The ``shots`` instance variable is the number of
        shots to take per ExperimentSetting. The ``reset`` instance variable is whether to
        actively reset qubits instead of waiting several times the coherence length for qubits
        to decay to ``|0>`` naturally. Setting this to True is much faster but there is a ~1%
        error per qubit in the reset operation. Thermal noise from "traditional" reset is not
        routinely characterized but is of the same order.
    :ivar symmetrization: the level of readout symmetrization to perform for the estimation
        and optional calibration of each observable. The following integer levels, encapsulated in
        the ``SymmetrizationLevel`` integer enum, are currently supported:

        * -1 -- exhaustive symmetrization uses every possible combination of flips
        * 0 -- no symmetrization
        * 1 -- symmetrization using an orthogonal array (OA) with strength 1
        * 2 -- symmetrization using an orthogonal array (OA) with strength 2
        * 3 -- symmetrization using an orthogonal array (OA) with strength 3

        Note that (default) exhaustive symmetrization requires a number of QPU calls exponential in
        the number of qubits in the union of the support of the observables in any group of settings
        in ``tomo_experiment``; the number of shots may need to be increased to accommodate this.
        see :func:`run_symmetrized_readout` in api._quantum_computer for more information.
    """

    def __init__(self,
                 settings: Union[List[ExperimentSetting], List[List[ExperimentSetting]]],
                 program: Program,
                 qubits: Optional[List[int]] = None,
                 *,
                 symmetrization: int = SymmetrizationLevel.EXHAUSTIVE):
        if len(settings) == 0:
            settings = []
        else:
            if isinstance(settings[0], ExperimentSetting):
                # convenience wrapping in lists of length 1
                settings = [[expt] for expt in settings]

        self._settings = settings  # type: List[List[ExperimentSetting]]
        self.program = program
        if qubits is not None:
            warnings.warn("The 'qubits' parameter has been deprecated and will be removed"
                          "in a future release of pyquil")
        self.qubits = qubits
        self.symmetrization = SymmetrizationLevel(symmetrization)
        self.shots = self.program.num_shots

        if 'RESET' in self.program.out():
            self.reset = True
            self.program = _remove_reset_from_program(self.program)
        else:
            self.reset = False

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
            setting_strs = (setting_strs[:first_n] + [f'... {excluded} settings not shown ...']
                            + setting_strs[-last_n:])
        return '   ' + '\n   '.join(setting_strs)

    def __repr__(self):
        string = f'shots: {self.shots}\n'
        if self.reset:
            string += f'active reset: enabled\n'
        else:
            string += f'active reset: disabled\n'
        string += f'symmetrization: {self.symmetrization} ({self.symmetrization.name.lower()})\n'
        string += f'program:\n{_abbrev_program(self.program)}\n'
        string += f'settings:\n{self.settings_string(abbrev_after=20)}'
        return string

    def serializable(self):
        return {
            'type': 'TomographyExperiment',
            'settings': self._settings,
            'program': self.program.out(),
            'symmetrization': self.symmetrization,
            'shots': self.shots,
            'reset': self.reset
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
    """
    Convenience method to save pyquil.experiment objects as a JSON file.

    See :py:func:`read_json`.
    """
    with open(fn, 'w') as f:
        json.dump(obj, f, cls=OperatorEncoder, indent=2, ensure_ascii=False)
    return fn


def _operator_object_hook(obj):
    if 'type' in obj and obj['type'] == 'TomographyExperiment':
        # I bet this doesn't work for grouped experiment settings
        settings = [[ExperimentSetting.from_str(s) for s in stt] for stt in obj['settings']]
        p = Program(obj['program'])
        p.wrap_in_numshots_loop(obj['shots'])
        ex = TomographyExperiment(settings=settings,
                                  program=p,
                                  symmetrization=obj['symmetrization'])
        ex.reset = obj['reset']
        return ex
    return obj


def read_json(fn):
    """
    Convenience method to read pyquil.experiment objects from a JSON file.

    See :py:func:`to_json`.
    """
    with open(fn) as f:
        return json.load(f, object_hook=_operator_object_hook)
