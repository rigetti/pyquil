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
Schema definition of an ExperimentSetting, which corresponds to preparing a collection of qubits
in a TensorProductState and measuring them in a PauliTerm-defined basis.
"""
import logging
import re
import sys
import warnings
from typing import Iterable, Tuple

from pyquil.paulis import PauliTerm, sI, is_identity
from pyquil.experiment._memory import (pauli_term_to_measurement_memory_map,
                                       pauli_term_to_preparation_memory_map)

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


def _pauli_to_product_state(in_state: PauliTerm) -> TensorProductState:
    """
    Convert a Pauli term to a TensorProductState.
    """
    if is_identity(in_state):
        in_state = TensorProductState()
    else:
        in_state = TensorProductState([
            _OneQState(label=pauli_label, index=0, qubit=qubit)
            for qubit, pauli_label in in_state._ops.items()
        ])
    return in_state


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
            in_state = _pauli_to_product_state(in_state)
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
