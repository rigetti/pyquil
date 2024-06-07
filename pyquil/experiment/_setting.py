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
"""Definition of an ExperimentSetting.

Each ExperimentSetting corresponds to preparing a collection of qubits in a TensorProductState and measuring them in a
PauliTerm-defined basis.
"""

import logging
import re
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import Any, Optional, cast

from pyquil.paulis import PauliTerm, sI

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _OneQState:
    """A description of a named one-qubit quantum state.

    This can be used to generate pre-rotations for quantum process tomography. For example,
    X0_14 will generate the +1 eigenstate of the X operator on qubit 14. X1_14 will generate the
    -1 eigenstate. SIC0_14 will generate the 0th SIC-basis state on qubit 14.
    """

    label: str
    index: int
    qubit: int

    def __str__(self) -> str:
        return f"{self.label}{self.index}_{self.qubit}"

    @classmethod
    def from_str(cls, s: str) -> "_OneQState":
        ma = re.match(r"\s*(\w+)(\d+)_(\d+)\s*", s)
        if ma is None:
            raise ValueError(f"Couldn't parse '{s}'")
        return _OneQState(label=ma.group(1), index=int(ma.group(2)), qubit=int(ma.group(3)))


@dataclass(frozen=True)
class TensorProductState:
    """A description of a multi-qubit quantum state that is a tensor product of many _OneQStates states."""

    states: list[_OneQState]

    def __init__(self, states: Optional[Iterable[_OneQState]] = None):
        if states is None:
            states = []
        object.__setattr__(self, "states", list(states))

    def __mul__(self, other: "TensorProductState") -> "TensorProductState":
        return TensorProductState(self.states + other.states)

    def __str__(self) -> str:
        return " * ".join(str(s) for s in self.states)

    def __repr__(self) -> str:
        return f"TensorProductState[{self}]"

    def __getitem__(self, qubit: int) -> _OneQState:
        """Return the _OneQState at the given qubit."""
        for oneq_state in self.states:
            if oneq_state.qubit == qubit:
                return oneq_state
        raise IndexError()

    def __iter__(self) -> Generator[_OneQState, None, None]:
        yield from self.states

    def __len__(self) -> int:
        return len(self.states)

    def states_as_set(self) -> frozenset[_OneQState]:
        return frozenset(self.states)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TensorProductState):
            return False

        return self.states_as_set() == other.states_as_set()

    def __hash__(self) -> int:
        return hash(self.states_as_set())

    @classmethod
    def from_str(cls, s: str) -> "TensorProductState":
        if s == "":
            return TensorProductState()
        return TensorProductState(list(_OneQState.from_str(x) for x in s.split("*")))


def SIC0(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label="SIC", index=0, qubit=q)])


def SIC1(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label="SIC", index=1, qubit=q)])


def SIC2(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label="SIC", index=2, qubit=q)])


def SIC3(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label="SIC", index=3, qubit=q)])


def plusX(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label="X", index=0, qubit=q)])


def minusX(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label="X", index=1, qubit=q)])


def plusY(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label="Y", index=0, qubit=q)])


def minusY(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label="Y", index=1, qubit=q)])


def plusZ(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label="Z", index=0, qubit=q)])


def minusZ(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label="Z", index=1, qubit=q)])


def zeros_state(qubits: Iterable[int]) -> TensorProductState:
    return TensorProductState([_OneQState(label="Z", index=0, qubit=q) for q in qubits])


@dataclass(frozen=True, init=False)
class ExperimentSetting:
    """Input and output settings for a tomography-like experiment.

    Many near-term quantum algorithms take the following form:

     - Start in a pauli state
     - Prepare some ansatz
     - Measure it w.r.t. pauli operators

    Where we typically use a large number of (start, measure) pairs but keep the ansatz preparation
    program consistent. This class represents the (start, measure) pairs. Typically a large
    number of these :py:class:`ExperimentSetting` objects will be created and grouped into
    an :py:class:`Experiment`.

    :ivar additional_expectations: A list of lists, where each inner list specifies a qubit subset
        to calculate the joint expectation value for. This attribute allows users to extract
        simultaneously measurable expectation values from a single experiment setting.
    """

    in_state: TensorProductState
    out_operator: PauliTerm
    additional_expectations: Optional[list[list[int]]] = None

    def __init__(
        self,
        in_state: TensorProductState,
        out_operator: PauliTerm,
        additional_expectations: Optional[list[list[int]]] = None,
    ):
        object.__setattr__(self, "in_state", in_state)
        object.__setattr__(self, "out_operator", out_operator)
        object.__setattr__(self, "additional_expectations", additional_expectations)

    def _in_operator(self) -> PauliTerm:
        # Backwards compat
        pt = sI()
        for oneq_state in self.in_state.states:
            if oneq_state.label not in ["X", "Y", "Z"]:
                raise ValueError(f"Can't shim {oneq_state.label} into a pauli term. Use in_state.")
            if oneq_state.index != 0:
                raise ValueError(f"Can't shim {oneq_state} into a pauli term. Use in_state.")

            new_pt = pt * PauliTerm(op=oneq_state.label, index=oneq_state.qubit)
            pt = cast(PauliTerm, new_pt)

        return pt

    def __str__(self) -> str:
        return f"{self.in_state}→{self.out_operator.compact_str()}"

    def __repr__(self) -> str:
        return f"ExperimentSetting[{self}]"

    def serializable(self) -> str:
        return str(self)

    @classmethod
    def from_str(cls, s: str) -> "ExperimentSetting":
        """Opposite of str(expt)."""
        instr, outstr = s.split("→")
        return ExperimentSetting(
            in_state=TensorProductState.from_str(instr),
            out_operator=PauliTerm.from_compact_str(outstr),
        )
