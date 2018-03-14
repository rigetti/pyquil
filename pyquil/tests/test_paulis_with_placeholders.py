##############################################################################
# Copyright 2016-2018 Rigetti Computing
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

import math
from functools import reduce
from itertools import product
from operator import mul

import numpy as np
import pytest
from six.moves import range

from pyquil.gates import RX, RZ, CNOT, H, X, PHASE
from pyquil.paulis import PauliTerm, PauliSum, exponential_map, exponentiate_commuting_pauli_sum, \
    ID, UnequalLengthWarning, exponentiate, trotterize, is_zero, check_commutation, commuting_sets, \
    term_with_coeff, sI, sX, sY, sZ, ZERO
from pyquil.quil import Program
from pyquil.quilatom import QubitPlaceholder


def test_simplify_terms():
    q = QubitPlaceholder.register(1)
    term = PauliTerm('Z', q[0]) * -1.0 * PauliTerm('Z', q[0])
    assert term.id() == ''
    assert term.coefficient == -1.0

    term = PauliTerm('Z', q[0]) + PauliTerm('Z', q[0], 1.0)
    assert str(term).startswith('(2+0j)*Z<QubitPlaceholder ')


def test_get_qubits():
    q = QubitPlaceholder.register(2)
    term = PauliTerm('Z', q[0]) * PauliTerm('X', q[1])
    assert term.get_qubits() == q

    q10 = QubitPlaceholder()
    sum_term = PauliTerm('X', q[0], 0.5) + 0.5j * PauliTerm('Y', q10) * PauliTerm('Y', q[0], 0.5j)
    assert sum_term.get_qubits() == [q[0], q10]


def test_simplify_term_single():
    q0, q1, q2 = QubitPlaceholder.register(3)
    term = (PauliTerm('Z', q0) * PauliTerm('I', q1)
            * PauliTerm('X', q2, 0.5j) * PauliTerm('Z', q0, 1.0))
    assert term.id() == 'X{}'.format(q2)
    assert term.coefficient == 0.5j


def test_simplify_term_xz():
    q0 = QubitPlaceholder()
    term1 = (-0.5 * PauliTerm('X', q0)) * (-1.0 * PauliTerm('Z', q0))
    term2 = -0.5 * PauliTerm('X', q0) * (-1.0) * PauliTerm('Z', q0)
    term3 = 0.5 * PauliTerm('X', q0) * PauliTerm('Z', q0)
    for term in [term1, term2, term3]:
        assert term.id() == 'Y{}'.format(q0)
        assert term.coefficient == -0.5j


def test_simplify_term_multindex():
    q0, q2 = QubitPlaceholder.register(2)
    term = (PauliTerm('X', q0, coefficient=-0.5) * PauliTerm('Z', q0, coefficient=-1.0)
            * PauliTerm('X', q2, 0.5))
    assert term.id(sort_ops=False) == 'Y{q0}X{q2}'.format(q0=q0, q2=q2)
    assert term.coefficient == -0.25j


def test_simplify_sum_terms():
    q0 = QubitPlaceholder()
    sum_term = PauliSum([PauliTerm('X', q0, 0.5), PauliTerm('Z', q0, 0.5j)])
    str_sum_term = str(sum_term + sum_term)
    assert (str_sum_term == '(1+0j)*X{q0} + 1j*Z{q0}'.format(q0=q0)
            or str_sum_term == '1j*Z{q0} + (1+0j)*X{q0}'.format(q0=q0))
    sum_term = PauliSum([PauliTerm('X', q0, 0.5), PauliTerm('X', q0, 0.5)])
    assert str(sum_term.simplify()) == '(1+0j)*X{q0}'.format(q0=q0)

    # test the simplify on multiplication
    sum_term = PauliSum([PauliTerm('X', q0, 0.5), PauliTerm('X', q0, 0.5)])
    assert str(sum_term * sum_term) == '(1+0j)*I'


def test_copy():
    q0, q1 = QubitPlaceholder.register(2)
    term = PauliTerm('X', q0, 0.5) * PauliTerm('X', q1, 0.5)
    new_term = term.copy()

    q2 = QubitPlaceholder()
    term = term * PauliTerm('X', q2, 0.5)
    new_term = new_term * PauliTerm('X', q2, 0.5)

    assert term == new_term  # value equality
    assert term is not new_term  # ref inequality
    assert term._ops is not new_term._ops

    term = PauliTerm('X', q0, 0.5) * PauliTerm('X', q1, 0.5)
    new_term = term * PauliTerm('X', q2, 0.5)
    assert term != new_term
    assert term is not new_term
    assert term._ops is not new_term._ops


def test_len():
    q0, q1 = QubitPlaceholder.register(2)
    term = PauliTerm("Z", q0, 1.0) * PauliTerm("Z", q1, 1.0)
    assert len(term) == 2


def test_sum_len():
    q0, q1 = QubitPlaceholder.register(2)
    pauli_sum = PauliTerm("Z", q0, 1.0) + PauliTerm("Z", q1, 1.0)
    assert len(pauli_sum) == 2


def test_enumerate():
    q0, q1, q5 = QubitPlaceholder.register(3)
    term = PauliTerm("Z", q0, 1.0) * PauliTerm("Z", q1, 1.0) * PauliTerm("X", q5, 5)
    position_op_pairs = [(q0, "Z"), (q1, "Z"), (q5, "X")]
    for key, val in term:
        assert (key, val) in position_op_pairs


def test_getitem():
    q = QubitPlaceholder.register(6)
    term = PauliTerm("Z", q[0], 1.0) * PauliTerm("Z", q[1], 1.0) * PauliTerm("X", q[5], 5)
    assert term[q[0]] == "Z"
    assert term[q[1]] == "Z"
    assert term[q[2]] == "I"
    assert term[q[3]] == "I"
    assert term[q[4]] == "I"
    assert term[q[5]] == "X"
    assert len(term) == 3


