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


