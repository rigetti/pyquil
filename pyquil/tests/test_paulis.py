#!/usr/bin/python
##############################################################################
# Copyright 2016-2017 Rigetti Computing
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
import warnings
from functools import reduce
from itertools import product
from operator import mul

import numpy as np
import pytest
from six.moves import range

from pyquil.gates import I, RX, RZ, CNOT, H, X, PHASE
from pyquil.paulis import PauliTerm, PauliSum, exponential_map, exponentiate_commuting_pauli_sum, \
    ID, UnequalLengthWarning, exponentiate, trotterize, is_zero, check_commutation, commuting_sets, \
    term_with_coeff, sI, sX, sY, sZ, ZERO, is_identity
from pyquil.quil import Program


def isclose(a, b, rel_tol=1e-10, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def test_init_pauli_term():
    with pytest.raises(ValueError):
        PauliTerm('X', 0, 'a')


def test_simplify_terms():
    term = PauliTerm('Z', 0) * -1.0 * PauliTerm('Z', 0)
    assert term.id() == ''
    assert term.coefficient == -1.0

    term = PauliTerm('Z', 0) + PauliTerm('Z', 0, 1.0)
    assert str(term) == '(2+0j)*Z0'


def test_get_qubits():
    term = PauliTerm('Z', 0) * PauliTerm('X', 1)
    assert term.get_qubits() == [0, 1]

    sum_term = PauliTerm('X', 0, 0.5) + 0.5j * PauliTerm('Y', 10) * PauliTerm('Y', 0, 0.5j)
    assert sum_term.get_qubits() == [0, 10]


def test_simplify_term_id_1():
    term = PauliTerm('I', 0, 0.5)
    assert term.id() == ''
    assert term.coefficient == 0.5


def test_simplify_term_id_2():
    term = 0.5 * ID()
    assert term.id() == ''
    assert term.coefficient == 0.5


def test_simplify_term_id_3():
    s = 0.25 + 0.25 * ID()
    terms = s.terms
    assert len(terms) == 1
    assert terms[0].id() == ''
    assert terms[0].coefficient == 0.5


def test_simplify_term_single():
    term = PauliTerm('Z', 0) * PauliTerm('I', 1) * PauliTerm('X', 2, 0.5j) * PauliTerm('Z', 0, 1.0)
    assert term.id() == 'X2'
    assert term.coefficient == 0.5j


def test_simplify_term_xz():
    term1 = (-0.5 * PauliTerm('X', 0)) * (-1.0 * PauliTerm('Z', 0))
    term2 = -0.5 * PauliTerm('X', 0) * (-1.0) * PauliTerm('Z', 0)
    term3 = 0.5 * PauliTerm('X', 0) * PauliTerm('Z', 0)
    for term in [term1, term2, term3]:
        assert term.id() == 'Y0'
        assert term.coefficient == -0.5j


def test_simplify_term_multindex():
    term = (PauliTerm('X', 0, coefficient=-0.5)
            * PauliTerm('Z', 0, coefficient=-1.0) * PauliTerm('X', 2, 0.5))
    assert term.id(sort_ops=False) == 'Y0X2'
    assert term.coefficient == -0.25j


def test_simplify_sum_terms():
    sum_term = PauliSum([PauliTerm('X', 0, 0.5), PauliTerm('Z', 0, 0.5j)])
    str_sum_term = str(sum_term + sum_term)
    assert str_sum_term == '(1+0j)*X0 + 1j*Z0' or str_sum_term == '1j*Z0 + (1+0j)*X0'
    sum_term = PauliSum([PauliTerm('X', 0, 0.5), PauliTerm('X', 0, 0.5)])
    assert str(sum_term.simplify()) == '(1+0j)*X0'

    # test the simplify on multiplication
    sum_term = PauliSum([PauliTerm('X', 0, 0.5), PauliTerm('X', 0, 0.5)])
    assert str(sum_term * sum_term) == '(1+0j)*I'


def test_copy():
    term = PauliTerm('X', 0, 0.5) * PauliTerm('X', 1, 0.5)
    new_term = term.copy()

    term = term * PauliTerm('X', 2, 0.5)
    new_term = new_term * PauliTerm('X', 2, 0.5)

    assert term == new_term  # value equality
    assert term is not new_term  # ref inequality
    assert term._ops is not new_term._ops

    term = PauliTerm('X', 0, 0.5) * PauliTerm('X', 1, 0.5)
    new_term = term * PauliTerm('X', 2, 0.5)
    assert term != new_term
    assert term is not new_term
    assert term._ops is not new_term._ops


def test_len():
    term = PauliTerm("Z", 0, 1.0) * PauliTerm("Z", 1, 1.0)
    assert len(term) == 2


def test_sum_len():
    pauli_sum = PauliTerm("Z", 0, 1.0) + PauliTerm("Z", 1, 1.0)
    assert len(pauli_sum) == 2


def test_enumerate():
    term = PauliTerm("Z", 0, 1.0) * PauliTerm("Z", 1, 1.0) * PauliTerm("X", 5, 5)
    position_op_pairs = [(0, "Z"), (1, "Z"), (5, "X")]
    for key, val in term:
        assert (key, val) in position_op_pairs


def test_getitem():
    term = PauliTerm("Z", 0, 1.0) * PauliTerm("Z", 1, 1.0) * PauliTerm("X", 5, 5)
    assert term[0] == "Z"
    assert term[1] == "Z"
    assert term[2] == "I"
    assert term[3] == "I"
    assert term[4] == "I"
    assert term[5] == "X"
    assert len(term) == 3


def test_ids():
    term_1 = PauliTerm("Z", 0, 1.0) * PauliTerm("Z", 1, 1.0) * PauliTerm("X", 5, 5)
    term_2 = PauliTerm("X", 5, 5) * PauliTerm("Z", 0, 1.0) * PauliTerm("Z", 1, 1.0)
    with pytest.warns(FutureWarning) as w:
        assert term_1.id() == term_2.id()
    assert 'should be avoided' in str(w[0])


def test_ids_no_sort():
    term_1 = PauliTerm("Z", 0, 1.0) * PauliTerm("Z", 1, 1.0) * PauliTerm("X", 5, 5)
    term_2 = PauliTerm("X", 5, 5) * PauliTerm("Z", 0, 1.0) * PauliTerm("Z", 1, 1.0)
    assert term_1.id(sort_ops=False) == 'Z0Z1X5'
    assert term_2.id(sort_ops=False) == 'X5Z0Z1'


def test_operations_as_set():
    term_1 = PauliTerm("Z", 0, 1.0) * PauliTerm("Z", 1, 1.0) * PauliTerm("X", 5, 5)
    term_2 = PauliTerm("X", 5, 5) * PauliTerm("Z", 0, 1.0) * PauliTerm("Z", 1, 1.0)
    assert term_1.operations_as_set() == term_2.operations_as_set()


def test_pauliop_inputs():
    with pytest.raises(AssertionError):
        PauliTerm('X', -2)


def test_pauli_sum():
    q_plus = 0.5 * PauliTerm('X', 0) + 0.5j * PauliTerm('Y', 0)
    the_sum = q_plus * PauliSum([PauliTerm('X', 0)])
    term_strings = [str(x) for x in the_sum.terms]
    assert '(0.5+0j)*I' in term_strings
    assert '(0.5+0j)*Z0' in term_strings
    assert len(term_strings) == 2
    assert len(the_sum.terms) == 2

    the_sum = q_plus * PauliTerm('X', 0)
    term_strings = [str(x) for x in the_sum.terms]
    assert '(0.5+0j)*I' in term_strings
    assert '(0.5+0j)*Z0' in term_strings
    assert len(term_strings) == 2
    assert len(the_sum.terms) == 2

    the_sum = PauliTerm('X', 0) * q_plus
    term_strings = [str(x) for x in the_sum.terms]
    assert '(0.5+0j)*I' in term_strings
    assert '(-0.5+0j)*Z0' in term_strings
    assert len(term_strings) == 2
    assert len(the_sum.terms) == 2

    with pytest.raises(ValueError):
        _ = PauliSum(sI(0))
    with pytest.raises(ValueError):
        _ = PauliSum([1, 1, 1, 1])
    with pytest.raises(ValueError):
        _ = the_sum * []


def test_ps_adds_pt_1():
    term = ID()
    b = term + term
    assert str(b) == "(2+0j)*I"
    assert str(b + term) == "(3+0j)*I"
    assert str(term + b) == "(3+0j)*I"


def test_ps_adds_pt_2():
    term = ID()
    b = term + 1.0
    assert str(b) == "(2+0j)*I"
    assert str(b + 1.0) == "(3+0j)*I"
    assert str(1.0 + b) == "(3+0j)*I"
    b = sX(0) + 1.0
    assert str(b) == "(1+0j)*X0 + (1+0j)*I"
    b = 1.0 + sX(0)
    assert str(b) == "(1+0j)*I + (1+0j)*X0"


def test_pauliterm_sub():
    assert str(sX(1) - 2.0) == str(sX(1) + -2.0)
    assert str(1.4 - sZ(1)) == str(1.4 + -1.0 * sZ(1))


def test_ps_sub():
    term = 3 * ID()
    b = term - 1.0
    assert str(b) == "(2+0j)*I"
    assert str(b - 1.0) == "(1+0j)*I"
    assert str(1.0 - b) == "(-1+0j)*I"
    b = 1.0 - sX(0)
    assert str(b) == "(1+0j)*I + (-1+0j)*X0"
    b = sX(0) - 1.0
    assert str(b) == "(1+0j)*X0 + (-1+0j)*I"


def test_zero_terms():
    term = PauliTerm("X", 0, 1.0) + PauliTerm("X", 0, -1.0) + PauliTerm("Y", 0, 0.5)
    assert str(term) == "(0.5+0j)*Y0"

    term = PauliTerm("X", 0, 1.0) + PauliTerm("X", 0, -1.0)
    assert str(term) == "0j*I"
    assert len(term.terms) == 1

    term2 = term * PauliTerm("Z", 2, 0.5)
    assert str(term2) == "0j*I"

    term3 = PauliTerm("Z", 2, 0.5) + term
    assert str(term3) == "(0.5+0j)*Z2"

    term4 = PauliSum([])
    assert str(term4) == "0j*I"

    term = PauliSum([PauliTerm("X", 0, 0.0), PauliTerm("Y", 1, 1.0) * PauliTerm("Z", 2)])
    assert str(term) == "0j*X0 + (1+0j)*Y1*Z2"
    term = term.simplify()
    assert str(term) == "(1+0j)*Y1*Z2"


def test_exponentiate_1():
    # test rotation of single qubit
    generator = PauliTerm("Z", 0, 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst(RZ(2.0, 0))
    assert prog == result_prog


def test_exponentiate_2():
    # testing general 2-circuit
    generator = PauliTerm("Z", 0, 1.0) * PauliTerm("Z", 1, 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst(CNOT(0, 1)).inst(RZ(2.0, 1)).inst(CNOT(0, 1))
    assert prog == result_prog


def test_exponentiate_bp0_ZX():
    # testing change of basis position 0
    generator = PauliTerm("X", 0, 1.0) * PauliTerm("Z", 1, 1.0)
    param_prog = exponential_map(generator)
    prog = param_prog(1)
    result_prog = Program().inst([H(0), CNOT(0, 1), RZ(2.0, 1), CNOT(0, 1), H(0)])
    assert prog == result_prog


def test_exponentiate_bp1_XZ():
    # testing change of basis position 1
    generator = PauliTerm("Z", 0, 1.0) * PauliTerm("X", 1, 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([H(1), CNOT(0, 1), RZ(2.0, 1), CNOT(0, 1), H(1)])
    assert prog == result_prog


def test_exponentiate_bp0_ZY():
    # testing change of basis position 0
    generator = PauliTerm("Y", 0, 1.0) * PauliTerm("Z", 1, 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([RX(math.pi / 2.0, 0), CNOT(0, 1), RZ(2.0, qubit=1),
                                  CNOT(0, 1), RX(-math.pi / 2, 0)])
    assert prog == result_prog


def test_exponentiate_bp1_YZ():
    # testing change of basis position 1
    generator = PauliTerm("Z", 0, 1.0) * PauliTerm("Y", 1, 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([RX(math.pi / 2.0, 1), CNOT(0, 1),
                                  RZ(2.0, 1), CNOT(0, 1), RX(-math.pi / 2.0, 1)])
    assert prog == result_prog


def test_exponentiate_3cob():
    # testing circuit for 3-terms with change of basis
    generator = PauliTerm("Z", 0, 1.0) * PauliTerm("Y", 1, 1.0) * PauliTerm("X", 2, 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([RX(math.pi / 2.0, 1), H(2), CNOT(0, 1),
                                  CNOT(1, 2), RZ(2.0, 2), CNOT(1, 2),
                                  CNOT(0, 1), RX(-math.pi / 2.0, 1), H(2)])
    assert prog == result_prog


def test_exponentiate_3ns():
    # testing circuit for 3-terms non-sequential
    generator = (PauliTerm("Y", 0, 1.0)
                 * PauliTerm("I", 1, 1.0)
                 * PauliTerm("Y", 2, 1.0)
                 * PauliTerm("Y", 3, 1.0))
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([RX(math.pi / 2.0, 0), RX(math.pi / 2.0, 2),
                                  RX(math.pi / 2.0, 3), CNOT(0, 2),
                                  CNOT(2, 3), RZ(2.0, 3), CNOT(2, 3),
                                  CNOT(0, 2), RX(-math.pi / 2.0, 0),
                                  RX(-math.pi / 2.0, 2), RX(-math.pi / 2.0, 3)])
    assert prog == result_prog


def test_exponentiate_commuting_pauli_sum():
    pauli_sum = PauliSum([PauliTerm('Z', 0, 0.5), PauliTerm('Z', 1, 0.5)])
    prog = Program().inst(RZ(1., 0)).inst(RZ(1., 1))
    result_prog = exponentiate_commuting_pauli_sum(pauli_sum)(1.)
    assert prog == result_prog


def test_exponentiate_prog():
    ham = PauliTerm("Z", 0)
    result_prog = Program(RZ(2.0, 0))
    prog = exponentiate(ham)
    assert prog == result_prog


def test_exponentiate_identity():
    generator = PauliTerm("I", 1, 0.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program()
    assert prog == result_prog

    generator = PauliTerm("I", 1, 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([X(0), PHASE(-1.0, 0), X(0), PHASE(-1.0, 0)])
    assert prog == result_prog

    generator = PauliTerm("I", 10, 0.08)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([X(0), PHASE(-0.08, 0), X(0), PHASE(-0.08, 0)])
    assert prog == result_prog


def test_trotterize():
    term_one = PauliTerm("X", 0, 1.0)
    term_two = PauliTerm("Z", 0, 1.0)

    with pytest.raises(ValueError):
        trotterize(term_one, term_two, trotter_order=0)
    with pytest.raises(ValueError):
        trotterize(term_one, term_two, trotter_order=5)

    prog = trotterize(term_one, term_one)
    result_prog = Program().inst([H(0), RZ(2.0, 0), H(0), H(0),
                                  RZ(2.0, 0), H(0)])
    assert prog == result_prog

    # trotter_order 1 steps 1
    prog = trotterize(term_one, term_two, trotter_steps=1)
    result_prog = Program().inst([H(0), RZ(2.0, 0), H(0), RZ(2.0, 0)])
    assert prog == result_prog

    # trotter_order 1 steps 2
    prog = trotterize(term_one, term_two, trotter_steps=2)
    result_prog = Program().inst([H(0), RZ(1.0, 0), H(0), RZ(1.0, 0),
                                  H(0), RZ(1.0, 0), H(0), RZ(1.0, 0)])
    assert prog == result_prog

    # trotter_order 2 steps 1
    prog = trotterize(term_one, term_two, trotter_order=2)
    result_prog = Program().inst([H(0), RZ(1.0, 0), H(0), RZ(2.0, 0),
                                  H(0), RZ(1.0, 0), H(0)])
    assert prog == result_prog

    # trotter_order 2 steps 2
    prog = trotterize(term_one, term_two, trotter_order=2, trotter_steps=2)
    result_prog = Program().inst([H(0), RZ(0.5, 0), H(0), RZ(1.0, 0),
                                  H(0), RZ(0.5, 0), H(0),
                                  H(0), RZ(0.5, 0), H(0), RZ(1.0, 0),
                                  H(0), RZ(0.5, 0), H(0)])
    assert prog == result_prog

    # trotter_order 3 steps 1
    prog = trotterize(term_one, term_two, trotter_order=3, trotter_steps=1)
    result_prog = Program().inst([H(0), RZ(14.0 / 24, 0), H(0), RZ(4.0 / 3.0, 0),
                                  H(0), RZ(1.5, 0), H(0), RZ(-4.0 / 3.0, 0),
                                  H(0), RZ(-2.0 / 24, 0), H(0), RZ(2.0, 0)])
    assert prog == result_prog


def test_is_zero():
    with pytest.raises(TypeError):
        is_zero(1)

    p_term = PauliTerm("X", 0)
    ps_term = p_term + PauliTerm("Z", 1)
    id_term = PauliTerm("I", 0)

    assert not is_zero(p_term)
    assert is_zero(p_term + -1 * p_term)
    assert not is_zero(ps_term)
    assert not is_zero(id_term)


def test_check_commutation():
    term1 = PauliTerm("X", 0) * PauliTerm("X", 1)
    term2 = PauliTerm("Y", 0) * PauliTerm("Y", 1)
    term3 = PauliTerm("Y", 0) * PauliTerm("Z", 2)
    # assert check_commutation(PauliSum([term1]), term2)
    assert check_commutation([term2], term3)
    assert check_commutation([term2], term3)
    assert not check_commutation([term1], term3)


def _commutator(t1, t2):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message=r"The term .+ will be combined with .+, "
                                        r"but they have different orders of operations.*",
                                category=UserWarning)
        return t1 * t2 + -1 * t2 * t1


def test_check_commutation_rigorous():
    # more rigorous test.  Get all operators in Pauli group
    p_n_group = ("I", "X", "Y", "Z")
    pauli_list = list(product(p_n_group, repeat=3))
    pauli_ops = [list(zip(x, range(3))) for x in pauli_list]
    pauli_ops_pq = [reduce(mul, (PauliTerm(*x) for x in op)) for op in pauli_ops]

    non_commuting_pairs = []
    commuting_pairs = []
    for x in range(len(pauli_ops_pq)):
        for y in range(x, len(pauli_ops_pq)):

            tmp_op = _commutator(pauli_ops_pq[x], pauli_ops_pq[y])
            assert len(tmp_op.terms) == 1
            if is_zero(tmp_op.terms[0]):
                commuting_pairs.append((pauli_ops_pq[x], pauli_ops_pq[y]))
            else:
                non_commuting_pairs.append((pauli_ops_pq[x], pauli_ops_pq[y]))

    # now that we have our sets let's check against our code.
    for t1, t2 in non_commuting_pairs:
        assert not check_commutation([t1], t2)

    for t1, t2 in commuting_pairs:
        assert check_commutation([t1], t2)


def test_commuting_sets():
    term1 = PauliTerm("X", 0) * PauliTerm("X", 1)
    term2 = PauliTerm("Y", 0) * PauliTerm("Y", 1)
    term3 = PauliTerm("Y", 0) * PauliTerm("Z", 2)
    pauli_sum = term1 + term2 + term3
    commuting_sets(pauli_sum)


def test_paulisum_iteration():
    term_list = [sX(2), sZ(4)]
    pauli_sum = sum(term_list)
    for ii, term in enumerate(pauli_sum):
        assert term_list[ii] == term


def test_paulisum_indexing():
    pauli_sum = 0.5 * sX(0) + 0.1 * sZ(1)
    assert pauli_sum[0] == 0.5 * sX(0)
    for ii, term in enumerate(pauli_sum.terms):
        assert pauli_sum[ii] == term


def test_term_powers():
    for qubit_id in range(2):
        pauli_terms = [sI(qubit_id), sX(qubit_id), sY(qubit_id), sZ(qubit_id)]
        for pauli_term in pauli_terms:
            assert pauli_term ** 0 == sI(qubit_id)
            assert pauli_term ** 1 == pauli_term
            assert pauli_term ** 2 == sI(qubit_id)
            assert pauli_term ** 3 == pauli_term
    with pytest.raises(ValueError):
        pauli_terms[0] ** -1
    # Test to make sure large powers can be computed
    (PauliTerm('X', 0, 2) * PauliTerm('Y', 0, 2)) ** 400


def test_sum_power():
    pauli_sum = (sY(0) - sX(0)) * (1.0 / np.sqrt(2))
    assert pauli_sum ** 2 == PauliSum([sI(0)])
    with pytest.raises(ValueError):
        _ = pauli_sum ** -1
    pauli_sum = sI(0) + sI(1)
    assert pauli_sum ** 0 == sI(0)
    # Test to make sure large powers can be computed
    pauli_sum ** 400


def test_term_equality():
    with pytest.raises(TypeError):
        sI(0) != 0
    assert sI(0) == sI(0)
    assert PauliTerm('X', 10, 1 + 1.j) == PauliTerm('X', 10, 1 + 1.j)
    assert PauliTerm('X', 10, 1 + 1.j) + PauliTerm('X', 10, 1 + 1.j) != PauliTerm('X', 10, 1 + 1.j)
    assert PauliTerm('X', 10, 1 + 1.j) != PauliTerm('X', 10, 1 + 1.j) + PauliTerm('X', 10, 1 + 1.j)


def test_term_with_coeff():
    assert PauliTerm('X', 0, 1.j) == term_with_coeff(sX(0), 1.j)
    assert PauliTerm('X', 0, -1.0) == term_with_coeff(sX(0), -1)
    with pytest.raises(ValueError):
        term_with_coeff(sI(0), None)


def test_sum_equality():
    pauli_sum = sY(0) - sX(0)
    assert pauli_sum != 2 * pauli_sum
    with pytest.warns(UnequalLengthWarning):
        assert pauli_sum != pauli_sum + sZ(0)
    with pytest.warns(UnequalLengthWarning):
        assert pauli_sum + sZ(0) != pauli_sum
    assert pauli_sum != sY(1) - sX(1)
    assert pauli_sum == -1.0 * sX(0) + sY(0)
    assert pauli_sum == pauli_sum * 1.0
    with pytest.raises(TypeError):
        assert pauli_sum != 0


def test_zero_term():
    qubit_id = 0
    coefficient = 10
    ps = sI(qubit_id) + sX(qubit_id)
    assert coefficient * ZERO() == ZERO()
    assert ZERO() * coefficient == ZERO()
    assert ZERO() * ID() == ZERO()
    assert ZERO() + ID() == ID()
    assert ZERO() + ps == ps
    assert ps + ZERO() == ps


def test_from_list():
    terms_list = [("X", 0), ("Y", 1), ("Z", 5)]
    term = reduce(lambda x, y: x * y, [PauliTerm(*x) for x in terms_list])

    pterm = PauliTerm.from_list(terms_list)
    assert pterm == term

    with pytest.raises(ValueError):
        # terms are not on disjoint qubits
        pterm = PauliTerm.from_list([("X", 0), ("Y", 0)])


def test_ordered():
    term = sZ(3) * sZ(2) * sZ(1)
    prog = exponential_map(term)(0.5)
    assert prog.out() == "CNOT 3 2\n" \
                         "CNOT 2 1\n" \
                         "RZ(1.0) 1\n" \
                         "CNOT 2 1\n" \
                         "CNOT 3 2\n"


def test_numpy_integer_types():
    idx_np, = np.arange(1, dtype=np.int64)
    assert isinstance(idx_np, np.int64)
    # on python 3 this fails unless explicitly allowing for numpy integer types
    PauliTerm("X", idx_np)


def test_simplify():
    t1 = sZ(0) * sZ(1)
    t2 = sZ(0) * sZ(1)
    assert (t1 + t2) == 2 * sZ(0) * sZ(1)


def test_dont_simplify():
    t1 = sZ(0) * sZ(1)
    t2 = sZ(2) * sZ(3)
    assert (t1 + t2) != 2 * sZ(0) * sZ(1)


def test_simplify_warning():
    t1 = sZ(0) * sZ(1)
    t2 = sZ(1) * sZ(0)
    with pytest.warns(UserWarning) as e:
        tsum = t1 + t2

    assert tsum == 2 * sZ(0) * sZ(1)
    assert str(e[0].message).startswith('The term Z1Z0 will be combined with Z0Z1')


def test_pauli_string():
    p = PauliTerm("X", 1) * PauliTerm("Z", 5)
    assert p.pauli_string([1, 5]) == "XZ"
    assert p.pauli_string([1]) == "X"
    assert p.pauli_string([5]) == "Z"
    assert p.pauli_string([5, 6]) == "ZI"
    assert p.pauli_string([0, 1]) == "IX"


def test_is_identity():
    pt1 = -1.5j * sI(2)
    pt2 = 1.5 * sX(1) * sZ(2)

    assert is_identity(pt1)
    assert is_identity(pt2 + (-1 * pt2) + sI(0))
    assert not is_identity(0 * pt1)
    assert not is_identity(pt2 + (-1 * pt2))
