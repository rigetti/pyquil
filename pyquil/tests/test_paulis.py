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

import pytest
from pyquil.paulis import (PauliTerm, PauliSum, exponential_map, ID, exponentiate,
                           trotterize, is_zero, check_commutation, commuting_sets, sZ, sX
                           )
from pyquil.quil import Program
from pyquil.gates import RX, RZ, CNOT, H, X, PHASE
import math
from itertools import product


def isclose(a, b, rel_tol=1e-10, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def compare_progs(test, reference):
    """
    compares two programs gate by gate, param by param
    """
    tinstr = test.actions
    rinstr = reference.actions
    assert len(tinstr) == len(rinstr)
    for idx in xrange(len(tinstr)):
        # check each field of the instruction object
        assert tinstr[idx][1].operator_name == rinstr[idx][1].operator_name
        assert len(tinstr[idx][1].parameters) == len(rinstr[idx][1].parameters)
        for pp in xrange(len(tinstr[idx][1].parameters)):
            cmp_val = isclose(tinstr[idx][1].parameters[pp], rinstr[idx][1].parameters[pp])
            assert cmp_val

        assert len(tinstr[idx][1].arguments) == len(rinstr[idx][1].arguments)
        for aa in xrange(len(tinstr[idx][1].arguments)):
            assert tinstr[idx][1].arguments[aa] == rinstr[idx][1].arguments[aa]


def test_simplify_terms():
    term = PauliTerm('Z', 0) * -1.0 * PauliTerm('Z', 0)
    assert term.id() == ''
    assert term.coefficient == -1.0

    term = PauliTerm('Z', 0) + PauliTerm('Z', 0, 1.0)
    assert str(term) == '2.0*Z0'


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
    term = 0.5 * ID
    assert term.id() == ''
    assert term.coefficient == 0.5


def test_simplify_term_id_3():
    s = 0.25 + 0.25 * ID
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
    term = PauliTerm('X', 0, coefficient=-0.5) * PauliTerm('Z', 0, coefficient=-1.0) \
           * PauliTerm('X', 2, 0.5)
    assert term.id() == 'Y0X2'
    assert term.coefficient == -0.25j


def test_simplify_sum_terms():
    sum_term = PauliSum([PauliTerm('X', 0, 0.5), PauliTerm('Z', 0, 0.5j)])
    assert str(sum_term + sum_term) == '1.0*X0 + 1j*Z0'
    sum_term = PauliSum([PauliTerm('X', 0, 0.5), PauliTerm('X', 0, 0.5)])
    assert str(sum_term.simplify()) == '1.0*X0'

    # test the simplify on multiplication
    sum_term = PauliSum([PauliTerm('X', 0, 0.5), PauliTerm('X', 0, 0.5)])
    assert str(sum_term * sum_term) == '1.0*I'


def test_len():
    term = PauliTerm("Z", 0, 1.0) * PauliTerm("Z", 1, 1.0)
    assert len(term) == 2


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
    assert term_1.id() == term_2.id()


def test_pauliop_inputs():
    with pytest.raises(AssertionError):
        PauliTerm('X', -2)


def test_pauli_sum():
    q_plus = 0.5 * PauliTerm('X', 0) + 0.5j * PauliTerm('Y', 0)
    the_sum = q_plus * PauliSum([PauliTerm('X', 0)])
    term_strings = map(lambda x: str(x), the_sum.terms)
    assert '0.5*I' in term_strings
    assert '(0.5+0j)*Z0' in term_strings
    assert len(term_strings) == 2
    assert len(the_sum.terms) == 2

    the_sum = q_plus * PauliTerm('X', 0)
    term_strings = map(lambda x: str(x), the_sum.terms)
    assert '0.5*I' in term_strings
    assert '(0.5+0j)*Z0' in term_strings
    assert len(term_strings) == 2
    assert len(the_sum.terms) == 2

    the_sum = PauliTerm('X', 0) * q_plus
    term_strings = map(lambda x: str(x), the_sum.terms)
    assert '0.5*I' in term_strings
    assert '(-0.5+0j)*Z0' in term_strings
    assert len(term_strings) == 2
    assert len(the_sum.terms) == 2


def test_ps_adds_pt_1():
    term = ID
    b = term + term
    assert str(b) == "2.0*I"
    assert str(b + term) == "3.0*I"
    assert str(term + b) == "3.0*I"


def test_ps_adds_pt_2():
    term = ID
    b = term + 1.0
    assert str(b) == "2.0*I"
    assert str(b + 1.0) == "3.0*I"
    assert str(1.0 + b) == "3.0*I"


def test_pauliterm_sub():
    assert str(sX(1) - 2.0) == str(sX(1) + -2.0)
    assert str(1.4 - sZ(1)) == str(1.4 + -1.0 * sZ(1))


def test_ps_sub():
    term = 3 * ID
    b = term - 1.0
    assert str(b) == "2.0*I"
    assert str(b - 1.0) == "1.0*I"
    assert str(1.0 - b) == "-1.0*I"


def test_zero_terms():
    term = PauliTerm("X", 0, 1.0) + PauliTerm("X", 0, -1.0) + \
           PauliTerm("Y", 0, 0.5)
    assert str(term) == "0.5*Y0"

    term = PauliTerm("X", 0, 1.0) + PauliTerm("X", 0, -1.0)
    assert str(term) == "0.0*I"
    assert len(term.terms) == 1

    term2 = term * PauliTerm("Z", 2, 0.5)
    assert str(term2) == "0.0*I"

    term3 = PauliTerm("Z", 2, 0.5) + term
    assert str(term3) == "0.5*Z2"

    term4 = PauliSum([])
    assert str(term4) == "0.0*I"

    term = PauliSum([PauliTerm("X", 0, 0.0), PauliTerm("Y", 1, 1.0) *
                     PauliTerm("Z", 2)])
    assert str(term) == "0.0*X0 + 1.0*Y1*Z2"
    term = term.simplify()
    assert str(term) == "1.0*Y1*Z2"


def test_exponentiate():
    # test rotation of single qubit
    generator = PauliTerm("Z", 0, 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst(RZ(2.0)(0))
    compare_progs(prog, result_prog)

    # testing general 2-circuit
    generator = PauliTerm("Z", 1, 1.0) * PauliTerm("Z", 0, 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst(CNOT(0, 1)).inst(RZ(2.0)(1)).inst(CNOT(0, 1))
    compare_progs(prog, result_prog)

    # testing change of basis position 0
    generator = PauliTerm("Z", 1, 1.0) * PauliTerm("X", 0, 1.0)
    param_prog = exponential_map(generator)
    prog = param_prog(1)
    result_prog = Program().inst([H(0), CNOT(0, 1), RZ(2.0)(1), CNOT(0, 1),
                                  H(0)])
    compare_progs(prog, result_prog)

    # testing change of basis position 1
    generator = PauliTerm("X", 1, 1.0) * PauliTerm("Z", 0, 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([H(1), CNOT(0, 1), RZ(2.0)(1), CNOT(0, 1),
                                  H(1)])
    compare_progs(prog, result_prog)

    # testing change of basis position 0
    generator = PauliTerm("Z", 1, 1.0) * PauliTerm("Y", 0, 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([RX(math.pi / 2.0)(0), CNOT(0, 1), RZ(2.0)(1),
                                  CNOT(0, 1), RX(-math.pi / 2)(0)])
    compare_progs(prog, result_prog)

    # testing change of basis position 1
    generator = PauliTerm("Y", 1, 1.0) * PauliTerm("Z", 0, 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([RX(math.pi / 2.0)(1), CNOT(0, 1), RZ(2.0)(1),
                                  CNOT(0, 1), RX(-math.pi / 2.0)(1)])
    compare_progs(prog, result_prog)

    # testing circuit for 3-terms with change of basis
    generator = PauliTerm("X", 2, 1.0) * PauliTerm("Y", 1, 1.0) * PauliTerm("Z", 0, 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([RX(math.pi / 2.0)(1), H(2), CNOT(0, 1),
                                  CNOT(1, 2), RZ(2.0)(2), CNOT(1, 2),
                                  CNOT(0, 1), RX(-math.pi / 2.0)(1), H(2)])
    compare_progs(prog, result_prog)

    # testing circuit for 3-terms non-sequential
    generator = PauliTerm("Y", 3, 1.0) * PauliTerm("Y", 2, 1.0) * PauliTerm("I", 1,
                                                                            1.0) * PauliTerm("Y", 0,
                                                                                             1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([RX(math.pi / 2.0)(0), RX(math.pi / 2.0)(2),
                                  RX(math.pi / 2.0)(3), CNOT(0, 2),
                                  CNOT(2, 3), RZ(2.0)(3), CNOT(2, 3),
                                  CNOT(0, 2), RX(-math.pi / 2.0)(0),
                                  RX(-math.pi / 2.0)(2), RX(-math.pi / 2.0)(3)])
    compare_progs(prog, result_prog)


def test_exponentiate_prog():
    ham = PauliTerm("Z", 0)
    result_prog = Program(RZ(2.0, 0))
    prog = exponentiate(ham)
    compare_progs(result_prog, prog)


def test_exponentiate_identity():
    generator = PauliTerm("I", 1, 0.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([X(0), PHASE(0)(0), X(0), PHASE(0)(0)])
    compare_progs(prog, result_prog)

    generator = PauliTerm("I", 1, 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([X(0), PHASE(-1.0)(0), X(0), PHASE(-1.0)(0)])
    compare_progs(prog, result_prog)

    generator = PauliTerm("I", 10, 0.08)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([X(0), PHASE(-0.08)(0), X(0), PHASE(-0.08)(0)])
    compare_progs(prog, result_prog)


def test_trotterize():
    term_one = PauliTerm("X", 0, 1.0)
    term_two = PauliTerm("Z", 0, 1.0)

    with pytest.raises(ValueError):
        trotterize(term_one, term_two, trotter_order=0)
    with pytest.raises(ValueError):
        trotterize(term_one, term_two, trotter_order=5)

    prog = trotterize(term_one, term_one)
    result_prog = Program().inst([H(0), RZ(2.0)(0), H(0), H(0),
                                  RZ(2.0)(0), H(0)])
    compare_progs(prog, result_prog)

    # trotter_order 1 steps 1
    prog = trotterize(term_one, term_two, trotter_steps=1)
    result_prog = Program().inst([H(0), RZ(2.0)(0), H(0), RZ(2.0)(0)])
    compare_progs(prog, result_prog)

    # trotter_order 1 steps 2
    prog  = trotterize(term_one, term_two, trotter_steps=2)
    result_prog = Program().inst([H(0), RZ(1.0)(0), H(0), RZ(1.0)(0),
                                  H(0), RZ(1.0)(0), H(0), RZ(1.0)(0)])
    compare_progs(prog, result_prog)

    # trotter_order 2 steps 1
    prog  = trotterize(term_one, term_two, trotter_order=2)
    result_prog = Program().inst([H(0), RZ(1.0)(0), H(0), RZ(2.0)(0),
                                  H(0), RZ(1.0)(0), H(0)])
    compare_progs(prog, result_prog)

    # trotter_order 2 steps 2
    prog = trotterize(term_one, term_two, trotter_order=2, trotter_steps=2)
    result_prog = Program().inst([H(0), RZ(0.5)(0), H(0), RZ(1.0)(0),
                                  H(0), RZ(0.5)(0), H(0),
                                  H(0), RZ(0.5)(0), H(0), RZ(1.0)(0),
                                  H(0), RZ(0.5)(0), H(0)])
    compare_progs(prog, result_prog)

    # trotter_order 3 steps 1
    prog = trotterize(term_one, term_two, trotter_order=3, trotter_steps=1)
    result_prog = Program().inst([H(0), RZ(14.0 / 24)(0), H(0), RZ(4.0 / 3.0)(0),
                                  H(0), RZ(1.5)(0), H(0), RZ(-4.0 / 3.0)(0),
                                  H(0), RZ(-2.0 / 24)(0), H(0), RZ(2.0)(0)])
    compare_progs(prog, result_prog)


def test_is_zeron():
    with pytest.raises(TypeError):
        is_zero(1)

    p_term = PauliTerm("X", 0)
    ps_term = p_term + PauliTerm("Z", 1)

    assert not is_zero(p_term)
    assert is_zero(p_term + -1 * p_term)
    assert not is_zero(ps_term)


def test_check_commutation():
    term1 = PauliTerm("X", 0) * PauliTerm("X", 1)
    term2 = PauliTerm("Y", 0) * PauliTerm("Y", 1)
    term3 = PauliTerm("Y", 0) * PauliTerm("Z", 2)
    # assert check_commutation(PauliSum([term1]), term2)
    assert check_commutation([term2], term3)
    assert check_commutation([term2], term3)
    assert not check_commutation([term1], term3)

    # more rigorous test.  Get all operators in Pauli group
    p_n_group = ("I", "X", "Y", "Z")
    pauli_list = list(product(p_n_group, repeat=3))
    pauli_ops = map(lambda x: zip(x, range(3)), pauli_list)
    pauli_ops_pq = []
    for op in pauli_ops:
        pauli_ops_pq.append(reduce(lambda x, y: x * PauliTerm(y[0], y[1]),
                                   op[1:],
                                   PauliTerm(op[0][0], op[0][1]))
                            )

    def commutator(t1, t2):
        return t1 * t2 + -1 * t2 * t1

    non_commuting_pairs = []
    commuting_pairs = []
    for x in xrange(len(pauli_ops_pq)):
        for y in xrange(x, len(pauli_ops_pq)):

            tmp_op = commutator(pauli_ops_pq[x], pauli_ops_pq[y])
            assert len(tmp_op.terms) == 1
            if tmp_op.terms[0].id() == '':
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
    commuting_sets(pauli_sum, 3)


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
