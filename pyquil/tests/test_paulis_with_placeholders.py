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
import warnings
from functools import reduce
from itertools import product
from operator import mul

import numpy as np
import pytest
import re
from six.moves import range

from pyquil.gates import RX, RZ, CNOT, H, X, PHASE
from pyquil.paulis import PauliTerm, PauliSum, exponential_map, exponentiate_commuting_pauli_sum, \
    ID, UnequalLengthWarning, exponentiate, trotterize, is_zero, check_commutation, commuting_sets, \
    term_with_coeff, sI, sX, sY, sZ, ZERO, is_identity
from pyquil.quil import Program, address_qubits, get_default_qubit_mapping
from pyquil.quilatom import QubitPlaceholder, Qubit


def test_simplify_terms():
    q = QubitPlaceholder.register(1)
    term = PauliTerm('Z', q[0]) * -1.0 * PauliTerm('Z', q[0])
    assert term.id() == ''
    assert term.coefficient == -1.0

    term = PauliTerm('Z', q[0]) + PauliTerm('Z', q[0], 1.0)
    assert str(term).startswith('(2+0j)*Zq')


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


def test_ids():
    q = QubitPlaceholder.register(6)
    term_1 = PauliTerm("Z", q[0], 1.0) * PauliTerm("Z", q[1], 1.0) * PauliTerm("X", q[5], 5)
    term_2 = PauliTerm("X", q[5], 5) * PauliTerm("Z", q[0], 1.0) * PauliTerm("Z", q[1], 1.0)
    with pytest.raises(TypeError):
        # Not sortable
        t = term_1.id() == term_2.id()


def test_ids_no_sort():
    q = QubitPlaceholder.register(6)
    term_1 = PauliTerm("Z", q[0], 1.0) * PauliTerm("Z", q[1], 1.0) * PauliTerm("X", q[5], 5)
    term_2 = PauliTerm("X", q[5], 5) * PauliTerm("Z", q[0], 1.0) * PauliTerm("Z", q[1], 1.0)
    assert re.match('Z.+Z.+X.+', term_1.id(sort_ops=False))
    assert re.match('X.+Z.+Z.+', term_2.id(sort_ops=False))


def test_operations_as_set():
    q = QubitPlaceholder.register(6)
    term_1 = PauliTerm("Z", q[0], 1.0) * PauliTerm("Z", q[1], 1.0) * PauliTerm("X", q[5], 5)
    term_2 = PauliTerm("X", q[5], 5) * PauliTerm("Z", q[0], 1.0) * PauliTerm("Z", q[1], 1.0)
    assert term_1.operations_as_set() == term_2.operations_as_set()


def test_pauli_sum():
    q = QubitPlaceholder.register(8)
    q_plus = 0.5 * PauliTerm('X', q[0]) + 0.5j * PauliTerm('Y', q[0])
    the_sum = q_plus * PauliSum([PauliTerm('X', q[0])])
    term_strings = [str(x) for x in the_sum.terms]
    assert '(0.5+0j)*I' in term_strings
    assert len(term_strings) == 2
    assert len(the_sum.terms) == 2

    the_sum = q_plus * PauliTerm('X', q[0])
    term_strings = [str(x) for x in the_sum.terms]
    assert '(0.5+0j)*I' in term_strings
    assert len(term_strings) == 2
    assert len(the_sum.terms) == 2

    the_sum = PauliTerm('X', q[0]) * q_plus
    term_strings = [str(x) for x in the_sum.terms]
    assert '(0.5+0j)*I' in term_strings
    assert len(term_strings) == 2
    assert len(the_sum.terms) == 2

    with pytest.raises(ValueError):
        _ = PauliSum(sI(q[0]))
    with pytest.raises(ValueError):
        _ = PauliSum([1, 1, 1, 1])
    with pytest.raises(ValueError):
        _ = the_sum * []


def test_pauliterm_sub():
    q = QubitPlaceholder.register(8)
    assert str(sX(q[1]) - 2.0) == str(sX(q[1]) + -2.0)
    assert str(1.4 - sZ(q[1])) == str(1.4 + -1.0 * sZ(q[1]))


def test_ps_sub():
    q0 = QubitPlaceholder()
    term = 3 * ID()
    b = term - 1.0
    assert str(b) == "(2+0j)*I"
    assert str(b - 1.0) == "(1+0j)*I"
    assert str(1.0 - b) == "(-1+0j)*I"
    b = 1.0 - sX(q0)
    assert re.match(r"\(1\+0j\)\*I \+ \(-1\+0j\)\*Xq\d+", str(b))
    b = sX(q0) - 1.0
    assert re.match(r"\(1\+0j\)\*Xq\d+ \+ \(-1\+0j\)\*I", str(b))


def test_exponentiate_1():
    # test rotation of single qubit
    q = QubitPlaceholder.register(8)
    generator = PauliTerm("Z", q[0], 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst(RZ(2.0, q[0]))
    assert address_qubits(prog) == address_qubits(result_prog)


def test_exponentiate_2():
    # testing general 2-circuit
    q = QubitPlaceholder.register(8)
    generator = PauliTerm("Z", q[0], 1.0) * PauliTerm("Z", q[1], 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst(CNOT(q[0], q[1])).inst(RZ(2.0, q[1])).inst(CNOT(q[0], q[1]))

    mapping = get_default_qubit_mapping(prog)
    assert address_qubits(prog, mapping) == address_qubits(result_prog, mapping)


def test_exponentiate_bp0_ZX():
    q = QubitPlaceholder.register(8)
    # testing change of basis position 0
    generator = PauliTerm("X", q[0], 1.0) * PauliTerm("Z", q[1], 1.0)
    param_prog = exponential_map(generator)
    prog = param_prog(1)
    result_prog = Program().inst(
        [H(q[0]), CNOT(q[0], q[1]), RZ(2.0, q[1]), CNOT(q[0], q[1]), H(q[0])])
    assert address_qubits(prog) == address_qubits(result_prog)


def test_exponentiate_bp1_XZ():
    # testing change of basis position 1
    q = QubitPlaceholder.register(8)
    generator = PauliTerm("Z", q[0], 1.0) * PauliTerm("X", q[1], 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst(
        [H(q[1]), CNOT(q[0], q[1]), RZ(2.0, q[1]), CNOT(q[0], q[1]), H(q[1])])
    assert address_qubits(prog) == address_qubits(result_prog)


def test_exponentiate_bp0_ZY():
    # testing change of basis position 0
    q = QubitPlaceholder.register(8)
    generator = PauliTerm("Y", q[0], 1.0) * PauliTerm("Z", q[1], 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([RX(math.pi / 2.0, q[0]), CNOT(q[0], q[1]), RZ(2.0, q[1]),
                                  CNOT(q[0], q[1]), RX(-math.pi / 2, q[0])])
    assert address_qubits(prog) == address_qubits(result_prog)


def test_exponentiate_bp1_YZ():
    q = QubitPlaceholder.register(8)
    # testing change of basis position 1
    generator = PauliTerm("Z", q[0], 1.0) * PauliTerm("Y", q[1], 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([RX(math.pi / 2.0, q[1]), CNOT(q[0], q[1]),
                                  RZ(2.0, q[1]), CNOT(q[0], q[1]), RX(-math.pi / 2.0, q[1])])
    assert address_qubits(prog) == address_qubits(result_prog)


def test_exponentiate_3cob():
    # testing circuit for 3-terms with change of basis
    q = QubitPlaceholder.register(8)
    generator = PauliTerm("Z", q[0], 1.0) * PauliTerm("Y", q[1], 1.0) * PauliTerm("X", q[2], 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([RX(math.pi / 2.0, q[1]), H(q[2]), CNOT(q[0], q[1]),
                                  CNOT(q[1], q[2]), RZ(2.0, q[2]), CNOT(q[1], q[2]),
                                  CNOT(q[0], q[1]), RX(-math.pi / 2.0, q[1]), H(q[2])])
    assert address_qubits(prog) == address_qubits(result_prog)


def test_exponentiate_3ns():
    # testing circuit for 3-terms non-sequential
    q = QubitPlaceholder.register(8)
    generator = (PauliTerm("Y", q[0], 1.0)
                 * PauliTerm("I", q[1], 1.0)
                 * PauliTerm("Y", q[2], 1.0)
                 * PauliTerm("Y", q[3], 1.0))
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([RX(math.pi / 2.0, q[0]), RX(math.pi / 2.0, q[2]),
                                  RX(math.pi / 2.0, q[3]), CNOT(q[0], q[2]),
                                  CNOT(q[2], q[3]), RZ(2.0, q[3]), CNOT(q[2], q[3]),
                                  CNOT(q[0], q[2]), RX(-math.pi / 2.0, q[0]),
                                  RX(-math.pi / 2.0, q[2]), RX(-math.pi / 2.0, q[3])])
    assert address_qubits(prog) == address_qubits(result_prog)


def test_exponentiate_commuting_pauli_sum():
    q = QubitPlaceholder.register(8)
    pauli_sum = PauliSum([PauliTerm('Z', q[0], 0.5), PauliTerm('Z', q[1], 0.5)])
    prog = Program().inst(RZ(1., q[0])).inst(RZ(1., q[1]))
    result_prog = exponentiate_commuting_pauli_sum(pauli_sum)(1.)
    assert address_qubits(prog) == address_qubits(result_prog)


def test_exponentiate_prog():
    q = QubitPlaceholder.register(8)
    ham = PauliTerm("Z", q[0])
    result_prog = Program(RZ(2.0, q[0]))
    prog = exponentiate(ham)
    assert address_qubits(prog) == address_qubits(result_prog)


def test_exponentiate_identity():
    q = QubitPlaceholder.register(11)
    mapping = {qp: Qubit(i) for i, qp in enumerate(q)}
    generator = PauliTerm("I", q[1], 0.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([X(q[0]), PHASE(-0.0, q[0]), X(q[0]), PHASE(-0.0, q[0])])
    assert address_qubits(prog, mapping) == address_qubits(result_prog, mapping)

    generator = PauliTerm("I", q[1], 1.0)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([X(q[0]), PHASE(-1.0, q[0]), X(q[0]), PHASE(-1.0, q[0])])
    assert address_qubits(prog, mapping) == address_qubits(result_prog, mapping)

    generator = PauliTerm("I", q[10], 0.08)
    para_prog = exponential_map(generator)
    prog = para_prog(1)
    result_prog = Program().inst([X(q[0]), PHASE(-0.08, q[0]), X(q[0]), PHASE(-0.08, q[0])])
    assert address_qubits(prog, mapping) == address_qubits(result_prog, mapping)


def test_trotterize():
    q = QubitPlaceholder.register(8)
    term_one = PauliTerm("X", q[0], 1.0)
    term_two = PauliTerm("Z", q[0], 1.0)

    with pytest.raises(ValueError):
        trotterize(term_one, term_two, trotter_order=0)
    with pytest.raises(ValueError):
        trotterize(term_one, term_two, trotter_order=5)

    prog = trotterize(term_one, term_one)
    result_prog = Program().inst([H(q[0]), RZ(2.0, q[0]), H(q[0]), H(q[0]),
                                  RZ(2.0, q[0]), H(q[0])])
    assert address_qubits(prog) == address_qubits(result_prog)

    # trotter_order 1 steps 1
    prog = trotterize(term_one, term_two, trotter_steps=1)
    result_prog = Program().inst([H(q[0]), RZ(2.0, q[0]), H(q[0]), RZ(2.0, q[0])])
    assert address_qubits(prog) == address_qubits(result_prog)

    # trotter_order 1 steps 2
    prog = trotterize(term_one, term_two, trotter_steps=2)
    result_prog = Program().inst([H(q[0]), RZ(1.0, q[0]), H(q[0]), RZ(1.0, q[0]),
                                  H(q[0]), RZ(1.0, q[0]), H(q[0]), RZ(1.0, q[0])])
    assert address_qubits(prog) == address_qubits(result_prog)

    # trotter_order 2 steps 1
    prog = trotterize(term_one, term_two, trotter_order=2)
    result_prog = Program().inst([H(q[0]), RZ(1.0, q[0]), H(q[0]), RZ(2.0, q[0]),
                                  H(q[0]), RZ(1.0, q[0]), H(q[0])])
    assert address_qubits(prog) == address_qubits(result_prog)

    # trotter_order 2 steps 2
    prog = trotterize(term_one, term_two, trotter_order=2, trotter_steps=2)
    result_prog = Program().inst([H(q[0]), RZ(0.5, q[0]), H(q[0]), RZ(1.0, q[0]),
                                  H(q[0]), RZ(0.5, q[0]), H(q[0]),
                                  H(q[0]), RZ(0.5, q[0]), H(q[0]), RZ(1.0, q[0]),
                                  H(q[0]), RZ(0.5, q[0]), H(q[0])])
    assert address_qubits(prog) == address_qubits(result_prog)

    # trotter_order 3 steps 1
    prog = trotterize(term_one, term_two, trotter_order=3, trotter_steps=1)
    result_prog = Program().inst([H(q[0]), RZ(14.0 / 24, q[0]), H(q[0]), RZ(4.0 / 3.0, q[0]),
                                  H(q[0]), RZ(1.5, q[0]), H(q[0]), RZ(-4.0 / 3.0, q[0]),
                                  H(q[0]), RZ(-2.0 / 24, q[0]), H(q[0]), RZ(2.0, q[0])])
    assert address_qubits(prog) == address_qubits(result_prog)


def test_is_zeron():
    q = QubitPlaceholder.register(8)
    with pytest.raises(TypeError):
        is_zero(1)

    p_term = PauliTerm("X", q[0])
    ps_term = p_term + PauliTerm("Z", q[1])

    assert not is_zero(p_term)
    assert is_zero(p_term + -1 * p_term)
    assert not is_zero(ps_term)


def test_check_commutation():
    q = QubitPlaceholder.register(8)
    term1 = PauliTerm("X", q[0]) * PauliTerm("X", q[1])
    term2 = PauliTerm("Y", q[0]) * PauliTerm("Y", q[1])
    term3 = PauliTerm("Y", q[0]) * PauliTerm("Z", q[2])
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
    pauli_ops = [list(zip(x, QubitPlaceholder.register(3))) for x in pauli_list]
    pauli_ops_pq = [reduce(mul, (PauliTerm(*x) for x in op)) for op in pauli_ops]

    non_commuting_pairs = []
    commuting_pairs = []
    for x in range(len(pauli_ops_pq)):
        for y in range(x, len(pauli_ops_pq)):

            tmp_op = _commutator(pauli_ops_pq[x], pauli_ops_pq[y])
            assert len(tmp_op.terms) == 1
            if is_identity(tmp_op.terms[0]):
                commuting_pairs.append((pauli_ops_pq[x], pauli_ops_pq[y]))
            else:
                non_commuting_pairs.append((pauli_ops_pq[x], pauli_ops_pq[y]))

    # now that we have our sets let's check against our code.
    for t1, t2 in non_commuting_pairs:
        assert not check_commutation([t1], t2)

    for t1, t2 in commuting_pairs:
        assert check_commutation([t1], t2)


def test_commuting_sets():
    q = QubitPlaceholder.register(8)
    term1 = PauliTerm("X", q[0]) * PauliTerm("X", q[1])
    term2 = PauliTerm("Y", q[0]) * PauliTerm("Y", q[1])
    term3 = PauliTerm("Y", q[0]) * PauliTerm("Z", q[2])
    pauli_sum = term1 + term2 + term3
    commuting_sets(pauli_sum)


def test_paulisum_iteration():
    q = QubitPlaceholder.register(8)
    term_list = [sX(q[2]), sZ(q[4])]
    pauli_sum = sum(term_list)
    for ii, term in enumerate(pauli_sum):
        assert term_list[ii] == term


def test_paulisum_indexing():
    q = QubitPlaceholder.register(8)
    pauli_sum = 0.5 * sX(q[0]) + 0.1 * sZ(q[1])
    assert pauli_sum[0] == 0.5 * sX(q[0])
    for ii, term in enumerate(pauli_sum.terms):
        assert pauli_sum[ii] == term


def test_term_powers():
    for qubit in QubitPlaceholder.register(2):
        pauli_terms = [sI(qubit), sX(qubit), sY(qubit), sZ(qubit)]
        for pauli_term in pauli_terms:
            assert pauli_term ** 0 == sI(qubit)
            assert pauli_term ** 1 == pauli_term
            assert pauli_term ** 2 == sI(qubit)
            assert pauli_term ** 3 == pauli_term
    with pytest.raises(ValueError):
        pauli_terms[0] ** -1


def test_term_large_powers():
    # Test to make sure large powers can be computed
    q = QubitPlaceholder.register(2)
    (PauliTerm('X', q[0], 2) * PauliTerm('Y', q[0], 2)) ** 400


def test_sum_power():
    q = QubitPlaceholder.register(8)
    pauli_sum = (sY(q[0]) - sX(q[0])) * (1.0 / np.sqrt(2))
    assert pauli_sum ** 2 == PauliSum([sI(q[0])])
    with pytest.raises(ValueError):
        _ = pauli_sum ** -1
    pauli_sum = sI(q[0]) + sI(q[1])
    assert pauli_sum ** 0 == sI(q[0])
    # Test to make sure large powers can be computed
    pauli_sum ** 400


def test_term_equality():
    q0, q10 = QubitPlaceholder.register(2)
    with pytest.raises(TypeError):
        sI(q0) != 0
    assert sI(q0) == sI(q0)
    assert PauliTerm('X', q10, 1 + 1.j) == PauliTerm('X', q10, 1 + 1.j)
    assert PauliTerm('X', q10, 1 + 1.j) + PauliTerm('X', q10, 1 + 1.j) != PauliTerm('X', q10,
                                                                                    1 + 1.j)
    assert PauliTerm('X', q10, 1 + 1.j) != PauliTerm('X', q10, 1 + 1.j) + PauliTerm('X', q10,
                                                                                    1 + 1.j)


def test_term_with_coeff():
    q0 = QubitPlaceholder()
    assert PauliTerm('X', q0, 1.j) == term_with_coeff(sX(q0), 1.j)
    assert PauliTerm('X', q0, -1.0) == term_with_coeff(sX(q0), -1)
    with pytest.raises(ValueError):
        term_with_coeff(sI(q0), None)


def test_sum_equality():
    q0, q1 = QubitPlaceholder.register(2)
    pauli_sum = sY(q0) - sX(q0)
    assert pauli_sum != 2 * pauli_sum
    with pytest.warns(UnequalLengthWarning):
        assert pauli_sum != pauli_sum + sZ(q0)
    with pytest.warns(UnequalLengthWarning):
        assert pauli_sum + sZ(q0) != pauli_sum
    assert pauli_sum != sY(q1) - sX(q1)
    assert pauli_sum == -1.0 * sX(q0) + sY(q0)
    assert pauli_sum == pauli_sum * 1.0
    with pytest.raises(TypeError):
        assert pauli_sum != 0


def test_zero_term():
    qubit_id = QubitPlaceholder()
    coefficient = 10
    ps = sI(qubit_id) + sX(qubit_id)
    assert coefficient * ZERO() == ZERO()
    assert ZERO() * coefficient == ZERO()
    assert ZERO() * ID() == ZERO()
    assert ZERO() + ID() == ID()
    assert ZERO() + ps == ps
    assert ps + ZERO() == ps


def test_from_list():
    q = QubitPlaceholder.register(8)
    terms_list = [("X", q[0]), ("Y", q[1]), ("Z", q[5])]
    term = reduce(lambda x, y: x * y, [PauliTerm(*x) for x in terms_list])

    pterm = PauliTerm.from_list(terms_list)
    assert pterm == term

    with pytest.raises(ValueError):
        # terms are not on disjoint qubits
        pterm = PauliTerm.from_list([("X", q[0]), ("Y", q[0])])


def test_ordered():
    q = QubitPlaceholder.register(8)
    mapping = {x: i for i, x in enumerate(q)}
    term = sZ(q[3]) * sZ(q[2]) * sZ(q[1])
    prog = address_qubits(exponential_map(term)(0.5), mapping)
    assert prog.out() == "CNOT 3 2\n" \
                         "CNOT 2 1\n" \
                         "RZ(1.0) 1\n" \
                         "CNOT 2 1\n" \
                         "CNOT 3 2\n"


def test_simplify():
    q = QubitPlaceholder.register(8)
    t1 = sZ(q[0]) * sZ(q[1])
    t2 = sZ(q[0]) * sZ(q[1])
    assert (t1 + t2) == 2 * sZ(q[0]) * sZ(q[1])


def test_dont_simplify():
    q = QubitPlaceholder.register(8)
    t1 = sZ(q[0]) * sZ(q[1])
    t2 = sZ(q[2]) * sZ(q[3])
    assert (t1 + t2) != 2 * sZ(q[0]) * sZ(q[1])


def test_simplify_warning():
    q = QubitPlaceholder.register(8)
    t1 = sZ(q[0]) * sZ(q[1])
    t2 = sZ(q[1]) * sZ(q[0])
    with pytest.warns(UserWarning) as e:
        tsum = t1 + t2

    assert tsum == 2 * sZ(q[0]) * sZ(q[1])
    assert 'will be combined with' in str(e[0].message)
