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
"""
Module for working with Pauli algebras.
"""

from itertools import product
import numpy as np
import copy
from pyquil.gates import H, RZ, RX, CNOT, X, PHASE
from pyquil.parametric import parametric, ParametricProgram
import pyquil.quil as pq
import pyquil.quilbase as pqb
from numbers import Number

PAULI_OPS = ["X", "Y", "Z", "I"]
PAULI_PROD = {'ZZ': 'I', 'YY': 'I', 'XX': 'I', 'II': 'I',
              'XY': 'Z', 'XZ': 'Y', 'YX': 'Z', 'YZ': 'X', 'ZX': 'Y',
              'ZY': 'X', 'IX': 'X', 'IY': 'Y', 'IZ': 'Z',
              'ZI': 'Z', 'YI': 'Y', 'XI': 'X',
              'X': 'X', 'Y': 'Y', 'Z': 'Z', 'I': 'I'}
PAULI_COEFF = {'ZZ': 1.0, 'YY': 1.0, 'XX': 1.0, 'II': 1.0,
               'XY': 1.0j, 'XZ': -1.0j, 'YX': -1.0j, 'YZ': 1.0j, 'ZX': 1.0j,
               'ZY': -1.0j, 'IX': 1.0, 'IY': 1.0, 'IZ': 1.0, 'ZI': 1.0,
               'YI': 1.0, 'XI': 1.0,
               'X': 1.0, 'Y': 1.0, 'Z': 1.0, 'I': 1.0}


class PauliTerm(object):
    """A term is a product of Pauli operators operating on different qubits.
    """

    def __init__(self, op, index, coefficient=1.0):
        """ Create a new Pauli Term with a Pauli operator at a particular index and a leading
        coefficient.

        :param string op: The Pauli operator as a string "X", "Y", "Z", or "I"
        :param int index: The qubit index that that operator is applied to.
        :param float coefficient: The coefficient multiplying the operator, e.g. 1.5 * Z_1
        """
        assert op in PAULI_OPS
        assert isinstance(index, int) and index >= 0

        self._ops = {}
        if op != "I":
            self._ops[index] = op

        self.coefficient = coefficient
        self._id = None

    def id(self):
        """
        Returns the unique identifier string for the PauliTerm.  Used in the
        simplify method of PauliSum.

        :return: The unique identifier for this term.
        :rtype: string
        """
        if self._id is not None:
            return self._id
        else:
            s = ""
            for index in sorted(self._ops.keys()):
                s += "%s%s" % (self[index], index)
            self._id = s
            return s

    def __len__(self):
        """l
        The length of the pauliterm is the number of Pauli operators in the term. This is
        equivalent to the length of self._ops dict
        """
        return len(self._ops)

    def get_qubits(self):
        """Gets all the qubits that this PauliTerm operates on.
        """
        return self._ops.keys()

    def __getitem__(self, i):
        return self._ops.get(i, "I")

    def __iter__(self):
        for i in self._ops.keys():
            yield i, self[i]

    def _multiply_factor(self, factor, index):
        new_term = PauliTerm("I", 0)
        new_coeff = self.coefficient
        new_ops = self._ops.copy()

        ops = self[index] + factor
        new_op = PAULI_PROD[ops]
        if new_op != "I":
            new_ops[index] = new_op
        else:
            del new_ops[index]
        new_coeff *= PAULI_COEFF[ops]

        new_term._ops = new_ops
        new_term.coefficient = new_coeff

        return new_term

    def __mul__(self, term):
        """Multiplies this Pauli Term with another PauliTerm, PauliSum, or number according to the Pauli algebra rules.

        :param term: (PauliTerm or PauliSum or Number) A term to multiply by.
        :returns: The product of this PauliTerm and term.
        :rtype: PauliTerm
        """
        if isinstance(term, Number):
            return term_with_coeff(self, self.coefficient * term)
        elif isinstance(term, PauliSum):
            return (PauliSum([self]) * term).simplify()
        else:
            new_term = PauliTerm("I", 0, 1.0)
            new_term._ops = self._ops.copy()
            new_coeff = self.coefficient * term.coefficient
            for index, op in term:
                new_term = new_term._multiply_factor(op, index)

            return term_with_coeff(new_term, new_term.coefficient * new_coeff)

    def __rmul__(self, other):
        """Multiplies this PauliTerm with another object, probably a number.

        :param Number term: A number to multiply by
        :returns: A new PauliTerm
        :rtype: PauliTerm
        """
        assert isinstance(other, Number)
        return self * other

    def __add__(self, term):
        """Adds this PauliTerm with another one.

        :param PauliTerm term: A PauliTerm object
        :returns: A PauliSum object representing the sum of this PauliTerm and term
        :rtype: PauliSum
        """
        if isinstance(term, Number):
            return self + PauliTerm("I", 0, term)
        elif isinstance(term, PauliSum):
            return term + self
        else:
            new_sum = PauliSum([self, term])
            return new_sum.simplify()

    def __radd__(self, term):
        """Adds this PauliTerm with a Number.

        :param Number term: A number to multiply by
        :returns: A new PauliTerm
        :rtype: PauliTerm
        """
        assert isinstance(term, Number)
        return self + term

    def __str__(self):
        term_strs = []
        for index in sorted(self._ops.keys()):
            term_strs.append("%s%s" % (self[index], index))

        if len(term_strs) == 0:
            term_strs.append("I")
        out = "%s*%s" % (self.coefficient, '*'.join(term_strs))
        return out


# For convenience, a shorthand for several operators.
ID = PauliTerm("I", 0)
"""
The identity Pauli operator on the 0th qubit.
"""
sI = lambda q: PauliTerm("I", q)
"""
A function that returns the identity operator on a particular qubit.

:param int qubit_index: The index of the qubit
:returns: A PauliTerm object
:rtype: PauliTerm
"""
sX = lambda q: PauliTerm("X", q)
"""
A function that returns the sigma_X operator on a particular qubit.

:param int qubit_index: The index of the qubit
:returns: A PauliTerm object
:rtype: PauliTerm
"""
sY = lambda q: PauliTerm("Y", q)
"""
A function that returns the sigma_Y operator on a particular qubit.

:param int qubit_index: The index of the qubit
:returns: A PauliTerm object
:rtype: PauliTerm
"""
sZ = lambda q: PauliTerm("Z", q)
"""
A function that returns the sigma_Z operator on a particular qubit.

:param int qubit_index: The index of the qubit
:returns: A PauliTerm object
:rtype: PauliTerm
"""


def term_with_coeff(term, coeff):
    """
    Change the coefficient of a PauliTerm.

    :param PauliTerm term: A PauliTerm object
    :param float coeff: The coefficient to set on the PauliTerm
    :returns: A new PauliTerm that duplicates term but sets coeff
    :rtype: PauliTerm
    """
    new_pauli = copy.copy(term)
    new_pauli.coefficient = coeff
    return new_pauli


class PauliSum(object):
    """A sum of one or more PauliTerms.
    """

    def __init__(self, terms):
        """
        :param list terms: A list of PauliTerms.
        """
        if len(terms) == 0:
            self.terms = [0.0 * ID]
        else:
            self.terms = terms

    def __str__(self):
        return " + ".join([str(term) for term in self.terms])

    def __mul__(self, other):
        if isinstance(other, PauliTerm):
            other_terms = [other]
        elif isinstance(other, PauliSum):
            other_terms = other.terms
        new_terms = []
        for lterm, rterm in product(self.terms, other_terms):
            new_terms.append(lterm * rterm)
        new_sum = PauliSum(new_terms)
        return new_sum.simplify()

    def __rmul__(self, other):
        assert isinstance(other, Number)
        new_terms = copy.deepcopy(self.terms)
        for term in new_terms:
            term.coefficient *= other
        return PauliSum(new_terms).simplify()

    def __add__(self, other):
        if isinstance(other, PauliTerm):
            other = PauliSum([other])
        elif isinstance(other, Number):
            other = PauliSum([other * ID])
        new_terms = copy.deepcopy(self.terms)
        new_terms.extend(other.terms)
        new_sum = PauliSum(new_terms)
        return new_sum.simplify()

    def __radd__(self, other):
        assert isinstance(other, Number)
        return self + other

    def get_qubits(self):
        """
        The support of all the operators in the PauliSum object.

        :returns: A list of all the qubits in the sum of terms.
        :rtype: list
        """
        return list(set().union(*[term.get_qubits() for term in self.terms]))

    def simplify(self):
        """
        Simplifies the sum of Pauli operators according to Pauli algebra rules.
        """
        def coalesce(d):
            terms = []
            for term_list in d.values():
                if (len(term_list) == 1 and not
                   np.isclose(term_list[0].coefficient, 0.0)):
                    terms.append(term_list[0])
                else:
                    coeff = sum(t.coefficient for t in term_list)
                    if not np.isclose(coeff, 0.0):
                        terms.append(term_with_coeff(term_list[0], coeff))
            return PauliSum(terms)

        like_terms = {}
        for term in self.terms:
            id = term.id()
            if id not in like_terms:
                like_terms[id] = [term]
            else:
                like_terms[id] = like_terms[id] + [term]

        return coalesce(like_terms)


def check_commutation(pauli_list, pauli_two):
    """
    Check if commuting a PauliTerm commutes with a list of other terms by natural calculation.
    Derivation similar to arXiv:1405.5749v2 fo the check_commutation step in
    the Raesi, Wiebe, Sanders algorithm (arXiv:1108.4318, 2011).

    :param list pauli_list: A list of PauliTerm objects
    :param PauliTerm pauli_two_term: A PauliTerm object
    :returns: True if pauli_two object commutes with pauli_list, False otherwise
    :rtype: bool
    """
    def coincident_parity(p1, p2):
        non_similar = 0
        p1_indices = set(p1._ops.keys())
        p2_indices = set(p2._ops.keys())
        for idx in p1_indices.intersection(p2_indices):
            if p1[idx] != p2[idx]:
                non_similar += 1
        return non_similar % 2 == 0

    for i, term in enumerate(pauli_list):
        if not coincident_parity(term, pauli_two):
            return False
    return True


def commuting_sets(pauli_terms, nqubits):
    """Gather the Pauli terms of pauli_terms variable into commuting sets

    Uses algorithm defined in (Raeisi, Wiebe, Sanders, arXiv:1108.4318, 2011)
    to find commuting sets. Except uses commutation check from arXiv:1405.5749v2

    :param PauliSum pauli_terms: A PauliSum object
    :returns: List of lists where each list contains a commuting set
    :rtype: list
    """

    m_terms = len(pauli_terms.terms)
    m_s = 1
    groups = []
    groups.append([pauli_terms.terms[0]])
    for j in xrange(1, m_terms):
        isAssigned_bool = False
        for p in xrange(m_s):  # check if it commutes with each group
            if isAssigned_bool is False:

                if check_commutation(groups[p], pauli_terms.terms[j]):
                    isAssigned_bool = True
                    groups[p].append(pauli_terms.terms[j])
        if isAssigned_bool is False:
            m_s += 1
            groups.append([pauli_terms.terms[j]])
    return groups


def is_identity(term):
    """
    Check if Pauli Term is a scalar multiple of identity

    :param PauliTerm term: A PauliTerm object
    :returns: True if the PauliTerm is a scalar multiple of identity, false otherwise
    :rtype: bool
    """
    return len(term) == 0


def exponentiate(term):
    """
    Creates a pyQuil program that simulates the unitary evolution exp(-1j * term)

    :param PauliTerm term: Tests is a PauliTerm is the identity operator
    :returns: A Program object
    :rtype: Program
    """
    return exponential_map(term)(1.0)


def exponential_map(term):
    """
    Creates map alpha -> exp(-1j*alpha*term) represented as a ParametricProgram.

    :param PauliTerm term: Tests is a PauliTerm is the identity operator
    :returns: A ParametricProgram
    :rtype: ParametricProgram
    """
    if not np.isclose(np.imag(term.coefficient), 0.0):
        raise TypeError("PauliTerm coefficient must be real")

    coeff = term.coefficient

    @parametric
    def exp_wrap(param):
        prog = pq.Program()
        if is_identity(term):
            prog.inst(X(0))
            prog.inst(PHASE(-param*coeff)(0))
            prog.inst(X(0))
            prog.inst(PHASE(-param*coeff)(0))
        else:
            prog += _exponentiate_general_case(term, param)
        return prog

    return exp_wrap


def _exponentiate_general_case(pauli_term, param):
    """
    Returns a Quil (Program()) object corresponding to the exponential of
    the pauli_term object, i.e. exp[-1.0j * param * pauli_term]

    :param PauliTerm pauli_term: A PauliTerm to exponentiate
    :param float param: scalar, non-complex, value
    :returns: A Quil program object
    :rtype: Program
    """
    def reverse_hack(p):
        # A hack to produce a *temporary* program which reverses p.
        def translate(tup):
            action, obj = tup
            if tup == pqb.ACTION_RELEASE_QUBIT:
                return (pqb.ACTION_INSTANTIATE_QUBIT, obj)
            elif tup == pqb.ACTION_INSTANTIATE_QUBIT:
                return (pqb.ACTION_RELEASE_QUBIT, obj)
            else:
                return tup
        revp = pq.Program()
        revp.actions = map(translate, reversed(p.actions))
        return revp

    quil_prog = pq.Program()
    change_to_z_basis = pq.Program()
    change_to_original_basis = pq.Program()
    cnot_seq = pq.Program()
    prev_index = None
    highest_target_index = None

    for index, op in pauli_term:
        if 'X' == op:
            change_to_z_basis.inst(H(index))
            change_to_original_basis.inst(H(index))

        elif 'Y' == op:
            change_to_z_basis.inst(RX(np.pi/2.0)(index))
            change_to_original_basis.inst(RX(-np.pi/2.0)(index))

        elif 'I' == op:
            continue

        if prev_index is not None:
            cnot_seq.inst(CNOT(prev_index, index))

        prev_index = index
        highest_target_index = index

    # building rotation circuit
    quil_prog += change_to_z_basis
    quil_prog += cnot_seq
    quil_prog.inst(RZ(2.0 * pauli_term.coefficient * param)(highest_target_index))
    quil_prog += reverse_hack(cnot_seq)
    quil_prog += change_to_original_basis

    return quil_prog


def suzuki_trotter(trotter_order, trotter_steps):
    """
    Generate trotterization coefficients for a given number of Trotter steps.

    U = exp(A + B) is approximated as exp(w1*o1)exp(w2*o2)... This method returns
    a list [(w1, o1), (w2, o2), ... , (wm, om)] of tuples where o=0 corresponds
    to the A operator, o=1 corresponds to the B operator, and w is the
    coefficient in the exponential. For example, a second order Suzuki-Trotter
    approximation to exp(A + B) results in the following
    [(0.5/trotter_steps, 0), (1/trotteri_steps, 1),
    (0.5/trotter_steps, 0)] * trotter_steps.

    :param int trotter_order: order of Suzuki-Trotter approximation
    :param int trotter_steps: number of steps in the approximation
    :returns: List of tuples corresponding to the coefficient and operator
              type: o=0 is A and o=1 is B.
    :rtype: list
    """
    p1 = p2 = p4 = p5 = 1.0/(4 - (4**(1./3)))
    p3 = 1 - 4 * p1
    trotter_dict = {1: [(1, 0), (1, 1)],
                    2: [(0.5, 0), (1, 1), (0.5, 0)],
                    3: [(7.0/24, 0), (2.0/3.0, 1), (3.0/4.0, 0), (-2.0/3.0, 1),
                        (-1.0/24, 0), (1.0, 1)],
                    4: [(p5/2, 0), (p5, 1), (p5/2, 0),
                        (p4/2, 0), (p4, 1), (p4/2, 0),
                        (p3/2, 0), (p3, 1), (p3/2, 0),
                        (p2/2, 0), (p2, 1), (p2/2, 0),
                        (p1/2, 0), (p1, 1), (p1/2, 0)]}

    order_slices = map(lambda x: (x[0]/float(trotter_steps), x[1]),
                       trotter_dict[trotter_order])

    order_slices = order_slices * trotter_steps
    return order_slices


def is_zero(pauli_object):
    """
    Tests to see if a PauliTerm or PauliSum is zero.

    :param pauli_object: Either a PauliTerm or PauliSum
    :returns: True if PauliTerm is zero, False otherwise
    :rtype: bool
    """
    if isinstance(pauli_object, PauliTerm):
        if pauli_object.id() == '':
            return True
        else:
            return False
    elif isinstance(pauli_object, PauliSum):
        if len(pauli_object.terms) == 1 and pauli_object.terms[0].id() == '':
            return True
        else:
            return False
    else:
        raise TypeError("is_zero only checks PauliTerms and PauliSum objects!")


def trotterize(first_pauli_term, second_pauli_term, trotter_order=1,
               trotter_steps=1):
    """
    Create a Quil program that approximates exp( (A + B)t) where A and B are
    PauliTerm operators.

    :param PauliTerm first_pauli_term: PauliTerm denoted `A`
    :param PauliTerm second_pauli_term: PauliTerm denoted `B`
    :param int trotter_order: Optional argument indicating the Suzuki-Trotter
                          approximation order--only accepts orders 1, 2, 3, 4.
    :param int trotter_steps: Optional argument indicating the number of products
                          to decompose the exponential into.

    :return: Quil program
    :rtype: Program
    """

    if not (1 <= trotter_order < 5):
        raise ValueError("trotterize only accepts trotter_order in {1, 2, 3, 4}.")

    commutator = (first_pauli_term * second_pauli_term) +\
                 (-1 * second_pauli_term * first_pauli_term)

    prog = pq.Program()
    fused_param_prog = ParametricProgram(lambda: pq.Program())
    if is_zero(commutator):
        param_exp_prog_one = exponential_map(first_pauli_term)
        exp_prog = param_exp_prog_one(1)
        prog += exp_prog
        param_exp_prog_two = exponential_map(second_pauli_term)
        exp_prog = param_exp_prog_two(1)
        prog += exp_prog
        fused_param_prog = param_exp_prog_one.fuse(param_exp_prog_two)
        return prog, fused_param_prog

    order_slices = suzuki_trotter(trotter_order, trotter_steps)
    for coeff, operator in order_slices:
        if operator == 0:
            param_prog = exponential_map(coeff * first_pauli_term)
            exp_prog = param_prog(1)
            fused_param_prog = fused_param_prog.fuse(param_prog)
            prog += exp_prog
        else:
            param_prog = exponential_map(coeff * second_pauli_term)
            exp_prog = param_prog(1)
            fused_param_prog = fused_param_prog.fuse(param_prog)
            prog += exp_prog
    return prog, fused_param_prog
