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
"""
Module for working with Pauli algebras.
"""

from __future__ import division
from itertools import product
import numpy as np
import copy

from pyquil.quilatom import QubitPlaceholder

from .quil import Program
from .gates import H, RZ, RX, CNOT, X, PHASE, QUANTUM_GATES
from numbers import Number
from collections import Sequence, OrderedDict
import warnings
from six import integer_types as six_integer_types
from six.moves import range

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


class UnequalLengthWarning(Warning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


integer_types = six_integer_types + (np.int64, np.int32, np.int16, np.int8)
"""Explicitly include numpy integer dtypes (for python 3)."""

HASH_PRECISION = 1e6
"""The precision used when hashing terms to check equality. The simplify() method
uses np.isclose() for coefficient comparisons to 0 which has its own default precision. We
can't use np.isclose() for hashing terms though.
"""


def _valid_qubit(index):
    return ((isinstance(index, integer_types) and index >= 0)
            or isinstance(index, QubitPlaceholder))


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
        assert _valid_qubit(index)

        self._ops = OrderedDict()
        if op != "I":
            self._ops[index] = op
        if not isinstance(coefficient, Number):
            raise ValueError("coefficient of PauliTerm must be a Number.")
        self.coefficient = complex(coefficient)

    def id(self, sort_ops=True):
        """
        Returns an identifier string for the PauliTerm (ignoring the coefficient).

        Don't use this to compare terms. This function will not work with qubits that
        aren't sortable.

        :param sort_ops: Whether to sort operations by qubit. This is True by default for
            backwards compatibility but will change in pyQuil 2.0. Callers should never rely
            on comparing id's for testing equality. See ``operations_as_set`` instead.
        :return: A string representation of this term's operations.
        :rtype: string
        """
        if sort_ops and len(self._ops) > 1:
            warnings.warn("`PauliTerm.id()` will not work on PauliTerms where the qubits are not "
                          "sortable and should be avoided in favor of `operations_as_set`.",
                          FutureWarning)
            return ''.join("{}{}".format(self._ops[q], q) for q in sorted(self._ops.keys()))
        else:
            return ''.join("{}{}".format(p, q) for q, p in self._ops.items())

    def operations_as_set(self):
        """
        Return a frozenset of operations in this term.

        Use this in place of :py:func:`id` if the order of operations in the term does not
        matter.

        :return: frozenset of strings representing Pauli operations
        """
        return frozenset(self._ops.items())

    def __eq__(self, other):
        if not isinstance(other, (PauliTerm, PauliSum)):
            raise TypeError("Can't compare PauliTerm with object of type {}.".format(type(other)))
        elif isinstance(other, PauliSum):
            return other == self
        else:
            return (self.operations_as_set() == other.operations_as_set()
                    and np.isclose(self.coefficient, other.coefficient))

    def __hash__(self):
        return hash((
            round(self.coefficient.real * HASH_PRECISION),
            round(self.coefficient.imag * HASH_PRECISION),
            self.operations_as_set()
        ))

    def __len__(self):
        """
        The length of the PauliTerm is the number of Pauli operators in the term. A term that
        consists of only a scalar has a length of zero.
        """
        return len(self._ops)

    def copy(self):
        """
        Properly creates a new PauliTerm, with a completely new dictionary
        of operators
        """
        new_term = PauliTerm("I", 0, 1.0)  # create new object
        # manually copy all attributes over
        for key in self.__dict__.keys():
            val = self.__dict__[key]
            if isinstance(val, (dict, list, set)):  # mutable types
                new_term.__dict__[key] = copy.copy(val)
            else:  # immutable types
                new_term.__dict__[key] = val

        return new_term

    @property
    def program(self):
        return Program([QUANTUM_GATES[gate](q) for q, gate in self])

    def get_qubits(self):
        """Gets all the qubits that this PauliTerm operates on.
        """
        return list(self._ops.keys())

    def __getitem__(self, i):
        return self._ops.get(i, "I")

    def __iter__(self):
        for i in self.get_qubits():
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
        """Multiplies this Pauli Term with another PauliTerm, PauliSum, or number according to the
        Pauli algebra rules.

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

        :param other: A number or PauliTerm to multiply by
        :returns: A new PauliTerm
        :rtype: PauliTerm
        """
        assert isinstance(other, Number)
        return self * other

    def __pow__(self, power):
        """Raises this PauliTerm to power.

        :param int power: The power to raise this PauliTerm to.
        :return: The power-fold product of power.
        :rtype: PauliTerm
        """
        if not isinstance(power, int) or power < 0:
            raise ValueError("The power must be a non-negative integer.")

        if len(self.get_qubits()) == 0:
            # There weren't any nontrivial operators
            return term_with_coeff(self, 1)

        result = ID()
        for _ in range(power):
            result *= self
        return result

    def __add__(self, other):
        """Adds this PauliTerm with another one.

        :param other: A PauliTerm object or a Number
        :returns: A PauliSum object representing the sum of this PauliTerm and other
        :rtype: PauliSum
        """
        if isinstance(other, Number):
            return self + PauliTerm("I", 0, other)
        elif isinstance(other, PauliSum):
            return other + self
        else:
            new_sum = PauliSum([self, other])
            return new_sum.simplify()

    def __radd__(self, other):
        """Adds this PauliTerm with a Number.

        :param other: A PauliTerm object or a Number
        :returns: A new PauliTerm
        :rtype: PauliTerm
        """
        assert isinstance(other, Number)
        return PauliTerm("I", 0, other) + self

    def __sub__(self, other):
        """Subtracts a PauliTerm from this one.

        :param other: A PauliTerm object or a Number
        :returns: A PauliSum object representing the difference of this PauliTerm and term
        :rtype: PauliSum
        """
        return self + -1. * other

    def __rsub__(self, other):
        """Subtracts this PauliTerm from a Number or PauliTerm.

        :param other: A PauliTerm object or a Number
        :returns: A PauliSum object representing the difference of this PauliTerm and term
        :rtype: PauliSum
        """
        return other + -1. * self

    def __str__(self):
        term_strs = []
        for index in self._ops.keys():
            term_strs.append("%s%s" % (self[index], index))

        if len(term_strs) == 0:
            term_strs.append("I")
        out = "%s*%s" % (self.coefficient, '*'.join(term_strs))
        return out

    @classmethod
    def from_list(cls, terms_list, coefficient=1.0):
        """
        Allocates a Pauli Term from a list of operators and indices. This is more efficient than
        multiplying together individual terms.

        :param list terms_list: A list of tuples, e.g. [("X", 0), ("Y", 1)]
        :return: PauliTerm
        """
        pterm = PauliTerm("I", 0)
        assert all([op[0] in PAULI_OPS for op in terms_list])

        indices = [op[1] for op in terms_list]
        assert all(_valid_qubit(index) for index in indices)

        # this is because from_list doesn't call simplify in order to be more efficient.
        if len(set(indices)) != len(indices):
            raise ValueError("Elements of PauliTerm that are allocated using from_list must "
                             "be on disjoint qubits. Use PauliTerm multiplication to simplify "
                             "terms instead.")

        for op, index in terms_list:
            if op != "I":
                pterm._ops[index] = op
        if not isinstance(coefficient, Number):
            raise ValueError("coefficient of PauliTerm must be a Number.")
        pterm.coefficient = complex(coefficient)
        return pterm

    def pauli_string(self, qubits=None):
        """
        Return a string representation of this PauliTerm mod its phase, as a concatenation of the string representation
        of the
        >>> p = PauliTerm("X", 0) * PauliTerm("Y", 1, 1.j)
        >>> p.pauli_string()
        "XY"
        >>> p.pauli_string([0])
        "X"
        >>> p.pauli_string([0, 2])
        "XI"

        :param list qubits: The list of qubits to represent, given as ints. If None, defaults to all qubits in this
         PauliTerm.
        :return: The string representation of this PauliTerm, modulo its phase.
        :rtype: String
        """
        qubit_term_mapping = dict(self.operations_as_set())
        if qubits is None:
            qubits = [qubit for qubit, _ in qubit_term_mapping.items()]
        ps = ""
        for qubit in qubits:
            try:
                ps += qubit_term_mapping[qubit]
            except KeyError:
                ps += "I"
        return ps


# For convenience, a shorthand for several operators.
def ID():
    """
    The identity operator.
    """
    return PauliTerm("I", 0, 1)


def ZERO():
    """
    The zero operator.
    """
    return PauliTerm("I", 0, 0)


def sI(q):
    """
    A function that returns the identity operator on a particular qubit.

    :param int qubit_index: The index of the qubit
    :returns: A PauliTerm object
    :rtype: PauliTerm
    """
    return PauliTerm("I", q)


def sX(q):
    """
    A function that returns the sigma_X operator on a particular qubit.

    :param int qubit_index: The index of the qubit
    :returns: A PauliTerm object
    :rtype: PauliTerm
    """
    return PauliTerm("X", q)


def sY(q):
    """
    A function that returns the sigma_Y operator on a particular qubit.

    :param int qubit_index: The index of the qubit
    :returns: A PauliTerm object
    :rtype: PauliTerm
    """
    return PauliTerm("Y", q)


def sZ(q):
    """
    A function that returns the sigma_Z operator on a particular qubit.

    :param int qubit_index: The index of the qubit
    :returns: A PauliTerm object
    :rtype: PauliTerm
    """
    return PauliTerm("Z", q)


def term_with_coeff(term, coeff):
    """
    Change the coefficient of a PauliTerm.

    :param PauliTerm term: A PauliTerm object
    :param Number coeff: The coefficient to set on the PauliTerm
    :returns: A new PauliTerm that duplicates term but sets coeff
    :rtype: PauliTerm
    """
    if not isinstance(coeff, Number):
        raise ValueError("coeff must be a Number")
    new_pauli = term.copy()
    # We cast to a complex number to ensure that internally the coefficients remain compatible.
    new_pauli.coefficient = complex(coeff)
    return new_pauli


class PauliSum(object):
    """A sum of one or more PauliTerms.
    """

    def __init__(self, terms):
        """
        :param Sequence terms: A Sequence of PauliTerms.
        """
        if not (isinstance(terms, Sequence)
                and all([isinstance(term, PauliTerm) for term in terms])):
            raise ValueError("PauliSum's are currently constructed from Sequences of PauliTerms.")
        if len(terms) == 0:
            self.terms = [0.0 * ID()]
        else:
            self.terms = terms

    def __eq__(self, other):
        """Equality testing to see if two PauliSum's are equivalent.

        :param PauliSum other: The PauliSum to compare this PauliSum with.
        :return: True if other is equivalent to this PauliSum, False otherwise.
        :rtype: bool
        """
        if not isinstance(other, (PauliTerm, PauliSum)):
            raise TypeError("Can't compare PauliSum with object of type {}.".format(type(other)))
        elif isinstance(other, PauliTerm):
            return self == PauliSum([other])
        elif len(self.terms) != len(other.terms):
            warnings.warn(UnequalLengthWarning("These PauliSums have a different number of terms."))
            return False

        return set(self.terms) == set(other.terms)

    def __str__(self):
        return " + ".join([str(term) for term in self.terms])

    def __len__(self):
        """
        The length of the PauliSum is the number of PauliTerms in the sum.
        """
        return len(self.terms)

    def __getitem__(self, item):
        """
        :param int item: The index of the term in the sum to return
        :return: The PauliTerm at the index-th position in the PauliSum
        :rtype: PauliTerm
        """
        return self.terms[item]

    def __iter__(self):
        return self.terms.__iter__()

    def __mul__(self, other):
        """
        Multiplies together this PauliSum with PauliSum, PauliTerm or Number objects. The new term
        is then simplified according to the Pauli Algebra rules.

        :param other: a PauliSum, PauliTerm or Number object
        :return: A new PauliSum object given by the multiplication.
        :rtype: PauliSum
        """
        if not isinstance(other, (Number, PauliTerm, PauliSum)):
            raise ValueError("Cannot multiply PauliSum by term that is not a Number, PauliTerm, or"
                             "PauliSum")
        elif isinstance(other, PauliSum):
            other_terms = other.terms
        else:
            other_terms = [other]
        new_terms = [lterm * rterm for lterm, rterm in product(self.terms, other_terms)]
        new_sum = PauliSum(new_terms)
        return new_sum.simplify()

    def __rmul__(self, other):
        """
        Multiples together this PauliSum with PauliSum, PauliTerm or Number objects. The new term
        is then simplified according to the Pauli Algebra rules.

        :param other: a PauliSum, PauliTerm or Number object
        :return: A new PauliSum object given by the multiplication.
        :rtype: PauliSum
        """
        assert isinstance(other, Number)
        new_terms = [term.copy() for term in self.terms]
        for term in new_terms:
            term.coefficient *= other
        return PauliSum(new_terms).simplify()

    def __pow__(self, power):
        """Raises this PauliSum to power.

        :param int power: The power to raise this PauliSum to.
        :return: The power-th power of this PauliSum.
        :rtype: PauliSum
        """
        if not isinstance(power, int) or power < 0:
            raise ValueError("The power must be a non-negative integer.")
        result = PauliSum([ID()])

        if not self.get_qubits():
            # There aren't any nontrivial operators
            terms = [term_with_coeff(term, 1) for term in self.terms]
            for term in terms:
                result *= term
        else:
            for term in self.terms:
                for qubit_id in term.get_qubits():
                    result *= PauliTerm("I", qubit_id)

        for _ in range(power):
            result *= self
        return result

    def __add__(self, other):
        """
        Adds together this PauliSum with PauliSum, PauliTerm or Number objects. The new term
        is then simplified according to the Pauli Algebra rules.

        :param other: a PauliSum, PauliTerm or Number object
        :return: A new PauliSum object given by the addition.
        :rtype: PauliSum
        """
        if isinstance(other, PauliTerm):
            other = PauliSum([other])
        elif isinstance(other, Number):
            other = PauliSum([other * ID()])
        new_terms = [term.copy() for term in self.terms]
        new_terms.extend(other.terms)
        new_sum = PauliSum(new_terms)
        return new_sum.simplify()

    def __radd__(self, other):
        """
        Adds together this PauliSum with PauliSum, PauliTerm or Number objects. The new term
        is then simplified according to the Pauli Algebra rules.

        :param other: a PauliSum, PauliTerm or Number object
        :return: A new PauliSum object given by the addition.
        :rtype: PauliSum
        """
        assert isinstance(other, Number)
        return self + other

    def __sub__(self, other):
        """
        Finds the difference of this PauliSum with PauliSum, PauliTerm or Number objects. The new
        term is then simplified according to the Pauli Algebra rules.

        :param other: a PauliSum, PauliTerm or Number object
        :return: A new PauliSum object given by the subtraction.
        :rtype: PauliSum
        """
        return self + -1. * other

    def __rsub__(self, other):
        """
        Finds the different of this PauliSum with PauliSum, PauliTerm or Number objects. The new
        term is then simplified according to the Pauli Algebra rules.

        :param other: a PauliSum, PauliTerm or Number object
        :return: A new PauliSum object given by the subtraction.
        :rtype: PauliSum
        """
        return other + -1. * self

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
        return simplify_pauli_sum(self)

    def get_programs(self):
        """
        Get a Pyquil Program corresponding to each term in the PauliSum and a coefficient
        for each program

        :return: (programs, coefficients)
        """
        programs = [term.program for term in self.terms]
        coefficients = np.array([term.coefficient for term in self.terms])
        return programs, coefficients


def simplify_pauli_sum(pauli_sum):
    """Simplify the sum of Pauli operators according to Pauli algebra rules."""

    # You might want to use a defaultdict(list) here, but don't because
    # we want to do our best to preserve the order of terms.
    like_terms = OrderedDict()
    for term in pauli_sum.terms:
        key = term.operations_as_set()
        if key in like_terms:
            like_terms[key].append(term)
        else:
            like_terms[key] = [term]

    terms = []
    for term_list in like_terms.values():
        first_term = term_list[0]
        if len(term_list) == 1 and not np.isclose(first_term.coefficient, 0.0):
            terms.append(first_term)
        else:
            coeff = sum(t.coefficient for t in term_list)
            for t in term_list:
                if list(t._ops.items()) != list(first_term._ops.items()):
                    warnings.warn("The term {} will be combined with {}, but they have different "
                                  "orders of operations. This doesn't matter for QVM or "
                                  "wavefunction simulation but may be important when "
                                  "running on an actual device."
                                  .format(t.id(sort_ops=False), first_term.id(sort_ops=False)))

            if not np.isclose(coeff, 0.0):
                terms.append(term_with_coeff(term_list[0], coeff))
    return PauliSum(terms)


def check_commutation(pauli_list, pauli_two):
    """
    Check if commuting a PauliTerm commutes with a list of other terms by natural calculation.
    Uses the result in Section 3 of arXiv:1405.5749v2, modified slightly here to check for the
    number of anti-coincidences (which must always be even for commuting PauliTerms)
    instead of the no. of coincidences, as in the paper.

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

    for term in pauli_list:
        if not coincident_parity(term, pauli_two):
            return False
    return True


def commuting_sets(pauli_terms):
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
    for j in range(1, m_terms):
        isAssigned_bool = False
        for p in range(m_s):  # check if it commutes with each group
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


def exponentiate(term: PauliTerm):
    """
    Creates a pyQuil program that simulates the unitary evolution exp(-1j * term)

    :param term: A pauli term to exponentiate
    :returns: A Program object
    :rtype: Program
    """
    return exponential_map(term)(1.0)


def exponential_map(term):
    """
    Returns a function f(alpha) that constructs the Program corresponding to exp(-1j*alpha*term).

    :param term: A pauli term to exponentiate
    :returns: A function that takes an angle parameter and returns a program.
    :rtype: Function
    """
    if not np.isclose(np.imag(term.coefficient), 0.0):
        raise TypeError("PauliTerm coefficient must be real")

    coeff = term.coefficient.real
    term.coefficient = term.coefficient.real

    def exp_wrap(param):
        prog = Program()
        if is_identity(term):
            prog.inst(X(0))
            prog.inst(PHASE(-param * coeff, 0))
            prog.inst(X(0))
            prog.inst(PHASE(-param * coeff, 0))
        else:
            prog += _exponentiate_general_case(term, param)
        return prog

    return exp_wrap


def exponentiate_commuting_pauli_sum(pauli_sum):
    """
    Returns a function that maps all substituent PauliTerms and sums them into a program. NOTE: Use
    this function with care. Substituent PauliTerms should commute.

    :param PauliSum pauli_sum: PauliSum to exponentiate.
    :returns: A function that parametrizes the exponential.
    :rtype: function
    """
    if not isinstance(pauli_sum, PauliSum):
        raise TypeError("Argument 'pauli_sum' must be a PauliSum.")

    fns = [exponential_map(term) for term in pauli_sum]

    def combined_exp_wrap(param):
        return Program([f(param) for f in fns])

    return combined_exp_wrap


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
        revp = Program()
        revp.inst(list(reversed(p.instructions)))
        return revp

    quil_prog = Program()
    change_to_z_basis = Program()
    change_to_original_basis = Program()
    cnot_seq = Program()
    prev_index = None
    highest_target_index = None

    for index, op in pauli_term:
        if 'X' == op:
            change_to_z_basis.inst(H(index))
            change_to_original_basis.inst(H(index))

        elif 'Y' == op:
            change_to_z_basis.inst(RX(np.pi / 2.0, index))
            change_to_original_basis.inst(RX(-np.pi / 2.0, index))

        elif 'I' == op:
            continue

        if prev_index is not None:
            cnot_seq.inst(CNOT(prev_index, index))

        prev_index = index
        highest_target_index = index

    # building rotation circuit
    quil_prog += change_to_z_basis
    quil_prog += cnot_seq
    quil_prog.inst(RZ(2.0 * pauli_term.coefficient * param, highest_target_index))
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
    p1 = p2 = p4 = p5 = 1.0 / (4 - (4 ** (1. / 3)))
    p3 = 1 - 4 * p1
    trotter_dict = {1: [(1, 0), (1, 1)],
                    2: [(0.5, 0), (1, 1), (0.5, 0)],
                    3: [(7.0 / 24, 0), (2.0 / 3.0, 1), (3.0 / 4.0, 0), (-2.0 / 3.0, 1),
                        (-1.0 / 24, 0), (1.0, 1)],
                    4: [(p5 / 2, 0), (p5, 1), (p5 / 2, 0),
                        (p4 / 2, 0), (p4, 1), (p4 / 2, 0),
                        (p3 / 2, 0), (p3, 1), (p3 / 2, 0),
                        (p2 / 2, 0), (p2, 1), (p2 / 2, 0),
                        (p1 / 2, 0), (p1, 1), (p1 / 2, 0)]}

    order_slices = [(x0 / trotter_steps, x1) for x0, x1 in trotter_dict[trotter_order]]
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

    commutator = (first_pauli_term * second_pauli_term) + \
                 (-1 * second_pauli_term * first_pauli_term)

    prog = Program()
    if is_zero(commutator):
        param_exp_prog_one = exponential_map(first_pauli_term)
        exp_prog = param_exp_prog_one(1)
        prog += exp_prog
        param_exp_prog_two = exponential_map(second_pauli_term)
        exp_prog = param_exp_prog_two(1)
        prog += exp_prog
        return prog

    order_slices = suzuki_trotter(trotter_order, trotter_steps)
    for coeff, operator in order_slices:
        if operator == 0:
            param_prog = exponential_map(coeff * first_pauli_term)
            exp_prog = param_prog(1)
            prog += exp_prog
        else:
            param_prog = exponential_map(coeff * second_pauli_term)
            exp_prog = param_prog(1)
            prog += exp_prog
    return prog
