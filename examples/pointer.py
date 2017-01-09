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

import numpy as np
import pyquil.quil as pq
from pyquil.gates import X
from math import floor


def changed_bit_pos(a, b):
    """
    Return the index of the first bit that changed between `a` an `b`.
    Return None if there are no changed bits.
    """
    c = a ^ b
    n = 0
    while c > 0:
        if c & 1 == 1:
            return n
        c >>= 1
        n += 1
    return None


def gray(num_bits):
    """
    Generate the Gray code for `num_bits` bits.
    """
    last = 0
    x = 0
    n = 1 << num_bits
    while n > x:
        bit_string = bin(n + x ^ x/2)[3:]
        value = int(bit_string, 2)
        yield bit_string, value, changed_bit_pos(last, value)
        last = value
        x += 1


def controlled(num_ptr_bits, U):
    """
    Given a one-qubit gate matrix U, construct a controlled-U on all pointer
    qubits.
    """
    d = 2 ** (1 + num_ptr_bits)
    m = np.eye(d)
    m[d-2:, d-2:] = U
    return m


def fixup(p, data_bits, ptr_bits, bits_set):
    """
    Flip back the pointer qubits that were previously flipped indicated by
    the flags `bits_set`.
    """
    for i in xrange(ptr_bits):
        if 0 != bits_set & (1 << i):
            p.inst(X(data_bits + i))


def pointer_gate(num_qubits, U):
    """
    Make a pointer gate on `num_qubits`. The one-qubit gate U will act on the
    qubit addressed by the pointer qubits interpreted as an unsigned binary
    integer.

    There are P = floor(lg(num_qubits)) pointer qubits, and qubits numbered

        N - 1
        N - 2
        ...
        N - P

    are those reserved to represent the pointer. The first N - P qubits
    are the qubits which the one-qubit gate U can act on.
    """
    ptr_bits = int(floor(np.log2(num_qubits)))
    data_bits = num_qubits - ptr_bits
    ptr_state = 0
    assert ptr_bits > 0

    p = pq.Program()

    p.defgate("CU", controlled(ptr_bits, U))

    for _, target_qubit, changed in gray(ptr_bits):
        if changed is None:
            for ptr_qubit in xrange(num_qubits - ptr_bits, num_qubits):
                p.inst(X(ptr_qubit))
                ptr_state ^= 1 << (ptr_qubit - data_bits)
        else:
            p.inst(X(data_bits + changed))
            ptr_state ^= 1 << changed

        if target_qubit < data_bits:
            control_qubits = tuple(data_bits + i for i in xrange(ptr_bits))
            p.inst(("CU",) + control_qubits + (target_qubit,))

    fixup(p, data_bits, ptr_bits, ptr_state)
    return p


if __name__ == '__main__':
    H = np.matrix([[1, 1], [1, -1]])/np.sqrt(2)
    print pointer_gate(11, H)
