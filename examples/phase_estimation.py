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

import pyquil.quil as pq
from pyquil.gates import H
import numpy as np
from math import log
from pyquil.algorithms.fourier import qft


def controlled(m):
    """
    Make a one-qubit-controlled version of a matrix.
    :param m: (numpy.ndarray) A matrix.
    :return: A controlled version of that matrix.
    """
    rows, cols = m.shape
    assert rows == cols
    n = rows
    I = np.eye(n)
    Z = np.zeros((n, n))
    controlled_m = np.bmat([[I, Z],
                            [Z, m]])
    return controlled_m


def phase_estimation(U, accuracy, reg_offset=0):
    """
    Generate a circuit for quantum phase estimation.
    :param U: (numpy.ndarray) A unitary matrix.
    :param accuracy: (int) Number of bits of accuracy desired.
    :param reg_offset: (int) Where to start writing measurements (default 0).
    :return: A Quil program to perform phase estimation.
    """
    assert isinstance(accuracy, int)
    rows, cols = U.shape
    m = int(log(rows, 2))
    output_qubits = range(0, accuracy)
    U_qubits = range(accuracy, accuracy + m)
    p = pq.Program()

    # Hadamard initialization
    for i in output_qubits:
        p.inst(H(i))
    # Controlled unitaries
    for i in output_qubits:
        if i > 0:
            U = np.dot(U, U)
        cU = controlled(U)
        name = "CONTROLLED-U{0}".format(2 ** i)
        # define the gate
        p.defgate(name, cU)
        # apply it
        p.inst((name, i) + tuple(U_qubits))
    # Compute the QFT
    p = p + qft(output_qubits)
    # Perform the measurements
    for i in output_qubits:
        p.measure(i, reg_offset + i)

    return p


if __name__ == '__main__':
    import pyquil.forest as forest
    qvm = forest.Connection()
    X = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    Y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]])
    Rx = np.exp(X * np.pi / 8)
    Ry = np.exp(Y * np.pi / 16)
    U = np.kron(Rx, Ry)
    p = phase_estimation(U, 3)
    print p
    print qvm.run(p, range(3+2),10)
    print qvm.wavefunction(p)
