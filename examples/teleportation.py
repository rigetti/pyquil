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

from pyquil.quil import Program
import pyquil.api as forest
from pyquil.gates import X, Z, H, CNOT


def make_bell_pair(q1, q2):
    """Makes a bell pair between qubits q1 and q2
    """
    return Program(H(q1), CNOT(q1, q2))


def teleport(start_index, end_index, ancilla_index):
    """Teleport a qubit from start to end using an ancilla qubit
    """
    p = make_bell_pair(end_index, ancilla_index)

    # do the teleportation
    p.inst(CNOT(start_index, ancilla_index))
    p.inst(H(start_index))

    # measure the results and store them in classical registers [0] and [1]
    p.measure(start_index, 0)
    p.measure(ancilla_index, 1)

    p.if_then(1, X(2))
    p.if_then(0, Z(2))

    p.measure(end_index, 2)

    return p


if __name__ == '__main__':
    qvm = forest.SyncConnection()

    # initialize qubit 0 in |1>
    teleport_demo = Program(X(0))
    teleport_demo += teleport(0, 2, 1)
    print("Teleporting |1> state: {}".format(qvm.run(teleport_demo, [2])))

    # initialize qubit 0 in |0>
    teleport_demo = Program()
    teleport_demo += teleport(0, 2, 1)
    print("Teleporting |0> state: {}".format(qvm.run(teleport_demo, [2])))

    # initialize qubit 0 in |+>
    teleport_demo = Program(H(0))
    teleport_demo += teleport(0, 2, 1)
    print("Teleporting |+> state: {}".format(qvm.run(teleport_demo, [2], 10)))

    print(Program(X(0)).measure(0, 0).if_then(0, Program(X(1))))
