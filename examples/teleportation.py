#!/usr/bin/env python

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
from pyquil import api
from pyquil.gates import X, Z, H, CNOT


def make_bell_pair(q1, q2):
    """Makes a bell pair between qubits q1 and q2
    """
    return Program(H(q1), CNOT(q1, q2))


def teleport(start_index, end_index, ancilla_index):
    """Teleport a qubit from start to end using an ancilla qubit
    """
    program = make_bell_pair(end_index, ancilla_index)

    ro = program.declare("ro", memory_size=3)

    # do the teleportation
    program.inst(CNOT(start_index, ancilla_index))
    program.inst(H(start_index))

    # measure the results and store them in classical registers [0] and [1]
    program.measure(start_index, ro[0])
    program.measure(ancilla_index, ro[1])

    program.if_then(ro[1], X(2))
    program.if_then(ro[0], Z(2))

    program.measure(end_index, ro[2])

    print(program)
    return program


if __name__ == "__main__":
    qvm = api.QVMConnection()

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
