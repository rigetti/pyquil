#!/usr/bin/env python

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
Example program run during Rigetti's live programming session at QIP 2018. For a more complete
implementation of QAOA, refer to the grove example and docs:
http://grove-docs.readthedocs.io/en/latest/qaoa.html
"""

from pyquil import get_qc
from pyquil.quil import Program
from pyquil.gates import H
from pyquil.paulis import sI, sX, sZ, exponentiate_commuting_pauli_sum
from pyquil.api import QVMConnection

# Create a 4-node array graph: 0-1-2-3.
graph = [(0, 1), (1, 2), (2, 3)]
# Nodes [0, 1, 2, 3].
nodes = range(4)

# Create the initial state program, a sum over all bitstrings, via Hadamards on all qubits.
init_state_prog = Program([H(i) for i in nodes])

# The cost Hamiltonian is sum of the application of 0.5 * (1 - \sigma_z^i * \sigma_z^j) for all
# qubit pairs (i, j).
h_cost = -0.5 * sum(sI(nodes[0]) - sZ(i) * sZ(j) for i, j in graph)

# The driver Hamiltonian is the sum of the application of \sigma_x^i for all qubits i.
h_driver = -1.0 * sum(sX(i) for i in nodes)


def qaoa_ansatz(gammas, betas):
    """
    Function that returns a QAOA ansatz program for a list of angles betas and gammas. len(betas) ==
    len(gammas) == P for a QAOA program of order P.

    :param list(float) gammas: Angles over which to parameterize the cost Hamiltonian.
    :param list(float) betas: Angles over which to parameterize the driver Hamiltonian.
    :return: The QAOA ansatz program.
    :rtype: Program.
    """
    return Program(
        [
            exponentiate_commuting_pauli_sum(h_cost)(g)
            + exponentiate_commuting_pauli_sum(h_driver)(b)
            for g, b in zip(gammas, betas)
        ]
    )


# Create a program, the state initialization plus a QAOA ansatz program, for P = 2.
program = init_state_prog + qaoa_ansatz([0.0, 0.5], [0.75, 1.0])

# Initialize the QVM and run the program.
qc = get_qc("9q-generic-qvm")

results = qc.run_and_measure(program, trials=2)
