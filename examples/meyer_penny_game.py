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

import pyquil.quil as pq
from pyquil import api
from pyquil.gates import I, H, X

import numpy as np


def meyer_penny_program():
    """
    Returns the program to simulate the Meyer-Penny Game
    The full description is available in ../docs/source/exercises.rst

    :return: pyQuil Program
    """
    prog = pq.Program()
    ro = prog.declare('ro', memory_size=2)
    picard_register = ro[1]
    answer_register = ro[0]

    then_branch = pq.Program(X(0))
    else_branch = pq.Program(I(0))

    # Prepare Qubits in Heads state or superposition, respectively
    prog.inst(X(0), H(1))
    # Q puts the coin into a superposition
    prog.inst(H(0))
    # Picard makes a decision and acts accordingly
    prog.measure(1, picard_register)
    prog.if_then(picard_register, then_branch, else_branch)
    # Q undoes his superposition operation
    prog.inst(H(0))
    # The outcome is recorded into the answer register
    prog.measure(0, answer_register)

    return prog


if __name__ == "__main__":
    n_trials = 10
    qvm = api.QVMConnection()
    outcomes = np.asarray(qvm.run(meyer_penny_program(), [0, 1],
                                  trials=n_trials))

    print("Number of games: {}".format(n_trials))
    print("Q's winning average: {}".format(outcomes[:, 0].mean()))
    print("Picard's flip-decision average: {}".format(outcomes[:, 1].mean()))
