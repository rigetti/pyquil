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

import math
from functools import reduce
import pyquil.quil as pq
import pyquil.api as forest
from pyquil.gates import H
from six.moves import range

def qubits_needed(n):
    """
    The number of qubits needed for a die of n faces.
    """
    return int(math.ceil(math.log(n, 2)))

def die_program(n):
    """
    Generate a quantum program to roll a die of n faces.
    """
    prog = pq.Program()
    qubits = qubits_needed(n)
    # Hadamard initialize.
    for q in range(qubits):
        prog.inst(H(q))
    # Measure everything.
    for q in range(qubits):
        prog.measure(q, [q])
    return prog

def process_result(r):
    """
    Convert a list of measurements to a die value.
    """
    return reduce(lambda s, x: 2*s + x, r, 0)

BATCH_SIZE = 10

dice = {}

CXN = forest.Connection()

def roll_die(n):
    """
    Roll an n-sided quantum die.
    """
    addresses = list(range(qubits_needed(n)))
    if not n in dice:
        dice[n] = die_program(n)
    die = dice[n]
    # Generate results and do rejection sampling.
    while True:
        results = CXN.run(die, addresses, BATCH_SIZE)
        for r in results:
            x = process_result(r)
            if 0 < x <= n:
                return x

if __name__ == '__main__':
    while True:
        print(roll_die(16384))
